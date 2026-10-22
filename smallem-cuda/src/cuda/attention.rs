//! L'attention : fusionnée pour l'entraînement, sur le cache pour la
//! génération.

use super::*;

impl Cuda {
    /// Les dimensions d'une attention `q [H…, N, Dk]`, `k [H…, T, Dk]`,
    /// `v [H…, T, Dv]` que les kernels fusionnés savent traiter :
    /// `(têtes, N, T, Dk, Dv)`. `None` si elles dépassent leurs limites, ou
    /// si l'attention fusionnée est désactivée.
    pub(crate) fn fused_attention_dims(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
    ) -> Option<(usize, usize, usize, usize, usize)> {
        let rank = q.shape.len();
        assert!(
            rank >= 2 && k.shape.len() == rank && v.shape.len() == rank,
            "attention : formes {:?}, {:?}, {:?}",
            q.shape,
            k.shape,
            v.shape,
        );
        let (queries, key_dim) = (q.shape[rank - 2], q.shape[rank - 1]);
        let (positions, value_dim) = (k.shape[rank - 2], v.shape[rank - 1]);
        assert!(
            q.shape[..rank - 2] == k.shape[..rank - 2]
                && k.shape[..rank - 2] == v.shape[..rank - 2]
                && k.shape[rank - 1] == key_dim
                && v.shape[rank - 2] == positions
                && queries <= positions,
            "attention : formes incompatibles {:?}, {:?}, {:?}",
            q.shape,
            k.shape,
            v.shape,
        );
        let heads: usize = q.shape[..rank - 2].iter().product();
        let fits = key_dim <= ATTENTION_MAX_DIM
            && value_dim <= ATTENTION_MAX_DIM
            && heads <= MAX_GRID_BLOCKS as usize;
        (self.fused_attention && fits).then_some((heads, queries, positions, key_dim, value_dim))
    }

    /// L'attention sur le cache de génération (voir
    /// `smallem::backend::ops::CachedAttention`), par `attention_cached`, ou
    /// par `attention_cached_wide` pour les têtes de plus de
    /// [`ATTENTION_MAX_DIM`] composantes. Elles doivent en faire au plus
    /// [`CACHED_WIDE_DIM`].
    ///
    /// Les clés de chaque requête sont découpées en tranches pour que la
    /// grille compte au moins `NARROW_BLOCKS` (256) blocs, sans tranche de moins
    /// de 32 positions. Avec plusieurs tranches, un second kernel les réunit.
    #[allow(clippy::too_many_arguments)]
    pub fn cached_attention(
        &self,
        q: &CudaTensor,
        keys: &CudaTensor,
        values: &CudaTensor,
        slots: &CudaTensor,
        rows: &CudaTensor,
        dims: CachedAttentionDims,
        scale: f32,
    ) -> CudaTensor {
        assert!(
            dims.key_dim <= CACHED_WIDE_DIM && dims.value_dim <= CACHED_WIDE_DIM,
            "attention sur le cache : têtes de {} et {} composantes, {CACHED_WIDE_DIM} au plus",
            dims.key_dim,
            dims.value_dim,
        );
        let wide = dims.key_dim > ATTENTION_MAX_DIM || dims.value_dim > ATTENTION_MAX_DIM;
        let lines = dims.queries * dims.heads;
        let (splits, chunk) = split_depth(dims.max_len, lines, WARP, 16);
        let grid = (lines as u32, splits as u32);
        let len = lines * dims.value_dim;
        let shape = vec![dims.queries, dims.heads, dims.value_dim];
        let kernel_dims = CachedDims {
            heads: dims.heads as u32,
            kv_heads: dims.kv_heads as u32,
            key_dim: dims.key_dim as u32,
            value_dim: dims.value_dim as u32,
            value_stride: dims.value_stride as u32,
            max_len: dims.max_len as u32,
        };
        let parts = self.alloc(splits * len);
        let stats = self.alloc(splits * lines * 2);
        let config = LaunchConfig2D::new(grid, (WARP as u32, 4), 0);
        // SAFETY: la grille couvre les tranches et les lignes, `parts` et
        // `stats` ont les tailles qu'attend le kernel, et les têtes tiennent
        // dans ses registres (vérifié ci-dessus).
        unsafe {
            if wide {
                launch!(
                    self,
                    cached,
                    attention_cached_wide,
                    prepare_attention_cached_wide,
                    config,
                    &q.storage.buffer,
                    &keys.storage.buffer,
                    &values.storage.buffer,
                    &slots.storage.buffer,
                    &rows.storage.buffer,
                    kernel_dims,
                    chunk as u32,
                    scale,
                    parts.cu_deviceptr() as *mut f32,
                    stats.cu_deviceptr() as *mut f32
                );
            } else {
                launch!(
                    self,
                    cached,
                    attention_cached,
                    prepare_attention_cached,
                    config,
                    &q.storage.buffer,
                    &keys.storage.buffer,
                    &values.storage.buffer,
                    &slots.storage.buffer,
                    &rows.storage.buffer,
                    kernel_dims,
                    chunk as u32,
                    scale,
                    parts.cu_deviceptr() as *mut f32,
                    stats.cu_deviceptr() as *mut f32
                );
            }
        }
        if splits == 1 {
            return self.wrap(parts, shape);
        }
        let mut out = self.alloc(len);
        launch!(
            self,
            cached,
            attention_cached_combine,
            prepare_attention_cached_combine,
            launch_config(len),
            &parts,
            &stats,
            splits as u32,
            dims.value_dim as u32,
            &mut out
        );
        self.wrap(out, shape)
    }

    /// Côté des tuiles des kernels d'attention, et forme de leurs blocs :
    /// 64 lignes et quatre warps sur les Tensor Cores, 32 lignes et
    /// `16 × 16` threads sinon.
    pub(super) fn attention_tiles(&self) -> (u32, (u32, u32)) {
        if self.mixed_precision {
            (WIDE_TILE as u32, (32, 4))
        } else {
            (ATTENTION_TILE as u32, (16, 16))
        }
    }

    /// L'attention fusionnée : la sortie `[H…, N, Dv]`, et `log_sum`
    /// `[H…, N]` pour la passe arrière.
    pub(crate) fn attention(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: f32,
        (heads, queries, positions, key_dim, value_dim): (usize, usize, usize, usize, usize),
    ) -> (CudaTensor, CudaTensor) {
        let out = self.alloc(heads * queries * value_dim);
        let log_sum = self.alloc(heads * queries);
        let (tile, block) = self.attention_tiles();
        let grid = ((queries as u32).div_ceil(tile), heads as u32);
        let config = LaunchConfig2D::new(grid, block, 0);
        let dims = (
            queries as u32,
            positions as u32,
            key_dim as u32,
            value_dim as u32,
        );
        let (out_ptr, log_sum_ptr) = (
            out.cu_deviceptr() as *mut f32,
            log_sum.cu_deviceptr() as *mut f32,
        );
        // Les deux versions ont les mêmes arguments.
        macro_rules! forward {
            ($family:ident, $kernel:ident, $prepare:ident) => {
                launch!(
                    self,
                    $family,
                    $kernel,
                    $prepare,
                    config,
                    &q.storage.buffer,
                    &k.storage.buffer,
                    &v.storage.buffer,
                    dims.0,
                    dims.1,
                    dims.2,
                    dims.3,
                    scale,
                    out_ptr,
                    log_sum_ptr
                )
            };
        }
        // SAFETY: une tuile de requêtes par bloc en x, une tête en y, des
        // blocs de la forme qu'attend chaque kernel ; les sorties ont les
        // tailles voulues, et les largeurs ont été vérifiées par
        // `fused_attention_dims`.
        unsafe {
            if self.mixed_precision {
                forward!(
                    attention_bf16,
                    attention_forward_bf16,
                    prepare_attention_forward_bf16
                );
            } else {
                forward!(attention, attention_forward, prepare_attention_forward);
            }
        }
        let mut shape = q.shape.clone();
        *shape.last_mut().expect("rang vérifié") = value_dim;
        let rows = shape[..shape.len() - 1].to_vec();
        (self.wrap(out, shape), self.wrap(log_sum, rows))
    }

    /// Les gradients de `q`, `k` et `v`, à partir de la sortie, de son
    /// gradient et de ce que la passe avant a gardé.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_backward(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        output: &CudaTensor,
        log_sum: &CudaTensor,
        grad: &CudaTensor,
        scale: f32,
        (heads, queries, positions, key_dim, value_dim): (usize, usize, usize, usize, usize),
    ) -> [CudaTensor; 3] {
        // `Σ dO × O` par ligne : le terme de la règle du softmax.
        let delta = self.row_dot(grad, output);
        let grad_q = self.alloc(q.storage.buffer.len());
        let grad_k = self.alloc(k.storage.buffer.len());
        let grad_v = self.alloc(v.storage.buffer.len());
        let (tile, block) = self.attention_tiles();
        let config =
            |len: usize| LaunchConfig2D::new(((len as u32).div_ceil(tile), heads as u32), block, 0);
        let dims = (
            queries as u32,
            positions as u32,
            key_dim as u32,
            value_dim as u32,
        );
        let (grad_q_ptr, grad_k_ptr, grad_v_ptr) = (
            grad_q.cu_deviceptr() as *mut f32,
            grad_k.cu_deviceptr() as *mut f32,
            grad_v.cu_deviceptr() as *mut f32,
        );
        // Les quatre kernels ont les mêmes arguments, à leurs sorties près.
        macro_rules! backward {
            ($family:ident, $kernel:ident, $prepare:ident, $len:expr, $($out:expr),+) => {
                launch!(
                    self,
                    $family,
                    $kernel,
                    $prepare,
                    config($len),
                    &q.storage.buffer,
                    &k.storage.buffer,
                    &v.storage.buffer,
                    &grad.storage.buffer,
                    &log_sum.storage.buffer,
                    &delta,
                    dims.0,
                    dims.1,
                    dims.2,
                    dims.3,
                    scale,
                    $($out),+
                )
            };
        }
        // SAFETY: grilles et tailles comme pour `attention`, une tuile de
        // clés par bloc pour le premier kernel de chaque paire ; les
        // gradients ont la taille de leurs entrées.
        unsafe {
            if self.mixed_precision {
                backward!(
                    attention_bf16,
                    attention_backward_keys_bf16,
                    prepare_attention_backward_keys_bf16,
                    positions,
                    grad_k_ptr,
                    grad_v_ptr
                );
                backward!(
                    attention_bf16,
                    attention_backward_queries_bf16,
                    prepare_attention_backward_queries_bf16,
                    queries,
                    grad_q_ptr
                );
            } else {
                backward!(
                    attention,
                    attention_backward_keys,
                    prepare_attention_backward_keys,
                    positions,
                    grad_k_ptr,
                    grad_v_ptr
                );
                backward!(
                    attention,
                    attention_backward_queries,
                    prepare_attention_backward_queries,
                    queries,
                    grad_q_ptr
                );
            }
        }
        [
            self.wrap(grad_q, q.shape.clone()),
            self.wrap(grad_k, k.shape.clone()),
            self.wrap(grad_v, v.shape.clone()),
        ]
    }

    /// L'attention sans fusion, telle qu'elle était avant l'étape 12 : les
    /// scores, le softmax causal, le mélange des valeurs. Rend aussi les
    /// poids, que la passe arrière relira.
    pub(crate) fn attention_unfused(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: f32,
    ) -> (CudaTensor, CudaTensor) {
        let rank = k.shape.len();
        let scores = self.matmul(q, &self.transpose(k, rank - 2, rank - 1));
        let weights = self.causal_softmax(&scores, scale);
        (self.matmul(&weights, v), weights)
    }

    /// Les gradients de [`Cuda::attention_unfused`], dans l'ordre où la
    /// tape les calculait.
    pub(crate) fn attention_unfused_backward(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        weights: &CudaTensor,
        grad: &CudaTensor,
        scale: f32,
    ) -> [CudaTensor; 3] {
        let rank = k.shape.len();
        let transposed = |t: &CudaTensor| self.transpose(t, rank - 2, rank - 1);
        let grad_v = self.matmul(&transposed(weights), grad);
        let grad_weights = self.matmul(grad, &transposed(v));
        let grad_scores = self.softmax_backward(weights, &grad_weights, scale);
        let grad_q = self.matmul(&grad_scores, k);
        let grad_k = self.matmul(&transposed(&grad_scores), q);
        [grad_q, grad_k, grad_v]
    }
}
