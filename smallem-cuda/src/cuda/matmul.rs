//! Les produits matriciels sur la carte : la primitive `matmul`, la couche
//! affine, les poids quantifiés et le produit par une transposée.

use super::*;

impl Cuda {
    /// Le produit matriciel `a · b`, plus `bias` `[cols]` sur chaque ligne
    /// s'il est donné : la primitive `matmul`, et la couche affine.
    ///
    /// Trois chemins, selon le nombre de lignes de `a` :
    ///
    /// - au plus `narrow_rows` : le kernel des matrices étroites, celui de
    ///   la génération token par token ;
    /// - au plus [`SPLIT_ROWS`] : les kernels en tuiles, avec la dimension
    ///   commune découpée en tranches si la grille est trop petite pour
    ///   occuper la carte, comme en génération par lots ;
    /// - au-delà, les kernels en tuiles tels quels, puis l'addition du
    ///   biais : les produits de l'entraînement, qui gardent ainsi
    ///   exactement leurs nombres.
    pub fn product(&self, a: &CudaTensor, b: &CudaTensor, bias: Option<&CudaTensor>) -> CudaTensor {
        assert!(
            a.shape.len() >= 2 && b.shape.len() >= 2,
            "matmul demande au moins 2 dimensions : {:?} et {:?}",
            a.shape,
            b.shape,
        );
        let (batch_a, matrix_a) = a.shape.split_at(a.shape.len() - 2);
        let (batch_b, matrix_b) = b.shape.split_at(b.shape.len() - 2);
        let (rows, inner) = (matrix_a[0], matrix_a[1]);
        let cols = matrix_b[1];
        assert_eq!(
            inner, matrix_b[0],
            "matmul : formes incompatibles {:?} et {:?}",
            a.shape, b.shape,
        );
        if let Some(bias) = bias {
            assert_eq!(bias.shape, [cols], "biais de forme inattendue");
        }

        let batch = broadcast_shapes(batch_a, batch_b);
        let layout_a = layout(&batch, &broadcast_strides(batch_a, &batch));
        let layout_b = layout(&batch, &broadcast_strides(batch_b, &batch));

        let mut shape = batch;
        shape.extend([rows, cols]);
        assert!(
            shape.len() <= MAX_RANK,
            "le backend CUDA gère au plus {MAX_RANK} dimensions, reçu {shape:?}",
        );
        let batch_len: usize = shape[..shape.len() - 2].iter().product();
        if rows <= self.narrow_rows {
            return self.matmul_narrow(a, layout_a, b, layout_b, batch_len, shape, inner, bias);
        }

        // Une grille à deux dimensions de blocs `16 × 16` threads, chacun
        // pour une tuile carrée de sortie : les colonnes en `x`, les lignes de
        // toutes les matrices du lot en `y`. Les tranches de la dimension
        // commune, s'il y en a, s'enchaînent aussi en `x`.
        let tile = if self.mixed_precision {
            MATMUL_BF16_TILE
        } else {
            MATMUL_TILE
        };
        let threads = (MATMUL_TILE / MATMUL_PATCH) as u32;
        let tiles = cols.div_ceil(tile) * batch_len * rows.div_ceil(tile);
        // Au-delà de `SPLIT_ROWS` lignes, la grille occupe la carte : une
        // seule tranche.
        let blocks = if rows <= SPLIT_ROWS {
            tiles
        } else {
            NARROW_BLOCKS
        };
        let (splits, depth) = split_depth(inner, blocks, SPLIT_MIN_DEPTH, MATMUL_DEPTH);
        let grid = (
            (splits * cols.div_ceil(tile)) as u32,
            (batch_len * rows.div_ceil(tile)) as u32,
        );
        assert!(
            grid.0 <= MAX_GRID_BLOCKS && grid.1 <= MAX_GRID_BLOCKS,
            "matmul trop grande pour une seule grille : {} × {} blocs, {MAX_GRID_BLOCKS} au plus \
             par dimension",
            grid.0,
            grid.1,
        );
        let len = batch_len * rows * cols;
        let out = self.alloc(splits * len);
        let config = LaunchConfig2D::new(grid, (threads, threads), 0);
        // Adresse en mémoire GPU : à ne jamais déréférencer sur l'hôte.
        let out_ptr = out.cu_deviceptr() as *mut f32;
        let (rows_u, inner_u, cols_u, depth_u) =
            (rows as u32, inner as u32, cols as u32, depth as u32);
        let (buffer_a, buffer_b) = (&a.storage.buffer, &b.storage.buffer);
        // SAFETY: dans les deux cas, la grille couvre exactement
        // `tranches × cols` en x et `batch_len × rows` en y, par tuiles de la
        // taille que le kernel attend, et `out` a `tranches × batch_len ×
        // rows × cols` éléments.
        unsafe {
            if self.mixed_precision {
                launch!(
                    self,
                    matmul,
                    matmul_bf16,
                    prepare_matmul_bf16,
                    config,
                    buffer_a,
                    layout_a,
                    buffer_b,
                    layout_b,
                    rows_u,
                    inner_u,
                    cols_u,
                    depth_u,
                    out_ptr
                );
            } else {
                launch!(
                    self,
                    matmul,
                    matmul_tiled,
                    prepare_matmul_tiled,
                    config,
                    buffer_a,
                    layout_a,
                    buffer_b,
                    layout_b,
                    rows_u,
                    inner_u,
                    cols_u,
                    depth_u,
                    out_ptr
                );
            }
        }
        if splits > 1 {
            // Les tranches sont rangées `[tranches, toute la sortie]`.
            let total = self.sum_splits(&out, 1, splits, len, bias, cols);
            return self.wrap(total, shape);
        }
        let product = self.wrap(out, shape);
        match bias {
            Some(bias) => self.add(&product, bias),
            None => product,
        }
    }

    /// `x · W (+ b)` pour des poids quantifiés au format `format` : `words`
    /// et `scales` leur forme compacte, `dequantized` `[K, N]` les mêmes
    /// poids en `f32` (voir `smallem::backend::ops::QuantizedMatmul`).
    ///
    /// Les produits de la génération lisent la forme compacte : le kernel
    /// des matrices étroites jusqu'à `narrow_rows` lignes, les tuiles
    /// découpées en tranches jusqu'à [`SPLIT_ROWS`]. Au-delà, les produits
    /// d'une évaluation lisent la forme déquantifiée, par [`Cuda::product`].
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul(
        &self,
        x: &CudaTensor,
        words: &CudaTensor,
        scales: &CudaTensor,
        dequantized: &CudaTensor,
        bias: Option<&CudaTensor>,
        format: Format,
    ) -> CudaTensor {
        let (inner, cols) = matrix_dims("quantized_matmul", x, dequantized);
        let rows = x.storage.buffer.len() / inner;
        let mut shape = x.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = cols;
        let flat = self.reshape(x, &[rows, inner]);
        if rows > SPLIT_ROWS {
            let out = self.product(&flat, dequantized, bias);
            return self.reshape(&out, &shape);
        }
        let (group, groups, per_group) = Quantized::layout(format, inner);
        assert_eq!(
            words.shape,
            [groups * per_group, cols],
            "mots de forme inattendue"
        );
        assert_eq!(scales.shape, [groups, cols], "échelles de forme inattendue");
        let dims = QuantizedDims {
            cols: cols as u32,
            bits: format.bits(),
            group: group as u32,
            codes_per_word: format.codes_per_word() as u32,
            words_per_group: per_group as u32,
        };
        let (buffer_x, buffer_w, buffer_s) = (
            &flat.storage.buffer,
            &words.storage.buffer,
            &scales.storage.buffer,
        );
        let (out, splits) = if rows <= self.narrow_rows {
            // Les tranches comptent des mots.
            let total_words = groups * per_group;
            let column_blocks = cols.div_ceil(WARP);
            let (splits, depth) = split_depth(total_words, column_blocks, NARROW_MIN_WORDS, 1);
            let out = self.alloc(splits * rows * cols);
            let (bias_buffer, has_bias) = match bias {
                Some(bias) if splits == 1 => (&*bias.storage.buffer, 1u32),
                _ => (&**buffer_x, 0u32),
            };
            let config =
                LaunchConfig2D::new((column_blocks as u32, splits as u32), (WARP as u32, 8), 0);
            // SAFETY: la grille couvre `cols` par 32 en x et les tranches en
            // y, et `out` a `tranches × rows × cols` éléments.
            unsafe {
                launch!(
                    self,
                    quantized,
                    matmul_narrow_quantized,
                    prepare_matmul_narrow_quantized,
                    config,
                    buffer_x,
                    buffer_w,
                    buffer_s,
                    dims,
                    rows as u32,
                    inner as u32,
                    depth as u32,
                    bias_buffer,
                    has_bias,
                    out.cu_deviceptr() as *mut f32
                );
            }
            if splits == 1 {
                return self.wrap(out, shape);
            }
            (out, splits)
        } else {
            let column_tiles = cols.div_ceil(MATMUL_TILE);
            let tiles = column_tiles * rows.div_ceil(MATMUL_TILE);
            let (splits, depth) = split_depth(inner, tiles, SPLIT_MIN_DEPTH, MATMUL_DEPTH);
            let out = self.alloc(splits * rows * cols);
            let threads = (MATMUL_TILE / MATMUL_PATCH) as u32;
            let grid = (
                (splits * column_tiles) as u32,
                rows.div_ceil(MATMUL_TILE) as u32,
            );
            let config = LaunchConfig2D::new(grid, (threads, threads), 0);
            // SAFETY: la grille couvre `tranches × cols` en x et `rows` en y,
            // par tuiles de 64, et `out` a `tranches × rows × cols` éléments.
            unsafe {
                launch!(
                    self,
                    quantized,
                    matmul_tiled_quantized,
                    prepare_matmul_tiled_quantized,
                    config,
                    buffer_x,
                    buffer_w,
                    buffer_s,
                    dims,
                    rows as u32,
                    inner as u32,
                    depth as u32,
                    out.cu_deviceptr() as *mut f32
                );
            }
            if splits == 1 {
                let product = self.wrap(out, shape);
                return match bias {
                    Some(bias) => self.add(&product, bias),
                    None => product,
                };
            }
            (out, splits)
        };
        let total = self.sum_splits(&out, 1, splits, rows * cols, bias, cols);
        self.wrap(total, shape)
    }

    /// Additionne les tranches d'un produit, `[lots, tranches, inner]`, en
    /// `[lots, inner]`, plus `bias` `[cols]` s'il est donné.
    pub(super) fn sum_splits(
        &self,
        parts: &Pooled<f32>,
        batches: usize,
        splits: usize,
        inner: usize,
        bias: Option<&CudaTensor>,
        cols: usize,
    ) -> Pooled<f32> {
        let len = batches * inner;
        let mut total = self.alloc(len);
        // Sans biais, le kernel reçoit un tableau qu'il ne lira pas.
        let (bias_buffer, has_bias) = match bias {
            Some(bias) => (&*bias.storage.buffer, 1u32),
            None => (&**parts, 0u32),
        };
        launch!(
            self,
            matmul,
            sum_splits,
            prepare_sum_splits,
            launch_config(len),
            parts,
            splits as u32,
            inner as u32,
            bias_buffer,
            has_bias,
            cols as u32,
            &mut total
        );
        total
    }

    /// Le produit matriciel quand `a` a au plus [`Cuda::narrow_rows`] lignes,
    /// par le kernel `matmul_narrow`. `shape` est celle de la sortie. Avec
    /// `bias`, `[cols]`, le biais est ajouté à chaque ligne.
    ///
    /// Le nombre de tranches le long de `inner` est choisi pour que la
    /// grille compte au moins [`NARROW_BLOCKS`] blocs, sans tranche plus
    /// courte que [`NARROW_MIN_DEPTH`]. Les sommes partielles des tranches
    /// sont ensuite additionnées.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_narrow(
        &self,
        a: &CudaTensor,
        layout_a: Layout,
        b: &CudaTensor,
        layout_b: Layout,
        batch_len: usize,
        shape: Vec<usize>,
        inner: usize,
        bias: Option<&CudaTensor>,
    ) -> CudaTensor {
        let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let column_blocks = cols.div_ceil(WARP);
        // Une tranche est un multiple de 32 : quatre indices par warp, et
        // huit warps.
        let (splits, depth) = split_depth(inner, column_blocks * batch_len, NARROW_MIN_DEPTH, WARP);
        let grid = (column_blocks as u32, (batch_len * splits) as u32);
        assert!(
            grid.0 <= MAX_GRID_BLOCKS && grid.1 <= MAX_GRID_BLOCKS,
            "matmul trop grande pour une seule grille : {} × {} blocs",
            grid.0,
            grid.1,
        );
        let out = self.alloc(batch_len * splits * rows * cols);
        let out_ptr = out.cu_deviceptr() as *mut f32;
        // Sans biais, le kernel reçoit un tableau qu'il ne lira pas.
        let (bias_buffer, has_bias) = match bias {
            Some(bias) => (&bias.storage.buffer, 1u32),
            None => (&a.storage.buffer, 0u32),
        };
        let config = LaunchConfig2D::new(grid, (WARP as u32, 8), 0);
        // SAFETY: la grille couvre `cols` par 32 en x et `lots × tranches`
        // en y, et `out` a `lots × tranches × rows × cols` éléments.
        unsafe {
            launch!(
                self,
                matmul,
                matmul_narrow,
                prepare_matmul_narrow,
                config,
                &a.storage.buffer,
                layout_a,
                &b.storage.buffer,
                layout_b,
                rows as u32,
                inner as u32,
                cols as u32,
                depth as u32,
                bias_buffer,
                has_bias,
                out_ptr
            );
        }
        if splits == 1 {
            return self.wrap(out, shape);
        }
        let total = self.sum_splits(&out, batch_len, splits, rows * cols, bias, cols);
        self.wrap(total, shape)
    }

    /// `x · W + b` pour `x` `[…, K]`, une matrice `W` et un biais `b` : la
    /// couche affine, par [`Cuda::product`].
    pub fn affine(&self, x: &CudaTensor, weight: &CudaTensor, bias: &CudaTensor) -> CudaTensor {
        let (inner, cols) = matrix_dims("affine", x, weight);
        let rows = x.storage.buffer.len() / inner;
        let flat = self.reshape(x, &[rows, inner]);
        let out = self.product(&flat, weight, Some(bias));
        let mut shape = x.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = cols;
        self.reshape(&out, &shape)
    }

    /// `a · bᵀ` pour `a` `[…, K]` et une matrice `b` `[N, K]`, par
    /// `transposed_matmul_narrow`. Le kernel traite les lignes de `a` par
    /// paquets de 8, et en accepte autant qu'on veut ; il n'est rapide que
    /// pour quelques-unes, celles de la génération (voir
    /// [`Cuda::narrow_rows`]).
    pub fn transposed_matmul_narrow(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        assert!(
            b.shape.len() == 2 && b.shape[1] > 0 && a.shape.last() == Some(&b.shape[1]),
            "transposed_matmul : {:?} · {:?}ᵀ",
            a.shape,
            b.shape,
        );
        let (n, inner) = (b.shape[0], b.shape[1]);
        let rows = a.storage.buffer.len() / inner;
        let out = self.alloc(rows * n);
        let config = LaunchConfig2D::new((n.div_ceil(8) as u32, 1), (WARP as u32, 8), 0);
        // SAFETY: la grille couvre les `n` lignes de `b` par 8, et `out` a
        // `rows × n` éléments.
        unsafe {
            launch!(
                self,
                matmul,
                transposed_matmul_narrow,
                prepare_transposed_matmul_narrow,
                config,
                &a.storage.buffer,
                &b.storage.buffer,
                rows as u32,
                inner as u32,
                n as u32,
                out.cu_deviceptr() as *mut f32
            );
        }
        let mut shape = a.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = n;
        self.wrap(out, shape)
    }
}

/// Les dimensions `(K, N)` d'un produit de `x` `[…, K]` par une matrice
/// `[K, N]`, vérifiées avant d'être lues. `K` n'est pas nul : on divise par
/// lui pour compter les lignes de `x`.
fn matrix_dims(name: &str, x: &CudaTensor, matrix: &CudaTensor) -> (usize, usize) {
    assert!(
        matrix.shape.len() == 2 && matrix.shape[0] > 0 && x.shape.last() == Some(&matrix.shape[0]),
        "{name} : {:?} · {:?}",
        x.shape,
        matrix.shape,
    );
    (matrix.shape[0], matrix.shape[1])
}
