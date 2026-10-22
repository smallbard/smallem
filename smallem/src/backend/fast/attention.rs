//! L'attention fusionnée sur le processeur.
//!
//! Écrite avec les primitives, l'attention range la matrice des scores
//! `[…, N, T]` entière, puis celle des poids, et les relit : à l'entraînement
//! du modèle `xl`, 128 matrices de 512 × 512, 128 Mo par tenseur, bien plus
//! que le L3. Ici, comme dans FlashAttention, chaque tâche prend une tuile
//! de `QUERIES` requêtes et parcourt les clés par blocs de `KEYS` : la tuile
//! de scores, 32 Ko, reste dans le L1 et le L2, et n'est jamais écrite en
//! mémoire. Le softmax se fait en ligne : à chaque bloc de clés, le maximum
//! de la ligne peut monter, et ce qui a été accumulé est corrigé d'autant.
//!
//! Les blocs qui sont entièrement dans le futur de toute la tuile ne sont pas
//! calculés : c'est la moitié du travail à l'entraînement.
//!
//! La passe avant garde `L = m + ln(l)` par requête. La passe arrière
//! recalcule les poids tuile par tuile, `p = exp(s - L)`, sans refaire le
//! softmax.

use super::gemm::{self, Matrix};
use super::{Buffer, Fast, FastTensor, math};

/// Requêtes par tuile.
const QUERIES: usize = 128;
/// Clés par bloc.
const KEYS: usize = 128;

/// Les dimensions d'une attention : `matrices` matrices de `queries`
/// requêtes face à `positions` positions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dims {
    matrices: usize,
    queries: usize,
    positions: usize,
    key_dim: usize,
    value_dim: usize,
}

impl Dims {
    /// Les dimensions, si les trois entrées ont les mêmes dimensions de lot
    /// (sinon, le calcul passe par la définition de référence).
    pub(crate) fn of(q: &FastTensor, k: &FastTensor, v: &FastTensor) -> Option<Self> {
        let rank = q.shape.len();
        if rank < 2 || k.shape.len() != rank || v.shape.len() != rank {
            return None;
        }
        let lead = &q.shape[..rank - 2];
        if &k.shape[..rank - 2] != lead || &v.shape[..rank - 2] != lead {
            return None;
        }
        let (queries, key_dim) = (q.shape[rank - 2], q.shape[rank - 1]);
        let (positions, value_dim) = (v.shape[rank - 2], v.shape[rank - 1]);
        if k.shape[rank - 2] != positions || k.shape[rank - 1] != key_dim || queries > positions {
            return None;
        }
        Some(Self {
            matrices: lead.iter().product(),
            queries,
            positions,
            key_dim,
            value_dim,
        })
    }

    /// La dernière position (exclue) que voit la ligne `row` : les requêtes
    /// sont les dernières des positions.
    fn visible(&self, row: usize) -> usize {
        self.positions - self.queries + row + 1
    }
}

/// `C (+)= A · B` sur de petites matrices, dans une tâche : sans partage
/// entre fils.
fn product(fast: &Fast, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    gemm::gemm(fast.pool(), fast.isa(), a, b, c, accumulate);
}

/// Les scores d'une tuile de requêtes face à un bloc de clés, multipliés par
/// `scale`, avec `-∞` pour le futur.
#[allow(clippy::too_many_arguments)]
fn scores(
    fast: &Fast,
    dims: &Dims,
    q: &[f32],
    k: &[f32],
    row: usize,
    rows: usize,
    key: usize,
    keys: usize,
    scale: f32,
    out: &mut [f32],
) {
    let d = dims.key_dim;
    let tile = Matrix::new(&q[row * d..(row + rows) * d], rows, d);
    let block = Matrix::new(&k[key * d..(key + keys) * d], keys, d).t();
    product(fast, tile, block, &mut out[..rows * keys], false);
    for (i, line) in out[..rows * keys].chunks_mut(keys).enumerate() {
        math::scale(fast.isa(), line, scale);
        let visible = dims.visible(row + i);
        if visible < key + keys {
            line[visible.saturating_sub(key)..].fill(f32::NEG_INFINITY);
        }
    }
}

/// La passe avant : la sortie `[…, N, Dv]` et `L` `[…, N]`.
pub(crate) fn forward(
    fast: &Fast,
    dims: Dims,
    q: &FastTensor,
    k: &FastTensor,
    v: &FastTensor,
    scale: f32,
) -> (FastTensor, FastTensor) {
    let Dims {
        matrices,
        queries,
        positions,
        key_dim,
        value_dim,
    } = dims;
    let tiles = queries.div_ceil(QUERIES);
    let mut out = Buffer::any(matrices * queries * value_dim);
    let mut log_sum = Buffer::any(matrices * queries);
    let shared_out = super::pool::SharedMut::new(&mut out);
    let shared_log = super::pool::SharedMut::new(&mut log_sum);
    let isa = fast.isa();
    fast.pool().run(matrices * tiles, |task| {
        let (matrix, tile) = (task / tiles, task % tiles);
        let row = tile * QUERIES;
        let rows = QUERIES.min(queries - row);
        let q = &q.data()[matrix * queries * key_dim..][..queries * key_dim];
        let k = &k.data()[matrix * positions * key_dim..][..positions * key_dim];
        let v = &v.data()[matrix * positions * value_dim..][..positions * value_dim];
        // SAFETY : chaque tâche écrit ses propres lignes.
        let o = unsafe { shared_out.slice((matrix * queries + row) * value_dim, rows * value_dim) };
        // SAFETY : de même pour les sommes des exponentielles.
        let l_out = unsafe { shared_log.slice(matrix * queries + row, rows) };
        o.fill(0.0);
        let mut max = [f32::NEG_INFINITY; QUERIES];
        let mut sum = [0.0f32; QUERIES];
        let mut s = vec![0.0f32; QUERIES * KEYS];
        let end = dims.visible(row + rows - 1);
        for key in (0..end).step_by(KEYS) {
            let keys = KEYS.min(positions - key);
            scores(fast, &dims, q, k, row, rows, key, keys, scale, &mut s);
            for i in 0..rows {
                let line = &mut s[i * keys..(i + 1) * keys];
                let block_max = math::max(isa, line);
                let new_max = max[i].max(block_max);
                if new_max == f32::NEG_INFINITY {
                    // Rien de visible dans ce bloc pour cette ligne.
                    line.fill(0.0);
                    continue;
                }
                let correction = (max[i] - new_max).exp();
                let total = math::exp_shifted_sum(isa, line, new_max);
                sum[i] = sum[i] * correction + total;
                if correction != 1.0 {
                    math::scale(isa, &mut o[i * value_dim..(i + 1) * value_dim], correction);
                }
                max[i] = new_max;
            }
            let weights = Matrix::new(&s[..rows * keys], rows, keys);
            let values = Matrix::new(
                &v[key * value_dim..(key + keys) * value_dim],
                keys,
                value_dim,
            );
            product(fast, weights, values, o, true);
        }
        for i in 0..rows {
            math::scale(
                isa,
                &mut o[i * value_dim..(i + 1) * value_dim],
                1.0 / sum[i],
            );
            l_out[i] = max[i] + sum[i].ln();
        }
    });
    let rank = q.shape.len();
    let mut shape = q.shape.clone();
    shape[rank - 1] = value_dim;
    let log_shape = &q.shape[..rank - 1];
    (
        FastTensor::new(out, &shape),
        FastTensor::new(log_sum, log_shape),
    )
}

/// La passe arrière : les gradients de `q`, `k` et `v`. En notant `P` les
/// poids, `O` la sortie et `dO` son gradient :
///
/// ```text
/// D  = Σ dO × O              par ligne
/// dV = Pᵀ · dO
/// dP = dO · Vᵀ
/// dS = scale × P × (dP - D)
/// dQ = dS · K
/// dK = dSᵀ · Q
/// ```
///
/// Une tâche par matrice : `dK` et `dV` reçoivent des contributions de toutes
/// les tuiles de requêtes, additionnées dans l'ordre des tuiles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward(
    fast: &Fast,
    dims: Dims,
    q: &FastTensor,
    k: &FastTensor,
    v: &FastTensor,
    out: &FastTensor,
    log_sum: &FastTensor,
    grad: &FastTensor,
    scale: f32,
) -> [FastTensor; 3] {
    let Dims {
        matrices,
        queries,
        positions,
        key_dim,
        value_dim,
    } = dims;
    let isa = fast.isa();
    let mut grad_q = Buffer::any(q.len());
    let mut grad_k = Buffer::any(k.len());
    let mut grad_v = Buffer::any(v.len());
    let (shared_q, shared_k, shared_v) = (
        super::pool::SharedMut::new(&mut grad_q),
        super::pool::SharedMut::new(&mut grad_k),
        super::pool::SharedMut::new(&mut grad_v),
    );
    fast.pool().run(matrices, |matrix| {
        fn slice(t: &FastTensor, matrix: usize, rows: usize, width: usize) -> &[f32] {
            &t.data()[matrix * rows * width..][..rows * width]
        }
        let q = slice(q, matrix, queries, key_dim);
        let k = slice(k, matrix, positions, key_dim);
        let v = slice(v, matrix, positions, value_dim);
        let o = slice(out, matrix, queries, value_dim);
        let d_o = slice(grad, matrix, queries, value_dim);
        let l = &log_sum.data()[matrix * queries..][..queries];
        // SAFETY : chaque tâche écrit sa propre matrice.
        let dq = unsafe { shared_q.slice(matrix * queries * key_dim, queries * key_dim) };
        // SAFETY : de même pour les gradients des clés et des valeurs.
        let dk = unsafe { shared_k.slice(matrix * positions * key_dim, positions * key_dim) };
        // SAFETY : idem.
        let dv = unsafe { shared_v.slice(matrix * positions * value_dim, positions * value_dim) };
        dq.fill(0.0);
        dk.fill(0.0);
        dv.fill(0.0);
        let projection: Vec<f32> = (0..queries)
            .map(|i| {
                math::dot(
                    isa,
                    &o[i * value_dim..][..value_dim],
                    &d_o[i * value_dim..][..value_dim],
                )
            })
            .collect();
        let mut p = vec![0.0f32; QUERIES * KEYS];
        let mut dp = vec![0.0f32; QUERIES * KEYS];
        for row in (0..queries).step_by(QUERIES) {
            let rows = QUERIES.min(queries - row);
            let end = dims.visible(row + rows - 1);
            let d_o_tile = Matrix::new(
                &d_o[row * value_dim..(row + rows) * value_dim],
                rows,
                value_dim,
            );
            let q_tile = Matrix::new(&q[row * key_dim..(row + rows) * key_dim], rows, key_dim);
            for key in (0..end).step_by(KEYS) {
                let keys = KEYS.min(positions - key);
                scores(fast, &dims, q, k, row, rows, key, keys, scale, &mut p);
                // Les poids, à partir de `L` : `exp(-∞) = 0` pour le futur.
                for i in 0..rows {
                    let line = &mut p[i * keys..(i + 1) * keys];
                    math::exp_shifted_sum(isa, line, l[row + i]);
                }
                let weights = Matrix::new(&p[..rows * keys], rows, keys);
                // dV += Pᵀ · dO
                product(
                    fast,
                    weights.t(),
                    d_o_tile,
                    &mut dv[key * value_dim..(key + keys) * value_dim],
                    true,
                );
                // dP = dO · Vᵀ
                let v_block = Matrix::new(
                    &v[key * value_dim..(key + keys) * value_dim],
                    keys,
                    value_dim,
                );
                product(fast, d_o_tile, v_block.t(), &mut dp[..rows * keys], false);
                // dS = scale × P × (dP - D), rangé à la place de dP.
                for i in 0..rows {
                    let d = projection[row + i];
                    let (weights, line) = (
                        &p[i * keys..(i + 1) * keys],
                        &mut dp[i * keys..(i + 1) * keys],
                    );
                    for (g, &w) in line.iter_mut().zip(weights) {
                        *g = scale * w * (*g - d);
                    }
                }
                let ds = Matrix::new(&dp[..rows * keys], rows, keys);
                // dQ += dS · K
                let k_block = Matrix::new(&k[key * key_dim..(key + keys) * key_dim], keys, key_dim);
                product(
                    fast,
                    ds,
                    k_block,
                    &mut dq[row * key_dim..(row + rows) * key_dim],
                    true,
                );
                // dK += dSᵀ · Q
                product(
                    fast,
                    ds.t(),
                    q_tile,
                    &mut dk[key * key_dim..(key + keys) * key_dim],
                    true,
                );
            }
        }
    });
    [
        FastTensor::new(grad_q, &q.shape),
        FastTensor::new(grad_k, &k.shape),
        FastTensor::new(grad_v, &v.shape),
    ]
}
