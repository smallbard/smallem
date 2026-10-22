//! L'attention fusionnée sur les Tensor Cores, en précision mixte : le même
//! algorithme qu'en `f32`, par tuiles de `bf16` rangées par paires.

use super::*;

/// Pas d'une ligne de tuile rangée par paires de `bf16` : 32 paires, plus
/// une case pour décaler les bancs d'une ligne à l'autre.
const PAIR_ROW: usize = WIDE_TILE / 2 + 1;

/// Maximum de `value` sur les quatre threads d'un même groupe : ceux qui,
/// dans les fragments de `mma.sync`, se partagent une ligne.
#[inline(always)]
pub fn quad_max(value: f32) -> f32 {
    let value = value.max(warp::shuffle_xor_f32(value, 1));
    value.max(warp::shuffle_xor_f32(value, 2))
}

/// Somme de `value` sur les quatre threads d'un même groupe.
#[inline(always)]
pub fn quad_sum(value: f32) -> f32 {
    let value = value + warp::shuffle_xor_f32(value, 1);
    value + warp::shuffle_xor_f32(value, 2)
}

/// Fragment `A` de `mma.sync` formé de deux résultats `16 × 8` voisins,
/// arrondis en `bf16` : `left` couvre les colonnes `0 … 7` d'un pas de
/// seize, `right` les colonnes `8 … 15`. La disposition d'un résultat (lignes
/// `groupe` et `groupe + 8`, colonnes `2t` et `2t + 1`) est exactement celle
/// qu'attend un fragment `A` pour ces colonnes : aucun échange n'est
/// nécessaire.
#[inline(always)]
pub fn fragment_a(left: [f32; 4], right: [f32; 4]) -> [u32; 4] {
    [
        bf16_pair(left[0], left[1]),
        bf16_pair(left[2], left[3]),
        bf16_pair(right[0], right[1]),
        bf16_pair(right[2], right[3]),
    ]
}

/// Fragment `A` lu directement en mémoire globale : les lignes `rows` d'une
/// matrice `[limit, dim]` qui commence à `start` dans `src`, colonnes
/// `column`, `column + 1`, `column + 8` et `column + 9`, multipliées par
/// `factor`.
#[inline(always)]
pub fn global_fragment(
    src: &[f32],
    start: usize,
    rows: [usize; 2],
    limit: usize,
    column: usize,
    dim: usize,
    factor: f32,
) -> [u32; 4] {
    [
        global_pair(src, start, rows[0], limit, column, dim, factor),
        global_pair(src, start, rows[1], limit, column, dim, factor),
        global_pair(src, start, rows[0], limit, column + 8, dim, factor),
        global_pair(src, start, rows[1], limit, column + 8, dim, factor),
    ]
}

/// Les éléments `col` et `col + 1` de la ligne `row` d'une matrice
/// `[rows, dim]` qui commence à `start` dans `src`, multipliés par
/// `factor` et rangés en une paire de `bf16`. Hors de la matrice, 0.
#[inline(always)]
pub fn global_pair(
    src: &[f32],
    start: usize,
    row: usize,
    rows: usize,
    col: usize,
    dim: usize,
    factor: f32,
) -> u32 {
    let read = |col: usize| {
        if row < rows && col < dim {
            src[start + row * dim + col] * factor
        } else {
            0.0
        }
    };
    bf16_pair(read(col), read(col + 1))
}

#[cuda_module]
pub mod device {
    use super::*;

    // Les tuiles des kernels d'attention sur Tensor Cores sont rangées par
    // paires de `bf16`, `PAIR_ROW` cases par ligne.
    //
    // `load_pairs!` range les lignes `$first .. $first + 64` d'une matrice
    // `[$rows, $dim]` de `$src` (qui commence à `$start`), multipliées par
    // `$factor`, par paires le long de chaque ligne : la case
    // `ligne × PAIR_ROW + p` tient les composantes `2p` et `2p + 1`. Hors de
    // la matrice, des zéros. Les 128 threads du bloc chargent chacun seize
    // paires.
    macro_rules! load_pairs {
        ($tile:ident, $src:expr, $start:expr, $first:expr, $rows:expr, $dim:expr, $factor:expr, $slot:expr) => {{
            let mut cell = $slot;
            while cell < WIDE_TILE * WIDE_TILE / 2 {
                let (row, pair) = (cell / (WIDE_TILE / 2), cell % (WIDE_TILE / 2));
                $tile[row * PAIR_ROW + pair] =
                    global_pair($src, $start, $first + row, $rows, 2 * pair, $dim, $factor);
                cell += 128;
            }
        }};
    }

    // La même chose, transposée : la case `d × PAIR_ROW + p` tient la
    // composante `d` des lignes `2p` et `2p + 1`. Les threads voisins lisent
    // des composantes voisines, donc de la mémoire contiguë.
    macro_rules! load_pairs_transposed {
        ($tile:ident, $src:expr, $start:expr, $first:expr, $rows:expr, $dim:expr, $factor:expr, $slot:expr) => {{
            let mut cell = $slot;
            while cell < WIDE_TILE * WIDE_TILE / 2 {
                let (pair, d) = (cell / WIDE_TILE, cell % WIDE_TILE);
                let (row, rows, dim) = ($first + 2 * pair, $rows, $dim);
                let read = |row: usize| {
                    if row < rows && d < dim {
                        $src[$start + row * dim + d] * $factor
                    } else {
                        0.0
                    }
                };
                $tile[d * PAIR_ROW + pair] = bf16_pair(read(row), read(row + 1));
                cell += 128;
            }
        }};
    }

    // `$acc[n] += $a[kk] · B(n, kk)` pour les huit tuiles `n` de 8 colonnes
    // et les quatre pas `kk` de 16 : un produit `[16, 64] × [64, 64]` par
    // warp. Le fragment `B` est lu dans une tuile rangée par paires le long
    // de la dimension commune : colonne `8n + groupe`, paires `8kk + t` et
    // `8kk + t + 4`.
    macro_rules! product {
        ($acc:ident, $a:ident, $tile:ident, $group:expr, $t:expr) => {{
            unroll!(kk in [0, 1, 2, 3] {
                unroll!(n in [0, 1, 2, 3, 4, 5, 6, 7] {
                    let column = (8 * n + $group) * PAIR_ROW + 8 * kk + $t;
                    let b = [$tile[column], $tile[column + 4]];
                    $acc[n] = wmma::mma_m16n8k16_f32_bf16($acc[n], $a[kk], b);
                });
            });
        }};
    }

    // Le même produit, quand le facteur de gauche est lui-même le résultat
    // `[16, 64]` d'un produit : ses tuiles `2kk` et `2kk + 1` forment le
    // fragment `A` du pas `kk` (voir [`fragment_a`]).
    macro_rules! product_from {
        ($acc:ident, $left:ident, $tile:ident, $group:expr, $t:expr) => {{
            let mut a = [[0u32; 4]; 4];
            unroll!(kk in [0, 1, 2, 3] {
                a[kk] = fragment_a($left[2 * kk], $left[2 * kk + 1]);
            });
            product!($acc, a, $tile, $group, $t)
        }};
    }

    // Écrit les huit tuiles `$acc` d'un warp dans une matrice `[_, $dim]` de
    // `$dst` : l'élément `i` de la tuile `n` est à la ligne `$rows[i / 2]`
    // et à la colonne `8n + 2t + i % 2`. Les lignes à partir de `$limit` et
    // les colonnes à partir de `$dim` n'existent pas.
    macro_rules! store {
        ($dst:expr, $acc:ident, $rows:expr, $limit:expr, $start:expr, $dim:expr, $t:expr, $factor:expr) => {{
            unroll!(n in [0, 1, 2, 3, 4, 5, 6, 7] {
                unroll!(i in [0, 1, 2, 3] {
                    let (row, d) = ($rows[i / 2], 8 * n + 2 * $t + i % 2);
                    if row < $limit && d < $dim {
                        // SAFETY: `(row, d)` n'appartient qu'à ce thread.
                        unsafe {
                            *$dst.add($start + row * $dim + d) = $acc[n][i] * $factor[i / 2];
                        }
                    }
                });
            });
        }};
    }

    /// La passe avant de l'attention fusionnée sur les Tensor Cores, en
    /// précision mixte : le même algorithme que `attention_forward`, avec
    /// les deux produits de chaque tuile faits par `mma.sync`.
    ///
    /// Un bloc de quatre warps prend 64 requêtes, seize par warp, et
    /// parcourt les clés par tuiles de 64. Chaque warp garde ses requêtes
    /// dans ses registres, sous la forme des fragments `A` de l'instruction,
    /// et calcule ses scores `16 × 64` en huit tuiles `16 × 8`.
    ///
    /// Le résultat d'une multiplication a la disposition d'un fragment `A` :
    /// les poids passent des scores au produit par les valeurs sans toucher
    /// la mémoire partagée. Chaque thread tient des éléments de deux lignes,
    /// `groupe` et `groupe + 8` ; les quatre threads d'une même ligne
    /// réunissent maximum et somme par deux échanges de registres.
    ///
    /// Comme dans `matmul_bf16`, les opérandes sont arrondis en `bf16` et les
    /// sommes restent en `f32`.
    ///
    /// # Safety
    ///
    /// Comme `attention_forward`, avec des tuiles de [`WIDE_TILE`] requêtes
    /// et des blocs de quatre warps.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, block = (32, 4, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_forward_bf16(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        queries: u32,
        positions: u32,
        key_dim: u32,
        value_dim: u32,
        scale: f32,
        out: *mut f32,
        log_sum: *mut f32,
    ) {
        // Les clés par paires le long de leurs composantes ; les valeurs
        // par paires le long des positions.
        static mut KEYS: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut VALUES: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;

        let (queries, positions) = (queries as usize, positions as usize);
        let (key_dim, value_dim) = (key_dim as usize, value_dim as usize);
        let (lane, warp) = (
            thread::threadIdx_x() as usize,
            thread::threadIdx_y() as usize,
        );
        let (slot, group, t) = (warp * 32 + lane, lane / 4, lane % 4);
        let head = thread::blockIdx_y() as usize;
        let first_query = thread::blockIdx_x() as usize * WIDE_TILE;
        let past = positions - queries;
        // Les deux lignes de ce thread.
        let rows = [
            first_query + 16 * warp + group,
            first_query + 16 * warp + group + 8,
        ];

        let q_start = head * queries * key_dim;
        let mut qa = [[0u32; 4]; 4];
        unroll!(kk in [0, 1, 2, 3] {
            qa[kk] = global_fragment(q, q_start, rows, queries, 16 * kk + 2 * t, key_dim, scale);
        });
        let mut max = [f32::NEG_INFINITY; 2];
        let mut sum = [0.0f32; 2];
        let mut o = [[0.0f32; 4]; 8];

        let end = positions.min(past + first_query + WIDE_TILE);
        let (k_start, v_start) = (head * positions * key_dim, head * positions * value_dim);
        let mut first_key = 0;
        while first_key < end {
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                load_pairs!(KEYS, k, k_start, first_key, positions, key_dim, 1.0, slot);
                load_pairs_transposed!(
                    VALUES, v, v_start, first_key, positions, value_dim, 1.0, slot
                );
            }
            thread::sync_threads();

            // 1. Les scores : huit tuiles de 8 clés.
            let mut s = [[0.0f32; 4]; 8];
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe { product!(s, qa, KEYS, group, t) }

            // 2. Le futur reçoit moins l'infini, puis le softmax en ligne.
            let mut tile_max = [f32::NEG_INFINITY; 2];
            unroll!(n in [0, 1, 2, 3, 4, 5, 6, 7] {
                unroll!(i in [0, 1, 2, 3] {
                    let key = first_key + 8 * n + 2 * t + i % 2;
                    if key > past + rows[i / 2] || key >= positions {
                        s[n][i] = f32::NEG_INFINITY;
                    }
                    tile_max[i / 2] = tile_max[i / 2].max(s[n][i]);
                });
            });
            let new_max = [
                max[0].max(quad_max(tile_max[0])),
                max[1].max(quad_max(tile_max[1])),
            ];
            let fix = [(max[0] - new_max[0]).exp(), (max[1] - new_max[1]).exp()];
            let mut tile_sum = [0.0f32; 2];
            unroll!(n in [0, 1, 2, 3, 4, 5, 6, 7] {
                unroll!(i in [0, 1, 2, 3] {
                    s[n][i] = (s[n][i] - new_max[i / 2]).exp();
                    tile_sum[i / 2] += s[n][i];
                    o[n][i] *= fix[i / 2];
                });
            });
            sum = [
                sum[0] * fix[0] + quad_sum(tile_sum[0]),
                sum[1] * fix[1] + quad_sum(tile_sum[1]),
            ];
            max = new_max;

            // 3. Les poids, arrondis en `bf16`, mélangent les valeurs.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe { product_from!(o, s, VALUES, group, t) }
            thread::sync_threads();
            first_key += WIDE_TILE;
        }

        let start = head * queries * value_dim;
        store!(
            out,
            o,
            rows,
            queries,
            start,
            value_dim,
            t,
            [1.0 / sum[0], 1.0 / sum[1]]
        );
        if t == 0 {
            unroll!(r in [0, 1] {
                if rows[r] < queries {
                    // SAFETY: un seul thread par ligne écrit.
                    unsafe {
                        *log_sum.add(head * queries + rows[r]) = max[r] + sum[r].ln();
                    }
                }
            });
        }
    }

    /// Passe arrière côté clés sur les Tensor Cores : `attention_backward_keys`
    /// en précision mixte.
    ///
    /// Un bloc de quatre warps prend 64 clés, seize par warp, et parcourt les
    /// requêtes par tuiles de 64. Les lignes et les colonnes sont celles des
    /// transposées : chaque warp calcule `Sᵀ = K · Qᵀ` et `dPᵀ = V · dOᵀ`
    /// pour ses clés, puis accumule `dV += Pᵀ · dO` et `dK += dSᵀ · Q`, où
    /// `Pᵀ` et `dSᵀ` servent directement de fragments `A`. Requêtes et
    /// gradient de la sortie sont rangés deux fois en mémoire partagée, par
    /// paires le long de chacun de leurs axes, pour servir aux deux sortes de
    /// produits.
    ///
    /// # Safety
    ///
    /// Comme `attention_backward_keys`, avec des tuiles de [`WIDE_TILE`]
    /// clés et des blocs de quatre warps.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, block = (32, 4, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_backward_keys_bf16(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        grad: &[f32],
        log_sum: &[f32],
        delta: &[f32],
        queries: u32,
        positions: u32,
        key_dim: u32,
        value_dim: u32,
        scale: f32,
        grad_k: *mut f32,
        grad_v: *mut f32,
    ) {
        static mut QUERIES: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut QUERIES_T: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut GRADS: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut GRADS_T: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut ROW_LOG_SUM: SharedArray<f32, WIDE_TILE> = SharedArray::UNINIT;
        static mut ROW_DELTA: SharedArray<f32, WIDE_TILE> = SharedArray::UNINIT;

        let (queries, positions) = (queries as usize, positions as usize);
        let (key_dim, value_dim) = (key_dim as usize, value_dim as usize);
        let (lane, warp) = (
            thread::threadIdx_x() as usize,
            thread::threadIdx_y() as usize,
        );
        let (slot, group, t) = (warp * 32 + lane, lane / 4, lane % 4);
        let head = thread::blockIdx_y() as usize;
        let first_key = thread::blockIdx_x() as usize * WIDE_TILE;
        let past = positions - queries;
        // Les deux clés de ce thread.
        let keys = [
            first_key + 16 * warp + group,
            first_key + 16 * warp + group + 8,
        ];

        let (k_start, v_start) = (head * positions * key_dim, head * positions * value_dim);
        let (q_start, g_start) = (head * queries * key_dim, head * queries * value_dim);
        let mut ka = [[0u32; 4]; 4];
        let mut va = [[0u32; 4]; 4];
        unroll!(kk in [0, 1, 2, 3] {
            let column = 16 * kk + 2 * t;
            ka[kk] = global_fragment(k, k_start, keys, positions, column, key_dim, 1.0);
            va[kk] = global_fragment(v, v_start, keys, positions, column, value_dim, 1.0);
        });
        let mut dk = [[0.0f32; 4]; 8];
        let mut dv = [[0.0f32; 4]; 8];

        let mut first_query = (first_key.saturating_sub(past) / WIDE_TILE) * WIDE_TILE;
        while first_query < queries {
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                load_pairs!(
                    QUERIES,
                    q,
                    q_start,
                    first_query,
                    queries,
                    key_dim,
                    scale,
                    slot
                );
                load_pairs_transposed!(
                    QUERIES_T,
                    q,
                    q_start,
                    first_query,
                    queries,
                    key_dim,
                    scale,
                    slot
                );
                load_pairs!(
                    GRADS,
                    grad,
                    g_start,
                    first_query,
                    queries,
                    value_dim,
                    1.0,
                    slot
                );
                load_pairs_transposed!(
                    GRADS_T,
                    grad,
                    g_start,
                    first_query,
                    queries,
                    value_dim,
                    1.0,
                    slot
                );
                if slot < WIDE_TILE {
                    let query = first_query + slot;
                    let inside = query < queries;
                    ROW_LOG_SUM[slot] = if inside {
                        log_sum[head * queries + query]
                    } else {
                        0.0
                    };
                    ROW_DELTA[slot] = if inside {
                        delta[head * queries + query]
                    } else {
                        0.0
                    };
                }
            }
            thread::sync_threads();

            // 1. `Sᵀ` et `dPᵀ` : huit tuiles de 8 requêtes.
            let mut s = [[0.0f32; 4]; 8];
            let mut dp = [[0.0f32; 4]; 8];
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                product!(s, ka, QUERIES, group, t);
                product!(dp, va, GRADS, group, t);
            }

            // 2. Les poids `Pᵀ` (dans `s`) et le gradient des scores `dSᵀ`
            //    (dans `dp`). L'élément `i` de la tuile `n` relie la clé
            //    `keys[i / 2]` à la requête de la colonne `8n + 2t + i % 2`.
            unroll!(n in [0, 1, 2, 3, 4, 5, 6, 7] {
                unroll!(i in [0, 1, 2, 3] {
                    let (key, column) = (keys[i / 2], 8 * n + 2 * t + i % 2);
                    let query = first_query + column;
                    let visible = query < queries && key < positions && key <= past + query;
                    // SAFETY: les statistiques ne sont plus que lues.
                    let (row_log_sum, row_delta) =
                        unsafe { (ROW_LOG_SUM[column], ROW_DELTA[column]) };
                    let p = if visible {
                        (s[n][i] - row_log_sum).exp()
                    } else {
                        0.0
                    };
                    s[n][i] = p;
                    dp[n][i] = p * (dp[n][i] - row_delta);
                });
            });

            // 3. `dV += Pᵀ · dO` et `dK += dSᵀ · Q`.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                product_from!(dv, s, GRADS_T, group, t);
                product_from!(dk, dp, QUERIES_T, group, t);
            }
            thread::sync_threads();
            first_query += WIDE_TILE;
        }

        let start = head * positions;
        store!(
            grad_k,
            dk,
            keys,
            positions,
            start * key_dim,
            key_dim,
            t,
            [1.0f32; 2]
        );
        store!(
            grad_v,
            dv,
            keys,
            positions,
            start * value_dim,
            value_dim,
            t,
            [1.0f32; 2]
        );
    }

    /// Passe arrière côté requêtes sur les Tensor Cores :
    /// `attention_backward_queries` en précision mixte. Les tuiles se
    /// parcourent comme dans `attention_forward_bf16` ; chaque warp garde
    /// en registres ses requêtes et le gradient de leurs sorties.
    ///
    /// # Safety
    ///
    /// Comme `attention_backward_queries`, avec des tuiles de [`WIDE_TILE`]
    /// requêtes et des blocs de quatre warps.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, block = (32, 4, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_backward_queries_bf16(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        grad: &[f32],
        log_sum: &[f32],
        delta: &[f32],
        queries: u32,
        positions: u32,
        key_dim: u32,
        value_dim: u32,
        scale: f32,
        grad_q: *mut f32,
    ) {
        static mut KEYS: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut KEYS_T: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;
        static mut VALUES: SharedArray<u32, { WIDE_TILE * PAIR_ROW }> = SharedArray::UNINIT;

        let (queries, positions) = (queries as usize, positions as usize);
        let (key_dim, value_dim) = (key_dim as usize, value_dim as usize);
        let (lane, warp) = (
            thread::threadIdx_x() as usize,
            thread::threadIdx_y() as usize,
        );
        let (slot, group, t) = (warp * 32 + lane, lane / 4, lane % 4);
        let head = thread::blockIdx_y() as usize;
        let first_query = thread::blockIdx_x() as usize * WIDE_TILE;
        let past = positions - queries;
        let rows = [
            first_query + 16 * warp + group,
            first_query + 16 * warp + group + 8,
        ];

        let (q_start, g_start) = (head * queries * key_dim, head * queries * value_dim);
        let mut qa = [[0u32; 4]; 4];
        let mut ga = [[0u32; 4]; 4];
        unroll!(kk in [0, 1, 2, 3] {
            let column = 16 * kk + 2 * t;
            qa[kk] = global_fragment(q, q_start, rows, queries, column, key_dim, scale);
            ga[kk] = global_fragment(grad, g_start, rows, queries, column, value_dim, 1.0);
        });
        // Les statistiques des deux lignes. Celles qui sont hors de la
        // matrice ne seront pas écrites : leur valeur importe peu.
        let mut row_log_sum = [0.0f32; 2];
        let mut row_delta = [0.0f32; 2];
        unroll!(r in [0, 1] {
            if rows[r] < queries {
                row_log_sum[r] = log_sum[head * queries + rows[r]];
                row_delta[r] = delta[head * queries + rows[r]];
            }
        });
        let mut dq = [[0.0f32; 4]; 8];

        let end = positions.min(past + first_query + WIDE_TILE);
        let (k_start, v_start) = (head * positions * key_dim, head * positions * value_dim);
        let mut first_key = 0;
        while first_key < end {
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                load_pairs!(KEYS, k, k_start, first_key, positions, key_dim, 1.0, slot);
                load_pairs_transposed!(
                    KEYS_T, k, k_start, first_key, positions, key_dim, 1.0, slot
                );
                load_pairs!(
                    VALUES, v, v_start, first_key, positions, value_dim, 1.0, slot
                );
            }
            thread::sync_threads();

            let mut s = [[0.0f32; 4]; 8];
            let mut dp = [[0.0f32; 4]; 8];
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                product!(s, qa, KEYS, group, t);
                product!(dp, ga, VALUES, group, t);
            }
            // `dS`, dans `dp`.
            unroll!(n in [0, 1, 2, 3, 4, 5, 6, 7] {
                unroll!(i in [0, 1, 2, 3] {
                    let (row, key) = (rows[i / 2], first_key + 8 * n + 2 * t + i % 2);
                    let visible = row < queries && key < positions && key <= past + row;
                    let p = if visible {
                        (s[n][i] - row_log_sum[i / 2]).exp()
                    } else {
                        0.0
                    };
                    dp[n][i] = p * (dp[n][i] - row_delta[i / 2]);
                });
            });
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe { product_from!(dq, dp, KEYS_T, group, t) }
            thread::sync_threads();
            first_key += WIDE_TILE;
        }

        store!(grad_q, dq, rows, queries, q_start, key_dim, t, [scale; 2]);
    }
}
