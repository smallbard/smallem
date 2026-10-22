//! L'attention fusionnée en `f32` : la passe avant et les deux passes
//! arrière, sans jamais ranger la matrice des scores.

use super::*;

/// Pas d'une ligne de tuile `[32, 64]` en mémoire partagée : une case de
/// plus que la largeur. Les 32 bancs de la mémoire partagée servent chacun
/// une adresse par cycle ; avec un pas de 64, deux threads qui lisent la
/// même colonne de lignes différentes tomberaient dans le même banc et
/// attendraient leur tour. Avec 65, chaque ligne commence un banc plus loin.
const ROW: usize = ATTENTION_MAX_DIM + 1;

/// Pas d'une ligne de tuile `[32, 32]`, pour la même raison.
const SQUARE_ROW: usize = ATTENTION_TILE + 1;

/// Maximum de `value` sur les 16 threads du demi-warp : les threads
/// `0 … 15` d'un côté, `16 … 31` de l'autre. Dans les kernels d'attention,
/// ce sont les 16 threads qui se partagent une même ligne.
#[inline(always)]
pub fn half_warp_max(mut value: f32) -> f32 {
    let mut distance = WARP as u32 / 4;
    while distance > 0 {
        let other = warp::shuffle_xor_f32(value, distance);
        if other > value {
            value = other;
        }
        distance /= 2;
    }
    value
}

/// Somme de `value` sur les 16 threads du demi-warp.
#[inline(always)]
pub fn half_warp_sum(mut value: f32) -> f32 {
    let mut distance = WARP as u32 / 4;
    while distance > 0 {
        value += warp::shuffle_xor_f32(value, distance);
        distance /= 2;
    }
    value
}

#[cuda_module]
pub mod device {
    use super::*;

    /// L'attention causale en un seul kernel : la passe avant de
    /// FlashAttention.
    ///
    /// Un bloc de `16 × 16` threads prend une tuile de 32 requêtes d'une
    /// tête, et parcourt les clés et les valeurs par tuiles de 32. La matrice
    /// des scores n'existe jamais en entier : seule la tuile `32 × 32` du
    /// moment passe par la mémoire partagée.
    ///
    /// Le softmax demande le maximum et la somme de toute une ligne, que l'on
    /// ne connaît qu'à la fin. Le **softmax en ligne** les tient à jour tuile
    /// après tuile : quand une tuile fait monter le maximum de `m` à `m'`,
    /// tout ce qui a été accumulé jusque-là l'a été avec `exp(s - m)`, et il
    /// suffit de le multiplier par `exp(m - m')` pour le remettre à jour.
    /// La division par la somme attend la toute fin.
    ///
    /// Chaque thread tient deux lignes de la tuile : deux scores par ligne,
    /// quatre composantes de la sortie par ligne. Les 16 threads d'une même
    /// ligne forment un demi-warp et réunissent maximum et somme par
    /// échanges de registres.
    ///
    /// Les tuiles qui ne contiennent que du futur ne sont pas parcourues :
    /// pour une séquence entière, c'est la moitié du travail. La sortie
    /// `log_sum` garde `m + ln(somme)` pour chaque ligne, tout ce dont la
    /// passe arrière aura besoin pour recalculer les poids.
    ///
    /// # Safety
    ///
    /// La grille a une tuile de requêtes par bloc en `x` et une tête par
    /// bloc en `y` ; `out` a `têtes × queries × value_dim` éléments,
    /// `log_sum` `têtes × queries`. Les largeurs ne dépassent pas
    /// [`ATTENTION_MAX_DIM`].
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_forward(
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
        static mut TILE_Q: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_K: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_V: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_P: SharedArray<f32, { ATTENTION_TILE * SQUARE_ROW }> = SharedArray::UNINIT;

        const TILE: usize = ATTENTION_TILE;
        let (queries, positions) = (queries as usize, positions as usize);
        let (key_dim, value_dim) = (key_dim as usize, value_dim as usize);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let slot = ty * 16 + tx;
        let head = thread::blockIdx_y() as usize;
        let first_query = thread::blockIdx_x() as usize * TILE;
        // Les requêtes sont les dernières positions : la requête `i` est à
        // la position `past + i`.
        let past = positions - queries;

        // Les requêtes de la tuile, déjà multipliées par `scale`. Hors des
        // matrices, des zéros, qui ne changent aucun produit.
        let q_start = head * queries * key_dim;
        let mut cell = slot;
        while cell < TILE * ATTENTION_MAX_DIM {
            let (row, d) = (cell / ATTENTION_MAX_DIM, cell % ATTENTION_MAX_DIM);
            let query = first_query + row;
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                TILE_Q[row * ROW + d] = if query < queries && d < key_dim {
                    q[q_start + query * key_dim + d] * scale
                } else {
                    0.0
                };
            }
            cell += 256;
        }

        // Les deux lignes de ce thread, et leurs positions.
        let (row0, row1) = (ty * 2, ty * 2 + 1);
        let (position0, position1) = (past + first_query + row0, past + first_query + row1);
        // Maximum courant, somme courante, et quatre composantes de la
        // sortie, pour chacune des deux lignes.
        let (mut max0, mut max1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (mut sum0, mut sum1) = (0.0f32, 0.0f32);
        let (mut o00, mut o01, mut o02, mut o03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut o10, mut o11, mut o12, mut o13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        // Au-delà de la dernière position de la tuile, tout est futur.
        let end = positions.min(past + first_query + TILE);
        let (k_start, v_start) = (head * positions * key_dim, head * positions * value_dim);
        let mut first_key = 0;
        while first_key < end {
            // 1. Les clés et les valeurs de la tuile.
            let mut cell = slot;
            while cell < TILE * ATTENTION_MAX_DIM {
                let (row, d) = (cell / ATTENTION_MAX_DIM, cell % ATTENTION_MAX_DIM);
                let key = first_key + row;
                // SAFETY: chaque case est écrite par un seul thread.
                unsafe {
                    TILE_K[row * ROW + d] = if key < positions && d < key_dim {
                        k[k_start + key * key_dim + d]
                    } else {
                        0.0
                    };
                    TILE_V[row * ROW + d] = if key < positions && d < value_dim {
                        v[v_start + key * value_dim + d]
                    } else {
                        0.0
                    };
                }
                cell += 256;
            }
            thread::sync_threads();

            // 2. Quatre scores : lignes `row0` et `row1`, clés `2 tx` et
            //    `2 tx + 1`. Le futur reçoit moins l'infini.
            let (col0, col1) = (tx * 2, tx * 2 + 1);
            let (mut s00, mut s01, mut s10, mut s11) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                let mut d = 0;
                while d < key_dim {
                    let (q0, q1) = (TILE_Q[row0 * ROW + d], TILE_Q[row1 * ROW + d]);
                    let (k0, k1) = (TILE_K[col0 * ROW + d], TILE_K[col1 * ROW + d]);
                    s00 += q0 * k0;
                    s01 += q0 * k1;
                    s10 += q1 * k0;
                    s11 += q1 * k1;
                    d += 1;
                }
            }
            let (key0, key1) = (first_key + col0, first_key + col1);
            if key0 > position0 || key0 >= positions {
                s00 = f32::NEG_INFINITY;
            }
            if key1 > position0 || key1 >= positions {
                s01 = f32::NEG_INFINITY;
            }
            if key0 > position1 || key0 >= positions {
                s10 = f32::NEG_INFINITY;
            }
            if key1 > position1 || key1 >= positions {
                s11 = f32::NEG_INFINITY;
            }

            // 3. Le softmax en ligne. La première tuile contient toujours la
            //    clé 0, que toute requête voit : le maximum est fini dès
            //    elle, et `exp(-∞ - m')` vaut bien 0.
            let new_max0 = max0.max(half_warp_max(s00.max(s01)));
            let new_max1 = max1.max(half_warp_max(s10.max(s11)));
            let (p00, p01) = ((s00 - new_max0).exp(), (s01 - new_max0).exp());
            let (p10, p11) = ((s10 - new_max1).exp(), (s11 - new_max1).exp());
            let (fix0, fix1) = ((max0 - new_max0).exp(), (max1 - new_max1).exp());
            sum0 = sum0 * fix0 + half_warp_sum(p00 + p01);
            sum1 = sum1 * fix1 + half_warp_sum(p10 + p11);
            (max0, max1) = (new_max0, new_max1);
            o00 *= fix0;
            o01 *= fix0;
            o02 *= fix0;
            o03 *= fix0;
            o10 *= fix1;
            o11 *= fix1;
            o12 *= fix1;
            o13 *= fix1;
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                TILE_P[row0 * SQUARE_ROW + col0] = p00;
                TILE_P[row0 * SQUARE_ROW + col1] = p01;
                TILE_P[row1 * SQUARE_ROW + col0] = p10;
                TILE_P[row1 * SQUARE_ROW + col1] = p11;
            }
            thread::sync_threads();

            // 4. Les poids mélangent les valeurs : composantes `4 tx` à
            //    `4 tx + 3` des deux lignes.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                let d = tx * 4;
                let mut key = 0;
                while key < TILE {
                    let (p0, p1) = (
                        TILE_P[row0 * SQUARE_ROW + key],
                        TILE_P[row1 * SQUARE_ROW + key],
                    );
                    let at = key * ROW + d;
                    let (v0, v1, v2, v3) =
                        (TILE_V[at], TILE_V[at + 1], TILE_V[at + 2], TILE_V[at + 3]);
                    o00 += p0 * v0;
                    o01 += p0 * v1;
                    o02 += p0 * v2;
                    o03 += p0 * v3;
                    o10 += p1 * v0;
                    o11 += p1 * v1;
                    o12 += p1 * v2;
                    o13 += p1 * v3;
                    key += 1;
                }
            }
            // 5. Personne ne recharge avant que tout le bloc ait lu.
            thread::sync_threads();
            first_key += TILE;
        }

        // La division par la somme, enfin, et ce que garde la passe arrière.
        let rows = [
            (row0, max0, sum0, [o00, o01, o02, o03]),
            (row1, max1, sum1, [o10, o11, o12, o13]),
        ];
        let mut i = 0;
        while i < 2 {
            let (row, max, sum, values) = rows[i];
            let query = first_query + row;
            if query < queries {
                let line = head * queries + query;
                let mut j = 0;
                while j < 4 {
                    let d = tx * 4 + j;
                    if d < value_dim {
                        // SAFETY: `(line, d)` n'appartient qu'à ce thread.
                        unsafe {
                            *out.add(line * value_dim + d) = values[j] / sum;
                        }
                    }
                    j += 1;
                }
                if tx == 0 {
                    // SAFETY: un seul thread par ligne écrit.
                    unsafe {
                        *log_sum.add(line) = max + sum.ln();
                    }
                }
            }
            i += 1;
        }
    }

    /// Passe arrière de l'attention fusionnée, côté clés : les gradients
    /// des clés et des valeurs.
    ///
    /// Un bloc prend une tuile de 32 clés d'une tête et parcourt les
    /// requêtes qui les voient. Pour chaque tuile de requêtes, il recalcule
    /// les poids `P = exp(s - log_sum)`, sans softmax à refaire puisque
    /// `log_sum` contient déjà maximum et somme, puis
    ///
    /// ```text
    /// dV += Pᵀ · dO
    /// dS  = P × (dO · Vᵀ - delta)        delta = Σ dO × O, par ligne
    /// dK += dSᵀ · Q
    /// ```
    ///
    /// Les requêtes étant chargées multipliées par `scale`, `dK` reçoit
    /// directement le bon facteur. Chaque thread tient deux clés : quatre
    /// composantes de chacun des deux gradients par clé.
    ///
    /// # Safety
    ///
    /// La grille a une tuile de clés par bloc en `x` et une tête en `y` ;
    /// `grad_k` et `grad_v` ont la taille de `k` et de `v`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_backward_keys(
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
        static mut TILE_Q: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_G: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_K: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_V: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_P: SharedArray<f32, { ATTENTION_TILE * SQUARE_ROW }> = SharedArray::UNINIT;
        static mut TILE_DS: SharedArray<f32, { ATTENTION_TILE * SQUARE_ROW }> = SharedArray::UNINIT;
        static mut ROW_LOG_SUM: SharedArray<f32, ATTENTION_TILE> = SharedArray::UNINIT;
        static mut ROW_DELTA: SharedArray<f32, ATTENTION_TILE> = SharedArray::UNINIT;

        const TILE: usize = ATTENTION_TILE;
        let (queries, positions) = (queries as usize, positions as usize);
        let (key_dim, value_dim) = (key_dim as usize, value_dim as usize);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let slot = ty * 16 + tx;
        let head = thread::blockIdx_y() as usize;
        let first_key = thread::blockIdx_x() as usize * TILE;
        let past = positions - queries;

        let (q_start, k_start) = (head * queries * key_dim, head * positions * key_dim);
        let (g_start, v_start) = (head * queries * value_dim, head * positions * value_dim);
        let mut cell = slot;
        while cell < TILE * ATTENTION_MAX_DIM {
            let (row, d) = (cell / ATTENTION_MAX_DIM, cell % ATTENTION_MAX_DIM);
            let key = first_key + row;
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                TILE_K[row * ROW + d] = if key < positions && d < key_dim {
                    k[k_start + key * key_dim + d]
                } else {
                    0.0
                };
                TILE_V[row * ROW + d] = if key < positions && d < value_dim {
                    v[v_start + key * value_dim + d]
                } else {
                    0.0
                };
            }
            cell += 256;
        }

        // Les deux clés de ce thread pour les gradients ; les deux requêtes
        // de chaque tuile qu'il calcule, `2 tx` et `2 tx + 1`.
        let (key0, key1) = (ty * 2, ty * 2 + 1);
        let (row0, row1) = (tx * 2, tx * 2 + 1);
        let (mut dk00, mut dk01, mut dk02, mut dk03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut dk10, mut dk11, mut dk12, mut dk13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut dv00, mut dv01, mut dv02, mut dv03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut dv10, mut dv11, mut dv12, mut dv13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        // La première requête qui voit la première clé de la tuile.
        let mut first_query = (first_key.saturating_sub(past) / TILE) * TILE;
        while first_query < queries {
            // 1. Requêtes, gradient de la sortie, et les deux statistiques
            //    de chaque ligne.
            let mut cell = slot;
            while cell < TILE * ATTENTION_MAX_DIM {
                let (row, d) = (cell / ATTENTION_MAX_DIM, cell % ATTENTION_MAX_DIM);
                let query = first_query + row;
                // SAFETY: chaque case est écrite par un seul thread.
                unsafe {
                    TILE_Q[row * ROW + d] = if query < queries && d < key_dim {
                        q[q_start + query * key_dim + d] * scale
                    } else {
                        0.0
                    };
                    TILE_G[row * ROW + d] = if query < queries && d < value_dim {
                        grad[g_start + query * value_dim + d]
                    } else {
                        0.0
                    };
                }
                cell += 256;
            }
            if slot < TILE {
                let query = first_query + slot;
                // SAFETY: chaque case est écrite par un seul thread.
                unsafe {
                    ROW_LOG_SUM[slot] = if query < queries {
                        log_sum[head * queries + query]
                    } else {
                        0.0
                    };
                    ROW_DELTA[slot] = if query < queries {
                        delta[head * queries + query]
                    } else {
                        0.0
                    };
                }
            }
            thread::sync_threads();

            // 2. Poids et gradient des scores, pour les clés `key0`, `key1`
            //    et les requêtes `row0`, `row1`.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues ;
            // chaque case de `TILE_P` et `TILE_DS` est écrite par un seul
            // thread.
            unsafe {
                let mut i = 0;
                while i < 4 {
                    let (key, row) = (
                        if i < 2 { key0 } else { key1 },
                        if i % 2 == 0 { row0 } else { row1 },
                    );
                    let (mut score, mut grad_weight) = (0.0f32, 0.0f32);
                    let mut d = 0;
                    while d < key_dim {
                        score += TILE_Q[row * ROW + d] * TILE_K[key * ROW + d];
                        d += 1;
                    }
                    let mut d = 0;
                    while d < value_dim {
                        grad_weight += TILE_G[row * ROW + d] * TILE_V[key * ROW + d];
                        d += 1;
                    }
                    let query = first_query + row;
                    let visible = query < queries
                        && first_key + key < positions
                        && first_key + key <= past + query;
                    let p = if visible {
                        (score - ROW_LOG_SUM[row]).exp()
                    } else {
                        0.0
                    };
                    TILE_P[key * SQUARE_ROW + row] = p;
                    TILE_DS[key * SQUARE_ROW + row] = p * (grad_weight - ROW_DELTA[row]);
                    i += 1;
                }
            }
            thread::sync_threads();

            // 3. Les deux gradients, composantes `4 tx` à `4 tx + 3`.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                let d = tx * 4;
                let mut row = 0;
                while row < TILE {
                    let at = row * ROW + d;
                    let (g0, g1, g2, g3) =
                        (TILE_G[at], TILE_G[at + 1], TILE_G[at + 2], TILE_G[at + 3]);
                    let (q0, q1, q2, q3) =
                        (TILE_Q[at], TILE_Q[at + 1], TILE_Q[at + 2], TILE_Q[at + 3]);
                    let (p0, p1) = (
                        TILE_P[key0 * SQUARE_ROW + row],
                        TILE_P[key1 * SQUARE_ROW + row],
                    );
                    let (s0, s1) = (
                        TILE_DS[key0 * SQUARE_ROW + row],
                        TILE_DS[key1 * SQUARE_ROW + row],
                    );
                    dv00 += p0 * g0;
                    dv01 += p0 * g1;
                    dv02 += p0 * g2;
                    dv03 += p0 * g3;
                    dv10 += p1 * g0;
                    dv11 += p1 * g1;
                    dv12 += p1 * g2;
                    dv13 += p1 * g3;
                    dk00 += s0 * q0;
                    dk01 += s0 * q1;
                    dk02 += s0 * q2;
                    dk03 += s0 * q3;
                    dk10 += s1 * q0;
                    dk11 += s1 * q1;
                    dk12 += s1 * q2;
                    dk13 += s1 * q3;
                    row += 1;
                }
            }
            thread::sync_threads();
            first_query += TILE;
        }

        let keys = [
            (key0, [dk00, dk01, dk02, dk03], [dv00, dv01, dv02, dv03]),
            (key1, [dk10, dk11, dk12, dk13], [dv10, dv11, dv12, dv13]),
        ];
        let mut i = 0;
        while i < 2 {
            let (key, dk, dv) = keys[i];
            let position = first_key + key;
            if position < positions {
                let line = head * positions + position;
                let mut j = 0;
                while j < 4 {
                    let d = tx * 4 + j;
                    // SAFETY: `(line, d)` n'appartient qu'à ce thread.
                    unsafe {
                        if d < key_dim {
                            *grad_k.add(line * key_dim + d) = dk[j];
                        }
                        if d < value_dim {
                            *grad_v.add(line * value_dim + d) = dv[j];
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }

    /// Passe arrière de l'attention fusionnée, côté requêtes : `dQ = scale ×
    /// dS · K`.
    ///
    /// FlashAttention-2 calcule `dQ` dans le même kernel que `dK` et `dV`,
    /// chaque bloc de clés y ajoutant sa part par des additions atomiques.
    /// Ici, un second kernel parcourt les tuiles comme la passe avant, une
    /// tuile de requêtes par bloc, et recalcule les poids et `dS` : trois
    /// produits de plus, mais aucune addition atomique, donc un résultat qui
    /// ne dépend pas de l'ordre d'exécution des blocs.
    ///
    /// # Safety
    ///
    /// Grille de la passe avant ; `grad_q` a la taille de `q`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_backward_queries(
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
        static mut TILE_Q: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_G: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_K: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_V: SharedArray<f32, { ATTENTION_TILE * ROW }> = SharedArray::UNINIT;
        static mut TILE_DS: SharedArray<f32, { ATTENTION_TILE * SQUARE_ROW }> = SharedArray::UNINIT;

        const TILE: usize = ATTENTION_TILE;
        let (queries, positions) = (queries as usize, positions as usize);
        let (key_dim, value_dim) = (key_dim as usize, value_dim as usize);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let slot = ty * 16 + tx;
        let head = thread::blockIdx_y() as usize;
        let first_query = thread::blockIdx_x() as usize * TILE;
        let past = positions - queries;

        let (q_start, g_start) = (head * queries * key_dim, head * queries * value_dim);
        let mut cell = slot;
        while cell < TILE * ATTENTION_MAX_DIM {
            let (row, d) = (cell / ATTENTION_MAX_DIM, cell % ATTENTION_MAX_DIM);
            let query = first_query + row;
            // SAFETY: chaque case est écrite par un seul thread.
            unsafe {
                TILE_Q[row * ROW + d] = if query < queries && d < key_dim {
                    q[q_start + query * key_dim + d] * scale
                } else {
                    0.0
                };
                TILE_G[row * ROW + d] = if query < queries && d < value_dim {
                    grad[g_start + query * value_dim + d]
                } else {
                    0.0
                };
            }
            cell += 256;
        }

        let (row0, row1) = (ty * 2, ty * 2 + 1);
        let (query0, query1) = (first_query + row0, first_query + row1);
        // Statistiques des deux lignes de ce thread : les lignes hors de la
        // matrice ne sont jamais écrites, leurs valeurs importent peu.
        let stat = |values: &[f32], query: usize| {
            if query < queries {
                values[head * queries + query]
            } else {
                0.0
            }
        };
        let (log_sum0, log_sum1) = (stat(log_sum, query0), stat(log_sum, query1));
        let (delta0, delta1) = (stat(delta, query0), stat(delta, query1));
        let (mut dq00, mut dq01, mut dq02, mut dq03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut dq10, mut dq11, mut dq12, mut dq13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        let end = positions.min(past + first_query + TILE);
        let (k_start, v_start) = (head * positions * key_dim, head * positions * value_dim);
        let mut first_key = 0;
        while first_key < end {
            let mut cell = slot;
            while cell < TILE * ATTENTION_MAX_DIM {
                let (row, d) = (cell / ATTENTION_MAX_DIM, cell % ATTENTION_MAX_DIM);
                let key = first_key + row;
                // SAFETY: chaque case est écrite par un seul thread.
                unsafe {
                    TILE_K[row * ROW + d] = if key < positions && d < key_dim {
                        k[k_start + key * key_dim + d]
                    } else {
                        0.0
                    };
                    TILE_V[row * ROW + d] = if key < positions && d < value_dim {
                        v[v_start + key * value_dim + d]
                    } else {
                        0.0
                    };
                }
                cell += 256;
            }
            thread::sync_threads();

            // SAFETY: après la barrière, les tuiles ne sont plus que lues ;
            // chaque case de `TILE_DS` est écrite par un seul thread.
            unsafe {
                let mut i = 0;
                while i < 4 {
                    let (row, col) = (if i < 2 { row0 } else { row1 }, tx * 2 + i % 2);
                    let (mut score, mut grad_weight) = (0.0f32, 0.0f32);
                    let mut d = 0;
                    while d < key_dim {
                        score += TILE_Q[row * ROW + d] * TILE_K[col * ROW + d];
                        d += 1;
                    }
                    let mut d = 0;
                    while d < value_dim {
                        grad_weight += TILE_G[row * ROW + d] * TILE_V[col * ROW + d];
                        d += 1;
                    }
                    let (query, log_sum, delta) = if i < 2 {
                        (query0, log_sum0, delta0)
                    } else {
                        (query1, log_sum1, delta1)
                    };
                    let key = first_key + col;
                    let visible = query < queries && key < positions && key <= past + query;
                    let p = if visible {
                        (score - log_sum).exp()
                    } else {
                        0.0
                    };
                    TILE_DS[row * SQUARE_ROW + col] = p * (grad_weight - delta);
                    i += 1;
                }
            }
            thread::sync_threads();

            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                let d = tx * 4;
                let mut col = 0;
                while col < TILE {
                    let at = col * ROW + d;
                    let (k0, k1, k2, k3) =
                        (TILE_K[at], TILE_K[at + 1], TILE_K[at + 2], TILE_K[at + 3]);
                    let (s0, s1) = (
                        TILE_DS[row0 * SQUARE_ROW + col],
                        TILE_DS[row1 * SQUARE_ROW + col],
                    );
                    dq00 += s0 * k0;
                    dq01 += s0 * k1;
                    dq02 += s0 * k2;
                    dq03 += s0 * k3;
                    dq10 += s1 * k0;
                    dq11 += s1 * k1;
                    dq12 += s1 * k2;
                    dq13 += s1 * k3;
                    col += 1;
                }
            }
            thread::sync_threads();
            first_key += TILE;
        }

        let rows = [
            (query0, [dq00, dq01, dq02, dq03]),
            (query1, [dq10, dq11, dq12, dq13]),
        ];
        let mut i = 0;
        while i < 2 {
            let (query, dq) = rows[i];
            if query < queries {
                let line = head * queries + query;
                let mut j = 0;
                while j < 4 {
                    let d = tx * 4 + j;
                    if d < key_dim {
                        // SAFETY: `(line, d)` n'appartient qu'à ce thread.
                        unsafe {
                            *grad_q.add(line * key_dim + d) = scale * dq[j];
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }
}
