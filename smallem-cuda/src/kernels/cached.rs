//! L'attention de la génération, sur le cache des clés et des valeurs par
//! pages, et la réunion de ses tranches.

use super::*;

#[cuda_module]
pub mod device {
    use super::*;

    /// L'attention de la génération : chaque requête lit les clés et les
    /// valeurs de sa séquence dans les tables du cache, à travers la liste
    /// de ses lignes (voir `CachedAttention` dans `smallem::backend::ops`).
    ///
    /// En génération, une séquence n'apporte qu'une requête par token.
    /// Les kernels de l'attention fusionnée, taillés pour des tuiles de 32
    /// ou 64 requêtes, n'auraient alors qu'un bloc par tête, qui parcourrait
    /// seul toutes les clés. Ici, le parallélisme vient des clés : chaque
    /// bloc traite une tranche de `chunk` positions pour une requête et une
    /// tête, et les tranches sont réunies ensuite. C'est l'idée du
    /// *flash-decoding*.
    ///
    /// Un bloc a 4 warps de 4 groupes de 8 threads. Chaque groupe lit une
    /// clé à la fois : ses 8 threads se partagent les composantes, 8 chacun
    /// au plus, et réunissent leur produit scalaire par trois échanges entre
    /// registres. Chaque groupe tient son propre softmax en ligne (maximum,
    /// somme, sortie non normalisée), et les 16 groupes sont réunis en
    /// mémoire partagée à la fin.
    ///
    /// Avec une seule tranche, la sortie est écrite normalisée. Sinon chaque
    /// tranche écrit sa sortie non normalisée dans `out`, `[tranches, R, H,
    /// Dv]`, et son maximum et sa somme dans `stats`, `[tranches, R, H, 2]`,
    /// que `attention_cached_combine` réunit.
    ///
    /// # Safety
    ///
    /// La grille couvre `R × H` en `x` et les tranches en `y`, par blocs de
    /// `32 × 4` threads. Les lignes sont en `x`, qui accepte bien plus de
    /// blocs que `y` : une passe peut lire des centaines d'amorces. Les têtes font au plus 64 composantes. `out` et
    /// `stats` ont la taille ci-dessus, et les lignes de `slots` sont des
    /// lignes des tables.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, block = (32, 4, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_cached(
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        slots: &[f32],
        rows: &[f32],
        dims: CachedDims,
        chunk: u32,
        scale: f32,
        out: *mut f32,
        stats: *mut f32,
    ) {
        // Par groupe : maximum, somme, puis 64 composantes de sortie.
        static mut PARTS: SharedArray<f32, 1056> = SharedArray::UNINIT;
        const STRIDE: usize = ATTENTION_MAX_DIM + 2;

        let lane = thread::threadIdx_x() as usize;
        let group = thread::threadIdx_y() as usize * 4 + lane / 8;
        let sub = lane % 8;
        let heads = dims.heads as usize;
        let (key_dim, value_dim) = (dims.key_dim as usize, dims.value_dim as usize);
        let value_stride = dims.value_stride as usize;
        let kv_heads = dims.kv_heads as usize;
        let line = thread::blockIdx_x() as usize;
        let (row, head) = (line / heads, line % heads);
        let kv_head = head / (heads / kv_heads);
        let sequence = rows[2 * row] as usize;
        let count = rows[2 * row + 1] as usize;
        let split = thread::blockIdx_y() as usize;
        let splits = thread::gridDim_y() as usize;
        let begin = split * chunk as usize;
        let end = if begin + (chunk as usize) < count {
            begin + chunk as usize
        } else {
            count
        };
        let table = sequence * dims.max_len as usize;

        // Les composantes `sub`, `sub + 8`… de la requête, déjà multipliées
        // par l'échelle.
        let mut query = [0.0f32; 8];
        unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7] {
            if sub + 8 * i < key_dim {
                query[i] = q[line * key_dim + sub + 8 * i] * scale;
            }
        });

        let mut max = f32::NEG_INFINITY;
        let mut total = 0.0f32;
        let mut mixed = [0.0f32; 8];
        // Tous les groupes d'un warp font le même nombre de tours : les
        // échanges entre registres demandent que le warp entier y prenne
        // part. Un groupe sans clé à ce tour calcule pour rien.
        let mut first = begin;
        while first < end {
            let key = first + group;
            let valid = key < end;
            let slot = if valid {
                slots[table + key] as usize
            } else {
                0
            };
            let base = (slot * kv_heads + kv_head) * key_dim;
            let mut dot = 0.0f32;
            if valid {
                unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7] {
                    if sub + 8 * i < key_dim {
                        dot += query[i] * keys[base + sub + 8 * i];
                    }
                });
            }
            dot += warp::shuffle_xor_f32(dot, 1);
            dot += warp::shuffle_xor_f32(dot, 2);
            dot += warp::shuffle_xor_f32(dot, 4);
            if valid {
                let new_max = if dot > max { dot } else { max };
                let correction = (max - new_max).exp();
                let weight = (dot - new_max).exp();
                total = total * correction + weight;
                let base = (slot * kv_heads + kv_head) * value_stride;
                unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7] {
                    if sub + 8 * i < value_dim {
                        mixed[i] = mixed[i] * correction + weight * values[base + sub + 8 * i];
                    }
                });
                max = new_max;
            }
            first += 16;
        }

        // SAFETY: chaque groupe écrit sa propre ligne.
        unsafe {
            if sub == 0 {
                PARTS[group * STRIDE] = max;
                PARTS[group * STRIDE + 1] = total;
            }
            unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7] {
                PARTS[group * STRIDE + 2 + sub + 8 * i] = mixed[i];
            });
        }
        thread::sync_threads();

        // Les 16 groupes réunis : le thread `t` s'occupe de la composante `t`.
        let t = thread::threadIdx_y() as usize * 32 + lane;
        if t < value_dim {
            let mut best = f32::NEG_INFINITY;
            // SAFETY: après la barrière, les parts ne sont plus que lues.
            unsafe {
                let mut g = 0;
                while g < 16 {
                    if PARTS[g * STRIDE] > best {
                        best = PARTS[g * STRIDE];
                    }
                    g += 1;
                }
                let (mut sum, mut value) = (0.0f32, 0.0f32);
                if best > f32::NEG_INFINITY {
                    let mut g = 0;
                    while g < 16 {
                        let factor = (PARTS[g * STRIDE] - best).exp();
                        sum += PARTS[g * STRIDE + 1] * factor;
                        value += PARTS[g * STRIDE + 2 + t] * factor;
                        g += 1;
                    }
                }
                let lines = thread::gridDim_x() as usize;
                // SAFETY: `(split, line, t)` n'appartient qu'à ce thread.
                if splits == 1 {
                    *out.add(line * value_dim + t) = value / sum;
                } else {
                    *out.add((split * lines + line) * value_dim + t) = value;
                    if t == 0 {
                        *stats.add((split * lines + line) * 2) = best;
                        *stats.add((split * lines + line) * 2 + 1) = sum;
                    }
                }
            }
        }
    }

    /// Réunit les tranches de `attention_cached` : chaque tranche apporte
    /// sa sortie non normalisée, son maximum et sa somme, et la sortie
    /// finale est celle d'un seul softmax sur toutes les clés. Un thread par
    /// élément de sortie, `[R, H, Dv]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn attention_cached_combine(
        parts: &[f32],
        stats: &[f32],
        splits: u32,
        value_dim: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        let len = out.len();
        if let Some(out_elem) = out.get_mut(idx) {
            let line = flat / value_dim as usize;
            let lines = len / value_dim as usize;
            let mut best = f32::NEG_INFINITY;
            for split in 0..splits as usize {
                let max = stats[(split * lines + line) * 2];
                if max > best {
                    best = max;
                }
            }
            let (mut sum, mut value) = (0.0f32, 0.0f32);
            for split in 0..splits as usize {
                let factor = (stats[(split * lines + line) * 2] - best).exp();
                sum += stats[(split * lines + line) * 2 + 1] * factor;
                value += parts[split * len + flat] * factor;
            }
            *out_elem = value / sum;
        }
    }

    /// `attention_cached` pour des têtes larges, jusqu'à 512 composantes :
    /// celles de l'attention latente, dont la clé est le latent suivi de la
    /// partie tournée. Un warp entier lit une clé : ses 32 threads s'en
    /// partagent les composantes, 16 chacun au plus, et réunissent leur
    /// produit scalaire par cinq échanges. Les 4 warps d'un bloc lisent 4
    /// clés à la fois. Le reste est celui de `attention_cached`.
    ///
    /// # Safety
    ///
    /// Comme `attention_cached`, avec des têtes d'au plus 512 composantes.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 2, block = (32, 4, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_cached_wide(
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        slots: &[f32],
        rows: &[f32],
        dims: CachedDims,
        chunk: u32,
        scale: f32,
        out: *mut f32,
        stats: *mut f32,
    ) {
        // Par warp : maximum, somme, puis 512 composantes de sortie.
        static mut PARTS: SharedArray<f32, 2056> = SharedArray::UNINIT;
        const STRIDE: usize = 514;

        let lane = thread::threadIdx_x() as usize;
        let group = thread::threadIdx_y() as usize;
        let heads = dims.heads as usize;
        let (key_dim, value_dim) = (dims.key_dim as usize, dims.value_dim as usize);
        let value_stride = dims.value_stride as usize;
        let kv_heads = dims.kv_heads as usize;
        let line = thread::blockIdx_x() as usize;
        let (row, head) = (line / heads, line % heads);
        let kv_head = head / (heads / kv_heads);
        let sequence = rows[2 * row] as usize;
        let count = rows[2 * row + 1] as usize;
        let split = thread::blockIdx_y() as usize;
        let splits = thread::gridDim_y() as usize;
        let begin = split * chunk as usize;
        let end = if begin + (chunk as usize) < count {
            begin + chunk as usize
        } else {
            count
        };
        let table = sequence * dims.max_len as usize;

        let mut query = [0.0f32; 16];
        unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
            if lane + 32 * i < key_dim {
                query[i] = q[line * key_dim + lane + 32 * i] * scale;
            }
        });

        let mut max = f32::NEG_INFINITY;
        let mut total = 0.0f32;
        let mut mixed = [0.0f32; 16];
        let mut key = begin + group;
        // Un warp par clé : chaque warp fait ses propres tours, sans avoir à
        // attendre les autres.
        while key < end {
            let slot = slots[table + key] as usize;
            let base = (slot * kv_heads + kv_head) * key_dim;
            let mut dot = 0.0f32;
            unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                if lane + 32 * i < key_dim {
                    dot += query[i] * keys[base + lane + 32 * i];
                }
            });
            let dot = warp_sum(dot);
            let new_max = if dot > max { dot } else { max };
            let correction = (max - new_max).exp();
            let weight = (dot - new_max).exp();
            total = total * correction + weight;
            let base = (slot * kv_heads + kv_head) * value_stride;
            unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                if lane + 32 * i < value_dim {
                    mixed[i] = mixed[i] * correction + weight * values[base + lane + 32 * i];
                }
            });
            max = new_max;
            key += 4;
        }

        // SAFETY: chaque warp écrit sa propre ligne.
        unsafe {
            if lane == 0 {
                PARTS[group * STRIDE] = max;
                PARTS[group * STRIDE + 1] = total;
            }
            unroll!(i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                PARTS[group * STRIDE + 2 + lane + 32 * i] = mixed[i];
            });
        }
        thread::sync_threads();

        // Les 4 warps réunis : le thread `t` s'occupe des composantes `t`,
        // `t + 128`…
        let lines = thread::gridDim_x() as usize;
        let mut best = f32::NEG_INFINITY;
        // SAFETY: après la barrière, les parts ne sont plus que lues.
        unsafe {
            let mut g = 0;
            while g < 4 {
                if PARTS[g * STRIDE] > best {
                    best = PARTS[g * STRIDE];
                }
                g += 1;
            }
        }
        let mut t = group * 32 + lane;
        while t < value_dim {
            let (mut sum, mut value) = (0.0f32, 0.0f32);
            // SAFETY: après la barrière, les parts ne sont plus que lues.
            unsafe {
                if best > f32::NEG_INFINITY {
                    let mut g = 0;
                    while g < 4 {
                        let factor = (PARTS[g * STRIDE] - best).exp();
                        sum += PARTS[g * STRIDE + 1] * factor;
                        value += PARTS[g * STRIDE + 2 + t] * factor;
                        g += 1;
                    }
                }
                // SAFETY: `(split, line, t)` n'appartient qu'à ce thread.
                if splits == 1 {
                    *out.add(line * value_dim + t) = value / sum;
                } else {
                    *out.add((split * lines + line) * value_dim + t) = value;
                    if t == 0 {
                        *stats.add((split * lines + line) * 2) = best;
                        *stats.add((split * lines + line) * 2 + 1) = sum;
                    }
                }
            }
            t += 128;
        }
    }
}
