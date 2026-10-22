//! Les produits matriciels sur des poids quantifiés (voir
//! `smallem::quantize`) : ils lisent la forme compacte des poids.

use super::*;

#[cuda_module]
pub mod device {
    use super::*;

    /// `matmul_narrow` sur des poids quantifiés : chaque thread lit des
    /// mots de 24 bits plutôt que des `f32`, et en tire trois ou six poids.
    ///
    /// Le travail se découpe selon les mots : un bloc de 8 warps couvre 32
    /// colonnes et une tranche de `depth` mots de chaque colonne, un mot
    /// sur huit par warp. Chaque mot n'est lu qu'une fois. Ses codes sont
    /// des poids consécutifs le long de `K` d'un même groupe : une seule
    /// échelle pour tous.
    ///
    /// # Safety
    ///
    /// Comme `matmul_narrow`, les tranches comptant des mots : `out` a
    /// `tranches × rows × cols` éléments, où `tranches` vaut
    /// `mots.div_ceil(depth)`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_narrow_quantized(
        a: &[f32],
        words: &[f32],
        scales: &[f32],
        format: QuantizedDims,
        rows: u32,
        inner: u32,
        depth: u32,
        bias: &[f32],
        has_bias: u32,
        out: *mut f32,
    ) {
        static mut SUMS: SharedArray<f32, 2048> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let warp = thread::threadIdx_y() as usize;
        let (rows, inner, cols) = (rows as usize, inner as usize, format.cols as usize);
        let (group, per_word) = (format.group as usize, format.codes_per_word as usize);
        let per_group = format.words_per_group as usize;
        let total_words = inner.div_ceil(group) * per_group;
        let depth = depth as usize;
        let splits = total_words.div_ceil(depth);
        let split = thread::blockIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * WARP + lane;
        let begin = split * depth;
        let end = if begin + depth < total_words {
            begin + depth
        } else {
            total_words
        };
        let bits = format.bits;
        let mask = (1u32 << bits) - 1;
        let offset = 1i32 << (bits - 1);

        let mut first_row = 0;
        while first_row < rows {
            let mut sums = [0.0f32; 8];
            if col < cols {
                let mut w = begin + warp;
                // Le groupe du mot et son rang dans le groupe, suivis
                // d'un mot à l'autre sans division.
                let (mut g, mut slot_word) = (w / per_group, w % per_group);
                while w < end {
                    // SAFETY: `w < mots` et `col < cols`.
                    let word = unsafe { *words.as_ptr().add(w * cols + col) } as u32;
                    let scale = scales[g * cols + col];
                    let first_k = g * group + slot_word * per_word;
                    // Les poids du mot ; zéro pour les places vides en fin
                    // de groupe.
                    let mut weight = [0.0f32; 6];
                    unroll!(slot in [0, 1, 2, 3, 4, 5] {
                        if slot < per_word && slot_word * per_word + slot < group {
                            let stored = (word >> (bits * slot as u32)) & mask;
                            weight[slot] = (stored as i32 - offset) as f32 * scale;
                        }
                    });
                    unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                        if first_row + r < rows {
                            let base = (first_row + r) * inner + first_k;
                            unroll!(slot in [0, 1, 2, 3, 4, 5] {
                                if slot < per_word && first_k + slot < inner {
                                    // SAFETY: la ligne existe, `k < inner`.
                                    sums[r] += unsafe { *a.as_ptr().add(base + slot) }
                                        * weight[slot];
                                }
                            });
                        }
                    });
                    w += 8;
                    slot_word += 8;
                    while slot_word >= per_group {
                        slot_word -= per_group;
                        g += 1;
                    }
                }
            }
            // SAFETY: chaque thread écrit ses propres cases.
            unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                unsafe { SUMS[(warp * 8 + r) * WARP + lane] = sums[r] };
            });
            thread::sync_threads();
            let row = first_row + warp;
            if row < rows && col < cols {
                let mut total = if has_bias != 0 && splits == 1 {
                    bias[col]
                } else {
                    0.0
                };
                // SAFETY: après la barrière, les sommes ne sont plus que lues.
                unroll!(w in [0, 1, 2, 3, 4, 5, 6, 7] {
                    total += unsafe { SUMS[(w * 8 + warp) * WARP + lane] };
                });
                // SAFETY: `(split, row, col)` n'appartient qu'à ce thread.
                unsafe {
                    *out.add((split * rows + row) * cols + col) = total;
                }
            }
            thread::sync_threads();
            first_row += 8;
        }
    }

    /// `matmul_tiled` sur des poids quantifiés, pour les produits de la
    /// génération par lots : la tuile de `b` est déquantifiée au moment où
    /// elle entre en mémoire partagée. Un seul lot, `a` `[rows, inner]`.
    ///
    /// # Safety
    ///
    /// Comme `matmul_tiled`, pour un lot d'une matrice.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_tiled_quantized(
        a: &[f32],
        words: &[f32],
        scales: &[f32],
        format: QuantizedDims,
        rows: u32,
        inner: u32,
        depth: u32,
        out: *mut f32,
    ) {
        static mut TILE_A: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 1024> = SharedArray::UNINIT;

        const TILE: usize = MATMUL_TILE;
        const DEPTH: usize = MATMUL_DEPTH;
        const PATCH: usize = MATMUL_PATCH;

        let (rows, inner, cols) = (rows as usize, inner as usize, format.cols as usize);
        let (group, per_word) = (format.group as usize, format.codes_per_word as usize);
        let per_group = format.words_per_group as usize;
        let bits = format.bits;
        let mask = (1u32 << bits) - 1;
        let offset = 1i32 << (bits - 1);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let slot = ty * 16 + tx;
        let tile_row = thread::blockIdx_y() as usize * TILE;
        let column_tiles = cols.div_ceil(TILE);
        let block_x = thread::blockIdx_x() as usize;
        let (split, tile_col) = (block_x / column_tiles, (block_x % column_tiles) * TILE);
        let depth = depth as usize;
        let begin = split * depth;
        let end = if begin + depth < inner {
            begin + depth
        } else {
            inner
        };

        let (mut c00, mut c01, mut c02, mut c03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c10, mut c11, mut c12, mut c13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c20, mut c21, mut c22, mut c23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c30, mut c31, mut c32, mut c33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut tile_start = begin;
        while tile_start < end {
            let mut i = 0;
            while i < PATCH {
                let cell = slot * PATCH + i;
                let (a_row, a_col) = (tile_row + cell / DEPTH, tile_start + cell % DEPTH);
                // SAFETY: `cell` est propre à ce thread.
                unsafe {
                    TILE_A[cell] = if a_row < rows && a_col < end {
                        a[a_row * inner + a_col]
                    } else {
                        0.0
                    };
                }
                i += 1;
            }
            // La tuile de `b` : chaque thread en déquantifie quatre poids
            // voisins d'une même ligne `k`, donc d'un même rang dans leurs
            // mots. La position du code n'est calculée qu'une fois.
            {
                let cell = slot * PATCH;
                let (b_row, first_col) = (tile_start + cell / TILE, tile_col + cell % TILE);
                let (g, j) = (b_row / group, b_row % group);
                let word_row = (g * per_group + j / per_word) * cols;
                let shift = bits * (j % per_word) as u32;
                let mut i = 0;
                while i < PATCH {
                    let col = first_col + i;
                    let value = if b_row < end && col < cols {
                        let stored = ((words[word_row + col] as u32) >> shift) & mask;
                        (stored as i32 - offset) as f32 * scales[g * cols + col]
                    } else {
                        0.0
                    };
                    // SAFETY: `cell + i` est propre à ce thread.
                    unsafe { TILE_B[cell + i] = value };
                    i += 1;
                }
            }
            thread::sync_threads();
            // SAFETY: après la barrière, les tuiles ne sont plus que lues.
            unsafe {
                let mut k = 0;
                while k < DEPTH {
                    let a0 = TILE_A[(ty * PATCH) * DEPTH + k];
                    let a1 = TILE_A[(ty * PATCH + 1) * DEPTH + k];
                    let a2 = TILE_A[(ty * PATCH + 2) * DEPTH + k];
                    let a3 = TILE_A[(ty * PATCH + 3) * DEPTH + k];
                    let b0 = TILE_B[k * TILE + tx * PATCH];
                    let b1 = TILE_B[k * TILE + tx * PATCH + 1];
                    let b2 = TILE_B[k * TILE + tx * PATCH + 2];
                    let b3 = TILE_B[k * TILE + tx * PATCH + 3];
                    c00 += a0 * b0;
                    c01 += a0 * b1;
                    c02 += a0 * b2;
                    c03 += a0 * b3;
                    c10 += a1 * b0;
                    c11 += a1 * b1;
                    c12 += a1 * b2;
                    c13 += a1 * b3;
                    c20 += a2 * b0;
                    c21 += a2 * b1;
                    c22 += a2 * b2;
                    c23 += a2 * b3;
                    c30 += a3 * b0;
                    c31 += a3 * b1;
                    c32 += a3 * b2;
                    c33 += a3 * b3;
                    k += 1;
                }
            }
            thread::sync_threads();
            tile_start += DEPTH;
        }

        let (row, col) = (tile_row + ty * PATCH, tile_col + tx * PATCH);
        let values = [
            c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33,
        ];
        let mut i = 0;
        while i < PATCH * PATCH {
            let (row, col) = (row + i / PATCH, col + i % PATCH);
            if row < rows && col < cols {
                // SAFETY: `(split, row, col)` n'appartient qu'au pavé de ce
                // thread.
                unsafe {
                    *out.add((split * rows + row) * cols + col) = values[i];
                }
            }
            i += 1;
        }
    }
}
