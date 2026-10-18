//! Le produit d'octets par des octets, avec `vpdpbusd` (AVX-512 VNNI).
//!
//! Le noyau de `quantized.rs` convertit chaque code en `f32` avant de le
//! multiplier : trois instructions pour seize poids. Quand les poids sont
//! dans les caches, c'est cette conversion qui borne le produit, pas la
//! mémoire. `vpdpbusd` multiplie 64 octets non signés par 64 octets signés,
//! et additionne les produits quatre par quatre dans seize entiers de 32
//! bits : 64 multiplications-additions en une instruction, sans conversion.
//!
//! Il faut pour cela des activations en octets. Chaque ligne de `x` est
//! quantifiée à son tour, symétriquement : `x ≈ s · q`, `s = max |x| / 127`,
//! `q` entier entre -127 et 127, rangé `a = q + 128` pour être positif. Le
//! produit se corrige ensuite des deux décalages :
//!
//! ```text
//! int8 : Σ q·w = Σ a·w − 128 Σ w
//! int4 : Σ q·w = Σ a·u − 8 Σ a − 128 Σ u + 1024 n      (u = w + 8, n lignes du groupe)
//! ```
//!
//! `Σ w` et `Σ u` sont calculées une fois par colonne et par groupe, `Σ a`
//! une fois par ligne de `x` et par groupe.
//!
//! Le rangement des codes suit `vpdpbusd` : pour chaque quatre lignes de
//! `K` et chaque colonne, quatre octets contigus ; seize colonnes font un
//! registre. En int4, un octet porte deux codes : celui de la ligne `8i + j`
//! en bas, celui de `8i + 4 + j` en haut, pour qu'un masque et un décalage
//! redonnent deux registres au même rangement.
//!
//! Ce n'est plus le produit par les poids déquantifiés : l'erreur des
//! activations s'ajoute à celle des poids. Le mode est donc explicite
//! ([`Fast::with_int8_activations`]), et sa perte se mesure à part.

use super::Fast;
use super::pool::SharedMut;
use super::quantized::{BLOCK_COLUMNS, BLOCK_TASKS};
use crate::quantize::{Format, Quantized};

/// Les lignes de `K` d'une tranche de calcul en int8, où une seule échelle
/// sert toute la colonne.
const INT8_GROUP: usize = 128;

/// Le processeur a-t-il les instructions ?
pub(crate) fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        super::Isa::Avx512.available() && is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Une matrice quantifiée, rangée pour `vpdpbusd`.
pub(crate) struct Packed {
    format: Format,
    rows: usize,
    cols: usize,
    /// `rows` arrondi au multiple de 8 : les lignes ajoutées ont des poids
    /// nuls.
    padded: usize,
    /// Les lignes par groupe de calcul (voir `quantized.rs`), multiple de 8.
    group: usize,
    /// Les lignes par échelle.
    scale_group: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
    /// `[groupes de calcul, cols]` : `Σ w` en int8, `Σ u` en int4.
    sums: Vec<i32>,
}

impl Packed {
    pub(crate) fn new(
        format: Format,
        rows: usize,
        cols: usize,
        words: &[f32],
        scales: &[f32],
    ) -> Self {
        let quantized = Quantized {
            format,
            rows,
            cols,
            words: words.to_vec(),
            scales: scales.to_vec(),
        };
        let padded = rows.div_ceil(8) * 8;
        let scale_group = format.group(rows);
        let group = match format {
            Format::Int8 => INT8_GROUP,
            Format::Int4 { .. } => scale_group,
        };
        assert!(
            group.is_multiple_of(8),
            "vnni : des groupes de {group} lignes"
        );
        // Le code d'un poids, nul au-delà des lignes de la matrice.
        let code = |k: usize, col: usize| if k < rows { quantized.code(k, col) } else { 0 };
        let codes = match format {
            Format::Int8 => {
                let mut codes = vec![0u8; padded * cols];
                for quad in 0..padded / 4 {
                    for col in 0..cols {
                        for j in 0..4 {
                            codes[(quad * cols + col) * 4 + j] =
                                code(4 * quad + j, col) as i8 as u8;
                        }
                    }
                }
                codes
            }
            Format::Int4 { .. } => {
                let mut codes = vec![0u8; padded / 2 * cols];
                for oct in 0..padded / 8 {
                    for col in 0..cols {
                        for j in 0..4 {
                            let low = (code(8 * oct + j, col) + 8) as u8;
                            let high = (code(8 * oct + 4 + j, col) + 8) as u8;
                            codes[(oct * cols + col) * 4 + j] = low | (high << 4);
                        }
                    }
                }
                codes
            }
        };
        let groups = padded.div_ceil(group);
        let mut sums = vec![0i32; groups * cols];
        for g in 0..groups {
            for col in 0..cols {
                sums[g * cols + col] = (g * group..padded.min((g + 1) * group))
                    .map(|k| match format {
                        Format::Int8 => code(k, col),
                        Format::Int4 { .. } => code(k, col) + 8,
                    })
                    .sum();
            }
        }
        Self {
            format,
            rows,
            cols,
            padded,
            group,
            scale_group,
            codes,
            scales: scales.to_vec(),
            sums,
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.codes.len() + 4 * (self.scales.len() + self.sums.len())
    }
}

/// Les activations d'une ligne, en octets : `a = q + 128`, l'échelle `s`,
/// et `Σ a` par groupe de calcul.
struct Activations {
    bytes: Vec<u8>,
    scales: Vec<f32>,
    sums: Vec<i32>,
}

fn quantize_rows(w: &Packed, x: &[f32], m: usize) -> Activations {
    let k = w.rows;
    let groups = w.padded.div_ceil(w.group);
    let mut bytes = vec![128u8; m * w.padded];
    let mut scales = vec![1.0f32; m];
    let mut sums = vec![0i32; m * groups];
    for r in 0..m {
        let row = &x[r * k..(r + 1) * k];
        let largest = row.iter().fold(0.0f32, |max, &v| max.max(v.abs()));
        let scale = if largest > 0.0 { largest / 127.0 } else { 1.0 };
        scales[r] = scale;
        for (p, &v) in row.iter().enumerate() {
            bytes[r * w.padded + p] = ((v / scale).round().clamp(-127.0, 127.0) as i32 + 128) as u8;
        }
        for g in 0..groups {
            let range = g * w.group..w.padded.min((g + 1) * w.group);
            sums[r * groups + g] = bytes[r * w.padded..][range].iter().map(|&b| b as i32).sum();
        }
    }
    Activations {
        bytes,
        scales,
        sums,
    }
}

/// `out ≈ x · W`, `x` `[m, K]` et `out` `[m, N]`, découpé comme dans
/// `quantized.rs`.
pub(crate) fn product(fast: &Fast, w: &Packed, x: &[f32], m: usize, out: &mut [f32]) {
    let n = w.cols;
    let activations = quantize_rows(w, x, m);
    let groups = w.padded.div_ceil(w.group);
    let blocks = n.div_ceil(BLOCK_COLUMNS);
    let slices = BLOCK_TASKS.div_ceil(blocks).min(groups).max(1);
    let per_slice = groups.div_ceil(slices);
    let slices = groups.div_ceil(per_slice);
    let mut partial = if slices > 1 {
        vec![0.0f32; (slices - 1) * m * n]
    } else {
        Vec::new()
    };
    let first_out = SharedMut::new(out);
    let rest = SharedMut::new(&mut partial);
    fast.pool().run(blocks * slices, |task| {
        let (slice, block) = (task / blocks, task % blocks);
        let start = block * BLOCK_COLUMNS;
        let width = BLOCK_COLUMNS.min(n - start);
        let groups = slice * per_slice..groups.min((slice + 1) * per_slice);
        let target = if slice == 0 {
            first_out
        } else {
            unsafe { SharedMut::from_ptr(rest.ptr((slice - 1) * m * n)) }
        };
        for row in (0..m).step_by(4) {
            let job = Job {
                w,
                a: &activations,
                row,
                start,
                width,
                groups: groups.clone(),
                out: target,
            };
            #[cfg(target_arch = "x86_64")]
            unsafe {
                match m - row {
                    1 => x86::block::<1>(job),
                    2 => x86::block::<2>(job),
                    3 => x86::block::<3>(job),
                    _ => x86::block::<4>(job),
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            let _ = job;
        }
    });
    for part in partial.chunks(m * n) {
        for (value, &x) in out.iter_mut().zip(part) {
            *value += x;
        }
    }
}

struct Job<'a> {
    w: &'a Packed,
    a: &'a Activations,
    row: usize,
    start: usize,
    width: usize,
    groups: std::ops::Range<usize>,
    out: SharedMut<f32>,
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{BLOCK_COLUMNS, Job};
    use crate::quantize::Format;

    const V: usize = BLOCK_COLUMNS / 16;

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn block<const R: usize>(job: Job) {
        let Job {
            w,
            a,
            row,
            start,
            width,
            groups,
            out,
        } = job;
        let n = w.cols;
        let total_groups = w.padded.div_ceil(w.group);
        // Une colonne par entier de 32 bits : quatre octets.
        let masks: [__mmask16; V] = std::array::from_fn(|v| {
            let lanes = width.saturating_sub(16 * v).min(16);
            ((1u32 << lanes) - 1) as u16
        });
        let mut total = [[_mm512_setzero_ps(); V]; R];
        let codes = w.codes.as_ptr();
        let nibble = _mm512_set1_epi8(15);
        for g in groups {
            let (first, last) = (g * w.group, w.padded.min((g + 1) * w.group));
            let mut acc = [[_mm512_setzero_si512(); V]; R];
            unsafe {
                match w.format {
                    Format::Int8 => {
                        for quad in first / 4..last / 4 {
                            let line = codes.add((quad * n + start) * 4);
                            let xs: [__m512i; R] = std::array::from_fn(|r| {
                                let at = a.bytes.as_ptr().add((row + r) * w.padded + 4 * quad);
                                _mm512_set1_epi32((at as *const i32).read_unaligned())
                            });
                            for (v, &mask) in masks.iter().enumerate() {
                                let weights =
                                    _mm512_maskz_loadu_epi32(mask, line.add(64 * v) as *const i32);
                                for r in 0..R {
                                    acc[r][v] = _mm512_dpbusd_epi32(acc[r][v], xs[r], weights);
                                }
                            }
                        }
                    }
                    Format::Int4 { .. } => {
                        for oct in first / 8..last / 8 {
                            let line = codes.add((oct * n + start) * 4);
                            let quad = |r: usize, half: usize| {
                                let at = a
                                    .bytes
                                    .as_ptr()
                                    .add((row + r) * w.padded + 8 * oct + 4 * half);
                                _mm512_set1_epi32((at as *const i32).read_unaligned())
                            };
                            let low_x: [__m512i; R] = std::array::from_fn(|r| quad(r, 0));
                            let high_x: [__m512i; R] = std::array::from_fn(|r| quad(r, 1));
                            for (v, &mask) in masks.iter().enumerate() {
                                let packed =
                                    _mm512_maskz_loadu_epi32(mask, line.add(64 * v) as *const i32);
                                let low = _mm512_and_si512(packed, nibble);
                                let high = _mm512_and_si512(_mm512_srli_epi16(packed, 4), nibble);
                                for r in 0..R {
                                    acc[r][v] = _mm512_dpbusd_epi32(acc[r][v], low_x[r], low);
                                    acc[r][v] = _mm512_dpbusd_epi32(acc[r][v], high_x[r], high);
                                }
                            }
                        }
                    }
                }
                // Les corrections des décalages, puis les échelles.
                let sums = w.sums.as_ptr().add(g * n + start);
                let scales = w.scales.as_ptr().add((first / w.scale_group) * n + start);
                let count = (last - first) as i32;
                for (v, &mask) in masks.iter().enumerate() {
                    let column_sums = _mm512_maskz_loadu_epi32(mask, sums.add(16 * v));
                    let scale = _mm512_maskz_loadu_ps(mask, scales.add(16 * v));
                    let shifted = _mm512_slli_epi32(column_sums, 7);
                    for r in 0..R {
                        let mut exact = _mm512_sub_epi32(acc[r][v], shifted);
                        if let Format::Int4 { .. } = w.format {
                            let row_sum = a.sums[(row + r) * total_groups + g];
                            exact = _mm512_add_epi32(
                                exact,
                                _mm512_set1_epi32(1024 * count - 8 * row_sum),
                            );
                        }
                        let factor = _mm512_mul_ps(scale, _mm512_set1_ps(a.scales[row + r]));
                        total[r][v] =
                            _mm512_fmadd_ps(_mm512_cvtepi32_ps(exact), factor, total[r][v]);
                    }
                }
            }
        }
        for (r, sums) in total.iter().enumerate() {
            unsafe {
                let target = out.ptr((row + r) * n + start);
                for (v, &mask) in masks.iter().enumerate() {
                    _mm512_mask_storeu_ps(target.wrapping_add(16 * v), mask, sums[v]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Primitives;
    use crate::backend::ops::{QuantizedMatmul, Run};

    /// Les octets ne changent que les activations : sur des activations
    /// déjà entières entre -127 et 127, le produit est celui des poids
    /// déquantifiés, aux arrondis du `f32` près. Ailleurs, l'écart reste de
    /// l'ordre de l'erreur d'arrondi des activations, un sur 254.
    #[test]
    fn integer_activations_give_the_dequantized_product() {
        if !available() {
            return;
        }
        let fast = Fast::with_threads(4).with_int8_activations(true);
        let reference = Fast::with_threads(4);
        for (format, m, k, n) in [
            (Format::Int8, 1, 512, 300),
            (Format::Int8, 5, 1365, 70),
            (Format::Int4 { group: 64 }, 3, 1365, 130),
            (Format::Int4 { group: 32 }, 4, 64, 17),
        ] {
            let mut state = 7u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
            };
            let weight: Vec<f64> = (0..k * n).map(|_| next() as f64).collect();
            // Chaque ligne atteint 127 : son échelle vaut exactement 1.
            let mut x: Vec<f32> = (0..m * k).map(|_| (next() * 127.0).round()).collect();
            for r in 0..m {
                x[r * k] = 127.0;
            }
            let q = Quantized::round_to_nearest(&weight, k, n, format);
            let inputs = [
                reference.from_vec(x.clone(), &[m, k]),
                reference.from_vec(q.words.clone(), &[q.words.len() / n, n]),
                reference.from_vec(q.scales.clone(), &[q.scales.len() / n, n]),
                reference.from_vec(q.dequantize().iter().map(|&w| w as f32).collect(), &[k, n]),
            ];
            let refs: Vec<_> = inputs.iter().collect();
            let op = QuantizedMatmul { format };
            let expected = reference.to_vec(&reference.run(&op, &refs));
            let actual = fast.to_vec(&fast.run(&op, &refs));
            let largest = expected.iter().fold(0.0f32, |max, v| max.max(v.abs()));
            for (e, a) in expected.iter().zip(&actual) {
                assert!(
                    (e - a).abs() <= 1e-5 * largest,
                    "{format:?} {m}×{k}×{n} : {e} contre {a}"
                );
            }
        }
    }
}
