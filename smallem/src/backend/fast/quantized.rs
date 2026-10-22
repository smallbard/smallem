//! Le produit par des poids quantifiés, en génération.
//!
//! Les poids quantifiés du projet sont rangés dans des `f32`, 24 bits de codes
//! par nombre (voir [`crate::quantize`]) : 1,33 octet par poids en int8, 0,67
//! en int4. Le processeur lit mieux de vrais octets. À la première
//! utilisation d'une matrice, ses codes sont donc recopiés dans un rangement
//! natif, gardé tant que la matrice existe :
//!
//! - int8 : un octet signé par poids, `[K, N]` ligne par ligne ;
//! - int4 : deux codes par octet, ceux des lignes `2i` et `2i + 1` d'une même
//!   colonne, décalés de 8 pour être positifs.
//!
//! Le noyau ressemble au produit sans emballage de `gemm.rs` : quelques
//! lignes de `x`, des blocs de 64 colonnes gardés dans les registres. Chaque
//! ligne de codes est convertie en `f32` dans les registres, multipliée par
//! l'élément de `x`, et l'échelle du groupe s'applique à la fin du groupe.
//! Le résultat est celui du produit par les poids déquantifiés, aux
//! arrondis près.

use std::sync::{Arc, Mutex, Weak};

use super::pool::SharedMut;
use super::{Buffer, Fast, FastTensor, Isa};
use crate::quantize::{Format, Quantized};

/// Au plus ce nombre de lignes, le produit lit les codes ; au-delà, il y a
/// assez de calcul pour que le produit par blocs sur les poids déquantifiés
/// gagne.
pub(crate) const ROWS: usize = 16;

/// Les colonnes d'un bloc et le nombre de tâches, partagés avec `vnni.rs`.
pub(crate) const BLOCK_COLUMNS: usize = COLUMNS;
pub(crate) const BLOCK_TASKS: usize = TASKS;

/// Des colonnes par blocs de cette largeur : quatre registres AVX-512.
const COLUMNS: usize = 64;
/// Le nombre de tâches visé, indépendant du nombre de fils.
const TASKS: usize = 16;
/// Les tranches de `K` d'une matrice int8.
const INT8_GROUP: usize = 128;

/// Une matrice quantifiée, rangée pour le processeur.
pub(crate) struct Native {
    format: Format,
    rows: usize,
    cols: usize,
    /// Les lignes par groupe de calcul : celles d'un groupe d'échelles en
    /// int4 ; en int8, où une seule échelle sert à toute la colonne, des
    /// tranches de `INT8_GROUP` lignes, pour pouvoir partager `K`.
    group: usize,
    /// Les lignes par échelle.
    scale_group: usize,
    /// int8 : `rows × cols` octets signés. int4 : `⌈rows / 2⌉ × cols`
    /// octets, deux codes chacun.
    codes: Vec<u8>,
    /// `[groupes, cols]`.
    scales: Vec<f32>,
}

impl Native {
    fn new(format: Format, rows: usize, cols: usize, words: &[f32], scales: &[f32]) -> Self {
        let quantized = Quantized {
            format,
            rows,
            cols,
            words: words.to_vec(),
            scales: scales.to_vec(),
        };
        let codes = match format {
            Format::Int8 => (0..rows * cols)
                .map(|i| quantized.code(i / cols, i % cols) as i8 as u8)
                .collect(),
            Format::Int4 { .. } => {
                let pairs = rows.div_ceil(2);
                let mut codes = vec![0u8; pairs * cols];
                for pair in 0..pairs {
                    for col in 0..cols {
                        let low = (quantized.code(2 * pair, col) + 8) as u8;
                        let high = if 2 * pair + 1 < rows {
                            (quantized.code(2 * pair + 1, col) + 8) as u8
                        } else {
                            8
                        };
                        codes[pair * cols + col] = low | (high << 4);
                    }
                }
                codes
            }
        };
        Self {
            format,
            rows,
            cols,
            group: match format {
                Format::Int8 => INT8_GROUP.min(rows),
                Format::Int4 { .. } => format.group(rows),
            },
            scale_group: format.group(rows),
            codes,
            scales: scales.to_vec(),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.codes.len() + 4 * self.scales.len()
    }
}

/// Les matrices déjà rangées, retrouvées par l'adresse de leurs mots. La
/// référence faible garde l'adresse occupée tant que l'entrée existe : une
/// autre matrice ne peut pas la reprendre.
pub(crate) struct Natives<T>(Mutex<Vec<(Weak<Buffer>, Arc<T>)>>);

impl<T> Default for Natives<T> {
    fn default() -> Self {
        Self(Mutex::new(Vec::new()))
    }
}

impl<T> Natives<T> {
    pub(crate) fn get(&self, words: &FastTensor, build: impl FnOnce() -> T) -> Arc<T> {
        let mut entries = self.0.lock().unwrap_or_else(|e| e.into_inner());
        entries.retain(|(weak, _)| weak.strong_count() > 0);
        let address = Arc::as_ptr(&words.data);
        if let Some((_, native)) = entries.iter().find(|(weak, _)| weak.as_ptr() == address) {
            return Arc::clone(native);
        }
        let native = Arc::new(build());
        entries.push((Arc::downgrade(&words.data), Arc::clone(&native)));
        native
    }

    fn alive(&self) -> Vec<Arc<T>> {
        let entries = self.0.lock().unwrap_or_else(|e| e.into_inner());
        entries
            .iter()
            .filter(|(weak, _)| weak.strong_count() > 0)
            .map(|(_, native)| Arc::clone(native))
            .collect()
    }
}

impl Fast {
    /// `x · W (+ b)` pour une matrice quantifiée, en lisant ses codes ; `None`
    /// quand `x` a trop de lignes pour que cela vaille la peine.
    pub(crate) fn quantized_product(
        &self,
        format: Format,
        x: &FastTensor,
        words: &FastTensor,
        scales: &FastTensor,
        dequantized: &FastTensor,
        bias: Option<&FastTensor>,
    ) -> Option<FastTensor> {
        let (k, n) = (dequantized.shape[0], dequantized.shape[1]);
        let width = *x.shape.last()?;
        let m = x.len() / width.max(1);
        if m > ROWS || width != k || !format.group(k).is_multiple_of(2) {
            return None;
        }
        let mut out = Buffer::any(m * n);
        if self.int8_activations && super::vnni::available() {
            let packed = self.packed.get(words, || {
                super::vnni::Packed::new(format, k, n, words.data(), scales.data())
            });
            super::vnni::product(self, &packed, x.data(), m, &mut out);
        } else {
            let native = self.natives.get(words, || {
                Native::new(format, k, n, words.data(), scales.data())
            });
            product(self, &native, x.data(), m, &mut out);
        }
        if let Some(bias) = bias {
            for line in out.chunks_mut(n) {
                for (value, &b) in line.iter_mut().zip(bias.data()) {
                    *value += b;
                }
            }
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = n;
        Some(FastTensor::new(out, &shape))
    }

    /// Octets des matrices quantifiées rangées pour le processeur.
    pub fn native_bytes(&self) -> usize {
        let native: usize = self
            .natives
            .alive()
            .iter()
            .map(|native| native.bytes())
            .sum();
        let packed: usize = self
            .packed
            .alive()
            .iter()
            .map(|packed| packed.bytes())
            .sum();
        native + packed
    }
}

/// `out = x · W`, `x` `[m, K]` et `out` `[m, N]`. Des blocs de colonnes et,
/// s'il le faut pour faire assez de tâches, des tranches de groupes le long
/// de `K`, chacune dans sa somme partielle.
fn product(fast: &Fast, w: &Native, x: &[f32], m: usize, out: &mut [f32]) {
    let (k, n) = (w.rows, w.cols);
    let groups = k.div_ceil(w.group);
    let blocks = n.div_ceil(COLUMNS);
    let slices = TASKS.div_ceil(blocks).min(groups).max(1);
    let per_slice = groups.div_ceil(slices);
    let slices = groups.div_ceil(per_slice);
    let isa = fast.isa();
    super::gemm::split_k(fast.pool(), out, slices, blocks, |slice, block, target| {
        let start = block * COLUMNS;
        let width = COLUMNS.min(n - start);
        let groups = slice * per_slice..groups.min((slice + 1) * per_slice);
        for row in (0..m).step_by(4) {
            let rows = 4.min(m - row);
            let job = Job {
                w,
                x,
                row,
                rows,
                start,
                width,
                groups: groups.clone(),
                out: target,
            };
            match isa {
                #[cfg(target_arch = "x86_64")]
                // SAFETY : `fast.isa()` a été vérifiée à la construction du
                // backend ; le bloc écrit ses lignes et ses colonnes.
                Isa::Avx512 => unsafe {
                    match rows {
                        1 => x86::block::<1>(job),
                        2 => x86::block::<2>(job),
                        3 => x86::block::<3>(job),
                        _ => x86::block::<4>(job),
                    }
                },
                _ => portable(job),
            }
        }
    });
}

/// Un bloc à calculer : `rows` lignes de `x` à partir de `row`, `width`
/// colonnes à partir de `start`, sur les groupes `groups`.
#[derive(Clone)]
struct Job<'a> {
    w: &'a Native,
    x: &'a [f32],
    row: usize,
    rows: usize,
    start: usize,
    width: usize,
    groups: std::ops::Range<usize>,
    out: SharedMut<'a, f32>,
}

impl Native {
    /// Le code du poids `(p, col)`, dans le rangement natif.
    #[inline(always)]
    fn code(&self, p: usize, col: usize) -> f32 {
        match self.format {
            Format::Int8 => self.codes[p * self.cols + col] as i8 as f32,
            Format::Int4 { .. } => {
                let byte = self.codes[(p / 2) * self.cols + col];
                let nibble = if p.is_multiple_of(2) {
                    byte & 15
                } else {
                    byte >> 4
                };
                nibble as f32 - 8.0
            }
        }
    }
}

/// Le calcul en Rust ordinaire, sans instructions vectorielles explicites.
fn portable(job: Job) {
    let Job {
        w,
        x,
        row,
        rows,
        start,
        width,
        groups,
        out,
    } = job;
    let k = w.rows;
    for r in row..row + rows {
        // SAFETY : chaque tâche écrit ses propres colonnes.
        let target = unsafe { out.slice(r * w.cols + start, width) };
        target.fill(0.0);
        for g in groups.clone() {
            let (first, last) = (g * w.group, k.min((g + 1) * w.group));
            for (j, value) in target.iter_mut().enumerate() {
                let col = start + j;
                let sum: f32 = (first..last).map(|p| x[r * k + p] * w.code(p, col)).sum();
                *value += w.scales[(first / w.scale_group) * w.cols + col] * sum;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{COLUMNS, Job};
    use crate::quantize::Format;

    const V: usize = COLUMNS / 16;

    /// `R` lignes de 64 colonnes : seize accumulateurs pour le groupe en
    /// cours, seize pour le total, en registres. Les codes d'une ligne de
    /// poids, 64 octets en int8, deviennent quatre registres de `f32`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub(super) unsafe fn block<const R: usize>(job: Job) {
        let Job {
            w,
            x,
            row,
            start,
            width,
            groups,
            out,
            ..
        } = job;
        let (k, n) = (w.rows, w.cols);
        let masks: [__mmask16; V] = std::array::from_fn(|v| {
            let lanes = width.saturating_sub(16 * v).min(16);
            ((1u32 << lanes) - 1) as u16
        });
        let mut total = [[_mm512_setzero_ps(); V]; R];
        let codes = w.codes.as_ptr();
        let low_nibble = _mm512_set1_epi32(15);
        let eight = _mm512_set1_epi32(8);
        for g in groups {
            let (first, last) = (g * w.group, k.min((g + 1) * w.group));
            let mut acc = [[_mm512_setzero_ps(); V]; R];
            // SAFETY : les groupes, les lignes et les colonnes lus existent
            // (formes vérifiées par l'appelant) ; les colonnes masquées ne
            // sont pas lues.
            unsafe {
                match w.format {
                    Format::Int8 => {
                        for p in first..last {
                            let line = codes.add(p * n + start);
                            let xs: [__m512; R] =
                                std::array::from_fn(|r| _mm512_set1_ps(x[(row + r) * k + p]));
                            for (v, &mask) in masks.iter().enumerate() {
                                let bytes =
                                    _mm_maskz_loadu_epi8(mask, line.add(16 * v) as *const i8);
                                let weights = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes));
                                for r in 0..R {
                                    acc[r][v] = _mm512_fmadd_ps(xs[r], weights, acc[r][v]);
                                }
                            }
                        }
                    }
                    Format::Int4 { .. } => {
                        // Les lignes deux par deux : un octet porte les deux.
                        for p in (first..last).step_by(2) {
                            let line = codes.add((p / 2) * n + start);
                            let even: [__m512; R] =
                                std::array::from_fn(|r| _mm512_set1_ps(x[(row + r) * k + p]));
                            let odd: [__m512; R] = std::array::from_fn(|r| {
                                if p + 1 < last {
                                    _mm512_set1_ps(x[(row + r) * k + p + 1])
                                } else {
                                    _mm512_setzero_ps()
                                }
                            });
                            for (v, &mask) in masks.iter().enumerate() {
                                let bytes = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(
                                    mask,
                                    line.add(16 * v) as *const i8,
                                ));
                                let low =
                                    _mm512_sub_epi32(_mm512_and_si512(bytes, low_nibble), eight);
                                let high = _mm512_sub_epi32(_mm512_srli_epi32(bytes, 4), eight);
                                let (low, high) =
                                    (_mm512_cvtepi32_ps(low), _mm512_cvtepi32_ps(high));
                                for r in 0..R {
                                    acc[r][v] = _mm512_fmadd_ps(even[r], low, acc[r][v]);
                                    acc[r][v] = _mm512_fmadd_ps(odd[r], high, acc[r][v]);
                                }
                            }
                        }
                    }
                }
                let scales = w.scales.as_ptr().add((first / w.scale_group) * n + start);
                for (v, &mask) in masks.iter().enumerate() {
                    let scale = _mm512_maskz_loadu_ps(mask, scales.add(16 * v));
                    for r in 0..R {
                        total[r][v] = _mm512_fmadd_ps(scale, acc[r][v], total[r][v]);
                    }
                }
            }
        }
        for (r, sums) in total.iter().enumerate() {
            // SAFETY : les colonnes masquées ne sont pas écrites ; chaque
            // tâche écrit ses propres colonnes.
            unsafe {
                let target = out.ptr((row + r) * n + start);
                for (v, &mask) in masks.iter().enumerate() {
                    _mm512_mask_storeu_ps(target.wrapping_add(16 * v), mask, sums[v]);
                }
            }
        }
    }
}
