//! Les micro-kernels du produit matriciel : AVX-512, AVX2 et scalaire.
//!
//! Un micro-kernel calcule un petit bloc `MR × NR` de `C` en entier, le long
//! de toute une tranche `kc` de la dimension commune. Le bloc reste dans les
//! registres vectoriels du début à la fin : à chaque indice de la tranche,
//! le kernel charge une ligne de `NR` éléments de `B`, diffuse un à un les
//! `MR` éléments de la colonne de `A`, et fait `MR × NR / largeur` FMA. Il
//! ne relit `C` qu'une fois, à la fin.
//!
//! Les panneaux de `A` et de `B` arrivent emballés (voir `gemm.rs`) : pour
//! chaque indice `p` de la tranche, les `MR` éléments de `A`, puis les `NR`
//! éléments de `B`, sont contigus. Les bords qui ne tombent pas juste ont
//! été complétés par des zéros à l'emballage : le kernel calcule toujours un
//! bloc entier, et n'écrit que la partie utile.

/// Ce que le produit matriciel demande à un micro-kernel.
pub trait MicroKernel: Sync {
    /// Lignes du bloc de `C`.
    const MR: usize;
    /// Colonnes du bloc de `C`.
    const NR: usize;

    /// `C[..rows, ..cols] (+)= A · B` sur une tranche de `kc` indices.
    ///
    /// `a` contient `kc × MR` éléments, `b` `kc × NR`. `c` pointe sur
    /// l'élément `(0, 0)` du bloc, dans une matrice de `ldc` colonnes. Avec
    /// `accumulate`, le produit s'ajoute à `C` ; sinon il le remplace.
    ///
    /// # Safety
    ///
    /// Les pointeurs doivent couvrir ce qui est lu et écrit, et le
    /// processeur avoir les instructions du kernel.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    );
}

/// Écrit un bloc calculé dans un tampon local vers `C`, en n'en gardant que
/// `rows × cols` éléments.
///
/// # Safety
///
/// `c` doit couvrir `rows` lignes de `cols` éléments, à `ldc` d'intervalle.
#[inline(always)]
unsafe fn write_back<const MR: usize, const NR: usize>(
    block: &[[f32; NR]; MR],
    c: *mut f32,
    ldc: usize,
    rows: usize,
    cols: usize,
    accumulate: bool,
) {
    for (i, line) in block.iter().enumerate().take(rows) {
        for (j, &value) in line.iter().enumerate().take(cols) {
            unsafe {
                let target = c.add(i * ldc + j);
                *target = if accumulate { *target + value } else { value };
            }
        }
    }
}

// --- Scalaire ---

/// Le repli sans instructions vectorielles explicites : un bloc `4 × 8`, que
/// le compilateur vectorise comme il peut.
pub struct Scalar;

impl MicroKernel for Scalar {
    const MR: usize = 4;
    const NR: usize = 8;

    unsafe fn run(
        kc: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut block = [[0.0f32; 8]; 4];
        for p in 0..kc {
            let (a, b) = unsafe {
                (
                    std::slice::from_raw_parts(a.add(p * 4), 4),
                    std::slice::from_raw_parts(b.add(p * 8), 8),
                )
            };
            for i in 0..4 {
                for j in 0..8 {
                    block[i][j] += a[i] * b[j];
                }
            }
        }
        unsafe { write_back(&block, c, ldc, rows, cols, accumulate) };
    }
}

// --- AVX2 ---

#[cfg(target_arch = "x86_64")]
pub use x86::{Avx2, Avx512, bf16_14x32};

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{MicroKernel, write_back};

    /// AVX2 et FMA : seize registres de 256 bits. Un bloc `6 × 16`, soit
    /// douze accumulateurs, deux registres pour la ligne de `B` et un pour
    /// l'élément de `A` diffusé.
    pub struct Avx2;

    impl MicroKernel for Avx2 {
        const MR: usize = 6;
        const NR: usize = 16;

        unsafe fn run(
            kc: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
            rows: usize,
            cols: usize,
            accumulate: bool,
        ) {
            unsafe { avx2_6x16(kc, a, b, c, ldc, rows, cols, accumulate) }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn avx2_6x16(
        kc: usize,
        mut a: *const f32,
        mut b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut acc = [[_mm256_setzero_ps(); 2]; 6];
        for _ in 0..kc {
            unsafe {
                let b0 = _mm256_loadu_ps(b);
                let b1 = _mm256_loadu_ps(b.add(8));
                for (i, line) in acc.iter_mut().enumerate() {
                    let ai = _mm256_broadcast_ss(&*a.add(i));
                    line[0] = _mm256_fmadd_ps(ai, b0, line[0]);
                    line[1] = _mm256_fmadd_ps(ai, b1, line[1]);
                }
                a = a.add(6);
                b = b.add(16);
            }
        }
        if rows == 6 && cols == 16 {
            for (i, line) in acc.iter().enumerate() {
                unsafe {
                    let target = c.add(i * ldc);
                    let (mut low, mut high) = (line[0], line[1]);
                    if accumulate {
                        low = _mm256_add_ps(low, _mm256_loadu_ps(target));
                        high = _mm256_add_ps(high, _mm256_loadu_ps(target.add(8)));
                    }
                    _mm256_storeu_ps(target, low);
                    _mm256_storeu_ps(target.add(8), high);
                }
            }
        } else {
            let mut block = [[0.0f32; 16]; 6];
            for (line, values) in block.iter_mut().zip(&acc) {
                unsafe {
                    _mm256_storeu_ps(line.as_mut_ptr(), values[0]);
                    _mm256_storeu_ps(line.as_mut_ptr().add(8), values[1]);
                }
            }
            unsafe { write_back(&block, c, ldc, rows, cols, accumulate) };
        }
    }

    // --- AVX-512 ---

    /// AVX-512 : trente-deux registres de 512 bits. Un bloc `14 × 32`, soit
    /// vingt-huit accumulateurs, deux registres pour la ligne de `B` et un
    /// pour l'élément de `A` diffusé.
    ///
    /// Zen 4 exécute une instruction de 512 bits en deux moitiés : ce kernel
    /// ne calcule pas plus vite que celui d'AVX2 par cycle, mais il charge
    /// deux fois moins de `A` et de `B` par FMA, et décode deux fois moins
    /// d'instructions.
    pub struct Avx512;

    impl MicroKernel for Avx512 {
        const MR: usize = 14;
        const NR: usize = 32;

        unsafe fn run(
            kc: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
            rows: usize,
            cols: usize,
            accumulate: bool,
        ) {
            unsafe { avx512_14x32(kc, a, b, c, ldc, rows, cols, accumulate) }
        }
    }

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn avx512_14x32(
        kc: usize,
        mut a: *const f32,
        mut b: *const f32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 14];
        for _ in 0..kc {
            unsafe {
                let b0 = _mm512_loadu_ps(b);
                let b1 = _mm512_loadu_ps(b.add(16));
                for (i, line) in acc.iter_mut().enumerate() {
                    let ai = _mm512_set1_ps(*a.add(i));
                    line[0] = _mm512_fmadd_ps(ai, b0, line[0]);
                    line[1] = _mm512_fmadd_ps(ai, b1, line[1]);
                }
                a = a.add(14);
                b = b.add(32);
            }
        }
        // Les colonnes utiles de chaque moitié, en masque.
        let low_mask: __mmask16 = if cols >= 16 {
            0xffff
        } else {
            ((1u32 << cols) - 1) as u16
        };
        let high_mask: __mmask16 = if cols >= 32 {
            0xffff
        } else if cols > 16 {
            ((1u32 << (cols - 16)) - 1) as u16
        } else {
            0
        };
        for (i, line) in acc.iter().enumerate().take(rows) {
            unsafe {
                let target = c.add(i * ldc);
                let (mut low, mut high) = (line[0], line[1]);
                if accumulate {
                    low = _mm512_add_ps(low, _mm512_maskz_loadu_ps(low_mask, target));
                    high = _mm512_add_ps(
                        high,
                        _mm512_maskz_loadu_ps(high_mask, target.wrapping_add(16)),
                    );
                }
                _mm512_mask_storeu_ps(target, low_mask, low);
                _mm512_mask_storeu_ps(target.wrapping_add(16), high_mask, high);
            }
        }
    }

    // --- bf16 ---

    /// Le micro-kernel de la précision mixte : `vdpbf16ps` multiplie deux
    /// paires de `bf16` par voie et ajoute leurs deux produits à un `f32`,
    /// soit deux fois les opérations d'une FMA `f32` par instruction. Le
    /// bloc de `C` est le même, `14 × 32`, en `f32`. `a` et `b` arrivent
    /// emballés par paires d'indices communs : pour chaque paire, les `MR`
    /// paires de `A` (quatre octets chacune), puis les `NR` paires de `B`.
    ///
    /// # Safety
    ///
    /// Comme [`MicroKernel::run`], `pairs` étant le nombre de paires.
    #[target_feature(enable = "avx512f,avx512bf16")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bf16_14x32(
        pairs: usize,
        mut a: *const u32,
        mut b: *const u32,
        c: *mut f32,
        ldc: usize,
        rows: usize,
        cols: usize,
        accumulate: bool,
    ) {
        let mut acc = [[_mm512_setzero_ps(); 2]; 14];
        for _ in 0..pairs {
            unsafe {
                let b0: __m512bh = std::mem::transmute(_mm512_loadu_si512(b as *const _));
                let b1: __m512bh = std::mem::transmute(_mm512_loadu_si512(b.add(16) as *const _));
                for (i, line) in acc.iter_mut().enumerate() {
                    let ai: __m512bh = std::mem::transmute(_mm512_set1_epi32(*a.add(i) as i32));
                    line[0] = _mm512_dpbf16_ps(line[0], ai, b0);
                    line[1] = _mm512_dpbf16_ps(line[1], ai, b1);
                }
                a = a.add(14);
                b = b.add(32);
            }
        }
        let low_mask: __mmask16 = if cols >= 16 {
            0xffff
        } else {
            ((1u32 << cols) - 1) as u16
        };
        let high_mask: __mmask16 = if cols >= 32 {
            0xffff
        } else if cols > 16 {
            ((1u32 << (cols - 16)) - 1) as u16
        } else {
            0
        };
        for (i, line) in acc.iter().enumerate().take(rows) {
            unsafe {
                let target = c.add(i * ldc);
                let (mut low, mut high) = (line[0], line[1]);
                if accumulate {
                    low = _mm512_add_ps(low, _mm512_maskz_loadu_ps(low_mask, target));
                    high = _mm512_add_ps(
                        high,
                        _mm512_maskz_loadu_ps(high_mask, target.wrapping_add(16)),
                    );
                }
                _mm512_mask_storeu_ps(target, low_mask, low);
                _mm512_mask_storeu_ps(target.wrapping_add(16), high_mask, high);
            }
        }
    }
}
