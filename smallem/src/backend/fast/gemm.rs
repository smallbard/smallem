//! Le produit matriciel par blocs, à la manière de Goto et de BLIS.
//!
//! Le produit en triple boucle relit `B` en entier pour chaque ligne de `A`.
//! Dès que `B` ne tient plus dans les caches, chaque FMA attend la mémoire.
//! Le produit par blocs découpe le calcul pour que chaque niveau de cache
//! garde ce qu'il relit :
//!
//! ```text
//! pour chaque tranche de nc colonnes de B         (jc)
//!   pour chaque tranche de kc indices communs     (pc)  B[pc, jc] emballé, en L3
//!     pour chaque bloc de mc lignes de A          (ic)  A[ic, pc] emballé, en L2
//!       pour chaque panneau de NR colonnes        (jr)  un panneau de B, en L1
//!         pour chaque panneau de MR lignes        (ir)
//!           micro-kernel : C[MR × NR] += A[MR × kc] · B[kc × NR]
//! ```
//!
//! Emballer, c'est recopier un bloc de `A` ou de `B` dans l'ordre exact où
//! le micro-kernel le lira, complété par des zéros au bord. C'est aussi là
//! que se traitent les matrices transposées : l'emballage lit n'importe quels
//! pas, et le micro-kernel ne voit jamais que des panneaux contigus.
//!
//! Les fils se partagent les blocs de lignes et, quand il n'y en a pas
//! assez, des groupes de panneaux de colonnes. Ce partage ne change pas le
//! calcul d'un élément de `C` : sa somme se fait toujours tranche `kc` après
//! tranche `kc`, dans le même ordre. Le résultat ne dépend donc pas du nombre
//! de fils.

use std::cell::RefCell;

use super::Isa;
use super::kernels::{MicroKernel, Scalar};
use super::pool::{Pool, SharedMut};

/// Une matrice lue avec des pas quelconques : l'élément `(i, j)` est
/// `data[i × row_stride + j × col_stride]`. Une matrice transposée n'est
/// qu'un échange des pas.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
}

impl<'a> Matrix<'a> {
    /// Une matrice rangée ligne par ligne.
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
        }
    }

    /// La transposée, sans rien recopier.
    pub fn t(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    #[inline(always)]
    fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.row_stride + j * self.col_stride]
    }
}

/// Les tailles de blocs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blocks {
    /// La tranche de la dimension commune. C'est la seule taille qui change
    /// le résultat : elle fixe l'ordre des additions.
    pub kc: usize,
    /// Le plus grand bloc de lignes de `A`.
    pub mc: usize,
    /// La plus large tranche de colonnes de `B`.
    pub nc: usize,
    /// Comment les fils se partagent le produit.
    pub split: Split,
    /// Au plus ce nombre de lignes, le produit se fait sans emballage :
    /// chaque ligne de `B` est lue une fois et sert à toutes les lignes de
    /// `A`.
    pub few_rows: usize,
}

/// Le partage du produit entre les fils.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Chaque fil prend des blocs de lignes de `A` et de `C`, et lit la même
    /// tranche emballée de `B`, partagée dans le L3. Les colonnes ne se
    /// partagent que s'il n'y a pas assez de lignes.
    Rows,
    /// Chaque fil prend un groupe de colonnes de `B` et de `C`, qui reste
    /// dans son L2, et emballe lui-même chaque bloc de `A`.
    Columns,
}

impl Blocks {
    /// Les tailles retenues pour Zen 4 (voir `PLAN-CPU.md`, étape C2) : un
    /// panneau de `B` de `kc × NR` dans la moitié du L1 de 32 Ko, un bloc de
    /// `A` de `mc × kc` dans la moitié du L2 de 1 Mo, et la tranche de `B`
    /// de `kc × nc` dans une part du L3.
    pub const ZEN4: Blocks = Blocks {
        kc: 512,
        mc: 392,
        nc: 4096,
        split: Split::Rows,
        few_rows: FEW_ROWS,
    };
}

/// Le nombre de tâches visé par fil : les fils prennent les tâches à mesure
/// qu'ils finissent les précédentes, et des tâches plus nombreuses que les
/// fils les font finir ensemble.
const TASKS_PER_THREAD: usize = 3;

/// Les frontières entre les deux chemins, mesurées par `cpubench rows`
/// (étapes C2 et C5) : jusqu'à 256 lignes face à une matrice rangée ligne
/// par ligne, où le bloc de `C` reste dans les registres et chaque ligne de
/// `B` se lit d'un trait ; jusqu'à 16 face à une transposée, où chaque
/// élément est un produit scalaire.
pub const FEW_ROWS: usize = 256;
pub const FEW_ROWS_TRANSPOSED: usize = 16;

thread_local! {
    /// Le bloc de `A` emballé, propre à chaque fil.
    static PACKED_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// La tranche de `B` emballée, propre au fil qui lance le produit.
    static PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// `C = A · B`, ou `C += A · B` avec `accumulate`. `C` est rangée ligne par
/// ligne, `a.rows × b.cols`.
pub fn gemm(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    gemm_with_blocks(pool, isa, a, b, c, accumulate, Blocks::ZEN4);
}

/// Comme [`gemm`], avec des tailles de blocs données : pour les mesurer.
pub fn gemm_with_blocks(
    pool: &Pool,
    isa: Isa,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    assert_eq!(a.cols, b.rows, "gemm : formes incompatibles");
    assert_eq!(
        c.len(),
        a.rows * b.cols,
        "gemm : sortie de taille inattendue"
    );
    if c.is_empty() {
        return;
    }
    if a.cols == 0 {
        if !accumulate {
            c.fill(0.0);
        }
        return;
    }
    let limit = if b.col_stride == 1 {
        blocks.few_rows
    } else {
        blocks.few_rows.min(FEW_ROWS_TRANSPOSED)
    };
    if a.rows <= limit {
        few_rows(pool, isa, a, b, c, accumulate);
        return;
    }
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => blocked::<super::kernels::Avx512>(pool, a, b, c, accumulate, blocks),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => blocked::<super::kernels::Avx2>(pool, a, b, c, accumulate, blocks),
        _ => blocked::<Scalar>(pool, a, b, c, accumulate, blocks),
    }
}

/// Le produit par blocs, pour un micro-kernel donné.
fn blocked<K: MicroKernel>(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let (mr, nr) = (K::MR, K::NR);
    let kc = blocks.kc;
    let nc = blocks.nc.div_ceil(nr) * nr;
    let threads = pool.threads();

    // Les blocs de lignes : au plus `mc` lignes, et de quoi faire au moins
    // `TASKS_PER_THREAD` tâches par fil, pour que les fils finissent ensemble
    // même si les tâches ne tombent pas juste. S'il n'y a pas assez de
    // lignes, les colonnes sont partagées aussi.
    let wanted = TASKS_PER_THREAD * threads;
    let row_panels = m.div_ceil(mr);
    let row_blocks = match blocks.split {
        Split::Rows => m.div_ceil(blocks.mc.max(mr)).max(wanted.min(row_panels)),
        Split::Columns => m.div_ceil(blocks.mc.max(mr)),
    };
    let mc = row_panels.div_ceil(row_blocks) * mr;
    let row_blocks = m.div_ceil(mc);

    let out = SharedMut::new(c);
    PACKED_B.with(|packed_b| {
        let mut packed_b = packed_b.borrow_mut();
        for jc in (0..n).step_by(nc) {
            let width = nc.min(n - jc);
            let panels = width.div_ceil(nr);
            // Des groupes d'au moins deux panneaux, quand les lignes ne
            // suffisent pas à occuper les fils.
            let col_groups = match blocks.split {
                Split::Rows => wanted.div_ceil(row_blocks),
                Split::Columns => wanted,
            };
            let col_groups = col_groups.min(panels.div_ceil(2)).max(1);
            let per_group = panels.div_ceil(col_groups);
            let col_groups = panels.div_ceil(per_group);
            for pc in (0..k).step_by(kc) {
                let depth = kc.min(k - pc);
                let accumulate = accumulate || pc > 0;

                // La tranche de B, emballée par panneaux de NR colonnes.
                let size = panels * nr * depth;
                if packed_b.len() < size {
                    packed_b.resize(size, 0.0);
                }
                let shared_b = SharedMut::new(&mut packed_b[..size]);
                pool.run(panels, |panel| {
                    let target = unsafe { shared_b.slice(panel * nr * depth, nr * depth) };
                    pack_b(
                        b,
                        pc,
                        depth,
                        jc + panel * nr,
                        nr.min(width - panel * nr),
                        nr,
                        target,
                    );
                });
                let packed_b = &packed_b[..size];

                pool.run(row_blocks * col_groups, |task| {
                    let (row_block, group) = (task / col_groups, task % col_groups);
                    let ic = row_block * mc;
                    let height = mc.min(m - ic);
                    let first = group * per_group;
                    let last = panels.min(first + per_group);
                    PACKED_A.with(|packed_a| {
                        let mut packed_a = packed_a.borrow_mut();
                        let row_panels = height.div_ceil(mr);
                        let size = row_panels * mr * depth;
                        if packed_a.len() < size {
                            packed_a.resize(size, 0.0);
                        }
                        for panel in 0..row_panels {
                            let rows = mr.min(height - panel * mr);
                            pack_a(
                                a,
                                ic + panel * mr,
                                rows,
                                pc,
                                depth,
                                mr,
                                &mut packed_a[panel * mr * depth..][..mr * depth],
                            );
                        }
                        for jr in first..last {
                            let col = jc + jr * nr;
                            let cols = nr.min(n - col);
                            let panel_b = packed_b[jr * nr * depth..].as_ptr();
                            for ir in 0..row_panels {
                                let row = ic + ir * mr;
                                let rows = mr.min(m - row);
                                // SAFETY : chaque tâche écrit ses propres
                                // lignes et colonnes de C.
                                unsafe {
                                    let target = out.ptr(row * n + col);
                                    K::run(
                                        depth,
                                        packed_a[ir * mr * depth..].as_ptr(),
                                        panel_b,
                                        target,
                                        n,
                                        rows,
                                        cols,
                                        accumulate,
                                    );
                                }
                            }
                        }
                    });
                });
            }
        }
    });
}

// --- Précision mixte ---

/// Le processeur sait-il multiplier des `bf16` (`vdpbf16ps`) ?
pub fn bf16_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        Isa::Avx512.available() && is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// `f32` arrondi au `bf16` le plus proche, à égalité vers le pair : on garde
/// les 16 bits de poids fort.
#[inline(always)]
fn to_bf16(x: f32) -> u32 {
    let bits = x.to_bits();
    (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
}

/// Comme [`gemm`], les opérandes d'un grand produit arrondis en `bf16` et
/// leurs produits accumulés en `f32` : la précision mixte. Les produits de
/// quelques lignes, limités par la mémoire, restent en `f32`, comme sur la
/// carte.
pub fn gemm_mixed(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    let limit = if b.col_stride == 1 {
        FEW_ROWS
    } else {
        FEW_ROWS_TRANSPOSED
    };
    if a.rows <= limit || a.cols == 0 || c.is_empty() || isa != Isa::Avx512 || !bf16_available() {
        gemm(pool, isa, a, b, c, accumulate);
        return;
    }
    assert_eq!(a.cols, b.rows, "gemm : formes incompatibles");
    assert_eq!(
        c.len(),
        a.rows * b.cols,
        "gemm : sortie de taille inattendue"
    );
    blocked_bf16(pool, a, b, c, accumulate, Blocks::ZEN4);
}

thread_local! {
    static PAIRS_A: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static PAIRS_B: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Le produit par blocs de [`blocked`], avec des paires de `bf16` : la
/// tranche `kc` est comptée en indices, et emballée par paires.
#[cfg(target_arch = "x86_64")]
fn blocked_bf16(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    const MR: usize = 14;
    const NR: usize = 32;
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let kc = blocks.kc;
    let nc = blocks.nc.div_ceil(NR) * NR;
    let threads = pool.threads();
    let wanted = TASKS_PER_THREAD * threads;
    let row_panels = m.div_ceil(MR);
    let row_blocks = m.div_ceil(blocks.mc.max(MR)).max(wanted.min(row_panels));
    let mc = row_panels.div_ceil(row_blocks) * MR;
    let row_blocks = m.div_ceil(mc);
    let out = SharedMut::new(c);
    PAIRS_B.with(|packed_b| {
        let mut packed_b = packed_b.borrow_mut();
        for jc in (0..n).step_by(nc) {
            let width = nc.min(n - jc);
            let panels = width.div_ceil(NR);
            let col_groups = wanted.div_ceil(row_blocks).min(panels.div_ceil(2)).max(1);
            let per_group = panels.div_ceil(col_groups);
            let col_groups = panels.div_ceil(per_group);
            for pc in (0..k).step_by(kc) {
                let depth = kc.min(k - pc);
                let pairs = depth.div_ceil(2);
                let accumulate = accumulate || pc > 0;
                let size = panels * NR * pairs;
                if packed_b.len() < size {
                    packed_b.resize(size, 0);
                }
                let shared_b = SharedMut::new(&mut packed_b[..size]);
                pool.run(panels, |panel| {
                    let target = unsafe { shared_b.slice(panel * NR * pairs, NR * pairs) };
                    let (col, cols) = (jc + panel * NR, NR.min(width - panel * NR));
                    target.fill(0);
                    for p in 0..pairs {
                        for j in 0..cols {
                            let low = to_bf16(b.at(pc + 2 * p, col + j));
                            let high = if 2 * p + 1 < depth {
                                to_bf16(b.at(pc + 2 * p + 1, col + j))
                            } else {
                                0
                            };
                            target[p * NR + j] = low | (high << 16);
                        }
                    }
                });
                let packed_b = &packed_b[..size];
                pool.run(row_blocks * col_groups, |task| {
                    let (row_block, group) = (task / col_groups, task % col_groups);
                    let ic = row_block * mc;
                    let height = mc.min(m - ic);
                    let first = group * per_group;
                    let last = panels.min(first + per_group);
                    PAIRS_A.with(|packed_a| {
                        let mut packed_a = packed_a.borrow_mut();
                        let panels_a = height.div_ceil(MR);
                        let size = panels_a * MR * pairs;
                        if packed_a.len() < size {
                            packed_a.resize(size, 0);
                        }
                        for panel in 0..panels_a {
                            let target = &mut packed_a[panel * MR * pairs..][..MR * pairs];
                            target.fill(0);
                            let rows = MR.min(height - panel * MR);
                            for p in 0..pairs {
                                for i in 0..rows {
                                    let row = ic + panel * MR + i;
                                    let low = to_bf16(a.at(row, pc + 2 * p));
                                    let high = if 2 * p + 1 < depth {
                                        to_bf16(a.at(row, pc + 2 * p + 1))
                                    } else {
                                        0
                                    };
                                    target[p * MR + i] = low | (high << 16);
                                }
                            }
                        }
                        for jr in first..last {
                            let col = jc + jr * NR;
                            let cols = NR.min(n - col);
                            let panel_b = packed_b[jr * NR * pairs..].as_ptr();
                            for ir in 0..panels_a {
                                let row = ic + ir * MR;
                                let rows = MR.min(m - row);
                                unsafe {
                                    super::kernels::bf16_14x32(
                                        pairs,
                                        packed_a[ir * MR * pairs..].as_ptr(),
                                        panel_b,
                                        out.ptr(row * n + col),
                                        n,
                                        rows,
                                        cols,
                                        accumulate,
                                    );
                                }
                            }
                        }
                    });
                });
            }
        }
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn blocked_bf16(
    pool: &Pool,
    a: Matrix,
    b: Matrix,
    c: &mut [f32],
    accumulate: bool,
    blocks: Blocks,
) {
    blocked::<Scalar>(pool, a, b, c, accumulate, blocks)
}

/// Emballe `rows` lignes de `A`, de `row`, sur `depth` indices communs à
/// partir de `pc` : pour chaque indice, les `mr` éléments de la colonne,
/// complétés par des zéros.
fn pack_a(a: Matrix, row: usize, rows: usize, pc: usize, depth: usize, mr: usize, out: &mut [f32]) {
    if rows < mr {
        out.fill(0.0);
    }
    if a.col_stride == 1 {
        for i in 0..rows {
            let line = &a.data[(row + i) * a.row_stride + pc..][..depth];
            for (p, &value) in line.iter().enumerate() {
                out[p * mr + i] = value;
            }
        }
    } else {
        for p in 0..depth {
            for i in 0..rows {
                out[p * mr + i] = a.at(row + i, pc + p);
            }
        }
    }
}

/// Emballe `cols` colonnes de `B`, de `col`, sur `depth` indices communs à
/// partir de `pc` : pour chaque indice, les `nr` éléments de la ligne,
/// complétés par des zéros.
fn pack_b(b: Matrix, pc: usize, depth: usize, col: usize, cols: usize, nr: usize, out: &mut [f32]) {
    if cols < nr {
        out.fill(0.0);
    }
    if b.col_stride == 1 {
        for p in 0..depth {
            let line = &b.data[(pc + p) * b.row_stride + col..][..cols];
            out[p * nr..p * nr + cols].copy_from_slice(line);
        }
    } else {
        for j in 0..cols {
            for p in 0..depth {
                out[p * nr + j] = b.at(pc + p, col + j);
            }
        }
    }
}

// --- Quelques lignes ---

/// Des colonnes par groupes de cette taille : la tâche d'un fil. Une ligne
/// seule les garde toutes dans ses accumulateurs ; plusieurs lignes les
/// prennent par blocs de `FEW_ROWS_BLOCK`, pour que leurs accumulateurs
/// tiennent dans les registres.
const FEW_ROWS_COLUMNS: usize = 256;
const FEW_ROWS_BLOCK: usize = 64;
/// Le nombre de tâches visé : de quoi occuper tous les fils plusieurs fois.
/// Il ne dépend pas du nombre de fils, pour que le résultat n'en dépende
/// pas.
const FEW_ROWS_TASKS: usize = 32;
/// Une tranche de la dimension commune a au moins cette longueur.
const FEW_ROWS_DEPTH: usize = 128;
/// Des lignes de `Bᵀ` par groupes de cette taille, pour le produit par une
/// transposée.
const FEW_ROWS_DOTS: usize = 64;

/// Le produit quand `A` n'a que quelques lignes, comme en génération. Il n'y
/// a rien à réutiliser dans les caches : chaque poids sert une fois par
/// ligne. Emballer `B` coûterait plus cher que le produit lui-même.
fn few_rows(pool: &Pool, isa: Isa, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    if b.col_stride == 1 {
        // Des groupes de colonnes, et des tranches de la dimension commune
        // s'il n'y a pas assez de colonnes pour faire `FEW_ROWS_TASKS`
        // tâches : chaque tranche écrit sa somme partielle, et les sommes
        // s'additionnent ensuite dans l'ordre des tranches. Ce découpage ne
        // dépend que des formes.
        let width = if m == 1 && n >= FEW_ROWS_COLUMNS {
            FEW_ROWS_COLUMNS
        } else {
            FEW_ROWS_BLOCK
        };
        let groups = n.div_ceil(width);
        let slices = FEW_ROWS_TASKS
            .div_ceil(groups)
            .min(k.div_ceil(FEW_ROWS_DEPTH))
            .max(1);
        let depth = k.div_ceil(slices);
        let slices = k.div_ceil(depth);
        let mut partial = if slices > 1 {
            vec![0.0f32; (slices - 1) * m * n]
        } else {
            Vec::new()
        };
        let out = SharedMut::new(c);
        let rest = SharedMut::new(&mut partial);
        pool.run(groups * slices, |task| {
            let (slice, group) = (task / groups, task % groups);
            let first = group * width;
            let end = n.min(first + width);
            let (begin, stop) = (slice * depth, k.min((slice + 1) * depth));
            let a = Matrix {
                data: &a.data[begin * a.col_stride..],
                cols: stop - begin,
                ..a
            };
            let b = Matrix {
                data: &b.data[begin * b.row_stride..],
                rows: stop - begin,
                ..b
            };
            // La première tranche écrit dans C, les autres à part.
            let (target, accumulate) = if slice == 0 {
                (out, accumulate)
            } else {
                (
                    unsafe { SharedMut::from_ptr(rest.ptr((slice - 1) * m * n)) },
                    false,
                )
            };
            if width == FEW_ROWS_COLUMNS {
                let block = Block {
                    a,
                    row: 0,
                    b,
                    start: first,
                    width: end - first,
                    out: target,
                    n,
                    accumulate,
                };
                direct::<FEW_ROWS_COLUMNS>(isa, block, 1);
                return;
            }
            for row in (0..m).step_by(4) {
                let block = Block {
                    a,
                    row,
                    b,
                    start: first,
                    width: end - first,
                    out: target,
                    n,
                    accumulate,
                };
                direct::<FEW_ROWS_BLOCK>(isa, block, m - row);
            }
        });
        for part in partial.chunks(m * n) {
            for (value, &x) in c.iter_mut().zip(part) {
                *value += x;
            }
        }
    } else if b.row_stride == 1 {
        // `B` est une transposée : chaque élément est le produit scalaire de
        // deux lignes contiguës.
        let out = SharedMut::new(c);
        let groups = n.div_ceil(FEW_ROWS_DOTS);
        pool.run(groups, |group| {
            let start = group * FEW_ROWS_DOTS;
            let width = FEW_ROWS_DOTS.min(n - start);
            for i in 0..m {
                let target = unsafe { out.slice(i * n + start, width) };
                for (offset, value) in target.iter_mut().enumerate() {
                    let column = &b.data[(start + offset) * b.col_stride..][..b.rows];
                    let dot = dot_with_row(isa, a, i, column);
                    *value = if accumulate { *value + dot } else { dot };
                }
            }
        });
    } else {
        blocked::<Scalar>(pool, a, b, c, accumulate, Blocks::ZEN4);
    }
}

/// Un bloc de `C` du produit sans emballage : quelques lignes à partir de
/// `row`, `width` colonnes à partir de `start`.
#[derive(Clone, Copy)]
struct Block<'a> {
    a: Matrix<'a>,
    row: usize,
    b: Matrix<'a>,
    start: usize,
    width: usize,
    out: SharedMut<f32>,
    n: usize,
    accumulate: bool,
}

/// `C[R lignes, W colonnes] (+)= A · B`, sans emballage : le bloc de `C`
/// reste dans les registres (seize registres AVX-512 pour quatre lignes de
/// 64 colonnes, ou une ligne de 256), et chaque ligne de `B` sert aux `R`
/// lignes de `A` à la fois.
#[inline(always)]
fn direct_body<const R: usize, const W: usize, const FMA: bool>(block: Block) {
    let Block {
        a,
        row,
        b,
        start,
        width,
        out,
        n,
        accumulate,
    } = block;
    let mut acc = [[0.0f32; W]; R];
    let mut padded = [0.0f32; W];
    for p in 0..a.cols {
        let line = &b.data[p * b.row_stride + start..][..width];
        let line: &[f32; W] = if width == W {
            line.try_into().unwrap()
        } else {
            padded[..width].copy_from_slice(line);
            &padded
        };
        for (r, sums) in acc.iter_mut().enumerate() {
            let x = a.at(row + r, p);
            for j in 0..W {
                sums[j] = if FMA {
                    x.mul_add(line[j], sums[j])
                } else {
                    sums[j] + x * line[j]
                };
            }
        }
    }
    for (r, sums) in acc.iter().enumerate() {
        // SAFETY : chaque tâche écrit ses propres colonnes.
        let target = unsafe { out.slice((row + r) * n + start, width) };
        for (value, &sum) in target.iter_mut().zip(sums.iter()) {
            *value = if accumulate { *value + sum } else { sum };
        }
    }
}

fn direct<const W: usize>(isa: Isa, block: Block, rows: usize) {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2<const R: usize, const W: usize>(block: Block) {
        direct_body::<R, W, true>(block)
    }

    fn with<const R: usize, const W: usize>(isa: Isa, block: Block) {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 if W == 64 => unsafe { x86::direct_avx512::<R, 4>(block) },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 if W == 256 && R == 1 => unsafe { x86::direct_avx512::<1, 16>(block) },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 | Isa::Avx512 => unsafe { avx2::<R, W>(block) },
            _ => direct_body::<R, W, false>(block),
        }
    }

    match rows {
        1 => with::<1, W>(isa, block),
        2 => with::<2, W>(isa, block),
        3 => with::<3, W>(isa, block),
        _ => with::<4, W>(isa, block),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Block;

    /// Le produit sans emballage en AVX-512 : `R` lignes de `16 × V`
    /// colonnes, soit `R × V` accumulateurs, qui restent dans les registres
    /// du début à la fin. Les colonnes au-delà de `width` sont masquées.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn direct_avx512<const R: usize, const V: usize>(block: Block) {
        let Block {
            a,
            row,
            b,
            start,
            width,
            out,
            n,
            accumulate,
        } = block;
        let masks: [__mmask16; V] = std::array::from_fn(|v| {
            let lanes = width.saturating_sub(16 * v).min(16);
            ((1u32 << lanes) - 1) as u16
        });
        let mut acc = [[_mm512_setzero_ps(); V]; R];
        let a_data = a.data.as_ptr();
        let mut line = b.data.as_ptr().wrapping_add(start);
        for p in 0..a.cols {
            // SAFETY : l'appelant garantit que les lignes, les colonnes et
            // la tranche existent ; les colonnes masquées ne sont pas lues.
            unsafe {
                let x: [__m512; R] = std::array::from_fn(|r| {
                    _mm512_set1_ps(*a_data.add((row + r) * a.row_stride + p * a.col_stride))
                });
                for (v, &mask) in masks.iter().enumerate() {
                    let w = _mm512_maskz_loadu_ps(mask, line.wrapping_add(16 * v));
                    for r in 0..R {
                        acc[r][v] = _mm512_fmadd_ps(x[r], w, acc[r][v]);
                    }
                }
            }
            line = line.wrapping_add(b.row_stride);
        }
        for (r, sums) in acc.iter().enumerate() {
            unsafe {
                let target = out.ptr((row + r) * n + start);
                for (v, &mask) in masks.iter().enumerate() {
                    let at = target.wrapping_add(16 * v);
                    let mut value = sums[v];
                    if accumulate {
                        value = _mm512_add_ps(value, _mm512_maskz_loadu_ps(mask, at));
                    }
                    _mm512_mask_storeu_ps(at, mask, value);
                }
            }
        }
    }
}

/// Le produit scalaire de la ligne `i` de `A` et d'une ligne contiguë.
fn dot_with_row(isa: Isa, a: Matrix, i: usize, row: &[f32]) -> f32 {
    if a.col_stride == 1 {
        let line = &a.data[i * a.row_stride..][..a.cols];
        super::math::dot(isa, line, row)
    } else {
        (0..a.cols).map(|p| a.at(i, p) * row[p]).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: Matrix, b: Matrix) -> Vec<f64> {
        let mut c = vec![0.0; a.rows * b.cols];
        for i in 0..a.rows {
            for j in 0..b.cols {
                c[i * b.cols + j] = (0..a.cols)
                    .map(|p| a.at(i, p) as f64 * b.at(p, j) as f64)
                    .sum();
            }
        }
        c
    }

    fn values(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn check(isa: Isa, m: usize, k: usize, n: usize, transpose_a: bool, transpose_b: bool) {
        let pool = Pool::new(4);
        let data_a = values(m * k, 1);
        let data_b = values(k * n, 2);
        let a = if transpose_a {
            Matrix::new(&data_a, k, m).t()
        } else {
            Matrix::new(&data_a, m, k)
        };
        let b = if transpose_b {
            Matrix::new(&data_b, n, k).t()
        } else {
            Matrix::new(&data_b, k, n)
        };
        let expected = reference(a, b);
        let mut c = vec![f32::NAN; m * n];
        let blocks = Blocks {
            kc: 48,
            mc: 40,
            nc: 64,
            split: if (m + k + n).is_multiple_of(2) {
                Split::Rows
            } else {
                Split::Columns
            },
            few_rows: FEW_ROWS,
        };
        gemm_with_blocks(&pool, isa, a, b, &mut c, false, blocks);
        for (index, (&got, &want)) in c.iter().zip(&expected).enumerate() {
            assert!(
                (got as f64 - want).abs() < 1e-4 * (k as f64).sqrt(),
                "{isa:?} {m}×{k}×{n} (tA {transpose_a}, tB {transpose_b}), élément {index} : {got} au lieu de {want}",
            );
        }
        // Accumuler ajoute le même produit une seconde fois.
        gemm_with_blocks(&pool, isa, a, b, &mut c, true, blocks);
        for (&got, &want) in c.iter().zip(&expected) {
            assert!((got as f64 - 2.0 * want).abs() < 2e-4 * (k as f64).sqrt());
        }
    }

    fn isas() -> Vec<Isa> {
        let mut isas = vec![Isa::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if Isa::Avx2.available() {
                isas.push(Isa::Avx2);
            }
            if Isa::Avx512.available() {
                isas.push(Isa::Avx512);
            }
        }
        isas
    }

    #[test]
    fn blocked_product_matches_the_definition() {
        for isa in isas() {
            // Des bords partout : aucune dimension multiple de MR, NR, kc, mc ou nc.
            for &(m, k, n) in &[(5, 7, 3), (33, 100, 70), (130, 49, 150), (29, 200, 33)] {
                for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                    check(isa, m, k, n, ta, tb);
                }
            }
        }
    }

    #[test]
    fn few_rows_match_the_definition() {
        for isa in isas() {
            for &(m, k, n) in &[
                (1, 512, 300),
                (3, 70, 513),
                (4, 5, 1),
                (13, 100, 70),
                (16, 33, 300),
            ] {
                for (ta, tb) in [(false, false), (false, true), (true, true)] {
                    check(isa, m, k, n, ta, tb);
                }
            }
        }
    }

    #[test]
    fn mixed_products_are_exact_on_small_integers_and_close_otherwise() {
        if !bf16_available() {
            return;
        }
        let pool = Pool::new(4);
        for &(m, k, n, ta, tb) in &[
            (300, 70, 150, false, false),
            (290, 33, 65, true, false),
            (400, 129, 70, false, true),
        ] {
            // Des entiers entre -8 et 8 sont exacts en bf16, et leurs sommes
            // le restent en f32 : le résultat doit être celui du f32.
            let small = |len: usize, seed: u64| -> Vec<f32> {
                values(len, seed)
                    .iter()
                    .map(|v| (v * 8.0).round())
                    .collect()
            };
            for (data_a, data_b, exact) in [
                (small(m * k, 5), small(k * n, 6), true),
                (values(m * k, 7), values(k * n, 8), false),
            ] {
                let a = if ta {
                    Matrix::new(&data_a, k, m).t()
                } else {
                    Matrix::new(&data_a, m, k)
                };
                let b = if tb {
                    Matrix::new(&data_b, n, k).t()
                } else {
                    Matrix::new(&data_b, k, n)
                };
                let mut c = vec![0.0; m * n];
                gemm_mixed(&pool, Isa::Avx512, a, b, &mut c, false);
                let expected = reference(a, b);
                for (i, (&got, &want)) in c.iter().zip(&expected).enumerate() {
                    if exact {
                        assert_eq!(got as f64, want, "{m}×{k}×{n}, élément {i}");
                    } else {
                        // Chaque opérande est arrondi à 2⁻⁹ près : l'erreur est
                        // bornée par 2⁻⁸ × Σ |a| × |b| ≤ 2⁻⁸ × k.
                        assert!(
                            (got as f64 - want).abs() <= k as f64 / 256.0,
                            "{m}×{k}×{n} : {got} contre {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_number_of_threads() {
        let (m, k, n) = (300, 700, 200);
        let data_a = values(m * k, 3);
        let data_b = values(k * n, 4);
        let (a, b) = (Matrix::new(&data_a, m, k), Matrix::new(&data_b, k, n));
        let run = |threads: usize, isa: Isa| {
            let mut c = vec![0.0; m * n];
            gemm(&Pool::new(threads), isa, a, b, &mut c, false);
            c
        };
        for isa in isas() {
            let one = run(1, isa);
            assert_eq!(one, run(2, isa));
            assert_eq!(one, run(7, isa));
        }
    }
}
