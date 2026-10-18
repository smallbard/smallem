//! Les kernels GPU : le code exécuté par chaque thread de la carte.
//!
//! La plupart suivent le schéma le plus simple. Le lancement crée au moins
//! un thread par élément de sortie ; chaque thread reçoit un indice unique,
//! calcule l'élément correspondant et l'écrit. Comme deux threads n'écrivent
//! jamais au même endroit, ils n'ont pas à se coordonner.
//!
//! Trois familles s'en écartent, parce que leurs threads coopèrent :
//!
//! - les produits matriciels (`matmul_tiled`,
//!   `matmul_bf16`) : un bloc de threads calcule une tuile de la
//!   sortie, en passant par la mémoire partagée du bloc et des barrières ;
//! - les réductions par ligne (`row_stats` et ses voisines) : un
//!   warp de 32 threads par ligne, qui réunit ses résultats partiels par
//!   échanges entre registres ([`warp_sum`], [`warp_max`]) ;
//! - `scatter_add_rows` : plusieurs threads peuvent viser la même
//!   case, et l'addition est atomique.
//!
//! Les fonctions ordinaires de ce fichier ([`Layout::source_index`],
//! [`bf16_pair`], [`warp_sum`]…) sont compilées à la fois pour l'hôte et
//! pour le GPU.

// Les boucles des kernels indexent explicitement : c'est l'accès mémoire que
// chaque thread effectue, et on veut le lire tel quel.
#![allow(clippy::needless_range_loop)]

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp, wmma,
};
use cuda_host::cuda_module;

use smallem::backend::float;

/// Nombre maximal de dimensions d'un tenseur sur GPU. Les formes sont
/// passées aux kernels par valeur, dans des tableaux de taille fixe : un
/// kernel ne peut pas allouer de mémoire.
pub const MAX_RANK: usize = 4;

/// Côté des tuiles de sortie de la matmul : un bloc de `16 × 16` threads
/// calcule une tuile `64 × 64`.
pub const MATMUL_TILE: usize = 64;

/// Côté du pavé de sortie que calcule chaque thread de la matmul.
pub const MATMUL_PATCH: usize = 4;

/// Côté des tuiles de sortie de la matmul sur Tensor Cores.
pub const MATMUL_BF16_TILE: usize = 128;

/// Épaisseur des tranches de la matmul le long de la dimension commune.
pub const MATMUL_DEPTH: usize = 16;

/// Nombre de threads par bloc. Les contrats de lancement des kernels
/// exigent cette valeur exacte.
pub const BLOCK_SIZE: u32 = 256;

/// Comment lire un tenseur source à travers une forme de sortie : la forme
/// parcourue, et le pas du tenseur source le long de chaque dimension.
///
/// Les formes de moins de [`MAX_RANK`] dimensions sont complétées à gauche
/// par des dimensions de taille 1 et de pas 0, ce qui ne change rien au
/// calcul. Un pas nul sur une vraie dimension est du broadcasting.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub shape: [u32; MAX_RANK],
    pub strides: [u32; MAX_RANK],
}

impl Layout {
    /// Convertit un indice à plat dans `shape` en indice dans le stockage du
    /// tenseur source. Même calcul que `shape::source_index` côté CPU.
    pub fn source_index(&self, mut flat: u32) -> usize {
        let mut index = 0;
        let mut dim = MAX_RANK;
        while dim > 0 {
            dim -= 1;
            index += (flat % self.shape[dim]) * self.strides[dim];
            flat /= self.shape[dim];
        }
        index as usize
    }
}

/// Opérations du kernel `binary`.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `param` là où le second opérande est non nul, le premier sinon.
    MaskedFill,
    /// Second opérande multiplié par la pente de GELU au premier.
    GeluBackward,
    /// Second opérande multiplié par la pente de SiLU au premier.
    SiluBackward,
}

/// Opérations du kernel `unary`.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum UnaryOp {
    Neg,
    Exp,
    Ln,
    Sqrt,
    Tanh,
    Gelu,
    Silu,
}

/// Sortie du kernel `softmax_rows`.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum SoftmaxMode {
    Probability,
    Log,
}

/// Les réglages d'un pas d'AdamW, passés par valeur au kernel
/// `adamw_step`.
#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub bias_correction1: f32,
    pub bias_correction2: f32,
    pub grad_scale: f32,
}

/// Arrondit deux `f32` en `bf16`, au plus proche, et les range dans un
/// `u32`, le premier dans les 16 bits bas.
///
/// Un `bf16` est la moitié haute d'un `f32` : même signe, même exposant, et
/// 7 des 23 bits de mantisse. Ajouter un demi-pas avant de couper arrondit
/// au plus proche (au pair en cas d'égalité) ; couper sans arrondir tirerait
/// toutes les valeurs vers zéro, un biais qui s'accumulerait à
/// l'entraînement. Le calcul est sans branchement : un NaN peut en sortir
/// sous la forme d'un infini.
#[inline(always)]
pub fn bf16_pair(lo: f32, hi: f32) -> u32 {
    let lo = lo.to_bits();
    let hi = hi.to_bits();
    let lo = lo.wrapping_add(0x7FFF + ((lo >> 16) & 1)) >> 16;
    let hi = hi.wrapping_add(0x7FFF + ((hi >> 16) & 1)) & 0xFFFF_0000;
    lo | hi
}

/// Nombre de threads d'un warp : le groupe de threads que le GPU fait
/// avancer d'un même pas, et qui peuvent s'échanger des valeurs de registre
/// à registre.
pub const WARP: usize = 32;

/// Somme de `value` sur les 32 threads du warp ; chacun reçoit le total.
///
/// À chaque tour, un thread ajoute à sa valeur celle du thread dont le rang
/// diffère par un bit (16, puis 8, 4, 2, 1). Après cinq tours, chaque valeur
/// a parcouru tout le warp. L'échange (`shuffle`) se fait entre registres,
/// sans mémoire ni barrière.
///
/// Tous les threads du warp doivent appeler cette fonction ensemble.
#[inline(always)]
pub fn warp_sum(mut value: f32) -> f32 {
    let mut distance = WARP as u32 / 2;
    while distance > 0 {
        value += warp::shuffle_xor_f32(value, distance);
        distance /= 2;
    }
    value
}

/// Maximum de `value` sur les 32 threads du warp, sur le modèle de
/// [`warp_sum`].
#[inline(always)]
pub fn warp_max(mut value: f32) -> f32 {
    let mut distance = WARP as u32 / 2;
    while distance > 0 {
        let other = warp::shuffle_xor_f32(value, distance);
        if other > value {
            value = other;
        }
        distance /= 2;
    }
    value
}

/// Côté des tuiles de l'attention fusionnée : un bloc traite 32 requêtes
/// (ou 32 clés) à la fois.
pub const ATTENTION_TILE: usize = 32;

/// Taille maximale des têtes que traitent les kernels d'attention : la
/// largeur des requêtes, des clés et des valeurs. Au-delà, l'opération est
/// calculée avec les primitives.
pub const ATTENTION_MAX_DIM: usize = 64;

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

/// Côté des tuiles de l'attention fusionnée sur les Tensor Cores : quatre
/// warps de seize lignes.
pub const WIDE_TILE: usize = 64;

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

/// Les dimensions d'une attention sur le cache, telles que les reçoit
/// `attention_cached`.
#[derive(Clone, Copy, Debug)]
pub struct CachedDims {
    pub heads: u32,
    pub kv_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    /// Largeur des lignes de la table des valeurs, dont seules les
    /// `value_dim` premières composantes servent.
    pub value_stride: u32,
    /// Longueur des lignes de la table des lignes de chaque séquence.
    pub max_len: u32,
}

/// La forme d'une matrice quantifiée, telle que la reçoivent les kernels
/// qui la lisent (voir `smallem::quantize`).
#[derive(Clone, Copy, Debug)]
pub struct QuantizedDims {
    /// Colonnes de la matrice, `N`.
    pub cols: u32,
    /// Bits par code : 8 ou 4.
    pub bits: u32,
    /// Poids par groupe le long de `K`.
    pub group: u32,
    pub codes_per_word: u32,
    pub words_per_group: u32,
}

#[cuda_module]
pub mod device {
    use super::*;

    /// `out[i] = op(a[…], b[…])`, avec broadcasting des deux opérandes vers
    /// la forme de sortie.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn binary(
        op: BinaryOp,
        param: f32,
        a: &[f32],
        layout_a: Layout,
        b: &[f32],
        layout_b: Layout,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        // Le dernier bloc contient des threads en trop : ils n'ont pas
        // d'élément de sortie et ne font rien.
        if let Some(out_elem) = out.get_mut(idx) {
            let x = a[layout_a.source_index(flat)];
            let y = b[layout_b.source_index(flat)];
            *out_elem = match op {
                BinaryOp::Add => x + y,
                BinaryOp::Sub => x - y,
                BinaryOp::Mul => x * y,
                BinaryOp::Div => x / y,
                BinaryOp::MaskedFill => {
                    if y != 0.0 {
                        param
                    } else {
                        x
                    }
                }
                BinaryOp::GeluBackward => y * float::gelu_slope(x),
                BinaryOp::SiluBackward => y * float::silu_slope(x),
            };
        }
    }

    /// `out[i] = op(a[i])`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn unary(op: UnaryOp, a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let x = a[i];
            *out_elem = match op {
                UnaryOp::Neg => -x,
                UnaryOp::Exp => x.exp(),
                UnaryOp::Ln => x.ln(),
                UnaryOp::Sqrt => x.sqrt(),
                UnaryOp::Tanh => x.tanh(),
                UnaryOp::Gelu => float::gelu(x),
                UnaryOp::Silu => float::silu(x),
            };
        }
    }

    /// Réarrange les éléments : `out[i] = a[layout.source_index(i)]`.
    /// Avec deux pas échangés, c'est une transposition.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn permute(a: &[f32], layout: Layout, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a[layout.source_index(flat)];
        }
    }

    /// Somme le long d'un axe. Le tenseur est vu comme `[outer, n, inner]`
    /// et la sortie comme `[outer, inner]` : chaque thread additionne les
    /// `n` éléments de sa colonne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_axis(a: &[f32], n: u32, inner: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        if let Some(out_elem) = out.get_mut(idx) {
            let o = flat / inner;
            let i = flat % inner;
            let mut total = 0.0;
            for k in 0..n {
                total += a[((o * n + k) * inner + i) as usize];
            }
            *out_elem = total;
        }
    }

    /// Somme par paquets : le thread `t` additionne les éléments
    /// `[t × chunk, (t + 1) × chunk)`. Répété jusqu'à ce qu'il ne reste
    /// qu'un élément, cela donne la somme de tout un tenseur.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_chunks(a: &[f32], chunk: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let start = idx.get() * chunk as usize;
        if let Some(out_elem) = out.get_mut(idx) {
            // Le dernier paquet peut être incomplet.
            let mut end = start + chunk as usize;
            if end > a.len() {
                end = a.len();
            }
            let mut total = 0.0;
            for i in start..end {
                total += a[i];
            }
            *out_elem = total;
        }
    }

    /// Produit matriciel par lots, sur les deux dernières dimensions.
    ///
    /// Les `Layout` décrivent les dimensions de lot, et leurs pas comptent
    /// en matrices entières : un pas nul fait relire la même matrice à tous
    /// les lots, comme les poids d'une couche linéaire.
    ///
    /// Un bloc de `16 × 16` threads calcule une tuile `64 × 64` de la
    /// sortie, et ses threads coopèrent. Si chacun calculait seul un élément,
    /// il relirait en mémoire globale toute une ligne de `a` et toute une
    /// colonne de `b`, que ses voisins relisent aussi. Ici le bloc avance le
    /// long de `inner` par tranches de 16 : à chaque tranche, il charge
    /// ensemble une tuile `64 × 16` de `a` et une tuile `16 × 64` de `b`
    /// dans sa mémoire partagée, bien plus rapide. Chaque valeur n'est lue
    /// qu'une fois en mémoire globale.
    ///
    /// Chaque thread calcule un pavé `4 × 4` de la sortie, et non un seul
    /// élément. Pour une tranche, il lit 4 valeurs de `a` et 4 de `b` par
    /// indice `k`, et en tire 16 produits : une lecture pour deux produits,
    /// là où un thread par élément en ferait deux par produit. Les 16
    /// accumulateurs restent dans des registres.
    ///
    /// La grille est à deux dimensions : `x` parcourt les colonnes, `y` les
    /// lignes de toutes les matrices du lot, mises bout à bout. Quand il y a
    /// trop peu de tuiles pour occuper la carte, la dimension commune est en
    /// plus découpée en tranches, que `x` enchaîne aussi : chacune écrit sa
    /// somme partielle, et `sum_splits` les additionne.
    ///
    /// # Safety
    ///
    /// Le lancement doit couvrir exactement `tranches × cols` en `x` et
    /// `lots × rows` en `y`, par blocs de `16 × 16` threads pour `64 × 64`
    /// éléments, où `tranches` vaut `inner.div_ceil(depth)` et `depth` est un
    /// multiple de 16. `out` doit pointer sur un tampon GPU de
    /// `tranches × lots × rows × cols` éléments : chaque tranche de la
    /// dimension commune écrit sa somme partielle dans sa propre copie de
    /// la sortie. `out` est un pointeur brut parce que
    /// chaque thread écrit plusieurs éléments, ce que `DisjointSlice` ne
    /// sait pas vérifier.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_tiled(
        a: &[f32],
        batch_a: Layout,
        b: &[f32],
        batch_b: Layout,
        rows: u32,
        inner: u32,
        cols: u32,
        depth: u32,
        out: *mut f32,
    ) {
        // Mémoire partagée par les threads d'un même bloc : `a` en
        // `[64, 16]`, `b` en `[16, 64]`.
        static mut TILE_A: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 1024> = SharedArray::UNINIT;

        const TILE: usize = MATMUL_TILE;
        const DEPTH: usize = MATMUL_DEPTH;
        const PATCH: usize = MATMUL_PATCH;

        let (rows, inner, cols) = (rows as usize, inner as usize, cols as usize);
        // Position du thread dans son bloc.
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let slot = ty * 16 + tx;

        // L'axe `y` de la grille enchaîne les matrices du lot : on retrouve
        // d'abord la matrice, puis la ligne dans cette matrice.
        let tiles_per_matrix = rows.div_ceil(TILE);
        let block_y = thread::blockIdx_y() as usize;
        let batch = block_y / tiles_per_matrix;
        let tile_row = (block_y % tiles_per_matrix) * TILE;
        // L'axe `x` enchaîne les tranches de la dimension commune, chacune
        // sur toutes les tuiles de colonnes.
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
        let matrices = thread::gridDim_y() as usize / tiles_per_matrix;

        let start_a = batch_a.source_index(batch as u32) * rows * inner;
        let start_b = batch_b.source_index(batch as u32) * inner * cols;

        // Les 16 sommes du pavé de ce thread, `cij` pour la ligne `i` et la
        // colonne `j`. Ce sont seize variables et non un tableau : le
        // compilateur de kernels garde les variables dans des registres,
        // mais range un tableau en mémoire locale, bien plus lente.
        let (mut c00, mut c01, mut c02, mut c03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c10, mut c11, mut c12, mut c13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c20, mut c21, mut c22, mut c23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut c30, mut c31, mut c32, mut c33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut tile_start = begin;
        while tile_start < end {
            // 1. Chargement : chaque thread apporte 4 éléments voisins de
            //    chaque tuile. Hors des matrices, on met 0, qui ne change
            //    pas les sommes.
            let mut i = 0;
            while i < PATCH {
                let cell = slot * PATCH + i;
                let (a_row, a_col) = (tile_row + cell / DEPTH, tile_start + cell % DEPTH);
                let (b_row, b_col) = (tile_start + cell / TILE, tile_col + cell % TILE);
                // SAFETY: `cell` est propre à ce thread, donc deux threads
                // n'écrivent jamais la même case.
                unsafe {
                    TILE_A[cell] = if a_row < rows && a_col < end {
                        a[start_a + a_row * inner + a_col]
                    } else {
                        0.0
                    };
                    TILE_B[cell] = if b_row < end && b_col < cols {
                        b[start_b + b_row * cols + b_col]
                    } else {
                        0.0
                    };
                }
                i += 1;
            }
            // 2. Barrière : personne ne lit avant que tout le bloc ait écrit.
            thread::sync_threads();

            // 3. Calcul : 4 lignes de la tuile de `a` par 4 colonnes de la
            //    tuile de `b`.
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
            // 4. Barrière : personne ne recharge avant que tout le bloc ait lu.
            thread::sync_threads();

            tile_start += DEPTH;
        }

        // Les éléments du pavé qui dépassent de la matrice ne sont pas écrits.
        let (row, col) = (tile_row + ty * PATCH, tile_col + tx * PATCH);
        let values = [
            c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33,
        ];
        let mut i = 0;
        while i < PATCH * PATCH {
            let (row, col) = (row + i / PATCH, col + i % PATCH);
            if row < rows && col < cols {
                // SAFETY: `(batch, row, col)` n'appartient qu'au pavé de ce
                // thread et désigne un élément de `out`, dont l'hôte
                // garantit la taille.
                unsafe {
                    *out.add(((split * matrices + batch) * rows + row) * cols + col) = values[i];
                }
            }
            i += 1;
        }
    }

    /// Le même produit matriciel sur les Tensor Cores, en précision mixte :
    /// les opérandes sont arrondis en `bf16` (16 bits, dont 8 de mantisse),
    /// les sommes restent en `f32`.
    ///
    /// Un Tensor Core multiplie d'un coup une tuile `16 × 16` par une tuile
    /// `16 × 8` (instruction `mma.sync`). L'instruction est exécutée par un
    /// warp entier, c'est-à-dire 32 threads consécutifs : chacun fournit dans
    /// ses registres un fragment des deux tuiles et reçoit un fragment du
    /// résultat. Quel thread tient quel élément est fixé par l'instruction ;
    /// avec `groupe = thread / 4` et `t = thread % 4` dans le warp :
    ///
    /// ```text
    /// a : lignes groupe et groupe + 8, colonnes 2t, 2t+1, 2t+8, 2t+9
    /// b : colonne groupe, lignes 2t, 2t+1, 2t+8, 2t+9
    /// c : lignes groupe et groupe + 8, colonnes 2t et 2t+1
    /// ```
    ///
    /// Les `bf16` voyagent par paires dans un `u32`. Les tuiles en mémoire
    /// partagée sont rangées pour que chaque registre soit une seule
    /// lecture : `a` par paires le long de `k`, ligne par ligne ; `b` par
    /// paires le long de `k`, colonne par colonne.
    ///
    /// Un bloc de `16 × 16` threads, soit 8 warps, calcule une tuile de
    /// sortie `128 × 128`. Chaque warp en calcule une région `32 × 64`, en
    /// seize multiplications par tranche de 16 le long de `inner`.
    ///
    /// Les multiplications elles-mêmes sont si rapides que c'est le
    /// chargement des tuiles qui limite ce kernel. Deux précautions le
    /// rendent supportable : une grande tuile de sortie, qui demande moins
    /// de lectures par produit, et un chemin rapide, sans aucun test, pour
    /// les tranches entièrement à l'intérieur des matrices.
    ///
    /// Le `bf16` garde les exposants du `f32` : contrairement au `f16`, il
    /// ne déborde ni ne s'annule sur les petits gradients, et la perte n'a
    /// pas besoin d'être mise à l'échelle.
    ///
    /// # Safety
    ///
    /// Mêmes exigences que `matmul_tiled`, avec des tuiles de
    /// [`MATMUL_BF16_TILE`].
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_bf16(
        a: &[f32],
        batch_a: Layout,
        b: &[f32],
        batch_b: Layout,
        rows: u32,
        inner: u32,
        cols: u32,
        depth: u32,
        out: *mut f32,
    ) {
        // `a` : 128 lignes de 8 paires. `b` : 128 colonnes de 8 paires.
        static mut TILE_A: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<u32, 1024> = SharedArray::UNINIT;

        const TILE: u32 = MATMUL_BF16_TILE as u32;
        const DEPTH: u32 = MATMUL_DEPTH as u32;
        /// Paires de `bf16` par ligne de `a` ou colonne de `b`.
        const PAIRS: u32 = DEPTH / 2;
        /// Paires que chaque thread charge dans chaque tuile.
        const LOADS: u32 = TILE * PAIRS / 256;

        // Rang du thread dans son bloc : les warps sont formés de 32 threads
        // consécutifs dans cet ordre.
        let slot = thread::threadIdx_y() * 16 + thread::threadIdx_x();
        let warp = slot / 32;
        let group = (slot % 32) / 4;
        let t = slot % 4;
        // Coin de la région `32 × 64` de ce warp dans la tuile de sortie.
        let warp_row = (warp / 2) * 32;
        let warp_col = (warp % 2) * 64;

        let tiles_per_matrix = rows.div_ceil(TILE);
        let block_y = thread::blockIdx_y();
        let batch = block_y / tiles_per_matrix;
        let tile_row = (block_y % tiles_per_matrix) * TILE;
        // L'axe `x` enchaîne les tranches de la dimension commune, chacune
        // sur toutes les tuiles de colonnes.
        let column_tiles = cols.div_ceil(TILE);
        let (split, tile_col) = (
            thread::blockIdx_x() / column_tiles,
            (thread::blockIdx_x() % column_tiles) * TILE,
        );
        let begin = split * depth;
        let end = if begin + depth < inner {
            begin + depth
        } else {
            inner
        };
        let matrices = thread::gridDim_y() / tiles_per_matrix;

        let start_a = batch_a.source_index(batch) * (rows * inner) as usize;
        let start_b = batch_b.source_index(batch) * (inner * cols) as usize;
        // La tuile de sortie tient-elle entièrement dans la matrice ? La
        // réponse est la même pour tous les threads du bloc.
        let inside = tile_row + TILE <= rows && tile_col + TILE <= cols;

        // Un accumulateur par multiplication : seize tuiles `16 × 8`, deux
        // rangées de huit. Ce sont des variables séparées, que le
        // compilateur garde dans des registres.
        let (mut c00, mut c01, mut c02, mut c03) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c04, mut c05, mut c06, mut c07) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c10, mut c11, mut c12, mut c13) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut c14, mut c15, mut c16, mut c17) =
            ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        let mut tile_start = begin;
        while tile_start < end {
            // 1. Chargement : chaque thread convertit et range quatre paires
            //    de chaque tuile.
            if inside && tile_start + DEPTH <= end {
                // Chemin rapide : tout ce qu'on lit existe.
                let mut i = 0;
                while i < LOADS {
                    let cell = slot * LOADS + i;
                    let index_a = start_a
                        + ((tile_row + cell / PAIRS) * inner + tile_start + 2 * (cell % PAIRS))
                            as usize;
                    // Les threads voisins lisent des colonnes voisines de `b`.
                    let index_b = start_b
                        + ((tile_start + 2 * (cell / TILE)) * cols + tile_col + cell % TILE)
                            as usize;
                    // SAFETY: les indices sont dans les matrices, d'après le
                    // test ci-dessus et les tailles garanties par l'hôte ;
                    // les deux cases écrites sont propres à ce thread.
                    unsafe {
                        let pa = a.as_ptr().add(index_a);
                        let pb = b.as_ptr().add(index_b);
                        TILE_A[cell as usize] = bf16_pair(*pa, *pa.add(1));
                        TILE_B[((cell % TILE) * PAIRS + cell / TILE) as usize] =
                            bf16_pair(*pb, *pb.add(cols as usize));
                    }
                    i += 1;
                }
            } else {
                // Au bord : hors des matrices, on met 0, qui ne change pas
                // les sommes.
                let mut i = 0;
                while i < LOADS {
                    let cell = slot * LOADS + i;

                    let a_row = tile_row + cell / PAIRS;
                    let a_col = tile_start + 2 * (cell % PAIRS);
                    let (mut lo, mut hi) = (0.0, 0.0);
                    if a_row < rows {
                        let base = start_a + (a_row * inner + a_col) as usize;
                        if a_col < end {
                            lo = a[base];
                        }
                        if a_col + 1 < end {
                            hi = a[base + 1];
                        }
                    }
                    let pair_a = bf16_pair(lo, hi);

                    let b_col = tile_col + cell % TILE;
                    let b_row = tile_start + 2 * (cell / TILE);
                    let (mut lo, mut hi) = (0.0, 0.0);
                    if b_col < cols {
                        let base = start_b + (b_row * cols + b_col) as usize;
                        if b_row < end {
                            lo = b[base];
                        }
                        if b_row + 1 < end {
                            hi = b[base + cols as usize];
                        }
                    }
                    let pair_b = bf16_pair(lo, hi);

                    // SAFETY: les deux cases sont propres à ce thread.
                    unsafe {
                        TILE_A[cell as usize] = pair_a;
                        TILE_B[((cell % TILE) * PAIRS + cell / TILE) as usize] = pair_b;
                    }
                    i += 1;
                }
            }
            // 2. Barrière : personne ne lit avant que tout le bloc ait écrit.
            thread::sync_threads();

            // 3. Calcul. Tous les threads du warp exécutent les mêmes seize
            //    instructions, sans branchement.
            // SAFETY: après la barrière, les tuiles ne sont plus que lues,
            // et les fragments suivent la disposition décrite plus haut.
            unsafe {
                let col_b = ((warp_col + group) * PAIRS + t) as usize;
                // Les huit fragments de `b`, communs aux deux rangées.
                macro_rules! frag_b {
                    ($n:expr) => {
                        [
                            TILE_B[col_b + $n * 8 * PAIRS as usize],
                            TILE_B[col_b + $n * 8 * PAIRS as usize + 4],
                        ]
                    };
                }
                let (b0, b1, b2, b3) = (frag_b!(0), frag_b!(1), frag_b!(2), frag_b!(3));
                let (b4, b5, b6, b7) = (frag_b!(4), frag_b!(5), frag_b!(6), frag_b!(7));

                let row_a = ((warp_row + group) * PAIRS + t) as usize;
                macro_rules! frag_a {
                    ($down:expr) => {{
                        let start = row_a + $down * 16 * PAIRS as usize;
                        [
                            TILE_A[start],
                            TILE_A[start + 8 * PAIRS as usize],
                            TILE_A[start + 4],
                            TILE_A[start + 8 * PAIRS as usize + 4],
                        ]
                    }};
                }
                let a0 = frag_a!(0);
                c00 = wmma::mma_m16n8k16_f32_bf16(c00, a0, b0);
                c01 = wmma::mma_m16n8k16_f32_bf16(c01, a0, b1);
                c02 = wmma::mma_m16n8k16_f32_bf16(c02, a0, b2);
                c03 = wmma::mma_m16n8k16_f32_bf16(c03, a0, b3);
                c04 = wmma::mma_m16n8k16_f32_bf16(c04, a0, b4);
                c05 = wmma::mma_m16n8k16_f32_bf16(c05, a0, b5);
                c06 = wmma::mma_m16n8k16_f32_bf16(c06, a0, b6);
                c07 = wmma::mma_m16n8k16_f32_bf16(c07, a0, b7);
                let a1 = frag_a!(1);
                c10 = wmma::mma_m16n8k16_f32_bf16(c10, a1, b0);
                c11 = wmma::mma_m16n8k16_f32_bf16(c11, a1, b1);
                c12 = wmma::mma_m16n8k16_f32_bf16(c12, a1, b2);
                c13 = wmma::mma_m16n8k16_f32_bf16(c13, a1, b3);
                c14 = wmma::mma_m16n8k16_f32_bf16(c14, a1, b4);
                c15 = wmma::mma_m16n8k16_f32_bf16(c15, a1, b5);
                c16 = wmma::mma_m16n8k16_f32_bf16(c16, a1, b6);
                c17 = wmma::mma_m16n8k16_f32_bf16(c17, a1, b7);
            }
            // 4. Barrière : personne ne recharge avant que tout le bloc ait lu.
            thread::sync_threads();

            tile_start += DEPTH;
        }

        // Chaque thread tient quatre éléments de chacune des seize tuiles.
        let tiles = [
            c00, c01, c02, c03, c04, c05, c06, c07, c10, c11, c12, c13, c14, c15, c16, c17,
        ];
        let mut n = 0;
        while n < 16 {
            let mut j = 0;
            while j < 4 {
                let row = tile_row + warp_row + 16 * (n / 8) + group + 8 * (j / 2);
                let col = tile_col + warp_col + 8 * (n % 8) + 2 * t + j % 2;
                if row < rows && col < cols {
                    // SAFETY: `(batch, row, col)` n'appartient qu'à ce thread
                    // et désigne un élément de `out`.
                    unsafe {
                        *out.add(
                            (((split * matrices + batch) * rows + row) * cols + col) as usize,
                        ) = tiles[n as usize][j as usize];
                    }
                }
                j += 1;
            }
            n += 1;
        }
    }

    /// Les deux réductions dont le softmax a besoin, pour chaque ligne de
    /// largeur `width` : son maximum, et `ln(Σ exp(x - max))`.
    ///
    /// Un warp par ligne, comme toutes les réductions par ligne de ce
    /// fichier : le thread de rang `l` dans son warp parcourt les éléments
    /// `l`, `l + 32`, `l + 64`… de la ligne, puis les 32 résultats partiels
    /// sont réunis par [`warp_sum`] ou [`warp_max`]. Les 32 threads lisent à
    /// chaque tour 32 éléments voisins, ce que la mémoire sert en une seule
    /// transaction. Avec un seul thread par ligne, chacun lirait seul dans
    /// son coin, et la ligne serait parcourue 32 fois plus lentement.
    ///
    /// Le lancement doit donc créer 32 threads par ligne.
    ///
    /// Les deux valeurs sont gardées séparées. Leur somme est `ln(Σ exp(x))`,
    /// mais l'additionner tout de suite noierait le second terme, petit,
    /// dans le maximum, qui peut être grand : on perdrait des chiffres.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_stats(
        a: &[f32],
        width: u32,
        mut row_max: DisjointSlice<f32>,
        mut row_log_sum: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        // Vrai ou faux pour tout le warp à la fois.
        if row < row_max.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let mut max = f32::NEG_INFINITY;
            let mut i = start + lane;
            while i < end {
                if a[i] > max {
                    max = a[i];
                }
                i += WARP;
            }
            let max = warp_max(max);
            let mut total = 0.0;
            let mut i = start + lane;
            while i < end {
                total += (a[i] - max).exp();
                i += WARP;
            }
            let total = warp_sum(total);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *row_max.as_mut_ptr().add(row) = max;
                    *row_log_sum.as_mut_ptr().add(row) = total.ln();
                }
            }
        }
    }

    /// Softmax ou log-softmax d'un élément à partir des réductions de sa
    /// ligne : `ln softmax(x) = (x - max) - ln(Σ exp(x - max))`. Soustraire
    /// le maximum d'abord garde la précision, quelle que soit la taille des
    /// valeurs.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_rows(
        mode: SoftmaxMode,
        a: &[f32],
        row_max: &[f32],
        row_log_sum: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let log_probability = (a[flat] - row_max[row]) - row_log_sum[row];
            *out_elem = match mode {
                SoftmaxMode::Log => log_probability,
                SoftmaxMode::Probability => log_probability.exp(),
            };
        }
    }

    /// Les deux réductions de la normalisation, pour chaque ligne de
    /// largeur `width` : sa moyenne, et `1 / √(variance + epsilon)`. Un
    /// warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_moments(
        a: &[f32],
        width: u32,
        epsilon: f32,
        mut mean: DisjointSlice<f32>,
        mut inverse_std: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < mean.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let mut total = 0.0;
            let mut i = start + lane;
            while i < end {
                total += a[i];
                i += WARP;
            }
            let row_mean = warp_sum(total) / width as f32;
            let mut squares = 0.0;
            let mut i = start + lane;
            while i < end {
                let centered = a[i] - row_mean;
                squares += centered * centered;
                i += WARP;
            }
            let squares = warp_sum(squares);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *mean.as_mut_ptr().add(row) = row_mean;
                    *inverse_std.as_mut_ptr().add(row) =
                        1.0 / (squares / width as f32 + epsilon).sqrt();
                }
            }
        }
    }

    /// `1 / √(moyenne(x²) + epsilon)` pour chaque ligne de largeur `width` :
    /// la réduction de RMSNorm. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_inverse_rms(a: &[f32], width: u32, epsilon: f32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let mut squares = 0.0;
            let mut i = start + lane;
            while i < end {
                squares += a[i] * a[i];
                i += WARP;
            }
            let squares = warp_sum(squares);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *out.as_mut_ptr().add(row) = 1.0 / (squares / width as f32 + epsilon).sqrt();
                }
            }
        }
    }

    /// Multiplie chaque élément par le facteur de sa ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scale_rows(a: &[f32], row_scale: &[f32], width: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a[flat] * row_scale[flat / width as usize];
        }
    }

    /// Rétropropagation de RMSNorm, un thread par élément :
    /// `s × (g - x̂ × moyenne(g × x̂))`. `grad_dot` est la somme par ligne
    /// de `g × x̂`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn rms_grad_rows(
        normalized: &[f32],
        grad: &[f32],
        grad_dot: &[f32],
        inverse_rms: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let mean_projection = grad_dot[row] / width as f32;
            *out_elem = inverse_rms[row] * (grad[flat] - normalized[flat] * mean_projection);
        }
    }

    /// La rotation de RoPE, un thread par élément. `a` est vu comme des
    /// lignes de `width` composantes, qui se suivent position après
    /// position sur `length` positions ; `cos` et `sin` sont
    /// `[length, width / 2]`. `sign` vaut 1, ou -1 pour la rotation inverse.
    ///
    /// Chaque thread lit les deux composantes de sa paire et écrit la
    /// sienne : la première reçoit `x₀·cos - x₁·sin`, la seconde
    /// `x₀·sin + x₁·cos`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn rotate_pairs(
        a: &[f32],
        cos: &[f32],
        sin: &[f32],
        width: u32,
        length: u32,
        sign: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (width, length) = (width as usize, length as usize);
            let component = flat % width;
            let position = (flat / width) % length;
            let angle = position * (width / 2) + component / 2;
            let (c, s) = (cos[angle], sin[angle] * sign);
            // Le premier élément de la paire, que le thread soit le premier
            // ou le second.
            let first = flat - component % 2;
            let (x0, x1) = (a[first], a[first + 1]);
            *out_elem = if component % 2 == 0 {
                x0 * c - x1 * s
            } else {
                x0 * s + x1 * c
            };
        }
    }

    /// Concaténation le long du dernier axe : chaque ligne de la sortie est
    /// une ligne de `a`, de largeur `width_a`, suivie de la ligne de même
    /// rang de `b`, de largeur `width_b`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn concat_last(
        a: &[f32],
        b: &[f32],
        width_a: u32,
        width_b: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (width_a, width_b) = (width_a as usize, width_b as usize);
            let row = flat / (width_a + width_b);
            let column = flat % (width_a + width_b);
            *out_elem = if column < width_a {
                a[row * width_a + column]
            } else {
                b[row * width_b + column - width_a]
            };
        }
    }

    /// L'inverse de `concat_last` : de chaque ligne de `a`, de largeur
    /// `total`, garde les `width` composantes qui commencent à `offset`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn slice_last(a: &[f32], total: u32, offset: u32, width: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (total, offset, width) = (total as usize, offset as usize, width as usize);
            let row = flat / width;
            let column = flat % width;
            *out_elem = a[row * total + offset + column];
        }
    }

    /// Normalisation d'un élément à partir des réductions de sa ligne :
    /// `(x - moyenne) × écart-type inverse`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn normalize_rows(
        a: &[f32],
        mean: &[f32],
        inverse_std: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            *out_elem = (a[flat] - mean[row]) * inverse_std[row];
        }
    }

    /// Produit scalaire de chaque ligne de `a` par la ligne correspondante
    /// de `b` : `Σ a × b`. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_dot(a: &[f32], b: &[f32], width: u32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let mut total = 0.0;
            let mut i = start + lane;
            while i < end {
                total += a[i] * b[i];
                i += WARP;
            }
            let total = warp_sum(total);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *out.as_mut_ptr().add(row) = total;
                }
            }
        }
    }

    /// Somme de chaque ligne de largeur `width`. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn row_sum(a: &[f32], width: u32, mut out: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let mut total = 0.0;
            let mut i = start + lane;
            while i < end {
                total += a[i];
                i += WARP;
            }
            let total = warp_sum(total);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *out.as_mut_ptr().add(row) = total;
                }
            }
        }
    }

    /// Rétropropagation de la normalisation, un thread par élément :
    ///
    /// ```text
    /// s × (g - moyenne(g) - x̂ × moyenne(g × x̂))
    /// ```
    ///
    /// `grad_sum` et `grad_dot` sont les sommes par ligne de `g` et de
    /// `g × x̂` ; on les divise ici par `width` pour en faire des moyennes.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn normalize_grad_rows(
        normalized: &[f32],
        grad: &[f32],
        grad_sum: &[f32],
        grad_dot: &[f32],
        inverse_std: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let mean_grad = grad_sum[row] / width as f32;
            let mean_projection = grad_dot[row] / width as f32;
            *out_elem =
                inverse_std[row] * (grad[flat] - mean_grad - normalized[flat] * mean_projection);
        }
    }

    /// Comme `row_stats`, pour le softmax causal de scores `[…, N, T]` :
    /// `queries` requêtes face à `width` positions, dont elles sont les
    /// dernières. La ligne `i` d'une matrice est à la position
    /// `width - queries + i` et ne tient compte que des colonnes jusqu'à la
    /// sienne, les suivantes étant son futur. Les scores sont multipliés par
    /// `scale` avant tout calcul.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn causal_row_stats(
        a: &[f32],
        width: u32,
        queries: u32,
        scale: f32,
        mut row_max: DisjointSlice<f32>,
        mut row_log_sum: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < row_max.len() {
            let start = row * width as usize;
            // Position de cette ligne parmi les `width` positions : c'est
            // aussi le nombre de colonnes visibles, moins une.
            let position = (width - queries) as usize + row % queries as usize;
            let end = start + position + 1;
            let mut max = f32::NEG_INFINITY;
            let mut i = start + lane;
            while i < end {
                let x = a[i] * scale;
                if x > max {
                    max = x;
                }
                i += WARP;
            }
            let max = warp_max(max);
            let mut total = 0.0;
            let mut i = start + lane;
            while i < end {
                total += (a[i] * scale - max).exp();
                i += WARP;
            }
            let total = warp_sum(total);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *row_max.as_mut_ptr().add(row) = max;
                    *row_log_sum.as_mut_ptr().add(row) = total.ln();
                }
            }
        }
    }

    /// Softmax causal d'un élément à partir des réductions de sa ligne. Les
    /// colonnes du futur reçoivent exactement 0.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn causal_softmax_rows(
        a: &[f32],
        row_max: &[f32],
        row_log_sum: &[f32],
        width: u32,
        queries: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (width, queries) = (width as usize, queries as usize);
            let row = flat / width;
            let col = flat % width;
            let position = width - queries + row % queries;
            *out_elem = if col > position {
                0.0
            } else {
                ((a[flat] * scale - row_max[row]) - row_log_sum[row]).exp()
            };
        }
    }

    /// Rétropropagation d'un softmax, un thread par élément :
    /// `scale × p × (g - Σ g × p)`. `row_dot` contient `Σ g × p` par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn softmax_grad_rows(
        probabilities: &[f32],
        grad: &[f32],
        row_dot: &[f32],
        width: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            *out_elem = scale * probabilities[flat] * (grad[flat] - row_dot[row]);
        }
    }

    /// Recherche d'embedding : la ligne `r` de la sortie est la ligne
    /// `indices[r]` de `table`. Un thread par élément de sortie.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn gather_rows(table: &[f32], indices: &[u32], row_len: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let flat = idx.get() as u32;
        if let Some(out_elem) = out.get_mut(idx) {
            let row = indices[(flat / row_len) as usize];
            *out_elem = table[(row * row_len + flat % row_len) as usize];
        }
    }

    /// Écrit la ligne `r` de `rows` à la ligne `indices[r]` de `table`, sur
    /// place. Les indices sont des entiers rangés en `f32`. Un thread par
    /// élément écrit.
    ///
    /// # Safety
    ///
    /// `table` doit pointer sur un tampon GPU d'au moins
    /// `(max(indices) + 1) × row_len` éléments, que rien d'autre n'utilise
    /// pendant le kernel, et les indices doivent être distincts.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub unsafe fn write_rows(table: *mut f32, indices: &[f32], rows: &[f32], row_len: u32) {
        let flat = thread::index_1d().get();
        if flat < rows.len() {
            let row_len = row_len as usize;
            let index = indices[flat / row_len] as usize;
            // SAFETY: d'après le contrat, la ligne `index` existe et
            // n'appartient qu'à la ligne `flat / row_len` de `rows`.
            unsafe {
                *table.add(index * row_len + flat % row_len) = rows[flat];
            }
        }
    }

    /// Met `b` à la suite de `a` : `out = a ++ b`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn concat(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = if i < a.len() { a[i] } else { b[i - a.len()] };
        }
    }

    /// Inverse de `gather_rows` : la ligne `r` de `src` est ajoutée à la
    /// ligne `indices[r]` de `out`, qui doit être nul au départ. Un thread
    /// par élément de `src`.
    ///
    /// C'est le seul kernel où deux threads peuvent viser la même case :
    /// deux lignes de `src` qui portent le même indice s'additionnent au
    /// même endroit. Une addition ordinaire (lire, ajouter, écrire) perdrait
    /// des contributions quand deux threads lisent la même ancienne valeur.
    /// L'addition atomique fait les trois étapes d'un bloc.
    ///
    /// `out` est un pointeur brut et non un `DisjointSlice`, qui garantit
    /// justement qu'aucune case n'est partagée.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    ///
    /// # Safety
    ///
    /// `out` doit pointer sur un tampon GPU assez grand pour toutes les
    /// lignes désignées par `indices`, et rien d'autre ne doit y accéder
    /// pendant le lancement.
    pub unsafe fn scatter_add_rows(src: &[f32], indices: &[u32], row_len: u32, out: *mut f32) {
        let flat = thread::index_1d().get();
        if flat < src.len() {
            let row_len = row_len as usize;
            let row = indices[flat / row_len] as usize;
            let target = row * row_len + flat % row_len;
            // SAFETY: l'hôte a vérifié que chaque indice désigne une ligne
            // de `out`, et `out` n'est lu ou écrit que par des opérations
            // atomiques pendant ce lancement.
            let cell = unsafe { DeviceAtomicF32::from_ptr(out.add(target)) };
            // `const { … }` : le compilateur de kernels exige que l'ordre
            // soit une constante au point d'appel, ce qu'une simple valeur
            // n'est pas en compilation non optimisée.
            cell.fetch_add(src[flat], const { AtomicOrdering::Relaxed });
        }
    }

    /// Perte de chaque ligne de logits : `-ln softmax(x)[cible]`, soit
    /// `ln(Σ exp(x - max)) - (x[cible] - max)`. Un warp par ligne.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn cross_entropy_rows(
        logits: &[f32],
        targets: &[u32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let (row, lane) = (index / WARP, index % WARP);
        if row < out.len() {
            let start = row * width as usize;
            let end = start + width as usize;
            let mut max = f32::NEG_INFINITY;
            let mut i = start + lane;
            while i < end {
                if logits[i] > max {
                    max = logits[i];
                }
                i += WARP;
            }
            let max = warp_max(max);
            let mut total = 0.0;
            let mut i = start + lane;
            while i < end {
                total += (logits[i] - max).exp();
                i += WARP;
            }
            let total = warp_sum(total);
            if lane == 0 {
                // SAFETY: un seul thread par warp écrit, et chaque warp a
                // sa propre ligne.
                unsafe {
                    *out.as_mut_ptr().add(row) =
                        total.ln() - (logits[start + targets[row] as usize] - max);
                }
            }
        }
    }

    /// Gradient de l'entropie croisée moyenne par rapport aux logits :
    /// `(softmax(x) - onehot(cible)) × scale`, avec `scale = 1 / N`.
    /// Un thread par logit ; `row_max` et `row_log_sum` viennent de
    /// `row_stats`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn cross_entropy_grad(
        logits: &[f32],
        row_max: &[f32],
        row_log_sum: &[f32],
        targets: &[u32],
        width: u32,
        scale: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = flat / width as usize;
            let col = (flat % width as usize) as u32;
            let probability = ((logits[flat] - row_max[row]) - row_log_sum[row]).exp();
            let expected = if col == targets[row] { 1.0 } else { 0.0 };
            *out_elem = (probability - expected) * scale;
        }
    }

    /// Descente de gradient sur place : `param[i] -= learning_rate × grad[i]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sgd_step(mut param: DisjointSlice<f32>, grad: &[f32], learning_rate: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(p) = param.get_mut(idx) {
            *p -= learning_rate * grad[i];
        }
    }

    /// Un pas d'AdamW sur place : chaque thread met à jour un élément du
    /// paramètre et les deux moments qui lui correspondent. Tout l'optimiseur
    /// tient dans ce seul lancement par paramètre.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn adamw_step(
        mut param: DisjointSlice<f32>,
        grad: &[f32],
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        step: AdamWParams,
    ) {
        let i = thread::index_1d().get();
        if let Some(p) = param.get_mut(thread::index_1d())
            && let Some(m) = first_moment.get_mut(thread::index_1d())
            && let Some(v) = second_moment.get_mut(thread::index_1d())
        {
            // L'écrêtage de la norme réduit tout le gradient d'un même facteur.
            let g = grad[i] * step.grad_scale;
            // Moyennes glissantes du gradient et de son carré.
            *m = step.beta1 * *m + (1.0 - step.beta1) * g;
            *v = step.beta2 * *v + (1.0 - step.beta2) * g * g;

            // Les moments partent de zéro : on corrige ce biais.
            let m_hat = *m / step.bias_correction1;
            let v_hat = *v / step.bias_correction2;

            // La décroissance des poids s'applique directement au
            // paramètre, sans passer par les moments.
            let update = m_hat / (v_hat.sqrt() + step.epsilon);
            *p -= step.learning_rate * (update + step.weight_decay * *p);
        }
    }

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

    // Répète `$body` pour chaque valeur de la liste, `$var` y valant cette
    // valeur, constante. Une boucle ordinaire n'est pas déroulée par le
    // compilateur de kernels : les tableaux qu'elle indexe iraient en
    // mémoire locale, bien plus lente que les registres. Avec des indices
    // constants, chaque case devient une variable.
    macro_rules! unroll {
        ($var:ident in [$($value:literal),*] $body:block) => {
            $({
                #[allow(non_upper_case_globals)]
                const $var: usize = $value;
                $body
            })*
        };
    }

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

    /// Le produit matriciel quand `a` n'a que quelques lignes : celui de la
    /// génération, où chaque séquence n'apporte qu'un vecteur par token.
    ///
    /// Les tuiles de `matmul_tiled` y gaspillent presque tout. Une tuile de
    /// 64 lignes n'en a qu'une d'utile, et la grille, qui compte un bloc par
    /// tuile, n'a que 8 blocs pour une sortie de 512 colonnes : 8 des 82
    /// multiprocesseurs de la carte travaillent. Chaque poids sert à une
    /// seule multiplication, si bien que le temps se passe à attendre la
    /// mémoire, avec trop peu de lectures en cours pour l'occuper.
    ///
    /// Ici, le travail est découpé finement. Un bloc de 8 warps couvre 32
    /// colonnes et une tranche de `depth` indices le long de `inner`. Les 32
    /// threads d'un warp lisent 32 colonnes voisines d'une même ligne de
    /// `b`, une seule transaction. Les 8 warps se partagent la tranche, un
    /// indice sur huit chacun, quatre indices à la fois pour garder
    /// plusieurs lectures en cours. Chaque poids lu sert aux lignes de `a`
    /// par paquets de 8, et les sommes des 8 warps sont réunies en mémoire
    /// partagée.
    ///
    /// Quand une tranche ne couvre pas tout `inner`, chacune écrit sa somme
    /// partielle dans sa propre copie de la sortie, et `sum_splits` les
    /// additionne ensuite. Pas d'addition atomique : le résultat ne dépend
    /// pas de l'ordre dans lequel les blocs s'exécutent.
    ///
    /// Avec `has_bias`, le biais `bias` `[cols]` est ajouté à chaque ligne :
    /// c'est la couche affine `x · W + b` en un seul passage.
    ///
    /// La grille est à deux dimensions : `x` parcourt les colonnes par 32,
    /// `y` les matrices du lot, chacune découpée en tranches.
    ///
    /// # Safety
    ///
    /// Le lancement doit couvrir `cols` en `x` par blocs de `32 × 8`
    /// threads, et `lots × tranches` en `y`, où `tranches` vaut
    /// `inner.div_ceil(depth)`. `out` doit pointer sur un tampon GPU de
    /// `lots × tranches × rows × cols` éléments, rangés dans cet ordre.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_narrow(
        a: &[f32],
        batch_a: Layout,
        b: &[f32],
        batch_b: Layout,
        rows: u32,
        inner: u32,
        cols: u32,
        depth: u32,
        bias: &[f32],
        has_bias: u32,
        out: *mut f32,
    ) {
        // La somme partielle de chaque warp, pour 8 lignes et 32 colonnes.
        static mut SUMS: SharedArray<f32, 2048> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let warp = thread::threadIdx_y() as usize;
        let (rows, inner, cols) = (rows as usize, inner as usize, cols as usize);
        let depth = depth as usize;
        let splits = inner.div_ceil(depth);
        let block_y = thread::blockIdx_y() as usize;
        let (batch, split) = (block_y / splits, block_y % splits);
        let col = thread::blockIdx_x() as usize * WARP + lane;
        let begin = split * depth;
        let end = if begin + depth < inner {
            begin + depth
        } else {
            inner
        };
        let start_a = batch_a.source_index(batch as u32) * rows * inner;
        let start_b = batch_b.source_index(batch as u32) * inner * cols;

        let mut first_row = 0;
        while first_row < rows {
            let mut sums = [0.0f32; 8];
            if col < cols {
                let mut k = begin + warp;
                while k < end {
                    // Les quatre poids d'abord : leurs lectures partent
                    // ensemble, au lieu d'attendre chacune la précédente.
                    let mut weights = [0.0f32; 4];
                    unroll!(i in [0, 1, 2, 3] {
                        if k + 8 * i < end {
                            // SAFETY: `k + 8i < inner` et `col < cols`.
                            weights[i] = unsafe {
                                *b.as_ptr().add(start_b + (k + 8 * i) * cols + col)
                            };
                        }
                    });
                    unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                        if first_row + r < rows {
                            let row = start_a + (first_row + r) * inner + k;
                            unroll!(i in [0, 1, 2, 3] {
                                if k + 8 * i < end {
                                    // SAFETY: la ligne existe, `k + 8i < inner`.
                                    sums[r] += unsafe { *a.as_ptr().add(row + 8 * i) }
                                        * weights[i];
                                }
                            });
                        }
                    });
                    k += 32;
                }
            }
            // SAFETY: chaque thread écrit ses propres cases.
            unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                unsafe { SUMS[(warp * 8 + r) * WARP + lane] = sums[r] };
            });
            thread::sync_threads();

            // Le warp `w` réunit la ligne `w` du paquet.
            let row = first_row + warp;
            if row < rows && col < cols {
                // Le biais, une seule fois : par la seule tranche quand il
                // n'y en a qu'une, sinon par la réunion des tranches.
                let mut total = if has_bias != 0 && splits == 1 {
                    bias[col]
                } else {
                    0.0
                };
                // SAFETY: après la barrière, les sommes ne sont plus que lues.
                unroll!(w in [0, 1, 2, 3, 4, 5, 6, 7] {
                    total += unsafe { SUMS[(w * 8 + warp) * WARP + lane] };
                });
                // SAFETY: `(batch, split, row, col)` n'appartient qu'à ce
                // thread et désigne un élément de `out`.
                unsafe {
                    *out.add(((batch * splits + split) * rows + row) * cols + col) = total;
                }
            }
            // Personne ne réécrit les sommes avant qu'elles aient été lues.
            thread::sync_threads();
            first_row += 8;
        }
    }

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

    /// `a · bᵀ` quand `a` n'a que quelques lignes : la sortie du modèle en
    /// génération, où `b` est la table d'embedding `[N, K]`.
    ///
    /// Chaque élément de sortie est le produit scalaire d'une ligne de `a`
    /// et d'une ligne de `b`. Un warp s'occupe d'une ligne de `b` : ses 32
    /// threads la lisent par morceaux voisins, quatre à la fois, puis
    /// réunissent leurs sommes partielles par échanges entre registres. La
    /// ligne de `b` sert aux lignes de `a` par paquets de 8. Un bloc de 8
    /// warps couvre 8 lignes de `b`.
    ///
    /// # Safety
    ///
    /// Le lancement doit couvrir `n` lignes de `b` en `x`, par blocs de
    /// `32 × 8` threads, et `out` doit pointer sur un tampon GPU de
    /// `rows × n` éléments.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub unsafe fn transposed_matmul_narrow(
        a: &[f32],
        b: &[f32],
        rows: u32,
        inner: u32,
        n: u32,
        out: *mut f32,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let (rows, inner, n) = (rows as usize, inner as usize, n as usize);
        let line = thread::blockIdx_x() as usize * 8 + thread::threadIdx_y() as usize;
        if line < n {
            let start_b = line * inner;
            let mut first_row = 0;
            while first_row < rows {
                let mut sums = [0.0f32; 8];
                let mut k = lane;
                while k < inner {
                    let mut weights = [0.0f32; 4];
                    unroll!(i in [0, 1, 2, 3] {
                        if k + 32 * i < inner {
                            // SAFETY: `line < n` et `k + 32i < inner`.
                            weights[i] = unsafe { *b.as_ptr().add(start_b + k + 32 * i) };
                        }
                    });
                    unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                        if first_row + r < rows {
                            let row = (first_row + r) * inner + k;
                            unroll!(i in [0, 1, 2, 3] {
                                if k + 32 * i < inner {
                                    // SAFETY: la ligne existe, `k + 32i < inner`.
                                    sums[r] += unsafe { *a.as_ptr().add(row + 32 * i) }
                                        * weights[i];
                                }
                            });
                        }
                    });
                    k += 128;
                }
                unroll!(r in [0, 1, 2, 3, 4, 5, 6, 7] {
                    let total = warp_sum(sums[r]);
                    if lane == 0 && first_row + r < rows {
                        // SAFETY: `(row, line)` n'appartient qu'à ce warp.
                        unsafe { *out.add((first_row + r) * n + line) = total };
                    }
                });
                first_row += 8;
            }
        }
    }

    /// Réunit les tranches de `matmul_narrow` : `parts` est
    /// `[lots, tranches, inner]`, la sortie `[lots, inner]`, et chaque
    /// élément est la somme de ses tranches, plus le biais de sa colonne
    /// avec `has_bias`. Un thread par élément de sortie.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn sum_splits(
        parts: &[f32],
        splits: u32,
        inner: u32,
        bias: &[f32],
        has_bias: u32,
        cols: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let flat = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let (splits, inner) = (splits as usize, inner as usize);
            let (batch, rest) = (flat / inner, flat % inner);
            let mut total = if has_bias != 0 {
                bias[rest % cols as usize]
            } else {
                0.0
            };
            for split in 0..splits {
                total += parts[(batch * splits + split) * inner + rest];
            }
            *out_elem = total;
        }
    }

    /// `out[i] = value` : un tenseur rempli d'une même valeur, sans rien
    /// envoyer depuis l'hôte.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn fill(value: f32, mut out: DisjointSlice<f32>) {
        if let Some(out_elem) = out.get_mut(thread::index_1d()) {
            *out_elem = value;
        }
    }

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
