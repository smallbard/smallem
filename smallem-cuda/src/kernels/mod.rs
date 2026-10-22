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
//!
//! Les kernels sont rangés par famille, un fichier chacune, avec son
//! `#[cuda_module]` : [`elementwise`], [`rows`], [`matmul`], [`quantized`],
//! [`attention`], [`attention_bf16`] et [`cached`]. Ce qu'elles partagent,
//! formes, réglages et échanges dans un warp, est ici.

// Les boucles des kernels indexent explicitement : c'est l'accès mémoire que
// chaque thread effectue, et on veut le lire tel quel.
#![allow(clippy::needless_range_loop)]

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{
    DisjointSlice, SharedArray, device, kernel, launch_bounds, launch_contract, thread, warp, wmma,
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

/// Somme des valeurs `value(i)`, pour `i` de `start` à `end` exclus, que
/// les 32 threads d'un warp se partagent : celui de rang `lane` lit
/// `start + lane`, puis de 32 en 32, et [`warp_sum`] réunit les sommes
/// partielles. Les 32 threads lisent ainsi à chaque tour 32 éléments
/// voisins, ce que la mémoire sert en une seule transaction. Chaque thread
/// reçoit le total.
///
/// Tous les threads du warp doivent appeler cette fonction ensemble.
#[inline(always)]
pub fn warp_row_sum(start: usize, end: usize, lane: usize, value: impl Fn(usize) -> f32) -> f32 {
    let mut total = 0.0;
    let mut i = start + lane;
    while i < end {
        total += value(i);
        i += WARP;
    }
    warp_sum(total)
}

/// Le maximum des valeurs `value(i)`, pour `i` de `start` à `end` exclus,
/// et `Σ exp(value(i) - max)`, partagés entre les threads d'un warp comme
/// dans [`warp_row_sum`]. Chaque thread reçoit les deux.
///
/// Les deux sont gardés séparés : leur somme `max + ln(Σ…)` est
/// `ln(Σ exp(x))`, mais l'additionner tout de suite noierait le second
/// terme, petit, dans le maximum, qui peut être grand.
#[inline(always)]
pub fn warp_row_max_and_exp_sum(
    start: usize,
    end: usize,
    lane: usize,
    value: impl Fn(usize) -> f32,
) -> (f32, f32) {
    let mut max = f32::NEG_INFINITY;
    let mut i = start + lane;
    while i < end {
        let x = value(i);
        if x > max {
            max = x;
        }
        i += WARP;
    }
    let max = warp_max(max);
    (
        max,
        warp_row_sum(start, end, lane, |i| (value(i) - max).exp()),
    )
}

/// Écrit `value` à la ligne du warp de ce thread dans `out`, pour les
/// kernels qui donnent un warp à chaque ligne : la ligne `indice / 32`.
/// Seul le premier thread du warp écrit ; les warps au-delà de la dernière
/// ligne n'écrivent rien.
///
/// Les 32 threads d'un warp ont la même ligne, et deux warps n'ont jamais
/// la même : chaque case n'est écrite que par un thread.
#[device]
pub fn write_warp_row(out: &mut DisjointSlice<f32>, value: f32) {
    let index = thread::index_1d().get();
    let (row, lane) = (index / WARP, index % WARP);
    if lane == 0 && row < out.len() {
        // SAFETY: `row < out.len()`, et seul le premier thread du warp de
        // la ligne `row` arrive ici.
        unsafe { *out.as_mut_ptr().add(row) = value };
    }
}

/// Côté des tuiles de l'attention fusionnée : un bloc traite 32 requêtes
/// (ou 32 clés) à la fois.
pub const ATTENTION_TILE: usize = 32;

/// Taille maximale des têtes que traitent les kernels d'attention : la
/// largeur des requêtes, des clés et des valeurs. Au-delà, l'opération est
/// calculée avec les primitives.
pub const ATTENTION_MAX_DIM: usize = 64;

/// Côté des tuiles de l'attention fusionnée sur les Tensor Cores : quatre
/// warps de seize lignes.
pub const WIDE_TILE: usize = 64;

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

pub mod attention;
pub mod attention_bf16;
pub mod cached;
pub mod elementwise;
pub mod matmul;
pub mod quantized;
pub mod rows;
