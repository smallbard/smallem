//! Un backend pour le processeur, qui en exploite les cœurs et les
//! instructions vectorielles.
//!
//! [`Cpu`](super::cpu::Cpu) reste la référence : un seul fil, la triple
//! boucle, le calcul écrit pour qu'on le croie. [`Fast`] calcule la même
//! chose, en `f32`, aussi vite que la machine le permet :
//!
//! - les opérations se partagent entre un groupe de fils permanent
//!   ([`pool`]) ;
//! - le produit matriciel est découpé en blocs qui tiennent dans les caches,
//!   autour d'un micro-kernel AVX-512 ou AVX2 ([`gemm`], [`kernels`]) ;
//! - les calculs d'une ligne, exponentielle comprise, sont vectorisés
//!   ([`math`]) ;
//! - chaque opération a une écriture directe, souvent en une passe par
//!   ligne, au lieu de l'enchaînement de primitives de sa définition
//!   (`ops.rs`), et l'attention se calcule en tuiles (`attention.rs`) ;
//! - les poids quantifiés se lisent en octets (`quantized.rs`, `vnni.rs`) ;
//! - les tampons abandonnés retournent dans une réserve ([`Buffer`]).
//!
//! Les instructions sont détectées à l'exécution : la même compilation tourne
//! sur un processeur sans AVX-512, avec les kernels AVX2, ou sans AVX2.
//!
//! Le découpage du travail entre les fils est fixé par la forme des tenseurs,
//! jamais par le nombre de fils : avec un, deux ou seize fils, un calcul
//! donne les mêmes bits.

/// Compile une fonction écrite une fois en trois versions : AVX-512, AVX2,
/// et sans instructions vectorielles explicites. `$body` est une fonction
/// `#[inline(always)]` générique sur `FMA`, qui dit si l'on peut fusionner
/// multiplication et addition. La fonction obtenue vérifie les instructions
/// qu'on lui demande.
macro_rules! multiversion {
    (
        $(#[$attr:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty = $body:ident;
    ) => {
        $(#[$attr])*
        pub fn $name(isa: $crate::backend::fast::Isa, $($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
            unsafe fn wide($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn narrow($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            isa.check();
            match isa {
                // SAFETY : `check` a vérifié que le processeur a ces
                // instructions.
                #[cfg(target_arch = "x86_64")]
                $crate::backend::fast::Isa::Avx512 => unsafe { wide($($arg),*) },
                #[cfg(target_arch = "x86_64")]
                $crate::backend::fast::Isa::Avx2 => unsafe { narrow($($arg),*) },
                _ => $body::<false>($($arg),*),
            }
        }
    };
}

mod attention;
mod buffer;
pub mod gemm;
mod isa;
pub mod kernels;
pub mod math;
mod ops;
pub mod pool;
mod primitives;
mod quantized;
mod tensor;
#[cfg(test)]
mod tests;
mod vnni;

use std::sync::Arc;

pub use buffer::Buffer;
use gemm::Matrix;
pub use isa::{Isa, default_threads};
use pool::Pool;
pub use tensor::FastTensor;

/// `a × b + c`, fusionné quand le processeur sait le faire.
#[inline(always)]
pub(crate) fn fma<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// Le backend.
#[derive(Clone)]
pub struct Fast {
    pool: Arc<Pool>,
    isa: Isa,
    /// Les matrices quantifiées, rangées pour le processeur.
    natives: Arc<quantized::Natives<quantized::Native>>,
    /// Les mêmes, rangées pour `vpdpbusd`.
    packed: Arc<quantized::Natives<vnni::Packed>>,
    /// Quantifier aussi les activations en 8 bits devant une matrice
    /// quantifiée (voir `vnni.rs`).
    int8_activations: bool,
    /// Les grands produits en `bf16`, accumulés en `f32`.
    mixed: bool,
}

impl Default for Fast {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Fast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Fast({} fils, {:?})", self.threads(), self.isa)
    }
}

/// Une tâche fait au moins ce nombre d'éléments : en dessous, partager le
/// travail coûte plus que le travail lui-même (mesuré à l'étape C1).
const CHUNK: usize = 1 << 14;

impl Fast {
    /// Un fil par cœur physique, les meilleures instructions du processeur.
    pub fn new() -> Self {
        Self::with_threads(default_threads())
    }

    pub fn with_threads(threads: usize) -> Self {
        Self {
            pool: Arc::new(Pool::new(threads)),
            isa: Isa::detect(),
            natives: Arc::default(),
            packed: Arc::default(),
            int8_activations: false,
            mixed: false,
        }
    }

    /// Les mêmes fils, avec d'autres instructions : pour comparer les
    /// kernels entre eux.
    pub fn with_isa(&self, isa: Isa) -> Self {
        isa.check();
        Self {
            pool: Arc::clone(&self.pool),
            isa,
            natives: Arc::clone(&self.natives),
            packed: Arc::clone(&self.packed),
            int8_activations: self.int8_activations,
            mixed: self.mixed,
        }
    }

    /// La précision mixte : les opérandes des grands produits arrondis en
    /// `bf16`, leurs produits accumulés en `f32` avec `vdpbf16ps` (AVX-512
    /// BF16), deux fois plus vite qu'en `f32`. Tout le reste, poids compris,
    /// reste en `f32`, comme sur la carte. Sans ces instructions, rien ne
    /// change.
    pub fn with_mixed_precision(&self, on: bool) -> Self {
        Self {
            mixed: on,
            ..self.clone()
        }
    }

    /// `C (+)= A · B`, dans la précision du backend.
    pub(crate) fn gemm_into(&self, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
        if self.mixed {
            gemm::gemm_mixed(&self.pool, self.isa, a, b, c, accumulate);
        } else {
            gemm::gemm(&self.pool, self.isa, a, b, c, accumulate);
        }
    }

    /// Devant une matrice quantifiée, quantifier aussi les activations en
    /// 8 bits, pour multiplier des octets par des octets avec `vpdpbusd`
    /// (AVX-512 VNNI). Plus rapide quand les poids sont dans les caches,
    /// mais le produit n'est plus celui des poids déquantifiés : les
    /// activations perdent leur précision (étape C5 de `PLAN-CPU.md`).
    pub fn with_int8_activations(&self, on: bool) -> Self {
        Self {
            int8_activations: on,
            ..self.clone()
        }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }
}
