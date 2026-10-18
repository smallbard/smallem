//! Le backend GPU de Smallem : les primitives et les opérations de
//! `smallem::backend`, exécutées par des kernels CUDA écrits en Rust.
//!
//! - [`kernels`] : le code qui tourne sur la carte ;
//! - [`cuda`] : le backend, qui garde les tenseurs sur la carte et lance les
//!   kernels ;
//! - [`ops`] : les opérations de `smallem::backend::ops` sur GPU ;
//! - [`profile`] : le même backend, qui chronomètre chaque opération.
//!
//! Ce paquet se compile avec `cargo oxide`, qui traduit les kernels pour la
//! carte. Tout le reste du projet, dans le paquet `smallem`, est du Rust
//! ordinaire et s'en passe.

pub mod cuda;
pub mod kernels;
pub mod ops;
pub mod profile;

use std::error::Error;

pub use cuda::{Cuda, CudaTensor};

/// Ouvre le premier GPU. Avec `mixed`, les produits matriciels passent par
/// les Tensor Cores.
pub fn gpu(mixed: bool) -> Result<Cuda, Box<dyn Error>> {
    let gpu = Cuda::new(0)?;
    Ok(if mixed {
        gpu.with_mixed_precision()
    } else {
        gpu
    })
}

/// Mémoire GPU utilisée, en Mio, d'après `nvidia-smi`. `None` si la
/// commande n'est pas disponible. Comme la réserve de tampons ne rend rien
/// au pilote, c'est à peu près le maximum atteint depuis le début.
pub fn gpu_memory_mib() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}
