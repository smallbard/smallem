//! La mémoire GPU quand les tailles demandées ne se répètent pas.
//!
//! Ce test remplit la carte : il est dans son propre fichier pour ne pas
//! tourner en même temps que les autres.
//!
//! `cargo oxide test -- --test memory`

use smallem::backend::Primitives;
use smallem_cuda::Cuda;

#[test]
fn pool_gives_memory_back_when_the_card_is_full() {
    // 40 tenseurs d'environ 1 Gio, tous de tailles différentes : aucun
    // tampon rendu à la réserve n'est réutilisable par le suivant. Sans
    // vidage de la réserve, la carte (24 Gio) serait pleine à mi-chemin.
    const GIB_OF_F32: usize = 1 << 28;
    let gpu = Cuda::new(0).unwrap();
    for extra in 0..40 {
        let len = GIB_OF_F32 + extra;
        let tensor = gpu.ones(&[len]);
        // Au-delà de 2²⁴, un f32 ne représente plus tous les entiers :
        // la somme se compare avec une tolérance.
        let sum = gpu.item(&gpu.sum_all(&tensor)) as f64;
        assert!((sum - len as f64).abs() < 64.0, "somme {sum} pour {len}");
    }
}
