//! Les instructions du processeur, et le nombre de fils.

use std::sync::OnceLock;

/// Les instructions vectorielles dont se servent les kernels.
///
/// N'importe qui peut nommer une valeur, mais les fonctions qui en prennent
/// une la vérifient ([`Isa::check`]) avant de lancer un kernel qui s'en
/// sert : demander AVX-512 sur un processeur qui ne l'a pas fait paniquer,
/// au lieu d'exécuter une instruction inconnue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AVX-512 (fondation, mots et octets, longueurs variables) et FMA.
    Avx512,
    /// AVX2 et FMA.
    Avx2,
    /// Rien de plus que ce que tout processeur `x86_64` garantit.
    Scalar,
}

impl Isa {
    /// Les meilleures instructions du processeur.
    pub fn detect() -> Self {
        [Isa::Avx512, Isa::Avx2]
            .into_iter()
            .find(|isa| isa.available())
            .unwrap_or(Isa::Scalar)
    }

    /// Le rang des instructions : chacune contient les précédentes.
    fn rank(self) -> u8 {
        match self {
            Isa::Scalar => 0,
            Isa::Avx2 => 1,
            Isa::Avx512 => 2,
        }
    }

    /// Panique si le processeur n'a pas ces instructions. La détection est
    /// faite une fois : le contrôle ne coûte qu'une lecture.
    #[inline]
    pub fn check(self) {
        static BEST: OnceLock<Isa> = OnceLock::new();
        if self.rank() > BEST.get_or_init(Isa::detect).rank() {
            unavailable(self);
        }
    }

    /// Le processeur a-t-il ces instructions ?
    pub fn available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                Isa::Avx2.available()
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512dq")
                    && is_x86_feature_detected!("avx512vl")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

#[cold]
#[inline(never)]
fn unavailable(isa: Isa) -> ! {
    panic!("le processeur n'a pas les instructions {isa:?}")
}

/// Lit le nom des instructions. Le nom seul ne dit pas si le processeur
/// les a : voir [`Isa::available`].
impl std::str::FromStr for Isa {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "avx512" => Ok(Isa::Avx512),
            "avx2" => Ok(Isa::Avx2),
            "scalar" => Ok(Isa::Scalar),
            other => Err(format!(
                "instructions inconnues : {other} (avx512, avx2, scalar)"
            )),
        }
    }
}

/// Le nombre de fils par défaut : un par cœur physique. Le SMT partage les
/// unités de calcul d'un cœur entre deux fils, et ne fait rien gagner au
/// produit matriciel (étape C1 de `PLAN-CPU.md`). `SMALLEM_THREADS` le
/// remplace.
pub fn default_threads() -> usize {
    if let Some(threads) = std::env::var("SMALLEM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        return threads;
    }
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    physical_cores().unwrap_or(logical).clamp(1, logical)
}

/// Le nombre de cœurs physiques, d'après `/proc/cpuinfo`.
fn physical_cores() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut cores = std::collections::BTreeSet::new();
    let mut physical = None;
    for line in info.lines() {
        let mut parts = line.splitn(2, ':').map(str::trim);
        match (parts.next(), parts.next()) {
            (Some("physical id"), Some(id)) => physical = Some(id.to_string()),
            (Some("core id"), Some(id)) => {
                cores.insert((physical.clone(), id.to_string()));
            }
            _ => {}
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_refuses_exactly_the_missing_instructions() {
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512] {
            let accepted = std::panic::catch_unwind(|| isa.check()).is_ok();
            assert_eq!(accepted, isa.available(), "{isa:?}");
        }
    }

    #[test]
    fn a_vectorized_function_checks_its_instructions() {
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512] {
            let accepted =
                std::panic::catch_unwind(|| super::super::math::sum(isa, &[1.0, 2.0])).is_ok();
            assert_eq!(accepted, isa.available(), "{isa:?}");
        }
    }
}
