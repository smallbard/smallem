//! Ce que vaut le processeur, et ce qu'en tire le backend `Fast`.
//!
//! Usage : `cargo run --release --bin cpubench -- <mesure> [options]`
//!
//! ```text
//! peak                      débit maximal de FMA, un cœur puis tous, AVX2 et AVX-512
//! memory [fils]             copie et triad à la STREAM, selon la taille des tableaux
//! pool                      coût d'un appel vide au groupe de fils, et seuil de partage
//! kernel                    le micro-kernel seul, données dans L1
//! matmul [reference|mixed]  nos produits, forme par forme ; en bf16 avec `mixed`
//! blocks [jeu] [kc,…] [mc,…]  balayage des tailles de blocs du produit
//! threads                   le produit et un pas selon le nombre de fils
//! rows                      la frontière du produit sans emballage
//! ladder                    le produit 1024³, de la triple boucle au micro-kernel
//! step <config> <lot> [reference|fast] [fils] [gpt2|modern] [adamw|muon] [mixed]
//!                           un pas d'entraînement, et la répartition de son temps
//! mfu [small,large,xl] [mixed]
//!                           débit et rendement, comme le tableau des lois d'échelle
//! generate [modèle] [f32|int8|int4/G][+out][+vnni] [lots]
//!                           la génération : un token seul, son profil, puis par lots
//! quantized [lignes,…]      le produit par une matrice quantifiée, dans les caches
//! levels                    le même, sur des matrices de plus en plus nombreuses
//! quality [formats] [séquences]
//!                           la perte de validation, token par token, comme en génération
//! ```
//!
//! Les mesures se font machine libre : un entraînement sur la carte prend
//! lui aussi du processeur et de la bande passante.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smallem::backend::cpu::Cpu;
use smallem::backend::fast::gemm::{self, Blocks, Matrix, Split};
use smallem::backend::fast::pool::Pool;
use smallem::backend::fast::{Fast, Isa};
use smallem::backend::profile::Profiled;
use smallem::backend::{Backend, Primitives};
use smallem::model::{Gpt, GptConfig};
use smallem::nn::Architecture;
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind};
use smallem::training::train_step_clipped;

unsafe extern "C" {
    fn mallopt(param: i32, value: i32) -> i32;
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var("KEEP").is_ok() {
        // Expérience : M_TRIM_THRESHOLD = -1, M_MMAP_THRESHOLD = -3.
        unsafe {
            mallopt(-1, i32::MAX);
            mallopt(-3, 1 << 30);
        }
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = &args[1.min(args.len())..];
    let wanted = args.first().map(String::as_str);
    match MEASURES.iter().find(|(name, _)| Some(*name) == wanted) {
        Some((_, measure)) => measure(rest),
        None => {
            let names: Vec<&str> = MEASURES.iter().map(|(name, _)| *name).collect();
            Err(format!("mesure inconnue : {}", names.join(", ")).into())
        }
    }
}

type Measure = fn(&[String]) -> Result<(), Box<dyn Error>>;

/// Les mesures, sous le nom qui les lance.
const MEASURES: &[(&str, Measure)] = &[
    ("peak", |_| {
        peak();
        Ok(())
    }),
    ("memory", memory),
    ("pool", |_| {
        pool();
        Ok(())
    }),
    ("kernel", |_| {
        kernel();
        Ok(())
    }),
    ("matmul", matmul),
    ("blocks", blocks),
    ("threads", |_| {
        threads();
        Ok(())
    }),
    ("rows", |_| {
        rows();
        Ok(())
    }),
    ("ladder", |_| ladder()),
    ("step", step),
    ("mfu", mfu),
    ("generate", generate),
    ("quantized", quantized),
    ("levels", |_| levels()),
    ("quality", quality),
];

/// Les mesures écrites pour AVX-512 refusent de tourner sans lui.
fn require_avx512(measure: &str) -> Result<(), Box<dyn Error>> {
    if Isa::Avx512.available() {
        Ok(())
    } else {
        Err(format!("la mesure `{measure}` demande un processeur AVX-512").into())
    }
}

/// Répète `work` jusqu'à `duration` au moins, et rend le temps d'un appel.
fn time(duration: Duration, mut work: impl FnMut()) -> f64 {
    work();
    let mut calls = 0u64;
    let start = Instant::now();
    while start.elapsed() < duration {
        work();
        calls += 1;
    }
    start.elapsed().as_secs_f64() / calls as f64
}

/// La meilleure de plusieurs mesures : le minimum écarte les interruptions.
fn best_of(runs: usize, duration: Duration, mut work: impl FnMut()) -> f64 {
    (0..runs)
        .map(|_| time(duration, &mut work))
        .fold(f64::INFINITY, f64::min)
}

// --- Débit maximal ---

/// Des FMA indépendantes, sur `ACCUMULATORS` registres : assez pour couvrir
/// la latence de 4 cycles de deux unités FMA.
const ACCUMULATORS: usize = 12;
/// Tours de boucle par appel.
const ROUNDS: usize = 1 << 20;

#[cfg(target_arch = "x86_64")]
mod fma {
    use std::arch::x86_64::*;

    use super::{ACCUMULATORS, ROUNDS};

    /// Opérations `f32` faites par un appel de [`avx512`].
    pub const AVX512_FLOPS: f64 = (ROUNDS * ACCUMULATORS * 16 * 2) as f64;
    /// Une instruction `vdpbf16ps` fait 32 produits de paires `bf16`.
    pub const BF16_FLOPS: f64 = (ROUNDS * ACCUMULATORS * 32 * 2) as f64;
    /// Une instruction `vpdpbusd` fait 64 produits d'octets.
    pub const VNNI_OPS: f64 = (ROUNDS * ACCUMULATORS * 64 * 2) as f64;

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn bf16(seed: f32) -> f32 {
        let a = _mm512_set1_ps(0.5 + seed * 1e-9);
        let pairs = _mm512_cvtne2ps_pbh(a, a);
        let mut acc = [_mm512_set1_ps(seed); ACCUMULATORS];
        for _ in 0..ROUNDS {
            for x in acc.iter_mut() {
                *x = _mm512_dpbf16_ps(*x, pairs, pairs);
            }
        }
        let mut total = _mm512_setzero_ps();
        for x in acc {
            total = _mm512_add_ps(total, x);
        }
        _mm512_reduce_add_ps(total)
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    pub unsafe fn vnni(seed: f32) -> f32 {
        let a = _mm512_set1_epi8(3);
        let b = _mm512_set1_epi8(-2);
        let mut acc = [_mm512_set1_epi32(seed as i32); ACCUMULATORS];
        for _ in 0..ROUNDS {
            for x in acc.iter_mut() {
                *x = _mm512_dpbusd_epi32(*x, a, b);
            }
        }
        let mut total = _mm512_setzero_si512();
        for x in acc {
            total = _mm512_add_epi32(total, x);
        }
        _mm512_reduce_add_epi32(total) as f32
    }
    pub const AVX2_FLOPS: f64 = (ROUNDS * ACCUMULATORS * 8 * 2) as f64;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn avx512(seed: f32) -> f32 {
        let factor = _mm512_set1_ps(0.999_999);
        let offset = _mm512_set1_ps(1e-6);
        let mut acc = [_mm512_set1_ps(seed); ACCUMULATORS];
        for (i, a) in acc.iter_mut().enumerate() {
            *a = _mm512_add_ps(*a, _mm512_set1_ps(i as f32));
        }
        for _ in 0..ROUNDS {
            for a in acc.iter_mut() {
                *a = _mm512_fmadd_ps(*a, factor, offset);
            }
        }
        let mut total = _mm512_setzero_ps();
        for a in acc {
            total = _mm512_add_ps(total, a);
        }
        _mm512_reduce_add_ps(total)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn avx2(seed: f32) -> f32 {
        let factor = _mm256_set1_ps(0.999_999);
        let offset = _mm256_set1_ps(1e-6);
        let mut acc = [_mm256_set1_ps(seed); ACCUMULATORS];
        for (i, a) in acc.iter_mut().enumerate() {
            *a = _mm256_add_ps(*a, _mm256_set1_ps(i as f32));
        }
        for _ in 0..ROUNDS {
            for a in acc.iter_mut() {
                *a = _mm256_fmadd_ps(*a, factor, offset);
            }
        }
        let mut total = _mm256_setzero_ps();
        for a in acc {
            total = _mm256_add_ps(total, a);
        }
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), total) };
        lanes.iter().sum()
    }

    /// Une chaîne d'additions entières dépendantes : une par cycle. Le
    /// nombre d'additions par seconde est la fréquence du cœur.
    pub fn cycles(count: u64) -> u64 {
        let mut x: u64 = 0;
        let mut n = count / 8;
        unsafe {
            std::arch::asm!(
                "2:",
                "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
                "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
                "dec {n}",
                "jnz 2b",
                x = inout(reg) x,
                n = inout(reg) n,
                options(nomem, nostack),
            );
        }
        let _ = n;
        x
    }
}

/// La fréquence d'un cœur, en GHz, mesurée par la chaîne d'additions.
fn frequency() -> f64 {
    const COUNT: u64 = 1 << 30;
    let start = Instant::now();
    black_box(fma::cycles(COUNT));
    COUNT as f64 / start.elapsed().as_secs_f64() / 1e9
}

/// La plus haute fréquence de cœur selon le noyau, en GHz.
fn kernel_frequency() -> Option<f64> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let values: Vec<f64> = info
        .lines()
        .filter(|line| line.starts_with("cpu MHz"))
        .filter_map(|line| line.split(':').nth(1)?.trim().parse().ok())
        .collect();
    (!values.is_empty()).then(|| values.iter().fold(0.0, |max: f64, &v| max.max(v)) / 1000.0)
}

/// Lance `work` sur `threads` fils à la fois, et rend le débit total en
/// Gop/s ainsi que la plus haute fréquence de cœur relevée pendant le
/// calcul.
fn on_threads(threads: usize, flops: f64, work: fn(f32) -> f32) -> (f64, Option<f64>) {
    const CALLS: usize = 200;
    let (seconds, sampled) = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let start = Instant::now();
                    let mut total = 0.0;
                    for call in 0..CALLS {
                        total += work((t * CALLS + call) as f32);
                    }
                    black_box(total);
                    start.elapsed().as_secs_f64()
                })
            })
            .collect();
        std::thread::sleep(Duration::from_millis(100));
        let sampled = kernel_frequency();
        let seconds = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0.0, f64::max);
        (seconds, sampled)
    });
    (
        threads as f64 * CALLS as f64 * flops / seconds / 1e9,
        sampled,
    )
}

fn peak() {
    let idle = frequency();
    println!("fréquence d'un cœur, chaîne d'additions : {idle:.2} GHz");
    let physical = smallem::backend::fast::default_threads();
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    type Variant = (&'static str, bool, f64, fn(f32) -> f32);
    let avx512 = Isa::Avx512.available();
    // SAFETY : chaque variante ne tourne que si le processeur a ses
    // instructions (le booléen).
    let variants: [Variant; 4] = [
        (
            "AVX2   ",
            Isa::Avx2.available(),
            fma::AVX2_FLOPS,
            |s| unsafe { fma::avx2(s) },
        ),
        ("AVX-512", avx512, fma::AVX512_FLOPS, |s| unsafe {
            fma::avx512(s)
        }),
        (
            "bf16",
            gemm::bf16_available(),
            fma::BF16_FLOPS,
            |s| unsafe { fma::bf16(s) },
        ),
        (
            "vnni",
            avx512 && is_x86_feature_detected!("avx512vnni"),
            fma::VNNI_OPS,
            |s| unsafe { fma::vnni(s) },
        ),
    ];
    println!(
        "{:<8} {:>5} {:>12} {:>14} {:>14}",
        "jeu", "fils", "Gop/s", "op/cycle/cœur", "GHz relevés"
    );
    for (name, available, flops, work) in variants {
        if !available {
            println!("{name:<8} absent de ce processeur");
            continue;
        }
        for threads in [1, physical, logical] {
            let (gflops, sampled) = on_threads(threads, flops, work);
            let ghz = sampled.unwrap_or(idle);
            let cores = threads.min(physical) as f64;
            println!(
                "{name:<8} {threads:>5} {gflops:>12.1} {:>14.1} {:>14}",
                gflops / cores / ghz,
                sampled.map_or("?".into(), |f| format!("{f:.2}")),
            );
        }
    }
}

// --- Bande passante ---

#[inline(always)]
fn copy_body<const FMA: bool>(source: &[f32], target: &mut [f32]) {
    target.copy_from_slice(source);
}

#[inline(always)]
fn triad_body<const FMA: bool>(a: &mut [f32], b: &[f32], c: &[f32]) {
    for ((x, &y), &z) in a.iter_mut().zip(b).zip(c) {
        *x = if FMA {
            3.0f32.mul_add(z, y)
        } else {
            y + 3.0 * z
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn copy_wide(source: &[f32], target: &mut [f32]) {
    copy_body::<true>(source, target)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn triad_wide(a: &mut [f32], b: &[f32], c: &[f32]) {
    triad_body::<true>(a, b, c)
}

#[inline(always)]
fn read_body<const FMA: bool>(source: &[f32]) -> f32 {
    // Huit registres de seize sommes : assez pour que la latence de
    // l'addition ne limite pas la lecture.
    let mut acc = [0.0f32; 128];
    for chunk in source.as_chunks::<128>().0 {
        for (sum, &value) in acc.iter_mut().zip(chunk) {
            *sum += value;
        }
    }
    acc.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn read_wide(source: &[f32]) -> f32 {
    read_body::<true>(source)
}

/// La lecture seule, la copie et la triad de STREAM, sur des tableaux qui
/// font ensemble `bytes` octets pour chaque mesure, partagés entre les fils.
/// Chaque fil repasse sur ses propres tableaux, assez de fois pour que
/// l'appel au groupe de fils ne compte pas. Rend les débits en Go/s, en
/// comptant comme STREAM : un tableau pour la lecture, deux pour la copie,
/// trois pour la triad (les lignes que l'écriture charge d'abord en cache
/// ne sont pas comptées).
fn stream(pool: &Pool, bytes: usize) -> (f64, f64, f64) {
    let threads = pool.threads();
    let duration = Duration::from_millis(150);
    let measure = |arrays: usize, kernel: &(dyn Fn(&mut [Vec<f32>]) + Sync)| {
        let len = (bytes / 4 / arrays / threads).max(128) / 128 * 128;
        // De l'ordre de 4 Mo traités par fil et par appel.
        let repeats = (1 << 20) / len + 1;
        let mut data: Vec<Vec<Vec<f32>>> = (0..threads)
            .map(|_| (0..arrays).map(|i| vec![1.0 + i as f32; len]).collect())
            .collect();
        let shared = smallem::backend::fast::pool::SharedMut::new(&mut data);
        let seconds = best_of(3, duration, || {
            pool.run(threads, |t| {
                // SAFETY : chaque fil prend ses propres tableaux.
                let mine = unsafe { &mut shared.slice(t, 1)[0] };
                for _ in 0..repeats {
                    kernel(mine);
                    black_box(&mut *mine);
                }
            })
        });
        (arrays * 4 * len * threads * repeats) as f64 / seconds / 1e9
    };
    // SAFETY : `memory` a vérifié que le processeur a AVX-512.
    let read = measure(1, &|a| {
        black_box(unsafe { read_wide(&a[0]) });
    });
    let copy = measure(2, &|a| {
        let (target, source) = a.split_at_mut(1);
        unsafe { copy_wide(&source[0], &mut target[0]) };
    });
    let triad = measure(3, &|a| {
        let (target, sources) = a.split_at_mut(1);
        unsafe { triad_wide(&mut target[0], &sources[0], &sources[1]) };
    });
    (read, copy, triad)
}

fn memory(args: &[String]) -> Result<(), Box<dyn Error>> {
    require_avx512("memory")?;
    let counts: Vec<usize> = match args.first() {
        Some(threads) => vec![
            threads
                .parse()
                .map_err(|_| format!("nombre de fils illisible : {threads}"))?,
        ],
        None => vec![1, smallem::backend::fast::default_threads()],
    };
    for threads in counts {
        let pool = Pool::new(threads);
        println!("{threads} fil(s)");
        println!(
            "{:>12} {:>13} {:>12} {:>12}",
            "taille", "lecture Go/s", "copie Go/s", "triad Go/s"
        );
        let mut bytes = 12 << 10;
        while bytes <= 3 << 29 {
            let (read, copy, triad) = stream(&pool, bytes);
            println!(
                "{:>12} {read:>13.1} {copy:>12.1} {triad:>12.1}",
                human(bytes)
            );
            bytes *= 2;
        }
    }
    Ok(())
}

fn human(bytes: usize) -> String {
    if bytes >= 1 << 30 {
        format!("{} Gio", bytes >> 30)
    } else if bytes >= 1 << 20 {
        format!("{} Mio", bytes >> 20)
    } else {
        format!("{} Kio", bytes >> 10)
    }
}

// --- Le groupe de fils ---

fn pool() {
    println!(
        "{:>5} {:>16} {:>20}",
        "fils", "appel vide (µs)", "appel endormi (µs)"
    );
    for threads in [2, 4, 8, 16] {
        let pool = Pool::new(threads);
        let empty = best_of(5, Duration::from_millis(100), || pool.run(threads, |_| {}));
        // Les fils se sont endormis : le premier appel les réveille.
        let mut asleep = f64::INFINITY;
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(5));
            let start = Instant::now();
            pool.run(threads, |_| {});
            asleep = asleep.min(start.elapsed().as_secs_f64());
        }
        println!("{threads:>5} {:>16.2} {:>20.1}", empty * 1e6, asleep * 1e6);
    }
    // Le seuil : à partir de quelle taille une addition gagne-t-elle à se
    // partager ?
    let fast = Fast::new();
    let single = Fast::with_threads(1);
    println!(
        "\naddition de deux tenseurs : un fil contre {}",
        fast.threads()
    );
    println!(
        "{:>10} {:>12} {:>12}",
        "éléments", "1 fil (µs)", "tous (µs)"
    );
    for exponent in 10..=22 {
        let len = 1usize << exponent;
        let measure = |backend: &Fast| {
            let a = backend.from_vec(vec![1.0; len], &[len]);
            let b = backend.from_vec(vec![2.0; len], &[len]);
            best_of(3, Duration::from_millis(60), || {
                black_box(backend.add(&a, &b));
            })
        };
        println!(
            "{len:>10} {:>12.2} {:>12.2}",
            measure(&single) * 1e6,
            measure(&fast) * 1e6
        );
    }
}

// --- Le micro-kernel ---

fn kernel() {
    use smallem::backend::fast::kernels::{Avx2, Avx512, MicroKernel, Scalar};
    fn measure<K: MicroKernel>(name: &str) {
        // A et B emballés, de quoi tenir dans L1 : kc = 128.
        const KC: usize = 128;
        let a = vec![0.5f32; KC * K::MR];
        let b = vec![0.25f32; KC * K::NR];
        let mut c = vec![0.0f32; K::MR * K::NR];
        let seconds = best_of(5, Duration::from_millis(100), || unsafe {
            for _ in 0..64 {
                K::run(
                    KC,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    K::NR,
                    K::MR,
                    K::NR,
                    true,
                );
            }
            black_box(&mut c);
        });
        let flops = (64 * 2 * KC * K::MR * K::NR) as f64;
        println!(
            "{name:<8} {} × {:<3} {:>8.1} Gop/s, un cœur",
            K::MR,
            K::NR,
            flops / seconds / 1e9
        );
    }
    measure::<Scalar>("scalaire");
    if Isa::Avx2.available() {
        measure::<Avx2>("AVX2");
    }
    if Isa::Avx512.available() {
        measure::<Avx512>("AVX-512");
    }
}

// --- Le produit matriciel ---

/// Nos produits : des carrés, ceux d'un pas `xl` (projections, MLP, sortie,
/// rétropropagation transposée), ceux de la génération et de Muon.
const SHAPES: &[(&str, usize, usize, usize, bool, bool)] = &[
    ("carré 1024", 1024, 1024, 1024, false, false),
    ("carré 2048", 2048, 2048, 2048, false, false),
    ("xl qkv", 8192, 512, 1536, false, false),
    ("xl mlp montée", 8192, 512, 2048, false, false),
    ("xl mlp descente", 8192, 2048, 512, false, false),
    ("xl sortie (· Eᵀ)", 8192, 512, 8192, false, true),
    ("xl dW = xᵀ · g", 512, 8192, 2048, true, false),
    ("xl dx = g · Wᵀ", 8192, 2048, 512, false, true),
    ("small qkv", 2048, 128, 384, false, false),
    ("small mlp descente", 2048, 512, 128, false, false),
    ("small dW = xᵀ · g", 128, 2048, 512, true, false),
    ("génération, 1 ligne", 1, 512, 2048, false, false),
    ("génération, sortie", 1, 512, 8192, false, true),
    ("génération, 8 lignes", 8, 512, 2048, false, false),
    ("Muon X · Xᵀ", 512, 2048, 512, false, true),
    ("Muon A · X", 512, 512, 2048, false, false),
];

fn random(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

/// Débit du produit `m × k × n` en Gop/s.
fn gemm_rate(
    pool: &Pool,
    isa: Isa,
    shape: (usize, usize, usize, bool, bool),
    blocks: Blocks,
) -> f64 {
    let (m, k, n, ta, tb) = shape;
    let data_a = random(m * k, 1);
    let data_b = random(k * n, 2);
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
    let mut c = vec![0.0f32; m * n];
    let seconds = best_of(3, Duration::from_millis(300), || {
        gemm::gemm_with_blocks(pool, isa, a, b, &mut c, false, blocks);
    });
    2.0 * (m * k * n) as f64 / seconds / 1e9
}

fn matmul(args: &[String]) -> Result<(), Box<dyn Error>> {
    let reference = args.iter().any(|arg| arg == "reference");
    if args.iter().any(|arg| arg == "mixed") {
        require_avx512("matmul mixed")?;
        let pool = Pool::new(smallem::backend::fast::default_threads());
        println!(
            "{:<22} {:>22} {:>12} {:>12}",
            "produit", "m × k × n", "f32 Gop/s", "bf16 Gop/s"
        );
        for &(name, m, k, n, ta, tb) in SHAPES {
            let f32_rate = gemm_rate(&pool, Isa::Avx512, (m, k, n, ta, tb), Blocks::ZEN4);
            let data_a = random(m * k, 1);
            let data_b = random(k * n, 2);
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
            let mut c = vec![0.0f32; m * n];
            let seconds = best_of(3, Duration::from_millis(300), || {
                gemm::gemm_mixed(&pool, Isa::Avx512, a, b, &mut c, false);
            });
            println!(
                "{name:<22} {:>22} {f32_rate:>12.1} {:>12.1}",
                format!("{m} × {k} × {n}"),
                2.0 * (m * k * n) as f64 / seconds / 1e9
            );
        }
        return Ok(());
    }
    let pool = Pool::new(smallem::backend::fast::default_threads());
    let single = Pool::new(1);
    let isa = Isa::detect();
    println!(
        "{:<22} {:>22} {:>12} {:>14}{}",
        "produit",
        "m × k × n",
        "1 fil Gop/s",
        format!("{} fils Gop/s", pool.threads()),
        if reference {
            "   référence Gop/s"
        } else {
            ""
        }
    );
    for &(name, m, k, n, ta, tb) in SHAPES {
        let one = gemm_rate(&single, isa, (m, k, n, ta, tb), Blocks::ZEN4);
        let all = gemm_rate(&pool, isa, (m, k, n, ta, tb), Blocks::ZEN4);
        let naive = if reference && m * k * n <= 1 << 31 {
            let cpu = Cpu::<f32>::new();
            let a = cpu.from_vec(random(m * k, 1), &[m, k]);
            let b = cpu.from_vec(random(k * n, 2), &[k, n]);
            let seconds = time(Duration::from_millis(1), || {
                black_box(cpu.matmul(&a, &b));
            });
            format!("{:>18.2}", 2.0 * (m * k * n) as f64 / seconds / 1e9)
        } else {
            String::new()
        };
        println!(
            "{name:<22} {:>22} {one:>12.1} {all:>14.1}{naive}",
            format!("{m} × {k} × {n}")
        );
    }
    Ok(())
}

fn blocks(args: &[String]) -> Result<(), Box<dyn Error>> {
    let pool = Pool::new(smallem::backend::fast::default_threads());
    let isa: Isa = match args.first() {
        Some(name) => name.parse()?,
        None => Isa::detect(),
    };
    if !isa.available() {
        return Err(format!("le processeur n'a pas les instructions {isa:?}").into());
    }
    let list = |index: usize, default: &[usize]| -> Result<Vec<usize>, String> {
        args.get(index).map_or(Ok(default.to_vec()), |text| {
            text.split(',')
                .map(|v| {
                    v.parse()
                        .map_err(|_| format!("taille de bloc illisible : {v}"))
                })
                .collect()
        })
    };
    let kcs = list(1, &[128, 192, 256, 384, 512])?;
    let mcs = list(2, &[112, 196, 280, 392, 560])?;
    let shapes = [
        (1024, 1024, 1024, false, false),
        (8192, 512, 1536, false, false),
        (8192, 2048, 512, false, false),
        (512, 8192, 2048, true, false),
    ];
    println!("{isa:?}");
    println!(
        "{:>5} {:>5} {:>8} {:>8} {:>15} {:>15} {:>15}",
        "kc", "mc", "partage", "1024³", "8192·512·1536", "8192·2048·512", "dW 512·8192·2048"
    );
    for &kc in &kcs {
        for &mc in &mcs {
            for split in [Split::Rows, Split::Columns] {
                let blocks = Blocks {
                    kc,
                    mc,
                    split,
                    ..Blocks::ZEN4
                };
                let rates: Vec<f64> = shapes
                    .iter()
                    .map(|&s| gemm_rate(&pool, isa, s, blocks))
                    .collect();
                println!(
                    "{kc:>5} {mc:>5} {:>8} {:>8.1} {:>15.1} {:>15.1} {:>15.1}",
                    format!("{split:?}"),
                    rates[0],
                    rates[1],
                    rates[2],
                    rates[3]
                );
            }
        }
    }
    Ok(())
}

/// Le produit `1024³`, du plus simple au plus rapide : ce que gagne chaque
/// idée.
fn ladder() -> Result<(), Box<dyn Error>> {
    require_avx512("ladder")?;
    const N: usize = 1024;
    let (a, b) = (random(N * N, 1), random(N * N, 2));
    let rate = |seconds: f64| 2.0 * (N * N * N) as f64 / seconds / 1e9;
    // 1. La triple boucle du backend de référence : i, j, puis k.
    let cpu = Cpu::<f32>::new();
    let (ta, tb) = (
        cpu.from_vec(a.clone(), &[N, N]),
        cpu.from_vec(b.clone(), &[N, N]),
    );
    let naive = time(Duration::from_millis(1), || {
        black_box(cpu.matmul(&ta, &tb));
    });
    println!(
        "{:<52} {:>8.2} Gop/s",
        "triple boucle i, j, k (référence)",
        rate(naive)
    );
    // 2. Les mêmes calculs, boucles i, k, j : la boucle intérieure parcourt
    // une ligne de B et une ligne de C, contiguës, et se vectorise.
    #[inline(always)]
    fn ikj_body<const FMA: bool>(a: &[f32], b: &[f32], c: &mut [f32]) {
        c.fill(0.0);
        for i in 0..N {
            for k in 0..N {
                let x = a[i * N + k];
                let (line, row) = (&b[k * N..][..N], &mut c[i * N..][..N]);
                for (out, &w) in row.iter_mut().zip(line) {
                    *out = if FMA {
                        x.mul_add(w, *out)
                    } else {
                        *out + x * w
                    };
                }
            }
        }
    }
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn ikj(a: &[f32], b: &[f32], c: &mut [f32]) {
        ikj_body::<true>(a, b, c)
    }
    let mut c = vec![0.0f32; N * N];
    // SAFETY : AVX-512 est vérifié en tête de la mesure.
    let reordered = best_of(2, Duration::from_millis(500), || unsafe {
        ikj(&a, &b, &mut c)
    });
    println!(
        "{:<52} {:>8.2} Gop/s",
        "boucles i, k, j, vectorisées (AVX-512)",
        rate(reordered)
    );
    // 3. Le produit par blocs, un fil, puis tous.
    let (ma, mb) = (Matrix::new(&a, N, N), Matrix::new(&b, N, N));
    for (name, isa) in [
        ("blocs, micro-kernel AVX2 6 × 16", Isa::Avx2),
        ("blocs, micro-kernel AVX-512 14 × 32", Isa::Avx512),
    ] {
        for threads in [1, smallem::backend::fast::default_threads()] {
            let pool = Pool::new(threads);
            let seconds = best_of(3, Duration::from_millis(300), || {
                gemm::gemm(&pool, isa, ma, mb, &mut c, false);
            });
            println!(
                "{:<52} {:>8.2} Gop/s",
                format!("{name}, {threads} fil(s)"),
                rate(seconds)
            );
        }
    }
    Ok(())
}

/// La frontière entre le produit sans emballage et le produit par blocs.
fn rows() {
    let pool = Pool::new(smallem::backend::fast::default_threads());
    let isa = Isa::detect();
    println!(
        "{:>6} {:>22} {:>22} {:>22} {:>22}",
        "lignes",
        "sans emballage Gop/s",
        "par blocs Gop/s",
        "· Bᵀ, produits scalaires",
        "· Bᵀ, par blocs"
    );
    for m in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256] {
        let shape = (m, 512, 2048, false, false);
        let direct = gemm_rate(
            &pool,
            isa,
            shape,
            Blocks {
                few_rows: usize::MAX,
                ..Blocks::ZEN4
            },
        );
        let blocked = gemm_rate(
            &pool,
            isa,
            shape,
            Blocks {
                few_rows: 0,
                ..Blocks::ZEN4
            },
        );
        let shape = (m, 512, 2048, false, true);
        let dots = gemm_rate(
            &pool,
            isa,
            shape,
            Blocks {
                few_rows: usize::MAX,
                ..Blocks::ZEN4
            },
        );
        let transposed = gemm_rate(
            &pool,
            isa,
            shape,
            Blocks {
                few_rows: 0,
                ..Blocks::ZEN4
            },
        );
        println!("{m:>6} {direct:>22.1} {blocked:>22.1} {dots:>22.1} {transposed:>22.1}");
    }
}

fn threads() {
    let isa = Isa::detect();
    println!(
        "{:>5} {:>14} {:>16}",
        "fils", "1024³ Gop/s", "pas small (ms)"
    );
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    for threads in [1, 2, 4, 6, 8, 12, 16]
        .into_iter()
        .filter(|&t| t <= logical)
    {
        let pool = Pool::new(threads);
        let rate = gemm_rate(&pool, isa, (1024, 1024, 1024, false, false), Blocks::ZEN4);
        let fast = Fast::with_threads(threads);
        let ms = step_time(
            &fast,
            GptConfig::small(257),
            32,
            Architecture::GPT2,
            OptimizerKind::AdamW,
            10,
        ) * 1e3;
        println!("{threads:>5} {rate:>14.1} {ms:>16.1}");
    }
}

// --- Le rendement ---

/// Le maximum mesuré par `peak` : 8 cœurs, AVX-512, 5,15 GHz.
const PEAK: f64 = 1314e9;

/// Les configurations du tableau de l'article 1 des lois d'échelle, mesurées
/// comme sur la carte : architecture moderne, vocabulaire de 8 192, lots de
/// 16 séquences de 512 tokens, Muon. Le processeur calcule en `f32`.
fn mfu(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mixed = args.iter().any(|arg| arg == "mixed");
    let fast = Fast::new().with_mixed_precision(mixed);
    let names: Vec<&str> = match args.first().filter(|arg| *arg != "mixed") {
        Some(list) => list.split(',').collect(),
        None => vec!["small", "large", "xl"],
    };
    let kind: OptimizerKind = "muon:short".parse()?;
    println!(
        "{:<8} {:>12} {:>14} {:>12} {:>10}",
        "modèle", "paramètres", "FLOP/token", "tokens/s", "rendement"
    );
    for name in names {
        let config = match name {
            "small" => GptConfig::small(8192),
            "large" => GptConfig::large(8192),
            "xl" => GptConfig::xl(8192),
            other => return Err(format!("configuration inconnue : {other}").into()),
        };
        let config = GptConfig {
            context: 512,
            ..config
        };
        let architecture = Architecture::MODERN;
        let modern = config.with_architecture(architecture);
        let per_token = smallem::flops::training_flops(&modern, 16, 512).expect("compté") as f64
            / (16.0 * 512.0);
        let parameters = {
            use smallem::nn::Module;
            Gpt::new(&fast, modern, &mut StdRng::seed_from_u64(0)).parameter_count(&fast)
        };
        let seconds = step_time(&fast, config, 16, architecture, kind, 3);
        let rate = 16.0 * 512.0 / seconds;
        println!(
            "{name:<8} {parameters:>12} {per_token:>14.3e} {rate:>12.0} {:>9.1} %",
            100.0 * rate * per_token / PEAK
        );
    }
    Ok(())
}

// --- Les produits quantifiés ---

/// Le produit d'une ligne par une matrice `512 × 2048` quantifiée, répété :
/// les codes restent dans les caches. Débit en opérations et en octets de
/// codes lus.
fn quantized(args: &[String]) -> Result<(), Box<dyn Error>> {
    use smallem::backend::ops::{QuantizedMatmul, Run};
    use smallem::quantize::{Format, Quantized};
    let (k, n) = (512, 2048);
    let rows: Vec<usize> = args
        .first()
        .map_or("1,4,8,16".to_string(), Clone::clone)
        .split(',')
        .map(|r| r.parse())
        .collect::<Result<_, _>>()?;
    let fast = Fast::new();
    let weight: Vec<f64> = random(k * n, 3).iter().map(|&w| w as f64).collect();
    println!(
        "{:<14} {:>6} {:>12} {:>16}",
        "format", "lignes", "Gop/s", "Go/s de poids"
    );
    for format in [None, Some(Format::Int8), Some(Format::Int4 { group: 64 })] {
        let (inputs, bytes) = match format {
            None => (
                vec![fast.from_vec(weight.iter().map(|&w| w as f32).collect(), &[k, n])],
                4 * k * n,
            ),
            Some(format) => {
                let q = Quantized::round_to_nearest(&weight, k, n, format);
                let words = fast.from_vec(q.words.clone(), &[q.words.len() / n, n]);
                let scales = fast.from_vec(q.scales.clone(), &[q.scales.len() / n, n]);
                let dequantized =
                    fast.from_vec(q.dequantize().iter().map(|&w| w as f32).collect(), &[k, n]);
                let bytes = match format {
                    Format::Int8 => k * n,
                    Format::Int4 { .. } => k * n / 2,
                };
                (vec![words, scales, dequantized], bytes)
            }
        };
        let modes: &[bool] = if format.is_some() {
            &[false, true]
        } else {
            &[false]
        };
        for &vnni in modes {
            let fast = fast.with_int8_activations(vnni);
            for &m in &rows {
                let x = fast.from_vec(random(m * k, 4), &[m, k]);
                let seconds = best_of(3, Duration::from_millis(200), || {
                    let out = match format {
                        None => fast.matmul(&x, &inputs[0]),
                        Some(format) => fast.run(
                            &QuantizedMatmul { format },
                            &[&x, &inputs[0], &inputs[1], &inputs[2]],
                        ),
                    };
                    black_box(out);
                });
                let name = format.map_or("f32".to_string(), |f| f.name())
                    + if vnni { " vnni" } else { "" };
                println!(
                    "{name:<14} {m:>6} {:>12.1} {:>16.1}",
                    2.0 * (m * k * n) as f64 / seconds / 1e9,
                    bytes as f64 / seconds / 1e9
                );
            }
        }
    }
    Ok(())
}

/// Une ligne face à une suite de matrices int8 `512 × 2048` d'un mégaoctet
/// chacune, parcourues l'une après l'autre comme les couches d'un modèle :
/// le débit des codes selon la taille totale. Au-delà du L3, il devrait
/// tomber à celui de la mémoire.
fn levels() -> Result<(), Box<dyn Error>> {
    use smallem::backend::ops::{QuantizedMatmul, Run};
    use smallem::quantize::{Format, Quantized};
    let (k, n) = (512, 2048);
    let fast = Fast::new();
    let format = Format::Int8;
    let weight: Vec<f64> = random(k * n, 3).iter().map(|&w| w as f64).collect();
    let q = Quantized::round_to_nearest(&weight, k, n, format);
    let make = || {
        [
            fast.from_vec(q.words.clone(), &[q.words.len() / n, n]),
            fast.from_vec(q.scales.clone(), &[q.scales.len() / n, n]),
            fast.from_vec(q.dequantize().iter().map(|&w| w as f32).collect(), &[k, n]),
        ]
    };
    let x = fast.from_vec(random(k, 4), &[1, k]);
    println!("{:>10} {:>16}", "total", "Go/s de codes");
    for count in [2, 4, 8, 16, 20, 24, 28, 32, 40, 48, 64, 96] {
        let matrices: Vec<_> = (0..count).map(|_| make()).collect();
        let seconds = best_of(3, Duration::from_millis(300), || {
            for m in &matrices {
                black_box(fast.run(&QuantizedMatmul { format }, &[&x, &m[0], &m[1], &m[2]]));
            }
        });
        println!(
            "{:>10} {:>16.1}",
            format!("{count} Mo"),
            (count * k * n) as f64 / seconds / 1e9
        );
    }
    Ok(())
}

// --- La génération ---

const GREEDY: smallem::sampling::Sampling = smallem::sampling::Sampling {
    temperature: 0.0,
    top_k: None,
    top_p: None,
};

/// Génère `tokens` tokens d'une séquence, puis par lots, avec le modèle
/// enregistré `--model`, quantifié ou non, et rend la répartition du temps
/// d'un token seul.
fn generate(args: &[String]) -> Result<(), Box<dyn Error>> {
    use smallem::generation::{Batch, Generator};
    use smallem::io::{SafeTensors, load_parameters};
    let path = args
        .first()
        .map_or("../data/step11/muon-short-long.safetensors", String::as_str);
    // `int8`, `int4/64`… ; avec `+out`, la table de sortie aussi ; avec
    // `+vnni`, des activations en octets.
    let name = args.get(1).map_or("f32", String::as_str);
    let (name, vnni) = name
        .strip_suffix("+vnni")
        .map_or((name, false), |name| (name, true));
    let (name, output) = name
        .strip_suffix("+out")
        .map_or((name, false), |name| (name, true));
    let format = match name {
        "f32" => None,
        name => Some(name.parse::<smallem::quantize::Format>()?),
    };
    let sizes: Vec<usize> = args
        .get(2)
        .map_or("1,8,32".to_string(), Clone::clone)
        .split(',')
        .map(|size| size.parse())
        .collect::<Result<_, _>>()?;
    let file = SafeTensors::read(path)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let backend = Profiled::new(Fast::new().with_int8_activations(vnni));
    let mut model = Gpt::new(&backend, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, &backend, &mut model)?;
    if let Some(format) = format {
        model.quantize(&backend, format);
        if output {
            model.quantize_output(&backend, format);
        }
    }
    let prompt: Vec<u32> = (0..16)
        .map(|i| (i * 37 + 11) % config.vocab as u32)
        .collect();
    let mut rng = StdRng::seed_from_u64(0);
    const TOKENS: usize = 200;
    for size in sizes {
        if size == 1 {
            let mut generator = Generator::new(&backend, &model, &prompt);
            generator.logits();
            println!(
                "poids rangés pour le processeur : {:.1} Mo",
                backend.inner().native_bytes() as f64 / 1e6
            );
            backend.reset();
            let start = Instant::now();
            generator.extend(TOKENS, &GREEDY, &mut rng);
            let seconds = start.elapsed().as_secs_f64();
            println!(
                "1 séquence : {:.0} tokens/s, {:.2} ms par token",
                TOKENS as f64 / seconds,
                1e3 * seconds / TOKENS as f64
            );
            println!(
                "{:<34} {:>10} {:>12} {:>7}",
                "opération", "appels", "µs/token", "part"
            );
            for (operation, timing) in backend.report().into_iter().take(14) {
                let us = timing.total.as_secs_f64() * 1e6 / TOKENS as f64;
                println!(
                    "{operation:<34} {:>10} {us:>12.1} {:>6.1} %",
                    timing.calls / TOKENS as u64,
                    100.0 * us / (1e6 * seconds / TOKENS as f64),
                );
            }
        } else {
            let mut batch = Batch::new(&backend, &model, size);
            for _ in 0..size {
                batch.add(&prompt);
            }
            batch.step(&GREEDY, &mut rng);
            backend.reset();
            let start = Instant::now();
            let steps = TOKENS / 2;
            for _ in 0..steps {
                batch.step(&GREEDY, &mut rng);
            }
            let seconds = start.elapsed().as_secs_f64();
            println!(
                "{size} séquences : {:.0} tokens/s, {:.2} ms par pas",
                (size * steps) as f64 / seconds,
                1e3 * seconds / steps as f64
            );
            for (operation, timing) in backend.report().into_iter().take(6) {
                let us = timing.total.as_secs_f64() * 1e6 / steps as f64;
                println!("    {operation:<30} {us:>10.1} µs/pas");
            }
        }
    }
    Ok(())
}

/// La perte de validation par le chemin de la génération : chaque token
/// d'une séquence passe à son tour par le cache, comme quand le modèle
/// écrit, et l'on compte `-ln p(token suivant)`. Les produits sont donc ceux
/// d'une ligne : ceux qui lisent les codes, et, avec `+vnni`, des activations
/// en octets. Rend la perte moyenne en nats par token.
fn quality(args: &[String]) -> Result<(), Box<dyn Error>> {
    use smallem::data::Dataset;
    use smallem::generation::Generator;
    use smallem::io::{SafeTensors, load_parameters};
    let path = "../data/step11/muon-short-long.safetensors";
    let data = Dataset::load("../data/bpe/000_00000.val.bin")?;
    let sequences: usize = args.get(1).map_or(Ok(16), |n| n.parse())?;
    const LENGTH: usize = 512;
    let file = SafeTensors::read(path)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let names = args.first().map_or(
        "f32,int8,int8+out,int8+out+vnni,int4/64,int4/64+out,int4/64+out+vnni".to_string(),
        Clone::clone,
    );
    println!("{:<20} {:>14} {:>10}", "format", "nats/token", "écart");
    let mut reference = None;
    for name in names.split(',') {
        let (base, vnni) = name
            .strip_suffix("+vnni")
            .map_or((name, false), |n| (n, true));
        let (base, output) = base
            .strip_suffix("+out")
            .map_or((base, false), |n| (n, true));
        let fast = Fast::new().with_int8_activations(vnni);
        let mut model = Gpt::new(&fast, config, &mut StdRng::seed_from_u64(0));
        load_parameters(&file, &fast, &mut model)?;
        if base != "f32" {
            let format: smallem::quantize::Format = base.parse()?;
            model.quantize(&fast, format);
            if output {
                model.quantize_output(&fast, format);
            }
        }
        let stride = data.len() / (sequences + 1);
        let mut total = 0.0;
        let mut count = 0usize;
        for sequence in 0..sequences {
            let tokens: Vec<u32> = data.tokens()[sequence * stride..][..LENGTH + 1]
                .iter()
                .map(|&t| u32::from(t))
                .collect();
            let mut generator = Generator::new(&fast, &model, &tokens[..1]);
            for &next in &tokens[1..] {
                let logits = generator.logits();
                let max = logits.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                let log_sum = max + logits.iter().map(|&v| (v - max).exp()).sum::<f64>().ln();
                total += log_sum - logits[next as usize];
                count += 1;
                generator.push(next);
            }
        }
        let loss = total / count as f64;
        let gap = reference.map_or(String::new(), |r: f64| format!("{:+.4}", loss - r));
        reference.get_or_insert(loss);
        println!("{name:<20} {loss:>14.4} {gap:>10}");
    }
    Ok(())
}

// --- Un pas d'entraînement ---

/// Le temps moyen d'un pas, en secondes, après deux pas d'échauffement.
fn step_time<B: Backend>(
    backend: &B,
    config: GptConfig,
    batch: usize,
    architecture: Architecture,
    optimizer: OptimizerKind,
    steps: usize,
) -> f64 {
    step_run(
        backend,
        config,
        batch,
        architecture,
        optimizer,
        steps,
        || {},
    )
}

fn step_run<B: Backend>(
    backend: &B,
    config: GptConfig,
    batch: usize,
    architecture: Architecture,
    kind: OptimizerKind,
    steps: usize,
    mut after_warmup: impl FnMut(),
) -> f64 {
    const WARMUP: usize = 2;
    let config = config.with_architecture(architecture);
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(backend, config, &mut rng);
    let mut optimizer = AnyOptimizer::new(kind, 1e-3, 0.1);
    optimizer.set_progress(1.0);
    let mut start = Instant::now();
    for step in 0..WARMUP + steps {
        if step == WARMUP {
            after_warmup();
            start = Instant::now();
        }
        // Le temps de calcul ne dépend pas des tokens : on les tire au hasard.
        let mut tokens = || -> Vec<u32> {
            (0..batch * config.context)
                .map(|_| rng.random_range(0..config.vocab as u32))
                .collect()
        };
        let (inputs, targets) = (tokens(), tokens());
        let (loss, _) = train_step_clipped(backend, &mut model, &mut optimizer, 1.0, |m, t| {
            m.training_loss(t, &inputs, &targets, batch)
        });
        black_box(backend.item(&loss));
    }
    start.elapsed().as_secs_f64() / steps as f64
}

fn step(args: &[String]) -> Result<(), Box<dyn Error>> {
    let name = args.first().map_or("small", String::as_str);
    let config = match name {
        "small" => GptConfig::small(257),
        "large" => GptConfig::large(257),
        "xl" => GptConfig::xl(8192),
        other => return Err(format!("configuration inconnue : {other}").into()),
    };
    let batch: usize = args.get(1).map_or(Ok(16), |arg| arg.parse())?;
    let reference = args.get(2).is_some_and(|arg| arg == "reference");
    let threads: usize = args
        .get(3)
        .map_or(Ok(smallem::backend::fast::default_threads()), |arg| {
            arg.parse()
        })?;
    let architecture = match args.get(4).map(String::as_str) {
        None | Some("gpt2") => Architecture::GPT2,
        Some("modern") => Architecture::MODERN,
        Some(other) => return Err(format!("architecture inconnue : {other}").into()),
    };
    let kind: OptimizerKind = args
        .get(5)
        .map_or(Ok(OptimizerKind::AdamW), |arg| arg.parse())?;
    let mixed = args.iter().any(|arg| arg == "mixed");
    let steps = if reference { 1 } else { 5 };
    let tokens = (batch * config.context) as f64;
    let flops = smallem::flops::training_flops(
        &config.with_architecture(architecture),
        batch,
        config.context,
    );
    let report = |seconds: f64, profile: Vec<(String, smallem::backend::profile::Timing)>| {
        let rate = flops.map_or(String::new(), |f| {
            format!(", {:.1} Gop/s du modèle", f as f64 / seconds / 1e9)
        });
        println!(
            "{name}, lot {batch} × {} : {:.1} ms par pas, {:.0} tokens/s{rate}",
            config.context,
            seconds * 1e3,
            tokens / seconds,
        );
        let measured: f64 = profile
            .iter()
            .map(|(_, t)| t.total.as_secs_f64())
            .sum::<f64>()
            / steps as f64;
        println!(
            "{:<34} {:>8} {:>10} {:>7}",
            "opération", "appels", "ms/pas", "part"
        );
        for (operation, timing) in profile {
            let ms = timing.total.as_secs_f64() * 1e3 / steps as f64;
            println!(
                "{operation:<34} {:>8} {ms:>10.2} {:>6.1} %",
                timing.calls / steps as u64,
                100.0 * ms / (measured * 1e3),
            );
        }
    };
    if reference {
        let cpu = Profiled::new(Cpu::<f32>::new());
        let seconds = step_run(&cpu, config, batch, architecture, kind, steps, || {
            cpu.reset()
        });
        report(seconds, cpu.report());
    } else {
        let fast = Profiled::new(Fast::with_threads(threads).with_mixed_precision(mixed));
        let seconds = step_run(&fast, config, batch, architecture, kind, steps, || {
            fast.reset()
        });
        report(seconds, fast.report());
    }
    Ok(())
}
