//! Mesure le débit des produits matriciels d'un pas d'entraînement de la
//! configuration `xl`, kernel par kernel, en milliards d'opérations
//! flottantes par seconde (une multiplication et une addition par terme).
//!
//! Avec `generation`, mesure plutôt la durée des produits de la génération
//! par lots, selon le nombre de lignes, avec le kernel des matrices
//! étroites et avec les kernels en tuiles.
//!
//! Usage : `cargo oxide run --bin matmul -- [generation]`

use std::error::Error;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::backend::Primitives;
use smallem_cuda::Cuda;

/// Les formes rencontrées avec un lot de 16 séquences de 512 tokens,
/// `C = 512`, 8 têtes et un vocabulaire de 8 192.
const SHAPES: &[(&str, &[usize], &[usize])] = &[
    ("linéaire C → C", &[8192, 512], &[512, 512]),
    ("linéaire C → 4C", &[8192, 512], &[512, 2048]),
    ("linéaire 4C → C", &[8192, 2048], &[2048, 512]),
    ("logits C → V", &[8192, 512], &[512, 8192]),
    ("gradient de poids", &[512, 8192], &[8192, 2048]),
    ("scores q · kᵀ", &[128, 512, 64], &[128, 64, 512]),
    ("attention p · v", &[128, 512, 512], &[128, 512, 64]),
];

const REPEATS: usize = 20;

/// Durée moyenne d'un produit, en microsecondes.
fn micros(gpu: &Cuda, shape_a: &[usize], shape_b: &[usize]) -> f64 {
    let mut rng = StdRng::seed_from_u64(0);
    let a = gpu.randn(shape_a, &mut rng);
    let b = gpu.randn(shape_b, &mut rng);
    drop(gpu.matmul(&a, &b));
    gpu.synchronize();
    let start = Instant::now();
    for _ in 0..REPEATS * 10 {
        gpu.matmul(&a, &b);
    }
    gpu.synchronize();
    start.elapsed().as_secs_f64() * 1e6 / (REPEATS * 10) as f64
}

/// Les produits d'un token par séquence, pour un lot de 1 à 256 séquences.
fn generation() -> Result<(), Box<dyn Error>> {
    let narrow = smallem_cuda::gpu(false)?.with_narrow_rows(usize::MAX);
    let tiled = smallem_cuda::gpu(false)?.with_narrow_rows(0);
    let tiled_bf16 = smallem_cuda::gpu(true)?.with_narrow_rows(0);
    for (name, inner, cols) in [
        ("C → C", 512, 512),
        ("C → 8C/3", 512, 1368),
        ("8C/3 → C", 1368, 512),
    ] {
        println!("{name} : µs par produit");
        println!(
            "{:>6} {:>10} {:>10} {:>10}",
            "lignes", "étroit", "tuiles", "bf16"
        );
        for rows in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
            let (a, b) = ([rows, inner], [inner, cols]);
            println!(
                "{rows:>6} {:>10.1} {:>10.1} {:>10.1}",
                micros(&narrow, &a, &b),
                micros(&tiled, &a, &b),
                micros(&tiled_bf16, &a, &b),
            );
        }
    }
    Ok(())
}

fn gflops(gpu: &Cuda, shape_a: &[usize], shape_b: &[usize]) -> f64 {
    let mut rng = StdRng::seed_from_u64(0);
    let a = gpu.randn(shape_a, &mut rng);
    let b = gpu.randn(shape_b, &mut rng);
    // Premier appel hors chronomètre : il alloue le tampon de sortie.
    drop(gpu.matmul(&a, &b));
    gpu.synchronize();
    let start = Instant::now();
    for _ in 0..REPEATS {
        gpu.matmul(&a, &b);
    }
    gpu.synchronize();
    let seconds = start.elapsed().as_secs_f64() / REPEATS as f64;

    let batch: usize = shape_a[..shape_a.len() - 2].iter().product();
    let (rows, inner) = (shape_a[shape_a.len() - 2], shape_a[shape_a.len() - 1]);
    let cols = shape_b[shape_b.len() - 1];
    2.0 * (batch * rows * inner * cols) as f64 / seconds / 1e9
}

/// Les produits de la génération sur poids quantifiés, contre `f32`.
fn quantized() -> Result<(), Box<dyn Error>> {
    use smallem::quantize::{Format, Quantized};
    let gpu = smallem_cuda::gpu(false)?;
    let mut rng = StdRng::seed_from_u64(0);
    for (inner, cols) in [(512, 512), (512, 1368), (1368, 512)] {
        println!("[{inner}, {cols}] : µs par produit");
        println!(
            "{:>6} {:>10} {:>10} {:>10}",
            "lignes", "f32", "int8", "int4/64"
        );
        let weight = gpu.randn(&[inner, cols], &mut rng);
        let values: Vec<f64> = gpu.to_vec(&weight).iter().map(|&v| f64::from(v)).collect();
        let packed: Vec<_> = [Format::Int8, Format::INT4]
            .iter()
            .map(|&format| {
                let q = Quantized::round_to_nearest(&values, inner, cols, format);
                let (_, groups, per_group) = Quantized::layout(format, inner);
                (
                    format,
                    gpu.from_vec(q.words.clone(), &[groups * per_group, cols]),
                    gpu.from_vec(q.scales.clone(), &[groups, cols]),
                )
            })
            .collect();
        for rows in [1, 8, 16, 64, 256] {
            let x = gpu.randn(&[rows, inner], &mut rng);
            let time = |run: &dyn Fn()| {
                run();
                gpu.synchronize();
                let start = Instant::now();
                for _ in 0..REPEATS * 10 {
                    run();
                }
                gpu.synchronize();
                start.elapsed().as_secs_f64() * 1e6 / (REPEATS * 10) as f64
            };
            let f32_time = time(&|| drop(gpu.matmul(&x, &weight)));
            let times: Vec<f64> = packed
                .iter()
                .map(|(format, words, scales)| {
                    time(&|| drop(gpu.quantized_matmul(&x, words, scales, &weight, None, *format)))
                })
                .collect();
            println!(
                "{rows:>6} {f32_time:>10.1} {:>10.1} {:>10.1}",
                times[0], times[1]
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::args().any(|arg| arg == "generation") {
        return generation();
    }
    if std::env::args().any(|arg| arg == "quantized") {
        return quantized();
    }
    let exact = smallem_cuda::gpu(false)?;
    let mixed = smallem_cuda::gpu(true)?;
    println!("{:<20} {:>12} {:>12}", "produit", "f32", "bf16");
    for &(name, shape_a, shape_b) in SHAPES {
        println!(
            "{name:<20} {:>6.0} GFLOP/s {:>6.0} GFLOP/s",
            gflops(&exact, shape_a, shape_b),
            gflops(&mixed, shape_a, shape_b),
        );
    }
    Ok(())
}
