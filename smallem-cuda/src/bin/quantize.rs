//! Quantifie un modèle, et mesure ce que cela coûte et ce que cela
//! rapporte : perte de validation, mémoire des poids, vitesse de
//! génération.
//!
//! Usage : `cargo oxide run --bin quantize -- [options]`
//!
//! ```text
//! --model FICHIER    modèle (défaut : ../data/step11/muon-short-long.safetensors)
//! --data DOSSIER     tokens et tokeniseur (défaut : ../data/bpe)
//! --formats LISTE    formats, séparés par des virgules (défaut :
//!                    f32,int8,int4/128,int4/64,int4/32,gptq-int4/64) ; le
//!                    préfixe `gptq-` quantifie par GPTQ au lieu d'arrondir
//! --batches N        lots de validation de 16 séquences (défaut : 200, comme l'entraînement)
//! --calibration N    lots d'entraînement de 16 séquences pour GPTQ (défaut : 8)
//! --outliers         analyse les entrées des couches au lieu de quantifier
//! ```
//!
//! Tout est calculé en `f32`. La perte se mesure sur les mêmes lots que la
//! perte finale d'un entraînement.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::backend::Primitives;
use smallem::cli::Args;
use smallem::data::Batches;
use smallem::data::{Shards, bytes_per_token};
use smallem::generation::{Batch, Generator};
use smallem::io::{SafeTensors, gpt_from_file};
use smallem::model::Gpt;
use smallem::nn::Module;
use smallem::quantize::{Format, Quantized};
use smallem::sampling::Sampling;
use smallem::tokenizer::{Bpe, TOKENIZER_FILE};
use smallem::training::{evaluate, validation_loss};
use smallem_cuda::Cuda;

const GREEDY: Sampling = Sampling {
    temperature: 0.0,
    top_k: None,
    top_p: None,
};

fn parse_format(name: &str) -> Result<Option<Format>, Box<dyn Error>> {
    let name = name.strip_prefix("gptq-").unwrap_or(name);
    Ok(match name {
        "f32" => None,
        "int8" => Some(Format::Int8),
        other => match other.strip_prefix("int4/") {
            Some(group) => Some(Format::Int4 {
                group: group.parse()?,
            }),
            None => return Err(format!("format inconnu : {other}").into()),
        },
    })
}

/// Tokens par seconde d'une génération de `tokens` tokens, seule ou par
/// lot de `size` séquences, et les tokens de la première séquence.
fn speed(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    prompt: &[u32],
    size: usize,
    tokens: usize,
) -> (f64, Vec<u32>) {
    let mut rng = StdRng::seed_from_u64(0);
    if size == 1 {
        let mut generator = Generator::new(gpu, model, prompt);
        generator.logits();
        let start = Instant::now();
        generator.extend(tokens, &GREEDY, &mut rng);
        let rate = tokens as f64 / start.elapsed().as_secs_f64();
        return (rate, generator.into_tokens());
    }
    let mut batch = Batch::new(gpu, model, size);
    for _ in 0..size {
        batch.add(prompt);
    }
    batch.step(&GREEDY, &mut rng);
    let start = Instant::now();
    for _ in 1..tokens {
        batch.step(&GREEDY, &mut rng);
    }
    let rate = (size * (tokens - 1)) as f64 / start.elapsed().as_secs_f64();
    (rate, batch.tokens(0).to_vec())
}

/// Le hessien de chaque matrice, avec son nom.
type Hessians = Vec<(String, Vec<f64>)>;

/// Fait passer `batches` lots d'entraînement de 16 séquences dans le
/// modèle, et rend le hessien `Σ xᵀ · x` des entrées de chaque matrice.
fn calibrate(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    train: &Shards,
    batches: usize,
) -> Result<Hessians, Box<dyn Error>> {
    model.start_calibration(gpu, false);
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..batches {
        let batch = train.batch(&mut rng, 16, model.config().context)?;
        let (tokens, targets) = batch.examples();
        evaluate(gpu, model, |m, t| m.loss(t, &tokens, &targets, 16));
    }
    Ok(model
        .take_calibration()
        .into_iter()
        .map(|(name, calibration)| {
            let hessian = calibration.hessian.expect("des entrées observées");
            let values = gpu.to_vec(&hessian).iter().map(|&v| f64::from(v)).collect();
            (name, values)
        })
        .collect())
}

/// Les entrées des couches linéaires, composante par composante : y
/// trouve-t-on, comme dans les grands modèles, quelques composantes bien
/// plus grandes que les autres ?
fn analyze_outliers(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    train: &Shards,
    batches: usize,
) -> Result<(), Box<dyn Error>> {
    model.start_calibration(gpu, true);
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..batches {
        let batch = train.batch(&mut rng, 16, model.config().context)?;
        let (tokens, targets) = batch.examples();
        evaluate(gpu, model, |m, t| m.loss(t, &tokens, &targets, 16));
    }
    println!(
        "{:<40} {:>8} {:>10} {:>10} {:>14}",
        "entrée de", "max |x|", "médiane", "comp. ≥ 6", "valeurs ≥ 6"
    );
    for (name, calibration) in model.take_calibration() {
        let mut largest = calibration.largest.clone();
        largest.sort_by(f64::total_cmp);
        let median = largest[largest.len() / 2];
        let max = *largest.last().expect("des composantes");
        let channels = calibration.beyond_six.iter().filter(|&&n| n > 0).count();
        let values: u64 = calibration.beyond_six.iter().sum();
        // Les composantes les plus grandes, avec la part des tokens où
        // elles dépassent 6.
        let mut order: Vec<usize> = (0..largest.len()).collect();
        order.sort_by(|&a, &b| calibration.largest[b].total_cmp(&calibration.largest[a]));
        let top: Vec<String> = order
            .iter()
            .take(3)
            .map(|&k| {
                format!(
                    "{k}:{:.1} ({:.0} %)",
                    calibration.largest[k],
                    100.0 * calibration.beyond_six[k] as f64 / calibration.rows as f64
                )
            })
            .collect();
        println!(
            "{name:<40} {max:>8.1} {median:>10.2} {channels:>10} {:>13.4}%   {}",
            100.0 * values as f64 / (calibration.rows * largest.len()) as f64,
            top.join("  "),
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_path: PathBuf = "../data/step11/muon-short-long.safetensors".into();
    let mut data: PathBuf = "../data/bpe".into();
    let mut formats = "f32,int8,int4/128,int4/64,int4/32,gptq-int4/64".to_string();
    let mut batches = 200usize;
    let mut calibration = 8usize;
    let mut outliers = false;
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = args.value(&flag)?,
            "--data" => data = args.value(&flag)?,
            "--formats" => formats = args.value(&flag)?,
            "--batches" => batches = args.value(&flag)?,
            "--calibration" => calibration = args.value(&flag)?,
            "--outliers" => outliers = true,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let file = SafeTensors::read(&model_path)?;
    let tokenizer = Bpe::load(data.join(TOKENIZER_FILE))?;
    let validation = Shards::open_dir(&data, "val.bin")?;
    let bytes = bytes_per_token(&validation, &tokenizer)?;
    let prompt: Vec<u32> = tokenizer
        .encode("Photosynthesis is the process by which plants")
        .iter()
        .map(|&t| u32::from(t))
        .collect();
    let gpu = smallem_cuda::gpu(false)?;
    let train = Shards::open_dir(&data, "train.bin")?;
    if outliers {
        let model = gpt_from_file(&gpu, &file, true)?;
        return analyze_outliers(&gpu, &model, &train, 2);
    }

    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>10} {:>12}",
        "format", "nats/oct.", "écart", "poids (Mo)", "B = 1", "B = 64"
    );
    let mut reference: Option<(f64, Vec<u32>)> = None;
    for name in formats.split(',') {
        let format = parse_format(name)?;
        let mut model = gpt_from_file(&gpu, &file, true)?;
        let f32_bytes: usize = model
            .parameters()
            .iter()
            .map(|p| 4 * gpu.shape(p.value()).iter().product::<usize>())
            .sum();
        let mut weight_bytes = f32_bytes;
        if let Some(format) = format {
            let done = if name.starts_with("gptq-") {
                let hessians = calibrate(&gpu, &model, &train, calibration)?;
                model.quantize_with(&gpu, |parameter, weight, rows, cols| {
                    let hessian = &hessians
                        .iter()
                        .find(|(name, _)| name == parameter)
                        .expect("hessien de chaque matrice")
                        .1;
                    Quantized::gptq(weight, rows, cols, format, hessian)
                })
            } else {
                model.quantize(&gpu, format)
            };
            for (_, quantized) in done {
                weight_bytes -= 4 * quantized.rows * quantized.cols;
                weight_bytes += quantized.bytes();
            }
        }
        let loss = validation_loss(&gpu, &model, &validation, batches, 16)? / bytes;
        let (single, tokens) = speed(&gpu, &model, &prompt, 1, 300);
        let (many, _) = speed(&gpu, &model, &prompt, 64, 64);
        let (gap, same) = match &reference {
            Some((base, base_tokens)) => {
                let same = tokens
                    .iter()
                    .zip(base_tokens)
                    .take_while(|(a, b)| a == b)
                    .count();
                (loss - base, format!("{same} tokens communs"))
            }
            None => {
                reference = Some((loss, tokens.clone()));
                (0.0, String::new())
            }
        };
        // Avec le même format, la génération avec et sans cache doit dire
        // la même chose : l'une lit les codes, l'autre les poids
        // déquantifiés.
        let mut uncached = Generator::without_cache(&gpu, &model, &prompt);
        uncached.extend(
            tokens.len() - prompt.len(),
            &GREEDY,
            &mut StdRng::seed_from_u64(0),
        );
        let consistent = uncached.tokens() == tokens.as_slice();
        println!(
            "{name:<10} {loss:>10.4} {gap:>+10.4} {:>12.1} {single:>10.0} {many:>12.0}   {same}{}",
            weight_bytes as f64 / 1e6,
            if consistent {
                ""
            } else {
                "  (cache ≠ recalcul !)"
            },
        );
    }
    Ok(())
}
