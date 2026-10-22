//! Entraîne le modèle de langage à contexte fixe sur FineWeb-Edu.
//!
//! Usage : `cargo oxide run --bin mlp -- [options]`
//!
//! ```text
//! --cpu              sur le backend de référence du processeur, sans la carte
//! --steps N          pas (défaut : 20 000)
//! --data DOSSIER     où trouver train.bin et val.bin (défaut : ../data)
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::cpu::Cpu;
use smallem::backend::{Backend, Float};
use smallem::cli::{Args, at_least_one};
use smallem::data::Dataset;
use smallem::mlp::{Mlp, MlpConfig, next_token_examples};
use smallem::nn::Module;
use smallem::optim::AdamW;
use smallem::sampling::sample;
use smallem::tokenizer::{self, VOCAB_SIZE};
use smallem::training::{evaluate, train_step};
use smallem_cuda::Cuda;

const SEED: u64 = 42;

const CONFIG: MlpConfig = MlpConfig {
    vocab: VOCAB_SIZE,
    context: 16,
    embed: 32,
    hidden: 512,
};

const BATCH_SIZE: usize = 256;
const DEFAULT_STEPS: usize = 20_000;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 0.1;

/// Lire la perte force le GPU à finir son travail : on ne le fait pas à
/// chaque pas.
const LOG_EVERY: usize = 500;
const EVAL_EVERY: usize = 2_000;
/// Nombre de lots de validation moyennés à chaque évaluation.
const EVAL_BATCHES: usize = 50;

const GENERATED_TOKENS: usize = 400;

/// Perte moyenne sur des lots de validation. La graine est fixe : chaque
/// évaluation voit les mêmes lots, donc les valeurs sont comparables.
fn validation_loss<B: Backend>(backend: &B, model: &Mlp<B>, val: &Dataset) -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let mut total = 0.0;
    for _ in 0..EVAL_BATCHES {
        let batch = val.sample_batch(&mut rng, BATCH_SIZE, CONFIG.context);
        let (contexts, targets) = next_token_examples(&batch);
        let loss = evaluate(backend, model, |m, t| m.loss(t, &contexts, &targets));
        total += loss.to_f64();
    }
    total / EVAL_BATCHES as f64
}

/// Prolonge `prompt` token par token : le modèle donne une distribution sur
/// le token suivant, on en tire un, on l'ajoute, on recommence.
fn generate<B: Backend>(
    backend: &B,
    model: &Mlp<B>,
    prompt: &[u16],
    length: usize,
    rng: &mut StdRng,
) -> Vec<u16> {
    let mut tokens = prompt.to_vec();
    for _ in 0..length {
        let context: Vec<u32> = tokens[tokens.len() - CONFIG.context..]
            .iter()
            .map(|&t| u32::from(t))
            .collect();
        let tape = Tape::new(backend);
        let probabilities = tape.softmax(model.logits(&tape, &context));
        let probabilities: Vec<f64> = backend
            .to_vec(&tape.value(probabilities))
            .iter()
            .map(|p| p.to_f64())
            .collect();
        tokens.push(sample(rng, &probabilities) as u16);
    }
    tokens
}

fn run<B: Backend>(
    backend: &B,
    steps: usize,
    train: &Dataset,
    val: &Dataset,
) -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut model = Mlp::new(backend, CONFIG, &mut rng);
    let mut optimizer = AdamW::new(LEARNING_RATE, WEIGHT_DECAY);
    println!("{} paramètres", model.parameter_count(backend));
    println!(
        "perte de validation initiale : {:.4} (uniforme : {:.4})",
        validation_loss(backend, &model, val),
        (VOCAB_SIZE as f64).ln(),
    );

    let mut last_log = Instant::now();
    for step in 1..=steps {
        let batch = train.sample_batch(&mut rng, BATCH_SIZE, CONFIG.context);
        let (contexts, targets) = next_token_examples(&batch);
        let loss = train_step(backend, &mut model, &mut optimizer, |m, t| {
            m.loss(t, &contexts, &targets)
        });

        if step % LOG_EVERY == 0 {
            // `item` attend que le GPU ait tout fini : le temps mesuré
            // couvre donc bien le calcul des `LOG_EVERY` derniers pas.
            let loss = backend.item(&loss).to_f64();
            let elapsed = last_log.elapsed().as_secs_f64();
            println!(
                "pas {step:>6}  perte {loss:.4}  {:>7.0} exemples/s",
                (LOG_EVERY * BATCH_SIZE) as f64 / elapsed,
            );
            last_log = Instant::now();
        }
        if step % EVAL_EVERY == 0 || step == steps {
            println!(
                "pas {step:>6}  perte de validation {:.4}",
                validation_loss(backend, &model, val),
            );
            last_log = Instant::now();
        }
    }

    // Le début du premier document de validation sert d'amorce.
    let prompt = &val.tokens()[..CONFIG.context];
    let generated = generate(backend, &model, prompt, GENERATED_TOKENS, &mut rng);
    println!("\n--- texte généré ---\n{}", tokenizer::decode(&generated));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (mut cpu, mut steps, mut data_dir) = (false, DEFAULT_STEPS, PathBuf::from("../data"));
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--cpu" => cpu = true,
            "--steps" => steps = args.value(&flag)?,
            "--data" => data_dir = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    at_least_one(&[("--steps", steps)])?;

    let train = Dataset::load(data_dir.join("train.bin"))?;
    let val = Dataset::load(data_dir.join("val.bin"))?;

    if cpu {
        run(&Cpu::<f32>::new(), steps, &train, &val)
    } else {
        run(&Cuda::new(0)?, steps, &train, &val)
    }
}
