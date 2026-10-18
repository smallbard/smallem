//! DPO : apprendre d'une préférence entre deux réponses plutôt que d'une
//! réponse modèle.
//!
//! Usage : `cargo oxide run --bin dpo -- [options]`
//!
//! ```text
//! --model FICHIER    modèle de départ et de référence (défaut : ../data/post/sft.safetensors)
//! --out FICHIER      modèle obtenu (défaut : ../data/post/dpo.safetensors)
//! --data DOSSIER     tokens et tokeniseur (défaut : ../data/bpe)
//! --prompts N        consignes d'entraînement par famille (défaut : 1500)
//! --samples N        réponses tirées par consigne pour former les paires (défaut : 8)
//! --beta X           (défaut : 0.1)
//! --steps N          pas (défaut : 600)
//! --pairs N          paires par lot (défaut : 32)
//! --lr X             (défaut : 2e-5)
//! --eval-every N     (défaut : 100)
//! --nll X            ajoute X fois la log-vraisemblance négative moyenne des réponses
//!                    préférées, par token (défaut : 0)
//! ```
//!
//! Les paires sont faites par le modèle lui-même : pour chaque consigne, il
//! tire plusieurs réponses, le vérificateur les juge, et l'on garde une
//! réponse juste et une fausse quand il y a les deux.

use std::error::Error;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::chat::{Encoded, Roles};
use smallem::cli::Args;
use smallem::data::{Shards, bytes_per_token};
use smallem::io::{SafeTensors, load_parameters, save_model};
use smallem::model::{Gpt, GptConfig};
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind, cosine_schedule};
use smallem::post_training::{
    Asking, SftBatch, accuracy, answer_all, dpo_loss, sequence_log_probs, sft_batch,
};
use smallem::sampling::Sampling;
use smallem::tasks::Family;
use smallem::tokenizer::Bpe;
use smallem::training::{train_step_clipped, validation_loss};
use smallem_cuda::Cuda;

/// Une paire : la consigne, la réponse juste, la fausse.
#[derive(Clone)]
struct Pair {
    prompt: Vec<u32>,
    preferred: Vec<u32>,
    rejected: Vec<u32>,
}

fn load(gpu: &Cuda, path: &PathBuf) -> Result<Gpt<Cuda>, Box<dyn Error>> {
    let file = SafeTensors::read(path)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let mut model = Gpt::new(gpu, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, gpu, &mut model)?;
    Ok(model)
}

/// Les paires d'une famille : `count` consignes tirées avec la graine
/// `seed`, `samples` réponses chacune.
#[allow(clippy::too_many_arguments)]
fn pairs(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    tokenizer: &Bpe,
    roles: Roles,
    count: usize,
    samples: usize,
    seed: u64,
    rng: &mut StdRng,
) -> Vec<Pair> {
    let asking = Asking::Chat(roles);
    let sampling = Sampling::default();
    let mut pairs = Vec::new();
    for (index, family) in Family::ALL.iter().enumerate() {
        let mut questions = StdRng::seed_from_u64(seed * 1_000_000 + index as u64);
        let turns: Vec<_> = (0..count)
            .map(|_| family.sample(&mut questions, false))
            .collect();
        for chunk in turns.chunks(256) {
            let answers = answer_all(
                gpu, model, tokenizer, &asking, chunk, samples, 24, &sampling, rng,
            );
            for (turn, answers) in chunk.iter().zip(answers) {
                let right: Vec<_> = answers
                    .iter()
                    .filter(|(text, _)| Family::check(turn, text))
                    .collect();
                let wrong: Vec<_> = answers
                    .iter()
                    .filter(|(text, _)| !Family::check(turn, text))
                    .collect();
                if let (Some(right), Some(wrong)) = (right.first(), wrong.first()) {
                    pairs.push(Pair {
                        prompt: asking.prompt(tokenizer, &turn.question),
                        preferred: right.1.clone(),
                        rejected: wrong.1.clone(),
                    });
                }
            }
        }
    }
    pairs
}

/// Les réponses préférées puis les rejetées des paires, en lignes
/// remplies : ce que lit [`sequence_log_probs`].
fn rows(pairs: &[Pair]) -> (SftBatch, usize) {
    let encoded: Vec<Encoded> = [true, false]
        .iter()
        .flat_map(|&preferred| {
            pairs.iter().map(move |pair| {
                let answer = if preferred {
                    &pair.preferred
                } else {
                    &pair.rejected
                };
                let mut tokens = pair.prompt.clone();
                tokens.extend(answer);
                let mut assistant = vec![false; pair.prompt.len()];
                assistant.extend(std::iter::repeat_n(true, answer.len()));
                Encoded { tokens, assistant }
            })
        })
        .collect();
    let length = encoded
        .iter()
        .map(|e| e.tokens.len())
        .max()
        .unwrap_or(1)
        .next_multiple_of(8);
    let mut items = encoded.into_iter();
    let count = 2 * pairs.len();
    (
        sft_batch(|| items.next().unwrap(), count, length, false, 0),
        count,
    )
}

/// Les log-probabilités des réponses des paires sous `model`, sans
/// gradient.
fn log_probs(gpu: &Cuda, model: &Gpt<Cuda>, pairs: &[Pair]) -> Vec<f64> {
    pairs
        .chunks(64)
        .flat_map(|chunk| {
            let (batch, count) = rows(chunk);
            let tape = Tape::new(gpu);
            let lp = sequence_log_probs(
                &tape,
                model,
                &batch.tokens,
                &batch.targets,
                &batch.scored,
                count,
            );
            let values = gpu.to_vec(&tape.value(lp));
            // `[préférées, rejetées]` du morceau.
            values.iter().map(|v| v.to_f64()).collect::<Vec<_>>()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_path: PathBuf = "../data/post/sft.safetensors".into();
    let mut out: PathBuf = "../data/post/dpo.safetensors".into();
    let mut data: PathBuf = "../data/bpe".into();
    let (mut prompts, mut samples, mut steps, mut batch_pairs, mut eval_every) =
        (1500usize, 8usize, 600usize, 32usize, 100usize);
    let (mut beta, mut lr, mut nll) = (0.1f64, 2e-5f64, 0.0f64);
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = args.value(&flag)?,
            "--out" => out = args.value(&flag)?,
            "--data" => data = args.value(&flag)?,
            "--prompts" => prompts = args.value(&flag)?,
            "--samples" => samples = args.value(&flag)?,
            "--beta" => beta = args.value(&flag)?,
            "--steps" => steps = args.value(&flag)?,
            "--pairs" => batch_pairs = args.value(&flag)?,
            "--lr" => lr = args.value(&flag)?,
            "--eval-every" => eval_every = args.value(&flag)?,
            "--nll" => nll = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = load(&gpu, &model_path)?;
    let reference = load(&gpu, &model_path)?;
    let tokenizer = Bpe::load(data.join("tokenizer.bpe"))?;
    let roles = Roles::after(tokenizer.vocab_size());
    let validation = Shards::open_dir(&data, "val.bin")?;
    let bytes = bytes_per_token(&validation, &tokenizer)?;
    let mut rng = StdRng::seed_from_u64(4);

    let mut train = pairs(
        &gpu, &model, &tokenizer, roles, prompts, samples, 5, &mut rng,
    );
    let held_out = pairs(&gpu, &model, &tokenizer, roles, 100, samples, 6, &mut rng);
    println!(
        "{} paires d'entraînement, {} de validation, β = {beta}",
        train.len(),
        held_out.len()
    );
    // Les log-probabilités de référence, une fois pour toutes, rangées
    // comme les paires : `[préférées, rejetées]` par morceaux de 64.
    let reference_of = |pairs: &[Pair]| -> Vec<(f64, f64)> {
        let values = log_probs(&gpu, &reference, pairs);
        let mut out = Vec::new();
        let mut offset = 0;
        for chunk in pairs.chunks(64) {
            let n = chunk.len();
            for i in 0..n {
                out.push((values[offset + i], values[offset + n + i]));
            }
            offset += 2 * n;
        }
        out
    };
    let held_out_reference = reference_of(&held_out);

    let report = |step: usize, model: &Gpt<Cuda>, rng: &mut StdRng| -> Result<(), Box<dyn Error>> {
        let current = log_probs(&gpu, model, &held_out);
        let (mut implicit, mut raw, mut drift) = (0, 0, 0.0);
        let mut offset = 0;
        let mut index = 0;
        for chunk in held_out.chunks(64) {
            let n = chunk.len();
            for i in 0..n {
                let (preferred, rejected) = (current[offset + i], current[offset + n + i]);
                let (ref_preferred, ref_rejected) = held_out_reference[index];
                implicit += usize::from(preferred - ref_preferred > rejected - ref_rejected);
                raw += usize::from(preferred > rejected);
                drift += preferred - ref_preferred;
                index += 1;
            }
            offset += 2 * n;
        }
        let total = held_out.len() as f64;
        let mut line = format!(
            "pas {step:>4}  préférence {:.1} % (brute {:.1} %)  écart à la référence {:+.2}",
            100.0 * implicit as f64 / total,
            100.0 * raw as f64 / total,
            drift / total,
        );
        for family in Family::ALL {
            let turns = family.evaluation_set(200);
            let right = accuracy(&gpu, model, &tokenizer, &Asking::Chat(roles), &turns, rng);
            line += &format!("  {} {:.1} %", family.name(), 100.0 * right);
        }
        let fineweb = validation_loss(&gpu, model, &validation, 50, 16)? / bytes;
        gpu.release_memory();
        println!("{line}  FineWeb {fineweb:.4}");
        Ok(())
    };

    report(0, &model, &mut rng)?;
    let mut optimizer = AnyOptimizer::<Cuda>::new(OptimizerKind::AdamW, lr, 0.0);
    let mut cursor = train.len();
    let mut train_reference = Vec::new();
    for step in 0..steps {
        if cursor + batch_pairs > train.len() {
            train.shuffle(&mut rng);
            train_reference = reference_of(&train);
            cursor = 0;
        }
        let chunk = &train[cursor..cursor + batch_pairs];
        let mut reference_values: Vec<f64> = train_reference[cursor..cursor + batch_pairs]
            .iter()
            .map(|&(preferred, _)| preferred)
            .collect();
        reference_values.extend(
            train_reference[cursor..cursor + batch_pairs]
                .iter()
                .map(|&(_, rejected)| rejected),
        );
        cursor += batch_pairs;
        // Les tailles des lots changent à chaque pas : la réserve de
        // tampons garderait un tampon de chaque taille.
        if step % 10 == 0 {
            gpu.release_memory();
        }
        optimizer.set_learning_rate(cosine_schedule(step, 20, steps, lr, 0.1 * lr));
        let (batch, count) = rows(chunk);
        // Le nombre de tokens des réponses préférées, pour le terme de
        // vraisemblance.
        let preferred_tokens: usize = chunk.iter().map(|pair| pair.preferred.len()).sum();
        let (loss, _) = train_step_clipped(&gpu, &mut model, &mut optimizer, 1.0, |m, t| {
            let lp = sequence_log_probs(t, m, &batch.tokens, &batch.targets, &batch.scored, count);
            let loss = dpo_loss(t, lp, &reference_values, beta);
            if nll == 0.0 {
                return loss;
            }
            // `− λ × Σ ln π(y⁺) / tokens` : garder les réponses préférées
            // probables, et pas seulement plus probables que les rejetées.
            let preferred = t.gather_rows(t.reshape(lp, &[2, batch_pairs]), &[0]);
            let total = t.sum(t.reshape(preferred, &[batch_pairs]), 0, false);
            let factor = -nll / preferred_tokens as f64;
            t.add(loss, t.mul(total, t.scalar(Float::from_f64(factor))))
        });
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            println!("pas {:>4}  perte {:.4}", step + 1, gpu.item(&loss).to_f64());
            report(step + 1, &model, &mut rng)?;
        }
    }
    let _ = rng.random::<u8>();
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_model(&out, &gpu, &model, model.config().to_metadata())?;
    println!("modèle enregistré dans {}", out.display());
    Ok(())
}
