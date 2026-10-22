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
//! --warmup N         pas de montée du taux (défaut : 20)
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

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::chat::{Encoded, Roles};
use smallem::cli::{Args, at_least_one, check_positive, check_range};
use smallem::data::{Shards, bytes_per_token};
use smallem::io::{load_gpt, save_model};
use smallem::model::Gpt;
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind, cosine_schedule};
use smallem::post_training::{
    Asking, SftBatch, accuracy, answer_all, dpo_loss, sequence_log_probs, sft_batch,
};
use smallem::sampling::Sampling;
use smallem::tasks::Family;
use smallem::tokenizer::{Bpe, TOKENIZER_FILE};
use smallem::training::{train_step_clipped, validation_loss};
use smallem_cuda::Cuda;

/// Une paire : la consigne, la réponse juste, la fausse.
#[derive(Clone)]
struct Pair {
    prompt: Vec<u32>,
    preferred: Vec<u32>,
    rejected: Vec<u32>,
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

/// Le modèle de départ, celui qu'on obtient et les données, par défaut.
const DEFAULT_MODEL: &str = "../data/post/sft.safetensors";
const DEFAULT_OUT: &str = "../data/post/dpo.safetensors";
const DEFAULT_DATA: &str = "../data/bpe";

struct Options {
    model: PathBuf,
    out: PathBuf,
    data: PathBuf,
    prompts: usize,
    samples: usize,
    steps: usize,
    pairs: usize,
    warmup: usize,
    eval_every: usize,
    beta: f64,
    lr: f64,
    nll: f64,
}

impl Options {
    fn parse_from(mut args: Args) -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            model: DEFAULT_MODEL.into(),
            out: DEFAULT_OUT.into(),
            data: DEFAULT_DATA.into(),
            prompts: 1500,
            samples: 8,
            steps: 600,
            pairs: 32,
            warmup: 20,
            eval_every: 100,
            beta: 0.1,
            lr: 2e-5,
            nll: 0.0,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => options.model = args.value(&flag)?,
                "--out" => options.out = args.value(&flag)?,
                "--data" => options.data = args.value(&flag)?,
                "--prompts" => options.prompts = args.value(&flag)?,
                "--samples" => options.samples = args.value(&flag)?,
                "--beta" => options.beta = args.value(&flag)?,
                "--steps" => options.steps = args.value(&flag)?,
                "--pairs" => options.pairs = args.value(&flag)?,
                "--lr" => options.lr = args.value(&flag)?,
                "--warmup" => options.warmup = args.value(&flag)?,
                "--eval-every" => options.eval_every = args.value(&flag)?,
                "--nll" => options.nll = args.value(&flag)?,
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        at_least_one(&[
            ("--prompts", options.prompts),
            ("--samples", options.samples),
            ("--steps", options.steps),
            ("--pairs", options.pairs),
            ("--eval-every", options.eval_every),
        ])?;
        check_positive("--beta", options.beta)?;
        check_positive("--lr", options.lr)?;
        check_range("--nll", options.nll, 0.0..)?;
        Ok(options)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Options {
        model: model_path,
        out,
        data,
        prompts,
        samples,
        steps,
        pairs: batch_pairs,
        warmup,
        eval_every,
        beta,
        lr,
        nll,
    } = Options::parse_from(Args::from_env())?;
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = load_gpt(&gpu, &model_path, true)?;
    let reference = load_gpt(&gpu, &model_path, true)?;
    let tokenizer = Bpe::load(data.join(TOKENIZER_FILE))?;
    let roles = Roles::after(tokenizer.vocab_size());
    let validation = Shards::open_dir(&data, "val.bin")?;
    let bytes = bytes_per_token(&validation, &tokenizer)?;
    let mut rng = StdRng::seed_from_u64(4);

    let mut train = pairs(
        &gpu, &model, &tokenizer, roles, prompts, samples, 5, &mut rng,
    );
    let held_out = pairs(&gpu, &model, &tokenizer, roles, 100, samples, 6, &mut rng);
    if train.len() < batch_pairs {
        return Err(format!(
            "{} paires d'entraînement, moins que --pairs {batch_pairs}",
            train.len()
        )
        .into());
    }
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
        optimizer.set_learning_rate(cosine_schedule(step, warmup, steps, lr, 0.1 * lr));
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
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_model(&out, &gpu, &model, model.config().to_metadata())?;
    println!("modèle enregistré dans {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(args: &[&str]) -> Result<Options, Box<dyn Error>> {
        Options::parse_from(Args::new(
            args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>(),
        ))
    }

    #[test]
    fn options_are_read_from_the_command_line() {
        let options = options(&["--beta", "0.5", "--warmup", "0", "--nll", "1"]).unwrap();
        assert_eq!((options.beta, options.warmup, options.nll), (0.5, 0, 1.0));
        assert_eq!((options.pairs, options.steps), (32, 600));
    }

    #[test]
    fn options_that_would_break_the_loop_are_refused() {
        for (refused, flag) in [
            (&["--eval-every", "0"][..], "--eval-every"),
            (&["--pairs", "0"], "--pairs"),
            (&["--beta", "0"], "--beta"),
            (&["--nll", "-0.5"], "--nll"),
            (&["--lr", "inf"], "--lr"),
        ] {
            let error = options(refused).err().expect(flag).to_string();
            assert!(error.contains(flag), "{error}");
        }
    }
}
