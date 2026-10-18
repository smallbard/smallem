//! Apprendre par essais et erreurs : le modèle tire plusieurs réponses à
//! chaque consigne, le vérificateur les note, et les meilleures de chaque
//! groupe sont rendues plus probables (GRPO, ou sa forme la plus simple,
//! REINFORCE).
//!
//! Usage : `cargo oxide run --bin rl -- [options]`
//!
//! ```text
//! --model FICHIER    modèle de départ et de référence (défaut : ../data/post/sft.safetensors)
//! --out FICHIER      modèle obtenu (défaut : ../data/post/rl.safetensors)
//! --data DOSSIER     tokens et tokeniseur (défaut : ../data/bpe)
//! --method M         reinforce ou grpo (défaut : grpo)
//! --steps N          pas (défaut : 300)
//! --prompts N        consignes par pas (défaut : 16)
//! --group N          réponses par consigne (défaut : 8)
//! --lr X             (défaut : 2e-6)
//! --clip X           borne du rapport, GRPO (défaut : 0.2)
//! --kl X             poids de la divergence, GRPO (défaut : 0.04)
//! --per-token        moyenne sur tous les tokens au lieu de réponse par réponse
//! --inner N          pas de gradient par lot de réponses tirées (défaut : 1)
//! --eval-every N     (défaut : 50)
//! --task T           long-sort, long-addition ou sort (défaut : long-sort)
//! ```
//!
//! Les consignes sont plus difficiles que celles du fine-tuning supervisé :
//! trier de 6 à 8 nombres au lieu de 3 à 5, ou additionner des nombres de 4
//! chiffres. Une consigne sur quatre est la tâche ordinaire correspondante.

use std::error::Error;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::chat::{Encoded, Roles};
use smallem::cli::Args;
use smallem::data::{Shards, bytes_per_token};
use smallem::io::{SafeTensors, load_parameters, save_model};
use smallem::model::{Gpt, GptConfig};
use smallem::optim::{AnyOptimizer, OptimizerKind};
use smallem::post_training::{
    Asking, GrpoSettings, accuracy, answer_all, group_advantages, grpo_loss, sft_batch,
    token_log_probs,
};
use smallem::sampling::Sampling;
use smallem::tasks::Family;
use smallem::tokenizer::Bpe;
use smallem::training::{train_step_clipped, validation_loss};
use smallem_cuda::Cuda;

fn load(gpu: &Cuda, path: &PathBuf) -> Result<Gpt<Cuda>, Box<dyn Error>> {
    let file = SafeTensors::read(path)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let mut model = Gpt::new(gpu, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, gpu, &mut model)?;
    Ok(model)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_path: PathBuf = "../data/post/sft.safetensors".into();
    let mut out: PathBuf = "../data/post/rl.safetensors".into();
    let mut data: PathBuf = "../data/bpe".into();
    let mut method = "grpo".to_string();
    let mut task = "long-sort".to_string();
    let (mut steps, mut prompts, mut group, mut inner, mut eval_every) =
        (300usize, 16usize, 8usize, 1usize, 50usize);
    let (mut lr, mut clip, mut kl, mut per_token) = (2e-6f64, 0.2f64, 0.04f64, false);
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = args.value(&flag)?,
            "--out" => out = args.value(&flag)?,
            "--data" => data = args.value(&flag)?,
            "--method" => method = args.value(&flag)?,
            "--steps" => steps = args.value(&flag)?,
            "--prompts" => prompts = args.value(&flag)?,
            "--group" => group = args.value(&flag)?,
            "--lr" => lr = args.value(&flag)?,
            "--clip" => clip = args.value(&flag)?,
            "--kl" => kl = args.value(&flag)?,
            "--per-token" => per_token = true,
            "--inner" => inner = args.value(&flag)?,
            "--eval-every" => eval_every = args.value(&flag)?,
            "--task" => task = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let (hard, ordinary) = match task.as_str() {
        "long-sort" => (Family::LongSort, Family::Sort),
        "long-addition" => (Family::LongAddition, Family::Addition),
        // Le tri ordinaire, que le fine-tuning a déjà à moitié appris : un
        // signal abondant, pour voir si l'apprentissage par essais améliore
        // ce qui est déjà en partie su.
        "sort" => (Family::Sort, Family::Sort),
        other => return Err(format!("tâche inconnue : {other}").into()),
    };
    let reinforce = match method.as_str() {
        "reinforce" => true,
        "grpo" => false,
        other => return Err(format!("méthode inconnue : {other}").into()),
    };
    // REINFORCE : ni borne, ni divergence, et la moyenne du groupe comme
    // seule ligne de base.
    let settings = if reinforce {
        GrpoSettings {
            clip: f64::INFINITY,
            kl: 0.0,
            per_answer: false,
        }
    } else {
        GrpoSettings {
            clip,
            kl,
            per_answer: !per_token,
        }
    };
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = load(&gpu, &model_path)?;
    let reference = load(&gpu, &model_path)?;
    let tokenizer = Bpe::load(data.join("tokenizer.bpe"))?;
    let roles = Roles::after(tokenizer.vocab_size());
    let asking = Asking::Chat(roles);
    let validation = Shards::open_dir(&data, "val.bin")?;
    let bytes = bytes_per_token(&validation, &tokenizer)?;
    let mut rng = StdRng::seed_from_u64(8);
    let mut questions = StdRng::seed_from_u64(9);

    let report = |step: usize, model: &Gpt<Cuda>, rng: &mut StdRng| -> Result<(), Box<dyn Error>> {
        let mut line = format!("pas {step:>4}");
        let mut families = vec![hard, ordinary, Family::Lookup];
        if hard == ordinary {
            families = vec![Family::LongSort, Family::Sort, Family::Lookup];
        }
        for family in families {
            let turns = family.evaluation_set(200);
            let right = accuracy(&gpu, model, &tokenizer, &asking, &turns, rng);
            line += &format!("  {} {:.1} %", family.name(), 100.0 * right);
        }
        let fineweb = validation_loss(&gpu, model, &validation, 50, 16)? / bytes;
        gpu.release_memory();
        println!("{line}  FineWeb {fineweb:.4}");
        Ok(())
    };
    report(0, &model, &mut rng)?;

    let mut optimizer = AnyOptimizer::<Cuda>::new(OptimizerKind::AdamW, lr, 0.0);
    let sampling = Sampling::default();
    let (mut reward_sum, mut length_sum, mut kl_sum, mut seen) = (0.0, 0.0, 0.0, 0usize);
    for step in 0..steps {
        // Le lot de consignes, et `group` réponses tirées pour chacune.
        let turns: Vec<_> = (0..prompts)
            .map(|i| {
                let family = if i % 4 == 3 { ordinary } else { hard };
                family.sample(&mut questions, false)
            })
            .collect();
        let answers = answer_all(
            &gpu, &model, &tokenizer, &asking, &turns, group, 32, &sampling, &mut rng,
        );
        let mut encoded = Vec::new();
        let mut advantages = Vec::new();
        for (turn, answers) in turns.iter().zip(&answers) {
            let rewards: Vec<f64> = answers
                .iter()
                .map(|(text, _)| f64::from(u8::from(Family::check(turn, text))))
                .collect();
            reward_sum += rewards.iter().sum::<f64>();
            let group_advantage = if reinforce {
                let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
                rewards.iter().map(|r| r - mean).collect()
            } else {
                group_advantages(&rewards)
            };
            advantages.extend(group_advantage);
            let prompt = asking.prompt(&tokenizer, &turn.question);
            for (_, tokens) in answers {
                length_sum += tokens.len() as f64;
                seen += 1;
                let mut all = prompt.clone();
                all.extend(tokens);
                let mut assistant = vec![false; prompt.len()];
                assistant.extend(std::iter::repeat_n(true, tokens.len()));
                encoded.push(Encoded {
                    tokens: all,
                    assistant,
                });
            }
        }
        let rows = encoded.len();
        let length = encoded
            .iter()
            .map(|e| e.tokens.len())
            .max()
            .unwrap_or(1)
            .next_multiple_of(8);
        let mut items = encoded.into_iter();
        let batch = sft_batch(|| items.next().unwrap(), rows, length, false, 0);
        // Les log-probabilités des réponses sous la référence, et sous le
        // modèle qui les a tirées : sans gradient.
        let values = |model: &Gpt<Cuda>| -> Vec<f64> {
            let tape = Tape::new(&gpu);
            let (lp, _) = token_log_probs(
                &tape,
                model,
                &batch.tokens,
                &batch.targets,
                &batch.scored,
                rows,
            );
            gpu.to_vec(&tape.value(lp))
                .iter()
                .map(|v| v.to_f64())
                .collect()
        };
        let reference_values = values(&reference);
        let old = values(&model);
        kl_sum += old
            .iter()
            .zip(&reference_values)
            .map(|(lp, r)| (r - lp).exp() - (r - lp) - 1.0)
            .sum::<f64>()
            / old.len() as f64;
        // Les tailles des lots changent à chaque pas : la réserve de
        // tampons garderait un tampon de chaque taille.
        if step % 10 == 0 {
            gpu.release_memory();
        }
        for _ in 0..inner {
            train_step_clipped(&gpu, &mut model, &mut optimizer, 1.0, |m, t| {
                let (lp, owners) =
                    token_log_probs(t, m, &batch.tokens, &batch.targets, &batch.scored, rows);
                grpo_loss(
                    t,
                    lp,
                    &owners,
                    &advantages,
                    &old,
                    &reference_values,
                    &settings,
                )
            });
        }
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            let answers = seen as f64;
            println!(
                "pas {:>4}  récompense {:.3}  longueur {:.2}  divergence {:.4}",
                step + 1,
                reward_sum / answers,
                length_sum / answers,
                kl_sum / eval_every as f64,
            );
            (reward_sum, length_sum, kl_sum, seen) = (0.0, 0.0, 0.0, 0);
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
