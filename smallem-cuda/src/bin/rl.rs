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

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::chat::{Encoded, Roles};
use smallem::cli::{Args, at_least_one, check_positive, check_range};
use smallem::data::{Shards, bytes_per_token};
use smallem::io::{load_gpt, save_model};
use smallem::model::Gpt;
use smallem::optim::{AnyOptimizer, OptimizerKind};
use smallem::post_training::{
    Asking, GrpoSettings, accuracy, answer_all, group_advantages, grpo_loss, sft_batch,
    token_log_probs,
};
use smallem::sampling::Sampling;
use smallem::tasks::Family;
use smallem::tokenizer::{Bpe, TOKENIZER_FILE};
use smallem::training::{train_step_clipped, validation_loss};
use smallem_cuda::Cuda;

/// Le modèle de départ, celui qu'on obtient et les données, par défaut.
const DEFAULT_MODEL: &str = "../data/post/sft.safetensors";
const DEFAULT_OUT: &str = "../data/post/rl.safetensors";
const DEFAULT_DATA: &str = "../data/bpe";

struct Options {
    model: PathBuf,
    out: PathBuf,
    data: PathBuf,
    /// REINFORCE plutôt que GRPO.
    reinforce: bool,
    /// La tâche difficile, et la tâche ordinaire qui l'accompagne.
    tasks: (Family, Family),
    steps: usize,
    prompts: usize,
    group: usize,
    inner: usize,
    eval_every: usize,
    lr: f64,
    clip: f64,
    kl: f64,
    per_token: bool,
}

impl Options {
    fn parse_from(mut args: Args) -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            model: DEFAULT_MODEL.into(),
            out: DEFAULT_OUT.into(),
            data: DEFAULT_DATA.into(),
            reinforce: false,
            tasks: (Family::LongSort, Family::Sort),
            steps: 300,
            prompts: 16,
            group: 8,
            inner: 1,
            eval_every: 50,
            lr: 2e-6,
            clip: 0.2,
            kl: 0.04,
            per_token: false,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => options.model = args.value(&flag)?,
                "--out" => options.out = args.value(&flag)?,
                "--data" => options.data = args.value(&flag)?,
                "--method" => {
                    options.reinforce = match args.value::<String>(&flag)?.as_str() {
                        "reinforce" => true,
                        "grpo" => false,
                        other => return Err(format!("méthode inconnue : {other}").into()),
                    }
                }
                "--steps" => options.steps = args.value(&flag)?,
                "--prompts" => options.prompts = args.value(&flag)?,
                "--group" => options.group = args.value(&flag)?,
                "--lr" => options.lr = args.value(&flag)?,
                "--clip" => options.clip = args.value(&flag)?,
                "--kl" => options.kl = args.value(&flag)?,
                "--per-token" => options.per_token = true,
                "--inner" => options.inner = args.value(&flag)?,
                "--eval-every" => options.eval_every = args.value(&flag)?,
                "--task" => {
                    options.tasks = match args.value::<String>(&flag)?.as_str() {
                        "long-sort" => (Family::LongSort, Family::Sort),
                        "long-addition" => (Family::LongAddition, Family::Addition),
                        // Le tri ordinaire, que le fine-tuning a déjà à
                        // moitié appris : un signal abondant, pour voir si
                        // l'apprentissage par essais améliore ce qui est déjà
                        // en partie su.
                        "sort" => (Family::Sort, Family::Sort),
                        other => return Err(format!("tâche inconnue : {other}").into()),
                    }
                }
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        at_least_one(&[
            ("--steps", options.steps),
            ("--prompts", options.prompts),
            ("--group", options.group),
            ("--inner", options.inner),
            ("--eval-every", options.eval_every),
        ])?;
        check_positive("--lr", options.lr)?;
        check_positive("--clip", options.clip)?;
        check_range("--kl", options.kl, 0.0..)?;
        Ok(options)
    }

    /// Les réglages de la perte. REINFORCE : ni borne, ni divergence, et la
    /// moyenne du groupe comme seule ligne de base.
    fn settings(&self) -> GrpoSettings {
        if self.reinforce {
            GrpoSettings {
                clip: f64::INFINITY,
                kl: 0.0,
                per_answer: false,
            }
        } else {
            GrpoSettings {
                clip: self.clip,
                kl: self.kl,
                per_answer: !self.per_token,
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse_from(Args::from_env())?;
    let settings = options.settings();
    let Options {
        model: model_path,
        out,
        data,
        reinforce,
        tasks: (hard, ordinary),
        steps,
        prompts,
        group,
        inner,
        eval_every,
        lr,
        ..
    } = options;
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = load_gpt(&gpu, &model_path, true)?;
    let reference = load_gpt(&gpu, &model_path, true)?;
    let tokenizer = Bpe::load(data.join(TOKENIZER_FILE))?;
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
    // Les sommes depuis le dernier compte rendu : récompenses et longueurs
    // par réponse, divergence par pas.
    let (mut reward_sum, mut length_sum, mut seen) = (0.0, 0.0, 0usize);
    let (mut kl_sum, mut kl_steps) = (0.0, 0usize);
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
        kl_steps += 1;
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
                kl_sum / kl_steps as f64,
            );
            (reward_sum, length_sum, seen) = (0.0, 0.0, 0);
            (kl_sum, kl_steps) = (0.0, 0);
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
        let options =
            options(&["--method", "reinforce", "--task", "sort", "--group", "4"]).unwrap();
        assert!(options.reinforce);
        assert_eq!(options.tasks, (Family::Sort, Family::Sort));
        assert_eq!((options.group, options.steps), (4, 300));
        assert_eq!(options.settings().kl, 0.0);
        let grpo = self::options(&["--per-token"]).unwrap().settings();
        assert!(!grpo.per_answer && grpo.clip == 0.2);
    }

    #[test]
    fn options_that_would_break_the_loop_are_refused() {
        for (refused, flag) in [
            (&["--eval-every", "0"][..], "--eval-every"),
            (&["--group", "0"], "--group"),
            (&["--clip", "0"], "--clip"),
            (&["--kl", "-1"], "--kl"),
            (&["--method", "ppo"], "ppo"),
            (&["--task", "chess"], "chess"),
        ] {
            let error = options(refused).err().expect(flag).to_string();
            assert!(error.contains(flag), "{error}");
        }
    }
}
