//! Le fine-tuning supervisé : apprendre au modèle à répondre, sur des
//! conversations faites de tâches synthétiques.
//!
//! Usage : `cargo oxide run --bin sft -- [options]`
//!
//! ```text
//! --model FICHIER    modèle de départ (défaut : ../data/step11/muon-short-long.safetensors)
//! --out FICHIER      modèle obtenu (défaut : ../data/post/sft.safetensors)
//! --data DOSSIER     tokens et tokeniseur (défaut : ../data/bpe)
//! --examples N       conversations d'entraînement par famille (défaut : 2000)
//! --steps N          pas (défaut : 1500)
//! --batch N          lignes par lot (défaut : 32)
//! --length N         tokens par ligne (défaut : 64)
//! --packed           conversations mises bout à bout dans chaque ligne
//! --lr X             taux d'apprentissage au pic (défaut : 1e-4)
//! --warmup N         pas de montée du taux (défaut : 50)
//! --mix X            part des lignes prises dans FineWeb-Edu (défaut : 0)
//! --lora R           n'entraîne que des adaptateurs de rang R
//! --eval-every N     pas entre deux évaluations (défaut : 250)
//! --questions N      questions d'évaluation par famille (défaut : 200)
//! ```
//!
//! À chaque évaluation : la perte sur des conversations de validation
//! (tirées comme celles d'entraînement, mais jamais vues), l'exactitude sur
//! chaque famille, et la perte sur la validation de FineWeb-Edu. Le modèle
//! enregistré est celui dont la perte de validation des conversations est la
//! plus basse.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use smallem::backend::{Float, Primitives};
use smallem::chat::{self, Encoded, Roles};
use smallem::cli::Args;
use smallem::data::{Batches, Shards, bytes_per_token};
use smallem::io::{SafeTensors, load_parameters, save_model};
use smallem::model::{Gpt, GptConfig};
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind, cosine_schedule};
use smallem::post_training::{Asking, accuracy, sft_batch};
use smallem::tasks::Family;
use smallem::tokenizer::Bpe;
use smallem::training::{evaluate, train_step_clipped, validation_loss};
use smallem_cuda::Cuda;

struct Options {
    model: PathBuf,
    out: PathBuf,
    data: PathBuf,
    examples: usize,
    steps: usize,
    batch: usize,
    length: usize,
    packed: bool,
    lr: f64,
    warmup: usize,
    mix: f64,
    lora: Option<usize>,
    eval_every: usize,
    questions: usize,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            model: "../data/step11/muon-short-long.safetensors".into(),
            out: "../data/post/sft.safetensors".into(),
            data: "../data/bpe".into(),
            examples: 2000,
            steps: 1500,
            batch: 32,
            length: 64,
            packed: false,
            lr: 1e-4,
            warmup: 50,
            mix: 0.0,
            lora: None,
            eval_every: 250,
            questions: 200,
        };
        let mut args = Args::from_env();
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => options.model = args.value(&flag)?,
                "--out" => options.out = args.value(&flag)?,
                "--data" => options.data = args.value(&flag)?,
                "--examples" => options.examples = args.value(&flag)?,
                "--steps" => options.steps = args.value(&flag)?,
                "--batch" => options.batch = args.value(&flag)?,
                "--length" => options.length = args.value(&flag)?,
                "--packed" => options.packed = true,
                "--lr" => options.lr = args.value(&flag)?,
                "--warmup" => options.warmup = args.value(&flag)?,
                "--mix" => options.mix = args.value(&flag)?,
                "--lora" => options.lora = Some(args.value(&flag)?),
                "--eval-every" => options.eval_every = args.value(&flag)?,
                "--questions" => options.questions = args.value(&flag)?,
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        Ok(options)
    }
}

/// Les conversations d'une famille, tirées avec les graines
/// `seed × 1 000 000 + i`.
fn conversations(tokenizer: &Bpe, roles: Roles, count: usize, seed: u64) -> Vec<Encoded> {
    Family::ALL
        .iter()
        .enumerate()
        .flat_map(|(index, family)| {
            let mut rng = StdRng::seed_from_u64(seed * 1_000_000 + index as u64);
            (0..count)
                .map(|_| chat::encode(tokenizer, roles, &family.sample(&mut rng, false)))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let file = SafeTensors::read(&options.model)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = Gpt::new(&gpu, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, &gpu, &mut model)?;
    let tokenizer = Bpe::load(options.data.join("tokenizer.bpe"))?;
    let roles = Roles::after(tokenizer.vocab_size());
    if model.config().vocab < roles.vocab() {
        model.grow_vocab(&gpu, roles.vocab());
        println!(
            "vocabulaire agrandi de trois tokens de rôle : {}",
            roles.vocab()
        );
    }
    if let Some(rank) = options.lora {
        let count = model.add_adapters(&gpu, rank, &mut StdRng::seed_from_u64(1));
        println!("adaptateurs de rang {rank} : {count} paramètres entraînés");
    }
    let train_data = Shards::open_dir(&options.data, "train.bin")?;
    let validation = Shards::open_dir(&options.data, "val.bin")?;
    let bytes = bytes_per_token(&validation, &tokenizer)?;

    let train = conversations(&tokenizer, roles, options.examples, 1);
    let held_out = conversations(&tokenizer, roles, 100, 2);
    println!(
        "{} conversations d'entraînement, {} de validation, lignes de {} tokens{}",
        train.len(),
        held_out.len(),
        options.length,
        if options.packed {
            ", conversations mises bout à bout"
        } else {
            ""
        },
    );

    let fineweb_rows = (options.batch as f64 * options.mix).round() as usize;
    let chat_rows = options.batch - fineweb_rows;
    let mut optimizer = AnyOptimizer::<Cuda>::new(OptimizerKind::AdamW, options.lr, 0.0);
    let mut rng = StdRng::seed_from_u64(3);
    // Les conversations d'entraînement, dans un ordre tiré à chaque époque.
    let mut order: Vec<usize> = Vec::new();
    let mut served = 0usize;
    let mut next_conversation = |rng: &mut StdRng, train: &[Encoded]| {
        if order.is_empty() {
            order = (0..train.len()).collect();
            order.shuffle(rng);
        }
        served += 1;
        train[order.pop().expect("une conversation")].clone()
    };

    // La perte sur les conversations de validation, en lignes remplies.
    let held_out_loss = |model: &Gpt<Cuda>| -> f64 {
        let mut total = 0.0;
        let chunks = held_out.chunks(64);
        let count = chunks.len();
        for chunk in held_out.chunks(64) {
            let mut items = chunk.iter().cloned();
            let batch = sft_batch(
                || items.next().unwrap(),
                chunk.len(),
                options.length,
                false,
                0,
            );
            total += evaluate(&gpu, model, |m, t| {
                m.masked_loss(t, &batch.tokens, &batch.targets, &batch.scored, batch.rows)
            })
            .to_f64();
        }
        total / count as f64
    };
    let report =
        |step: usize, model: &Gpt<Cuda>, rng: &mut StdRng| -> Result<f64, Box<dyn Error>> {
            let loss = held_out_loss(model);
            let mut line = format!("pas {step:>5}  validation des conversations {loss:.4}");
            for family in Family::ALL {
                let turns = family.evaluation_set(options.questions);
                let right = accuracy(&gpu, model, &tokenizer, &Asking::Chat(roles), &turns, rng);
                line += &format!("  {} {:.1} %", family.name(), 100.0 * right);
            }
            let fineweb = validation_loss(&gpu, model, &validation, 50, 16)? / bytes;
            gpu.release_memory();
            println!("{line}  FineWeb {fineweb:.4}");
            Ok(loss)
        };

    let start = Instant::now();
    let mut best = report(0, &model, &mut rng)?;
    if let Some(parent) = options.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    for step in 0..options.steps {
        optimizer.set_learning_rate(cosine_schedule(
            step,
            options.warmup,
            options.steps,
            options.lr,
            options.lr * 0.1,
        ));
        let chat = sft_batch(
            || next_conversation(&mut rng, &train),
            chat_rows,
            options.length,
            options.packed,
            0,
        );
        let (mut tokens, mut targets, mut scored) = (chat.tokens, chat.targets, chat.scored);
        if fineweb_rows > 0 {
            // Du texte ordinaire, noté partout : le modèle continue d'y
            // apprendre à prédire chaque token.
            let text = train_data.batch(&mut rng, fineweb_rows, options.length)?;
            let (inputs, outputs) = text.examples();
            tokens.extend(inputs);
            targets.extend(outputs);
            scored.extend(std::iter::repeat_n(true, fineweb_rows * options.length));
        }
        let rows = options.batch;
        let (loss, _) = train_step_clipped(&gpu, &mut model, &mut optimizer, 1.0, |m, t| {
            m.masked_loss(t, &tokens, &targets, &scored, rows)
        });
        if (step + 1) % options.eval_every == 0 || step + 1 == options.steps {
            let train_loss = gpu_item(&gpu, &loss);
            println!(
                "pas {:>5}  entraînement {train_loss:.4}  époques {:.2}  {:.0} s",
                step + 1,
                ((step + 1) * chat_rows) as f64 / train.len() as f64,
                start.elapsed().as_secs_f64()
            );
            let loss = report(step + 1, &model, &mut rng)?;
            // Avec LoRA, le modèle n'est enregistré qu'à la fin, fusionné.
            if loss < best && options.lora.is_none() {
                best = loss;
                save_model(&options.out, &gpu, &model, model.config().to_metadata())?;
                println!("enregistré");
            }
        }
    }
    if options.lora.is_some() {
        model.merge_adapters(&gpu);
        save_model(&options.out, &gpu, &model, model.config().to_metadata())?;
        println!("adaptateurs fusionnés, modèle enregistré");
    } else {
        println!("meilleure validation des conversations : {best:.4}");
    }
    Ok(())
}

fn gpu_item(gpu: &Cuda, tensor: &<Cuda as Primitives>::Tensor) -> f64 {
    gpu.item(tensor).to_f64()
}
