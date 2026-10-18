//! Évalue un modèle : son exactitude sur chaque famille de tâches, et sa
//! perte sur la validation de FineWeb-Edu.
//!
//! Usage : `cargo oxide run --bin evaluate -- [options]`
//!
//! ```text
//! --model FICHIER    modèle (défaut : ../data/step11/muon-short-long.safetensors)
//! --data DOSSIER     tokens et tokeniseur (défaut : ../data/bpe)
//! --questions N      questions par famille (défaut : 200)
//! --shots N          exemples dans la consigne, pour un modèle sans tokens de rôle
//!                    (défaut : 3)
//! --long             ajoute les additions de 4 chiffres et les tris de 6 à 8 nombres
//! --show N           affiche les N premières réponses de chaque famille
//! ```
//!
//! Un modèle dont le vocabulaire a les trois tokens de rôle est interrogé en
//! conversation ; un autre, avec des exemples dans la consigne.

use std::error::Error;
use std::path::PathBuf;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::chat::Roles;
use smallem::cli::Args;
use smallem::data::{Shards, bytes_per_token};
use smallem::io::{SafeTensors, load_parameters};
use smallem::model::{Gpt, GptConfig};
use smallem::post_training::{Asking, accuracy, answer_all};
use smallem::sampling::Sampling;
use smallem::tasks::Family;
use smallem::tokenizer::Bpe;
use smallem::training::validation_loss;

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_path: PathBuf = "../data/step11/muon-short-long.safetensors".into();
    let mut data: PathBuf = "../data/bpe".into();
    let (mut questions, mut shots, mut long, mut show) = (200usize, 3usize, false, 0usize);
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = args.value(&flag)?,
            "--data" => data = args.value(&flag)?,
            "--questions" => questions = args.value(&flag)?,
            "--shots" => shots = args.value(&flag)?,
            "--long" => long = true,
            "--show" => show = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let file = SafeTensors::read(&model_path)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = Gpt::new(&gpu, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, &gpu, &mut model)?;
    let tokenizer = Bpe::load(data.join("tokenizer.bpe"))?;
    let validation = Shards::open_dir(&data, "val.bin")?;
    let bytes = bytes_per_token(&validation, &tokenizer)?;

    let roles = Roles::after(tokenizer.vocab_size());
    let chat = config.vocab == roles.vocab();
    let mut families = Family::ALL.to_vec();
    if long {
        families.extend([Family::LongAddition, Family::LongSort]);
    }
    let mut rng = StdRng::seed_from_u64(0);
    for family in families {
        let asking = if chat {
            Asking::Chat(roles)
        } else {
            // Les exemples viennent du générateur d'entraînement.
            let mut examples = StdRng::seed_from_u64(7);
            Asking::FewShot(
                (0..shots)
                    .map(|_| family.sample(&mut examples, false))
                    .collect(),
            )
        };
        let turns = family.evaluation_set(questions);
        if show > 0 {
            let greedy = Sampling {
                temperature: 0.0,
                top_k: None,
                top_p: None,
            };
            let shown = &turns[..show.min(turns.len())];
            let answers = answer_all(
                &gpu, &model, &tokenizer, &asking, shown, 1, 24, &greedy, &mut rng,
            );
            for (turn, answers) in shown.iter().zip(answers) {
                println!(
                    "  {} → {:?} (attendu {:?})",
                    turn.question, answers[0].0, turn.answer
                );
            }
        }
        let right = accuracy(&gpu, &model, &tokenizer, &asking, &turns, &mut rng);
        println!("{:<16} {:>6.1} %", family.name(), 100.0 * right);
    }
    let loss = validation_loss(&gpu, &model, &validation, 200, 16)? / bytes;
    println!("FineWeb-Edu      {loss:.4} nats par octet");
    Ok(())
}
