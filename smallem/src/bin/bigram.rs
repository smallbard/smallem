//! Modèle bigramme par comptage : la probabilité d'un token ne dépend que du
//! token précédent. Aucun apprentissage, seulement des fréquences.
//!
//! Usage : `bigram [dossier_des_données]`

use std::error::Error;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use smallem::data::Dataset;
use smallem::sampling::sample;
use smallem::tokenizer::{self, END_OF_DOCUMENT, VOCAB_SIZE};

const SEED: u64 = 42;
const GENERATED_TOKENS: usize = 600;

/// Table `VOCAB_SIZE × VOCAB_SIZE` rangée ligne par ligne : la ligne `a`
/// est la distribution du token qui suit `a`.
struct Bigram {
    probabilities: Vec<f64>,
}

impl Bigram {
    fn fit(tokens: &[u16]) -> Self {
        // Lissage +1 : chaque paire est comptée une fois d'avance. Sans lui,
        // une paire absente du train aurait une probabilité nulle, et une
        // seule occurrence en validation donnerait une perte infinie.
        let mut counts = vec![1u64; VOCAB_SIZE * VOCAB_SIZE];
        for pair in tokens.windows(2) {
            counts[pair[0] as usize * VOCAB_SIZE + pair[1] as usize] += 1;
        }

        let mut probabilities = Vec::with_capacity(counts.len());
        for row in counts.chunks(VOCAB_SIZE) {
            let total: u64 = row.iter().sum();
            probabilities.extend(row.iter().map(|&count| count as f64 / total as f64));
        }
        Self { probabilities }
    }

    /// Distribution du token qui suit `token`.
    fn next_distribution(&self, token: u16) -> &[f64] {
        let start = token as usize * VOCAB_SIZE;
        &self.probabilities[start..start + VOCAB_SIZE]
    }

    /// Entropie croisée moyenne, en nats par token : `-ln p(suivant | courant)`.
    fn loss(&self, tokens: &[u16]) -> f64 {
        let total: f64 = tokens
            .windows(2)
            .map(|pair| -self.next_distribution(pair[0])[pair[1] as usize].ln())
            .sum();
        total / (tokens.len() - 1) as f64
    }

    fn generate(&self, rng: &mut impl Rng, length: usize) -> Vec<u16> {
        // Le token de fin de document précède le premier octet de chaque
        // document : partir de lui, c'est commencer un nouveau texte.
        let mut token = END_OF_DOCUMENT;
        let mut tokens = Vec::with_capacity(length);
        for _ in 0..length {
            token = sample(rng, self.next_distribution(token)) as u16;
            tokens.push(token);
        }
        tokens
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../data".into()));
    let train = Dataset::load(data_dir.join("train.bin"))?;
    let val = Dataset::load(data_dir.join("val.bin"))?;

    let model = Bigram::fit(train.tokens());

    println!("perte uniforme   : {:.4}", (VOCAB_SIZE as f64).ln());
    println!("perte train      : {:.4}", model.loss(train.tokens()));
    println!("perte validation : {:.4}", model.loss(val.tokens()));

    let generated = model.generate(&mut StdRng::seed_from_u64(SEED), GENERATED_TOKENS);
    println!("\n--- texte généré ---\n{}", tokenizer::decode(&generated));
    Ok(())
}
