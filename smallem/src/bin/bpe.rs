//! Apprend une table de fusions BPE sur le début d'un fichier Parquet de
//! FineWeb-Edu, puis mesure le résultat sur des documents de validation.
//!
//! Usage : `cargo run --release --bin bpe -- [options] [fichier.parquet]`
//!
//! ```text
//! --vocab N       taille du vocabulaire (défaut : 8192)
//! --text-mb N     volume de texte d'entraînement, en Mo (défaut : 100)
//! --out FICHIER   où enregistrer la table (défaut : ../data/bpe/tokenizer.bpe)
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use smallem::cli::{self, Args};
use smallem::corpus::{ParquetTexts, is_validation};
use smallem::tokenizer::{Bpe, FIRST_MERGE, count_words};

struct Options {
    vocab: usize,
    text_mb: u64,
    out: PathBuf,
    input: PathBuf,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            vocab: 8192,
            text_mb: 100,
            out: "../data/bpe/tokenizer.bpe".into(),
            input: "../data/000_00000.parquet".into(),
        };
        let mut args = Args::from_env();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--vocab" => options.vocab = args.value(&arg)?,
                "--text-mb" => options.text_mb = args.value(&arg)?,
                "--out" => options.out = args.value(&arg)?,
                input => options.input = cli::positional(input)?.into(),
            }
        }
        if !(FIRST_MERGE as usize..=u16::MAX as usize + 1).contains(&options.vocab) {
            return Err(format!(
                "--vocab doit être compris entre {FIRST_MERGE} et {} : les tokens tiennent sur 16 bits",
                u16::MAX as usize + 1
            )
            .into());
        }
        Ok(options)
    }
}

/// Un token tel qu'on peut l'afficher : les blancs et les octets qui ne
/// forment pas de l'UTF-8 sont rendus visibles.
fn printable(bytes: &[u8]) -> String {
    format!("{:?}", String::from_utf8_lossy(bytes)).replace(' ', "␣")
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let corpus = ParquetTexts::open(&options.input)?;
    let limit = options.text_mb * 1024 * 1024;

    // Les documents de validation ne servent pas à apprendre les fusions :
    // ils mesurent ensuite le tokeniseur sur du texte qu'il n'a pas vu.
    let start = Instant::now();
    let mut word_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut validation: Vec<String> = Vec::new();
    let (mut document_index, mut train_bytes, mut words) = (0u64, 0u64, 0u64);
    'groups: for group in 0..corpus.row_groups() {
        for text in corpus.read_group(group)? {
            if is_validation(document_index) {
                validation.push(text);
            } else {
                train_bytes += text.len() as u64;
                count_words(&text, &mut word_counts);
            }
            document_index += 1;
            if train_bytes >= limit {
                break 'groups;
            }
        }
    }
    for count in word_counts.values() {
        words += count;
    }
    println!(
        "{:.1} Mo de texte : {words} mots, dont {} distincts ({:.1} s)",
        train_bytes as f64 / (1024.0 * 1024.0),
        word_counts.len(),
        start.elapsed().as_secs_f64(),
    );

    let start = Instant::now();
    let bpe = Bpe::train(&word_counts, options.vocab);
    println!(
        "{} fusions apprises en {:.1} s : vocabulaire de {} tokens",
        bpe.merges().len(),
        start.elapsed().as_secs_f64(),
        bpe.vocab_size(),
    );
    if let Some(dir) = options.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    bpe.save(&options.out)?;
    println!("table enregistrée dans {}", options.out.display());

    let show = |range: std::ops::Range<usize>| {
        range
            .map(|token| printable(bpe.token_bytes(token as u16)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    // Moins de fusions que prévu : le vocabulaire demandé est petit, ou le
    // corpus n'avait plus de paire à fusionner.
    let (first, end) = (FIRST_MERGE as usize, bpe.vocab_size());
    if end < options.vocab {
        println!(
            "attention : {} tokens demandés, il ne restait rien à fusionner après {end}",
            options.vocab
        );
    }
    println!("premières fusions : {}", show(first..end.min(first + 20)));
    println!(
        "dernières fusions : {}",
        show(end.saturating_sub(20).max(first)..end)
    );

    // Validation : le décodage doit redonner chaque document, à l'octet près.
    let start = Instant::now();
    let mut encoder = bpe.encoder();
    let (mut bytes, mut tokens) = (0usize, 0usize);
    let mut longest = 0;
    for text in &validation {
        let mut encoded = Vec::new();
        encoder.encode_into(text, &mut encoded);
        if bpe.decode(&encoded) != *text {
            return Err("decode(encode(texte)) != texte".into());
        }
        bytes += text.len();
        tokens += encoded.len();
        longest = longest.max(
            encoded
                .iter()
                .map(|&t| bpe.token_bytes(t).len())
                .max()
                .unwrap_or(0),
        );
    }
    println!(
        "validation : {} documents, {bytes} octets → {tokens} tokens, \
         soit {:.3} octets par token ; aller-retour exact ({:.1} s)",
        validation.len(),
        bytes as f64 / tokens as f64,
        start.elapsed().as_secs_f64(),
    );
    println!("plus long token rencontré : {longest} octets");
    Ok(())
}
