//! Compte les n-grammes du corpus, avant d'en faire une mémoire : combien de
//! bigrammes et de trigrammes distincts, la part des occurrences que
//! couvrent les plus fréquents, les collisions du hachage selon la taille
//! des tables, et la perte de deux modèles par comptage, bigramme et
//! trigramme, qui disent ce qu'une table sait à elle seule.
//!
//! Les n-grammes sont ceux que lit la mémoire : ils ne franchissent ni une
//! fin de document ni la frontière entre deux fichiers (voir
//! [`smallem::ngram`]).
//!
//! Usage : `ngrams [dossier_des_données]` (défaut : `../data/bpe`)

use std::error::Error;
use std::path::PathBuf;

use smallem::cli::{self, Args};
use smallem::data::{Shards, bytes_per_token};
use smallem::ngram::{NgramCounts, NgramHash, collisions, coverage, cross_entropy, interpolation};
use smallem::tokenizer::{Bpe, TOKENIZER_FILE};

/// Têtes par ordre, comme dans le plan.
const HEADS: usize = 8;
const SEED: u64 = 1234;

fn main() -> Result<(), Box<dyn Error>> {
    let mut dir = PathBuf::from("../data/bpe");
    for arg in Args::from_env() {
        dir = cli::positional(&arg)?.into();
    }
    let bpe = Bpe::load(dir.join(TOKENIZER_FILE))?;
    let vocab = bpe.vocab_size();
    let train = read_all(&Shards::open_dir(&dir, ".train.bin")?)?;
    let val_shards = Shards::open_dir(&dir, ".val.bin")?;
    let val = read_all(&val_shards)?;
    let bytes = bytes_per_token(&val_shards, &bpe)?;
    println!(
        "{} tokens d'entraînement, {} de validation, {bytes:.3} octets par token\n",
        train.iter().map(Vec::len).sum::<usize>(),
        val.iter().map(Vec::len).sum::<usize>(),
    );

    let train: Vec<&[u32]> = train.iter().map(Vec::as_slice).collect();
    let counts = NgramCounts::new(&train, vocab);
    let bigrams = counts.bigrams();
    let v = vocab as u64;
    println!(
        "bigrammes  : {:>12} distincts sur {} possibles",
        group(bigrams.len() as u64),
        group(v * v)
    );
    println!(
        "trigrammes : {:>12} distincts sur {} possibles\n",
        group(counts.trigrams().len() as u64),
        group(v * v * v)
    );
    print_coverage(&bigrams, counts.trigrams());
    print_collisions(&bigrams, counts.trigrams(), vocab);
    print_count_models(&counts, &val, bytes);
    Ok(())
}

/// Chaque fichier, à part : un n-gramme ne va pas de l'un à l'autre.
fn read_all(shards: &Shards) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    (0..shards.shard_count())
        .map(|shard| {
            let read = shards.read(shard, 0, shards.shard_len(shard))?;
            Ok(read.into_iter().map(u32::from).collect())
        })
        .collect()
}

fn print_coverage(bigrams: &[(u64, u32)], trigrams: &[(u64, u32)]) {
    println!("part des occurrences couverte par les n-grammes les plus fréquents :");
    println!(
        "{:>12} {:>10} {:>10}",
        "n-grammes", "bigrammes", "trigrammes"
    );
    for log in [14, 16, 18, 20, 22, 24] {
        println!(
            "{:>12} {:>9.2} % {:>9.2} %",
            format!("2^{log}"),
            100.0 * coverage(bigrams, 1 << log),
            100.0 * coverage(trigrams, 1 << log),
        );
    }
}

fn print_collisions(bigrams: &[(u64, u32)], trigrams: &[(u64, u32)], vocab: usize) {
    println!(
        "\npart des occurrences dont le n-gramme est le plus fréquent de sa ligne\n\
         (dans la première tête ; dans au moins une des {HEADS} têtes) :"
    );
    println!(
        "{:>10} {:>11} {:>11} {:>11} {:>11}",
        "lignes", "bi, 1", "bi, 8", "tri, 1", "tri, 8"
    );
    for log in [16, 18, 20, 22] {
        let hash = NgramHash::new(3, HEADS, 1 << log, SEED);
        let (bi_one, bi_all) = collisions(&hash, bigrams, 2, vocab);
        let (tri_one, tri_all) = collisions(&hash, trigrams, 3, vocab);
        println!(
            "{:>10} {:>9.2} % {:>9.4} % {:>9.2} % {:>9.4} %",
            format!("2^{log}"),
            100.0 * bi_one,
            100.0 * bi_all,
            100.0 * tri_one,
            100.0 * tri_all
        );
    }
}

/// Les poids d'interpolation s'ajustent sur la première moitié des
/// positions de validation, la perte se mesure sur l'autre.
fn print_count_models(counts: &NgramCounts, val: &[Vec<u32>], bytes: f64) {
    // On prédit chaque token d'après ce qui le précède : à partir du
    // deuxième de chaque fichier.
    let rows: Vec<[f64; 4]> = val
        .iter()
        .flat_map(|sequence| (1..sequence.len()).map(|t| counts.components(sequence, t)))
        .collect();
    let (fit, report) = rows.split_at(rows.len() / 2);
    println!("\nmodèles par comptage, interpolés (poids ajustés sur la moitié de la validation,");
    println!("perte sur l'autre moitié) :");
    for (name, used) in [("bigramme", 3), ("trigramme", 4)] {
        let weights = interpolation(fit, used);
        let loss = cross_entropy(report, &weights);
        println!(
            "  {name:<10} {loss:.4} nat par token, {:.4} nat par octet, poids {:?}",
            loss / bytes,
            weights
                .iter()
                .map(|w| (w * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
    }
}

/// Un entier avec des espaces entre les milliers.
fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}
