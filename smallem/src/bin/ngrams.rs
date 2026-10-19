//! Compte les n-grammes du corpus, avant d'en faire une mémoire : combien de
//! bigrammes et de trigrammes distincts, la part des occurrences que
//! couvrent les plus fréquents, les collisions du hachage selon la taille
//! des tables, et la perte de deux modèles par comptage, bigramme et
//! trigramme, qui disent ce qu'une table sait à elle seule.
//!
//! Les n-grammes sont ceux que lit la mémoire : ils ne franchissent pas une
//! fin de document (voir [`smallem::ngram`]).
//!
//! Usage : `ngrams [dossier_des_données]` (défaut : `../data/bpe`)

use std::error::Error;
use std::path::PathBuf;
use std::thread;

use smallem::data::{Shards, bytes_per_token};
use smallem::ngram::{NgramHash, count_of, suffix, trigram_counts};
use smallem::tokenizer::Bpe;

/// Le vocabulaire du tokeniseur BPE ; vérifié au chargement.
const VOCAB: usize = 8192;
const V: u64 = VOCAB as u64;
/// Têtes par ordre, comme dans le plan.
const HEADS: usize = 8;
const SEED: u64 = 1234;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "../data/bpe".into()),
    );
    let train = read_all(&Shards::open_dir(&dir, ".train.bin")?)?;
    let val_shards = Shards::open_dir(&dir, ".val.bin")?;
    let val = read_all(&val_shards)?;
    let bpe = Bpe::load(dir.join("tokenizer.bpe"))?;
    if bpe.vocab_size() != VOCAB {
        return Err(format!(
            "vocabulaire de {} tokens, {VOCAB} attendus",
            bpe.vocab_size()
        )
        .into());
    }
    let bytes = bytes_per_token(&val_shards, &bpe)?;
    println!(
        "{} tokens d'entraînement, {} de validation, {bytes:.3} octets par token\n",
        train.len(),
        val.len()
    );

    // --- Les comptes ---
    let mut unigrams = vec![0u64; VOCAB];
    let mut bigrams = vec![0u32; VOCAB * VOCAB];
    for &t in &train {
        unigrams[t as usize] += 1;
    }
    let mut gram = [0u32; 3];
    for t in 0..train.len() {
        suffix(&train, t, &mut gram);
        bigrams[(gram[1] as u64 * V + gram[0] as u64) as usize] += 1;
    }
    let trigrams = trigram_counts(&train, VOCAB);

    let bigram_list: Vec<(u64, u32)> = bigrams
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(k, c)| (k as u64, *c))
        .collect();
    println!(
        "bigrammes  : {:>12} distincts sur {} possibles",
        group(bigram_list.len() as u64),
        group(V * V)
    );
    println!(
        "trigrammes : {:>12} distincts sur {} possibles\n",
        group(trigrams.len() as u64),
        group(V * V * V)
    );

    // --- La couverture ---
    println!("part des occurrences couverte par les n-grammes les plus fréquents :");
    println!(
        "{:>12} {:>10} {:>10}",
        "n-grammes", "bigrammes", "trigrammes"
    );
    let covered = |list: &[(u64, u32)], top: usize| -> f64 {
        let mut counts: Vec<u32> = list.iter().map(|(_, c)| *c).collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        let kept: u64 = counts.iter().take(top).map(|&c| u64::from(c)).sum();
        kept as f64 / total as f64
    };
    for log in [14, 16, 18, 20, 22, 24] {
        println!(
            "{:>12} {:>9.2} % {:>9.2} %",
            format!("2^{log}"),
            100.0 * covered(&bigram_list, 1 << log),
            100.0 * covered(&trigrams, 1 << log),
        );
    }

    // --- Les collisions ---
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
        let (bi_one, bi_all) = collisions(&hash, &bigram_list, 0, |k| [k % V, k / V, 0]);
        let (tri_one, tri_all) =
            collisions(&hash, &trigrams, 1, |k| [k % V, (k / V) % V, k / (V * V)]);
        println!(
            "{:>10} {:>9.2} % {:>9.4} % {:>9.2} % {:>9.4} %",
            format!("2^{log}"),
            100.0 * bi_one,
            100.0 * bi_all,
            100.0 * tri_one,
            100.0 * tri_all
        );
    }

    // --- Les modèles par comptage ---
    let (fit, report) = val.split_at(val.len() / 2);
    let components = |tokens: &[u32]| -> Vec<[f64; 4]> {
        let total = train.len() as f64;
        let mut gram = [0u32; 3];
        let mut out = Vec::with_capacity(tokens.len());
        for t in 1..tokens.len() {
            // On prédit `tokens[t]` d'après ce qui le précède.
            suffix(tokens, t, &mut gram);
            let (c, b, a) = (gram[0] as u64, gram[1] as u64, gram[2] as u64);
            let bigram = f64::from(bigrams[(b * V + c) as usize]);
            let context = unigrams[b as usize] as f64;
            let pair = f64::from(bigrams[(a * V + b) as usize]);
            let trigram = f64::from(count_of(&trigrams, (a * V + b) * V + c));
            out.push([
                1.0 / V as f64,
                unigrams[c as usize] as f64 / total,
                if context > 0.0 { bigram / context } else { 0.0 },
                if pair > 0.0 { trigram / pair } else { 0.0 },
            ]);
        }
        out
    };
    let fit = components(fit);
    let report = components(report);
    println!("\nmodèles par comptage, interpolés (poids ajustés sur la moitié de la validation,");
    println!("perte sur l'autre moitié) :");
    for (name, used) in [("bigramme", 3), ("trigramme", 4)] {
        let weights = interpolation(&fit, used);
        let loss = cross_entropy(&report, &weights);
        println!(
            "  {name:<10} {loss:.4} nat par token, {:.4} nat par octet, poids {:?}",
            loss / bytes,
            weights
                .iter()
                .map(|w| (w * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

fn read_all(shards: &Shards) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut tokens = Vec::with_capacity(shards.len());
    for shard in 0..shards.shard_count() {
        let read = shards.read(shard, 0, shards.shard_len(shard))?;
        tokens.extend(read.into_iter().map(u32::from));
    }
    Ok(tokens)
}

/// La part des occurrences dont le n-gramme est le plus fréquent de sa
/// ligne : dans la première tête de l'ordre, puis dans au moins une de ses
/// têtes. Une ligne partagée sert surtout son n-gramme le plus fréquent. `order_index` vaut 0 pour les bigrammes, 1 pour les
/// trigrammes ; `tokens` rend un n-gramme du plus récent au plus ancien.
fn collisions(
    hash: &NgramHash,
    grams: &[(u64, u32)],
    order_index: usize,
    tokens: impl Fn(u64) -> [u64; 3] + Sync,
) -> (f64, f64) {
    let heads = hash.heads();
    let owned: Vec<Vec<bool>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..heads)
            .map(|head| {
                let tokens = &tokens;
                scope.spawn(move || {
                    let table = order_index * heads + head;
                    let row = |k: u64| -> usize {
                        let [a, b, c] = tokens(k);
                        hash.hash(table, &[a as u32, b as u32, c as u32]) as usize
                    };
                    let mut most = vec![0u32; hash.rows(table)];
                    for (k, count) in grams {
                        let slot = &mut most[row(*k)];
                        *slot = (*slot).max(*count);
                    }
                    grams
                        .iter()
                        .map(|(k, count)| most[row(*k)] == *count)
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let total: u64 = grams.iter().map(|(_, c)| u64::from(*c)).sum();
    let (mut one, mut all) = (0u64, 0u64);
    for (i, (_, count)) in grams.iter().enumerate() {
        if owned[0][i] {
            one += u64::from(*count);
        }
        if owned.iter().any(|head| head[i]) {
            all += u64::from(*count);
        }
    }
    (one as f64 / total as f64, all as f64 / total as f64)
}

/// Les poids d'interpolation des `used` premières composantes, ajustés par
/// l'algorithme EM : chaque poids devient la part moyenne de la probabilité
/// qu'apporte sa composante.
fn interpolation(components: &[[f64; 4]], used: usize) -> Vec<f64> {
    let mut weights = vec![1.0 / used as f64; used];
    for _ in 0..50 {
        let mut shares = vec![0.0; used];
        for row in components {
            let mixed: f64 = (0..used).map(|i| weights[i] * row[i]).sum();
            for i in 0..used {
                shares[i] += weights[i] * row[i] / mixed;
            }
        }
        for (weight, share) in weights.iter_mut().zip(shares) {
            *weight = share / components.len() as f64;
        }
    }
    weights
}

fn cross_entropy(components: &[[f64; 4]], weights: &[f64]) -> f64 {
    let total: f64 = components
        .iter()
        .map(|row| {
            -weights
                .iter()
                .zip(row)
                .map(|(w, p)| w * p)
                .sum::<f64>()
                .ln()
        })
        .sum();
    total / components.len() as f64
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
