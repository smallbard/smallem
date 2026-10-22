//! Ce que retient la mémoire de n-grammes d'un modèle : sa perte de
//! validation avec la mémoire, puis la mémoire coupée, position par
//! position, rangée selon le nombre de fois que le trigramme lu à cette
//! position apparaît dans les données d'entraînement.
//!
//! Usage : `cargo oxide run --bin recall -- MODÈLE [options]`
//!
//! ```text
//! --baseline FICHIER  un modèle sans mémoire, mesuré de même, pour comparer
//! --data DOSSIER      les fichiers de tokens (défaut : ../data/bpe)
//! --batches N         lots de validation de 16 séquences (défaut : 20)
//! --cpu               sur le processeur, avec le backend `Fast`
//! ```
//!
//! Couper la mémoire, c'est la retirer du flux : le modèle fonctionne
//! comme s'il ne l'avait pas, avec tout ce qu'il a appris autour d'elle.

use std::error::Error;
use std::path::PathBuf;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::fast::Fast;
use smallem::backend::{Backend, Float};
use smallem::cli::Args;
use smallem::data::{Shards, bytes_per_token};
use smallem::io::load_gpt;
use smallem::ngram::{count_of, suffix, trigram_counts, trigram_key};
use smallem::tokenizer::{Bpe, TOKENIZER_FILE};

const BATCH: usize = 16;
/// La graine des lots de validation, celle de l'entraînement.
const EVAL_SEED: u64 = 0x5EED;
/// Les tranches : nombre d'apparitions du trigramme dans l'entraînement.
const BUCKETS: [(u32, u32, &str); 5] = [
    (0, 0, "jamais vu"),
    (1, 9, "1 à 9 fois"),
    (10, 999, "10 à 999"),
    (1000, 99_999, "1 000 à 99 999"),
    (100_000, u32::MAX, "100 000 et plus"),
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = Args::from_env();
    let model: PathBuf = args.next().ok_or("usage : recall MODÈLE [options]")?.into();
    let (mut baseline, mut data, mut batches, mut cpu) = (
        None::<PathBuf>,
        PathBuf::from("../data/bpe"),
        20usize,
        false,
    );
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--baseline" => baseline = Some(args.value(&flag)?),
            "--data" => data = args.value(&flag)?,
            "--batches" => batches = args.value(&flag)?,
            "--cpu" => cpu = true,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let tokenizer = Bpe::load(data.join(TOKENIZER_FILE))?;
    let vocab = tokenizer.vocab_size();
    let val = Shards::open_dir(&data, ".val.bin")?;
    let bytes = bytes_per_token(&val, &tokenizer)?;

    // Les trigrammes de l'entraînement, comptés comme la mémoire les lit.
    let train = Shards::open_dir(&data, ".train.bin")?;
    let mut tokens = Vec::with_capacity(train.len());
    for shard in 0..train.shard_count() {
        tokens.extend(
            train
                .read(shard, 0, train.shard_len(shard))?
                .into_iter()
                .map(u32::from),
        );
    }
    let counts = trigram_counts(&tokens, vocab);
    drop(tokens);
    println!("{} trigrammes distincts dans l'entraînement", counts.len());

    let measure = |path: &PathBuf| -> Result<Losses, Box<dyn Error>> {
        if cpu {
            losses(&Fast::new(), path, &val, batches, &counts, vocab)
        } else {
            losses(
                &smallem_cuda::gpu(true)?,
                path,
                &val,
                batches,
                &counts,
                vocab,
            )
        }
    };
    let with = measure(&model)?;
    let base = baseline.as_ref().map(measure).transpose()?;

    println!(
        "\npertes en nats par octet, sur {} positions de validation",
        with.on.iter().map(|b| b.1).sum::<usize>()
    );
    print!(
        "{:<18} {:>9} {:>10} {:>10} {:>10}",
        "trigramme vu", "positions", "mémoire", "coupée", "écart"
    );
    if base.is_some() {
        print!(" {:>10} {:>10}", "sans", "gain");
    }
    println!();
    let total: usize = with.on.iter().map(|b| b.1).sum();
    let mean = |(sum, count): (f64, usize)| sum / count.max(1) as f64 / bytes;
    let mut rows: Vec<(String, usize, f64, f64, Option<f64>)> = BUCKETS
        .iter()
        .enumerate()
        .map(|(i, (_, _, name))| {
            (
                name.to_string(),
                with.on[i].1,
                mean(with.on[i]),
                mean(with.off[i]),
                base.as_ref().map(|b| mean(b.on[i])),
            )
        })
        .collect();
    let sum = |buckets: &[(f64, usize)]| {
        buckets
            .iter()
            .fold((0.0, 0), |(s, c), (bs, bc)| (s + bs, c + bc))
    };
    rows.push((
        "toutes".into(),
        total,
        mean(sum(&with.on)),
        mean(sum(&with.off)),
        base.as_ref().map(|b| mean(sum(&b.on))),
    ));
    for (name, count, on, off, without) in rows {
        print!(
            "{name:<18} {:>8.1} % {on:>10.4} {off:>10.4} {:>+10.4}",
            100.0 * count as f64 / total as f64,
            off - on
        );
        if let Some(without) = without {
            print!(" {without:>10.4} {:>+10.4}", without - on);
        }
        println!();
    }
    Ok(())
}

/// La somme des pertes et le nombre de positions de chaque tranche, avec la
/// mémoire et sans.
struct Losses {
    on: Vec<(f64, usize)>,
    off: Vec<(f64, usize)>,
}

fn losses<B: Backend>(
    backend: &B,
    path: &PathBuf,
    val: &Shards,
    batches: usize,
    counts: &[(u64, u32)],
    vocab: usize,
) -> Result<Losses, Box<dyn Error>> {
    let model = load_gpt(backend, path, true)?;
    let config = model.config();
    let mut result = Losses {
        on: vec![(0.0, 0); BUCKETS.len()],
        off: vec![(0.0, 0); BUCKETS.len()],
    };
    let mut rng = StdRng::seed_from_u64(EVAL_SEED);
    for _ in 0..batches {
        let batch = val.sample_batch(&mut rng, BATCH, config.context)?;
        let (tokens, targets) = batch.examples();
        // La tranche de chaque position : le trigramme qui finit là.
        let mut gram = [0u32; 3];
        let buckets: Vec<usize> = tokens
            .chunks(config.context)
            .flat_map(|sequence| {
                (0..sequence.len())
                    .map(|t| {
                        suffix(sequence, t, &mut gram);
                        let seen = count_of(counts, trigram_key(&gram, vocab));
                        BUCKETS
                            .iter()
                            .position(|(low, high, _)| (*low..=*high).contains(&seen))
                            .expect("tranches complètes")
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for (enabled, out) in [(true, &mut result.on), (false, &mut result.off)] {
            if let Some(memory) = model.memory() {
                memory.set_enabled(enabled);
            }
            let tape = Tape::new(backend);
            let logits = model.logits(&tape, &tokens, BATCH);
            let logits = backend.to_vec(&tape.value(logits));
            for (position, (row, &target)) in logits.chunks(vocab).zip(&targets).enumerate() {
                let largest = row.iter().fold(f64::MIN, |m, x| m.max(x.to_f64()));
                let total: f64 = row.iter().map(|x| (x.to_f64() - largest).exp()).sum();
                let loss = largest + total.ln() - row[target as usize].to_f64();
                let bucket = &mut out[buckets[position]];
                bucket.0 += loss;
                bucket.1 += 1;
            }
        }
    }
    Ok(result)
}
