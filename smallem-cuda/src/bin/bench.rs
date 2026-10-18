//! Mesure le débit d'entraînement du Transformer sur GPU, en tokens par
//! seconde, pour les trois configurations.
//!
//! Usage : `cargo oxide run --bin bench -- [dossier_des_données]`

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::backend::Primitives;
use smallem::data::{Dataset, Shards};
use smallem::model::{Gpt, GptConfig};
use smallem::nn::{Architecture, AttentionKind, LatentDims, Module};
use smallem::optim::AdamW;
use smallem::tokenizer::VOCAB_SIZE;
use smallem::training::train_step;

/// Les premiers pas paient des coûts uniques (chargement des kernels,
/// premières allocations) : on ne les chronomètre pas.
const WARMUP_STEPS: usize = 5;

/// Avec `fused` faux, l'attention est calculée avec les primitives, comme
/// avant l'attention fusionnée.
#[allow(clippy::too_many_arguments)]
fn bench(
    name: &str,
    mixed: bool,
    fused: bool,
    config: GptConfig,
    batch_size: usize,
    steps: usize,
    train: &Dataset,
) -> Result<(), Box<dyn Error>> {
    let gpu = smallem_cuda::gpu(mixed)?;
    let gpu = if fused {
        gpu
    } else {
        gpu.without_fused_attention()
    };
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(&gpu, config, &mut rng);
    let mut optimizer = AdamW::new(1e-3, 0.1);

    let mut start = Instant::now();
    for step in 1..=WARMUP_STEPS + steps {
        let batch = train.sample_batch(&mut rng, batch_size, config.context);
        let (tokens, targets) = batch.examples();
        let loss = train_step(&gpu, &mut model, &mut optimizer, |m, t| {
            m.loss(t, &tokens, &targets, batch_size)
        });
        // Lire la perte attend que le GPU ait fini : sans cela, l'horloge
        // mesurerait la mise en file des kernels, pas leur exécution.
        if step == WARMUP_STEPS {
            gpu.item(&loss);
            start = Instant::now();
        } else if step == WARMUP_STEPS + steps {
            gpu.item(&loss);
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let tokens = steps * batch_size * config.context;
    println!(
        "{name:<20} {:>10} paramètres  lot {batch_size:>3} × {:<4} {:>8.0} tokens/s  {:>7.1} ms/pas",
        model.parameter_count(&gpu),
        config.context,
        tokens as f64 / seconds,
        seconds * 1000.0 / steps as f64,
    );
    Ok(())
}

/// Temps que l'hôte met à préparer un lot : tirage des fenêtres dans les
/// fichiers de tokens, puis conversion en indices. Pendant ce temps, le GPU
/// n'a rien à faire.
fn bench_batches(data_dir: &Path, batch_size: usize, context: usize) -> Result<(), Box<dyn Error>> {
    const BATCHES: usize = 2000;
    let shards = Shards::open_dir(data_dir, "train.bin")?;
    let mut rng = StdRng::seed_from_u64(0);
    let start = Instant::now();
    let mut tokens = 0;
    for _ in 0..BATCHES {
        let batch = shards.sample_batch(&mut rng, batch_size, context)?;
        tokens += batch.examples().0.len();
    }
    println!(
        "préparation d'un lot {batch_size} × {context} sur l'hôte : {:.0} µs ({tokens} tokens lus)",
        start.elapsed().as_secs_f64() * 1e6 / BATCHES as f64,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../data".into()));
    let train = Dataset::load(data_dir.join("train.bin"))?;
    bench(
        "petit",
        false,
        true,
        GptConfig::small(VOCAB_SIZE),
        32,
        100,
        &train,
    )?;
    bench(
        "grand",
        false,
        true,
        GptConfig::large(VOCAB_SIZE),
        16,
        20,
        &train,
    )?;
    // La configuration de l'étape 8, avec son vocabulaire BPE. Les tokens
    // tirés restent des octets : le temps de calcul n'en dépend pas.
    bench("xl", false, true, GptConfig::xl(8192), 16, 10, &train)?;
    // Les mêmes, avec les produits matriciels sur les Tensor Cores.
    bench(
        "grand, bf16",
        true,
        true,
        GptConfig::large(VOCAB_SIZE),
        16,
        20,
        &train,
    )?;
    bench("xl, bf16", true, true, GptConfig::xl(8192), 16, 10, &train)?;
    // Le socle moderne de l'étape 10, puis l'attention latente par-dessus.
    let xl = GptConfig::xl(8192);
    let modern = xl.with_architecture(Architecture::MODERN);
    let dims = LatentDims::for_model(xl.embed, xl.heads);
    let latent = xl.with_architecture(
        modern
            .architecture
            .with_attention(AttentionKind::Latent(dims)),
    );
    bench("moderne", true, true, modern, 16, 10, &train)?;
    // Le même, avec l'attention calculée comme avant l'étape 12.
    bench("moderne, sans fusion", true, false, modern, 16, 10, &train)?;
    bench("mla", true, true, latent, 16, 10, &train)?;
    bench_batches(&data_dir, 32, 64)?;
    bench_batches(&data_dir, 16, 512)?;
    Ok(())
}
