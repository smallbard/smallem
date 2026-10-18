//! Répartition du temps d'un pas d'entraînement entre les opérations GPU.
//!
//! Usage : `cargo oxide run --bin profile -- [small|large|xl] [lot] [mixed|f32] [gpt2|modern|mla|moe|moe-dense|mtp|loop|shared] [contexte] [optimiseur] [unfused]`
//!
//! Avec `unfused`, l'attention est calculée avec les primitives, comme avant
//! l'attention fusionnée.
//!
//! L'optimiseur se nomme comme pour `train` : `adamw`, `muon`, `muon:short,period=3`…

use std::error::Error;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smallem::backend::Primitives;
use smallem::model::{Gpt, GptConfig};
use smallem::nn::{Architecture, AttentionKind, ExpertDims, LatentDims, MultiToken};
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind};
use smallem::training::train_step_clipped;
use smallem_cuda::profile::Profiled;

const WARMUP_STEPS: usize = 3;
const STEPS: usize = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "xl".into());
    let config = match name.as_str() {
        "small" => GptConfig::small(257),
        "large" => GptConfig::large(257),
        "xl" => GptConfig::xl(8192),
        other => return Err(format!("configuration inconnue : {other}").into()),
    };
    let batch_size: usize = args.next().map_or(Ok(16), |arg| arg.parse())?;

    let mixed = args.next().is_some_and(|arg| arg == "mixed");
    let architecture = match args.next().as_deref() {
        None | Some("gpt2") => Architecture::GPT2,
        Some("modern") => Architecture::MODERN,
        Some("mla") => Architecture::MODERN.with_attention(AttentionKind::Latent(
            LatentDims::for_model(config.embed, config.heads),
        )),
        Some("moe" | "moe-dense") => {
            Architecture::MODERN.with_experts(ExpertDims::for_model(config.embed))
        }
        Some("mtp") => Architecture::MODERN.with_multi_token(MultiToken::DEEPSEEK),
        Some("loop") => Architecture::MODERN.with_passes(2),
        Some("shared") => Architecture::MODERN.with_own_keys(config.layers / 2),
        Some(other) => return Err(format!("architecture inconnue : {other}").into()),
    };
    let dense = std::env::args().any(|arg| arg == "moe-dense");
    let mut config = config.with_architecture(architecture);
    // Un contexte plus long, pour mesurer ce que coûte l'attention dense.
    if let Some(context) = args.next() {
        config.context = context.parse()?;
    }
    let unfused = std::env::args().any(|arg| arg == "unfused");
    let gpu = smallem_cuda::gpu(mixed)?;
    let gpu = Profiled::new(if unfused {
        gpu.without_fused_attention()
    } else {
        gpu
    });
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(&gpu, config, &mut rng);
    model.set_dense_experts(dense);
    let kind: OptimizerKind = args
        .next()
        .filter(|arg| arg != "unfused")
        .map_or(Ok(OptimizerKind::AdamW), |arg| arg.parse())?;
    let mut optimizer = AnyOptimizer::new(kind, 1e-3, 0.1);
    // Passé la montée de SAMuon-lite : son amplification est mesurée entière.
    optimizer.set_progress(1.0);

    let mut start = Instant::now();
    for step in 0..WARMUP_STEPS + STEPS {
        if step == WARMUP_STEPS {
            gpu.reset();
            start = Instant::now();
        }
        // Le temps de calcul ne dépend pas des tokens : on les tire au hasard.
        let mut tokens = || -> Vec<u32> {
            (0..batch_size * config.context)
                .map(|_| rng.random_range(0..config.vocab as u32))
                .collect()
        };
        let (inputs, targets) = (tokens(), tokens());
        let (loss, _) = train_step_clipped(&gpu, &mut model, &mut optimizer, 1.0, |m, t| {
            m.training_loss(t, &inputs, &targets, batch_size)
        });
        gpu.item(&loss);
    }
    let wall = start.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;

    let report = gpu.report();
    let measured: f64 = report
        .iter()
        .map(|(_, timing)| timing.total.as_secs_f64() * 1000.0 / STEPS as f64)
        .sum();
    println!(
        "{name}, lot {batch_size} × {} : {wall:.1} ms par pas, dont {measured:.1} dans les opérations",
        config.context
    );
    if let Some(mib) = smallem_cuda::gpu_memory_mib() {
        println!("mémoire GPU : {mib} Mio");
    }
    println!(
        "{:<28} {:>8} {:>10} {:>7}",
        "opération", "appels", "ms/pas", "part"
    );
    for (operation, timing) in report {
        let ms = timing.total.as_secs_f64() * 1000.0 / STEPS as f64;
        println!(
            "{operation:<28} {:>8} {ms:>10.2} {:>6.1} %",
            timing.calls / STEPS as u64,
            100.0 * ms / measured,
        );
    }
    Ok(())
}
