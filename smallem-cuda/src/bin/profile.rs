//! Répartition du temps d'un pas d'entraînement entre les opérations GPU.
//!
//! Usage : `cargo oxide run --bin profile -- [options]`
//!
//! ```text
//! --config small|large|xl  architecture de départ (défaut : xl)
//! --batch N                séquences par lot (défaut : 16)
//! --mixed                  produits matriciels en bf16, sur les Tensor Cores
//! --arch A                 gpt2, modern, mla, moe, mtp, loop ou shared (défaut : gpt2)
//! --dense-experts          avec moe, chaque expert calculé sur tous les tokens
//! --context N              longueur des séquences, à la place de celle de la configuration
//! --optimizer X            comme pour `train` : adamw, muon, muon:short,period=3… (défaut : adamw)
//! --unfused                l'attention calculée avec les primitives, comme avant
//!                          l'attention fusionnée
//! ```

use std::error::Error;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smallem::backend::Primitives;
use smallem::cli::{Args, at_least_one};
use smallem::model::{Gpt, GptConfig};
use smallem::nn::{Architecture, AttentionKind, ExpertDims, LatentDims, MultiToken};
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind};
use smallem::training::train_step_clipped;
use smallem_cuda::profile::Profiled;

const WARMUP_STEPS: usize = 3;
const STEPS: usize = 5;

struct Options {
    config: String,
    batch: usize,
    mixed: bool,
    arch: String,
    dense_experts: bool,
    context: Option<usize>,
    optimizer: OptimizerKind,
    unfused: bool,
}

impl Options {
    fn parse_from(mut args: Args) -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            config: "xl".into(),
            batch: 16,
            mixed: false,
            arch: "gpt2".into(),
            dense_experts: false,
            context: None,
            optimizer: OptimizerKind::AdamW,
            unfused: false,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--config" => options.config = args.value(&flag)?,
                "--batch" => options.batch = args.value(&flag)?,
                "--mixed" => options.mixed = true,
                "--arch" => options.arch = args.value(&flag)?,
                "--dense-experts" => options.dense_experts = true,
                "--context" => options.context = Some(args.value(&flag)?),
                "--optimizer" => options.optimizer = args.value(&flag)?,
                "--unfused" => options.unfused = true,
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        at_least_one(&[
            ("--batch", options.batch),
            ("--context", options.context.unwrap_or(1)),
        ])?;
        if options.dense_experts && options.arch != "moe" {
            return Err("--dense-experts demande --arch moe".into());
        }
        Ok(options)
    }

    fn config(&self) -> Result<GptConfig, Box<dyn Error>> {
        let config = match self.config.as_str() {
            "small" => GptConfig::small(257),
            "large" => GptConfig::large(257),
            "xl" => GptConfig::xl(8192),
            other => return Err(format!("configuration inconnue : {other}").into()),
        };
        let architecture = match self.arch.as_str() {
            "gpt2" => Architecture::GPT2,
            "modern" => Architecture::MODERN,
            "mla" => Architecture::MODERN.with_attention(AttentionKind::Latent(
                LatentDims::for_model(config.embed, config.heads),
            )),
            "moe" => Architecture::MODERN.with_experts(ExpertDims::for_model(config.embed)),
            "mtp" => Architecture::MODERN.with_multi_token(MultiToken::DEEPSEEK),
            "loop" => Architecture::MODERN.with_passes(2),
            "shared" => Architecture::MODERN.with_own_keys(config.layers / 2),
            other => return Err(format!("architecture inconnue : {other}").into()),
        };
        let mut config = config.with_architecture(architecture);
        // Un contexte plus long, pour mesurer ce que coûte l'attention dense.
        if let Some(context) = self.context {
            config.context = context;
        }
        Ok(config)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse_from(Args::from_env())?;
    let config = options.config()?;
    let batch_size = options.batch;
    let gpu = smallem_cuda::gpu(options.mixed)?;
    let gpu = Profiled::new(if options.unfused {
        gpu.without_fused_attention()
    } else {
        gpu
    });
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(&gpu, config, &mut rng);
    model.set_dense_experts(options.dense_experts);
    let kind = options.optimizer;
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
        "{}, lot {batch_size} × {} : {wall:.1} ms par pas, dont {measured:.1} dans les opérations",
        options.config, config.context
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

#[cfg(test)]
mod tests {
    use super::*;

    fn options(args: &[&str]) -> Result<Options, Box<dyn Error>> {
        Options::parse_from(Args::new(
            args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>(),
        ))
    }

    #[test]
    fn options_are_named() {
        let options = options(&["--config", "small", "--arch", "moe", "--unfused"]).unwrap();
        assert!(options.unfused && !options.mixed);
        let config = options.config().unwrap();
        assert!(config.architecture.experts.is_some());
        // L'ordre ne compte plus : `--unfused` n'est pas pris pour un contexte.
        let options = self::options(&["--unfused", "--context", "2048"]).unwrap();
        assert_eq!(options.config().unwrap().context, 2048);
    }

    #[test]
    fn mistakes_are_refused() {
        assert!(options(&["mixed"]).is_err());
        assert!(options(&["--dense-experts"]).is_err());
        assert!(options(&["--arch", "rnn"]).unwrap().config().is_err());
        assert!(options(&["--batch", "0"]).is_err());
    }
}
