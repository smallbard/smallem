//! Le rendement de la carte : les opérations utiles d'un pas
//! d'entraînement, par seconde, rapportées au maximum de la RTX 3090.
//!
//! Usage : `cargo oxide run --bin mfu -- [--widths 128,256,512]`
//!
//! Sans option, mesure les configurations `small`, `large` et `xl` en
//! architecture moderne. Avec `--widths`, mesure la famille des lois
//! d'échelle à ces largeurs : têtes de 64, autant de couches que de têtes.
//! Lots de 16 séquences de 512 tokens, précision mixte, Muon.

use std::error::Error;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smallem::backend::Primitives;
use smallem::cli::Args;
use smallem::flops::{six_n_d, training_flops};
use smallem::model::{Gpt, GptConfig};
use smallem::nn::{Architecture, Module};
use smallem::optim::{AnyOptimizer, OptimizerKind};
use smallem::training::train_step_clipped;

/// Le maximum de la carte en `bf16` sur les Tensor Cores, accumulation en
/// `f32` (fiche GA102, annexe A).
const PEAK_BF16: f64 = 71e12;

const BATCH: usize = 16;
const WARMUP: usize = 3;
const STEPS: usize = 10;

fn main() -> Result<(), Box<dyn Error>> {
    let vocab = 8192;
    let mut widths: Option<String> = None;
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--widths" => widths = Some(args.value(&flag)?),
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let configs: Vec<(String, GptConfig)> = match widths {
        None => vec![
            ("small".into(), GptConfig::small(vocab)),
            ("large".into(), GptConfig::large(vocab)),
            ("xl".into(), GptConfig::xl(vocab)),
        ],
        Some(list) => list
            .split(',')
            .map(|width| {
                let embed: usize = width
                    .parse()
                    .ok()
                    .filter(|embed| embed % 64 == 0 && *embed > 0)
                    .ok_or_else(|| format!("--widths : `{width}` n'est pas un multiple de 64"))?;
                let config = GptConfig {
                    embed,
                    heads: embed / 64,
                    layers: embed / 64,
                    context: 512,
                    ..GptConfig::xl(vocab)
                };
                Ok((format!("C = {embed}"), config))
            })
            .collect::<Result<_, String>>()?,
    };
    println!(
        "{:<10} {:>12} {:>6} {:>12} {:>12} {:>10} {:>12} {:>8}",
        "modèle", "paramètres", "T", "FLOP/token", "6N/token", "écart", "tokens/s", "MFU"
    );
    for (name, config) in configs {
        let config = GptConfig {
            context: 512,
            ..config.with_architecture(Architecture::MODERN)
        };
        let gpu = smallem_cuda::gpu(true)?;
        let mut rng = StdRng::seed_from_u64(0);
        let mut model = Gpt::new(&gpu, config, &mut rng);
        let parameters = model.parameter_count(&gpu) as u64;
        let tokens = (BATCH * config.context) as u64;
        let per_token =
            training_flops(&config, BATCH, config.context).expect("compté") as f64 / tokens as f64;
        let rule = six_n_d(parameters, 1) as f64;
        let kind: OptimizerKind = "muon:short".parse()?;
        let mut optimizer = AnyOptimizer::new(kind, 1e-3, 0.1);
        let mut start = Instant::now();
        for step in 0..WARMUP + STEPS {
            if step == WARMUP {
                gpu.synchronize();
                start = Instant::now();
            }
            let batch: Vec<u32> = (0..tokens)
                .map(|_| rng.random_range(0..vocab as u32))
                .collect();
            let (loss, _) = train_step_clipped(&gpu, &mut model, &mut optimizer, 1.0, |m, t| {
                m.loss(t, &batch, &batch, BATCH)
            });
            gpu.item(&loss);
        }
        let rate = (STEPS as u64 * tokens) as f64 / start.elapsed().as_secs_f64();
        println!(
            "{name:<10} {parameters:>12} {:>6} {:>12.3e} {:>12.3e} {:>+9.1}% {:>12.0} {:>7.1}%",
            config.context,
            per_token,
            rule,
            100.0 * (per_token / rule - 1.0),
            rate,
            100.0 * rate * per_token / PEAK_BF16,
        );
    }
    Ok(())
}
