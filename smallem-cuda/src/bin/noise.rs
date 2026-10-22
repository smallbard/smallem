//! Le bruit du gradient, et la taille de lot critique qui s'en déduit
//! (McCandlish et al., 2018).
//!
//! Usage : `cargo oxide run --bin noise -- --model FICHIER [options]`
//!
//! ```text
//! --model FICHIER      le modèle mesuré
//! --data DOSSIER       tokens d'entraînement (défaut : ../data/bpe)
//! --micro-batches K    micro-lots, au moins 2 (défaut : 32)
//! --sequences b        séquences par micro-lot (défaut : 4)
//! ```
//!
//! Le gradient d'un lot de `B` séquences est le vrai gradient `G`, plus un
//! bruit de variance `tr(Σ) / B`. On le mesure avec deux tailles de lot :
//! `K` micro-lots de `b` séquences chacun, et leur moyenne, un lot de `K·b`.
//!
//! ```text
//! E |g_b|²  = |G|² + tr(Σ) / b
//! |G|²      ≈ (B |g_B|² − b |g_b|²) / (B − b)
//! tr(Σ)     ≈ (|g_b|² − |g_B|²) / (1/b − 1/B)
//! B_bruit   = tr(Σ) / |G|²
//! ```
//!
//! En dessous de `B_bruit` séquences par lot, le bruit domine : doubler le
//! lot divise presque par deux le nombre de pas nécessaires. Au-dessus, le
//! gradient est déjà bien estimé, et un lot plus gros ne sert presque plus.

use std::error::Error;
use std::path::PathBuf;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::cli::{Args, at_least_one};
use smallem::data::{Batches, Shards};
use smallem::io::load_gpt;
use smallem::nn::Module;

struct Options {
    model: PathBuf,
    data: PathBuf,
    micro_batches: usize,
    sequences: usize,
}

impl Options {
    fn parse_from(mut args: Args) -> Result<Self, Box<dyn Error>> {
        let mut model = None;
        let mut options = Self {
            model: PathBuf::new(),
            data: "../data/bpe".into(),
            micro_batches: 32,
            sequences: 4,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => model = Some(args.value(&flag)?),
                "--data" => options.data = args.value(&flag)?,
                "--micro-batches" => options.micro_batches = args.value(&flag)?,
                "--sequences" => options.sequences = args.value(&flag)?,
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        options.model = model.ok_or("--model FICHIER est demandé")?;
        at_least_one(&[("--sequences", options.sequences)])?;
        // Les formules comparent deux tailles de lot : `b` et `K·b`.
        if options.micro_batches < 2 {
            return Err("--micro-batches doit valoir au moins 2".into());
        }
        Ok(options)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Options {
        model: path,
        data,
        micro_batches,
        sequences,
    } = Options::parse_from(Args::from_env())?;
    let gpu = smallem_cuda::gpu(true)?;
    let model = load_gpt(&gpu, &path, true)?;
    let config = model.config();
    let train = Shards::open_dir(&data, "train.bin")?;
    let mut rng = StdRng::seed_from_u64(11);

    // La somme des gradients des micro-lots, et la somme de leurs normes
    // au carré.
    let mut sum: Vec<Option<_>> = model.parameters().iter().map(|_| None).collect();
    let mut squares = 0.0;
    for _ in 0..micro_batches {
        let batch = train.batch(&mut rng, sequences, config.context)?;
        let (tokens, targets) = batch.examples();
        let tape = Tape::new(&gpu);
        let loss = model.loss(&tape, &tokens, &targets, sequences);
        let grads = tape.backward(loss);
        for (parameter, total) in model.parameters().iter().zip(&mut sum) {
            let Some(grad) = parameter.grad(tape.id(), &grads) else {
                continue;
            };
            squares += gpu.item(&gpu.sum_all(&gpu.mul(grad, grad))).to_f64();
            *total = Some(match total.take() {
                Some(previous) => gpu.add(&previous, grad),
                None => grad.clone(),
            });
        }
    }
    let k = micro_batches as f64;
    let small = sequences as f64;
    let big = small * k;
    let small_norm = squares / k;
    let mean_norm: f64 = sum
        .iter()
        .flatten()
        .map(|total| gpu.item(&gpu.sum_all(&gpu.mul(total, total))).to_f64() / (k * k))
        .sum();
    let true_norm = (big * mean_norm - small * small_norm) / (big - small);
    let trace = (small_norm - mean_norm) / (1.0 / small - 1.0 / big);
    let noise = trace / true_norm;
    println!(
        "|g_{small}|² = {small_norm:.4e}, |g_{big}|² = {mean_norm:.4e}, |G|² ≈ {true_norm:.4e}, \
         tr(Σ) ≈ {trace:.4e}"
    );
    println!(
        "taille de lot critique ≈ {noise:.1} séquences, soit {:.0} tokens",
        noise * config.context as f64
    );
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
    fn the_model_is_required_and_two_batch_sizes_are_needed() {
        let options = options(&["--model", "m.safetensors", "--sequences", "8"]).unwrap();
        assert_eq!((options.micro_batches, options.sequences), (32, 8));
        assert!(self::options(&[]).is_err());
        assert!(self::options(&["m.safetensors"]).is_err());
        assert!(self::options(&["--model", "m", "--micro-batches", "1"]).is_err());
    }
}
