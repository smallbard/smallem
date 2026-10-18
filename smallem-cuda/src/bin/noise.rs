//! Le bruit du gradient, et la taille de lot critique qui s'en déduit
//! (McCandlish et al., 2018).
//!
//! Usage : `cargo oxide run --bin noise -- MODÈLE [micro-lots] [séquences par micro-lot]`
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

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::data::{Batches, Shards};
use smallem::io::{SafeTensors, load_parameters};
use smallem::model::{Gpt, GptConfig};
use smallem::nn::Module;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage : noise MODÈLE [micro-lots] [séquences]")?;
    let micro_batches: usize = args.next().map_or(Ok(32), |a| a.parse())?;
    let sequences: usize = args.next().map_or(Ok(4), |a| a.parse())?;
    let file = SafeTensors::read(&path)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let gpu = smallem_cuda::gpu(true)?;
    let mut model = Gpt::new(&gpu, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, &gpu, &mut model)?;
    let train = Shards::open_dir("../data/bpe", "train.bin")?;
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
