//! Mesure le décodage spéculatif : taux d'acceptation, tokens par passe du
//! modèle visé, et vitesse, contre la génération ordinaire.
//!
//! Usage : `cargo oxide run --bin speculate -- [options]`
//!
//! ```text
//! --target FICHIER   modèle visé (défaut : ../data/step11/muon-short-long.safetensors)
//! --drafts LISTE     proposeurs, séparés par des virgules
//!                    (défaut : ../data/inference/draft2.safetensors,../data/inference/draft4.safetensors)
//! --mtp FICHIER      modèle muni d'un module multi-token (défaut : ../data/step10/mtp.safetensors)
//! --data FICHIER     tokens où prendre les amorces (défaut : ../data/bpe/000_00000.val.bin)
//! --prompts N        amorces de 32 tokens (défaut : 8)
//! --tokens N         tokens générés par amorce (défaut : 200)
//! --temperature X    0 pour le mode glouton (défaut : 1)
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::cli::Args;
use smallem::data::Dataset;
use smallem::generation::Generator;
use smallem::io::load_gpt;
use smallem::model::Gpt;
use smallem::sampling::Sampling;
use smallem::speculative::{DraftDecoder, MultiTokenDecoder, Statistics};
use smallem_cuda::Cuda;

/// Une façon de générer : prolonge une amorce d'au moins `tokens` tokens,
/// et rend les tokens produits et les statistiques de la spéculation.
type Run<'a> = dyn FnMut(&[u32], usize, &mut StdRng) -> (Vec<u32>, Statistics) + 'a;

/// Génère à partir de chaque amorce, et affiche le débit et ce qu'a donné
/// la spéculation. Rend les textes produits, pour les comparer.
fn measure(name: &str, prompts: &[Vec<u32>], tokens: usize, run: &mut Run) -> (f64, Vec<Vec<u32>>) {
    let mut rng = StdRng::seed_from_u64(1);
    let mut total = Statistics::default();
    let mut produced = 0;
    let mut texts = Vec::new();
    let start = Instant::now();
    for prompt in prompts {
        let (text, statistics) = run(prompt, tokens, &mut rng);
        produced += text.len() - prompt.len();
        total.passes += statistics.passes;
        total.proposed += statistics.proposed;
        total.accepted += statistics.accepted;
        texts.push(text);
    }
    let rate = produced as f64 / start.elapsed().as_secs_f64();
    if total.proposed > 0 {
        println!(
            "{name:<24} {rate:>8.0} tokens/s   acceptation {:>5.1} %   {:.2} tokens par passe",
            100.0 * total.acceptance(),
            total.tokens_per_pass(produced),
        );
    } else {
        println!("{name:<24} {rate:>8.0} tokens/s");
    }
    (rate, texts)
}

/// La génération ordinaire, sans spéculation.
fn alone(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    prompt: &[u32],
    tokens: usize,
    sampling: &Sampling,
    rng: &mut StdRng,
) -> (Vec<u32>, Statistics) {
    let mut generator = Generator::new(gpu, model, prompt);
    generator.extend(tokens, sampling, rng);
    (generator.into_tokens(), Statistics::default())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut target_path: PathBuf = "../data/step11/muon-short-long.safetensors".into();
    let mut drafts =
        "../data/inference/draft2.safetensors,../data/inference/draft4.safetensors".to_string();
    let mut mtp_path: PathBuf = "../data/step10/mtp.safetensors".into();
    let mut data_path: PathBuf = "../data/bpe/000_00000.val.bin".into();
    let (mut count, mut tokens, mut temperature) = (8usize, 200usize, 1.0f64);
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--target" => target_path = args.value(&flag)?,
            "--drafts" => drafts = args.value(&flag)?,
            "--mtp" => mtp_path = args.value(&flag)?,
            "--data" => data_path = args.value(&flag)?,
            "--prompts" => count = args.value(&flag)?,
            "--tokens" => tokens = args.value(&flag)?,
            "--temperature" => temperature = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let sampling = Sampling {
        temperature,
        ..Sampling::default()
    };
    let gpu = smallem_cuda::gpu(false)?;
    let data = Dataset::load(&data_path)?;
    let stride = data.len() / (count + 1);
    let prompts: Vec<Vec<u32>> = (0..count)
        .map(|i| {
            data.tokens()[i * stride..i * stride + 32]
                .iter()
                .map(|&t| u32::from(t))
                .collect()
        })
        .collect();
    println!("{count} amorces de 32 tokens, {tokens} tokens chacune, température {temperature}");

    let target = load_gpt(&gpu, &target_path, true)?;
    let (base, reference) = measure("seul", &prompts, tokens, &mut |prompt, tokens, rng| {
        alone(&gpu, &target, prompt, tokens, &sampling, rng)
    });
    for path in drafts.split(',') {
        let draft = load_gpt(&gpu, Path::new(path), true)?;
        let name = Path::new(path)
            .file_stem()
            .map_or(path.into(), |s| s.to_string_lossy());
        for k in [1, 2, 3, 4, 6, 8] {
            let (rate, texts) = measure(
                &format!("{name}, k = {k}"),
                &prompts,
                tokens,
                &mut |prompt, tokens, rng| {
                    let mut decoder = DraftDecoder::new(&gpu, &target, &draft, prompt, k);
                    while decoder.tokens().len() < prompt.len() + tokens {
                        decoder.step(&sampling, rng);
                    }
                    (
                        decoder.tokens()[..prompt.len() + tokens].to_vec(),
                        decoder.statistics,
                    )
                },
            );
            if temperature == 0.0 {
                let same = texts.iter().zip(&reference).filter(|(a, b)| a == b).count();
                println!(
                    "    ×{:.2} ; {same} textes sur {count} identiques",
                    rate / base
                );
            } else {
                println!("    ×{:.2}", rate / base);
            }
        }
    }

    let mtp = load_gpt(&gpu, &mtp_path, false)?;
    let (mtp_base, mtp_reference) = measure(
        "module MTP : seul",
        &prompts,
        tokens,
        &mut |prompt, tokens, rng| alone(&gpu, &mtp, prompt, tokens, &sampling, rng),
    );
    let (rate, texts) = measure(
        "module MTP, k = 1",
        &prompts,
        tokens,
        &mut |prompt, tokens, rng| {
            let mut decoder = MultiTokenDecoder::new(&gpu, &mtp, prompt);
            while decoder.tokens().len() < prompt.len() + tokens {
                decoder.step(&sampling, rng);
            }
            (
                decoder.tokens()[..prompt.len() + tokens].to_vec(),
                decoder.statistics,
            )
        },
    );
    let same = texts
        .iter()
        .zip(&mtp_reference)
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "    ×{:.2} ; {same} textes sur {count} identiques",
        rate / mtp_base
    );
    Ok(())
}
