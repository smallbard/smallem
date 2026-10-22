//! Charge GPT-2 small et compare ses sorties à celles de PyTorch.
//!
//! C'est la preuve la plus solide que l'architecture et les kernels sont
//! corrects : 124 millions de paramètres entraînés par d'autres ne donnent
//! les bons logits que si chaque opération fait exactement ce qu'elle doit.
//!
//! Préparation, une fois :
//!
//! 1. télécharger dans `../data/gpt2` les fichiers de
//!    `https://huggingface.co/openai-community/gpt2` ;
//! 2. lancer `python scripts/gpt2_reference.py ../data/gpt2`.
//!
//! Usage : `cargo oxide run --bin gpt2 -- [--cpu] [dossier]`

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::cpu::Cpu;
use smallem::backend::{Backend, Float};
use smallem::cli::{Args, positional};
use smallem::generation::generate;
use smallem::gpt2::{GPT2_SMALL, convert};
use smallem::io::{SafeTensors, load_parameters};
use smallem::model::Gpt;
use smallem::sampling::Sampling;
use smallem_cuda::Cuda;

/// Écart toléré sur un logit. Les logits de GPT-2 valent autour de -100 et
/// traversent 12 couches : les arrondis en f32, faits dans un ordre
/// différent de celui de PyTorch, s'accumulent jusqu'à ce niveau.
const TOLERANCE: f32 = 2e-3;

/// Lit un tenseur d'identifiants, enregistrés en f32 par le script Python.
fn token_ids(file: &SafeTensors, name: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let tensor = file
        .tensors
        .get(name)
        .ok_or_else(|| format!("`{name}` absent du fichier de référence"))?;
    Ok(tensor.data.iter().map(|&id| id as u32).collect())
}

fn run<B: Backend>(backend: &B, directory: &Path) -> Result<bool, Box<dyn Error>> {
    let start = Instant::now();
    let weights = convert(
        &SafeTensors::read(directory.join("model.safetensors"))?,
        GPT2_SMALL,
    )?;
    let mut model = Gpt::new(backend, GPT2_SMALL, &mut StdRng::seed_from_u64(0));
    load_parameters(&weights, backend, &mut model)?;
    println!("modèle chargé en {:.1} s", start.elapsed().as_secs_f64());

    let reference = SafeTensors::read(directory.join("reference.safetensors"))?;
    let tokens = token_ids(&reference, "tokens")?;
    let expected_greedy = token_ids(&reference, "greedy")?;
    let expected_logits = &reference
        .tensors
        .get("logits")
        .ok_or("`logits` absent du fichier de référence")?
        .data;

    // 1. Les logits de chaque position de l'amorce.
    let tape = Tape::new(backend);
    let logits = backend.to_vec(&tape.value(model.logits(&tape, &tokens, 1)));
    drop(tape);
    if logits.len() != expected_logits.len() {
        return Err("les logits n'ont pas la taille attendue".into());
    }
    let mut largest_gap = 0.0f32;
    let mut largest_logit = 0.0f32;
    for (actual, expected) in logits.iter().zip(expected_logits) {
        let actual = actual.to_f64() as f32;
        largest_gap = largest_gap.max((actual - expected).abs());
        largest_logit = largest_logit.max(expected.abs());
    }
    let logits_match = largest_gap <= TOLERANCE;
    println!(
        "logits  : {} valeurs, écart maximal {largest_gap:.2e} (tolérance {TOLERANCE:.0e}, \
         logits jusqu'à {largest_logit:.0} en valeur absolue) : {}",
        logits.len(),
        if logits_match { "OK" } else { "ÉCHEC" },
    );

    // 2. La suite la plus probable, token après token.
    let greedy = Sampling {
        temperature: 0.0,
        ..Default::default()
    };
    let generated = generate(
        backend,
        &model,
        &tokens,
        expected_greedy.len(),
        &greedy,
        &mut StdRng::seed_from_u64(0),
    );
    let continuation = &generated[tokens.len()..];
    let greedy_match = continuation == expected_greedy;
    println!(
        "suite   : {} tokens générés : {}",
        continuation.len(),
        if greedy_match {
            "identiques à PyTorch"
        } else {
            "DIFFÉRENTS"
        },
    );
    if !greedy_match {
        println!("  attendu : {expected_greedy:?}\n  obtenu  : {continuation:?}");
    }
    Ok(logits_match && greedy_match)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut cpu = false;
    let mut directory = PathBuf::from("../data/gpt2");
    for arg in Args::from_env() {
        if arg == "--cpu" {
            cpu = true;
        } else {
            directory = positional(&arg)?.into();
        }
    }
    let ok = if cpu {
        run(&Cpu::<f32>::new(), &directory)?
    } else {
        run(&Cuda::new(0)?, &directory)?
    };
    if !ok {
        return Err("les sorties diffèrent de la référence".into());
    }
    Ok(())
}
