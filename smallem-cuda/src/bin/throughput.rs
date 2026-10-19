//! Débit de la génération par lots : tokens par seconde, en tout et par
//! séquence, selon le nombre de séquences générées ensemble.
//!
//! Usage : `cargo oxide run --bin throughput -- [options]`
//!
//! ```text
//! --model FICHIER    modèle (défaut : ../data/step11/muon-short-long.safetensors)
//! --data FICHIER     tokens où prendre les amorces (défaut : ../data/bpe/000_00000.val.bin)
//! --batches LISTE    tailles de lot, séparées par des virgules (défaut : 1,2,4,…,256)
//! --prompt N         longueur des amorces (défaut : 16)
//! --tokens N         tokens générés par séquence (défaut : 128)
//! --mixed            précision mixte
//! --quantize FORMAT  poids des couches linéaires quantifiés : int8, int4, int4/G
//! --quantize-output  avec --quantize, la table de sortie aussi
//! --cpu [reference]  sur le processeur : le backend `Fast`, ou `Cpu<f32>`
//! --int8-activations avec --cpu, des activations en octets devant les poids
//!                    quantifiés (`vpdpbusd`)
//! --no-memory        coupe la mémoire de n-grammes du modèle, s'il en a une
//! --memory-host      garde la table de la mémoire en mémoire centrale : la
//!                    carte ne reçoit que les lignes lues
//! --check            vérifie que chaque séquence du plus grand lot donne les
//!                    tokens qu'elle donne seule
//! ```
//!
//! Les séquences sont générées en mode glouton. Tirer un token selon les
//! probabilités demande, sur l'hôte, un softmax sur tout le vocabulaire par
//! séquence : son coût croîtrait avec le lot et se mêlerait à la mesure.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::backend::Backend;
use smallem::backend::cpu::Cpu;
use smallem::backend::fast::Fast;
use smallem::cli::Args;
use smallem::data::Dataset;
use smallem::generation::{Batch, generate};
use smallem::io::{SafeTensors, load_parameters};
use smallem::model::{Gpt, GptConfig};
use smallem::quantize::Format;
use smallem::sampling::Sampling;

const GREEDY: Sampling = Sampling {
    temperature: 0.0,
    top_k: None,
    top_p: None,
};

struct Options {
    model: PathBuf,
    data: PathBuf,
    batches: Vec<usize>,
    prompt: usize,
    tokens: usize,
    mixed: bool,
    check: bool,
    quantize: Option<Format>,
    quantize_output: bool,
    cpu: Option<bool>,
    int8_activations: bool,
    no_memory: bool,
    memory_host: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut options = Options {
        model: "../data/step11/muon-short-long.safetensors".into(),
        data: "../data/bpe/000_00000.val.bin".into(),
        batches: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
        prompt: 16,
        tokens: 128,
        mixed: false,
        check: false,
        quantize: None,
        quantize_output: false,
        cpu: None,
        int8_activations: false,
        no_memory: false,
        memory_host: false,
    };
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => options.model = args.value(&flag)?,
            "--data" => options.data = args.value(&flag)?,
            "--batches" => {
                let list: String = args.value(&flag)?;
                options.batches = list.split(',').map(str::parse).collect::<Result<_, _>>()?;
            }
            "--prompt" => options.prompt = args.value(&flag)?,
            "--tokens" => options.tokens = args.value(&flag)?,
            "--mixed" => options.mixed = true,
            "--check" => options.check = true,
            "--quantize" => options.quantize = Some(args.value(&flag)?),
            "--quantize-output" => options.quantize_output = true,
            "--cpu" => options.cpu = Some(args.accept("reference")),
            "--int8-activations" => options.int8_activations = true,
            "--no-memory" => options.no_memory = true,
            "--memory-host" => options.memory_host = true,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    if options.quantize_output && options.quantize.is_none() {
        return Err("--quantize-output demande --quantize".into());
    }
    if options.int8_activations && options.cpu != Some(false) {
        return Err("--int8-activations demande --cpu".into());
    }
    match options.cpu {
        Some(false) => run(
            &Fast::new().with_int8_activations(options.int8_activations),
            &options,
        ),
        Some(true) => run(&Cpu::<f32>::new(), &options),
        None => {
            run(&smallem_cuda::gpu(options.mixed)?, &options)?;
            if let Some(mib) = smallem_cuda::gpu_memory_mib() {
                println!("mémoire GPU : {mib} Mio");
            }
            Ok(())
        }
    }
}

fn run<B: Backend>(backend: &B, options: &Options) -> Result<(), Box<dyn Error>> {
    let (prompt_len, tokens, check) = (options.prompt, options.tokens, options.check);
    let batches = &options.batches;
    let file = SafeTensors::read(&options.model)?;
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let gpu = backend;
    let mut model = Gpt::new(gpu, config, &mut StdRng::seed_from_u64(0));
    load_parameters(&file, gpu, &mut model)?;
    if options.memory_host {
        let memory = model
            .memory_mut()
            .ok_or("--memory-host : le modèle n'a pas de mémoire")?;
        memory.move_to_host(gpu)?;
    }
    if let Some(memory) = model.memory() {
        memory.set_enabled(!options.no_memory);
        println!(
            "mémoire de n-grammes : {} paramètres de table{}{}",
            memory.table_parameters(),
            if options.memory_host {
                ", en mémoire centrale"
            } else {
                ""
            },
            if options.no_memory { ", coupée" } else { "" }
        );
    }
    if let Some(format) = options.quantize {
        model.quantize(gpu, format);
        if options.quantize_output {
            model.quantize_output(gpu, format);
        }
    }
    let data = Dataset::load(&options.data)?;

    println!(
        "{:>6} {:>12} {:>14} {:>12} {:>10}",
        "lot", "amorces (ms)", "ms par pas", "tokens/s", "par séq."
    );
    for &size in batches {
        // Des amorces prises à des endroits différents du texte.
        let stride = data.len() / (size + 1);
        let mut batch = Batch::new(gpu, &model, size);
        for sequence in 0..size {
            let start = sequence * stride;
            let prompt: Vec<u32> = data.tokens()[start..start + prompt_len]
                .iter()
                .map(|&t| u32::from(t))
                .collect();
            batch.add(&prompt);
        }
        let mut rng = StdRng::seed_from_u64(0);
        // Le premier pas lit les amorces : il est chronométré à part.
        let start = Instant::now();
        batch.step(&GREEDY, &mut rng);
        let prompts = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 1..tokens {
            batch.step(&GREEDY, &mut rng);
        }
        let seconds = start.elapsed().as_secs_f64();
        let steps = (tokens - 1) as f64;
        let rate = size as f64 * steps / seconds;
        if check && size == *batches.iter().max().expect("au moins un lot") {
            // Chaque séquence, générée seule, avec le même nombre de tokens.
            let same = (0..size)
                .filter(|&sequence| {
                    let tokens_alone = batch.tokens(sequence);
                    let prompt = &tokens_alone[..prompt_len];
                    let alone = generate(gpu, &model, prompt, tokens, &GREEDY, &mut rng);
                    alone == tokens_alone
                })
                .count();
            println!("{same} séquences sur {size} identiques à leur génération seule");
        }
        println!(
            "{size:>6} {:>12.1} {:>14.2} {rate:>12.0} {:>10.0}",
            prompts * 1000.0,
            seconds * 1000.0 / steps,
            rate / size as f64,
        );
    }
    Ok(())
}
