//! Génère du texte avec un modèle enregistré par `train`.
//!
//! Usage : `cargo oxide run --bin generate -- [options]`
//!
//! ```text
//! --model FICHIER      modèle à charger (défaut : ../data/gpt-small.safetensors)
//! --tokenizer FICHIER  table de fusions BPE (défaut : `tokenizer.bpe` à côté
//!                      du modèle s'il existe, sinon des octets)
//! --prompt TEXTE       début du texte (défaut : "The ")
//! --tokens N           nombre de tokens à générer (défaut : 500)
//! --temperature X      0 = toujours le plus probable ; 1 = fidèle au modèle (défaut : 1)
//! --top-k N            ne garder que les N tokens les plus probables
//! --top-p X            ne garder que les tokens formant une probabilité cumulée X
//! --seed N             graine du hasard (défaut : 42)
//! --no-cache           recalculer toute la séquence à chaque token
//! --mixed              produits matriciels sur les Tensor Cores (bf16)
//! --unfused            attention calculée avec les primitives, sans le kernel fusionné
//! --quantize FORMAT    poids des couches linéaires quantifiés : int8, int4, int4/G
//! --quantize-output    avec --quantize, la table de sortie aussi
//! --cpu [reference]    sur le processeur, sans la carte : le backend `Fast`,
//!                      ou avec `reference`, le backend de référence `Cpu<f32>`
//! ```

use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::backend::Backend;
use smallem::backend::cpu::Cpu;
use smallem::backend::fast::Fast;
use smallem::cli::Args;
use smallem::generation::Generator;
use smallem::io::{SafeTensors, load_parameters};
use smallem::model::{Gpt, GptConfig};
use smallem::quantize::Format;
use smallem::sampling::Sampling;
use smallem::tokenizer::Bpe;

struct Options {
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    prompt: String,
    tokens: usize,
    sampling: Sampling,
    seed: u64,
    no_cache: bool,
    mixed: bool,
    unfused: bool,
    cpu: Option<Processor>,
    quantize: Option<Format>,
    quantize_output: bool,
}

/// Le backend du processeur choisi par `--cpu`.
#[derive(Clone, Copy, PartialEq)]
enum Processor {
    Fast,
    Reference,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            model: "../data/gpt-small.safetensors".into(),
            tokenizer: None,
            prompt: "The ".into(),
            tokens: 500,
            sampling: Sampling::default(),
            seed: 42,
            no_cache: false,
            mixed: false,
            unfused: false,
            cpu: None,
            quantize: None,
            quantize_output: false,
        };
        let mut args = Args::from_env();
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => options.model = args.value(&flag)?,
                "--tokenizer" => options.tokenizer = Some(args.value(&flag)?),
                "--prompt" => options.prompt = args.value(&flag)?,
                "--tokens" => options.tokens = args.value(&flag)?,
                "--temperature" => options.sampling.temperature = args.value(&flag)?,
                "--top-k" => options.sampling.top_k = Some(args.value(&flag)?),
                "--top-p" => options.sampling.top_p = Some(args.value(&flag)?),
                "--seed" => options.seed = args.value(&flag)?,
                "--no-cache" => options.no_cache = true,
                "--mixed" => options.mixed = true,
                "--unfused" => options.unfused = true,
                "--quantize" => options.quantize = Some(args.value(&flag)?),
                "--quantize-output" => options.quantize_output = true,
                "--cpu" if args.accept("reference") => options.cpu = Some(Processor::Reference),
                "--cpu" => options.cpu = Some(Processor::Fast),
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        if options.prompt.is_empty() {
            return Err("l'amorce ne peut pas être vide".into());
        }
        if options.quantize_output && options.quantize.is_none() {
            return Err("--quantize-output demande --quantize".into());
        }
        Ok(options)
    }

    /// Le tokeniseur du modèle : celui qu'on a désigné, sinon celui qui se
    /// trouve à côté du fichier de poids, sinon les octets.
    fn tokenizer(&self) -> Result<Bpe, Box<dyn Error>> {
        let beside = self.model.with_file_name("tokenizer.bpe");
        match &self.tokenizer {
            Some(path) => Ok(Bpe::load(path)?),
            None if beside.exists() => Ok(Bpe::load(beside)?),
            None => Ok(Bpe::bytes()),
        }
    }
}

fn run<B: Backend>(
    backend: &B,
    options: &Options,
    file: &SafeTensors,
) -> Result<(), Box<dyn Error>> {
    // Sans ce qui ne sert qu'à l'entraînement : le fichier peut contenir un
    // module de prédiction multi-token, que la génération ignore.
    let config = GptConfig::from_metadata(&file.metadata)?.for_inference();
    let tokenizer = options.tokenizer()?;
    if tokenizer.vocab_size() != config.vocab {
        return Err(format!(
            "le modèle attend {} tokens, le tokeniseur en a {} : préciser --tokenizer",
            config.vocab,
            tokenizer.vocab_size(),
        )
        .into());
    }
    // Les poids tirés ici sont aussitôt remplacés par ceux du fichier. Ils
    // ont leur propre générateur : celui du tirage des tokens ne doit pas
    // dépendre du nombre de paramètres du modèle.
    let mut model = Gpt::new(backend, config, &mut StdRng::seed_from_u64(0));
    load_parameters(file, backend, &mut model)?;
    if let Some(format) = options.quantize {
        model.quantize(backend, format);
        if options.quantize_output {
            model.quantize_output(backend, format);
        }
    }
    let mut rng = StdRng::seed_from_u64(options.seed);

    let prompt: Vec<u32> = tokenizer
        .encode(&options.prompt)
        .iter()
        .map(|&t| u32::from(t))
        .collect();
    let mut generator = if options.no_cache {
        Generator::without_cache(backend, &model, &prompt)
    } else {
        Generator::new(backend, &model, &prompt)
    };

    // Le texte s'affiche à mesure qu'il s'écrit.
    let mut decoder = tokenizer.decoder();
    let mut out = std::io::stdout();
    for &token in &prompt {
        write!(out, "{}", decoder.push(token as u16))?;
    }
    let start = Instant::now();
    for _ in 0..options.tokens {
        let token = generator.next_token(&options.sampling, &mut rng);
        write!(out, "{}", decoder.push(token as u16))?;
        out.flush()?;
    }
    let seconds = start.elapsed().as_secs_f64();
    writeln!(out, "{}", decoder.finish())?;
    // Sur la sortie d'erreur, pour ne pas se mêler au texte généré.
    eprintln!("{:.0} tokens par seconde", options.tokens as f64 / seconds);
    if let Some((cache, sequence)) = generator.cache() {
        let numbers = cache.numbers_per_position(backend);
        let mib =
            |lines: usize| (lines * numbers * size_of::<B::Scalar>()) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "cache : {} positions de {numbers} nombres ({} par couche), soit {:.1} Mio \
             dans des tables de {} lignes ({:.1} Mio)",
            cache.len(sequence),
            numbers / config.layers,
            mib(cache.len(sequence)),
            cache.capacity(),
            mib(cache.capacity()),
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let file = SafeTensors::read(&options.model)?;
    if let Some(processor) = options.cpu {
        match processor {
            Processor::Fast => run(&Fast::new(), &options, &file),
            Processor::Reference => run(&Cpu::<f32>::new(), &options, &file),
        }
    } else {
        let gpu = smallem_cuda::gpu(options.mixed)?;
        let gpu = if options.unfused {
            gpu.without_fused_attention()
        } else {
            gpu
        };
        run(&gpu, &options, &file)
    }
}
