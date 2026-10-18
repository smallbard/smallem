//! Convertit des fichiers Parquet de FineWeb-Edu en fichiers de tokens :
//! des `u16` petit-boutistes, sans en-tête.
//!
//! Usage : `prepare [options] [fichier.parquet …]`
//!
//! ```text
//! --tokenizer FICHIER   table de fusions BPE ; sans elle, un token par octet
//! --out DOSSIER         où écrire (défaut : le dossier du tokeniseur, sinon ../data)
//! --limit-mb N          s'arrêter après N Mo de texte d'entraînement par fichier
//! ```
//!
//! Deux usages :
//!
//! - sans tokeniseur, la tranche de mise au point des étapes 1 à 7 : les
//!   100 premiers Mo du fichier, en octets, dans `train.bin` et `val.bin` ;
//! - avec un tokeniseur, des fichiers entiers : chaque `nom.parquet` donne
//!   `nom.train.bin` et `nom.val.bin`.
//!
//! L'encodage BPE est réparti sur tous les cœurs, et chaque document est
//! décodé aussitôt pour vérifier qu'on retrouve le texte d'origine.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use smallem::cli::Args;
use smallem::corpus::{ParquetTexts, is_validation};
use smallem::tokenizer::{Bpe, END_OF_DOCUMENT, Encoder};

/// Volume de la tranche de mise au point, quand aucun tokeniseur n'est donné.
const TUNING_TEXT_MB: u64 = 100;

const DEFAULT_INPUT: &str = "../data/000_00000.parquet";

struct Options {
    tokenizer: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    limit_mb: Option<u64>,
    inputs: Vec<PathBuf>,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            tokenizer: None,
            out_dir: None,
            limit_mb: None,
            inputs: Vec::new(),
        };
        let mut args = Args::from_env();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokenizer" => options.tokenizer = Some(args.value(&arg)?),
                "--out" => options.out_dir = Some(args.value(&arg)?),
                "--limit-mb" => options.limit_mb = Some(args.value(&arg)?),
                other if other.starts_with("--") => {
                    return Err(format!("option inconnue : {other}").into());
                }
                input => options.inputs.push(input.into()),
            }
        }
        if options.inputs.is_empty() {
            options.inputs.push(DEFAULT_INPUT.into());
        }
        Ok(options)
    }
}

/// Un fichier de tokens en cours d'écriture, avec ses compteurs.
struct TokenFile {
    path: PathBuf,
    writer: BufWriter<File>,
    documents: u64,
    tokens: u64,
    text_bytes: u64,
}

impl TokenFile {
    fn create(path: PathBuf) -> std::io::Result<Self> {
        Ok(Self {
            writer: BufWriter::new(File::create(&path)?),
            path,
            documents: 0,
            tokens: 0,
            text_bytes: 0,
        })
    }

    /// Écrit un document déjà encodé, token de fin de document compris.
    fn write_document(&mut self, tokens: &[u16], text_bytes: usize) -> std::io::Result<()> {
        for token in tokens {
            self.writer.write_all(&token.to_le_bytes())?;
        }
        self.documents += 1;
        self.tokens += tokens.len() as u64;
        self.text_bytes += text_bytes as u64;
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        println!(
            "  {} : {} documents, {} tokens, {:.1} Mo, {:.3} octets par token",
            self.path.display(),
            self.documents,
            self.tokens,
            (self.tokens * 2) as f64 / (1024.0 * 1024.0),
            self.text_bytes as f64 / self.tokens as f64,
        );
        Ok(())
    }
}

/// Encode un document, suivi du token de fin de document, et vérifie que
/// le décodage redonne le texte.
fn encode_document(bpe: &Bpe, encoder: &mut Encoder, text: &str) -> Vec<u16> {
    let mut tokens = Vec::with_capacity(text.len() / 2);
    encoder.encode_into(text, &mut tokens);
    assert!(
        bpe.decode_bytes(&tokens) == text.as_bytes(),
        "le décodage ne redonne pas le texte d'origine",
    );
    tokens.push(END_OF_DOCUMENT);
    tokens
}

/// Encode des documents en parallèle : chaque fil reçoit une tranche
/// contiguë des textes et son propre encodeur, qui garde sa mémoire des
/// mots d'un appel à l'autre. Le résultat est dans l'ordre des textes.
fn encode_documents(bpe: &Bpe, encoders: &mut [Encoder], texts: &[String]) -> Vec<Vec<u16>> {
    let per_thread = texts.len().div_ceil(encoders.len()).max(1);
    std::thread::scope(|scope| {
        let workers: Vec<_> = texts
            .chunks(per_thread)
            .zip(encoders.iter_mut())
            .map(|(slice, encoder)| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|text| encode_document(bpe, encoder, text))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("un fil d'encodage a paniqué"))
            .collect()
    })
}

fn prepare_file(
    input: &Path,
    train_path: PathBuf,
    val_path: PathBuf,
    bpe: &Bpe,
    limit_bytes: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let corpus = ParquetTexts::open(input)?;
    println!(
        "{} : {} documents en {} groupes de lignes",
        input.display(),
        corpus.documents(),
        corpus.row_groups(),
    );
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut encoders: Vec<Encoder> = (0..threads).map(|_| bpe.encoder()).collect();

    let mut train = TokenFile::create(train_path)?;
    let mut val = TokenFile::create(val_path)?;
    let mut document_index = 0u64;

    'groups: for group in 0..corpus.row_groups() {
        let texts = corpus.read_group(group)?;
        let encoded = encode_documents(bpe, &mut encoders, &texts);
        for (text, tokens) in texts.iter().zip(&encoded) {
            if is_validation(document_index) {
                val.write_document(tokens, text.len())?;
            } else {
                train.write_document(tokens, text.len())?;
            }
            document_index += 1;
            if limit_bytes.is_some_and(|limit| train.text_bytes >= limit) {
                break 'groups;
            }
        }
    }

    let text_bytes = train.text_bytes + val.text_bytes;
    train.finish()?;
    val.finish()?;
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "  {:.1} Mo de texte en {seconds:.1} s ({:.1} Mo/s sur {threads} fils)",
        text_bytes as f64 / (1024.0 * 1024.0),
        text_bytes as f64 / (1024.0 * 1024.0) / seconds,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let megabytes = |mb: u64| mb * 1024 * 1024;

    let Some(tokenizer) = &options.tokenizer else {
        // Tranche de mise au point : un seul fichier, en octets.
        let [input] = options.inputs.as_slice() else {
            return Err("sans tokeniseur, un seul fichier est attendu".into());
        };
        let out_dir = options.out_dir.unwrap_or_else(|| "../data".into());
        let limit = megabytes(options.limit_mb.unwrap_or(TUNING_TEXT_MB));
        return prepare_file(
            input,
            out_dir.join("train.bin"),
            out_dir.join("val.bin"),
            &Bpe::bytes(),
            Some(limit),
        );
    };

    let bpe = Bpe::load(tokenizer)?;
    println!(
        "{} : vocabulaire de {} tokens",
        tokenizer.display(),
        bpe.vocab_size()
    );
    let out_dir = match options.out_dir {
        Some(dir) => dir,
        None => tokenizer.parent().unwrap_or(Path::new(".")).to_path_buf(),
    };
    for input in &options.inputs {
        let stem = input
            .file_stem()
            .ok_or("nom de fichier attendu")?
            .to_string_lossy();
        prepare_file(
            input,
            out_dir.join(format!("{stem}.train.bin")),
            out_dir.join(format!("{stem}.val.bin")),
            &bpe,
            options.limit_mb.map(megabytes),
        )?;
    }
    Ok(())
}
