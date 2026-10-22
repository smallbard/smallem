//! Lance une grille d'entraînements, un par ligne d'un fichier de plan, et
//! note les résultats. Reprend après une interruption : un entraînement
//! déjà noté n'est pas relancé.
//!
//! Usage : `cargo oxide run --bin sweep -- PLAN DOSSIER [options communes de train]`
//!
//! Chaque ligne du plan nomme un entraînement, suivi de ses options :
//!
//! ```text
//! # largeur 256, quatre couches, budget de 1e15 opérations
//! e256-l4-c1e15  --embed 256 --heads 4 --layers 4 --budget 1e15 --lr 2e-3
//! ```
//!
//! Les options données après `DOSSIER` s'ajoutent à toutes les lignes.
//! Chaque entraînement écrit son modèle et son journal dans `DOSSIER`, et
//! une ligne de `DOSSIER/results.tsv` :
//!
//! ```text
//! nom  paramètres  opérations  pas  nats/octet  tokens/s  secondes  table
//! ```
//!
//! `paramètres` compte le réseau, ce qui calcule ; `table` la table d'une
//! mémoire de n-grammes, 0 sans mémoire.
//!
//! La perte est la perte de validation finale, sur 200 lots. Un
//! entraînement qui échoue est noté avec une perte `NaN`, pour ne pas le
//! relancer sans fin.

use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use smallem::cli::{Args, positional};

/// Ce qu'on lit dans le journal d'un entraînement.
#[derive(Debug, Default)]
struct Outcome {
    parameters: Option<u64>,
    flops: Option<f64>,
    steps: Option<u64>,
    loss: Option<f64>,
    rate: Option<f64>,
    table: Option<u64>,
}

fn read_log(log: &str) -> Outcome {
    let mut outcome = Outcome::default();
    for line in log.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if line.starts_with("mémoire de n-grammes") {
            outcome.table = words.get(4).and_then(|w| w.parse().ok());
        } else if line.contains(" paramètres, vocabulaire") {
            outcome.parameters = words.first().and_then(|w| w.parse().ok());
        } else if line.contains("opérations par pas") {
            outcome.flops = words
                .iter()
                .position(|&w| w == "en")
                .and_then(|i| words.get(i - 1))
                .and_then(|w| w.trim_end_matches(',').parse().ok());
        } else if line.starts_with("pas ") && line.contains("tokens/s") {
            outcome.steps = words.get(1).and_then(|w| w.parse().ok());
            outcome.rate = words
                .iter()
                .position(|&w| w == "tokens/s")
                .and_then(|i| words.get(i - 1))
                .and_then(|w| w.parse().ok());
        } else if line.starts_with("perte de validation finale") {
            outcome.loss = words
                .iter()
                .position(|&w| w == "nats")
                .and_then(|i| words.get(i - 1))
                .and_then(|w| w.parse().ok());
        }
    }
    outcome
}

/// Les noms déjà notés dans `results.tsv`.
fn done(results: &Path) -> Vec<String> {
    fs::read_to_string(results)
        .unwrap_or_default()
        .lines()
        .skip(1)
        .filter_map(|line| line.split('\t').next().map(str::to_string))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    const USAGE: &str = "usage : sweep PLAN DOSSIER [options]";
    let mut args = Args::from_env();
    let plan = PathBuf::from(positional(&args.next().ok_or(USAGE)?)?);
    let dir = PathBuf::from(positional(&args.next().ok_or(USAGE)?)?);
    let common: Vec<String> = args.collect();
    fs::create_dir_all(&dir)?;
    let results = dir.join("results.tsv");
    if !results.exists() {
        fs::write(
            &results,
            "nom\tparamètres\topérations\tpas\tnats/octet\ttokens/s\tsecondes\ttable\n",
        )?;
    }
    let train = std::env::current_exe()?.with_file_name("train");
    for line in fs::read_to_string(&plan)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut words = line.split_whitespace();
        let name = words.next().expect("une ligne non vide");
        if done(&results).iter().any(|n| n == name) {
            println!("{name} : déjà fait");
            continue;
        }
        let options: Vec<&str> = words.collect();
        let model = dir.join(format!("{name}.safetensors"));
        println!("{name} : {}", options.join(" "));
        let start = Instant::now();
        let output = Command::new(&train)
            .args(&options)
            .args(&common)
            .arg("--out")
            .arg(&model)
            .output()?;
        let seconds = start.elapsed().as_secs_f64();
        let log = String::from_utf8_lossy(&output.stdout).to_string()
            + &String::from_utf8_lossy(&output.stderr);
        fs::write(dir.join(format!("{name}.log")), &log)?;
        let outcome = read_log(&log);
        let field = |value: Option<String>| value.unwrap_or_else(|| "?".into());
        let row = format!(
            "{name}\t{}\t{}\t{}\t{}\t{}\t{seconds:.0}\t{}\n",
            field(outcome.parameters.map(|p| p.to_string())),
            field(outcome.flops.map(|f| format!("{f:.3e}"))),
            field(outcome.steps.map(|s| s.to_string())),
            outcome.loss.map_or("NaN".into(), |l| format!("{l:.4}")),
            field(outcome.rate.map(|r| format!("{r:.0}"))),
            outcome.table.unwrap_or(0),
        );
        OpenOptions::new()
            .append(true)
            .open(&results)?
            .write_all(row.as_bytes())?;
        println!(
            "{name} : {} nats/octet en {seconds:.0} s",
            outcome.loss.map_or("échec".into(), |l| format!("{l:.4}"))
        );
    }
    Ok(())
}
