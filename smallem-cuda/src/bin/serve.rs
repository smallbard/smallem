//! Simule un service : des requêtes arrivent à tout moment, et le serveur
//! les génère par lots, avec un cache de taille fixe. Mesure le débit et la
//! latence.
//!
//! Usage : `cargo oxide run --bin serve -- [options]`
//!
//! ```text
//! --model FICHIER    modèle (défaut : ../data/step11/muon-short-long.safetensors)
//! --data FICHIER     tokens où prendre les amorces (défaut : ../data/bpe/000_00000.val.bin)
//! --requests N       nombre de requêtes (défaut : 2000)
//! --rates LISTE      requêtes par seconde, séparées par des virgules (défaut : 50,100,200)
//! --policies LISTE   static, continuous, paged (défaut : les trois)
//! --cache-mib N      mémoire du cache, en Mio (défaut : 1024)
//! --seed N           graine des requêtes (défaut : 1)
//! --reserve X        avec des pages : part des pages qu'une admission doit
//!                    laisser libres, pour que les séquences en cours
//!                    puissent grandir (défaut : 0)
//! ```
//!
//! Les requêtes arrivent selon un processus de Poisson. Chacune a une amorce
//! de 16 à 128 tokens prise dans le texte de validation, et demande de 32 à
//! 384 tokens, tirés uniformément. Le serveur génère en mode glouton.
//!
//! Trois façons de servir :
//!
//! - `static` : des lots fixes. Quand le lot en cours est fini, le serveur
//!   prend toutes les requêtes en attente, autant que le cache en permet, et
//!   les mène à leur terme ensemble. Chaque séquence réserve tout le
//!   contexte du modèle dans le cache.
//! - `continuous` : des lots continus. Une requête entre dès qu'une place se
//!   libère, et en sort dès qu'elle est finie. Même réservation.
//! - `paged` : des lots continus, avec un cache par pages de 16 lignes,
//!   prises à mesure que les séquences s'allongent. Une requête entre s'il
//!   reste de quoi lire son amorce. Si les pages viennent à manquer, la
//!   dernière requête entrée est suspendue : ses pages sont rendues, et elle
//!   repart plus tard, son texte déjà écrit servant d'amorce.

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smallem::cache::{KvCache, PAGE_SIZE};
use smallem::cli::Args;
use smallem::data::Dataset;
use smallem::generation::Batch;
use smallem::io::load_gpt;
use smallem::model::Gpt;
use smallem::sampling::Sampling;
use smallem_cuda::Cuda;

const GREEDY: Sampling = Sampling {
    temperature: 0.0,
    top_k: None,
    top_p: None,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Policy {
    Static,
    Continuous,
    Paged,
}

struct Request {
    /// Arrivée, en secondes depuis le début.
    arrival: f64,
    prompt: Vec<u32>,
    /// Tokens demandés.
    wanted: usize,
}

/// Ce qu'on mesure d'une requête.
#[derive(Clone, Copy, Default)]
struct Timing {
    first: Option<f64>,
    finish: Option<f64>,
    generated: usize,
}

fn requests(data: &Dataset, count: usize, rate: f64, seed: u64) -> Vec<Request> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0;
    (0..count)
        .map(|_| {
            // Écart exponentiel entre deux arrivées : un processus de Poisson.
            let u: f64 = rng.random();
            time += -(1.0 - u).ln() / rate;
            let length = rng.random_range(16..=128);
            let start = rng.random_range(0..data.len() - length);
            Request {
                arrival: time,
                prompt: data.tokens()[start..start + length]
                    .iter()
                    .map(|&t| u32::from(t))
                    .collect(),
                wanted: rng.random_range(32..=384),
            }
        })
        .collect()
}

/// Le `fraction`-ième quantile de `values`.
fn quantile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}

/// Sert `requests` avec la politique `policy` et un cache de `lines`
/// lignes, et affiche ce qu'on mesure.
fn serve(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    requests: &[Request],
    policy: Policy,
    lines: usize,
    reserve: f64,
) {
    let config = model.config();
    let stages = config.layers * config.architecture.passes;
    let page_size = match policy {
        Policy::Paged => PAGE_SIZE,
        Policy::Static | Policy::Continuous => config.context,
    };
    let pages = lines / page_size;
    let mut batch = Batch::with_cache(gpu, model, KvCache::new(stages, pages, page_size));
    let mut rng = StdRng::seed_from_u64(0);
    let mut timings = vec![Timing::default(); requests.len()];
    // Les requêtes en attente, et pour chacune le texte déjà écrit si elle
    // a été suspendue.
    let mut waiting: VecDeque<(usize, Option<Vec<u32>>)> = VecDeque::new();
    // Les séquences du lot, et leur requête.
    let mut running: BTreeMap<usize, usize> = BTreeMap::new();
    let mut arrived = 0;
    let (mut preemptions, mut largest) = (0, 0);
    let start = Instant::now();
    let now = || start.elapsed().as_secs_f64();
    loop {
        while arrived < requests.len() && requests[arrived].arrival <= now() {
            waiting.push_back((arrived, None));
            arrived += 1;
        }
        // Les entrées dans le lot.
        let admit = match policy {
            Policy::Static => running.is_empty(),
            Policy::Continuous | Policy::Paged => true,
        };
        while admit && let Some(&(request, ref written)) = waiting.front() {
            let prompt = written
                .clone()
                .unwrap_or_else(|| requests[request].prompt.clone());
            // Les séquences déjà admises n'ont pas encore pris leurs pages :
            // elles le feront au prochain pas, d'où `pages_needed`. Avec des
            // pages, on garde en plus de quoi faire grandir la séquence d'un
            // token.
            let margin = match policy {
                Policy::Paged => 1 + (reserve * pages as f64) as usize,
                Policy::Static | Policy::Continuous => 0,
            };
            let needed = batch.pages_needed() + prompt.len().div_ceil(page_size) + margin;
            let fits = batch.cache().free_pages() >= needed;
            if !fits {
                break;
            }
            waiting.pop_front();
            let limit = requests[request].prompt.len() + requests[request].wanted;
            running.insert(batch.add_limited(&prompt, Some(limit)), request);
        }
        if running.is_empty() {
            if arrived == requests.len() && waiting.is_empty() {
                break;
            }
            if arrived < requests.len() {
                let wait = (requests[arrived].arrival - now()).max(0.0);
                std::thread::sleep(Duration::from_secs_f64(wait));
            }
            continue;
        }
        // Avec des pages, le prochain pas doit tenir : sinon la dernière
        // requête entrée est suspendue.
        while policy == Policy::Paged && batch.pages_needed() > batch.cache().free_pages() {
            let (&sequence, &request) = running.iter().next_back().expect("une séquence");
            let written = batch.tokens(sequence).to_vec();
            batch.finish(sequence);
            running.remove(&sequence);
            waiting.push_front((request, Some(written)));
            preemptions += 1;
        }
        largest = largest.max(running.len());
        let drawn = batch.step(&GREEDY, &mut rng);
        let time = now();
        for (sequence, _) in drawn {
            let timing = &mut timings[running[&sequence]];
            timing.first.get_or_insert(time);
            timing.generated += 1;
        }
        running.retain(|&sequence, &mut request| {
            let done = !batch.is_running(sequence);
            if done {
                timings[request].finish = Some(time);
            }
            !done
        });
    }

    let end = timings
        .iter()
        .map(|t| t.finish.expect("toutes finies"))
        .fold(0.0, f64::max);
    let span = end - requests[0].arrival;
    let generated: usize = timings.iter().map(|t| t.generated).sum();
    let mut first: Vec<f64> = timings
        .iter()
        .zip(requests)
        .map(|(t, r)| t.first.expect("un token") - r.arrival)
        .collect();
    let mut latency: Vec<f64> = timings
        .iter()
        .zip(requests)
        .map(|(t, r)| t.finish.expect("finie") - r.arrival)
        .collect();
    let mut per_token: Vec<f64> = timings
        .iter()
        .map(|t| {
            (t.finish.expect("finie") - t.first.expect("commencée"))
                / (t.generated - 1).max(1) as f64
        })
        .collect();
    println!(
        "{:<11} {:>8.1} {:>9.0} {:>9.0} {:>9.0} {:>9.1} {:>9.1} {:>9.2} {:>9.2} {:>7} {:>6}",
        format!("{policy:?}").to_lowercase(),
        requests.len() as f64 / span,
        generated as f64 / span,
        1000.0 * quantile(&mut first, 0.5),
        1000.0 * quantile(&mut first, 0.99),
        1000.0 * quantile(&mut per_token, 0.5),
        1000.0 * quantile(&mut per_token, 0.99),
        quantile(&mut latency, 0.5),
        quantile(&mut latency, 0.99),
        largest,
        preemptions,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_path: PathBuf = "../data/step11/muon-short-long.safetensors".into();
    let mut data_path: PathBuf = "../data/bpe/000_00000.val.bin".into();
    let (mut count, mut cache_mib, mut seed) = (2000usize, 1024usize, 1u64);
    let mut reserve = 0.0f64;
    let mut rates = "50,100,200".to_string();
    let mut policies = "static,continuous,paged".to_string();
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model_path = args.value(&flag)?,
            "--data" => data_path = args.value(&flag)?,
            "--requests" => count = args.value(&flag)?,
            "--rates" => rates = args.value(&flag)?,
            "--policies" => policies = args.value(&flag)?,
            "--cache-mib" => cache_mib = args.value(&flag)?,
            "--seed" => seed = args.value(&flag)?,
            "--reserve" => reserve = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    let gpu = smallem_cuda::gpu(false)?;
    let model = load_gpt(&gpu, &model_path, true)?;
    let config = model.config();
    let data = Dataset::load(&data_path)?;
    let numbers = config.cache_numbers_per_position();
    let lines = (cache_mib << 20) / (4 * numbers);
    println!(
        "cache de {cache_mib} Mio : {numbers} nombres par position, {lines} lignes, \
         {} séquences de {} positions réservées",
        lines / config.context,
        config.context,
    );
    for rate in rates.split(',') {
        let rate: f64 = rate.parse()?;
        let requests = requests(&data, count, rate, seed);
        println!("\n{count} requêtes, {rate} par seconde");
        println!(
            "{:<11} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>6}",
            "politique",
            "req./s",
            "tokens/s",
            "1er p50",
            "1er p99",
            "tok. p50",
            "tok. p99",
            "tot. p50",
            "tot. p99",
            "lot max",
            "susp."
        );
        for policy in policies.split(',') {
            let policy = match policy {
                "static" => Policy::Static,
                "continuous" => Policy::Continuous,
                "paged" => Policy::Paged,
                other => return Err(format!("politique inconnue : {other}").into()),
            };
            serve(&gpu, &model, &requests, policy, lines, reserve);
        }
    }
    Ok(())
}
