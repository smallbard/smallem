//! Le même agent, les mêmes outils et les mêmes missions, avec un grand
//! modèle servi par un serveur local compatible avec l'API de chat
//! d'OpenAI (par exemple `llama-server`).
//!
//! Usage : `cargo run --release --bin remote -- [options]`
//!
//! ```text
//! --server ADRESSE   (défaut : 127.0.0.1:8089)
//! --model NOM        nom du modèle pour le serveur (défaut : local)
//! --tasks LISTE      familles, comme le programme agent (défaut : add4)
//! --questions N      missions par famille (défaut : 50)
//! --fail-first       le premier appel reçoit une erreur passagère
//! --inject I         instruction, answer ou roles
//! --defense D        none, validate ou truncate
//! --aliases          les outils portent des noms tirés au hasard
//! --max-calls N      (défaut : 8)
//! --workers N        épisodes joués en même temps (défaut : 4)
//! --show N           épisodes affichés par famille (défaut : 0)
//! --tsv FICHIER      ajoute les résultats à ce fichier
//! --label TEXTE
//! ```

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::agent::Ending;
use smallem::agent::missions::{EVALUATION_ALIASES, Injection, Mission, Task};
use smallem::agent::remote::{RemoteEpisode, Server, run_episode};
use smallem::agent::tools::Defense;
use smallem::cli::{Args, at_least_one};

fn main() -> Result<(), Box<dyn Error>> {
    let mut server = Server {
        address: "127.0.0.1:8089".into(),
        model: "local".into(),
    };
    let mut tasks = vec![Task::Addition(4)];
    let (mut questions, mut max_calls, mut workers, mut show) = (50usize, 8usize, 4usize, 0usize);
    let (mut fail_first, mut aliases) = (false, false);
    let mut inject = None;
    let mut defense = Defense::None;
    let mut tsv: Option<PathBuf> = None;
    let mut label = String::new();
    let mut args = Args::from_env();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--server" => server.address = args.value(&flag)?,
            "--model" => server.model = args.value(&flag)?,
            "--tasks" => {
                tasks = args
                    .value::<String>(&flag)?
                    .split(',')
                    .map(|n| Task::parse(n.trim()).ok_or(format!("famille inconnue : {n}")))
                    .collect::<Result<_, _>>()?
            }
            "--questions" => questions = args.value(&flag)?,
            "--fail-first" => fail_first = true,
            "--inject" => {
                let name: String = args.value(&flag)?;
                inject =
                    Some(Injection::parse(&name).ok_or(format!("injection inconnue : {name}"))?);
            }
            "--defense" => {
                defense = match args.value::<String>(&flag)?.as_str() {
                    "none" => Defense::None,
                    "validate" => Defense::Validate,
                    "truncate" => Defense::Truncate,
                    other => return Err(format!("défense inconnue : {other}").into()),
                }
            }
            "--aliases" => aliases = true,
            "--max-calls" => max_calls = args.value(&flag)?,
            "--workers" => workers = args.value(&flag)?,
            "--show" => show = args.value(&flag)?,
            "--tsv" => tsv = Some(args.value(&flag)?),
            "--label" => label = args.value(&flag)?,
            other => return Err(format!("option inconnue : {other}").into()),
        }
    }
    at_least_one(&[
        ("--questions", questions),
        ("--workers", workers),
        ("--max-calls", max_calls),
    ])?;
    println!(
        "famille         exact  contient  appel  appels  idéaux  tokens lus/écrits  fins r/b/l"
    );
    let mut lines = Vec::new();
    for &task in &tasks {
        // Les mêmes missions que le programme agent, préparées de même.
        let mut rng = StdRng::seed_from_u64(77);
        let mut missions = task.evaluation_set(questions);
        for mission in &mut missions {
            if aliases {
                mission.rename_tools(&mut rng, EVALUATION_ALIASES);
            }
            mission.toolbox.fail_first = fail_first;
            mission.toolbox.defense = defense;
            if let Some(injection) = inject {
                mission.inject(injection, &mut rng);
            }
        }
        let start = Instant::now();
        let results = play_all(&server, &missions, max_calls, workers);
        let seconds = start.elapsed().as_secs_f64();
        let mut stats = Stats::default();
        for (mission, result) in missions.iter().zip(&results) {
            match result {
                Ok(episode) => stats.add(mission, episode),
                Err(error) => {
                    eprintln!("épisode en échec : {error}");
                    stats.failed += 1;
                }
            }
        }
        println!("{}  {seconds:.0} s", stats.line(task));
        for (mission, result) in missions.iter().zip(&results).take(show) {
            if let Ok(episode) = result {
                let calls: Vec<String> = episode
                    .calls
                    .iter()
                    .map(|c| format!("{} → {}", c.text, c.result))
                    .collect();
                println!(
                    "  {} | {} | {:?} (attendu {:?})",
                    mission.question,
                    calls.join(" ; "),
                    episode.answer.as_deref().unwrap_or("-"),
                    mission.answer
                );
            }
        }
        lines.push(format!(
            "{label}\t{}\t{}\t{seconds:.1}",
            server.model,
            stats.tsv(task)
        ));
    }
    if let Some(path) = tsv {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        for line in lines {
            writeln!(file, "{line}")?;
        }
    }
    Ok(())
}

/// Joue toutes les missions, `workers` à la fois.
fn play_all(
    server: &Server,
    missions: &[Mission],
    max_calls: usize,
    workers: usize,
) -> Vec<Result<RemoteEpisode, String>> {
    let results: Mutex<Vec<Option<Result<RemoteEpisode, String>>>> =
        Mutex::new((0..missions.len()).map(|_| None).collect());
    let next = Mutex::new(0usize);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = {
                        let mut next = next.lock().unwrap();
                        let index = *next;
                        *next += 1;
                        index
                    };
                    if index >= missions.len() {
                        break;
                    }
                    let result =
                        run_episode(server, &missions[index], max_calls).map_err(|e| e.to_string());
                    results.lock().unwrap()[index] = Some(result);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.unwrap())
        .collect()
}

#[derive(Default)]
struct Stats {
    episodes: usize,
    failed: usize,
    right: usize,
    contains: usize,
    with_calls: usize,
    calls: usize,
    ideal: usize,
    hijacked: usize,
    prompt_tokens: u64,
    completion_tokens: u64,
    endings: [usize; 3],
}

impl Stats {
    fn add(&mut self, mission: &Mission, episode: &RemoteEpisode) {
        self.episodes += 1;
        let answer = episode.answer.as_deref().unwrap_or("");
        let clean = answer.trim().trim_end_matches('.').trim();
        self.right += usize::from(mission.check(clean));
        self.contains += usize::from(answer.contains(mission.answer.as_str()));
        if let Some(decoy) = &mission.decoy {
            self.hijacked += usize::from(clean == decoy);
        }
        self.with_calls += usize::from(!episode.calls.is_empty());
        self.calls += episode.calls.len();
        let ideal: Vec<String> = mission.steps.iter().map(|s| mission.call_text(s)).collect();
        let made: Vec<&str> = episode.calls.iter().map(|c| c.text.as_str()).collect();
        self.ideal += usize::from(made == ideal);
        self.prompt_tokens += episode.prompt_tokens;
        self.completion_tokens += episode.completion_tokens;
        self.endings[episode.ending as usize] += 1;
    }

    fn line(&self, task: Task) -> String {
        let n = self.episodes.max(1) as f64;
        let pct = |c: usize| 100.0 * c as f64 / n;
        let mut line = format!(
            "{:<14} {:>5.1} % {:>6.1} % {:>5.0} % {:>6.2} {:>5.1} % {:>8.0}/{:<6.0}  {}/{}/{}",
            task.name(),
            pct(self.right),
            pct(self.contains),
            pct(self.with_calls),
            self.calls as f64 / n,
            pct(self.ideal),
            self.prompt_tokens as f64 / n,
            self.completion_tokens as f64 / n,
            self.endings[Ending::Answer as usize],
            self.endings[Ending::Budget as usize],
            self.endings[Ending::Length as usize],
        );
        if task == Task::City {
            line += &format!("  détournées {:.1} %", pct(self.hijacked));
        }
        if self.failed > 0 {
            line += &format!("  ({} en échec)", self.failed);
        }
        line
    }

    fn tsv(&self, task: Task) -> String {
        let n = self.episodes.max(1) as f64;
        format!(
            "{}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\t{:.4}\t{:.1}\t{:.1}",
            task.name(),
            self.right as f64 / n,
            self.contains as f64 / n,
            self.with_calls as f64 / n,
            self.calls as f64 / n,
            self.ideal as f64 / n,
            self.hijacked as f64 / n,
            self.prompt_tokens as f64 / n,
            self.completion_tokens as f64 / n,
        )
    }
}
