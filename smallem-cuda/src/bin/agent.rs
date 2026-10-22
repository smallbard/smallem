//! Un agent : un modèle dans une boucle, qui appelle des outils. Ce
//! programme joue des épisodes, les évalue, et entraîne le modèle à agir.
//!
//! Usage : `cargo oxide run --bin agent -- COMMANDE [options]`
//!
//! ```text
//! show    joue et affiche quelques épisodes de chaque famille
//! eval    évalue un modèle sur des familles de missions
//! sft     fine-tuning supervisé sur des traces idéales
//! rl      GRPO dans la boucle : le modèle agit, la réponse finale est notée
//! ```
//!
//! Options communes :
//!
//! ```text
//! --model FICHIER    modèle (défaut : ../data/post/sft.safetensors)
//! --data DOSSIER     tokens et tokeniseur (défaut : ../data/bpe)
//! --tasks LISTE      familles, séparées par des virgules : add2…add6, reverse,
//!                    sort, long-sort, sentence, city, ages1…ages6,
//!                    sum-backwards, top-two, product
//! --questions N      missions d'évaluation par famille (défaut : 200)
//! --describe D       none, names ou aliases : décrire les outils dans la
//!                    consigne, sous leur nom ou sous des noms tirés au hasard
//! --fail-first       le premier appel de chaque épisode reçoit une erreur passagère
//! --inject I         instruction, answer ou roles : la ville lue est piégée
//! --defense D        none, validate ou truncate
//! --faulty N         la calculatrice ajoute N à chaque somme
//! --special          (faute exprès) les noms de tokens spéciaux d'un résultat
//!                    deviennent ces tokens
//! --no-cache         chaque reprise relit tout l'épisode
//! --max-calls N      budget d'appels (défaut : 8)
//! --temperature X    tirage des tokens : 0 pour le plus probable (défaut : 0 pour
//!                    évaluer, 1 pour rl)
//! --fineweb          mesure aussi la perte sur FineWeb-Edu
//! --show N           affiche N épisodes par famille (défaut : 0 ; 3 pour show)
//! --tsv FICHIER      ajoute les résultats à ce fichier
//! --label TEXTE      étiquette des lignes du fichier
//! ```
//!
//! Fine-tuning (`sft`) :
//!
//! ```text
//! --out FICHIER      (défaut : ../data/agent/sft.safetensors)
//! --examples N       traces par famille (défaut : 2000)
//! --steps N          (défaut : 1500)
//! --batch N          lignes par lot (défaut : 32)
//! --mix X            part des lignes prises dans FineWeb-Edu (défaut : 0.5)
//! --chat X           part des traces remplacées par des conversations sans
//!                    outil du post-entraînement (défaut : 0)
//! --lr X             (défaut : 3e-5)
//! --warmup N         (défaut : 50)
//! --eval-every N     (défaut : 250)
//! --eval-tasks LISTE familles évaluées pendant l'entraînement (défaut : --tasks)
//! --thoughts         une phrase avant chaque appel
//! --recovery X       part des traces dont le premier appel reçoit une erreur
//!                    passagère, refait
//! --malformed X      part des traces dont le premier appel est mal formé, puis corrigé
//! --injected X       part des traces de ville piégées, où l'agent répond la vraie ville
//! ```
//!
//! Apprentissage par renforcement (`rl`) :
//!
//! ```text
//! --out FICHIER      (défaut : ../data/agent/rl.safetensors)
//! --steps N          (défaut : 300)
//! --prompts N        missions par pas (défaut : 16)
//! --group N          épisodes par mission (défaut : 8)
//! --lr X             (défaut : 2e-6)
//! --clip X           (défaut : 0.2)
//! --kl X             (défaut : 0.04)
//! --call-penalty X   retiré de la récompense pour chaque appel (défaut : 0)
//! --format-reward X  ajouté pour chaque appel sans erreur (défaut : 0)
//! --partial X        ajouté : X fois la part des appels de la trace idéale que
//!                    l'épisode a faits (défaut : 0)
//! --eval-every N     (défaut : 50)
//! --eval-tasks LISTE (défaut : --tasks)
//! ```

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use smallem::agent::missions::{
    EVALUATION_ALIASES, Injection, Mission, Style, TRAINING_ALIASES, Task,
};
use smallem::agent::tools::Defense;
use smallem::agent::{Ending, Episode, Report, Settings, Tokens, evaluate, run_episodes};
use smallem::autograd::Tape;
use smallem::backend::{Float, Primitives};
use smallem::chat::{self, Encoded};
use smallem::cli::{Args, at_least_one, check_positive, check_range};
use smallem::data::{Batches, Shards, bytes_per_token};
use smallem::io::{load_gpt, save_model};
use smallem::model::Gpt;
use smallem::optim::{AnyOptimizer, Optimizer, OptimizerKind, cosine_schedule};
use smallem::post_training::{
    GrpoSettings, group_advantages, grpo_loss, sft_batch, token_log_probs,
};
use smallem::sampling::Sampling;
use smallem::tasks::Family;
use smallem::tokenizer::{Bpe, TOKENIZER_FILE};
use smallem::training::{evaluate as evaluate_loss, train_step_clipped, validation_loss};
use smallem_cuda::Cuda;

const DEFAULT_MODEL: &str = "../data/post/sft.safetensors";
const DEFAULT_DATA: &str = "../data/bpe";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Show,
    Eval,
    Sft,
    Rl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Describe {
    None,
    Names,
    Aliases,
}

struct Options {
    command: Command,
    model: PathBuf,
    out: Option<PathBuf>,
    data: PathBuf,
    tasks: Vec<Task>,
    eval_tasks: Option<Vec<Task>>,
    questions: usize,
    describe: Describe,
    fail_first: bool,
    inject: Option<Injection>,
    defense: Defense,
    special: bool,
    faulty: i64,
    no_cache: bool,
    max_calls: usize,
    temperature: Option<f64>,
    fineweb: bool,
    show: Option<usize>,
    tsv: Option<PathBuf>,
    label: String,
    // Fine-tuning.
    examples: usize,
    steps: Option<usize>,
    batch: usize,
    mix: f64,
    chat: f64,
    lr: Option<f64>,
    warmup: usize,
    eval_every: Option<usize>,
    thoughts: bool,
    recovery: f64,
    malformed: f64,
    injected: f64,
    // Renforcement.
    prompts: usize,
    group: usize,
    clip: f64,
    kl: f64,
    call_penalty: f64,
    format_reward: f64,
    partial: f64,
}

fn parse_tasks(list: &str) -> Result<Vec<Task>, Box<dyn Error>> {
    list.split(',')
        .map(|name| {
            Task::parse(name.trim()).ok_or_else(|| format!("famille inconnue : {name}").into())
        })
        .collect()
}

impl Options {
    fn parse_from(mut args: Args) -> Result<Self, Box<dyn Error>> {
        let command = match args.next().as_deref() {
            Some("show") => Command::Show,
            Some("eval") => Command::Eval,
            Some("sft") => Command::Sft,
            Some("rl") => Command::Rl,
            Some(other) => return Err(format!("commande inconnue : {other}").into()),
            None => return Err("une commande : show, eval, sft ou rl".into()),
        };
        let mut o = Self {
            command,
            model: DEFAULT_MODEL.into(),
            out: None,
            data: DEFAULT_DATA.into(),
            tasks: vec![Task::Addition(4)],
            eval_tasks: None,
            questions: 200,
            describe: Describe::None,
            fail_first: false,
            inject: None,
            defense: Defense::None,
            special: false,
            faulty: 0,
            no_cache: false,
            max_calls: 8,
            temperature: None,
            fineweb: false,
            show: None,
            tsv: None,
            label: String::new(),
            examples: 2000,
            steps: None,
            batch: 32,
            mix: 0.5,
            chat: 0.0,
            lr: None,
            warmup: 50,
            eval_every: None,
            thoughts: false,
            recovery: 0.0,
            malformed: 0.0,
            injected: 0.0,
            prompts: 16,
            group: 8,
            clip: 0.2,
            kl: 0.04,
            call_penalty: 0.0,
            format_reward: 0.0,
            partial: 0.0,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model" => o.model = args.value(&flag)?,
                "--out" => o.out = Some(args.value(&flag)?),
                "--data" => o.data = args.value(&flag)?,
                "--tasks" => o.tasks = parse_tasks(&args.value::<String>(&flag)?)?,
                "--eval-tasks" => o.eval_tasks = Some(parse_tasks(&args.value::<String>(&flag)?)?),
                "--questions" => o.questions = args.value(&flag)?,
                "--describe" => {
                    o.describe = match args.value::<String>(&flag)?.as_str() {
                        "none" => Describe::None,
                        "names" => Describe::Names,
                        "aliases" => Describe::Aliases,
                        other => {
                            return Err(
                                format!("--describe {other} : none, names ou aliases").into()
                            );
                        }
                    }
                }
                "--fail-first" => o.fail_first = true,
                "--inject" => {
                    let name: String = args.value(&flag)?;
                    o.inject = Some(
                        Injection::parse(&name).ok_or(format!("injection inconnue : {name}"))?,
                    );
                }
                "--defense" => {
                    o.defense = match args.value::<String>(&flag)?.as_str() {
                        "none" => Defense::None,
                        "validate" => Defense::Validate,
                        "truncate" => Defense::Truncate,
                        other => return Err(format!("défense inconnue : {other}").into()),
                    }
                }
                "--special" => o.special = true,
                "--faulty" => o.faulty = args.value(&flag)?,
                "--no-cache" => o.no_cache = true,
                "--max-calls" => o.max_calls = args.value(&flag)?,
                "--temperature" => o.temperature = Some(args.value(&flag)?),
                "--fineweb" => o.fineweb = true,
                "--show" => o.show = Some(args.value(&flag)?),
                "--tsv" => o.tsv = Some(args.value(&flag)?),
                "--label" => o.label = args.value(&flag)?,
                "--examples" => o.examples = args.value(&flag)?,
                "--steps" => o.steps = Some(args.value(&flag)?),
                "--batch" => o.batch = args.value(&flag)?,
                "--mix" => o.mix = args.value(&flag)?,
                "--chat" => o.chat = args.value(&flag)?,
                "--lr" => o.lr = Some(args.value(&flag)?),
                "--warmup" => o.warmup = args.value(&flag)?,
                "--eval-every" => o.eval_every = Some(args.value(&flag)?),
                "--thoughts" => o.thoughts = true,
                "--recovery" => o.recovery = args.value(&flag)?,
                "--malformed" => o.malformed = args.value(&flag)?,
                "--injected" => o.injected = args.value(&flag)?,
                "--prompts" => o.prompts = args.value(&flag)?,
                "--group" => o.group = args.value(&flag)?,
                "--clip" => o.clip = args.value(&flag)?,
                "--kl" => o.kl = args.value(&flag)?,
                "--call-penalty" => o.call_penalty = args.value(&flag)?,
                "--format-reward" => o.format_reward = args.value(&flag)?,
                "--partial" => o.partial = args.value(&flag)?,
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        at_least_one(&[
            ("--questions", o.questions),
            ("--examples", o.examples),
            ("--batch", o.batch),
            ("--prompts", o.prompts),
            ("--group", o.group),
            ("--max-calls", o.max_calls),
            ("--steps", o.steps.unwrap_or(1)),
            ("--eval-every", o.eval_every.unwrap_or(1)),
        ])?;
        check_range("--mix", o.mix, 0.0..1.0)?;
        for (flag, value) in [
            ("--chat", o.chat),
            ("--recovery", o.recovery),
            ("--malformed", o.malformed),
            ("--injected", o.injected),
        ] {
            check_range(flag, value, 0.0..=1.0)?;
        }
        if let Some(lr) = o.lr {
            check_positive("--lr", lr)?;
        }
        check_positive("--clip", o.clip)?;
        check_range("--kl", o.kl, 0.0..)?;
        Ok(o)
    }

    /// Les règles de la boucle pour évaluer : le token le plus probable,
    /// sauf `--temperature` pour la commande d'évaluation.
    fn settings(&self) -> Settings {
        let mut settings = Settings {
            max_calls: self.max_calls,
            reuse_cache: !self.no_cache,
            special_in_results: self.special,
            ..Settings::default()
        };
        if let Some(t) = self.temperature
            && self.command != Command::Rl
        {
            settings.sampling.temperature = t;
        }
        settings
    }

    /// Les missions d'évaluation d'une famille, préparées selon les options.
    fn evaluation_set(&self, task: Task) -> Vec<Mission> {
        let mut rng = StdRng::seed_from_u64(77);
        let mut missions = task.evaluation_set(self.questions);
        for mission in &mut missions {
            self.prepare(mission, &mut rng, EVALUATION_ALIASES);
            if let Some(injection) = self.inject {
                mission.inject(injection, &mut rng);
            }
        }
        missions
    }

    /// Applique à une mission ce que les options demandent : outils décrits,
    /// renommés, première erreur, défense.
    fn prepare(&self, mission: &mut Mission, rng: &mut StdRng, aliases: &[&str]) {
        match self.describe {
            Describe::None => {}
            Describe::Names => mission.describe = true,
            Describe::Aliases => {
                mission.describe = true;
                mission.rename_tools(rng, aliases);
            }
        }
        mission.toolbox.fail_first = self.fail_first;
        mission.toolbox.defense = self.defense;
        mission.toolbox.add_error = self.faulty;
    }
}

/// Ce qu'il faut pour parler au modèle.
struct Context {
    gpu: Cuda,
    tokenizer: Bpe,
    tokens: Tokens,
    validation: Shards,
    bytes: f64,
}

impl Context {
    fn new(data: &PathBuf) -> Result<Self, Box<dyn Error>> {
        let gpu = smallem_cuda::gpu(true)?;
        let tokenizer = Bpe::load(data.join(TOKENIZER_FILE))?;
        let tokens = Tokens::after(tokenizer.vocab_size());
        let validation = Shards::open_dir(data, "val.bin")?;
        let bytes = bytes_per_token(&validation, &tokenizer)?;
        Ok(Self {
            gpu,
            tokenizer,
            tokens,
            validation,
            bytes,
        })
    }

    /// Charge un modèle, et lui ajoute les deux tokens d'outil s'il ne les
    /// a pas encore.
    fn load(&self, path: &PathBuf, inference: bool) -> Result<Gpt<Cuda>, Box<dyn Error>> {
        let mut model = load_gpt(&self.gpu, path, inference)?;
        if model.config().vocab < self.tokens.vocab() {
            model.grow_vocab(&self.gpu, self.tokens.vocab());
            println!(
                "vocabulaire agrandi des tokens d'outil : {}",
                self.tokens.vocab()
            );
        }
        Ok(model)
    }

    fn fineweb(&self, model: &Gpt<Cuda>, batches: usize) -> Result<f64, Box<dyn Error>> {
        let loss = validation_loss(&self.gpu, model, &self.validation, batches, 16)? / self.bytes;
        self.gpu.release_memory();
        Ok(loss)
    }
}

const HEADER: &str = "famille         exact   appel  appels  sans-err  idéaux  1er-juste  tokens  réponse/budget/longueur";

fn report_line(task: Task, report: &Report) -> String {
    let n = report.episodes as f64;
    let pct = |count: usize| 100.0 * count as f64 / n;
    let mut line = format!(
        "{:<14} {:>6.1} % {:>5.0} % {:>6.2} {:>7.1} % {:>5.1} % {:>8.1} % {:>7.1}  {}/{}/{}",
        task.name(),
        100.0 * report.accuracy(),
        pct(report.with_calls),
        report.calls_per_episode(),
        100.0 * report.good_call_rate(),
        pct(report.ideal_calls),
        pct(report.first_call_right),
        report.tokens as f64 / n,
        report.endings[Ending::Answer as usize],
        report.endings[Ending::Budget as usize],
        report.endings[Ending::Length as usize],
    );
    line += &format!("  =résultat {:.1} %", pct(report.answer_is_result));
    if report.hijacked > 0 || task == Task::City {
        line += &format!("  détournées {:.1} %", pct(report.hijacked));
    }
    line
}

fn tsv_line(label: &str, model: &str, task: Task, report: &Report) -> String {
    let n = report.episodes as f64;
    format!(
        "{label}\t{model}\t{}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.1}\t{:.1}",
        task.name(),
        report.accuracy(),
        report.with_calls as f64 / n,
        report.calls_per_episode(),
        report.good_call_rate(),
        report.ideal_calls as f64 / n,
        report.first_call_right as f64 / n,
        report.tokens as f64 / n,
        report.endings[0],
        report.endings[1],
        report.endings[2],
        report.hijacked as f64 / n,
        report.answer_is_result as f64 / n,
        report.reads_cached as f64 / n,
        report.reads_uncached as f64 / n,
    )
}

/// Un épisode, lisible : les tokens spéciaux par leur nom.
fn show_episode(context: &Context, mission: &Mission, episode: &Episode) {
    let base = context.tokenizer.vocab_size() as u32;
    let names = [
        chat::ROLE_NAMES[0],
        chat::ROLE_NAMES[1],
        chat::ROLE_NAMES[2],
        chat::TOOL_NAMES[0],
        chat::TOOL_NAMES[1],
    ];
    let mut text = String::new();
    let mut run: Vec<u16> = Vec::new();
    for &token in &episode.tokens {
        if token < base {
            run.push(token as u16);
        } else {
            text += &context.tokenizer.decode(&run);
            run.clear();
            text += &format!(" {} ", names[(token - base) as usize]);
        }
    }
    text += &context.tokenizer.decode(&run);
    let verdict = match &episode.answer {
        Some(answer) if mission.check(answer) => "juste".to_string(),
        Some(_) => format!("faux, attendu {:?}", mission.answer),
        None => format!("sans réponse ({})", episode.ending.name()),
    };
    println!(
        "  {}\n    → {verdict}",
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    );
}

fn run_eval(options: &Options) -> Result<(), Box<dyn Error>> {
    let context = Context::new(&options.data)?;
    let model = context.load(&options.model, true)?;
    let settings = options.settings();
    let mut rng = StdRng::seed_from_u64(0);
    let show = options.show.unwrap_or(if options.command == Command::Show {
        3
    } else {
        0
    });
    println!("{HEADER}");
    let mut lines = Vec::new();
    for &task in &options.tasks {
        let missions = options.evaluation_set(task);
        let start = Instant::now();
        let (report, episodes) = evaluate(
            &context.gpu,
            &model,
            &context.tokenizer,
            context.tokens,
            &missions,
            &settings,
            &mut rng,
        );
        let seconds = start.elapsed().as_secs_f64();
        context.gpu.release_memory();
        println!(
            "{}  {:.2} s, lectures {:.1} par épisode (sans cache : {:.1})",
            report_line(task, &report),
            seconds,
            report.reads_cached as f64 / report.episodes as f64,
            report.reads_uncached as f64 / report.episodes as f64,
        );
        let mut used: Vec<(&String, &usize)> = report.first_tools.iter().collect();
        used.sort_by(|a, b| b.1.cmp(a.1));
        let used: Vec<String> = used
            .iter()
            .take(5)
            .map(|(t, n)| format!("{t} {n}"))
            .collect();
        println!("    premier outil : {}", used.join(", "));
        for (mission, episode) in missions.iter().zip(&episodes).take(show) {
            show_episode(&context, mission, episode);
        }
        let model_name = options.model.display().to_string();
        lines.push(format!(
            "{}\t{seconds:.3}",
            tsv_line(&options.label, &model_name, task, &report)
        ));
    }
    if options.fineweb {
        println!(
            "FineWeb-Edu {:.4} nats par octet",
            context.fineweb(&model, 200)?
        );
    }
    if let Some(path) = &options.tsv {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        for line in lines {
            writeln!(file, "{line}")?;
        }
    }
    Ok(())
}

/// Les traces d'entraînement d'une famille : `count` missions tirées avec la
/// graine `seed`, mises en traces selon les options.
fn traces(
    options: &Options,
    context: &Context,
    task: Task,
    index: usize,
    count: usize,
    seed: u64,
) -> Vec<Encoded> {
    let mut rng = StdRng::seed_from_u64(seed * 1_000_000 + index as u64);
    (0..count)
        .map(|_| {
            let mut mission = task.sample(&mut rng, false);
            options.prepare(&mut mission, &mut rng, TRAINING_ALIASES);
            // L'erreur passagère de l'évaluation n'a pas à figurer dans les
            // traces : c'est `Style::transient_error` qui l'y met.
            mission.toolbox.fail_first = false;
            if task == Task::City && rng.random_bool(options.injected) {
                let injections = [Injection::Instruction, Injection::Answer, Injection::Roles];
                let injection = injections[rng.random_range(0..injections.len())];
                mission.inject(injection, &mut rng);
            }
            let style = Style {
                thoughts: options.thoughts,
                transient_error: !mission.steps.is_empty() && rng.random_bool(options.recovery),
                malformed_first: !mission.steps.is_empty() && rng.random_bool(options.malformed),
            };
            mission.ideal(&context.tokenizer, context.tokens, style)
        })
        .collect()
}

/// Les familles évaluées pendant un entraînement, avec l'exactitude de
/// chacune, en mode glouton.
fn training_report(
    options: &Options,
    context: &Context,
    model: &Gpt<Cuda>,
    questions: usize,
    rng: &mut StdRng,
) -> Result<String, Box<dyn Error>> {
    let tasks = options
        .eval_tasks
        .clone()
        .unwrap_or_else(|| options.tasks.clone());
    let settings = options.settings();
    let mut line = String::new();
    for task in tasks {
        let mut missions = options.evaluation_set(task);
        missions.truncate(questions);
        let (report, _) = evaluate(
            &context.gpu,
            model,
            &context.tokenizer,
            context.tokens,
            &missions,
            &settings,
            rng,
        );
        context.gpu.release_memory();
        line += &format!("  {} {:.1} %", task.name(), 100.0 * report.accuracy());
    }
    line += &format!("  FineWeb {:.4}", context.fineweb(model, 50)?);
    Ok(line)
}

fn run_sft(options: &Options) -> Result<(), Box<dyn Error>> {
    let context = Context::new(&options.data)?;
    let gpu = &context.gpu;
    let mut model = context.load(&options.model, false)?;
    let out = options
        .out
        .clone()
        .unwrap_or_else(|| "../data/agent/sft.safetensors".into());
    let steps = options.steps.unwrap_or(1500);
    let lr = options.lr.unwrap_or(3e-5);
    let eval_every = options.eval_every.unwrap_or(250);
    let train_data = Shards::open_dir(&options.data, "train.bin")?;

    let mut train: Vec<Encoded> = Vec::new();
    let mut held_out: Vec<Encoded> = Vec::new();
    for (index, &task) in options.tasks.iter().enumerate() {
        train.extend(traces(options, &context, task, index, options.examples, 1));
        held_out.extend(traces(options, &context, task, index, 50, 2));
    }
    // Des conversations sans outil, celles du post-entraînement, pour ne pas
    // oublier de répondre directement.
    let chat_count =
        (train.len() as f64 * options.chat / (1.0 - options.chat).max(1e-9)).round() as usize;
    if chat_count > 0 {
        let mut rng = StdRng::seed_from_u64(5);
        for i in 0..chat_count {
            let family = Family::ALL[i % Family::ALL.len()];
            let turn = family.sample(&mut rng, false);
            train.push(chat::encode(
                &context.tokenizer,
                context.tokens.roles,
                &turn,
            ));
        }
    }
    let longest = train
        .iter()
        .chain(&held_out)
        .map(|e| e.tokens.len())
        .max()
        .unwrap_or(1);
    let length = (longest - 1).next_multiple_of(16).max(64);
    println!(
        "{} traces d'entraînement ({} sans outil), {} de validation, lignes de {length} tokens, la plus longue {longest}",
        train.len(),
        chat_count,
        held_out.len(),
    );
    let fineweb_rows = (options.batch as f64 * options.mix).round() as usize;
    let chat_rows = options.batch - fineweb_rows;
    let mut optimizer = AnyOptimizer::<Cuda>::new(OptimizerKind::AdamW, lr, 0.0);
    let mut rng = StdRng::seed_from_u64(3);
    let mut order: Vec<usize> = Vec::new();
    let mut next = |rng: &mut StdRng| {
        if order.is_empty() {
            order = (0..train.len()).collect();
            order.shuffle(rng);
        }
        train[order.pop().expect("une trace")].clone()
    };
    let held_out_loss = |model: &Gpt<Cuda>| -> f64 {
        let mut total = 0.0;
        let chunks = held_out.chunks(64);
        let count = chunks.len();
        for chunk in chunks {
            let mut items = chunk.iter().cloned();
            let batch = sft_batch(|| items.next().unwrap(), chunk.len(), length, false, 0);
            total += evaluate_loss(gpu, model, |m, t| {
                m.masked_loss(t, &batch.tokens, &batch.targets, &batch.scored, batch.rows)
            })
            .to_f64();
        }
        total / count as f64
    };
    let start = Instant::now();
    let mut eval_rng = StdRng::seed_from_u64(11);
    let questions = options.questions.min(100);
    let mut best = held_out_loss(&model);
    println!(
        "pas     0  validation des traces {best:.4}{}",
        training_report(options, &context, &model, questions, &mut eval_rng)?
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    for step in 0..steps {
        optimizer.set_learning_rate(cosine_schedule(step, options.warmup, steps, lr, lr * 0.1));
        let batch = sft_batch(|| next(&mut rng), chat_rows, length, false, 0);
        let (mut tokens, mut targets, mut scored) = (batch.tokens, batch.targets, batch.scored);
        if fineweb_rows > 0 {
            let text = train_data.batch(&mut rng, fineweb_rows, length)?;
            let (inputs, outputs) = text.examples();
            tokens.extend(inputs);
            targets.extend(outputs);
            scored.extend(std::iter::repeat_n(true, fineweb_rows * length));
        }
        let rows = options.batch;
        let (loss, _) = train_step_clipped(gpu, &mut model, &mut optimizer, 1.0, |m, t| {
            m.masked_loss(t, &tokens, &targets, &scored, rows)
        });
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            let held = held_out_loss(&model);
            println!(
                "pas {:>5}  entraînement {:.4}  validation des traces {held:.4}  époques {:.2}  {:.0} s{}",
                step + 1,
                gpu.item(&loss).to_f64(),
                ((step + 1) * chat_rows) as f64 / train.len() as f64,
                start.elapsed().as_secs_f64(),
                training_report(options, &context, &model, questions, &mut eval_rng)?
            );
            if held < best {
                best = held;
                save_model(&out, gpu, &model, model.config().to_metadata())?;
                println!("enregistré");
            }
        }
    }
    println!(
        "meilleure validation des traces : {best:.4}, modèle dans {}",
        out.display()
    );
    Ok(())
}

fn run_rl(options: &Options) -> Result<(), Box<dyn Error>> {
    let context = Context::new(&options.data)?;
    let gpu = &context.gpu;
    let mut model = context.load(&options.model, true)?;
    let reference = context.load(&options.model, true)?;
    let out = options
        .out
        .clone()
        .unwrap_or_else(|| "../data/agent/rl.safetensors".into());
    let steps = options.steps.unwrap_or(300);
    let lr = options.lr.unwrap_or(2e-6);
    let eval_every = options.eval_every.unwrap_or(50);
    let grpo = GrpoSettings {
        clip: options.clip,
        kl: options.kl,
        per_answer: true,
    };
    let mut settings = options.settings();
    settings.sampling = Sampling {
        temperature: options.temperature.unwrap_or(1.0),
        ..Sampling::default()
    };
    let mut rng = StdRng::seed_from_u64(8);
    let mut missions_rng = StdRng::seed_from_u64(9);
    let mut eval_rng = StdRng::seed_from_u64(11);
    let questions = options.questions.min(200);
    println!(
        "pas    0{}",
        training_report(options, &context, &model, questions, &mut eval_rng)?
    );
    let mut optimizer = AnyOptimizer::<Cuda>::new(OptimizerKind::AdamW, lr, 0.0);
    // Les sommes depuis le dernier compte rendu.
    let mut sums = RlSums::default();
    let start = Instant::now();
    for step in 0..steps {
        let missions: Vec<Mission> = (0..options.prompts)
            .map(|i| {
                let task = options.tasks[i % options.tasks.len()];
                let mut mission = task.sample(&mut missions_rng, false);
                options.prepare(&mut mission, &mut missions_rng, TRAINING_ALIASES);
                mission
            })
            .collect();
        let groups = run_episodes(
            gpu,
            &model,
            &context.tokenizer,
            context.tokens,
            &missions,
            options.group,
            &settings,
            &mut rng,
        );
        let mut encoded = Vec::new();
        let mut advantages = Vec::new();
        for (mission, episodes) in missions.iter().zip(&groups) {
            let rewards: Vec<f64> = episodes
                .iter()
                .map(|episode| {
                    let right = episode.answer.as_deref().is_some_and(|a| mission.check(a));
                    sums.add(mission, episode, right);
                    let good = episode
                        .calls
                        .iter()
                        .filter(|c| !c.result.starts_with("error"))
                        .count();
                    f64::from(u8::from(right)) - options.call_penalty * episode.calls.len() as f64
                        + options.format_reward * good as f64
                        + options.partial * ideal_share(mission, episode)
                })
                .collect();
            sums.reward += rewards.iter().sum::<f64>();
            advantages.extend(group_advantages(&rewards));
            encoded.extend(episodes.iter().map(Episode::encoded));
        }
        let rows = encoded.len();
        let length = encoded
            .iter()
            .map(|e| e.tokens.len())
            .max()
            .unwrap_or(2)
            .next_multiple_of(8);
        let mut items = encoded.into_iter();
        let batch = sft_batch(|| items.next().unwrap(), rows, length, false, 0);
        let values = |model: &Gpt<Cuda>| -> Vec<f64> {
            let tape = Tape::new(gpu);
            let (lp, _) = token_log_probs(
                &tape,
                model,
                &batch.tokens,
                &batch.targets,
                &batch.scored,
                rows,
            );
            gpu.to_vec(&tape.value(lp))
                .iter()
                .map(|v| v.to_f64())
                .collect()
        };
        let reference_values = values(&reference);
        let old = values(&model);
        sums.kl += old
            .iter()
            .zip(&reference_values)
            .map(|(lp, r)| (r - lp).exp() - (r - lp) - 1.0)
            .sum::<f64>()
            / old.len().max(1) as f64;
        sums.steps += 1;
        if step % 10 == 0 {
            gpu.release_memory();
        }
        train_step_clipped(gpu, &mut model, &mut optimizer, 1.0, |m, t| {
            let (lp, owners) =
                token_log_probs(t, m, &batch.tokens, &batch.targets, &batch.scored, rows);
            grpo_loss(t, lp, &owners, &advantages, &old, &reference_values, &grpo)
        });
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            println!(
                "pas {:>4}  {}  {:.0} s",
                step + 1,
                sums.line(),
                start.elapsed().as_secs_f64()
            );
            sums = RlSums::default();
            println!(
                "pas {:>4}{}",
                step + 1,
                training_report(options, &context, &model, questions, &mut eval_rng)?
            );
        }
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_model(&out, gpu, &model, model.config().to_metadata())?;
    println!("modèle enregistré dans {}", out.display());
    Ok(())
}

/// La part des appels de la trace idéale que l'épisode a faits, chacun
/// compté une fois : une récompense partielle, qui note le chemin et non
/// seulement l'arrivée.
fn ideal_share(mission: &Mission, episode: &Episode) -> f64 {
    let ideal: Vec<String> = mission.steps.iter().map(|s| mission.call_text(s)).collect();
    if ideal.is_empty() {
        return 0.0;
    }
    let mut made: Vec<&str> = episode.calls.iter().map(|c| c.text.as_str()).collect();
    let mut found = 0;
    for call in &ideal {
        if let Some(i) = made.iter().position(|m| m == call) {
            made.remove(i);
            found += 1;
        }
    }
    found as f64 / ideal.len() as f64
}

/// Ce que l'apprentissage par renforcement suit entre deux comptes rendus.
#[derive(Default)]
struct RlSums {
    episodes: usize,
    right: usize,
    reward: f64,
    calls: usize,
    tokens: usize,
    kl: f64,
    steps: usize,
    /// Justes, avec exactement les appels de la trace idéale.
    right_ideal: usize,
    /// Justes avec moins d'appels que la trace idéale : la réponse a été
    /// trouvée sans le chemin voulu.
    right_shortcut: usize,
    endings: [usize; 3],
}

impl RlSums {
    fn add(&mut self, mission: &Mission, episode: &Episode, right: bool) {
        self.episodes += 1;
        self.calls += episode.calls.len();
        self.tokens += episode.written.iter().filter(|&&w| w).count();
        self.endings[episode.ending as usize] += 1;
        if right {
            self.right += 1;
            let ideal: Vec<String> = mission.steps.iter().map(|s| mission.call_text(s)).collect();
            let made: Vec<&str> = episode.calls.iter().map(|c| c.text.as_str()).collect();
            if made == ideal {
                self.right_ideal += 1;
            }
            if episode.calls.len() < mission.steps.len() {
                self.right_shortcut += 1;
            }
        }
    }

    fn line(&self) -> String {
        let n = self.episodes.max(1) as f64;
        format!(
            "récompense {:.3}  justes {:.1} % (dont idéales {:.1} %, raccourcis {})  appels {:.2}  tokens écrits {:.1}  divergence {:.4}  fins {}/{}/{}",
            self.reward / n,
            100.0 * self.right as f64 / n,
            100.0 * self.right_ideal as f64 / self.right.max(1) as f64,
            self.right_shortcut,
            self.calls as f64 / n,
            self.tokens as f64 / n,
            self.kl / self.steps.max(1) as f64,
            self.endings[0],
            self.endings[1],
            self.endings[2],
        )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse_from(Args::from_env())?;
    match options.command {
        Command::Show | Command::Eval => run_eval(&options),
        Command::Sft => run_sft(&options),
        Command::Rl => run_rl(&options),
    }
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
    fn options_are_read_from_the_command_line() {
        let o = options(&[
            "eval",
            "--tasks",
            "add2,ages3,city",
            "--inject",
            "roles",
            "--no-cache",
        ])
        .unwrap();
        assert_eq!(o.command, Command::Eval);
        assert_eq!(
            o.tasks,
            [Task::Addition(2), Task::ages(3).unwrap(), Task::City]
        );
        assert_eq!(o.inject, Some(Injection::Roles));
        assert!(!o.settings().reuse_cache);
    }

    #[test]
    fn bad_options_are_refused() {
        for (refused, word) in [
            (&["fly"][..], "fly"),
            (&["eval", "--tasks", "chess"], "chess"),
            (&["sft", "--mix", "1"], "--mix"),
            (&["rl", "--group", "0"], "--group"),
            (&["eval", "--describe", "all"], "all"),
        ] {
            let error = options(refused).err().expect(word).to_string();
            assert!(error.contains(word), "{error}");
        }
    }
}
