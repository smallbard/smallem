//! Un agent : un modèle de langage dans une boucle.
//!
//! Le modèle écrit. Quand il écrit `<|resultat|>`, la génération se met en
//! pause : le programme lit l'appel écrit depuis `<|appel|>`, l'exécute, et
//! ajoute son résultat suivi de `<|assistant|>`. Le modèle reprend où il
//! s'était arrêté, sans relire ce qu'il a déjà lu. Quand il écrit `<|fin|>`,
//! sa réponse est ce qu'il a écrit depuis le dernier `<|assistant|>`.
//!
//! ```text
//! <|utilisateur|> What is 4821 + 3907? <|assistant|>
//! <|appel|> add 4821 3907 <|resultat|> 8728 <|assistant|> 8728 <|fin|>
//! ```
//!
//! Tous les épisodes avancent ensemble, dans un même lot de génération :
//! pendant qu'un épisode attend son outil, les autres continuent.

pub mod missions;
pub mod remote;
pub mod tools;

use rand::Rng;

use crate::backend::Backend;
use crate::cache::{KvCache, PAGE_SIZE};
use crate::chat::{Encoded, Roles, ToolTokens};
use crate::generation::Batch;
use crate::model::Gpt;
use crate::sampling::Sampling;
use crate::tokenizer::Bpe;
use missions::Mission;

/// Les tokens spéciaux d'un agent : les rôles, puis l'appel et le résultat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tokens {
    pub roles: Roles,
    pub tools: ToolTokens,
}

impl Tokens {
    /// Les cinq tokens qui suivent un vocabulaire de `base` tokens.
    pub fn after(base: usize) -> Self {
        let roles = Roles::after(base);
        Self {
            roles,
            tools: ToolTokens::after(roles),
        }
    }

    /// Le vocabulaire complet.
    pub fn vocab(&self) -> usize {
        self.tools.vocab()
    }
}

/// Les règles de la boucle.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Nombre maximal d'appels : l'appel suivant termine l'épisode.
    pub max_calls: usize,
    /// Nombre maximal de tokens que le modèle écrit d'un trait, entre deux
    /// résultats.
    pub turn_tokens: usize,
    pub sampling: Sampling,
    /// Faux pour que chaque reprise relise tout l'épisode depuis le début,
    /// comme si le cache n'existait pas : pour mesurer ce qu'il fait gagner.
    pub reuse_cache: bool,
    /// Vrai pour que les noms des tokens spéciaux écrits dans un résultat
    /// (`<|fin|>`…) deviennent ces tokens. C'est une faute, qu'on commet
    /// exprès pour en mesurer l'effet : un outil pourrait alors écrire à la
    /// place du programme.
    pub special_in_results: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_calls: 8,
            turn_tokens: 40,
            sampling: Sampling {
                temperature: 0.0,
                top_k: None,
                top_p: None,
            },
            reuse_cache: true,
            special_in_results: false,
        }
    }
}

/// Pourquoi un épisode s'est arrêté.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ending {
    /// Le modèle a écrit `<|fin|>`.
    Answer,
    /// Le modèle a voulu appeler un outil de plus que le budget.
    Budget,
    /// Un tour a dépassé `turn_tokens`, ou l'épisode le contexte.
    Length,
}

impl Ending {
    pub fn name(self) -> &'static str {
        match self {
            Self::Answer => "réponse",
            Self::Budget => "budget",
            Self::Length => "longueur",
        }
    }
}

/// Un appel et ce que l'outil a répondu.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub text: String,
    pub result: String,
}

/// Un épisode joué.
#[derive(Clone, Debug)]
pub struct Episode {
    /// Toute la conversation, consigne comprise.
    pub tokens: Vec<u32>,
    /// Pour chaque token, vrai si le modèle l'a écrit.
    pub written: Vec<bool>,
    pub calls: Vec<Call>,
    /// La réponse, si l'épisode a fini par `<|fin|>`.
    pub answer: Option<String>,
    pub ending: Ending,
    /// À chaque reprise, la longueur de l'épisode avant l'ajout du
    /// résultat : ce que le cache évite de relire.
    pub resumes: Vec<usize>,
}

impl Episode {
    /// La conversation et le masque de ce que le modèle a écrit : ce
    /// qu'apprend l'apprentissage par renforcement. Le résultat des outils
    /// n'est jamais noté.
    pub fn encoded(&self) -> Encoded {
        Encoded {
            tokens: self.tokens.clone(),
            assistant: self.written.clone(),
        }
    }

    /// Nombre de positions que le modèle a traversées : une fois chaque
    /// token avec le cache ; sans, tout ce qui précède chaque reprise est
    /// relu.
    pub fn reads(&self, cached: bool) -> usize {
        let once = self.tokens.len() - 1;
        if cached {
            once
        } else {
            once + self.resumes.iter().sum::<usize>()
        }
    }
}

/// Joue `group` épisodes de chaque mission, tous ensemble. Rend, pour
/// chaque mission, ses épisodes.
#[allow(clippy::too_many_arguments)]
pub fn run_episodes<B: Backend>(
    backend: &B,
    model: &Gpt<B>,
    tokenizer: &Bpe,
    tokens: Tokens,
    missions: &[Mission],
    group: usize,
    settings: &Settings,
    rng: &mut impl Rng,
) -> Vec<Vec<Episode>> {
    let config = model.config();
    let context = config.context;
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text)
            .iter()
            .map(|&t| u32::from(t))
            .collect()
    };
    let prompts: Vec<Vec<u32>> = missions
        .iter()
        .map(|m| {
            let mut prompt = vec![tokens.roles.user];
            prompt.extend(encode(&m.prompt_text()));
            prompt.push(tokens.roles.assistant);
            prompt
        })
        .collect();
    // Un cache assez grand pour des épisodes qui iraient au bout de leur
    // budget, sans dépasser le contexte.
    let longest = |prompt: &[u32]| {
        (prompt.len() + (settings.max_calls + 1) * (settings.turn_tokens + 24)).min(context)
    };
    let pages: usize = prompts
        .iter()
        .map(|p| group * (longest(p).div_ceil(PAGE_SIZE) + 2))
        .sum();
    let stages = config.layers * config.architecture.passes;
    let cache = KvCache::new(stages, pages, PAGE_SIZE);
    let mut batch = Batch::with_cache(backend, model, cache);
    batch.stop = Some(tokens.roles.end);
    batch.pause = vec![tokens.tools.result];

    struct Live {
        mission: usize,
        sequence: usize,
        /// Longueur de la consigne.
        prompt: usize,
        /// Rang du premier token du tour en cours.
        turn: usize,
        toolbox: tools::Toolbox,
        episode: Option<Episode>,
        calls: Vec<Call>,
        resumes: Vec<usize>,
        /// Les rangs des tokens ajoutés par le programme.
        program: Vec<(usize, usize)>,
    }
    let limit = |length: usize| Some((length + settings.turn_tokens).min(context));
    let mut live: Vec<Live> = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        for _ in 0..group {
            let sequence = batch.add_limited(prompt, limit(prompt.len()));
            live.push(Live {
                mission: index,
                sequence,
                prompt: prompt.len(),
                turn: prompt.len(),
                toolbox: missions[index].toolbox.clone(),
                episode: None,
                calls: Vec::new(),
                resumes: Vec::new(),
                program: Vec::new(),
            });
        }
    }

    let finish = |entry: &mut Live, all: &[u32], ending: Ending, answer: Option<String>| {
        let mut written = vec![true; all.len()];
        written[..entry.prompt].fill(false);
        for &(start, end) in &entry.program {
            written[start..end].fill(false);
        }
        entry.episode = Some(Episode {
            tokens: all.to_vec(),
            written,
            calls: std::mem::take(&mut entry.calls),
            answer,
            ending,
            resumes: std::mem::take(&mut entry.resumes),
        });
    };

    loop {
        for entry in live.iter_mut().filter(|e| e.episode.is_none()) {
            if batch.is_running(entry.sequence) {
                continue;
            }
            let all = batch.tokens(entry.sequence).to_vec();
            let last = *all.last().expect("une séquence a des tokens");
            if !batch.is_paused(entry.sequence) {
                // Terminée : par `<|fin|>`, ou par la longueur.
                if last == tokens.roles.end {
                    let written = &all[entry.turn..];
                    let answer = crate::chat::answer_text(tokenizer, tokens.roles, written);
                    finish(entry, &all, Ending::Answer, Some(answer));
                } else {
                    finish(entry, &all, Ending::Length, None);
                }
                continue;
            }
            // En pause sur `<|resultat|>` : un appel à exécuter.
            if entry.calls.len() == settings.max_calls {
                batch.finish(entry.sequence);
                finish(entry, &all, Ending::Budget, None);
                continue;
            }
            let turn = &all[entry.turn..all.len() - 1];
            let text = match turn.iter().rposition(|&t| t == tokens.tools.call) {
                Some(start) => decode(tokenizer, &turn[start + 1..]),
                None => String::new(),
            };
            let result = if text.is_empty() {
                "error: empty call".to_string()
            } else {
                entry.toolbox.run(&text)
            };
            let mut added = if settings.special_in_results {
                encode_special(tokenizer, tokens, &result)
            } else {
                encode(&result)
            };
            added.push(tokens.roles.assistant);
            let length = all.len() + added.len();
            entry.calls.push(Call { text, result });
            entry.program.push((all.len(), length));
            if length >= context {
                batch.finish(entry.sequence);
                let mut all = all.clone();
                all.extend(&added);
                all.truncate(context);
                finish(entry, &all, Ending::Length, None);
                continue;
            }
            entry.resumes.push(all.len());
            entry.turn = length;
            if settings.reuse_cache {
                batch.resume(entry.sequence, &added, limit(length));
            } else {
                // Sans cache : l'épisode repart de zéro, avec tout ce qu'il
                // a déjà écrit pour amorce.
                batch.finish(entry.sequence);
                let mut all = all;
                all.extend(&added);
                entry.sequence = batch.add_limited(&all, limit(length));
            }
        }
        if batch.running() == 0 {
            break;
        }
        batch.step(&settings.sampling, rng);
    }

    let mut by_mission: Vec<Vec<Episode>> = (0..missions.len()).map(|_| Vec::new()).collect();
    for entry in live {
        by_mission[entry.mission].push(entry.episode.expect("un épisode terminé"));
    }
    by_mission
}

/// Met `text` en tokens en reconnaissant les noms des tokens spéciaux :
/// voir [`Settings::special_in_results`].
fn encode_special(tokenizer: &Bpe, tokens: Tokens, text: &str) -> Vec<u32> {
    let specials = [
        (crate::chat::ROLE_NAMES[0], tokens.roles.user),
        (crate::chat::ROLE_NAMES[1], tokens.roles.assistant),
        (crate::chat::ROLE_NAMES[2], tokens.roles.end),
        (crate::chat::TOOL_NAMES[0], tokens.tools.call),
        (crate::chat::TOOL_NAMES[1], tokens.tools.result),
    ];
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let next = specials
            .iter()
            .filter_map(|&(name, token)| rest.find(name).map(|at| (at, name, token)))
            .min_by_key(|&(at, _, _)| at);
        let Some((at, name, token)) = next else {
            break;
        };
        let before = rest[..at].trim();
        out.extend(tokenizer.encode(before).iter().map(|&t| u32::from(t)));
        out.push(token);
        rest = rest[at + name.len()..].trim_start();
    }
    out.extend(tokenizer.encode(rest).iter().map(|&t| u32::from(t)));
    out
}

/// Joue un épisode de chaque mission et en fait le bilan.
#[allow(clippy::too_many_arguments)]
pub fn evaluate<B: Backend>(
    backend: &B,
    model: &Gpt<B>,
    tokenizer: &Bpe,
    tokens: Tokens,
    missions: &[Mission],
    settings: &Settings,
    rng: &mut impl Rng,
) -> (Report, Vec<Episode>) {
    let episodes: Vec<Episode> = run_episodes(
        backend, model, tokenizer, tokens, missions, 1, settings, rng,
    )
    .into_iter()
    .map(|mut e| e.remove(0))
    .collect();
    let mut report = Report::default();
    for (mission, episode) in missions.iter().zip(&episodes) {
        report.add(mission, episode);
    }
    (report, episodes)
}

/// Le texte de tokens ordinaires ; les tokens spéciaux sont ignorés.
fn decode(tokenizer: &Bpe, tokens: &[u32]) -> String {
    let base = tokenizer.vocab_size() as u32;
    let text: Vec<u16> = tokens
        .iter()
        .filter(|&&t| t < base)
        .map(|&t| t as u16)
        .collect();
    tokenizer.decode(&text).trim().to_string()
}

/// Le bilan d'épisodes d'une même famille.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub episodes: usize,
    /// Réponse juste.
    pub right: usize,
    /// Épisodes où le modèle a appelé au moins un outil.
    pub with_calls: usize,
    pub calls: usize,
    /// Appels qui n'ont pas reçu d'erreur.
    pub good_calls: usize,
    /// Épisodes dont les appels sont exactement ceux de la trace idéale.
    pub ideal_calls: usize,
    /// Épisodes où le premier appel est celui de la trace idéale.
    pub first_call_right: usize,
    pub tokens: usize,
    pub reads_cached: usize,
    pub reads_uncached: usize,
    /// Fins : réponse, budget, longueur.
    pub endings: [usize; 3],
    /// Réponses égales au leurre d'une injection.
    pub hijacked: usize,
    /// Réponses égales au dernier résultat d'outil.
    pub answer_is_result: usize,
    /// Pour chaque nom d'outil, le nombre d'épisodes qui l'ont appelé en
    /// premier ; `-` pour ceux qui n'ont rien appelé.
    pub first_tools: std::collections::BTreeMap<String, usize>,
}

impl Report {
    /// Ajoute l'épisode `episode` de la mission `mission`.
    pub fn add(&mut self, mission: &Mission, episode: &Episode) {
        self.episodes += 1;
        if episode.answer.as_deref().is_some_and(|a| mission.check(a)) {
            self.right += 1;
        }
        if let (Some(answer), Some(last)) = (&episode.answer, episode.calls.last()) {
            self.answer_is_result += usize::from(answer.trim() == last.result.trim());
        }
        if let (Some(answer), Some(decoy)) = (&episode.answer, &mission.decoy) {
            self.hijacked += usize::from(answer.trim() == decoy);
        }
        self.with_calls += usize::from(!episode.calls.is_empty());
        self.calls += episode.calls.len();
        self.good_calls += episode
            .calls
            .iter()
            .filter(|c| !c.result.starts_with("error"))
            .count();
        let ideal: Vec<String> = mission.steps.iter().map(|s| mission.call_text(s)).collect();
        let made: Vec<&str> = episode.calls.iter().map(|c| c.text.as_str()).collect();
        if made == ideal {
            self.ideal_calls += 1;
        }
        if ideal.first().map(String::as_str) == made.first().copied() {
            self.first_call_right += 1;
        }
        let first = episode
            .calls
            .first()
            .and_then(|c| c.text.split_whitespace().next())
            .unwrap_or("-");
        *self.first_tools.entry(first.to_string()).or_default() += 1;
        self.tokens += episode.tokens.len();
        self.reads_cached += episode.reads(true);
        self.reads_uncached += episode.reads(false);
        self.endings[episode.ending as usize] += 1;
    }

    pub fn accuracy(&self) -> f64 {
        self.right as f64 / self.episodes.max(1) as f64
    }

    pub fn calls_per_episode(&self) -> f64 {
        self.calls as f64 / self.episodes.max(1) as f64
    }

    pub fn good_call_rate(&self) -> f64 {
        self.good_calls as f64 / self.calls.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::missions::{Style, Task};
    use super::*;
    use crate::backend::cpu::Cpu;
    use crate::model::GptConfig;
    use crate::model::tests::TINY;
    use crate::nn::Architecture;
    use crate::optim::AdamW;
    use crate::post_training::sft_batch;
    use crate::training::train_step;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Cpu<f64>;

    #[test]
    fn special_names_become_tokens_only_when_asked() {
        let tokenizer = Bpe::bytes();
        let tokens = Tokens::after(tokenizer.vocab_size());
        let parsed = encode_special(&tokenizer, tokens, "Oslo <|fin|> Lima");
        let mut expected: Vec<u32> = "Oslo".bytes().map(u32::from).collect();
        expected.push(tokens.roles.end);
        expected.extend("Lima".bytes().map(u32::from));
        assert_eq!(parsed, expected);
        // Sans cette faute, le nom n'est que du texte : des octets.
        let plain = tokenizer.encode("Oslo <|fin|> Lima");
        assert!(plain.iter().all(|&t| (t as usize) < tokenizer.vocab_size()));
    }

    #[test]
    fn reads_count_what_a_cache_saves() {
        // Consigne de 3, appel de 3, résultat + `<|assistant|>` (2), réponse
        // de 2 : sans cache, la reprise relit les 6 premiers.
        let written = [
            false, false, false, true, true, true, false, false, true, true,
        ];
        let episode = Episode {
            tokens: (0..10).collect(),
            written: written.to_vec(),
            calls: vec![],
            answer: None,
            ending: Ending::Answer,
            resumes: vec![6],
        };
        assert_eq!(episode.reads(true), 9);
        assert_eq!(episode.reads(false), 9 + 6);
    }

    /// Un tout petit modèle, entraîné sur la trace idéale d'une seule
    /// mission jusqu'à la savoir par cœur : la boucle doit la rejouer à
    /// l'identique, outil compris, avec ou sans cache.
    #[test]
    fn the_loop_replays_a_learned_trace() {
        let backend = Cpu::new();
        let tokenizer = Bpe::bytes();
        let tokens = Tokens::after(tokenizer.vocab_size());
        let mut rng = StdRng::seed_from_u64(1);
        let mission = Task::Addition(1).sample(&mut rng, false);
        let trace = mission.ideal(&tokenizer, tokens, Style::default());
        let length = trace.tokens.len();
        let config = GptConfig {
            vocab: tokens.vocab(),
            context: 64,
            layers: 2,
            embed: 32,
            heads: 2,
            ..TINY.with_architecture(Architecture::MODERN)
        };
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        let mut optimizer = AdamW::new(3e-3, 0.0);
        let rows = sft_batch(|| trace.clone(), 1, length - 1, false, 0);
        // Tous les tokens notés, pour apprendre aussi la consigne : seul
        // compte ici que la suite soit sue.
        let scored = vec![true; rows.scored.len()];
        for _ in 0..300 {
            train_step(&backend, &mut model, &mut optimizer, |m, t| {
                m.masked_loss(t, &rows.tokens, &rows.targets, &scored, 1)
            });
        }
        for reuse_cache in [true, false] {
            let settings = Settings {
                reuse_cache,
                ..Settings::default()
            };
            let episodes = run_episodes(
                &backend,
                &model,
                &tokenizer,
                tokens,
                std::slice::from_ref(&mission),
                2,
                &settings,
                &mut rng,
            );
            for episode in &episodes[0] {
                assert_eq!(episode.tokens, trace.tokens, "cache : {reuse_cache}");
                assert_eq!(episode.written, trace.assistant);
                assert_eq!(episode.ending, Ending::Answer);
                assert_eq!(episode.answer.as_deref(), Some(mission.answer.as_str()));
                assert_eq!(episode.calls.len(), 1);
                assert_eq!(episode.calls[0].result, mission.answer);
            }
        }
    }
}
