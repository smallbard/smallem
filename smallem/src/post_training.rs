//! Le post-entraînement : évaluer un modèle sur des tâches, et préparer ce
//! qu'apprennent le fine-tuning supervisé, DPO et GRPO.

use rand::Rng;

use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};
use crate::cache::{KvCache, PAGE_SIZE};
use crate::chat::{self, Encoded, Roles, Turn};
use crate::generation::Batch;
use crate::model::Gpt;
use crate::sampling::Sampling;
use crate::tasks::Family;
use crate::tokenizer::Bpe;

/// Comment poser les questions au modèle.
#[derive(Clone, Debug)]
pub enum Asking {
    /// En conversation, avec les tokens de rôle : la réponse s'arrête au
    /// `<|fin|>`.
    Chat(Roles),
    /// Avec quelques exemples dans la consigne, sans tokens de rôle : la
    /// manière d'interroger un modèle qui n'a appris qu'à continuer un
    /// texte. La réponse s'arrête au premier saut de ligne.
    FewShot(Vec<Turn>),
}

impl Asking {
    /// Ce que le modèle lit avant de répondre à `question`.
    pub fn prompt(&self, tokenizer: &Bpe, question: &str) -> Vec<u32> {
        match self {
            Self::Chat(roles) => chat::prompt(tokenizer, *roles, question),
            Self::FewShot(examples) => {
                let mut text = String::new();
                for example in examples {
                    text += &format!(
                        "Question: {}\nAnswer: {}\n\n",
                        example.question, example.answer
                    );
                }
                text += &format!("Question: {question}\nAnswer:");
                tokenizer
                    .encode(&text)
                    .iter()
                    .map(|&t| u32::from(t))
                    .collect()
            }
        }
    }

    /// Le texte de la réponse dans les tokens générés.
    pub fn answer(&self, tokenizer: &Bpe, generated: &[u32]) -> String {
        match self {
            Self::Chat(roles) => chat::answer_text(tokenizer, *roles, generated),
            Self::FewShot(_) => {
                let text: Vec<u16> = generated.iter().map(|&t| t as u16).collect();
                let text = tokenizer.decode(&text);
                text.split('\n').next().unwrap_or("").trim().to_string()
            }
        }
    }
}

/// Les réponses du modèle à `turns`, toutes générées ensemble par un lot.
/// Avec `sampling`, chaque question reçoit `answers` réponses tirées ;
/// rend, pour chaque question, ses réponses et leurs tokens.
#[allow(clippy::too_many_arguments)]
pub fn answer_all<B: Backend>(
    backend: &B,
    model: &Gpt<B>,
    tokenizer: &Bpe,
    asking: &Asking,
    turns: &[Turn],
    answers: usize,
    max_tokens: usize,
    sampling: &Sampling,
    rng: &mut impl Rng,
) -> Vec<Vec<(String, Vec<u32>)>> {
    let prompts: Vec<Vec<u32>> = turns
        .iter()
        .map(|turn| asking.prompt(tokenizer, &turn.question))
        .collect();
    // Un cache juste assez grand pour ces amorces et ces réponses, plutôt
    // que pour des séquences de tout le contexte.
    let config = model.config();
    let pages: usize = prompts
        .iter()
        .map(|prompt| answers * ((prompt.len() + max_tokens).div_ceil(PAGE_SIZE) + 1))
        .sum();
    let stages = config.layers * config.architecture.passes;
    let cache = KvCache::new(stages, pages, PAGE_SIZE);
    let mut batch = Batch::with_cache(backend, model, cache);
    if let Asking::Chat(roles) = asking {
        batch.stop = Some(roles.end);
    }
    let mut numbers = Vec::new();
    for prompt in &prompts {
        for _ in 0..answers {
            numbers.push(batch.add_limited(prompt, Some(prompt.len() + max_tokens)));
        }
    }
    while batch.running() > 0 {
        batch.step(sampling, rng);
    }
    prompts
        .iter()
        .enumerate()
        .map(|(question, prompt)| {
            (0..answers)
                .map(|i| {
                    let tokens =
                        batch.tokens(numbers[question * answers + i])[prompt.len()..].to_vec();
                    (asking.answer(tokenizer, &tokens), tokens)
                })
                .collect()
        })
        .collect()
}

/// La part des questions de `turns` auxquelles le modèle répond juste, en
/// mode glouton.
pub fn accuracy<B: Backend>(
    backend: &B,
    model: &Gpt<B>,
    tokenizer: &Bpe,
    asking: &Asking,
    turns: &[Turn],
    rng: &mut impl Rng,
) -> f64 {
    let greedy = Sampling {
        temperature: 0.0,
        top_k: None,
        top_p: None,
    };
    let answers = answer_all(
        backend, model, tokenizer, asking, turns, 1, 40, &greedy, rng,
    );
    let right = turns
        .iter()
        .zip(&answers)
        .filter(|(turn, answers)| Family::check(turn, &answers[0].0))
        .count();
    right as f64 / turns.len() as f64
}

/// Un lot du fine-tuning supervisé : `rows` lignes de `length` tokens,
/// leurs cibles, et les positions notées.
pub struct SftBatch {
    pub tokens: Vec<u32>,
    pub targets: Vec<u32>,
    pub scored: Vec<bool>,
    pub rows: usize,
}

/// Range des conversations en `rows` lignes de `length` tokens.
///
/// Avec `packed`, les conversations sont mises bout à bout, autant qu'il en
/// tient dans une ligne : chacune peut alors regarder celles qui la
/// précèdent dans sa ligne. Sinon, une conversation par ligne, et le reste
/// de la ligne est rempli de `padding`, jamais noté : l'attention étant
/// causale, ces tokens de remplissage, placés après la conversation, ne
/// changent rien à son calcul.
///
/// `conversations` fournit les conversations une à une.
pub fn sft_batch(
    mut conversations: impl FnMut() -> Encoded,
    rows: usize,
    length: usize,
    packed: bool,
    padding: u32,
) -> SftBatch {
    let mut inputs = Vec::with_capacity(rows * length);
    let mut targets = Vec::with_capacity(rows * length);
    let mut scored = Vec::with_capacity(rows * length);
    let mut pending: Option<Encoded> = None;
    for _ in 0..rows {
        // Une ligne de `length + 1` tokens : les entrées sont les `length`
        // premiers, les cibles les `length` derniers.
        let mut row = Vec::with_capacity(length + 1);
        let mut written = Vec::with_capacity(length + 1);
        loop {
            let next = pending.take().unwrap_or_else(&mut conversations);
            assert!(
                next.tokens.len() <= length + 1,
                "conversation plus longue qu'une ligne"
            );
            if !row.is_empty() && row.len() + next.tokens.len() > length + 1 {
                pending = Some(next);
                break;
            }
            row.extend(&next.tokens);
            written.extend(&next.assistant);
            if !packed {
                break;
            }
        }
        row.resize(length + 1, padding);
        written.resize(length + 1, false);
        inputs.extend(&row[..length]);
        targets.extend(&row[1..]);
        // Une position est notée si sa cible a été écrite par l'assistant.
        scored.extend(&written[1..]);
    }
    SftBatch {
        tokens: inputs,
        targets,
        scored,
        rows,
    }
}

/// La somme des log-probabilités des cibles notées de chaque séquence,
/// sachant ce qui les précède : la log-probabilité d'une réponse. Les
/// séquences sont `[batch, T]`, mises bout à bout ; `scored` marque les
/// positions dont la cible compte. Rend `[batch]`.
pub fn sequence_log_probs<B: Backend>(
    tape: &Tape<B>,
    model: &Gpt<B>,
    tokens: &[u32],
    targets: &[u32],
    scored: &[bool],
    batch: usize,
) -> Var {
    let (per_position, owners) = token_log_probs(tape, model, tokens, targets, scored, batch);
    // La somme par séquence, en un produit : une matrice `[batch, n]` de 0
    // et de 1, dont la ligne d'une séquence marque ses positions notées.
    let n = owners.len();
    let mut owner = vec![B::Scalar::ZERO; batch * n];
    for (i, &sequence) in owners.iter().enumerate() {
        owner[sequence * n + i] = B::Scalar::ONE;
    }
    let owner = tape.constant(tape.backend().from_vec(owner, &[batch, n]));
    let per_position = tape.reshape(per_position, &[n, 1]);
    tape.reshape(tape.matmul(owner, per_position), &[batch])
}

/// La perte de DPO (Rafailov et al., 2023) sur `pairs` paires de réponses.
///
/// `log_probs` contient, `[2 × pairs]`, les log-probabilités des réponses
/// sous le modèle qu'on entraîne : les réponses préférées d'abord, puis les
/// rejetées, dans le même ordre. `reference` contient les mêmes sous le
/// modèle de référence, gelé. Pour chaque paire :
///
/// ```text
/// z     = β × [ (ln π(y⁺) − ln π_ref(y⁺)) − (ln π(y⁻) − ln π_ref(y⁻)) ]
/// perte = −ln σ(z) = ln(1 + e^(−z))
/// ```
///
/// La perte pousse la réponse préférée à gagner en probabilité par rapport
/// à la rejetée, chacune mesurée par rapport à la référence. `β` règle la
/// force du lien à la référence : plus il est petit, plus le modèle peut
/// s'en éloigner pour creuser l'écart.
pub fn dpo_loss<B: Backend>(tape: &Tape<B>, log_probs: Var, reference: &[f64], beta: f64) -> Var {
    let backend = tape.backend();
    let pairs = reference.len() / 2;
    assert_eq!(
        tape.shape(log_probs),
        [2 * pairs],
        "deux log-probabilités par paire"
    );
    let reference = backend.from_vec(
        reference.iter().map(|&r| B::Scalar::from_f64(r)).collect(),
        &[2, pairs],
    );
    let ratios = tape.sub(
        tape.reshape(log_probs, &[2, pairs]),
        tape.constant(reference),
    );
    let preferred = tape.gather_rows(ratios, &[0]);
    let rejected = tape.gather_rows(ratios, &[1]);
    let z = tape.mul(
        tape.sub(preferred, rejected),
        tape.scalar(B::Scalar::from_f64(beta)),
    );
    // `ln(1 + e^(−z))` : sans danger tant que `z` reste modéré, ce que
    // garantit un `β` petit.
    let softplus = tape.ln(tape.add(tape.exp(tape.neg(z)), tape.scalar(B::Scalar::ONE)));
    tape.mean(tape.reshape(softplus, &[pairs]), 0, false)
}

/// Les log-probabilités des cibles notées de `batch` séquences, une par
/// position notée, `[n]`, et pour chacune le rang de sa séquence.
pub fn token_log_probs<B: Backend>(
    tape: &Tape<B>,
    model: &Gpt<B>,
    tokens: &[u32],
    targets: &[u32],
    scored: &[bool],
    batch: usize,
) -> (Var, Vec<usize>) {
    let length = tokens.len() / batch;
    let vocab = model.config().vocab;
    let logits = model.logits(tape, tokens, batch);
    let flat = tape.reshape(logits, &[tokens.len(), vocab]);
    let rows: Vec<u32> = (0..tokens.len() as u32)
        .filter(|&i| scored[i as usize])
        .collect();
    let picked = target_log_probs(tape, tape.log_softmax(flat), &rows, targets, vocab);
    let owners = rows.iter().map(|&row| row as usize / length).collect();
    (picked, owners)
}

/// La log-probabilité de la cible de chaque ligne de `rows`, `[n]`, prise
/// dans `log_probs`, `[positions, vocab]`. Vue comme une seule colonne de
/// `positions × vocab` lignes, la table donne chacune par un `gather_rows`,
/// sans matrice de sélection.
fn target_log_probs<B: Backend>(
    tape: &Tape<B>,
    log_probs: Var,
    rows: &[u32],
    targets: &[u32],
    vocab: usize,
) -> Var {
    let positions = tape.shape(log_probs)[0];
    assert!(
        positions * vocab <= u32::MAX as usize,
        "trop de log-probabilités pour des indices sur 32 bits"
    );
    let column = tape.reshape(log_probs, &[positions * vocab, 1]);
    let indices: Vec<u32> = rows
        .iter()
        .map(|&row| row * vocab as u32 + targets[row as usize])
        .collect();
    tape.reshape(tape.gather_rows(column, &indices), &[rows.len()])
}

/// Les réglages de la perte de GRPO.
#[derive(Clone, Copy, Debug)]
pub struct GrpoSettings {
    /// `ε` : le rapport des probabilités est borné à `[1 − ε, 1 + ε]`.
    pub clip: f64,
    /// `β` : le poids de la pénalité de divergence avec la référence.
    pub kl: f64,
    /// Vrai pour moyenner chaque réponse sur ses tokens avant de moyenner
    /// les réponses, comme l'article ; faux pour moyenner tous les tokens
    /// ensemble, ce qui ne favorise plus les réponses courtes.
    pub per_answer: bool,
}

/// La perte de GRPO (Shao et al., 2024) sur des réponses dont `log_probs`
/// donne les log-probabilités des tokens, `[n]`, `owners` la réponse de
/// chaque token, et `advantages` l'avantage de chaque réponse. `old` est la
/// même chose sous le modèle qui a tiré les réponses, `reference` sous le
/// modèle de référence, gelé. Pour chaque token :
///
/// ```text
/// r       = π / π_old
/// objectif = min(r × A, borné(r) × A) − β × (π_ref/π − ln(π_ref/π) − 1)
/// ```
///
/// Le second terme est l'estimateur de la divergence de Kullback-Leibler
/// qu'utilise l'article : toujours positif, nul quand `π = π_ref`. La perte
/// est l'opposé de la moyenne de l'objectif.
///
/// Où le rapport est borné, son gradient est nul : le calcul le sait en
/// lisant la valeur du rapport, et ne garde à ces tokens que la constante.
#[allow(clippy::too_many_arguments)]
pub fn grpo_loss<B: Backend>(
    tape: &Tape<B>,
    log_probs: Var,
    owners: &[usize],
    advantages: &[f64],
    old: &[f64],
    reference: &[f64],
    settings: &GrpoSettings,
) -> Var {
    let backend = tape.backend();
    let n = owners.len();
    let constant = |values: Vec<f64>| {
        let values = values.into_iter().map(B::Scalar::from_f64).collect();
        tape.constant(backend.from_vec(values, &[n]))
    };
    let current: Vec<f64> = backend
        .to_vec(&tape.value(log_probs))
        .iter()
        .map(|v| v.to_f64())
        .collect();
    let advantage: Vec<f64> = owners.iter().map(|&o| advantages[o]).collect();
    // Les tokens dont le rapport est borné, dans le sens où la borne mord.
    let mut free = vec![1.0; n];
    let mut fixed = vec![0.0; n];
    for i in 0..n {
        let ratio = (current[i] - old[i]).exp();
        let a = advantage[i];
        let bounded = ratio.clamp(1.0 - settings.clip, 1.0 + settings.clip);
        if bounded * a < ratio * a {
            free[i] = 0.0;
            fixed[i] = bounded * a;
        }
    }
    let ratio = tape.exp(tape.sub(log_probs, constant(old.to_vec())));
    let surrogate = tape.add(
        tape.mul(tape.mul(ratio, constant(advantage)), constant(free)),
        constant(fixed),
    );
    // `π_ref/π − ln(π_ref/π) − 1`, avec `ln(π_ref/π) = ref − lp`.
    let gap = tape.sub(constant(reference.to_vec()), log_probs);
    let kl = tape.sub(tape.sub(tape.exp(gap), gap), tape.scalar(B::Scalar::ONE));
    let objective = tape.sub(
        surrogate,
        tape.mul(kl, tape.scalar(B::Scalar::from_f64(settings.kl))),
    );
    // Le poids de chaque token dans la moyenne.
    let answers = advantages.len();
    let mut lengths = vec![0usize; answers];
    for &o in owners {
        lengths[o] += 1;
    }
    let weights: Vec<f64> = owners
        .iter()
        .map(|&o| {
            if settings.per_answer {
                1.0 / (lengths[o] as f64 * answers as f64)
            } else {
                1.0 / n as f64
            }
        })
        .collect();
    let weighted = tape.mul(objective, constant(weights));
    tape.neg(tape.sum(weighted, 0, false))
}

/// Les avantages d'un groupe de réponses à une même consigne : chaque
/// récompense, moins la moyenne du groupe, divisée par son écart-type. Un
/// groupe où toutes les réponses se valent n'apprend rien : tous ses
/// avantages sont nuls.
pub fn group_advantages(rewards: &[f64]) -> Vec<f64> {
    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let deviation = variance.sqrt();
    rewards
        .iter()
        .map(|r| {
            if deviation > 1e-9 {
                (r - mean) / deviation
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::Cpu;
    use crate::model::GptConfig;
    use crate::model::tests::{TINY, randomize};
    use crate::nn::Module;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Cpu<f64>;

    fn conversation(length: usize, answer: usize) -> Encoded {
        Encoded {
            tokens: (0..length as u32).collect(),
            assistant: (0..length).map(|i| i >= length - answer).collect(),
        }
    }

    #[test]
    fn padded_rows_hold_one_conversation_each() {
        let mut next = [conversation(4, 2), conversation(3, 1)].into_iter();
        let batch = sft_batch(|| next.next().unwrap(), 2, 6, false, 99);
        assert_eq!(batch.tokens, [0, 1, 2, 3, 99, 99, 0, 1, 2, 99, 99, 99]);
        assert_eq!(batch.targets, [1, 2, 3, 99, 99, 99, 1, 2, 99, 99, 99, 99]);
        // Les cibles notées sont les deux derniers tokens de la première,
        // le dernier de la seconde.
        let scored: Vec<usize> = (0..12).filter(|&i| batch.scored[i]).collect();
        assert_eq!(scored, [1, 2, 7]);
    }

    #[test]
    fn packed_rows_put_conversations_end_to_end() {
        let mut next = std::iter::repeat_with(|| conversation(3, 1));
        let batch = sft_batch(|| next.next().unwrap(), 1, 6, true, 99);
        // Deux conversations de 3 tiennent dans 7 tokens, pas trois.
        assert_eq!(batch.tokens, [0, 1, 2, 0, 1, 2]);
        assert_eq!(batch.targets, [1, 2, 0, 1, 2, 99]);
        let scored: Vec<usize> = (0..6).filter(|&i| batch.scored[i]).collect();
        assert_eq!(scored, [1, 4]);
    }

    #[test]
    fn sequence_log_probs_add_the_scored_targets() {
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(5);
        let config = GptConfig { context: 4, ..TINY };
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        let tokens = [3u32, 7, 1, 9, 2, 2, 5, 6];
        let targets = [7u32, 1, 9, 4, 2, 5, 6, 0];
        let scored = [false, true, true, false, true, false, false, true];
        let tape = Tape::new(&backend);
        let sums = sequence_log_probs(&tape, &model, &tokens, &targets, &scored, 2);
        let sums = tape.value(sums).data().to_vec();
        // À la main : le log-softmax de chaque position notée.
        let logits = tape.value(model.logits(&tape, &tokens, 2)).data().to_vec();
        let mut expected = [0.0; 2];
        for i in 0..8 {
            if scored[i] {
                let row = &logits[i * 11..(i + 1) * 11];
                let max = row.iter().cloned().fold(f64::MIN, f64::max);
                let log_total = row.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
                expected[i / 4] += row[targets[i] as usize] - log_total;
            }
        }
        for (a, b) in sums.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }

    #[test]
    fn the_dpo_loss_is_ln_2_at_the_reference_and_has_the_right_gradient() {
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(5);
        let config = GptConfig { context: 4, ..TINY };
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        // Deux paires : quatre réponses de deux tokens après une consigne.
        let tokens = [3u32, 7, 1, 9, 3, 7, 2, 2, 4, 5, 6, 1, 4, 5, 0, 0];
        let targets = [7u32, 1, 9, 4, 7, 2, 2, 8, 5, 6, 1, 3, 5, 0, 0, 2];
        let scored = [false, true, true, false].repeat(4);
        let loss_of = |model: &Gpt<B>, reference: &[f64]| {
            let tape = Tape::new(&backend);
            let lp = sequence_log_probs(&tape, model, &tokens, &targets, &scored, 4);
            tape.value(dpo_loss(&tape, lp, reference, 0.5)).item()
        };
        let tape = Tape::new(&backend);
        let lp = sequence_log_probs(&tape, &model, &tokens, &targets, &scored, 4);
        let reference = tape.value(lp).data().to_vec();
        drop(tape);
        // Le modèle est sa propre référence : `z = 0`, la perte vaut ln 2.
        assert!((loss_of(&model, &reference) - std::f64::consts::LN_2).abs() < 1e-12);

        // Le gradient, contre des différences finies, sur quelques poids.
        let other: Vec<f64> = reference.iter().map(|r| r + 0.3).collect();
        let tape = Tape::new(&backend);
        let lp = sequence_log_probs(&tape, &model, &tokens, &targets, &scored, 4);
        let loss = dpo_loss(&tape, lp, &other, 0.5);
        let grads = tape.backward(loss);
        let names = model.parameter_names();
        let analytic: Vec<(usize, Vec<f64>)> = model
            .parameters()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    i,
                    p.grad(tape.id(), &grads)
                        .map_or(vec![], |g| g.data().to_vec()),
                )
            })
            .collect();
        drop(tape);
        for (index, grad) in analytic.iter().filter(|(_, g)| !g.is_empty()).take(4) {
            for element in [0, grad.len() / 2] {
                let epsilon = 1e-6;
                let mut shifted = |delta: f64| {
                    let parameter = &mut model.parameters_mut()[*index];
                    let mut data = parameter.value().data().to_vec();
                    data[element] += delta;
                    let shape = parameter.value().shape().to_vec();
                    *parameter.value_mut() = crate::backend::cpu::CpuTensor::from_vec(data, &shape);
                    loss_of(&model, &other)
                };
                let up = shifted(epsilon);
                let down = shifted(-2.0 * epsilon);
                shifted(epsilon);
                let numeric = (up - down) / (2.0 * epsilon);
                assert!(
                    (numeric - grad[element]).abs() < 1e-6 * (1.0 + numeric.abs()),
                    "{} [{element}] : {numeric} contre {}",
                    names[*index],
                    grad[element],
                );
            }
        }
    }

    #[test]
    fn group_advantages_compare_answers_with_their_group() {
        let advantages = group_advantages(&[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(advantages, [1.0, -1.0, -1.0, 1.0]);
        assert_eq!(group_advantages(&[1.0, 1.0, 1.0]), [0.0; 3]);
    }

    #[test]
    fn grpo_on_its_own_samples_is_reinforce() {
        // Sans divergence ni borne active, les réponses tirées par le modèle
        // lui-même : le gradient est celui de `−A × ln π(réponse)`, moyenné.
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(5);
        let config = GptConfig { context: 4, ..TINY };
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        let tokens = [3u32, 7, 1, 9, 3, 7, 2, 2];
        let targets = [7u32, 1, 9, 4, 7, 2, 2, 8];
        let scored = [false, true, true, true, false, true, true, false];
        let advantages = [0.7, -0.7];
        let settings = GrpoSettings {
            clip: 0.2,
            kl: 0.0,
            per_answer: false,
        };
        let gradient = |grpo: bool| {
            let tape = Tape::new(&backend);
            let (lp, owners) = token_log_probs(&tape, &model, &tokens, &targets, &scored, 2);
            let values = tape.value(lp).data().to_vec();
            let loss = if grpo {
                grpo_loss(&tape, lp, &owners, &advantages, &values, &values, &settings)
            } else {
                let weights: Vec<f64> = owners.iter().map(|&o| -advantages[o] / 5.0).collect();
                let weights =
                    tape.constant(crate::backend::cpu::CpuTensor::from_vec(weights, &[5]));
                tape.sum(tape.mul(lp, weights), 0, false)
            };
            let grads = tape.backward(loss);
            model.parameters()[0]
                .grad(tape.id(), &grads)
                .unwrap()
                .data()
                .to_vec()
        };
        for (a, b) in gradient(true).iter().zip(gradient(false)) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn a_saturated_ratio_stops_the_gradient() {
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(5);
        let config = GptConfig { context: 4, ..TINY };
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        let tokens = [3u32, 7, 1, 9];
        let targets = [7u32, 1, 9, 4];
        let scored = [false, true, true, true];
        let tape = Tape::new(&backend);
        let (lp, owners) = token_log_probs(&tape, &model, &tokens, &targets, &scored, 1);
        // Le modèle a déjà beaucoup monté ces tokens par rapport à celui qui
        // les a tirés, et l'avantage est positif : la borne mord partout.
        let old: Vec<f64> = tape.value(lp).data().iter().map(|v| v - 1.0).collect();
        let settings = GrpoSettings {
            clip: 0.2,
            kl: 0.0,
            per_answer: true,
        };
        let loss = grpo_loss(&tape, lp, &owners, &[1.0], &old, &old, &settings);
        assert!((tape.value(loss).item() + 1.2).abs() < 1e-12);
        let grads = tape.backward(loss);
        let grad = model.parameters()[0].grad(tape.id(), &grads);
        assert!(grad.is_none_or(|g| g.data().iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn grpo_makes_the_rewarded_answer_more_likely() {
        // De bout en bout, sur un problème trivial : la réponse est
        // récompensée si son premier token est 5. Le modèle tire des
        // groupes de réponses, GRPO pousse les meilleures.
        use crate::generation::Batch;
        use crate::optim::AdamW;
        use crate::sampling::Sampling;
        use crate::training::train_step;
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(3);
        let config = GptConfig {
            context: 8,
            layers: 2,
            ..TINY
        };
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        let prompt = [3u32, 7];
        let probability_of_five = |model: &Gpt<B>| {
            let tape = Tape::new(&backend);
            let logits = tape.value(model.logits(&tape, &prompt, 1)).data().to_vec();
            let row = &logits[11..22];
            let total: f64 = row.iter().map(|l| l.exp()).sum();
            row[5].exp() / total
        };
        let before = probability_of_five(&model);
        let settings = GrpoSettings {
            clip: 0.2,
            kl: 0.0,
            per_answer: true,
        };
        let mut optimizer = AdamW::new(1e-2, 0.0);
        for _ in 0..20 {
            let mut batch = Batch::new(&backend, &model, 16);
            let numbers: Vec<usize> = (0..16)
                .map(|_| batch.add_limited(&prompt, Some(4)))
                .collect();
            while batch.running() > 0 {
                batch.step(&Sampling::default(), &mut rng);
            }
            let answers: Vec<Vec<u32>> = numbers
                .iter()
                .map(|&n| batch.tokens(n)[2..].to_vec())
                .collect();
            drop(batch);
            let rewards: Vec<f64> = answers
                .iter()
                .map(|a| f64::from(u8::from(a[0] == 5)))
                .collect();
            let advantages = group_advantages(&rewards);
            let encoded: Vec<Encoded> = answers
                .iter()
                .map(|answer| Encoded {
                    tokens: prompt.iter().chain(answer).copied().collect(),
                    assistant: vec![false, false, true, true],
                })
                .collect();
            let mut items = encoded.into_iter();
            let rows = sft_batch(|| items.next().unwrap(), 16, 4, false, 0);
            let old = {
                let tape = Tape::new(&backend);
                let (lp, _) =
                    token_log_probs(&tape, &model, &rows.tokens, &rows.targets, &rows.scored, 16);
                tape.value(lp).data().to_vec()
            };
            train_step(&backend, &mut model, &mut optimizer, |m, t| {
                let (lp, owners) =
                    token_log_probs(t, m, &rows.tokens, &rows.targets, &rows.scored, 16);
                grpo_loss(t, lp, &owners, &advantages, &old, &old, &settings)
            });
        }
        let after = probability_of_five(&model);
        assert!(after > 3.0 * before, "{before} → {after}");
    }
}
