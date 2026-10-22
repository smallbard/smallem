//! Le décodage spéculatif : un modèle rapide propose, le modèle visé
//! vérifie.
//!
//! Générer un token demande une passe entière du modèle, et cette passe
//! coûte presque autant pour un token que pour huit (article 2 de la série
//! sur l'inférence). L'idée est d'en profiter : un **proposeur** bon marché
//! devine les `k` tokens suivants, et le modèle visé les vérifie tous en une
//! seule passe, comme à l'entraînement, où chaque position prédit la
//! suivante. Si les devinettes sont bonnes, une passe produit plusieurs
//! tokens.
//!
//! La vérification ne se contente pas de garder les tokens que le modèle
//! visé aurait tirés : ce serait un autre tirage. Elle accepte chaque
//! proposition `x`, tirée selon la distribution `q` du proposeur, avec la
//! probabilité `min(1, p(x) / q(x))`, où `p` est la distribution du modèle
//! visé. Au premier refus, elle tire un token de la distribution corrigée
//! `max(0, p − q)`, renormalisée, et s'arrête. Si tout est accepté, la passe
//! donne en prime un token de plus, tiré de `p`. Le texte produit suit alors
//! **exactement** la distribution du modèle visé (Leviathan, Kalman et
//! Matias, 2023 ; Chen et al., 2023).

use rand::{Rng, RngExt};

use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};
use crate::cache::{KvCache, SequenceId};
use crate::model::Gpt;
use crate::sampling::{Sampling, probabilities, sample};

/// Ce que donne la vérification de `k` propositions.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    /// Nombre de propositions acceptées, de 0 à `k`.
    pub accepted: usize,
    /// Le token qui suit les propositions acceptées : tiré de la
    /// distribution corrigée après un refus, de `p` si tout est accepté.
    pub next: u32,
}

/// Vérifie les propositions `proposed`, tirées une à une selon les
/// distributions `draft` du proposeur. `target` contient les distributions
/// du modèle visé aux mêmes positions, plus une : `k + 1` en tout.
///
/// Les distributions sont celles qu'on veut réellement tirer : température,
/// `top_k` et `top_p` déjà appliqués, des deux côtés. En mode glouton, ce
/// sont des distributions certaines, et la règle revient à garder les
/// propositions tant qu'elles sont le choix du modèle visé.
pub fn verify(
    rng: &mut impl Rng,
    proposed: &[u32],
    draft: &[Vec<f64>],
    target: &[Vec<f64>],
) -> Verdict {
    assert!(
        draft.len() == proposed.len() && target.len() == proposed.len() + 1,
        "{} propositions, {} distributions du proposeur et {} du modèle visé",
        proposed.len(),
        draft.len(),
        target.len(),
    );
    for (i, &token) in proposed.iter().enumerate() {
        let (p, q) = (target[i][token as usize], draft[i][token as usize]);
        // `q > 0` : le proposeur a tiré ce token.
        let keep: f64 = rng.random();
        if keep * q >= p {
            // Refus : la distribution corrigée, `max(0, p − q)`.
            let residual: Vec<f64> = target[i]
                .iter()
                .zip(&draft[i])
                .map(|(p, q)| (p - q).max(0.0))
                .collect();
            let total: f64 = residual.iter().sum();
            let next = if total > 0.0 {
                let residual: Vec<f64> = residual.iter().map(|r| r / total).collect();
                sample(rng, &residual)
            } else {
                // `p = q` : un refus n'arrive alors qu'avec une probabilité
                // nulle ; aux arrondis près, on tire simplement de `p`.
                sample(rng, &target[i])
            };
            return Verdict {
                accepted: i,
                next: next as u32,
            };
        }
    }
    let next = sample(rng, &target[proposed.len()]) as u32;
    Verdict {
        accepted: proposed.len(),
        next,
    }
}

/// Un modèle qui suit une séquence avec son cache.
struct Follower<'a, B: Backend> {
    model: &'a Gpt<B>,
    cache: KvCache<B>,
    id: SequenceId,
    /// Les tokens que le cache contient déjà.
    seen: Vec<u32>,
}

impl<'a, B: Backend> Follower<'a, B> {
    fn new(model: &'a Gpt<B>) -> Self {
        let mut cache = model.cache(1);
        let id = cache.add_sequence();
        Self {
            model,
            cache,
            id,
            seen: Vec::new(),
        }
    }

    /// Fait lire `tokens` au modèle, sur `tape`. Rend les états cachés et
    /// les logits de ces positions.
    fn read(&mut self, tape: &Tape<B>, tokens: &[u32]) -> (Var, Var) {
        assert!(
            self.seen.len() + tokens.len() <= self.model.config().context,
            "le décodage spéculatif reste dans le contexte du modèle",
        );
        self.seen.extend_from_slice(tokens);
        self.model
            .forward_cached_with_hidden(tape, &mut self.cache, &[(self.id, tokens)])
    }

    /// Oublie tout ce qui suit les `keep` premiers tokens lus. Avec une
    /// mémoire de n-grammes, le cache doit avoir été réglé pour revenir
    /// d'autant ([`KvCache::set_rollback`]).
    fn keep(&mut self, backend: &B, keep: usize) {
        self.cache.truncate(backend, self.id, keep);
        self.seen.truncate(keep);
    }
}

/// Les distributions de tirage des `count` dernières lignes de `logits`.
fn last_distributions<B: Backend>(
    tape: &Tape<B>,
    logits: Var,
    count: usize,
    sampling: &Sampling,
) -> Vec<Vec<f64>> {
    let backend = tape.backend();
    let rows = tape.shape(logits)[0];
    let indices: Vec<u32> = (rows - count..rows).map(|row| row as u32).collect();
    let values = backend.to_vec(&backend.gather_rows(&tape.value(logits), &indices));
    let vocab = values.len() / count;
    values
        .chunks(vocab)
        .map(|row| {
            let row: Vec<f64> = row.iter().map(|l| l.to_f64()).collect();
            probabilities(&row, sampling)
        })
        .collect()
}

/// Ce qu'a donné une génération spéculative.
#[derive(Clone, Copy, Debug, Default)]
pub struct Statistics {
    /// Passes du modèle visé.
    pub passes: usize,
    /// Propositions faites, et acceptées.
    pub proposed: usize,
    pub accepted: usize,
}

impl Statistics {
    /// Part des propositions acceptées.
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.proposed.max(1) as f64
    }

    /// Tokens produits par passe du modèle visé.
    pub fn tokens_per_pass(&self, tokens: usize) -> f64 {
        tokens as f64 / self.passes.max(1) as f64
    }
}

/// Le décodage spéculatif avec un petit modèle comme proposeur : à chaque
/// tour, il propose `k` tokens, un par un, et le modèle visé les vérifie en
/// une passe. Les deux modèles doivent partager leur tokeniseur.
pub struct DraftDecoder<'a, B: Backend> {
    backend: &'a B,
    target: Follower<'a, B>,
    draft: Follower<'a, B>,
    tokens: Vec<u32>,
    pub k: usize,
    pub statistics: Statistics,
}

impl<'a, B: Backend> DraftDecoder<'a, B> {
    pub fn new(
        backend: &'a B,
        target: &'a Gpt<B>,
        draft: &'a Gpt<B>,
        prompt: &[u32],
        k: usize,
    ) -> Self {
        assert!(!prompt.is_empty(), "la génération a besoin d'une amorce");
        assert_eq!(
            target.config().vocab,
            draft.config().vocab,
            "le proposeur et le modèle visé doivent partager leur vocabulaire",
        );
        Self {
            backend,
            target: Follower::new(target),
            draft: Follower::new(draft),
            tokens: prompt.to_vec(),
            k,
            statistics: Statistics::default(),
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Un tour : `k` propositions, une passe de vérification. Rend les
    /// nouveaux tokens, de 1 à `k + 1`.
    pub fn step(&mut self, sampling: &Sampling, rng: &mut impl Rng) -> Vec<u32> {
        let start = self.tokens.len();
        // Les deux caches reviendront sur au plus `k` propositions.
        self.target.cache.set_rollback(self.k);
        self.draft.cache.set_rollback(self.k);
        // Le proposeur lit ce qui lui manque, puis propose un token à la
        // fois, chacun d'après les précédents.
        let mut proposed = Vec::with_capacity(self.k);
        let mut draft = Vec::with_capacity(self.k);
        let mut pending = self.tokens[self.draft.seen.len()..].to_vec();
        for _ in 0..self.k {
            let tape = Tape::new(self.backend);
            let (_, logits) = self.draft.read(&tape, &pending);
            let distribution = last_distributions(&tape, logits, 1, sampling).remove(0);
            let token = sample(rng, &distribution) as u32;
            proposed.push(token);
            draft.push(distribution);
            pending = vec![token];
        }
        // Le modèle visé lit ce qui lui manque, suivi des propositions :
        // `k + 1` distributions, une par proposition et une de plus.
        let mut pending = self.tokens[self.target.seen.len()..].to_vec();
        pending.extend_from_slice(&proposed);
        let target = {
            let tape = Tape::new(self.backend);
            let (_, logits) = self.target.read(&tape, &pending);
            last_distributions(&tape, logits, self.k + 1, sampling)
        };
        let verdict = verify(rng, &proposed, &draft, &target);
        self.statistics.passes += 1;
        self.statistics.proposed += self.k;
        self.statistics.accepted += verdict.accepted;

        self.tokens.extend_from_slice(&proposed[..verdict.accepted]);
        self.tokens.push(verdict.next);
        // Les caches oublient les propositions refusées. Le dernier token,
        // tiré par la vérification, n'a encore été lu par personne.
        let kept = start + verdict.accepted;
        self.target.keep(self.backend, kept);
        let draft_kept = self.draft.seen.len().min(kept);
        self.draft.keep(self.backend, draft_kept);
        self.tokens[start..].to_vec()
    }
}

/// Le décodage spéculatif avec le module de prédiction multi-token du
/// modèle visé comme proposeur, une proposition par passe.
///
/// À la position `i`, le module reçoit l'état caché `h_i` du modèle et le
/// token `t_{i+1}`, et prédit `t_{i+2}`. Chaque passe du modèle lit donc le
/// dernier token accepté et la proposition qui le suit : elle vérifie la
/// proposition, et donne les états cachés d'où le module tire la suivante.
/// Le proposeur ne coûte qu'un bloc et la sortie, et il voit tout ce que le
/// modèle a calculé. C'est ainsi que DeepSeek-V3 se sert de son module.
pub struct MultiTokenDecoder<'a, B: Backend> {
    backend: &'a B,
    target: Follower<'a, B>,
    module_cache: KvCache<B>,
    module_id: SequenceId,
    /// Positions déjà passées par le module.
    module_seen: usize,
    tokens: Vec<u32>,
    /// La proposition pour le token qui suit `tokens`, et sa distribution.
    proposal: Option<(u32, Vec<f64>)>,
    pub statistics: Statistics,
}

impl<'a, B: Backend> MultiTokenDecoder<'a, B> {
    pub fn new(backend: &'a B, model: &'a Gpt<B>, prompt: &[u32]) -> Self {
        assert!(!prompt.is_empty(), "la génération a besoin d'une amorce");
        let mut module_cache = model.multi_token_cache();
        let module_id = module_cache.add_sequence();
        let mut target = Follower::new(model);
        // Chaque passe revient au plus sur la proposition qu'elle vérifie.
        target.cache.set_rollback(1);
        Self {
            backend,
            target,
            module_cache,
            module_id,
            module_seen: 0,
            tokens: prompt.to_vec(),
            proposal: None,
            statistics: Statistics::default(),
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Un tour : une passe du modèle, qui vérifie la proposition en cours
    /// s'il y en a une, puis une passe du module pour la suivante. Rend
    /// les nouveaux tokens, un ou deux.
    pub fn step(&mut self, sampling: &Sampling, rng: &mut impl Rng) -> Vec<u32> {
        let start = self.tokens.len();
        let read_from = self.target.seen.len();
        let mut pending = self.tokens[read_from..].to_vec();
        let proposal = self.proposal.take();
        pending.extend(proposal.as_ref().map(|(token, _)| *token));
        let tape = Tape::new(self.backend);
        let (hidden, logits) = self.target.read(&tape, &pending);

        let (accepted, next) = match &proposal {
            Some((token, draft)) => {
                let target = last_distributions(&tape, logits, 2, sampling);
                let verdict = verify(rng, &[*token], std::slice::from_ref(draft), &target);
                self.statistics.proposed += 1;
                self.statistics.accepted += verdict.accepted;
                (verdict.accepted, verdict.next)
            }
            None => {
                let target = last_distributions(&tape, logits, 1, sampling);
                (0, sample(rng, &target[0]) as u32)
            }
        };
        self.statistics.passes += 1;
        if let Some((token, _)) = &proposal
            && accepted == 1
        {
            self.tokens.push(*token);
        }
        self.tokens.push(next);
        self.target.keep(self.backend, start + accepted);

        // Le module passe sur chaque position dont le token suivant est
        // maintenant connu, et dont l'état caché sort de cette passe.
        let first = self.module_seen;
        let last = self.tokens.len() - 2;
        let rows: Vec<u32> = (first..=last).map(|i| (i - read_from) as u32).collect();
        let dim = tape.shape(hidden)[2];
        let flat = tape.reshape(hidden, &[pending.len(), dim]);
        let states = tape.gather_rows(flat, &rows);
        let module = self.target.model;
        let logits = module.multi_token_cached(
            &tape,
            &mut self.module_cache,
            self.module_id,
            states,
            &self.tokens[first + 1..=last + 1],
        );
        self.module_seen = last + 1;
        let draft = last_distributions(&tape, logits, 1, sampling).remove(0);
        let token = sample(rng, &draft) as u32;
        self.proposal = Some((token, draft));
        self.tokens[start..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn greedy_verification_keeps_what_the_target_would_choose() {
        let mut rng = StdRng::seed_from_u64(0);
        let one = |token: usize| {
            let mut p = vec![0.0; 4];
            p[token] = 1.0;
            p
        };
        // Le modèle visé choisirait 2, 1, puis 3 ; le proposeur dit 2, 0.
        let verdict = verify(
            &mut rng,
            &[2, 0],
            &[one(2), one(0)],
            &[one(2), one(1), one(3)],
        );
        assert_eq!(
            verdict,
            Verdict {
                accepted: 1,
                next: 1
            }
        );
        // Tout accepté : le token en prime vient de la dernière position.
        let verdict = verify(
            &mut rng,
            &[2, 1],
            &[one(2), one(1)],
            &[one(2), one(1), one(3)],
        );
        assert_eq!(
            verdict,
            Verdict {
                accepted: 2,
                next: 3
            }
        );
    }

    /// Le khi-deux d'une table de comptes contre des probabilités.
    fn chi_square(counts: &[u64], probabilities: &[f64]) -> f64 {
        let total: u64 = counts.iter().sum();
        counts
            .iter()
            .zip(probabilities)
            .map(|(&count, &p)| {
                let expected = p * total as f64;
                (count as f64 - expected).powi(2) / expected
            })
            .sum()
    }

    #[test]
    fn the_output_follows_the_target_distribution() {
        // Un vocabulaire de 5 tokens, et deux positions. Les distributions
        // du proposeur et du modèle visé sont différentes, et celles du
        // modèle visé dépendent du premier token : on compte les paires
        // produites par une passe de vérification, et on les compare à ce
        // que donnerait le modèle visé seul.
        let draft = [vec![0.4, 0.3, 0.1, 0.1, 0.1], vec![0.1, 0.1, 0.2, 0.3, 0.3]];
        let target_first = vec![0.1, 0.2, 0.3, 0.25, 0.15];
        let target_second = |first: usize| -> Vec<f64> {
            let mut p = vec![0.1; 5];
            p[first] += 0.5;
            p
        };
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = vec![0u64; 25];
        let trials = 200_000;
        for _ in 0..trials {
            let first = sample(&mut rng, &draft[0]) as u32;
            // Le proposeur tire la deuxième proposition sans regarder la
            // première, et le modèle visé la juge sachant la première.
            let second = sample(&mut rng, &draft[1]) as u32;
            let target = [
                target_first.clone(),
                target_second(first as usize),
                vec![0.2; 5],
            ];
            let verdict = verify(&mut rng, &[first, second], &draft, &target);
            let pair = match verdict.accepted {
                0 => {
                    let second = sample(&mut rng, &target_second(verdict.next as usize));
                    (verdict.next as usize, second)
                }
                1 => (first as usize, verdict.next as usize),
                _ => (first as usize, second as usize),
            };
            counts[pair.0 * 5 + pair.1] += 1;
        }
        let expected: Vec<f64> = (0..25)
            .map(|i| target_first[i / 5] * target_second(i / 5)[i % 5])
            .collect();
        // 24 degrés de liberté : au seuil de 0,1 %, le khi-deux reste sous
        // 51,2.
        let chi = chi_square(&counts, &expected);
        assert!(chi < 51.2, "khi-deux {chi}");
        // Et le même test rejette bien la distribution du proposeur.
        let wrong: Vec<f64> = (0..25).map(|i| draft[0][i / 5] * draft[1][i % 5]).collect();
        assert!(chi_square(&counts, &wrong) > 1000.0);
    }

    mod decoding {
        use super::super::*;
        use crate::backend::cpu::Cpu;
        use crate::generation::generate;
        use crate::model::GptConfig;
        use crate::model::tests::{TINY, randomize};
        use crate::nn::{Architecture, MemoryDims, MemoryStorage, MultiToken};
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        type B = Cpu<f64>;

        const GREEDY: Sampling = Sampling {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        };

        fn model(backend: &B, seed: u64, architecture: Architecture) -> Gpt<B> {
            let config = GptConfig {
                context: 40,
                layers: 2,
                ..TINY.with_architecture(architecture)
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut model = Gpt::<B>::new(backend, config, &mut rng);
            randomize(&mut model, &mut rng);
            model
        }

        #[test]
        fn greedy_speculation_with_a_draft_model_writes_the_same_tokens() {
            let backend = Cpu::new();
            let target = model(&backend, 1, Architecture::MODERN);
            let draft = model(&backend, 2, Architecture::MODERN);
            let mut rng = StdRng::seed_from_u64(0);
            let alone = generate(&backend, &target, &[3, 7], 20, &GREEDY, &mut rng);
            for k in [1, 3] {
                let mut decoder = DraftDecoder::new(&backend, &target, &draft, &[3, 7], k);
                while decoder.tokens().len() < alone.len() {
                    decoder.step(&GREEDY, &mut rng);
                }
                assert_eq!(&decoder.tokens()[..alone.len()], &alone[..], "k = {k}");
            }
        }

        /// Une mémoire de n-grammes, avec sa convolution : ce qu'elle garde
        /// des dernières positions doit reculer avec le cache.
        fn with_memory(architecture: Architecture) -> Architecture {
            architecture.with_memory(MemoryDims {
                heads: 2,
                storage: MemoryStorage::Sparse,
                table_std: 0.5,
                ..MemoryDims::engram(TINY.embed, 31)
            })
        }

        #[test]
        fn greedy_speculation_with_a_memory_writes_the_same_tokens() {
            let backend = Cpu::new();
            let target = model(&backend, 1, with_memory(Architecture::MODERN));
            let draft = model(&backend, 2, with_memory(Architecture::MODERN));
            let mut rng = StdRng::seed_from_u64(0);
            let alone = generate(&backend, &target, &[3, 7], 30, &GREEDY, &mut rng);
            for k in [1, 3] {
                let mut decoder = DraftDecoder::new(&backend, &target, &draft, &[3, 7], k);
                while decoder.tokens().len() < alone.len() {
                    decoder.step(&GREEDY, &mut rng);
                }
                assert_eq!(&decoder.tokens()[..alone.len()], &alone[..], "k = {k}");
                let statistics = decoder.statistics;
                assert!(
                    statistics.accepted < statistics.proposed,
                    "un refus au moins"
                );
            }
            let architecture =
                with_memory(Architecture::MODERN.with_multi_token(MultiToken::DEEPSEEK));
            let target = model(&backend, 3, architecture);
            let alone = generate(&backend, &target, &[3, 7, 1], 30, &GREEDY, &mut rng);
            let mut decoder = MultiTokenDecoder::new(&backend, &target, &[3, 7, 1]);
            while decoder.tokens().len() < alone.len() {
                decoder.step(&GREEDY, &mut rng);
            }
            assert_eq!(&decoder.tokens()[..alone.len()], &alone[..]);
            let statistics = decoder.statistics;
            assert!(
                statistics.accepted < statistics.proposed,
                "un refus au moins"
            );
        }

        #[test]
        fn a_draft_identical_to_the_target_is_always_accepted() {
            let backend = Cpu::new();
            let target = model(&backend, 1, Architecture::MODERN);
            let mut rng = StdRng::seed_from_u64(0);
            let mut decoder = DraftDecoder::new(&backend, &target, &target, &[3, 7], 4);
            for _ in 0..4 {
                decoder.step(&Sampling::default(), &mut rng);
            }
            assert_eq!(decoder.statistics.acceptance(), 1.0);
            assert_eq!(decoder.tokens().len(), 2 + 4 * 5);
        }

        #[test]
        fn greedy_speculation_with_the_multi_token_module_writes_the_same_tokens() {
            let backend = Cpu::new();
            let architecture = Architecture::MODERN.with_multi_token(MultiToken::DEEPSEEK);
            let target = model(&backend, 1, architecture);
            let mut rng = StdRng::seed_from_u64(0);
            let alone = generate(&backend, &target, &[3, 7, 1], 20, &GREEDY, &mut rng);
            let mut decoder = MultiTokenDecoder::new(&backend, &target, &[3, 7, 1]);
            while decoder.tokens().len() < alone.len() {
                decoder.step(&GREEDY, &mut rng);
            }
            assert_eq!(&decoder.tokens()[..alone.len()], &alone[..]);
            assert!(decoder.statistics.proposed > 0);
        }
    }
}
