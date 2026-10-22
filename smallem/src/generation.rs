//! La génération : prolonger une séquence, un token à la fois.
//!
//! Le modèle ne sait faire qu'une chose : donner, pour une séquence, un
//! score à chaque token qui pourrait la suivre. Générer, c'est répéter
//! trois gestes : demander ces scores, en tirer un token, l'ajouter à la
//! séquence. Le [`Generator`] tient cette boucle, et ce dont elle a besoin
//! d'un tour à l'autre : la séquence et le cache des couches d'attention.

use rand::Rng;

use crate::autograd::Tape;
use crate::backend::{Backend, Float};
use crate::cache::{KvCache, SequenceId};
use crate::model::Gpt;
use crate::sampling::{Sampling, argmax, sample_logits};

/// Une génération en cours.
///
/// ```text
/// let mut generator = Generator::new(&backend, &model, &prompt);
/// while generator.tokens().len() < 100 {
///     let token = generator.next_token(&sampling, &mut rng);
///     …                       // l'afficher, décider de s'arrêter
/// }
/// ```
///
/// L'amorce traverse le modèle une fois, en entier. Ensuite, chaque nouveau
/// token est le seul à le traverser : les clés et valeurs des positions
/// précédentes sont gardées dans un cache. Le coût d'un token est alors
/// proportionnel à la longueur de la séquence, et non à son carré.
///
/// Le modèle ne voit jamais plus de `context` tokens. D'une amorce plus
/// longue, il ne lit que la fin. Et quand la séquence dépasse le contexte,
/// la suite dépend de la façon dont il connaît les positions.
///
/// - **Positions apprises.** Toutes les positions se décalent d'un cran à
///   chaque token, et avec elles leurs embeddings : le cache ne vaut plus
///   rien. Il est abandonné, et tout est recalculé à chaque token.
/// - **RoPE.** Seules les distances comptent, et elles ne changent pas : le
///   cache reste bon. Il suffit d'oublier la plus ancienne position. Les
///   clés et valeurs gardées ont toutefois été calculées quand leurs
///   positions voyaient encore des tokens aujourd'hui oubliés : dès deux
///   couches, le résultat n'est donc plus celui d'un recalcul sur la seule
///   fenêtre visible. C'est le comportement habituel des générateurs à
///   cache glissant.
pub struct Generator<'a, B: Backend> {
    backend: &'a B,
    model: &'a Gpt<B>,
    tokens: Vec<u32>,
    /// Le cache et la séquence qu'il suit pour cette génération, ou `None`
    /// quand la génération se fait sans cache.
    cache: Option<(KvCache<B>, SequenceId)>,
    /// Rang du premier token qui n'a pas encore traversé le modèle.
    pending: usize,
    /// Les logits du prochain token, s'ils ont déjà été calculés pour la
    /// séquence actuelle.
    logits: Option<Vec<f64>>,
}

impl<'a, B: Backend> Generator<'a, B> {
    /// Une génération qui part de `prompt`.
    pub fn new(backend: &'a B, model: &'a Gpt<B>, prompt: &[u32]) -> Self {
        let mut cache = model.cache(1);
        let sequence = cache.add_sequence();
        Self::start(backend, model, prompt, Some((cache, sequence)))
    }

    /// Comme [`Generator::new`], sans cache : tout le modèle est recalculé
    /// sur toute la séquence visible à chaque token. C'est la définition de
    /// référence, à laquelle la version avec cache est comparée.
    pub fn without_cache(backend: &'a B, model: &'a Gpt<B>, prompt: &[u32]) -> Self {
        Self::start(backend, model, prompt, None)
    }

    fn start(
        backend: &'a B,
        model: &'a Gpt<B>,
        prompt: &[u32],
        cache: Option<(KvCache<B>, SequenceId)>,
    ) -> Self {
        assert!(!prompt.is_empty(), "la génération a besoin d'une amorce");
        Self {
            backend,
            model,
            tokens: prompt.to_vec(),
            cache,
            pending: 0,
            logits: None,
        }
    }

    /// La séquence : l'amorce, puis tout ce qui a été ajouté depuis.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn into_tokens(self) -> Vec<u32> {
        self.tokens
    }

    /// Le cache des couches d'attention, si la génération en utilise
    /// encore un, et la séquence qu'il suit.
    pub fn cache(&self) -> Option<(&KvCache<B>, SequenceId)> {
        self.cache
            .as_ref()
            .map(|(cache, sequence)| (cache, *sequence))
    }

    /// Nombre de positions gardées dans le cache, s'il y en a un.
    pub fn cached_len(&self) -> Option<usize> {
        self.cache().map(|(cache, sequence)| cache.len(sequence))
    }

    /// Les scores du modèle pour le token qui suivrait la séquence : un par
    /// token du vocabulaire. Ils ne sont calculés qu'une fois par séquence.
    pub fn logits(&mut self) -> &[f64] {
        if self.logits.is_none() {
            self.logits = Some(self.compute_logits());
        }
        self.logits.as_deref().expect("logits calculés")
    }

    /// Ajoute `token` à la séquence, sans le tirer : pour imposer une suite,
    /// ou reprendre la main sur le tirage.
    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
        self.logits = None;
    }

    /// Tire le token suivant, l'ajoute à la séquence et le renvoie.
    pub fn next_token(&mut self, sampling: &Sampling, rng: &mut impl Rng) -> u32 {
        let token = sample_logits(rng, self.logits(), sampling) as u32;
        self.push(token);
        token
    }

    /// Prolonge la séquence de `count` tokens.
    pub fn extend(&mut self, count: usize, sampling: &Sampling, rng: &mut impl Rng) {
        for _ in 0..count {
            self.next_token(sampling, rng);
        }
    }

    fn compute_logits(&mut self) -> Vec<f64> {
        let context = self.model.config().context;
        // Début de ce que le modèle peut voir de la séquence.
        let window = self.tokens.len().saturating_sub(context);
        if window > 0 && !self.model.has_relative_positions() {
            // Positions apprises, et la fenêtre a commencé à glisser.
            self.cache = None;
        }
        match &mut self.cache {
            Some((cache, sequence)) => {
                // Seuls les tokens que le modèle n'a pas encore vus le
                // traversent ; s'il y en a plus que le contexte, les
                // premiers ne seraient de toute façon plus visibles.
                let new = &self.tokens[self.pending.max(window)..];
                self.pending = self.tokens.len();
                let tape = Tape::new(self.backend);
                let logits = self.model.forward_cached(&tape, cache, &[(*sequence, new)]);
                last_row(self.backend, &tape.value(logits))
            }
            None => {
                let tokens = &self.tokens[window..];
                let tape = Tape::new(self.backend);
                let logits = self.model.logits(&tape, tokens, 1);
                let rows = tape.reshape(logits, &[tokens.len(), self.model.config().vocab]);
                last_row(self.backend, &tape.value(rows))
            }
        }
    }
}

/// Plusieurs générations menées ensemble, une passe du modèle pour toutes.
///
/// Générer un token lit tout le modèle pour une seule multiplication par
/// poids : la carte attend sa mémoire. Avec `B` séquences, chaque poids lu
/// sert `B` fois, pour la même lecture.
///
/// ```text
/// let mut batch = Batch::new(&backend, &model, 64);
/// let a = batch.add(&prompt_a);
/// let b = batch.add(&prompt_b);
/// while batch.running() > 0 {
///     for (sequence, token) in batch.step(&sampling, &mut rng) { … }
/// }
/// ```
///
/// Les séquences ont chacune leur longueur, leur position et leur fin. Il
/// n'y a ni remplissage ni masque : les tokens de toutes les séquences
/// traversent le modèle comme une seule suite, et seule l'attention sait, par
/// le cache, à quelle séquence appartient chacun. Une amorce traverse le
/// modèle en entier lors du premier pas qui suit son ajout, en même temps
/// que le token suivant des autres séquences.
///
/// Une séquence s'arrête quand elle tire `stop`, ou atteint `max_len`
/// tokens. Sa place dans le cache est alors rendue, et peut servir à une
/// séquence ajoutée ensuite.
pub struct Batch<'a, B: Backend> {
    backend: &'a B,
    model: &'a Gpt<B>,
    cache: KvCache<B>,
    sequences: Vec<BatchSequence>,
    /// Le token qui termine une séquence, s'il y en a un.
    pub stop: Option<u32>,
    /// Nombre maximal de tokens d'une séquence, amorce comprise.
    pub max_len: usize,
}

struct BatchSequence {
    /// `None` une fois la séquence terminée.
    id: Option<SequenceId>,
    tokens: Vec<u32>,
    /// Rang du premier token qui n'a pas encore traversé le modèle.
    pending: usize,
    /// Longueur à laquelle la séquence s'arrête, amorce comprise, si elle
    /// en a une à elle.
    limit: Option<usize>,
}

impl<'a, B: Backend> Batch<'a, B> {
    /// Un lot vide, dont le cache peut suivre `capacity` séquences de toute
    /// la longueur du contexte.
    pub fn new(backend: &'a B, model: &'a Gpt<B>, capacity: usize) -> Self {
        Self::with_cache(backend, model, model.cache(capacity))
    }

    /// Un lot vide, avec ce cache.
    ///
    /// Avec des positions apprises, une séquence ne peut pas dépasser le
    /// contexte : elle s'arrête là.
    pub fn with_cache(backend: &'a B, model: &'a Gpt<B>, cache: KvCache<B>) -> Self {
        let max_len = if model.has_relative_positions() {
            usize::MAX
        } else {
            model.config().context
        };
        Self {
            backend,
            model,
            cache,
            sequences: Vec::new(),
            stop: None,
            max_len,
        }
    }

    /// Ajoute une séquence qui part de `prompt`, et rend son numéro.
    pub fn add(&mut self, prompt: &[u32]) -> usize {
        self.add_limited(prompt, None)
    }

    /// Comme [`Batch::add`], pour une séquence qui s'arrête aussi à
    /// `limit` tokens, amorce comprise.
    pub fn add_limited(&mut self, prompt: &[u32], limit: Option<usize>) -> usize {
        assert!(!prompt.is_empty(), "la génération a besoin d'une amorce");
        let id = self.cache.add_sequence();
        self.sequences.push(BatchSequence {
            id: Some(id),
            tokens: prompt.to_vec(),
            pending: 0,
            limit,
        });
        self.sequences.len() - 1
    }

    /// Les numéros des séquences en cours.
    pub fn running_sequences(&self) -> Vec<usize> {
        (0..self.sequences.len())
            .filter(|&index| self.is_running(index))
            .collect()
    }

    /// Nombre de pages que le prochain pas prendra au cache : celles des
    /// tokens que les séquences en cours n'ont pas encore fait lire au
    /// modèle.
    pub fn pages_needed(&self) -> usize {
        self.sequences
            .iter()
            .filter_map(|sequence| {
                let id = sequence.id?;
                let new = sequence.tokens.len() - sequence.pending;
                Some(self.cache.pages_needed(id, new))
            })
            .sum()
    }

    /// Les tokens de la séquence `sequence` : son amorce, puis ce qui a été
    /// généré.
    pub fn tokens(&self, sequence: usize) -> &[u32] {
        &self.sequences[sequence].tokens
    }

    /// Vrai tant que la séquence n'est pas terminée.
    pub fn is_running(&self, sequence: usize) -> bool {
        self.sequences[sequence].id.is_some()
    }

    /// Nombre de séquences en cours.
    pub fn running(&self) -> usize {
        self.sequences.iter().filter(|s| s.id.is_some()).count()
    }

    pub fn cache(&self) -> &KvCache<B> {
        &self.cache
    }

    /// Termine la séquence `sequence` : sa place dans le cache est rendue.
    pub fn finish(&mut self, sequence: usize) {
        if let Some(id) = self.sequences[sequence].id.take() {
            self.cache.remove_sequence(id);
        }
    }

    /// Les logits du token qui suivrait chaque séquence en cours, en une
    /// passe : `(numéro, logits)`. Les tokens que le modèle n'a pas encore
    /// vus le traversent, et entrent dans le cache.
    pub fn logits(&mut self) -> Vec<(usize, Vec<f64>)> {
        let (owners, values) = self.forward();
        let vocab = self.model.config().vocab;
        owners
            .into_iter()
            .zip(values.chunks(vocab))
            .map(|(index, row)| (index, row.iter().map(|l| l.to_f64()).collect()))
            .collect()
    }

    /// La passe de [`Batch::logits`] : les numéros des séquences en cours,
    /// et leurs logits mis bout à bout, tels que la carte les rend.
    fn forward(&mut self) -> (Vec<usize>, Vec<B::Scalar>) {
        let context = self.model.config().context;
        let mut entries = Vec::new();
        let mut owners = Vec::new();
        for (index, sequence) in self.sequences.iter_mut().enumerate() {
            let Some(id) = sequence.id else { continue };
            // Au-delà du contexte, les premiers tokens ne seraient de toute
            // façon plus visibles.
            let window = sequence.tokens.len().saturating_sub(context);
            let start = sequence.pending.max(window);
            assert!(
                start < sequence.tokens.len(),
                "la séquence {index} n'a pas de nouveau token"
            );
            entries.push((id, start..sequence.tokens.len()));
            sequence.pending = sequence.tokens.len();
            owners.push(index);
        }
        if entries.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let tape = Tape::new(self.backend);
        let batch: Vec<(SequenceId, &[u32])> = entries
            .iter()
            .zip(&owners)
            .map(|((id, range), &index)| (*id, &self.sequences[index].tokens[range.clone()]))
            .collect();
        let logits = self.model.forward_cached(&tape, &mut self.cache, &batch);
        // La dernière ligne de chaque séquence : seule elle prédit un token
        // qu'on ne connaît pas encore.
        let mut last = Vec::with_capacity(entries.len());
        let mut end = 0u32;
        for (_, range) in &entries {
            end += range.len() as u32;
            last.push(end - 1);
        }
        let rows = self.backend.gather_rows(&tape.value(logits), &last);
        (owners, self.backend.to_vec(&rows))
    }

    /// Un pas de génération : un token de plus pour chaque séquence en
    /// cours. Rend les tokens tirés, `(numéro, token)`. Les séquences qui
    /// tirent `stop` ou atteignent `max_len` sont terminées.
    pub fn step(&mut self, sampling: &Sampling, rng: &mut impl Rng) -> Vec<(usize, u32)> {
        let (owners, values) = self.forward();
        let vocab = self.model.config().vocab;
        let drawn: Vec<(usize, u32)> = owners
            .into_iter()
            .zip(values.chunks(vocab))
            .map(|(index, row)| {
                // En mode glouton, le plus grand logit suffit : pas besoin
                // de la distribution entière.
                let token = if sampling.temperature <= 0.0 {
                    argmax(row)
                } else {
                    let row: Vec<f64> = row.iter().map(|l| l.to_f64()).collect();
                    sample_logits(rng, &row, sampling)
                };
                (index, token as u32)
            })
            .collect();
        for &(index, token) in &drawn {
            self.sequences[index].tokens.push(token);
            let sequence = &self.sequences[index];
            let length = sequence.tokens.len();
            let limit = sequence.limit.unwrap_or(usize::MAX).min(self.max_len);
            if Some(token) == self.stop || length >= limit {
                self.finish(index);
            }
        }
        drawn
    }
}

/// La dernière ligne de logits `[N, V]`, rapatriée sur l'hôte. Le modèle
/// calcule les logits de toutes les positions qu'on lui donne ; seule la
/// dernière prédit un token qu'on ne connaît pas encore.
fn last_row<B: Backend>(backend: &B, logits: &B::Tensor) -> Vec<f64> {
    let rows = backend.shape(logits)[0];
    let last = backend.gather_rows(logits, &[rows as u32 - 1]);
    backend.to_vec(&last).iter().map(|l| l.to_f64()).collect()
}

/// Prolonge `prompt` de `length` tokens et renvoie toute la séquence.
pub fn generate<B: Backend>(
    backend: &B,
    model: &Gpt<B>,
    prompt: &[u32],
    length: usize,
    sampling: &Sampling,
    rng: &mut impl Rng,
) -> Vec<u32> {
    let mut generator = Generator::new(backend, model, prompt);
    generator.extend(length, sampling, rng);
    generator.into_tokens()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::Cpu;
    use crate::model::GptConfig;
    use crate::model::tests::{EXPERTS, LATENT, LOOPED, SHARED_KEYS, TINY, randomize};
    use crate::nn::Architecture;
    use crate::optim::AdamW;
    use crate::training::train_step;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Cpu<f64>;

    const GREEDY: Sampling = Sampling {
        temperature: 0.0,
        top_k: None,
        top_p: None,
    };

    /// Un modèle aux poids agrandis.
    fn model(backend: &B, config: GptConfig) -> Gpt<B> {
        let mut rng = StdRng::seed_from_u64(4);
        let mut model = Gpt::<B>::new(backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        model
    }

    /// Un modèle RoPE à `layers` couches et contexte 6.
    fn rotary_model(backend: &B, layers: usize) -> Gpt<B> {
        let config = GptConfig {
            context: 6,
            layers,
            ..TINY.with_architecture(Architecture::MODERN)
        };
        model(backend, config)
    }

    /// Un modèle à attention latente, à `layers` couches et contexte 6.
    fn latent_model(backend: &B, layers: usize) -> Gpt<B> {
        let config = GptConfig {
            context: 6,
            layers,
            ..TINY.with_architecture(LATENT)
        };
        model(backend, config)
    }

    /// Prolonge `prompt` de `length` tokens sans cache.
    fn generate_uncached(
        backend: &B,
        model: &Gpt<B>,
        prompt: &[u32],
        length: usize,
        sampling: &Sampling,
        rng: &mut StdRng,
    ) -> Vec<u32> {
        let mut generator = Generator::without_cache(backend, model, prompt);
        generator.extend(length, sampling, rng);
        generator.into_tokens()
    }

    fn assert_close(a: &[f64], b: &[f64], tolerance: f64, context: &str) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < tolerance, "{context} : {a} != {b}");
        }
    }

    /// Fait lire `tokens` un par un à un générateur avec cache et à un
    /// générateur sans, et compare leurs logits après chaque token.
    fn compare_token_by_token(backend: &B, model: &Gpt<B>, tokens: &[u32], tolerance: f64) {
        let context = model.config().context;
        let mut cached = Generator::new(backend, model, &tokens[..1]);
        let mut full = Generator::without_cache(backend, model, &tokens[..1]);
        for length in 1..=tokens.len() {
            let expected = full.logits().to_vec();
            assert_eq!(expected.len(), model.config().vocab);
            assert_close(
                cached.logits(),
                &expected,
                tolerance,
                &format!("longueur {length}"),
            );
            let (cache, sequence) = cached.cache().expect("le cache sert encore");
            assert_eq!(cache.len(sequence), length.min(context));
            assert_eq!(cache.next_position(sequence), length);
            if let Some(&next) = tokens.get(length) {
                cached.push(next);
                full.push(next);
            }
        }
    }

    #[test]
    fn cached_logits_match_a_full_forward_pass() {
        let backend = Cpu::new();
        let config = GptConfig {
            context: 8,
            layers: 2,
            ..TINY
        };
        let model = model(&backend, config);
        compare_token_by_token(&backend, &model, &[3, 7, 1, 9, 10, 0, 4, 4], 1e-10);
    }

    #[test]
    fn cached_logits_match_a_full_pass_with_rope() {
        let backend = Cpu::new();
        let model = rotary_model(&backend, 2);
        compare_token_by_token(&backend, &model, &[3, 7, 1, 9, 10, 0], 1e-10);
    }

    #[test]
    fn cached_logits_match_a_full_pass_with_latent_attention() {
        let backend = Cpu::new();
        let model = latent_model(&backend, 2);
        compare_token_by_token(&backend, &model, &[3, 7, 1, 9, 10, 0], 1e-10);
    }

    #[test]
    fn cached_logits_match_a_full_pass_with_experts() {
        // L'aiguillage se fait token par token : un token seul choisit les
        // mêmes experts qu'au milieu de sa séquence.
        let backend = Cpu::new();
        let config = GptConfig {
            context: 6,
            layers: 2,
            ..TINY.with_architecture(EXPERTS)
        };
        let model = model(&backend, config);
        compare_token_by_token(&backend, &model, &[3, 7, 1, 9, 10, 0], 1e-10);
    }

    #[test]
    fn cached_logits_match_a_full_pass_when_the_stack_is_looped() {
        // Chaque traversée a son étage de cache : la seconde ne voit pas
        // les mêmes états cachés que la première.
        let backend = Cpu::new();
        let config = GptConfig {
            context: 6,
            layers: 2,
            ..TINY.with_architecture(LOOPED)
        };
        let model = model(&backend, config);
        compare_token_by_token(&backend, &model, &[3, 7, 1, 9, 10, 0], 1e-10);
        // Deux couches, deux traversées : quatre étages de 16 nombres.
        let mut generator = Generator::new(&backend, &model, &[3, 7, 1]);
        generator.logits();
        let (cache, _) = generator.cache().unwrap();
        assert_eq!(cache.numbers_per_position(&backend), 4 * 16);
    }

    #[test]
    fn cached_logits_match_a_full_pass_with_shared_keys() {
        let backend = Cpu::new();
        let config = GptConfig {
            context: 6,
            ..SHARED_KEYS
        };
        let model = model(&backend, config);
        compare_token_by_token(&backend, &model, &[3, 7, 1, 9, 10, 0], 1e-10);
        // Trois couches, mais seules deux gardent des clés et des valeurs.
        let mut generator = Generator::new(&backend, &model, &[3, 7, 1]);
        generator.logits();
        let (cache, _) = generator.cache().unwrap();
        assert_eq!(cache.numbers_per_position(&backend), 2 * 16);
    }

    #[test]
    fn sliding_cache_is_exact_for_one_latent_layer() {
        // La partie tournée des clés suffit à garder la propriété de RoPE :
        // le cache glisse au-delà du contexte, comme pour l'attention
        // classique.
        let backend = Cpu::new();
        let model = latent_model(&backend, 1);
        let tokens = [3u32, 7, 1, 9, 10, 0, 4, 4, 2, 8, 5, 1, 6, 3];
        compare_token_by_token(&backend, &model, &tokens, 1e-9);
    }

    #[test]
    fn latent_cache_is_smaller_than_keys_and_values() {
        // Par position et par couche : une clé et une valeur par tête,
        // soit `2 × C = 16` nombres, contre un latent de 4 et une clé
        // tournée de 2.
        let backend = Cpu::new();
        let size = |model: &Gpt<B>| {
            let mut generator = Generator::new(&backend, model, &[3, 7, 1]);
            generator.logits();
            generator.cache().unwrap().0.numbers_per_position(&backend)
        };
        assert_eq!(size(&rotary_model(&backend, 2)), 2 * 16);
        assert_eq!(size(&latent_model(&backend, 2)), 2 * (4 + 2));
        // Un cache vide ne garde rien.
        assert_eq!(
            KvCache::<B>::for_model(&TINY, 1).numbers_per_position(&backend),
            0
        );
    }

    #[test]
    fn sliding_cache_is_exact_for_one_layer() {
        // Au-delà du contexte, le cache glisse : la plus ancienne position
        // est oubliée et les positions continuent de croître. Avec une seule
        // couche, les clés et valeurs ne dépendent que de leur propre token :
        // le résultat doit être celui d'un recalcul sur la fenêtre visible,
        // bien que les positions absolues ne soient pas les mêmes des deux
        // côtés. C'est la propriété de RoPE, vérifiée de bout en bout.
        let backend = Cpu::new();
        let model = rotary_model(&backend, 1);
        let tokens = [3u32, 7, 1, 9, 10, 0, 4, 4, 2, 8, 5, 1, 6, 3];
        compare_token_by_token(&backend, &model, &tokens, 1e-9);
    }

    #[test]
    fn a_prompt_goes_through_the_cache_in_one_pass() {
        // L'amorce d'un coup, ou token par token : mêmes logits, et le cache
        // finit dans le même état. Avec les deux sortes de positions.
        let backend = Cpu::new();
        let tokens = [3u32, 7, 1, 9, 10];
        for architecture in [Architecture::GPT2, Architecture::MODERN] {
            let config = GptConfig {
                context: 8,
                layers: 2,
                ..TINY.with_architecture(architecture)
            };
            let model = model(&backend, config);
            let name = format!("{architecture:?}");

            let mut at_once = Generator::new(&backend, &model, &tokens);
            let mut one_by_one = Generator::new(&backend, &model, &tokens[..1]);
            for &token in &tokens[1..] {
                one_by_one.logits();
                one_by_one.push(token);
            }
            let whole = at_once.logits().to_vec();
            assert_close(&whole, one_by_one.logits(), 1e-10, &name);
            let (cache, sequence) = at_once.cache().unwrap();
            assert_eq!((cache.len(sequence), cache.next_position(sequence)), (5, 5));
            // La suite donne les mêmes logits depuis les deux caches.
            at_once.push(4);
            one_by_one.push(4);
            let next = at_once.logits().to_vec();
            assert_close(&next, one_by_one.logits(), 1e-10, &name);
        }
    }

    #[test]
    fn tokens_pushed_together_go_through_together() {
        // Plusieurs tokens imposés entre deux lectures des logits : ils
        // traversent le modèle en une passe, y compris quand le cache doit
        // glisser de plusieurs positions d'un coup.
        let backend = Cpu::new();
        let model = rotary_model(&backend, 1);
        let tokens = [3u32, 7, 1, 9, 10, 0, 4, 4, 2];
        let mut generator = Generator::new(&backend, &model, &tokens[..4]);
        generator.logits();
        for &token in &tokens[4..] {
            generator.push(token);
        }
        let mut reference = Generator::without_cache(&backend, &model, &tokens);
        let expected = reference.logits().to_vec();
        assert_close(
            generator.logits(),
            &expected,
            1e-9,
            "trois positions oubliées",
        );
        let (cache, sequence) = generator.cache().unwrap();
        assert_eq!((cache.len(sequence), cache.next_position(sequence)), (6, 9));
    }

    #[test]
    fn more_new_tokens_than_the_context_restart_the_cache() {
        let backend = Cpu::new();
        let model = rotary_model(&backend, 2);
        let tokens = [3u32, 7, 1, 9, 10, 0, 4, 4, 2, 8];
        let mut generator = Generator::new(&backend, &model, &tokens[..2]);
        generator.logits();
        for &token in &tokens[2..] {
            generator.push(token);
        }
        // Huit nouveaux tokens pour un contexte de six : rien de l'ancien
        // cache n'est plus visible, et le résultat est celui d'un recalcul.
        let mut reference = Generator::without_cache(&backend, &model, &tokens);
        let expected = reference.logits().to_vec();
        assert_close(generator.logits(), &expected, 1e-9, "cache repris de zéro");
        assert_eq!(generator.cached_len().unwrap(), 6);
    }

    #[test]
    fn logits_are_computed_once_per_sequence() {
        let backend = Cpu::new();
        let model = rotary_model(&backend, 2);
        let mut generator = Generator::new(&backend, &model, &[3, 7]);
        let first = generator.logits().to_vec();
        // Une seconde lecture ne fait pas avancer le cache.
        assert_eq!(generator.logits(), &first[..]);
        assert_eq!(generator.cached_len().unwrap(), 2);
    }

    #[test]
    fn cached_generation_matches_uncached_generation() {
        let backend = Cpu::new();
        let model = model(&backend, TINY);
        // Amorces plus courte, égale et plus longue que le contexte (4), et
        // assez de tokens pour le dépasser : le cache est alors abandonné.
        for prompt in [&[3u32][..], &[3, 7, 1, 9], &[3, 7, 1, 9, 10, 0]] {
            for sampling in [&GREEDY, &Sampling::default()] {
                let mut rng = StdRng::seed_from_u64(1);
                let cached = generate(&backend, &model, prompt, 10, sampling, &mut rng);
                let mut rng = StdRng::seed_from_u64(1);
                let uncached = generate_uncached(&backend, &model, prompt, 10, sampling, &mut rng);
                assert_eq!(cached, uncached, "amorce {prompt:?}");
                assert_eq!(cached.len(), prompt.len() + 10);
            }
        }
    }

    #[test]
    fn learned_positions_give_up_the_cache_past_the_context() {
        let backend = Cpu::new();
        let model = model(&backend, TINY);
        let mut generator = Generator::new(&backend, &model, &[3]);
        let mut rng = StdRng::seed_from_u64(1);
        generator.extend(4, &GREEDY, &mut rng);
        assert_eq!(generator.cached_len(), Some(4));
        generator.extend(1, &GREEDY, &mut rng);
        assert!(generator.cache().is_none());
    }

    #[test]
    fn rotary_generation_uses_the_cache_past_the_context() {
        let backend = Cpu::new();
        // Une couche : avec et sans cache, mêmes tokens, y compris au-delà
        // du contexte (6) et pour une amorce plus longue que lui.
        let model = rotary_model(&backend, 1);
        for prompt in [&[3u32][..], &[3, 7, 1, 9, 10, 0, 4, 4]] {
            let mut rng = StdRng::seed_from_u64(1);
            let mut generator = Generator::new(&backend, &model, prompt);
            generator.extend(12, &GREEDY, &mut rng);
            assert_eq!(generator.cached_len(), Some(6));
            let uncached = generate_uncached(&backend, &model, prompt, 12, &GREEDY, &mut rng);
            assert_eq!(generator.tokens(), &uncached[..], "amorce {prompt:?}");
        }
        // Deux couches : identiques tant que la séquence tient dans le
        // contexte ; au-delà, le cache glissant garde un passé que le
        // recalcul n'a plus, et les deux peuvent diverger.
        let model = rotary_model(&backend, 2);
        let mut rng = StdRng::seed_from_u64(1);
        let cached = generate(&backend, &model, &[3], 12, &GREEDY, &mut rng);
        let uncached = generate_uncached(&backend, &model, &[3], 12, &GREEDY, &mut rng);
        assert_eq!(cached[..6], uncached[..6]);
        assert_eq!(cached.len(), 13);
    }

    #[test]
    fn generation_extends_the_prompt() {
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(0);
        let model = Gpt::<B>::new(&backend, TINY, &mut rng);
        let prompt = [3u32, 7];
        // 10 tokens : la séquence dépasse le contexte de 4 en cours de route.
        let tokens = generate(
            &backend,
            &model,
            &prompt,
            10,
            &Sampling::default(),
            &mut rng,
        );
        assert_eq!(tokens.len(), 12);
        assert_eq!(&tokens[..2], &prompt);
        assert!(tokens.iter().all(|&t| (t as usize) < TINY.vocab));
    }

    #[test]
    fn a_generation_can_be_followed_and_extended() {
        // Token par token, puis prolongée : la même séquence qu'en une fois.
        let backend = Cpu::new();
        let model = rotary_model(&backend, 2);
        let sampling = Sampling::default();
        let mut rng = StdRng::seed_from_u64(5);
        let whole = generate(&backend, &model, &[3, 7], 9, &sampling, &mut rng);

        let mut rng = StdRng::seed_from_u64(5);
        let mut generator = Generator::new(&backend, &model, &[3, 7]);
        let mut seen = vec![3, 7];
        for _ in 0..4 {
            seen.push(generator.next_token(&sampling, &mut rng));
        }
        assert_eq!(generator.tokens(), &seen[..]);
        generator.extend(5, &sampling, &mut rng);
        assert_eq!(generator.into_tokens(), whole);
    }

    #[test]
    fn greedy_generation_is_deterministic() {
        let backend = Cpu::new();
        let model = model(&backend, TINY);
        let first = generate(
            &backend,
            &model,
            &[3],
            8,
            &GREEDY,
            &mut StdRng::seed_from_u64(1),
        );
        let second = generate(
            &backend,
            &model,
            &[3],
            8,
            &GREEDY,
            &mut StdRng::seed_from_u64(2),
        );
        assert_eq!(first, second);
    }

    #[test]
    fn generation_repeats_what_was_memorized() {
        // Après sur-apprentissage d'une séquence, la génération gloutonne
        // doit la réciter.
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(0);
        let mut model = Gpt::new(&backend, TINY, &mut rng);
        let mut optimizer = AdamW::new(0.01, 0.0);
        let tokens = [3u32, 7, 1, 9];
        let targets = [7u32, 1, 9, 2];
        for _ in 0..300 {
            train_step(&backend, &mut model, &mut optimizer, |m, t| {
                m.loss(t, &tokens, &targets, 1)
            });
        }
        let generated = generate(&backend, &model, &[3], 4, &GREEDY, &mut rng);
        assert_eq!(generated, vec![3, 7, 1, 9, 2]);
    }

    #[test]
    #[should_panic(expected = "a besoin d'une amorce")]
    fn an_empty_prompt_is_refused() {
        let backend = Cpu::new();
        let model = model(&backend, TINY);
        Generator::new(&backend, &model, &[]);
    }

    /// Les séquences d'un lot, chacune comparée à sa génération seule.
    fn batch_matches_alone(model: &Gpt<B>, prompts: &[&[u32]], length: usize) {
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(1);
        let mut batch = Batch::new(&backend, model, prompts.len());
        let numbers: Vec<usize> = prompts.iter().map(|prompt| batch.add(prompt)).collect();
        for _ in 0..length {
            batch.step(&GREEDY, &mut rng);
        }
        for (prompt, number) in prompts.iter().zip(numbers) {
            let alone = generate(&backend, model, prompt, length, &GREEDY, &mut rng);
            assert_eq!(batch.tokens(number), &alone[..], "amorce {prompt:?}");
        }
    }

    #[test]
    fn each_sequence_of_a_batch_is_generated_as_if_alone() {
        // Des amorces de longueurs différentes, et assez de tokens pour que
        // la plus longue dépasse le contexte (6) : son cache glisse, pas
        // celui des autres.
        let backend = Cpu::new();
        let prompts: [&[u32]; 3] = [&[3], &[3, 7, 1, 9], &[10, 0]];
        batch_matches_alone(&rotary_model(&backend, 2), &prompts, 5);
        batch_matches_alone(&latent_model(&backend, 2), &prompts, 5);
        let config = GptConfig {
            context: 9,
            layers: 2,
            ..TINY
        };
        batch_matches_alone(&model(&backend, config), &prompts, 5);
    }

    #[test]
    fn a_finished_sequence_gives_its_place_to_a_new_one() {
        let backend = Cpu::new();
        let model = rotary_model(&backend, 2);
        let mut rng = StdRng::seed_from_u64(1);
        // Le cache ne suit qu'une séquence : la seconde ne peut entrer
        // qu'une fois la première terminée.
        let mut batch = Batch::new(&backend, &model, 1);
        batch.max_len = 4;
        let first = batch.add(&[3, 7]);
        batch.step(&GREEDY, &mut rng);
        assert!(batch.is_running(first));
        batch.step(&GREEDY, &mut rng);
        assert!(!batch.is_running(first) && batch.running() == 0);
        let second = batch.add(&[10, 0, 4]);
        batch.step(&GREEDY, &mut rng);
        let alone = generate(&backend, &model, &[10, 0, 4], 1, &GREEDY, &mut rng);
        assert_eq!(batch.tokens(second), &alone[..]);
        assert_eq!(batch.tokens(first).len(), 4);
    }

    #[test]
    fn a_sequence_stops_on_its_stop_token() {
        let backend = Cpu::new();
        let model = rotary_model(&backend, 2);
        let mut rng = StdRng::seed_from_u64(1);
        let alone = generate(&backend, &model, &[3, 7], 3, &GREEDY, &mut rng);
        let mut batch = Batch::new(&backend, &model, 2);
        batch.stop = Some(alone[2]);
        let stopping = batch.add(&[3, 7]);
        let other = batch.add(&[10, 0]);
        batch.step(&GREEDY, &mut rng);
        assert_eq!(batch.tokens(stopping), &alone[..3]);
        assert!(!batch.is_running(stopping));
        // L'autre continue, à moins d'avoir tiré le même token.
        assert_eq!(batch.is_running(other), batch.tokens(other)[2] != alone[2]);
    }

    #[test]
    fn the_cache_size_is_known_from_the_configuration() {
        let backend = Cpu::new();
        let models = [
            rotary_model(&backend, 2),
            latent_model(&backend, 2),
            model(
                &backend,
                GptConfig {
                    context: 6,
                    ..SHARED_KEYS
                },
            ),
            model(
                &backend,
                GptConfig {
                    context: 6,
                    layers: 2,
                    ..TINY.with_architecture(LOOPED)
                },
            ),
        ];
        for model in &models {
            let mut generator = Generator::new(&backend, model, &[3, 7, 1]);
            generator.logits();
            let measured = generator.cache().unwrap().0.numbers_per_position(&backend);
            assert_eq!(model.config().cache_numbers_per_position(), measured);
        }
    }
}
