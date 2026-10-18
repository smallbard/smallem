//! Modèle de langage à contexte fixe (Bengio et al., 2003) : les `context`
//! tokens précédents prédisent le suivant.
//!
//! ```text
//! tokens [B, n] → embeddings [B, n, d] → mis bout à bout [B, n·d]
//!               → couche cachée + tanh [B, h] → logits [B, V]
//! ```

use rand::Rng;

use crate::autograd::{Tape, Var};
use crate::backend::Backend;
use crate::data::Batch;
use crate::nn::{Embedding, Linear, module};

#[derive(Clone, Copy, Debug)]
pub struct MlpConfig {
    pub vocab: usize,
    /// Nombre de tokens que le modèle voit pour prédire le suivant.
    pub context: usize,
    /// Dimension du vecteur associé à chaque token.
    pub embed: usize,
    /// Largeur de la couche cachée.
    pub hidden: usize,
}

pub struct Mlp<B: Backend> {
    config: MlpConfig,
    embedding: Embedding<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(backend: &B, config: MlpConfig, rng: &mut impl Rng) -> Self {
        Self {
            config,
            embedding: Embedding::new(backend, config.vocab, config.embed, rng),
            hidden: Linear::new(backend, config.context * config.embed, config.hidden, rng),
            output: Linear::new(backend, config.hidden, config.vocab, rng),
        }
    }

    pub fn config(&self) -> MlpConfig {
        self.config
    }

    /// `contexts` contient `B × context` tokens, un contexte après l'autre.
    /// Renvoie les logits `[B, vocab]` du token qui suit chaque contexte.
    pub fn logits(&self, tape: &Tape<B>, contexts: &[u32]) -> Var {
        let MlpConfig { context, embed, .. } = self.config;
        assert!(
            contexts.len().is_multiple_of(context),
            "{} tokens ne forment pas des contextes de {context}",
            contexts.len(),
        );
        let batch = contexts.len() / context;

        let embedded = self.embedding.forward(tape, contexts);
        // Les `context` vecteurs d'un même exemple sont mis bout à bout : la
        // couche cachée voit ainsi quel token occupe quelle position.
        let flat = tape.reshape(embedded, &[batch, context * embed]);
        let hidden = tape.tanh(self.hidden.forward(tape, flat));
        self.output.forward(tape, hidden)
    }

    /// Entropie croisée moyenne entre les prédictions et les `B` cibles.
    pub fn loss(&self, tape: &Tape<B>, contexts: &[u32], targets: &[u32]) -> Var {
        tape.cross_entropy(self.logits(tape, contexts), targets)
    }
}

module!(Mlp {
    embedding,
    hidden,
    output
});

/// Transforme un lot de fenêtres en exemples pour ce modèle : chaque ligne
/// de `inputs` est un contexte, et sa cible est le token qui suit la
/// fenêtre, c'est-à-dire la dernière cible de la ligne.
pub fn next_token_examples(batch: &Batch) -> (Vec<u32>, Vec<u32>) {
    let contexts = batch.inputs.iter().map(|&t| u32::from(t)).collect();
    let targets = batch
        .targets
        .chunks(batch.context)
        .map(|row| u32::from(row[batch.context - 1]))
        .collect();
    (contexts, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::Cpu;
    use crate::data::Dataset;
    use crate::nn::Module;
    use crate::optim::AdamW;
    use crate::training::{evaluate, train_step};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const CONFIG: MlpConfig = MlpConfig {
        vocab: 257,
        context: 4,
        embed: 8,
        hidden: 32,
    };

    /// Un petit corpus périodique : `0, 1, …, 9, 0, 1, …`
    fn batch(rng: &mut StdRng) -> (Vec<u32>, Vec<u32>) {
        let dataset = Dataset::from_tokens((0..500).map(|i| i % 10).collect());
        next_token_examples(&dataset.sample_batch(rng, 16, CONFIG.context))
    }

    #[test]
    fn examples_take_the_token_after_the_window() {
        let dataset = Dataset::from_tokens((0..100).collect());
        let batch = dataset.sample_batch(&mut StdRng::seed_from_u64(0), 3, 4);
        let (contexts, targets) = next_token_examples(&batch);
        assert_eq!(contexts.len(), 12);
        assert_eq!(targets.len(), 3);
        for (row, target) in contexts.chunks(4).zip(&targets) {
            assert_eq!(*target, row[3] + 1);
        }
    }

    #[test]
    fn parameter_count_matches_the_architecture() {
        let backend = Cpu::<f32>::new();
        let model = Mlp::new(&backend, CONFIG, &mut StdRng::seed_from_u64(0));
        let expected = 257 * 8 + (4 * 8 * 32 + 32) + (32 * 257 + 257);
        assert_eq!(model.parameter_count(&backend), expected);
    }

    #[test]
    fn initial_loss_is_ln_vocab() {
        // Des poids minuscules donnent des logits presque nuls, donc une
        // distribution presque uniforme sur le vocabulaire.
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(0);
        let model = Mlp::new(&backend, CONFIG, &mut rng);
        let (contexts, targets) = batch(&mut rng);
        let loss = evaluate(&backend, &model, |m, t| m.loss(t, &contexts, &targets));
        assert!((loss - 257f64.ln()).abs() < 0.01, "perte initiale {loss}");
    }

    #[test]
    fn overfits_a_single_batch() {
        // En répétant le même lot, le modèle doit finir par le connaître par
        // cœur. Sinon il y a un bug dans le forward, le backward ou
        // l'optimiseur.
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(0);
        let mut model = Mlp::new(&backend, CONFIG, &mut rng);
        let mut optimizer = AdamW::new(0.01, 0.0);
        let (contexts, targets) = batch(&mut rng);

        for _ in 0..300 {
            train_step(&backend, &mut model, &mut optimizer, |m, t| {
                m.loss(t, &contexts, &targets)
            });
        }
        let loss = evaluate(&backend, &model, |m, t| m.loss(t, &contexts, &targets));
        assert!(loss < 0.01, "perte après sur-apprentissage : {loss}");
    }
}
