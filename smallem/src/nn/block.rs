//! Le bloc de Transformer : une attention et un feed-forward, chacun
//! précédé d'une normalisation et contourné par une connexion résiduelle.

use rand::Rng;

use super::{
    Architecture, Attention, FeedForwardLayer, INIT_STD, Lent, Linear, Norm, Rope, module,
};
use crate::autograd::{Tape, Var};
use crate::backend::Backend;
use crate::cache::{AttentionCache, CacheStep};

/// Un bloc de Transformer, en version « pré-norme » : la normalisation est
/// appliquée à l'entrée de chaque sous-couche, pas à sa sortie.
///
/// ```text
/// x = x + attention(ln1(x))
/// x = x + feed_forward(ln2(x))
/// ```
///
/// Les deux `x + …` sont les connexions résiduelles : chaque sous-couche
/// n'apprend qu'une correction à ajouter, et le gradient dispose d'un
/// chemin direct de la sortie du modèle jusqu'à son entrée.
pub struct Block<B: Backend> {
    pub ln1: Norm<B>,
    pub attention: Attention<B>,
    pub ln2: Norm<B>,
    pub feed_forward: FeedForwardLayer<B>,
}

impl<B: Backend> Block<B> {
    /// `layers` est le nombre total de blocs du modèle. Les projections qui
    /// écrivent dans le flux résiduel sont initialisées avec un écart-type
    /// divisé par `√(2 × layers)` : `2 × layers` sous-couches s'additionnent
    /// dans ce flux, et sans cela sa variance grandirait avec la profondeur.
    ///
    /// Avec `borrows`, l'attention du bloc emprunte ses clés et ses valeurs
    /// à un bloc précédent (voir [`Architecture::own_keys`]).
    pub fn new(
        backend: &B,
        dim: usize,
        heads: usize,
        layers: usize,
        architecture: Architecture,
        borrows: bool,
        rng: &mut impl Rng,
    ) -> Self {
        let proj_std = INIT_STD / (2.0 * layers as f64).sqrt();
        Self {
            ln1: Norm::new(backend, architecture.norm, dim),
            attention: Attention::new(
                backend,
                architecture.attention,
                borrows,
                dim,
                heads,
                proj_std,
                rng,
            ),
            ln2: Norm::new(backend, architecture.norm, dim),
            feed_forward: FeedForwardLayer::new(
                backend,
                architecture.feed_forward,
                architecture.experts,
                dim,
                proj_std,
                rng,
            ),
        }
    }

    /// `x` contient les vecteurs de `N` positions, `[B, N, C]`. `rope`
    /// donne les angles de ces positions si le modèle utilise RoPE. `cache`
    /// est celui de [`Attention::forward`] : en génération, ce que la
    /// couche garde du passé, et le pas qui décrit les nouveaux tokens.
    /// `lent` porte, d'un bloc au suivant, les clés et les valeurs du
    /// dernier bloc qui en a calculé.
    pub fn forward(
        &self,
        tape: &Tape<B>,
        x: Var,
        rope: Option<Rope>,
        cache: Option<(&mut AttentionCache<B>, &CacheStep)>,
        lent: &mut Option<Lent>,
    ) -> Var {
        let attended = self
            .attention
            .forward(tape, self.ln1.forward(tape, x), rope, cache, lent);
        let x = tape.add(x, attended);
        let transformed = self.feed_forward.forward(tape, self.ln2.forward(tape, x));
        tape.add(x, transformed)
    }
}

impl<B: Backend> Block<B> {
    /// Les couches linéaires du bloc : celles de l'attention, puis celles
    /// du feed-forward. Les projections sans biais de l'attention latente
    /// et les experts n'y sont pas.
    pub fn linears_mut(&mut self) -> Vec<&mut Linear<B>> {
        let mut linears = self.attention.linears_mut();
        linears.extend(self.feed_forward.linears_mut());
        linears
    }
}

module!(Block {
    ln1,
    attention,
    ln2,
    feed_forward
});
