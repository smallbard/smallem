//! La prédiction multi-token (MTP) de DeepSeek-V3 : un module d'appoint qui,
//! pendant l'entraînement, prédit le token d'après le suivant.

use rand::Rng;

use super::{Architecture, Block, Projection, RmsNorm, Rope, module};
use crate::autograd::{Tape, Var};
use crate::backend::Backend;
use crate::cache::{AttentionCache, CacheStep};

/// Le module qui prédit un token plus loin que le modèle principal.
///
/// À la position `i`, le modèle principal a lu les tokens jusqu'à `t_i` et
/// prédit `t_{i+1}`. Ce module reprend l'état caché `h_i` du modèle, y joint
/// l'embedding du vrai token suivant `t_{i+1}`, et prédit `t_{i+2}` :
///
/// ```text
/// x      = [rms(h_i) ; rms(embedding(t_{i+1}))] · W      de 2C à C
/// h'_i   = bloc(x)                                        un bloc de Transformer
/// logits = sortie du modèle principal, appliquée à h'_i   embeddings et tête partagés
/// ```
///
/// Il ne sert qu'à l'entraînement. Son gradient remonte dans `h_i`, donc
/// dans tout le modèle : chaque état caché est poussé à contenir de quoi
/// prévoir deux tokens, pas un seul. Pour générer, le module est retiré, et
/// le modèle principal fonctionne sans lui.
pub struct MultiTokenHead<B: Backend> {
    pub hidden_norm: RmsNorm<B>,
    pub embedding_norm: RmsNorm<B>,
    /// `2C → C`
    pub proj: Projection<B>,
    pub block: Block<B>,
}

impl<B: Backend> MultiTokenHead<B> {
    /// `layers` et `architecture` sont ceux du modèle principal : le bloc
    /// du module est fait comme les siens.
    pub fn new(
        backend: &B,
        dim: usize,
        heads: usize,
        layers: usize,
        architecture: Architecture,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            hidden_norm: RmsNorm::new(backend, dim),
            embedding_norm: RmsNorm::new(backend, dim),
            proj: Projection::new(backend, 2 * dim, dim, rng),
            block: Block::new(backend, dim, heads, layers, architecture, false, rng),
        }
    }

    /// `hidden` contient les états cachés du modèle principal et `next`
    /// les embeddings des tokens qui suivent, `[B, N, C]` tous les deux.
    /// Renvoie `[B, N, C]`, à passer dans la sortie du modèle principal.
    pub fn forward(&self, tape: &Tape<B>, hidden: Var, next: Var, rope: Option<Rope>) -> Var {
        let joined = tape.concat(
            self.hidden_norm.forward(tape, hidden),
            self.embedding_norm.forward(tape, next),
        );
        self.block
            .forward(tape, self.proj.forward(tape, joined), rope, None, &mut None)
    }

    /// La même chose en génération : `hidden` et `next` sont `[1, n, C]`,
    /// pour `n` nouvelles positions que décrit `step`, et l'attention du
    /// bloc lit le passé dans `cache`.
    pub fn forward_cached(
        &self,
        tape: &Tape<B>,
        hidden: Var,
        next: Var,
        cache: &mut AttentionCache<B>,
        step: &CacheStep,
    ) -> Var {
        let joined = tape.concat(
            self.hidden_norm.forward(tape, hidden),
            self.embedding_norm.forward(tape, next),
        );
        let x = self.proj.forward(tape, joined);
        self.block
            .forward(tape, x, None, Some((cache, step)), &mut None)
    }
}

module!(MultiTokenHead {
    hidden_norm,
    embedding_norm,
    proj,
    block
});
