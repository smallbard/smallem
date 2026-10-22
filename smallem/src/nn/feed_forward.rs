//! Le petit réseau de chaque bloc, dans ses deux variantes : GELU, celle de
//! GPT-2, et SwiGLU, celle des modèles récents.

use rand::Rng;

use super::{BlockShape, FeedForwardKind, Linear, MixtureOfExperts, module};
use crate::autograd::{Tape, Var};
use crate::backend::Backend;

/// Le petit réseau de chaque bloc : il transforme chaque position
/// séparément, en passant par une couche quatre fois plus large.
pub struct FeedForward<B: Backend> {
    /// `C → 4C`
    pub expand: Linear<B>,
    /// `4C → C`
    pub proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(backend: &B, dim: usize, proj_std: f64, rng: &mut impl Rng) -> Self {
        Self {
            expand: Linear::new(backend, dim, 4 * dim, rng),
            proj: Linear::with_std(backend, 4 * dim, dim, proj_std, rng),
        }
    }

    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        self.proj
            .forward(tape, tape.gelu(self.expand.forward(tape, x)))
    }
}

module!(FeedForward { expand, proj });

/// SwiGLU, le feed-forward des modèles récents :
///
/// ```text
/// sortie = (silu(x · W_porte) × (x · W_valeur)) · W_proj
/// ```
///
/// Deux projections de l'entrée sont multipliées l'une par l'autre : la
/// première, passée par SiLU, sert de porte à la seconde. Le réseau peut
/// ainsi laisser passer ou éteindre chaque composante selon l'entrée.
///
/// Il y a trois matrices au lieu de deux. Pour garder le même nombre de
/// paramètres qu'un feed-forward de largeur `4C`, la largeur cachée est
/// ramenée à `8C/3`.
pub struct SwiGluFeedForward<B: Backend> {
    /// `C → h`, passée par SiLU.
    pub gate: Linear<B>,
    /// `C → h`
    pub up: Linear<B>,
    /// `h → C`
    pub proj: Linear<B>,
}

/// Largeur cachée d'un SwiGLU de dimension `dim` : `8C/3`, arrondi au
/// multiple de 8 supérieur.
pub fn swiglu_hidden(dim: usize) -> usize {
    (8 * dim).div_ceil(3).next_multiple_of(8)
}

impl<B: Backend> SwiGluFeedForward<B> {
    pub fn new(backend: &B, dim: usize, proj_std: f64, rng: &mut impl Rng) -> Self {
        Self::with_hidden(backend, dim, swiglu_hidden(dim), proj_std, rng)
    }

    /// Comme [`SwiGluFeedForward::new`], avec une largeur cachée choisie.
    pub fn with_hidden(
        backend: &B,
        dim: usize,
        hidden: usize,
        proj_std: f64,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            gate: Linear::new(backend, dim, hidden, rng),
            up: Linear::new(backend, dim, hidden, rng),
            proj: Linear::with_std(backend, hidden, dim, proj_std, rng),
        }
    }

    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        let gate = tape.silu(self.gate.forward(tape, x));
        self.proj
            .forward(tape, tape.mul(gate, self.up.forward(tape, x)))
    }
}

module!(SwiGluFeedForward { gate, up, proj });

/// Le feed-forward d'un bloc, selon l'architecture choisie.
pub enum FeedForwardLayer<B: Backend> {
    Gelu(FeedForward<B>),
    SwiGlu(SwiGluFeedForward<B>),
    Experts(MixtureOfExperts<B>),
}

impl<B: Backend> FeedForwardLayer<B> {
    /// Un mélange d'experts si l'architecture en a un ; sinon son
    /// feed-forward.
    pub fn new(backend: &B, shape: &BlockShape, rng: &mut impl Rng) -> Self {
        let (dim, proj_std) = (shape.dim, shape.proj_std());
        if let Some(dims) = shape.architecture.experts {
            return Self::Experts(MixtureOfExperts::new(backend, dim, dims, proj_std, rng));
        }
        match shape.architecture.feed_forward {
            FeedForwardKind::Gelu => Self::Gelu(FeedForward::new(backend, dim, proj_std, rng)),
            FeedForwardKind::SwiGlu => {
                Self::SwiGlu(SwiGluFeedForward::new(backend, dim, proj_std, rng))
            }
        }
    }

    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        match self {
            Self::Gelu(layer) => layer.forward(tape, x),
            Self::SwiGlu(layer) => layer.forward(tape, x),
            Self::Experts(layer) => layer.forward(tape, x),
        }
    }

    /// Les couches linéaires du feed-forward ; aucune pour un mélange
    /// d'experts.
    pub fn linears_mut(&mut self) -> Vec<&mut Linear<B>> {
        match self {
            Self::Gelu(layer) => vec![&mut layer.expand, &mut layer.proj],
            Self::SwiGlu(layer) => vec![&mut layer.gate, &mut layer.up, &mut layer.proj],
            Self::Experts(_) => Vec::new(),
        }
    }

    /// Le mélange d'experts, si c'en est un.
    pub fn experts(&self) -> Option<&MixtureOfExperts<B>> {
        match self {
            Self::Experts(layer) => Some(layer),
            _ => None,
        }
    }

    pub fn experts_mut(&mut self) -> Option<&mut MixtureOfExperts<B>> {
        match self {
            Self::Experts(layer) => Some(layer),
            _ => None,
        }
    }
}

module!(
    enum FeedForwardLayer {
        Gelu,
        SwiGlu,
        Experts,
    }
);
