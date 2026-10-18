//! Les normalisations : LayerNorm, celle de GPT-2, et RMSNorm, celle des
//! modèles récents.

use super::{NormKind, Parameter, module};
use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};

/// Normalisation de chaque vecteur du dernier axe : on lui retire sa
/// moyenne et on le divise par son écart-type, puis un gain et un biais
/// appris lui redonnent l'échelle qui convient.
///
/// Elle empêche les valeurs de grossir ou de s'éteindre au fil des couches.
pub struct LayerNorm<B: Backend> {
    /// `[dimension]`, initialisé à 1.
    pub gain: Parameter<B>,
    /// `[dimension]`, initialisé à 0.
    pub bias: Parameter<B>,
}

impl<B: Backend> LayerNorm<B> {
    /// Ajouté à la variance pour ne jamais diviser par zéro.
    pub const EPSILON: f64 = 1e-5;

    pub fn new(backend: &B, dim: usize) -> Self {
        Self {
            gain: Parameter::ones(backend, &[dim]),
            bias: Parameter::zeros(backend, &[dim]),
        }
    }

    /// `[…, dimension]` donne la même forme.
    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        let normalized = tape.normalize(x, B::Scalar::from_f64(Self::EPSILON));
        tape.add(
            tape.mul(normalized, self.gain.var(tape)),
            self.bias.var(tape),
        )
    }
}

module!(LayerNorm { gain, bias });

/// RMSNorm : chaque vecteur du dernier axe est divisé par la racine de la
/// moyenne de ses carrés, puis multiplié par un gain appris.
///
/// C'est une LayerNorm allégée. Elle ne retire pas la moyenne et n'a pas de
/// biais : l'expérience a montré que c'est la mise à l'échelle qui
/// stabilise l'entraînement, pas le recentrage.
pub struct RmsNorm<B: Backend> {
    /// `[dimension]`, initialisé à 1.
    pub gain: Parameter<B>,
}

impl<B: Backend> RmsNorm<B> {
    /// Ajouté à la moyenne des carrés pour ne jamais diviser par zéro.
    pub const EPSILON: f64 = 1e-5;

    pub fn new(backend: &B, dim: usize) -> Self {
        Self {
            gain: Parameter::ones(backend, &[dim]),
        }
    }

    /// `[…, dimension]` donne la même forme.
    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        let normalized = tape.rms_normalize(x, B::Scalar::from_f64(Self::EPSILON));
        tape.mul(normalized, self.gain.var(tape))
    }
}

module!(RmsNorm { gain });

/// La normalisation d'un bloc, selon l'architecture choisie.
pub enum Norm<B: Backend> {
    Layer(LayerNorm<B>),
    Rms(RmsNorm<B>),
}

impl<B: Backend> Norm<B> {
    pub fn new(backend: &B, kind: NormKind, dim: usize) -> Self {
        match kind {
            NormKind::Layer => Self::Layer(LayerNorm::new(backend, dim)),
            NormKind::Rms => Self::Rms(RmsNorm::new(backend, dim)),
        }
    }

    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        match self {
            Self::Layer(norm) => norm.forward(tape, x),
            Self::Rms(norm) => norm.forward(tape, x),
        }
    }
}

module!(
    enum Norm {
        Layer,
        Rms,
    }
);
