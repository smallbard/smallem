//! LoRA : ajuster un modèle en n'entraînant qu'une petite correction de rang
//! faible de chacune de ses matrices.

use rand::Rng;

use super::{Parameter, module};
use crate::autograd::{Tape, Var};
use crate::backend::Backend;

/// Une correction `ΔW = A · B` d'une matrice `W` `[entrée, sortie]`, avec
/// `A` `[entrée, r]` et `B` `[r, sortie]`, et `r` bien plus petit que les
/// deux dimensions (Hu et al., 2021).
///
/// `W` reste gelée ; seuls `A` et `B` apprennent : `r × (entrée + sortie)`
/// paramètres au lieu de `entrée × sortie`. `B` part de zéro, si bien que le
/// modèle de départ est exactement inchangé ; `A` part du hasard, sans quoi
/// le gradient de `B`, qui passe par `A`, serait nul lui aussi.
pub struct Adapter<B: Backend> {
    /// `A`, `[entrée, r]`.
    pub down: Parameter<B>,
    /// `B`, `[r, sortie]`.
    pub up: Parameter<B>,
}

impl<B: Backend> Adapter<B> {
    pub fn new(backend: &B, input: usize, output: usize, rank: usize, rng: &mut impl Rng) -> Self {
        Self {
            down: Parameter::normal(backend, &[input, rank], 1.0 / (input as f64).sqrt(), rng),
            up: Parameter::zeros(backend, &[rank, output]),
        }
    }

    /// `x · A · B`.
    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        tape.matmul(tape.matmul(x, self.down.var(tape)), self.up.var(tape))
    }

    /// `A · B`, la correction elle-même, `[entrée, sortie]`.
    pub fn delta(&self, tape: &Tape<B>) -> Var {
        tape.matmul(self.down.var(tape), self.up.var(tape))
    }

    /// Ajoute la correction à `weight`, sur place : `W ← W + A · B`.
    pub fn merge_into(&self, backend: &B, weight: &mut Parameter<B>) {
        let delta = backend.matmul(self.down.value(), self.up.value());
        let merged = backend.add(weight.value(), &delta);
        weight.set_value(merged);
    }
}

module!(Adapter { down, up });
