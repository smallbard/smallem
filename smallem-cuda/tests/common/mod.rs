//! Raccourcis communs aux tests GPU : les opérations appelées par leur nom,
//! comme des méthodes du backend.

#![allow(dead_code)]

use smallem::backend::ops::{
    CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu, Normalize, RmsNormalize,
    RotatePairs, Silu,
};
use smallem::backend::{Backend, Float};

/// Le gradient de la seule entrée qui en reçoit un.
fn first<T>(mut grads: Vec<Option<T>>) -> T {
    grads
        .swap_remove(0)
        .expect("gradient de la première entrée")
}

pub trait Shortcuts: Backend {
    fn gelu(&self, x: &Self::Tensor) -> Self::Tensor {
        self.run(&Gelu, &[x])
    }

    fn gelu_backward(&self, x: &Self::Tensor, grad: &Self::Tensor) -> Self::Tensor {
        first(self.run_backward(&Gelu, &[x], x, grad))
    }

    fn silu(&self, x: &Self::Tensor) -> Self::Tensor {
        self.run(&Silu, &[x])
    }

    fn silu_backward(&self, x: &Self::Tensor, grad: &Self::Tensor) -> Self::Tensor {
        first(self.run_backward(&Silu, &[x], x, grad))
    }

    fn normalize(&self, x: &Self::Tensor, epsilon: f64) -> Self::Tensor {
        self.run(&Normalize { epsilon }, &[x])
    }

    fn normalize_backward(
        &self,
        x: &Self::Tensor,
        normalized: &Self::Tensor,
        grad: &Self::Tensor,
        epsilon: f64,
    ) -> Self::Tensor {
        first(self.run_backward(&Normalize { epsilon }, &[x], normalized, grad))
    }

    fn rms_normalize(&self, x: &Self::Tensor, epsilon: f64) -> Self::Tensor {
        self.run(&RmsNormalize { epsilon }, &[x])
    }

    fn rms_normalize_backward(
        &self,
        x: &Self::Tensor,
        normalized: &Self::Tensor,
        grad: &Self::Tensor,
        epsilon: f64,
    ) -> Self::Tensor {
        first(self.run_backward(&RmsNormalize { epsilon }, &[x], normalized, grad))
    }

    /// Avec `inverse`, la rotation dans l'autre sens : c'est la
    /// rétropropagation de l'opération, appliquée à `x`.
    fn rotate_pairs(
        &self,
        x: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
        inverse: bool,
    ) -> Self::Tensor {
        if inverse {
            first(self.run_backward(&RotatePairs, &[x, cos, sin], x, x))
        } else {
            self.run(&RotatePairs, &[x, cos, sin])
        }
    }

    fn concat(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        self.run(&Concat, &[a, b])
    }

    /// Les deux morceaux du gradient d'une concaténation de `a` et `b`.
    fn concat_backward(
        &self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> (Self::Tensor, Self::Tensor) {
        let mut grads = self.run_backward(&Concat, &[a, b], grad, grad).into_iter();
        let mut part = || grads.next().flatten().expect("gradient d'une entrée");
        (part(), part())
    }

    fn causal_softmax(&self, scores: &Self::Tensor, scale: f64) -> Self::Tensor {
        self.run(&CausalSoftmax { scale }, &[scores])
    }

    fn softmax_backward(
        &self,
        probabilities: &Self::Tensor,
        grad: &Self::Tensor,
        scale: f64,
    ) -> Self::Tensor {
        let op = CausalSoftmax { scale };
        first(self.run_backward(&op, &[probabilities], probabilities, grad))
    }

    fn attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        scale: f64,
    ) -> Self::Tensor {
        self.run(&FusedAttention { scale }, &[q, k, v])
    }

    /// Le gradient de l'entrée `input` (0 pour `q`, 1 pour `k`, 2 pour
    /// `v`), par le même chemin que la tape : avec ce que la passe avant a
    /// gardé.
    #[allow(clippy::too_many_arguments)]
    fn attention_backward(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        grad: &Self::Tensor,
        scale: f64,
        input: usize,
    ) -> Self::Tensor {
        let op = FusedAttention { scale };
        let (output, saved) = self.run_saving(&op, &[q, k, v]);
        let mut grads = self.run_backward_saved(&op, &[q, k, v], &output, saved.as_ref(), grad);
        grads.swap_remove(input).expect("gradient de l'attention")
    }

    fn cross_entropy(&self, logits: &Self::Tensor, targets: &[u32]) -> Self::Tensor {
        let targets = targets.to_vec();
        self.run(&CrossEntropy { targets }, &[logits])
    }

    /// Gradient de la perte par rapport aux logits.
    fn cross_entropy_grad(&self, logits: &Self::Tensor, targets: &[u32]) -> Self::Tensor {
        let op = CrossEntropy {
            targets: targets.to_vec(),
        };
        let loss = self.run(&op, &[logits]);
        let one = self.scalar(Self::Scalar::ONE);
        first(self.run_backward(&op, &[logits], &loss, &one))
    }
}

impl<B: Backend> Shortcuts for B {}
