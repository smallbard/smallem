//! La normalisation de RMSNorm, sans son gain.

use super::{Operation, Reference, operands};
use crate::backend::{Float, Primitives};

/// Divise chaque vecteur du dernier axe par la racine de la moyenne de ses
/// carrés : `x / √(moyenne(x²) + epsilon)`. Une entrée.
///
/// Contrairement à [`Normalize`](super::Normalize), la moyenne n'est pas
/// retirée.
pub struct RmsNormalize {
    /// Ajouté à la moyenne des carrés pour ne jamais diviser par zéro.
    pub epsilon: f64,
}

impl Operation for RmsNormalize {
    const NAME: &'static str = "rms_normalize";
}

impl RmsNormalize {
    /// `1 / √(moyenne(x²) + epsilon)`, l'axe réduit étant gardé.
    fn inverse_rms<B: Primitives>(&self, backend: &B, x: &B::Tensor) -> B::Tensor {
        let last = backend.shape(x).len() - 1;
        let mean_square = backend.mean(&backend.mul(x, x), last, true);
        let epsilon = backend.scalar(B::Scalar::from_f64(self.epsilon));
        let rms = backend.sqrt(&backend.add(&mean_square, &epsilon));
        backend.div(&backend.scalar(B::Scalar::ONE), &rms)
    }
}

impl Reference for RmsNormalize {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let [x] = operands(Self::NAME, inputs);
        backend.mul(x, &self.inverse_rms(backend, x))
    }

    /// En notant `x̂` la sortie, `g` son gradient et `s` la racine inverse :
    ///
    /// ```text
    /// dx = s × (g - x̂ × moyenne(g × x̂))
    /// ```
    ///
    /// C'est la règle de [`Normalize`](super::Normalize) sans le terme de
    /// la moyenne, puisqu'elle n'est pas retirée.
    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [x] = operands(Self::NAME, inputs);
        let last = backend.shape(x).len() - 1;
        let mean_projection = backend.mean(&backend.mul(grad, output), last, true);
        let projected = backend.mul(output, &mean_projection);
        vec![Some(backend.mul(
            &backend.sub(grad, &projected),
            &self.inverse_rms(backend, x),
        ))]
    }
}
