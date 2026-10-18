//! La normalisation de LayerNorm, sans son gain ni son biais.

use super::{Operation, Reference, operands};
use crate::backend::{Float, Primitives};

/// Ramène chaque vecteur du dernier axe à une moyenne nulle et une variance
/// de 1 : `(x - moyenne) / √(variance + epsilon)`. Une entrée.
pub struct Normalize {
    /// Ajouté à la variance pour ne jamais diviser par zéro.
    pub epsilon: f64,
}

impl Operation for Normalize {
    const NAME: &'static str = "normalize";
}

impl Normalize {
    /// `x` centré, et `1 / √(variance + epsilon)`, l'axe réduit étant gardé.
    fn centered_and_inverse_std<B: Primitives>(
        &self,
        backend: &B,
        x: &B::Tensor,
    ) -> (B::Tensor, B::Tensor) {
        let last = backend.shape(x).len() - 1;
        let centered = backend.sub(x, &backend.mean(x, last, true));
        let variance = backend.mean(&backend.mul(&centered, &centered), last, true);
        let epsilon = backend.scalar(B::Scalar::from_f64(self.epsilon));
        let std = backend.sqrt(&backend.add(&variance, &epsilon));
        let inverse_std = backend.div(&backend.scalar(B::Scalar::ONE), &std);
        (centered, inverse_std)
    }
}

impl Reference for Normalize {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let [x] = operands(Self::NAME, inputs);
        let (centered, inverse_std) = self.centered_and_inverse_std(backend, x);
        backend.mul(&centered, &inverse_std)
    }

    /// En notant `x̂` la sortie, `g` son gradient et `s` l'écart-type
    /// inverse :
    ///
    /// ```text
    /// dx = s × (g - moyenne(g) - x̂ × moyenne(g × x̂))
    /// ```
    ///
    /// Les deux termes retranchés viennent des deux façons dont chaque
    /// entrée agit sur toute sa ligne : par la moyenne, et par
    /// l'écart-type.
    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [x] = operands(Self::NAME, inputs);
        let last = backend.shape(x).len() - 1;
        let (_, inverse_std) = self.centered_and_inverse_std(backend, x);
        let mean_grad = backend.mean(grad, last, true);
        let mean_projection = backend.mean(&backend.mul(grad, output), last, true);
        let centered_grad = backend.sub(grad, &mean_grad);
        let projected = backend.mul(output, &mean_projection);
        vec![Some(
            backend.mul(&backend.sub(&centered_grad, &projected), &inverse_std),
        )]
    }
}
