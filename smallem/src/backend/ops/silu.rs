//! SiLU, la fonction d'activation de SwiGLU.

use super::{Operation, Run, operands};
use crate::backend::Float;
use crate::backend::cpu::Cpu;

/// SiLU, élément par élément. Une entrée.
///
/// La formule est [`float::silu`](crate::backend::float::silu) : une fonction
/// ordinaire, compilée pour l'hôte et pour le GPU. Les deux backends
/// appliquent donc exactement le même calcul.
pub struct Silu;

impl Operation for Silu {
    const NAME: &'static str = "silu";
}

impl<F: Float> Run<Silu> for Cpu<F> {
    fn run(&self, _: &Silu, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x] = operands(Silu::NAME, inputs);
        x.silu()
    }

    /// `grad × silu'(x)`
    fn run_backward(
        &self,
        _: &Silu,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [x] = operands(Silu::NAME, inputs);
        vec![Some(x.silu_backward(grad))]
    }
}
