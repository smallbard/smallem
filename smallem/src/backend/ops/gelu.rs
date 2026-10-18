//! GELU, la fonction d'activation de GPT-2.

use super::{Operation, Run, operands};
use crate::backend::Float;
use crate::backend::cpu::Cpu;

/// GELU, élément par élément. Une entrée.
///
/// La formule est [`float::gelu`](crate::backend::float::gelu) : une fonction
/// ordinaire, compilée pour l'hôte et pour le GPU. Les deux backends
/// appliquent donc exactement le même calcul.
pub struct Gelu;

impl Operation for Gelu {
    const NAME: &'static str = "gelu";
}

impl<F: Float> Run<Gelu> for Cpu<F> {
    fn run(&self, _: &Gelu, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x] = operands(Gelu::NAME, inputs);
        x.gelu()
    }

    /// `grad × gelu'(x)`
    fn run_backward(
        &self,
        _: &Gelu,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [x] = operands(Gelu::NAME, inputs);
        vec![Some(x.gelu_backward(grad))]
    }
}
