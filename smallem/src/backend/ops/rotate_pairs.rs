//! La rotation de RoPE.

use super::{Operation, Run, operands};
use crate::backend::Float;
use crate::backend::cpu::Cpu;

/// Tourne chaque paire de composantes voisines d'un angle qui dépend de la
/// position. Trois entrées : `x`, `cos` et `sin`.
///
/// `x` est `[…, T, D]` avec `D` pair ; `cos` et `sin` sont `[T, D/2]` et
/// donnent le cosinus et le sinus de l'angle de la paire `i` à la position
/// `t`. La paire `(x₀, x₁)` devient `(x₀·cos - x₁·sin, x₀·sin + x₁·cos)`.
///
/// Seul `x` reçoit un gradient : les angles sont des données. Et comme une
/// rotation ne change pas les longueurs, ce gradient est simplement la
/// rotation inverse du gradient de la sortie.
pub struct RotatePairs;

impl Operation for RotatePairs {
    const NAME: &'static str = "rotate_pairs";
}

impl<F: Float> Run<RotatePairs> for Cpu<F> {
    fn run(&self, _: &RotatePairs, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x, cos, sin] = operands(RotatePairs::NAME, inputs);
        x.rotate_pairs(cos, sin, false)
    }

    fn run_backward(
        &self,
        _: &RotatePairs,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [_, cos, sin] = operands(RotatePairs::NAME, inputs);
        vec![Some(grad.rotate_pairs(cos, sin, true)), None, None]
    }
}
