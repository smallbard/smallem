//! La couche affine en une seule opération.

use super::{Operation, Reference, operands};
use crate::backend::Primitives;

/// `x · W + b`, pour `x` `[…, K]`, une matrice `W` `[K, N]` et un biais `b`
/// `[N]` : la sortie est `[…, N]`.
///
/// Avec les primitives, c'est un produit matriciel puis une addition. Le
/// GPU les fait en un seul passage quand `x` n'a que quelques lignes, comme
/// en génération : le biais est ajouté par le kernel du produit, au lieu
/// d'un second kernel qui relirait toute la sortie.
///
/// La rétropropagation est celle que la tape calculait pour le produit puis
/// l'addition, primitive par primitive : un entraînement donne les mêmes
/// nombres qu'avec deux opérations.
pub struct Affine;

impl Operation for Affine {
    const NAME: &'static str = "affine";
}

impl Reference for Affine {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let [x, weight, bias] = operands(Self::NAME, inputs);
        backend.add(&backend.matmul(x, weight), bias)
    }

    /// `dx = g · Wᵀ`, `dW = xᵀ · g` sommé sur le lot, `db = g` sommé sur
    /// toutes les lignes.
    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        _: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [x, weight, _] = operands(Self::NAME, inputs);
        let rank = backend.shape(x).len();
        let x_transposed = backend.transpose(x, rank - 2, rank - 1);
        let grad_x = backend.matmul(grad, &backend.transpose(weight, 0, 1));
        let mut grad_weight = backend.matmul(&x_transposed, grad);
        while backend.shape(&grad_weight).len() > 2 {
            grad_weight = backend.sum(&grad_weight, 0, false);
        }
        let mut grad_bias = grad.clone();
        while backend.shape(&grad_bias).len() > 1 {
            grad_bias = backend.sum(&grad_bias, 0, false);
        }
        vec![Some(grad_x), Some(grad_weight), Some(grad_bias)]
    }
}
