//! Le produit par une matrice transposée.

use super::{Operation, Reference, operands};
use crate::backend::Primitives;

/// `a · bᵀ`, pour `a` `[…, K]` et une matrice `b` `[N, K]` : la sortie est
/// `[…, N]`. L'élément `n` d'une ligne est le produit scalaire de cette
/// ligne et de la ligne `n` de `b`.
///
/// C'est le calcul de la sortie du modèle, où `b` est la table d'embedding
/// `[V, C]`. Avec les primitives, il faudrait d'abord la transposer, donc
/// recopier toute la table : en génération, où l'on ne calcule qu'une ligne
/// de logits par séquence, cette copie coûte plus que le produit lui-même.
pub struct TransposedMatmul;

impl Operation for TransposedMatmul {
    const NAME: &'static str = "transposed_matmul";
}

impl Reference for TransposedMatmul {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let [a, b] = operands(Self::NAME, inputs);
        assert_eq!(
            backend.shape(b).len(),
            2,
            "transposed_matmul : b doit être une matrice, reçu {:?}",
            backend.shape(b),
        );
        backend.matmul(a, &backend.transpose(b, 0, 1))
    }

    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        self.reference_backward_needed(backend, inputs, output, grad, &[true; 2])
    }

    /// `da = g · b` et `db = gᵀ · a`, les lignes de `a` et de `g` mises
    /// bout à bout ; seulement ceux qu'on attend.
    fn reference_backward_needed<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        _: &B::Tensor,
        grad: &B::Tensor,
        needed: &[bool],
    ) -> Vec<Option<B::Tensor>> {
        let [a, b] = operands(Self::NAME, inputs);
        let grad_a = needed[0].then(|| backend.matmul(grad, b));
        let grad_b = needed[1].then(|| {
            let (width, outputs) = (backend.shape(a).len(), backend.shape(b)[0]);
            let rows: usize = backend.shape(a)[..width - 1].iter().product();
            let flat_a = backend.reshape(a, &[rows, backend.shape(b)[1]]);
            let flat_grad = backend.reshape(grad, &[rows, outputs]);
            backend.matmul(&backend.transpose(&flat_grad, 0, 1), &flat_a)
        });
        vec![grad_a, grad_b]
    }
}
