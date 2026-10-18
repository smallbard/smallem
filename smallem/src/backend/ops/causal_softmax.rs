//! Le softmax causal de l'attention.

use super::{Operation, Reference, operands};
use crate::backend::{Float, Primitives};

/// Softmax causal de scores d'attention, multipliés d'abord par `scale`.
/// Une entrée.
///
/// Les scores sont `[…, N, T]` : `N` requêtes, face aux clés des `T`
/// positions connues, dont les `N` dernières sont celles des requêtes
/// elles-mêmes. La requête `i` occupe donc la position `T - N + i`. Sa
/// ligne devient une distribution sur les positions qui ne la suivent pas ;
/// les suivantes, son futur, reçoivent 0.
///
/// À l'entraînement, `N = T` : la matrice est carrée et la ligne `i` voit
/// les colonnes `j ≤ i`. En génération, `N` vaut souvent 1 : la seule
/// requête est la dernière position, et elle voit tout.
pub struct CausalSoftmax {
    pub scale: f64,
}

impl Operation for CausalSoftmax {
    const NAME: &'static str = "causal_softmax";
}

/// Masque causal `[queries, positions]` : 1 là où la colonne est dans le
/// futur de la ligne. Les requêtes sont les dernières des `positions` : la
/// ligne `i` est à la position `positions - queries + i`, et ne doit pas
/// voir les colonnes qui la suivent.
pub fn causal_mask<B: Primitives>(backend: &B, queries: usize, positions: usize) -> B::Tensor {
    assert!(
        queries <= positions,
        "{queries} requêtes pour {positions} positions seulement",
    );
    let past = positions - queries;
    let mut mask = vec![B::Scalar::ZERO; queries * positions];
    for row in 0..queries {
        for col in past + row + 1..positions {
            mask[row * positions + col] = B::Scalar::ONE;
        }
    }
    backend.from_vec(mask, &[queries, positions])
}

impl Reference for CausalSoftmax {
    /// Trois primitives : la mise à l'échelle, le masque qui met le futur à
    /// moins l'infini, et le softmax, pour lequel `exp(-∞) = 0`.
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let [scores] = operands(Self::NAME, inputs);
        let shape = backend.shape(scores);
        let rank = shape.len();
        assert!(
            rank >= 2 && shape[rank - 2] <= shape[rank - 1],
            "causal_softmax attend des scores […, N, T] avec N ≤ T, reçu {shape:?}",
        );
        let mask = causal_mask(backend, shape[rank - 2], shape[rank - 1]);
        let scaled = backend.mul(scores, &backend.scalar(B::Scalar::from_f64(self.scale)));
        backend.softmax(&backend.masked_fill(&scaled, &mask, B::Scalar::NEG_INFINITY))
    }

    /// En notant `p` la sortie et `g` son gradient :
    ///
    /// ```text
    /// dx = scale × p × (g - Σ g × p)
    /// ```
    ///
    /// C'est la règle du softmax. Les positions masquées ont `p = 0`, donc
    /// un gradient nul, sans qu'il faille y penser.
    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [_] = operands(Self::NAME, inputs);
        let last = backend.shape(output).len() - 1;
        let projection = backend.sum(&backend.mul(grad, output), last, true);
        let local = backend.mul(output, &backend.sub(grad, &projection));
        let scale = backend.scalar(B::Scalar::from_f64(self.scale));
        vec![Some(backend.mul(&local, &scale))]
    }
}
