//! L'attention causale en une seule opération.

use super::{CausalSoftmax, Operation, Reference, operands};
use crate::backend::{Float, Primitives};

/// L'attention causale entière : scores, softmax causal, mélange des
/// valeurs. Trois entrées, les requêtes `q` `[…, N, Dk]`, les clés `k`
/// `[…, T, Dk]` et les valeurs `v` `[…, T, Dv]` ; une sortie `[…, N, Dv]`.
///
/// ```text
/// sortie = causal_softmax(q · kᵀ, scale) · v
/// ```
///
/// Comme pour [`CausalSoftmax`], les `N` requêtes sont les dernières des `T`
/// positions : la requête `i` est à la position `T - N + i` et ne voit que
/// les clés qui ne la suivent pas.
///
/// Écrite avec les primitives, l'opération range la matrice des scores
/// `[…, N, T]`, puis celle des poids. Le GPU, lui, la calcule sans jamais
/// les ranger : c'est l'attention fusionnée. Les deux doivent donner le
/// même résultat, et cette écriture-ci est celle qui fait foi.
pub struct FusedAttention {
    pub scale: f64,
}

impl Operation for FusedAttention {
    const NAME: &'static str = "attention";
}

impl FusedAttention {
    /// Les poids de l'attention, `[…, N, T]` : le softmax causal des scores.
    fn weights<B: Primitives>(&self, backend: &B, q: &B::Tensor, k: &B::Tensor) -> B::Tensor {
        let rank = backend.shape(k).len();
        let scores = backend.matmul(q, &backend.transpose(k, rank - 2, rank - 1));
        CausalSoftmax { scale: self.scale }.reference(backend, &[&scores])
    }

    /// Le calcul avec les primitives, qui rend aussi les poids : un backend
    /// sans kernel fusionné peut les garder pour la passe arrière, comme
    /// le faisait la tape quand l'attention était trois opérations.
    pub fn with_weights<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
    ) -> (B::Tensor, B::Tensor) {
        let [q, k, v] = operands(Self::NAME, inputs);
        let weights = self.weights(backend, q, k);
        (backend.matmul(&weights, v), weights)
    }
}

impl Reference for FusedAttention {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        self.with_weights(backend, inputs).0
    }

    /// Les poids `P` sont recalculés à partir de `q` et `k` : c'est ce que
    /// fait aussi le GPU, qui ne les a jamais rangés. En notant `O` la
    /// sortie et `dO` son gradient :
    ///
    /// ```text
    /// dV = Pᵀ · dO
    /// dP = dO · Vᵀ
    /// dS = scale × P × (dP - Σ dO × O)      la règle du softmax
    /// dQ = dS · K
    /// dK = dSᵀ · Q
    /// ```
    ///
    /// La somme `Σ dP × P` de la règle du softmax, prise ligne par ligne,
    /// vaut `Σ dO × O` : `O` est justement `P · V`. C'est la forme qui sert
    /// au GPU, où `P` n'existe jamais en entier.
    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [q, k, _] = operands(Self::NAME, inputs);
        let weights = self.weights(backend, q, k);
        self.backward_from_weights(backend, inputs, output, &weights, grad)
    }
}

impl FusedAttention {
    /// La règle de [`Reference::reference_backward`], avec des poids déjà
    /// calculés.
    pub fn backward_from_weights<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        weights: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [q, k, v] = operands(Self::NAME, inputs);
        let rank = backend.shape(k).len();
        let last = rank - 1;
        let transposed = |t: &B::Tensor| backend.transpose(t, rank - 2, rank - 1);

        let grad_v = backend.matmul(&transposed(weights), grad);
        let grad_weights = backend.matmul(grad, &transposed(v));
        let projection = backend.sum(&backend.mul(grad, output), last, true);
        let grad_scores = backend.mul(
            &backend.mul(weights, &backend.sub(&grad_weights, &projection)),
            &backend.scalar(B::Scalar::from_f64(self.scale)),
        );
        let grad_q = backend.matmul(&grad_scores, k);
        let grad_k = backend.matmul(&transposed(&grad_scores), q);
        vec![Some(grad_q), Some(grad_k), Some(grad_v)]
    }
}
