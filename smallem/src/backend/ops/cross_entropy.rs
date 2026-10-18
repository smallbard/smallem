//! La perte d'un modèle de langage : softmax et entropie croisée, fusionnés.

use super::{Operation, Run, operands};
use crate::backend::cpu::Cpu;
use crate::backend::{Float, Primitives};

/// Entropie croisée moyenne entre des logits `[N, V]` et `N` cibles : la
/// moyenne de `-ln softmax(logits)[n, targets[n]]`, de forme `[]`. Une
/// entrée, les logits.
///
/// Le softmax et le logarithme sont fusionnés pour deux raisons. La première
/// est numérique : le logarithme d'une probabilité minuscule est instable.
/// La seconde est que leur gradient combiné est d'une simplicité
/// remarquable :
///
/// ```text
/// dlogits = (softmax(logits) - onehot(targets)) / N
/// ```
///
/// soit l'écart entre ce que le modèle a prédit et ce qu'il aurait dû
/// prédire.
pub struct CrossEntropy {
    /// Le token attendu pour chaque ligne de logits.
    pub targets: Vec<u32>,
}

impl Operation for CrossEntropy {
    const NAME: &'static str = "cross_entropy";
}

impl<F: Float> Run<CrossEntropy> for Cpu<F> {
    fn run(&self, op: &CrossEntropy, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [logits] = operands(CrossEntropy::NAME, inputs);
        logits.cross_entropy(&op.targets)
    }

    fn run_backward(
        &self,
        op: &CrossEntropy,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [logits] = operands(CrossEntropy::NAME, inputs);
        // `grad` est le gradient de la perte, un seul nombre : il s'étire.
        vec![Some(
            self.mul(&logits.cross_entropy_grad(&op.targets), grad),
        )]
    }
}
