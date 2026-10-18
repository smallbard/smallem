//! Les opérations : tout ce qu'un backend sait calculer en plus de ses
//! primitives.
//!
//! Une primitive est une méthode du trait [`Primitives`] : additionner,
//! sommer, multiplier deux matrices. Il y en a une vingtaine, et la liste
//! ne bouge plus. Une opération est construite au-dessus : GELU, une
//! normalisation, le softmax causal de l'attention, la perte.
//!
//! Chaque opération tient dans un fichier, qui réunit ce qui la définit :
//!
//! - une **structure**, qui porte ses réglages (`Normalize { epsilon }`) ;
//! - son **calcul** et sa **rétropropagation**, pour chaque backend. Le
//!   gradient d'une opération se vérifie contre sa définition : les deux
//!   sont côte à côte.
//!
//! Un backend sait exécuter l'opération `O` s'il implémente [`Run<O>`].
//! Quand l'opération s'écrit avec les primitives, elle fournit cette
//! écriture ([`Reference`]) : le backend CPU s'en sert telle quelle, et le
//! backend GPU, qui a un kernel dédié, est testé contre elle.
//!
//! Ajouter une opération, c'est ajouter un fichier ici et une ligne à la
//! liste de [`Backend`](super::Backend). L'autograd et le profileur n'ont
//! pas à la connaître : ils traitent toutes les opérations de la même façon.

mod affine;
mod attention;
mod cached_attention;
mod causal_softmax;
mod concat;
mod cross_entropy;
mod gelu;
mod normalize;
mod quantized_matmul;
mod rms_normalize;
mod rotate_pairs;
mod silu;
mod transposed_matmul;

pub use affine::Affine;
pub use attention::FusedAttention;
pub use cached_attention::{CachedAttention, CachedAttentionDims};
pub use causal_softmax::{CausalSoftmax, causal_mask};
pub use concat::{Concat, concat_widths};
pub use cross_entropy::CrossEntropy;
pub use gelu::Gelu;
pub use normalize::Normalize;
pub use quantized_matmul::QuantizedMatmul;
pub use rms_normalize::RmsNormalize;
pub use rotate_pairs::RotatePairs;
pub use silu::Silu;
pub use transposed_matmul::TransposedMatmul;

use super::cpu::Cpu;

#[cfg(test)]
mod tests;
use super::{Float, Primitives};

/// Une opération : un calcul nommé, avec ses réglages.
pub trait Operation: 'static {
    /// Le nom sous lequel le profileur la compte.
    const NAME: &'static str;
}

/// Un backend qui sait exécuter l'opération `O`.
pub trait Run<O: Operation>: Primitives {
    /// Calcule l'opération. Le nombre et le rôle des `inputs` sont ceux
    /// que documente la structure `O`.
    fn run(&self, op: &O, inputs: &[&Self::Tensor]) -> Self::Tensor;

    /// Comme [`Run::run`], et rend en plus ce que la rétropropagation
    /// voudra relire. La tape le garde et le rend à
    /// [`Run::run_backward_saved`].
    ///
    /// Par défaut, rien : la plupart des opérations retrouvent tout ce
    /// qu'il leur faut dans leurs entrées et leur sortie. L'attention
    /// fusionnée, elle, garde une statistique par ligne qu'il serait
    /// coûteux de recalculer.
    fn run_saving(&self, op: &O, inputs: &[&Self::Tensor]) -> (Self::Tensor, Option<Self::Tensor>) {
        (self.run(op, inputs), None)
    }

    /// Rétropropagation : à partir du gradient `grad` de la sortie
    /// `output`, rend le gradient de chaque entrée, dans l'ordre des
    /// `inputs`. `None` pour une entrée qui n'en reçoit pas, comme une
    /// table d'angles.
    fn run_backward(
        &self,
        op: &O,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>>;

    /// Comme [`Run::run_backward`], avec ce que [`Run::run_saving`] a gardé.
    fn run_backward_saved(
        &self,
        op: &O,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        _saved: Option<&Self::Tensor>,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        self.run_backward(op, inputs, output, grad)
    }
}

/// Une opération qui s'écrit avec les seules primitives.
///
/// Cette écriture est sa définition. Elle est lente, une primitive par
/// étape, et c'est son rôle : être assez simple pour qu'on la croie.
pub trait Reference: Operation {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor;

    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>>;
}

/// Le backend CPU exécute la définition de référence, sans rien y ajouter.
impl<F: Float, O: Reference> Run<O> for Cpu<F> {
    fn run(&self, op: &O, inputs: &[&Self::Tensor]) -> Self::Tensor {
        op.reference(self, inputs)
    }

    fn run_backward(
        &self,
        op: &O,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        op.reference_backward(self, inputs, output, grad)
    }
}

/// Les `N` entrées d'une opération. Se tromper sur leur nombre est une
/// erreur de programmation.
pub fn operands<'a, T, const N: usize>(name: &str, inputs: &[&'a T]) -> [&'a T; N] {
    inputs.try_into().unwrap_or_else(|_| {
        panic!(
            "l'opération `{name}` attend {N} entrée(s), reçu {}",
            inputs.len()
        )
    })
}
