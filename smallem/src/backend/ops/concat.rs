//! La concaténation le long du dernier axe.

use super::{Operation, Reference, operands};
use crate::backend::Primitives;

/// Met `b` à la suite de `a` le long du dernier axe. Deux entrées :
/// `[…, A]` et `[…, B]` donnent `[…, A + B]`. Les dimensions qui précèdent
/// doivent être les mêmes.
///
/// C'est l'opération qui réunit deux descriptions d'une même chose en un
/// seul vecteur : la partie « contenu » et la partie « position » d'une clé
/// d'attention, par exemple.
///
/// Rien n'est calculé, les nombres sont seulement rangés ailleurs. Le
/// gradient fait le chemin inverse : il est découpé en deux, chaque entrée
/// recevant la part qui correspond à ses composantes.
pub struct Concat;

impl Operation for Concat {
    const NAME: &'static str = "concat";
}

/// Les largeurs `A` et `B` des deux entrées, après vérification que leurs
/// autres dimensions sont les mêmes.
pub fn concat_widths(a: &[usize], b: &[usize]) -> (usize, usize) {
    assert!(
        !a.is_empty() && a.len() == b.len() && a[..a.len() - 1] == b[..b.len() - 1],
        "concat : formes incompatibles {a:?} et {b:?}",
    );
    (a[a.len() - 1], b[b.len() - 1])
}

impl Reference for Concat {
    /// Les primitives ne savent mettre bout à bout que le long du premier
    /// axe : on y amène le dernier, puis on le remet à sa place.
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let [a, b] = operands(Self::NAME, inputs);
        concat_widths(backend.shape(a), backend.shape(b));
        let last = backend.shape(a).len() - 1;
        let joined = backend.concat_rows(
            &backend.transpose(a, 0, last),
            &backend.transpose(b, 0, last),
        );
        backend.transpose(&joined, 0, last)
    }

    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        _: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        let [a, b] = operands(Self::NAME, inputs);
        let (width_a, width_b) = concat_widths(backend.shape(a), backend.shape(b));
        let last = backend.shape(a).len() - 1;
        let grad = backend.transpose(grad, 0, last);
        let part = |start: usize, width: usize| {
            let rows: Vec<u32> = (start as u32..(start + width) as u32).collect();
            Some(backend.transpose(&backend.gather_rows(&grad, &rows), 0, last))
        };
        vec![part(0, width_a), part(width_a, width_b)]
    }
}
