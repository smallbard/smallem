//! Le produit par une matrice de poids quantifiée.

use super::{Operation, Reference};
use crate::backend::Primitives;
use crate::quantize::Format;

/// `x · W (+ b)`, où `W` `[K, N]` est rangée quantifiée (voir
/// [`crate::quantize`]). Quatre ou cinq entrées :
///
/// - `x` `[…, K]` ;
/// - `words` `[groupes × mots par groupe, N]`, les codes ;
/// - `scales` `[groupes, N]`, les échelles ;
/// - `dequantized` `[K, N]`, les mêmes poids déquantifiés, `code × échelle` ;
/// - et, s'il y en a un, le biais `[N]`.
///
/// Les deux formes des poids donnent le même produit. La forme compacte se
/// lit vite : c'est elle que le GPU lit en génération, où l'on est limité
/// par la mémoire. La forme déquantifiée sert les gros produits, où l'on
/// calcule plus qu'on ne lit, et c'est la définition de référence.
///
/// Les poids quantifiés sont figés : ni les codes, ni les échelles, ni les
/// poids déquantifiés ne reçoivent de gradient. L'entrée et le biais en
/// reçoivent un, comme dans une couche affine : on peut ainsi entraîner un
/// adaptateur au-dessus d'un modèle quantifié.
pub struct QuantizedMatmul {
    pub format: Format,
}

impl Operation for QuantizedMatmul {
    const NAME: &'static str = "quantized_matmul";
}

impl QuantizedMatmul {
    /// Les entrées, avec le biais s'il y en a un.
    pub fn operands<'a, T>(inputs: &[&'a T]) -> (&'a T, &'a T, &'a T, &'a T, Option<&'a T>) {
        match inputs {
            [x, words, scales, dequantized] => (x, words, scales, dequantized, None),
            [x, words, scales, dequantized, bias] => (x, words, scales, dequantized, Some(bias)),
            _ => panic!(
                "l'opération `{}` attend 4 ou 5 entrées, reçu {}",
                Self::NAME,
                inputs.len()
            ),
        }
    }
}

impl Reference for QuantizedMatmul {
    fn reference<B: Primitives>(&self, backend: &B, inputs: &[&B::Tensor]) -> B::Tensor {
        let (x, _, _, dequantized, bias) = Self::operands(inputs);
        let product = backend.matmul(x, dequantized);
        match bias {
            Some(bias) => backend.add(&product, bias),
            None => product,
        }
    }

    /// `dx = g · Wᵀ` et `db = g` sommé sur toutes les lignes, où `W` est la
    /// matrice déquantifiée.
    fn reference_backward<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        self.reference_backward_needed(backend, inputs, output, grad, &[true; 5])
    }

    fn reference_backward_needed<B: Primitives>(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        _: &B::Tensor,
        grad: &B::Tensor,
        needed: &[bool],
    ) -> Vec<Option<B::Tensor>> {
        let (_, _, _, dequantized, bias) = Self::operands(inputs);
        let grad_x = needed[0].then(|| backend.matmul(grad, &backend.transpose(dequantized, 0, 1)));
        let mut grads = vec![grad_x, None, None, None];
        if bias.is_some() {
            grads.push(needed[4].then(|| {
                let mut grad_bias = grad.clone();
                while backend.shape(&grad_bias).len() > 1 {
                    grad_bias = backend.sum(&grad_bias, 0, false);
                }
                grad_bias
            }));
        }
        grads
    }
}
