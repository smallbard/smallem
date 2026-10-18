//! Les opérations sur GPU.
//!
//! Chaque opération de `smallem::backend::ops` reçoit ici son exécution par
//! kernel : un `impl Run<…> for Cuda`, qui lit les entrées et lance ce qu'il
//! faut. Les lancements eux-mêmes sont dans [`crate::cuda`], à côté de ceux
//! des primitives.

use smallem::backend::Primitives;
use smallem::backend::ops::{
    Affine, CachedAttention, CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu, Normalize,
    Operation, QuantizedMatmul, Reference, RmsNormalize, RotatePairs, Run, Silu, TransposedMatmul,
    concat_widths, operands,
};

use crate::cuda::CACHED_WIDE_DIM;
use crate::cuda::Cuda;

/// Deux kernels pour le calcul, deux pour le gradient. Aucun tenseur de
/// masque n'est construit : chaque thread déduit de sa position s'il
/// regarde le futur.
impl Run<CausalSoftmax> for Cuda {
    fn run(&self, op: &CausalSoftmax, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [scores] = operands(CausalSoftmax::NAME, inputs);
        self.causal_softmax(scores, op.scale as f32)
    }

    fn run_backward(
        &self,
        op: &CausalSoftmax,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [_] = operands(CausalSoftmax::NAME, inputs);
        vec![Some(self.softmax_backward(output, grad, op.scale as f32))]
    }
}

/// Trois kernels écrits pour elle : un pour le calcul, deux pour le
/// gradient. La passe avant garde `log_sum`, une valeur par requête, pour
/// que la passe arrière recalcule les poids sans refaire le softmax.
///
/// Des têtes plus larges que les kernels ne le permettent, ou un backend
/// dont l'attention fusionnée est désactivée, passent par les primitives :
/// la passe avant garde alors les poids entiers, et la passe arrière les
/// relit, comme quand l'attention était trois opérations sur la tape.
impl Run<FusedAttention> for Cuda {
    fn run(&self, op: &FusedAttention, inputs: &[&Self::Tensor]) -> Self::Tensor {
        self.run_saving(op, inputs).0
    }

    fn run_saving(
        &self,
        op: &FusedAttention,
        inputs: &[&Self::Tensor],
    ) -> (Self::Tensor, Option<Self::Tensor>) {
        let [q, k, v] = operands(FusedAttention::NAME, inputs);
        match self.fused_attention_dims(q, k, v) {
            Some(dims) => {
                let (out, log_sum) = self.attention(q, k, v, op.scale as f32, dims);
                (out, Some(log_sum))
            }
            None => {
                let (out, weights) = self.attention_unfused(q, k, v, op.scale as f32);
                (out, Some(weights))
            }
        }
    }

    fn run_backward(
        &self,
        op: &FusedAttention,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        self.run_backward_saved(op, inputs, output, None, grad)
    }

    fn run_backward_saved(
        &self,
        op: &FusedAttention,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        saved: Option<&Self::Tensor>,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [q, k, v] = operands(FusedAttention::NAME, inputs);
        // Ce qui a été gardé dépend du chemin : `log_sum` pour les kernels,
        // les poids pour les primitives.
        match (saved, self.fused_attention_dims(q, k, v)) {
            (Some(log_sum), Some(dims)) => self
                .attention_backward(q, k, v, output, log_sum, grad, op.scale as f32, dims)
                .map(Some)
                .into(),
            (Some(weights), None) => self
                .attention_unfused_backward(q, k, v, weights, grad, op.scale as f32)
                .map(Some)
                .into(),
            (None, _) => op.reference_backward(self, inputs, output, grad),
        }
    }
}

impl Run<Concat> for Cuda {
    fn run(&self, _: &Concat, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [a, b] = operands(Concat::NAME, inputs);
        self.concat_last(a, b)
    }

    fn run_backward(
        &self,
        _: &Concat,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [a, b] = operands(Concat::NAME, inputs);
        let (width_a, width_b) = concat_widths(self.shape(a), self.shape(b));
        vec![
            Some(self.slice_last(grad, 0, width_a)),
            Some(self.slice_last(grad, width_a, width_b)),
        ]
    }
}

impl Run<CrossEntropy> for Cuda {
    fn run(&self, op: &CrossEntropy, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [logits] = operands(CrossEntropy::NAME, inputs);
        self.cross_entropy(logits, &op.targets)
    }

    fn run_backward(
        &self,
        op: &CrossEntropy,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [logits] = operands(CrossEntropy::NAME, inputs);
        vec![Some(
            self.mul(&self.cross_entropy_grad(logits, &op.targets), grad),
        )]
    }
}

impl Run<Gelu> for Cuda {
    fn run(&self, _: &Gelu, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x] = operands(Gelu::NAME, inputs);
        self.gelu(x)
    }

    fn run_backward(
        &self,
        _: &Gelu,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [x] = operands(Gelu::NAME, inputs);
        vec![Some(self.gelu_backward(x, grad))]
    }
}

/// Deux kernels : les réductions par ligne, puis un thread par élément.
impl Run<Normalize> for Cuda {
    fn run(&self, op: &Normalize, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x] = operands(Normalize::NAME, inputs);
        self.normalize(x, op.epsilon as f32)
    }

    fn run_backward(
        &self,
        op: &Normalize,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [x] = operands(Normalize::NAME, inputs);
        vec![Some(self.normalize_backward(
            x,
            output,
            grad,
            op.epsilon as f32,
        ))]
    }
}

/// Deux kernels : la réduction par ligne, puis un thread par élément.
impl Run<RmsNormalize> for Cuda {
    fn run(&self, op: &RmsNormalize, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x] = operands(RmsNormalize::NAME, inputs);
        self.rms_normalize(x, op.epsilon as f32)
    }

    fn run_backward(
        &self,
        op: &RmsNormalize,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [x] = operands(RmsNormalize::NAME, inputs);
        vec![Some(self.rms_normalize_backward(
            x,
            output,
            grad,
            op.epsilon as f32,
        ))]
    }
}

/// Un kernel, plus un second pour réunir les tranches de clés quand il y en
/// a plusieurs. Des têtes plus larges que les kernels ne le permettent
/// passent par la définition de référence.
impl Run<CachedAttention> for Cuda {
    fn run(&self, op: &CachedAttention, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let dims = op.dims(self, inputs);
        if dims.key_dim > CACHED_WIDE_DIM || dims.value_dim > CACHED_WIDE_DIM {
            return op.reference(self, inputs);
        }
        let [q, keys, values, slots, rows] = operands(CachedAttention::NAME, inputs);
        self.cached_attention(q, keys, values, slots, rows, dims, op.scale as f32)
    }

    fn run_backward(
        &self,
        op: &CachedAttention,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        op.reference_backward(self, inputs, output, grad)
    }
}

/// Le biais est ajouté par le kernel du produit, ou par celui qui réunit ses
/// tranches, dans les produits de la génération. Ceux de l'entraînement font
/// le produit puis l'addition, comme avec les primitives (voir
/// [`Cuda::product`]).
impl Run<Affine> for Cuda {
    fn run(&self, op: &Affine, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x, weight, bias] = operands(Affine::NAME, inputs);
        if self.shape(weight).len() == 2 && self.shape(bias).len() == 1 {
            self.affine(x, weight, bias)
        } else {
            op.reference(self, inputs)
        }
    }

    fn run_backward(
        &self,
        op: &Affine,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        op.reference_backward(self, inputs, output, grad)
    }
}

/// Un kernel pour les produits de quelques lignes, ceux de la génération.
/// Au-delà, la définition de référence : la transposée, puis le produit
/// matriciel ordinaire.
impl Run<TransposedMatmul> for Cuda {
    fn run(&self, op: &TransposedMatmul, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [a, b] = operands(TransposedMatmul::NAME, inputs);
        let (shape_a, shape_b) = (self.shape(a), self.shape(b));
        let rows: usize = shape_a[..shape_a.len() - 1].iter().product();
        if shape_b.len() == 2 && rows <= self.narrow_rows() {
            self.transposed_matmul_narrow(a, b)
        } else {
            op.reference(self, inputs)
        }
    }

    fn run_backward(
        &self,
        op: &TransposedMatmul,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        op.reference_backward(self, inputs, output, grad)
    }
}

/// La forme compacte des poids pour les produits de la génération, la
/// forme déquantifiée pour les autres (voir [`Cuda::quantized_matmul`]).
impl Run<QuantizedMatmul> for Cuda {
    fn run(&self, op: &QuantizedMatmul, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let (x, words, scales, dequantized, bias) = QuantizedMatmul::operands(inputs);
        self.quantized_matmul(x, words, scales, dequantized, bias, op.format)
    }

    fn run_backward(
        &self,
        op: &QuantizedMatmul,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        op.reference_backward(self, inputs, output, grad)
    }
}

impl Run<RotatePairs> for Cuda {
    fn run(&self, _: &RotatePairs, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x, cos, sin] = operands(RotatePairs::NAME, inputs);
        self.rotate_pairs(x, cos, sin, false)
    }

    fn run_backward(
        &self,
        _: &RotatePairs,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [_, cos, sin] = operands(RotatePairs::NAME, inputs);
        vec![Some(self.rotate_pairs(grad, cos, sin, true)), None, None]
    }
}

impl Run<Silu> for Cuda {
    fn run(&self, _: &Silu, inputs: &[&Self::Tensor]) -> Self::Tensor {
        let [x] = operands(Silu::NAME, inputs);
        self.silu(x)
    }

    fn run_backward(
        &self,
        _: &Silu,
        inputs: &[&Self::Tensor],
        _: &Self::Tensor,
        grad: &Self::Tensor,
    ) -> Vec<Option<Self::Tensor>> {
        let [x] = operands(Silu::NAME, inputs);
        vec![Some(self.silu_backward(x, grad))]
    }
}
