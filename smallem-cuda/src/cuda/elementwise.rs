//! Les opérations élément par élément qui ont leurs propres kernels : GELU,
//! SiLU, la rotation de RoPE, et la concaténation le long du dernier axe.

use super::*;

impl Cuda {
    pub(crate) fn gelu(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Gelu, a)
    }

    pub(crate) fn gelu_backward(&self, a: &CudaTensor, grad: &CudaTensor) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        self.binary(BinaryOp::GeluBackward, 0.0, a, grad)
    }

    pub(crate) fn silu(&self, a: &CudaTensor) -> CudaTensor {
        self.unary(UnaryOp::Silu, a)
    }

    pub(crate) fn silu_backward(&self, a: &CudaTensor, grad: &CudaTensor) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        self.binary(BinaryOp::SiluBackward, 0.0, a, grad)
    }

    pub(crate) fn rotate_pairs(
        &self,
        a: &CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        inverse: bool,
    ) -> CudaTensor {
        let rank = a.shape.len();
        assert!(
            rank >= 2 && a.shape[rank - 1].is_multiple_of(2),
            "rotate_pairs attend […, T, D] avec D pair, reçu {:?}",
            a.shape,
        );
        let (length, width) = (a.shape[rank - 2], a.shape[rank - 1]);
        let half = width / 2;
        assert!(
            cos.shape == [length, half] && sin.shape == [length, half],
            "rotate_pairs : angles {:?} et {:?} pour un tenseur {:?}, [{length}, {half}] attendu",
            cos.shape,
            sin.shape,
            a.shape,
        );
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            elementwise,
            rotate_pairs,
            prepare_rotate_pairs,
            launch_config(len),
            &a.storage.buffer,
            &cos.storage.buffer,
            &sin.storage.buffer,
            width as u32,
            length as u32,
            if inverse { -1.0 } else { 1.0 },
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn concat_last(&self, a: &CudaTensor, b: &CudaTensor) -> CudaTensor {
        let (width_a, width_b) = concat_widths(&a.shape, &b.shape);
        let mut shape = a.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = width_a + width_b;
        let len = shape.iter().product();
        let mut out = self.alloc(len);
        launch!(
            self,
            elementwise,
            concat_last,
            prepare_concat_last,
            launch_config(len),
            &a.storage.buffer,
            &b.storage.buffer,
            width_a as u32,
            width_b as u32,
            &mut out
        );
        self.wrap(out, shape)
    }

    /// Les `width` composantes du dernier axe de `a` qui commencent à
    /// `offset`.
    pub(crate) fn slice_last(&self, a: &CudaTensor, offset: usize, width: usize) -> CudaTensor {
        let total = *a.shape.last().expect("au moins une dimension");
        assert!(
            offset + width <= total,
            "slice_last : {offset}..{} dans un axe de {total}",
            offset + width,
        );
        let mut shape = a.shape.clone();
        *shape.last_mut().expect("au moins une dimension") = width;
        let len = shape.iter().product();
        let mut out = self.alloc(len);
        launch!(
            self,
            elementwise,
            slice_last,
            prepare_slice_last,
            launch_config(len),
            &a.storage.buffer,
            total as u32,
            offset as u32,
            width as u32,
            &mut out
        );
        self.wrap(out, shape)
    }
}
