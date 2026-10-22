//! Les opérations par ligne du dernier axe : softmax, normalisations,
//! softmax causal et entropie croisée, avec les réductions qui les servent.

use super::*;

impl Cuda {
    /// Maximum et `ln(Σ exp(x - max))` de chaque ligne du dernier axe.
    pub(super) fn row_stats(&self, a: &CudaTensor) -> (Pooled<f32>, Pooled<f32>) {
        let width = *a
            .shape
            .last()
            .expect("softmax demande au moins une dimension");
        let rows = a.storage.buffer.len() / width;
        let mut row_max = self.alloc(rows);
        let mut row_log_sum = self.alloc(rows);
        launch!(
            self,
            rows,
            row_stats,
            prepare_row_stats,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            &mut row_max,
            &mut row_log_sum
        );
        (row_max, row_log_sum)
    }

    /// Deux lancements : les réductions par ligne, puis un thread par
    /// élément qui s'en sert.
    pub(super) fn softmax_rows(&self, mode: SoftmaxMode, a: &CudaTensor) -> CudaTensor {
        let (row_max, row_log_sum) = self.row_stats(a);
        let width = *a.shape.last().unwrap();
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            softmax_rows,
            prepare_softmax_rows,
            launch_config(len),
            mode,
            &a.storage.buffer,
            &row_max,
            &row_log_sum,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Nombre de lignes et largeur du dernier axe.
    pub(super) fn rows_and_width(a: &CudaTensor) -> (usize, usize) {
        let width = *a
            .shape
            .last()
            .expect("opération par ligne sur un tenseur sans dimension");
        (a.storage.buffer.len() / width, width)
    }

    /// Moyenne et écart-type inverse de chaque ligne du dernier axe.
    pub(super) fn row_moments(&self, a: &CudaTensor, epsilon: f32) -> (Pooled<f32>, Pooled<f32>) {
        let (rows, width) = Self::rows_and_width(a);
        let mut mean = self.alloc(rows);
        let mut inverse_std = self.alloc(rows);
        launch!(
            self,
            rows,
            row_moments,
            prepare_row_moments,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            epsilon,
            &mut mean,
            &mut inverse_std
        );
        (mean, inverse_std)
    }

    /// `1 / √(moyenne(x²) + epsilon)` de chaque ligne du dernier axe.
    pub(super) fn row_inverse_rms(&self, a: &CudaTensor, epsilon: f32) -> Pooled<f32> {
        let (rows, width) = Self::rows_and_width(a);
        let mut out = self.alloc(rows);
        launch!(
            self,
            rows,
            row_inverse_rms,
            prepare_row_inverse_rms,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            epsilon,
            &mut out
        );
        out
    }

    /// `Σ a × b` sur chaque ligne du dernier axe.
    pub(super) fn row_dot(&self, a: &CudaTensor, b: &CudaTensor) -> Pooled<f32> {
        assert_eq!(a.shape, b.shape, "row_dot : formes différentes");
        let (rows, width) = Self::rows_and_width(a);
        let mut out = self.alloc(rows);
        launch!(
            self,
            rows,
            row_dot,
            prepare_row_dot,
            row_launch_config(rows),
            &a.storage.buffer,
            &b.storage.buffer,
            width as u32,
            &mut out
        );
        out
    }

    /// Deux lancements : les réductions par ligne, puis un thread par
    /// élément qui s'en sert.
    pub(crate) fn normalize(&self, a: &CudaTensor, epsilon: f32) -> CudaTensor {
        let (mean, inverse_std) = self.row_moments(a, epsilon);
        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            normalize_rows,
            prepare_normalize_rows,
            launch_config(len),
            &a.storage.buffer,
            &mean,
            &inverse_std,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn normalize_backward(
        &self,
        a: &CudaTensor,
        normalized: &CudaTensor,
        grad: &CudaTensor,
        epsilon: f32,
    ) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        let (_, inverse_std) = self.row_moments(a, epsilon);
        let grad_sum = self.sum(grad, grad.shape.len() - 1, false);
        let grad_dot = self.row_dot(grad, normalized);

        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            normalize_grad_rows,
            prepare_normalize_grad_rows,
            launch_config(len),
            &normalized.storage.buffer,
            &grad.storage.buffer,
            &grad_sum.storage.buffer,
            &grad_dot,
            &inverse_std,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Deux lancements : la réduction par ligne, puis un thread par élément.
    pub(crate) fn rms_normalize(&self, a: &CudaTensor, epsilon: f32) -> CudaTensor {
        let inverse_rms = self.row_inverse_rms(a, epsilon);
        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            scale_rows,
            prepare_scale_rows,
            launch_config(len),
            &a.storage.buffer,
            &inverse_rms,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn rms_normalize_backward(
        &self,
        a: &CudaTensor,
        normalized: &CudaTensor,
        grad: &CudaTensor,
        epsilon: f32,
    ) -> CudaTensor {
        assert_eq!(a.shape, grad.shape, "gradient de forme inattendue");
        let inverse_rms = self.row_inverse_rms(a, epsilon);
        let grad_dot = self.row_dot(grad, normalized);
        let (_, width) = Self::rows_and_width(a);
        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            rms_grad_rows,
            prepare_rms_grad_rows,
            launch_config(len),
            &normalized.storage.buffer,
            &grad.storage.buffer,
            &grad_dot,
            &inverse_rms,
            width as u32,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    /// Le masque est calculé par les kernels à partir de la position :
    /// aucun tenseur de masque n'est construit ni envoyé.
    pub(crate) fn causal_softmax(&self, a: &CudaTensor, scale: f32) -> CudaTensor {
        let rank = a.shape.len();
        assert!(
            rank >= 2 && a.shape[rank - 2] <= a.shape[rank - 1],
            "causal_softmax attend des scores […, N, T] avec N ≤ T, reçu {:?}",
            a.shape,
        );
        let queries = a.shape[rank - 2] as u32;
        let (rows, width) = Self::rows_and_width(a);
        let mut row_max = self.alloc(rows);
        let mut row_log_sum = self.alloc(rows);
        launch!(
            self,
            rows,
            causal_row_stats,
            prepare_causal_row_stats,
            row_launch_config(rows),
            &a.storage.buffer,
            width as u32,
            queries,
            scale,
            &mut row_max,
            &mut row_log_sum
        );

        let len = a.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            causal_softmax_rows,
            prepare_causal_softmax_rows,
            launch_config(len),
            &a.storage.buffer,
            &row_max,
            &row_log_sum,
            width as u32,
            queries,
            scale,
            &mut out
        );
        self.wrap(out, a.shape.clone())
    }

    pub(crate) fn softmax_backward(
        &self,
        probabilities: &CudaTensor,
        grad: &CudaTensor,
        scale: f32,
    ) -> CudaTensor {
        let row_dot = self.row_dot(grad, probabilities);
        let (_, width) = Self::rows_and_width(probabilities);
        let len = probabilities.storage.buffer.len();
        let mut out = self.alloc(len);
        launch!(
            self,
            rows,
            softmax_grad_rows,
            prepare_softmax_grad_rows,
            launch_config(len),
            &probabilities.storage.buffer,
            &grad.storage.buffer,
            &row_dot,
            width as u32,
            scale,
            &mut out
        );
        self.wrap(out, probabilities.shape.clone())
    }

    pub(crate) fn cross_entropy(&self, logits: &CudaTensor, targets: &[u32]) -> CudaTensor {
        let (n, vocab) = cross_entropy_dims(&logits.shape, targets);
        let device_targets = self.upload_indices(targets);
        let mut losses = self.alloc(n);

        launch!(
            self,
            rows,
            cross_entropy_rows,
            prepare_cross_entropy_rows,
            row_launch_config(n),
            &logits.storage.buffer,
            &device_targets,
            vocab as u32,
            &mut losses
        );

        let total = self.sum_all(&self.wrap(losses, vec![n]));
        self.mul(&total, &self.scalar(1.0 / n as f32))
    }

    pub(crate) fn cross_entropy_grad(&self, logits: &CudaTensor, targets: &[u32]) -> CudaTensor {
        let (n, vocab) = cross_entropy_dims(&logits.shape, targets);
        let (row_max, row_log_sum) = self.row_stats(logits);
        let device_targets = self.upload_indices(targets);
        let len = n * vocab;
        let mut out = self.alloc(len);

        launch!(
            self,
            rows,
            cross_entropy_grad,
            prepare_cross_entropy_grad,
            launch_config(len),
            &logits.storage.buffer,
            &row_max,
            &row_log_sum,
            &device_targets,
            vocab as u32,
            1.0 / n as f32,
            &mut out
        );

        self.wrap(out, logits.shape.clone())
    }
}
