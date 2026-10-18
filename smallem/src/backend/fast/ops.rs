//! Les opérations sur le processeur.
//!
//! Une opération dont la définition s'écrit avec les primitives pourrait
//! s'exécuter telle quelle : chaque primitive est déjà parallèle. Mais elle
//! écrirait et relirait en mémoire chaque tenseur intermédiaire. Chaque
//! opération a donc ici une écriture directe, en une passe par ligne quand
//! c'est possible, comparée à sa définition dans `tests.rs`.

use crate::backend::Primitives;
use crate::backend::ops::{
    Affine, CachedAttention, CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu, Normalize,
    Operation, QuantizedMatmul, Reference, RmsNormalize, RotatePairs, Run, Silu, TransposedMatmul,
    concat_widths, operands,
};

use super::attention;
use super::gemm::Matrix;
use super::{Buffer, CHUNK, Fast, FastTensor, math};

type T = FastTensor;

impl Fast {
    /// Applique `f(ligne de a, ligne de b, ligne de sortie)` aux lignes du
    /// dernier axe, en parallèle. `a`, `b` et la sortie ont la même forme.
    fn rows2(&self, a: &T, b: &T, f: impl Fn(usize, &[f32], &[f32], &mut [f32]) + Sync) -> T {
        assert_eq!(
            a.shape, b.shape,
            "formes différentes : {:?} et {:?}",
            a.shape, b.shape
        );
        let width = *a.shape.last().expect("au moins une dimension");
        if width == 0 {
            return a.clone();
        }
        let (source_a, source_b) = (a.data(), b.data());
        let rows_per_task = (CHUNK / width).max(1);
        let data = self.generate(a.len(), rows_per_task * width, |start, part| {
            let first = start / width;
            for (offset, line) in part.chunks_mut(width).enumerate() {
                let row = first + offset;
                f(
                    row,
                    &source_a[row * width..][..width],
                    &source_b[row * width..][..width],
                    line,
                );
            }
        });
        FastTensor::new(data, &a.shape)
    }
}

/// La normalisation de LayerNorm en une passe par ligne : la moyenne, la
/// variance, puis la sortie, sans tenseur intermédiaire.
impl Run<Normalize> for Fast {
    fn run(&self, op: &Normalize, inputs: &[&T]) -> T {
        let [x] = operands(Normalize::NAME, inputs);
        let (isa, epsilon) = (self.isa, op.epsilon as f32);
        self.map_rows(x, |input, output| {
            let n = input.len() as f32;
            let mean = math::sum(isa, input) / n;
            for (out, &x) in output.iter_mut().zip(input) {
                *out = x - mean;
            }
            let inverse = 1.0 / (math::dot(isa, output, output) / n + epsilon).sqrt();
            math::scale(isa, output, inverse);
        })
    }

    /// `dx = s × (g - moyenne(g) - x̂ × moyenne(g × x̂))`, ligne par ligne.
    fn run_backward(&self, op: &Normalize, inputs: &[&T], output: &T, grad: &T) -> Vec<Option<T>> {
        let [x] = operands(Normalize::NAME, inputs);
        let (isa, epsilon) = (self.isa, op.epsilon as f32);
        let source = x.data();
        let width = *x.shape.last().unwrap();
        vec![Some(self.rows2(
            output,
            grad,
            |row, normalized, grad, out| {
                let input = &source[row * width..][..width];
                let n = width as f32;
                let mean = math::sum(isa, input) / n;
                let variance = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
                let inverse = 1.0 / (variance + epsilon).sqrt();
                let mean_grad = math::sum(isa, grad) / n;
                let mean_projection = math::dot(isa, grad, normalized) / n;
                for ((out, &g), &y) in out.iter_mut().zip(grad).zip(normalized) {
                    *out = inverse * (g - mean_grad - y * mean_projection);
                }
            },
        ))]
    }
}

/// RMSNorm en une passe par ligne.
impl Run<RmsNormalize> for Fast {
    fn run(&self, op: &RmsNormalize, inputs: &[&T]) -> T {
        let [x] = operands(RmsNormalize::NAME, inputs);
        let (isa, epsilon) = (self.isa, op.epsilon as f32);
        self.map_rows(x, |input, output| {
            let n = input.len() as f32;
            let inverse = 1.0 / (math::dot(isa, input, input) / n + epsilon).sqrt();
            output.copy_from_slice(input);
            math::scale(isa, output, inverse);
        })
    }

    /// `dx = s × (g - x̂ × moyenne(g × x̂))`, ligne par ligne.
    fn run_backward(
        &self,
        op: &RmsNormalize,
        inputs: &[&T],
        output: &T,
        grad: &T,
    ) -> Vec<Option<T>> {
        let [x] = operands(RmsNormalize::NAME, inputs);
        let (isa, epsilon) = (self.isa, op.epsilon as f32);
        let source = x.data();
        let width = *x.shape.last().unwrap();
        vec![Some(self.rows2(
            output,
            grad,
            |row, normalized, grad, out| {
                let input = &source[row * width..][..width];
                let n = width as f32;
                let inverse = 1.0 / (math::dot(isa, input, input) / n + epsilon).sqrt();
                let mean_projection = math::dot(isa, grad, normalized) / n;
                for ((out, &g), &y) in out.iter_mut().zip(grad).zip(normalized) {
                    *out = inverse * (g - y * mean_projection);
                }
            },
        ))]
    }
}

/// Le softmax causal ligne par ligne : chaque ligne ne lit que les positions
/// qu'elle voit, et le futur reçoit 0 sans masque construit.
impl Run<CausalSoftmax> for Fast {
    fn run(&self, op: &CausalSoftmax, inputs: &[&T]) -> T {
        let [scores] = operands(CausalSoftmax::NAME, inputs);
        let rank = scores.shape.len();
        assert!(
            rank >= 2 && scores.shape[rank - 2] <= scores.shape[rank - 1],
            "causal_softmax attend des scores […, N, T] avec N ≤ T, reçu {:?}",
            scores.shape,
        );
        let (queries, positions) = (scores.shape[rank - 2], scores.shape[rank - 1]);
        let (isa, scale) = (self.isa, op.scale as f32);
        self.rows2(scores, scores, |row, input, _, output| {
            let visible = positions - queries + row % queries + 1;
            let (seen, future) = output.split_at_mut(visible);
            for (out, &x) in seen.iter_mut().zip(input) {
                *out = x * scale;
            }
            let max = math::max(isa, seen);
            let total = math::exp_shifted_sum(isa, seen, max);
            math::scale(isa, seen, 1.0 / total);
            future.fill(0.0);
        })
    }

    /// `dx = scale × p × (g - Σ g × p)`.
    fn run_backward(&self, op: &CausalSoftmax, _: &[&T], output: &T, grad: &T) -> Vec<Option<T>> {
        let (isa, scale) = (self.isa, op.scale as f32);
        vec![Some(self.rows2(output, grad, |_, p, g, out| {
            let projection = math::dot(isa, g, p);
            for ((out, &g), &p) in out.iter_mut().zip(g).zip(p) {
                *out = scale * p * (g - projection);
            }
        }))]
    }
}

/// La concaténation le long du dernier axe : une copie par ligne.
impl Run<Concat> for Fast {
    fn run(&self, _: &Concat, inputs: &[&T]) -> T {
        let [a, b] = operands(Concat::NAME, inputs);
        let (width_a, width_b) = concat_widths(&a.shape, &b.shape);
        let width = width_a + width_b;
        let rows = a.len() / width_a.max(1);
        let (data_a, data_b) = (a.data(), b.data());
        let rows_per_task = (CHUNK / width.max(1)).max(1);
        let data = self.generate(rows * width, rows_per_task * width.max(1), |start, part| {
            let first = start / width;
            for (offset, line) in part.chunks_mut(width).enumerate() {
                let row = first + offset;
                line[..width_a].copy_from_slice(&data_a[row * width_a..][..width_a]);
                line[width_a..].copy_from_slice(&data_b[row * width_b..][..width_b]);
            }
        });
        let mut shape = a.shape.clone();
        *shape.last_mut().unwrap() = width;
        FastTensor::new(data, &shape)
    }

    fn run_backward(&self, _: &Concat, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [a, b] = operands(Concat::NAME, inputs);
        let (width_a, width_b) = concat_widths(&a.shape, &b.shape);
        let width = width_a + width_b;
        let source = grad.data();
        let part = |offset: usize, part_width: usize, shape: &[usize]| {
            let rows = grad.len() / width.max(1);
            let rows_per_task = (CHUNK / part_width.max(1)).max(1);
            let data = self.generate(
                rows * part_width,
                rows_per_task * part_width.max(1),
                |start, out| {
                    let first = start / part_width.max(1);
                    for (index, line) in out.chunks_mut(part_width.max(1)).enumerate() {
                        line.copy_from_slice(
                            &source[(first + index) * width + offset..][..part_width],
                        );
                    }
                },
            );
            Some(FastTensor::new(data, shape))
        };
        vec![part(0, width_a, &a.shape), part(width_a, width_b, &b.shape)]
    }
}

impl Fast {
    /// `A · B` dans un nouveau tampon.
    pub(crate) fn gemm(&self, a: Matrix, b: Matrix) -> Buffer {
        let mut out = Buffer::uninit(a.rows * b.cols);
        self.gemm_into(a, b, &mut out, false);
        out
    }

    /// Les lignes d'un tenseur `[…, K]`, vues comme une matrice `[R, K]`.
    fn rows_of(x: &T) -> Matrix<'_> {
        let width = *x.shape.last().expect("au moins une dimension");
        Matrix::new(x.data(), x.len() / width.max(1), width)
    }

    /// Ajoute `bias` `[N]` à chaque ligne de `out`.
    fn add_bias(&self, out: &mut [f32], bias: &[f32]) {
        let width = bias.len();
        let rows_per_task = (CHUNK / width.max(1)).max(1);
        self.for_parts(out, rows_per_task * width, |_, part| {
            for line in part.chunks_mut(width) {
                for (value, &b) in line.iter_mut().zip(bias) {
                    *value += b;
                }
            }
        });
    }
}

/// `x · W + b` : toutes les lignes de `x` d'un seul produit, puis le biais
/// sur place. La rétropropagation lit `x` et `W` transposées sans les
/// recopier, et réunit les lignes du lot dans chaque produit :
///
/// ```text
/// dx = g · Wᵀ          [R, N] · [N, K]
/// dW = xᵀ · g          [K, R] · [R, N]   la somme sur le lot est dans le produit
/// db = Σ g             sur les R lignes
/// ```
impl Run<Affine> for Fast {
    fn run(&self, _: &Affine, inputs: &[&T]) -> T {
        let [x, weight, bias] = operands(Affine::NAME, inputs);
        assert!(
            weight.shape.len() == 2 && bias.shape == [weight.shape[1]],
            "affine : poids {:?} et biais {:?}",
            weight.shape,
            bias.shape,
        );
        let (inner, outputs) = (weight.shape[0], weight.shape[1]);
        assert_eq!(
            x.shape.last(),
            Some(&inner),
            "affine : entrée {:?}, poids {:?}",
            x.shape,
            weight.shape
        );
        let mut out = self.gemm(Self::rows_of(x), Matrix::new(weight.data(), inner, outputs));
        self.add_bias(&mut out, bias.data());
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = outputs;
        FastTensor::new(out, &shape)
    }

    fn run_backward(&self, _: &Affine, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [x, weight, _] = operands(Affine::NAME, inputs);
        let (inner, outputs) = (weight.shape[0], weight.shape[1]);
        let rows = Self::rows_of(grad);
        let w = Matrix::new(weight.data(), inner, outputs);
        let grad_x = self.gemm(rows, w.t());
        let grad_weight = self.gemm(Self::rows_of(x).t(), rows);
        let grad_bias = self.sum(&self.reshape(grad, &[rows.rows, outputs]), 0, false);
        vec![
            Some(FastTensor::new(grad_x, &x.shape)),
            Some(FastTensor::new(grad_weight, &weight.shape)),
            Some(grad_bias),
        ]
    }
}

/// `a · bᵀ` : `b` est lue transposée, sans copie.
impl Run<TransposedMatmul> for Fast {
    fn run(&self, _: &TransposedMatmul, inputs: &[&T]) -> T {
        let [a, b] = operands(TransposedMatmul::NAME, inputs);
        assert_eq!(
            b.shape.len(),
            2,
            "transposed_matmul : b doit être une matrice, reçu {:?}",
            b.shape
        );
        let (outputs, inner) = (b.shape[0], b.shape[1]);
        assert_eq!(
            a.shape.last(),
            Some(&inner),
            "transposed_matmul : {:?} et {:?}",
            a.shape,
            b.shape
        );
        let out = self.gemm(Self::rows_of(a), Matrix::new(b.data(), outputs, inner).t());
        let mut shape = a.shape.clone();
        *shape.last_mut().unwrap() = outputs;
        FastTensor::new(out, &shape)
    }

    /// `da = g · b` et `db = gᵀ · a`, les lignes du lot réunies.
    fn run_backward(&self, _: &TransposedMatmul, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [a, b] = operands(TransposedMatmul::NAME, inputs);
        let (outputs, inner) = (b.shape[0], b.shape[1]);
        let rows = Self::rows_of(grad);
        let grad_a = self.gemm(rows, Matrix::new(b.data(), outputs, inner));
        let grad_b = self.gemm(rows.t(), Self::rows_of(a));
        vec![
            Some(FastTensor::new(grad_a, &a.shape)),
            Some(FastTensor::new(grad_b, &b.shape)),
        ]
    }
}

impl Fast {
    /// Applique `f(entrée, sortie)` par tranches, en parallèle.
    fn unary(&self, x: &T, f: impl Fn(&[f32], &mut [f32]) + Sync) -> T {
        let source = x.data();
        let data = self.generate(x.len(), CHUNK, |start, part| {
            f(&source[start..][..part.len()], part)
        });
        FastTensor::new(data, &x.shape)
    }

    /// Applique `f(entrée, gradient, sortie)` par tranches, en parallèle.
    fn binary(&self, x: &T, grad: &T, f: impl Fn(&[f32], &[f32], &mut [f32]) + Sync) -> T {
        assert_eq!(x.shape, grad.shape, "gradient de forme inattendue");
        let (source, grad) = (x.data(), grad.data());
        let data = self.generate(x.len(), CHUNK, |start, part| {
            let len = part.len();
            f(&source[start..][..len], &grad[start..][..len], part)
        });
        FastTensor::new(data, &x.shape)
    }
}

/// GELU et son gradient, vectorisés : la tangente hyperbolique passe par
/// l'exponentielle de `math.rs`.
impl Run<Gelu> for Fast {
    fn run(&self, _: &Gelu, inputs: &[&T]) -> T {
        let [x] = operands(Gelu::NAME, inputs);
        let isa = self.isa;
        self.unary(x, |x, out| math::gelu(isa, x, out))
    }

    fn run_backward(&self, _: &Gelu, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [x] = operands(Gelu::NAME, inputs);
        let isa = self.isa;
        vec![Some(self.binary(x, grad, |x, g, out| {
            math::gelu_backward(isa, x, g, out)
        }))]
    }
}

impl Run<Silu> for Fast {
    fn run(&self, _: &Silu, inputs: &[&T]) -> T {
        let [x] = operands(Silu::NAME, inputs);
        let isa = self.isa;
        self.unary(x, |x, out| math::silu(isa, x, out))
    }

    fn run_backward(&self, _: &Silu, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [x] = operands(Silu::NAME, inputs);
        let isa = self.isa;
        vec![Some(self.binary(x, grad, |x, g, out| {
            math::silu_backward(isa, x, g, out)
        }))]
    }
}

impl Fast {
    /// La rotation de RoPE, ou son inverse.
    fn rotate(&self, x: &T, cos: &T, sin: &T, inverse: bool) -> T {
        let rank = x.shape.len();
        assert!(
            rank >= 2 && x.shape[rank - 1].is_multiple_of(2),
            "rotate_pairs attend […, T, D] avec D pair, reçu {:?}",
            x.shape,
        );
        let (length, width) = (x.shape[rank - 2], x.shape[rank - 1]);
        let half = width / 2;
        assert!(
            cos.shape == [length, half] && sin.shape == [length, half],
            "rotate_pairs : angles {:?} et {:?} pour un tenseur {:?}, [{length}, {half}] attendu",
            cos.shape,
            sin.shape,
            x.shape,
        );
        let (source, cos, sin) = (x.data(), cos.data(), sin.data());
        let rows_per_task = (CHUNK / width).max(1);
        let data = self.generate(x.len(), rows_per_task * width, |start, part| {
            let first = start / width;
            for (offset, line) in part.chunks_mut(width).enumerate() {
                let row = first + offset;
                let position = row % length;
                let input = &source[row * width..][..width];
                for pair in 0..half {
                    let c = cos[position * half + pair];
                    let s = sin[position * half + pair];
                    let s = if inverse { -s } else { s };
                    let (a, b) = (input[2 * pair], input[2 * pair + 1]);
                    line[2 * pair] = a * c - b * s;
                    line[2 * pair + 1] = a * s + b * c;
                }
            }
        });
        FastTensor::new(data, &x.shape)
    }
}

impl Run<RotatePairs> for Fast {
    fn run(&self, _: &RotatePairs, inputs: &[&T]) -> T {
        let [x, cos, sin] = operands(RotatePairs::NAME, inputs);
        self.rotate(x, cos, sin, false)
    }

    fn run_backward(&self, _: &RotatePairs, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [_, cos, sin] = operands(RotatePairs::NAME, inputs);
        vec![Some(self.rotate(grad, cos, sin, true)), None, None]
    }
}

impl Fast {
    /// Vérifie les formes de l'entropie croisée et renvoie `(N, V)`.
    fn cross_entropy_dims(logits: &T, targets: &[u32]) -> (usize, usize) {
        assert!(
            logits.shape.len() == 2 && logits.shape[0] == targets.len(),
            "cross_entropy : logits {:?} pour {} cibles, [N, V] attendu",
            logits.shape,
            targets.len(),
        );
        let vocab = logits.shape[1];
        for &target in targets {
            assert!(
                (target as usize) < vocab,
                "cible {target} hors du vocabulaire ({vocab})"
            );
        }
        (logits.shape[0], vocab)
    }
}

impl Run<CrossEntropy> for Fast {
    /// `-ln softmax(logits)[cible]`, ligne par ligne, puis la moyenne.
    fn run(&self, op: &CrossEntropy, inputs: &[&T]) -> T {
        let [logits] = operands(CrossEntropy::NAME, inputs);
        let (n, vocab) = Self::cross_entropy_dims(logits, &op.targets);
        let log_probabilities = self.log_softmax(logits);
        let data = log_probabilities.data();
        let total: f32 = op
            .targets
            .iter()
            .enumerate()
            .fold(0.0, |total, (row, &target)| {
                total - data[row * vocab + target as usize]
            });
        FastTensor::new(vec![total / n as f32], &[])
    }

    /// `(softmax(logits) - onehot(cibles)) / N`, multiplié par le gradient
    /// de la perte.
    fn run_backward(&self, op: &CrossEntropy, inputs: &[&T], _: &T, grad: &T) -> Vec<Option<T>> {
        let [logits] = operands(CrossEntropy::NAME, inputs);
        let (n, vocab) = Self::cross_entropy_dims(logits, &op.targets);
        let mut probabilities = self.softmax(logits);
        let scale = grad.data()[0] / n as f32;
        let targets = &op.targets;
        let rows_per_task = (CHUNK / vocab).max(1);
        self.for_parts(
            probabilities.data_mut(),
            rows_per_task * vocab,
            |start, part| {
                let first = start / vocab;
                for (offset, line) in part.chunks_mut(vocab).enumerate() {
                    line[targets[first + offset] as usize] -= 1.0;
                    for x in line.iter_mut() {
                        *x *= scale;
                    }
                }
            },
        );
        vec![Some(probabilities)]
    }
}

/// L'attention en tuiles (voir `attention.rs`). La passe avant garde `L`,
/// une valeur par requête. Des entrées aux dimensions de lot différentes
/// passent par la définition de référence.
impl Run<FusedAttention> for Fast {
    fn run(&self, op: &FusedAttention, inputs: &[&T]) -> T {
        self.run_saving(op, inputs).0
    }

    fn run_saving(&self, op: &FusedAttention, inputs: &[&T]) -> (T, Option<T>) {
        let [q, k, v] = operands(FusedAttention::NAME, inputs);
        match attention::Dims::of(q, k, v) {
            Some(dims) => {
                let (out, log_sum) = attention::forward(self, dims, q, k, v, op.scale as f32);
                (out, Some(log_sum))
            }
            None => (op.reference(self, inputs), None),
        }
    }

    fn run_backward(
        &self,
        op: &FusedAttention,
        inputs: &[&T],
        output: &T,
        grad: &T,
    ) -> Vec<Option<T>> {
        self.run_backward_saved(op, inputs, output, None, grad)
    }

    fn run_backward_saved(
        &self,
        op: &FusedAttention,
        inputs: &[&T],
        output: &T,
        saved: Option<&T>,
        grad: &T,
    ) -> Vec<Option<T>> {
        let [q, k, v] = operands(FusedAttention::NAME, inputs);
        let Some(dims) = attention::Dims::of(q, k, v) else {
            return op.reference_backward(self, inputs, output, grad);
        };
        let recomputed;
        let log_sum = match saved {
            Some(log_sum) => log_sum,
            None => {
                recomputed = attention::forward(self, dims, q, k, v, op.scale as f32).1;
                &recomputed
            }
        };
        attention::backward(self, dims, q, k, v, output, log_sum, grad, op.scale as f32)
            .map(Some)
            .into()
    }
}

/// L'attention sur le cache, une tâche par requête et par tête : les scores
/// des positions de sa séquence, lues dans les tables à leurs lignes, le
/// softmax, puis le mélange des valeurs.
impl Run<CachedAttention> for Fast {
    fn run(&self, op: &CachedAttention, inputs: &[&T]) -> T {
        let dims = op.dims(self, inputs);
        let [q, keys, values, slots, rows] = operands(CachedAttention::NAME, inputs);
        let (q, keys, values, slots, rows) = (
            q.data(),
            keys.data(),
            values.data(),
            slots.data(),
            rows.data(),
        );
        let group = dims.heads / dims.kv_heads;
        let (key_dim, value_dim, stride) = (dims.key_dim, dims.value_dim, dims.value_stride);
        let isa = self.isa;
        let scale = op.scale as f32;
        for row in 0..dims.queries {
            let (sequence, count) = (rows[2 * row] as usize, rows[2 * row + 1] as usize);
            assert!(
                sequence < dims.sequences && (1..=dims.max_len).contains(&count),
                "cached_attention : la requête {row} voit {count} positions de la séquence {sequence}",
            );
        }
        let data = self.generate(
            dims.queries * dims.heads * value_dim,
            value_dim.max(1),
            |start, out| {
                let task = start / value_dim.max(1);
                let (row, head) = (task / dims.heads, task % dims.heads);
                let kv_head = head / group;
                let (sequence, count) = (rows[2 * row] as usize, rows[2 * row + 1] as usize);
                let lines = &slots[sequence * dims.max_len..][..count];
                let query = &q[(row * dims.heads + head) * key_dim..][..key_dim];
                let mut scores: Vec<f32> = lines
                    .iter()
                    .map(|&line| {
                        let key =
                            &keys[(line as usize * dims.kv_heads + kv_head) * key_dim..][..key_dim];
                        scale * math::dot(isa, query, key)
                    })
                    .collect();
                let max = math::max(isa, &scores);
                let total = math::exp_shifted_sum(isa, &mut scores, max);
                out.fill(0.0);
                for (&line, &weight) in lines.iter().zip(&scores) {
                    let value =
                        &values[(line as usize * dims.kv_heads + kv_head) * stride..][..value_dim];
                    math::axpy(isa, weight, value, out);
                }
                math::scale(isa, out, 1.0 / total);
            },
        );
        FastTensor::new(data, &[dims.queries, dims.heads, value_dim])
    }

    fn run_backward(
        &self,
        op: &CachedAttention,
        inputs: &[&T],
        output: &T,
        grad: &T,
    ) -> Vec<Option<T>> {
        op.reference_backward(self, inputs, output, grad)
    }
}

/// Le produit par des poids quantifiés : en lisant les codes quand `x` a peu
/// de lignes (voir `quantized.rs`), par les poids déquantifiés sinon.
impl Run<QuantizedMatmul> for Fast {
    fn run(&self, op: &QuantizedMatmul, inputs: &[&T]) -> T {
        let (x, words, scales, dequantized, bias) = QuantizedMatmul::operands(inputs);
        match self.quantized_product(op.format, x, words, scales, dequantized, bias) {
            Some(out) => out,
            None => op.reference(self, inputs),
        }
    }

    fn run_backward(
        &self,
        op: &QuantizedMatmul,
        inputs: &[&T],
        output: &T,
        grad: &T,
    ) -> Vec<Option<T>> {
        op.reference_backward(self, inputs, output, grad)
    }
}
