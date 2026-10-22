//! Les primitives de [`Fast`], et les parcours parallèles qui les écrivent.

use std::sync::Arc;

use super::super::shape::{
    broadcast_shapes, broadcast_strides, check_expand, check_mask, check_row_index, check_step,
    concat_rows_shape, gather_rows_dims, scatter_add_rows_dims, source_index, strides,
    write_rows_dims,
};
use super::super::{AdamWStep, Primitives, float};
use super::gemm::Matrix;
use super::pool::SharedMut;
use super::{Buffer, CHUNK, Fast, FastTensor, math};

impl Fast {
    /// Remplit un nouveau tampon de `len` éléments, par tâches d'au moins
    /// `chunk` éléments : `fill(début, tranche)`.
    pub(super) fn generate(
        &self,
        len: usize,
        chunk: usize,
        fill: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Buffer {
        // Chaque élément est écrit par exactement une tâche avant d'être lu.
        let mut out = Buffer::any(len);
        self.for_parts(&mut out, chunk, fill);
        out
    }

    /// Un tampon de zéros.
    fn zeros_buffer(&self, len: usize) -> Buffer {
        self.generate(len, 4 * CHUNK, |_, part| part.fill(0.0))
    }

    /// Applique `work(début, tranche)` à des tranches d'au moins `chunk`
    /// éléments de `data`, en parallèle.
    pub(super) fn for_parts(
        &self,
        data: &mut [f32],
        chunk: usize,
        work: impl Fn(usize, &mut [f32]) + Sync,
    ) {
        let len = data.len();
        let chunk = chunk.max(1);
        let shared = SharedMut::new(data);
        self.pool.run(len.div_ceil(chunk), |task| {
            let start = task * chunk;
            // SAFETY : chaque tâche prend sa propre tranche.
            let part = unsafe { shared.slice(start, chunk.min(len - start)) };
            work(start, part);
        });
    }

    /// Applique `f` à chaque élément.
    fn map(&self, a: &FastTensor, f: impl Fn(f32) -> f32 + Sync) -> FastTensor {
        let source = a.data();
        let data = self.generate(a.len(), CHUNK, |start, part| {
            for (out, &x) in part.iter_mut().zip(&source[start..]) {
                *out = f(x);
            }
        });
        FastTensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }

    /// Applique `f` aux paires d'éléments correspondants, après broadcasting.
    fn zip_with(
        &self,
        a: &FastTensor,
        b: &FastTensor,
        f: impl Fn(f32, f32) -> f32 + Sync,
    ) -> FastTensor {
        let shape = broadcast_shapes(&a.shape, &b.shape);
        let len: usize = shape.iter().product();
        let (data_a, data_b) = (a.data(), b.data());
        if a.shape == b.shape {
            let data = self.generate(len, CHUNK, |start, part| {
                let (xs, ys) = (&data_a[start..], &data_b[start..]);
                for ((out, &x), &y) in part.iter_mut().zip(xs).zip(ys) {
                    *out = f(x, y);
                }
            });
            return FastTensor::new(data, &shape);
        }
        // Ligne par ligne du dernier axe : chaque opérande y avance d'un pas
        // de 1, ou reste sur place s'il est étiré.
        let width = shape.last().copied().unwrap_or(1).max(1);
        let strides_a = broadcast_strides(&a.shape, &shape);
        let strides_b = broadcast_strides(&b.shape, &shape);
        let step_a = strides_a.last().copied().unwrap_or(0);
        let step_b = strides_b.last().copied().unwrap_or(0);
        let rows_per_task = (CHUNK / width).max(1);
        let data = self.generate(len, rows_per_task * width, |start, part| {
            for (offset, line) in part.chunks_mut(width).enumerate() {
                let flat = start + offset * width;
                let origin_a = source_index(flat, &shape, &strides_a);
                let origin_b = source_index(flat, &shape, &strides_b);
                match (step_a, step_b) {
                    (1, 1) => {
                        let (xs, ys) = (&data_a[origin_a..][..width], &data_b[origin_b..][..width]);
                        for ((out, &x), &y) in line.iter_mut().zip(xs).zip(ys) {
                            *out = f(x, y);
                        }
                    }
                    (1, _) => {
                        let y = data_b[origin_b];
                        for (out, &x) in line.iter_mut().zip(&data_a[origin_a..][..width]) {
                            *out = f(x, y);
                        }
                    }
                    (_, 1) => {
                        let x = data_a[origin_a];
                        for (out, &y) in line.iter_mut().zip(&data_b[origin_b..][..width]) {
                            *out = f(x, y);
                        }
                    }
                    _ => line.fill(f(data_a[origin_a], data_b[origin_b])),
                }
            }
        });
        FastTensor::new(data, &shape)
    }

    /// Copie les éléments de `a` vus à travers des pas quelconques, dans la
    /// forme `shape` : la transposition et l'étirement.
    fn gather_strided(
        &self,
        a: &FastTensor,
        shape: &[usize],
        source_strides: &[usize],
    ) -> FastTensor {
        let len: usize = shape.iter().product();
        let width = shape.last().copied().unwrap_or(1).max(1);
        let step = source_strides.last().copied().unwrap_or(0);
        let source = a.data();
        let rows_per_task = (CHUNK / width).max(1);
        let data = self.generate(len, rows_per_task * width, |start, part| {
            for (offset, line) in part.chunks_mut(width).enumerate() {
                let origin = source_index(start + offset * width, shape, source_strides);
                match step {
                    1 => line.copy_from_slice(&source[origin..][..width]),
                    0 => line.fill(source[origin]),
                    step => {
                        for (index, out) in line.iter_mut().enumerate() {
                            *out = source[origin + index * step];
                        }
                    }
                }
            }
        });
        FastTensor::new(data, shape)
    }

    /// Échange les deux derniers axes, par tuiles de 32 × 32 : chaque tuile
    /// lue et écrite reste dans le cache.
    fn transpose_last(&self, a: &FastTensor) -> FastTensor {
        const TILE: usize = 32;
        let rank = a.shape.len();
        let (rows, cols) = (a.shape[rank - 2], a.shape[rank - 1]);
        let matrices: usize = a.shape[..rank - 2].iter().product();
        let source = a.data();
        let mut shape = a.shape.clone();
        shape.swap(rank - 2, rank - 1);
        let tiles_r = rows.div_ceil(TILE);
        let tiles_c = cols.div_ceil(TILE);
        let mut data = Buffer::any(a.len());
        let out = SharedMut::new(&mut data);
        let tiles_per_task = (CHUNK / (TILE * TILE)).max(1);
        let total = matrices * tiles_r * tiles_c;
        self.pool.run(total.div_ceil(tiles_per_task), |task| {
            for tile in task * tiles_per_task..total.min((task + 1) * tiles_per_task) {
                let matrix = tile / (tiles_r * tiles_c);
                let (tr, tc) = ((tile / tiles_c) % tiles_r, tile % tiles_c);
                let base = matrix * rows * cols;
                for i in tr * TILE..rows.min((tr + 1) * TILE) {
                    for j in tc * TILE..cols.min((tc + 1) * TILE) {
                        // SAFETY : chaque tuile écrit ses propres éléments.
                        unsafe {
                            *out.ptr(base + j * rows + i) = source[base + i * cols + j];
                        }
                    }
                }
            }
        });
        FastTensor::new(data, &shape)
    }

    /// Le produit matriciel sur les deux dernières dimensions, avec un
    /// opérande éventuellement transposé (sans le recopier) et les
    /// dimensions de lot soumises au broadcasting.
    pub fn product(
        &self,
        a: &FastTensor,
        transpose_a: bool,
        b: &FastTensor,
        transpose_b: bool,
    ) -> FastTensor {
        assert!(
            a.shape.len() >= 2 && b.shape.len() >= 2,
            "matmul demande au moins 2 dimensions : {:?} et {:?}",
            a.shape,
            b.shape,
        );
        let (batch_a, matrix_a) = a.shape.split_at(a.shape.len() - 2);
        let (batch_b, matrix_b) = b.shape.split_at(b.shape.len() - 2);
        let (rows, inner) = if transpose_a {
            (matrix_a[1], matrix_a[0])
        } else {
            (matrix_a[0], matrix_a[1])
        };
        let (inner_b, cols) = if transpose_b {
            (matrix_b[1], matrix_b[0])
        } else {
            (matrix_b[0], matrix_b[1])
        };
        assert_eq!(
            inner, inner_b,
            "matmul : formes incompatibles {:?} et {:?}",
            a.shape, b.shape,
        );
        let batch = broadcast_shapes(batch_a, batch_b);
        // Une même matrice `B` pour tout le lot, comme dans une couche
        // linéaire : les matrices de `A` se suivent, et forment ensemble une
        // seule grande matrice. Un seul produit, une seule fois `B` emballée.
        if !transpose_a && batch_b.iter().all(|&dim| dim == 1) && batch_a == batch.as_slice() {
            let total_rows = a.len() / inner;
            let ma = Matrix::new(a.data(), total_rows, inner);
            let mb = Matrix::new(b.data(), matrix_b[0], matrix_b[1]);
            let mb = if transpose_b { mb.t() } else { mb };
            let mut data = Buffer::any(total_rows * cols);
            self.gemm_into(ma, mb, &mut data, false);
            let mut shape = batch;
            shape.extend([rows, cols]);
            return FastTensor::new(data, &shape);
        }
        let strides_a = broadcast_strides(batch_a, &batch);
        let strides_b = broadcast_strides(batch_b, &batch);
        let batch_len: usize = batch.iter().product();
        let size_a = matrix_a[0] * matrix_a[1];
        let size_b = matrix_b[0] * matrix_b[1];
        let mut data = Buffer::any(batch_len * rows * cols);
        fn matrix<'a>(
            data: &'a [f32],
            start: usize,
            shape: &[usize],
            transposed: bool,
        ) -> Matrix<'a> {
            let size = shape[0] * shape[1];
            let m = Matrix::new(&data[start..start + size], shape[0], shape[1]);
            if transposed { m.t() } else { m }
        }
        let work = rows * inner * cols;
        let out = SharedMut::new(&mut data);
        let one = |index: usize| {
            let start_a = source_index(index, &batch, &strides_a) * size_a;
            let start_b = source_index(index, &batch, &strides_b) * size_b;
            let ma = matrix(a.data(), start_a, matrix_a, transpose_a);
            let mb = matrix(b.data(), start_b, matrix_b, transpose_b);
            // SAFETY : chaque tâche écrit sa propre matrice du lot.
            let target = unsafe { out.slice(index * rows * cols, rows * cols) };
            self.gemm_into(ma, mb, target, false);
        };
        // De petits produits nombreux se partagent entre les fils, un par
        // tâche ; un gros produit se partage lui-même.
        if batch_len > 1 && work < 1 << 22 {
            self.pool.run(batch_len, one);
        } else {
            (0..batch_len).for_each(one);
        }
        let mut shape = batch;
        shape.extend([rows, cols]);
        FastTensor::new(data, &shape)
    }

    /// Applique `f` à chaque ligne du dernier axe, en parallèle.
    pub(super) fn map_rows(
        &self,
        a: &FastTensor,
        f: impl Fn(&[f32], &mut [f32]) + Sync,
    ) -> FastTensor {
        let width = *a.shape.last().expect("au moins une dimension");
        let source = a.data();
        if width == 0 {
            return a.clone();
        }
        let rows_per_task = (CHUNK / width).max(1);
        let data = self.generate(a.len(), rows_per_task * width, |start, part| {
            for (input, output) in source[start..].chunks(width).zip(part.chunks_mut(width)) {
                f(input, output);
            }
        });
        FastTensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
        }
    }
}

impl Primitives for Fast {
    type Scalar = f32;
    type Tensor = FastTensor;

    fn from_vec(&self, data: Vec<f32>, shape: &[usize]) -> FastTensor {
        FastTensor::new(data, shape)
    }

    fn to_vec(&self, tensor: &FastTensor) -> Vec<f32> {
        tensor.data.to_vec()
    }

    fn shape<'a>(&self, tensor: &'a FastTensor) -> &'a [usize] {
        &tensor.shape
    }

    fn add(&self, a: &FastTensor, b: &FastTensor) -> FastTensor {
        self.zip_with(a, b, |x, y| x + y)
    }

    fn sub(&self, a: &FastTensor, b: &FastTensor) -> FastTensor {
        self.zip_with(a, b, |x, y| x - y)
    }

    fn mul(&self, a: &FastTensor, b: &FastTensor) -> FastTensor {
        self.zip_with(a, b, |x, y| x * y)
    }

    fn div(&self, a: &FastTensor, b: &FastTensor) -> FastTensor {
        self.zip_with(a, b, |x, y| x / y)
    }

    fn neg(&self, a: &FastTensor) -> FastTensor {
        self.map(a, |x| -x)
    }

    fn exp(&self, a: &FastTensor) -> FastTensor {
        let isa = self.isa;
        let source = a.data();
        let data = self.generate(a.len(), CHUNK, |start, part| {
            part.copy_from_slice(&source[start..][..part.len()]);
            math::exp_in_place(isa, part);
        });
        FastTensor::new(data, &a.shape)
    }

    fn ln(&self, a: &FastTensor) -> FastTensor {
        self.map(a, f32::ln)
    }

    fn sqrt(&self, a: &FastTensor) -> FastTensor {
        self.map(a, f32::sqrt)
    }

    fn tanh(&self, a: &FastTensor) -> FastTensor {
        self.map(a, f32::tanh)
    }

    fn sum(&self, a: &FastTensor, axis: usize, keepdim: bool) -> FastTensor {
        assert!(
            axis < a.shape.len(),
            "axe {axis} hors de la forme {:?}",
            a.shape
        );
        let n = a.shape[axis];
        let outer: usize = a.shape[..axis].iter().product();
        let inner: usize = a.shape[axis + 1..].iter().product();
        let source = a.data();
        let isa = self.isa;
        let mut shape = a.shape.clone();
        if keepdim {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        if inner == 1 {
            // Des lignes contiguës.
            let rows_per_task = (CHUNK / n.max(1)).max(1);
            let data = self.generate(outer, rows_per_task, |start, part| {
                for (offset, out) in part.iter_mut().enumerate() {
                    *out = math::sum(isa, &source[(start + offset) * n..][..n]);
                }
            });
            return FastTensor::new(data, &shape);
        }
        // Des lignes `inner` additionnées les unes aux autres. Quand il y a
        // peu de colonnes et beaucoup de lignes, comme pour le gradient d'un
        // biais, les lignes sont coupées en paquets fixes de `ROWS` : chaque
        // paquet donne une somme partielle, et les sommes partielles
        // s'additionnent dans l'ordre.
        const COLUMNS: usize = 1024;
        const ROWS: usize = 256;
        let column_parts = inner.div_ceil(COLUMNS);
        let row_parts = if outer * column_parts < 16 && n > 2 * ROWS {
            n.div_ceil(ROWS)
        } else {
            1
        };
        let rows_per_part = n.div_ceil(row_parts);
        let mut partial = self.zeros_buffer(row_parts * outer * inner);
        let shared = SharedMut::new(&mut partial);
        self.pool.run(row_parts * outer * column_parts, |task| {
            let part = task / (outer * column_parts);
            let o = (task / column_parts) % outer;
            let first = (task % column_parts) * COLUMNS;
            let width = COLUMNS.min(inner - first);
            // SAFETY : chaque tâche écrit sa propre somme partielle.
            let target = unsafe { shared.slice((part * outer + o) * inner + first, width) };
            for k in part * rows_per_part..n.min((part + 1) * rows_per_part) {
                let line = &source[(o * n + k) * inner + first..][..width];
                for (sum, &x) in target.iter_mut().zip(line) {
                    *sum += x;
                }
            }
        });
        let total = outer * inner;
        for part in 1..row_parts {
            let (head, tail) = partial.split_at_mut(part * total);
            for (sum, &x) in head[..total].iter_mut().zip(&tail[..total]) {
                *sum += x;
            }
        }
        partial.truncate(total);
        FastTensor::new(partial, &shape)
    }

    fn sum_all(&self, a: &FastTensor) -> FastTensor {
        let source = a.data();
        let isa = self.isa;
        let parts = a.len().div_ceil(CHUNK);
        let partial = self.generate(parts, 1, |start, part| {
            for (offset, out) in part.iter_mut().enumerate() {
                let first = (start + offset) * CHUNK;
                *out = math::sum(isa, &source[first..a.len().min(first + CHUNK)]);
            }
        });
        let total = partial.iter().fold(0.0, |total, &x| total + x);
        FastTensor::new(vec![total], &[])
    }

    fn reshape(&self, a: &FastTensor, shape: &[usize]) -> FastTensor {
        let expected: usize = shape.iter().product();
        assert_eq!(
            a.len(),
            expected,
            "{} éléments fournis pour la forme {shape:?}",
            a.len(),
        );
        FastTensor {
            data: Arc::clone(&a.data),
            shape: shape.to_vec(),
        }
    }

    fn transpose(&self, a: &FastTensor, axis_a: usize, axis_b: usize) -> FastTensor {
        let rank = a.shape.len();
        if axis_a == axis_b {
            return a.clone();
        }
        if (axis_a.min(axis_b), axis_a.max(axis_b)) == (rank - 2, rank - 1) {
            return self.transpose_last(a);
        }
        let mut shape = a.shape.clone();
        shape.swap(axis_a, axis_b);
        let mut source_strides = strides(&a.shape);
        source_strides.swap(axis_a, axis_b);
        self.gather_strided(a, &shape, &source_strides)
    }

    fn expand(&self, a: &FastTensor, shape: &[usize]) -> FastTensor {
        check_expand(&a.shape, shape);
        let source_strides = broadcast_strides(&a.shape, shape);
        self.gather_strided(a, shape, &source_strides)
    }

    fn matmul(&self, a: &FastTensor, b: &FastTensor) -> FastTensor {
        self.product(a, false, b, false)
    }

    fn softmax(&self, a: &FastTensor) -> FastTensor {
        let isa = self.isa;
        self.map_rows(a, |input, output| {
            let max = math::max(isa, input);
            output.copy_from_slice(input);
            let total = math::exp_shifted_sum(isa, output, max);
            math::scale(isa, output, 1.0 / total);
        })
    }

    fn log_softmax(&self, a: &FastTensor) -> FastTensor {
        let isa = self.isa;
        self.map_rows(a, |input, output| {
            let max = math::max(isa, input);
            let shift = max + math::exp_shifted_total(isa, input, max).ln();
            for (out, &x) in output.iter_mut().zip(input) {
                *out = x - shift;
            }
        })
    }

    fn masked_fill(&self, a: &FastTensor, mask: &FastTensor, value: f32) -> FastTensor {
        check_mask(&a.shape, &mask.shape);
        self.zip_with(a, mask, |x, m| if m != 0.0 { value } else { x })
    }

    fn gather_rows(&self, a: &FastTensor, indices: &[u32]) -> FastTensor {
        let row_len = gather_rows_dims(&a.shape, indices);
        let source = a.data();
        let rows_per_task = (CHUNK / row_len.max(1)).max(1);
        let data = self.generate(
            indices.len() * row_len,
            rows_per_task * row_len.max(1),
            |start, part| {
                let first = start / row_len.max(1);
                for (offset, line) in part.chunks_mut(row_len.max(1)).enumerate() {
                    let index = indices[first + offset] as usize;
                    line.copy_from_slice(&source[index * row_len..][..line.len()]);
                }
            },
        );
        let mut shape = a.shape.clone();
        shape[0] = indices.len();
        FastTensor::new(data, &shape)
    }

    fn scatter_add_rows(&self, a: &FastTensor, indices: &[u32], num_rows: usize) -> FastTensor {
        let row_len = scatter_add_rows_dims(&a.shape, indices, num_rows);
        let source = a.data();
        let mut data = self.zeros_buffer(num_rows * row_len);
        // Chaque tâche prend un groupe de colonnes, pour toutes les lignes :
        // les lignes de même indice s'additionnent dans leur ordre.
        const COLUMNS: usize = 64;
        let out = SharedMut::new(&mut data);
        self.pool.run(row_len.div_ceil(COLUMNS), |task| {
            let first = task * COLUMNS;
            let width = COLUMNS.min(row_len - first);
            for (row, &index) in indices.iter().enumerate() {
                // SAFETY : chaque tâche écrit son groupe de colonnes, dans
                // toutes les lignes.
                let target = unsafe { out.slice(index as usize * row_len + first, width) };
                for (sum, &x) in target
                    .iter_mut()
                    .zip(&source[row * row_len + first..][..width])
                {
                    *sum += x;
                }
            }
        });
        let mut shape = a.shape.clone();
        shape[0] = num_rows;
        FastTensor::new(data, &shape)
    }

    fn concat_rows(&self, a: &FastTensor, b: &FastTensor) -> FastTensor {
        let shape = concat_rows_shape(&a.shape, &b.shape);
        let mut data = Buffer::any(a.len() + b.len());
        data[..a.len()].copy_from_slice(a.data());
        data[a.len()..].copy_from_slice(b.data());
        FastTensor::new(data, &shape)
    }

    fn write_rows(&self, table: &mut FastTensor, indices: &FastTensor, rows: &FastTensor) {
        let row_len = write_rows_dims(&table.shape, indices.len(), &rows.shape);
        let lines = table.shape[0];
        let target = table.data_mut();
        for (row, &index) in indices.data().iter().enumerate() {
            let index = index as usize;
            check_row_index(index, lines);
            target[index * row_len..(index + 1) * row_len]
                .copy_from_slice(&rows.data()[row * row_len..(row + 1) * row_len]);
        }
    }

    fn sgd_step(&self, param: &mut FastTensor, grad: &FastTensor, learning_rate: f64) {
        check_step(&param.shape, &grad.shape, &[]);
        let learning_rate = learning_rate as f32;
        let grad = grad.data();
        self.for_parts(param.data_mut(), CHUNK, |start, part| {
            for (p, &g) in part.iter_mut().zip(&grad[start..]) {
                *p -= learning_rate * g;
            }
        });
    }

    fn adamw_step(
        &self,
        param: &mut FastTensor,
        grad: &FastTensor,
        first_moment: &mut FastTensor,
        second_moment: &mut FastTensor,
        step: &AdamWStep,
    ) {
        check_step(
            &param.shape,
            &grad.shape,
            &[&first_moment.shape, &second_moment.shape],
        );
        let step = step.scalars::<f32>();
        let grad = grad.data();
        let first = SharedMut::new(first_moment.data_mut());
        let second = SharedMut::new(second_moment.data_mut());
        self.for_parts(param.data_mut(), CHUNK, |start, part| {
            let len = part.len();
            // SAFETY : chaque tâche prend la même tranche des deux moments
            // que du paramètre, et les tranches des tâches sont disjointes.
            let (m_part, v_part) = unsafe { (first.slice(start, len), second.slice(start, len)) };
            for i in 0..len {
                (part[i], m_part[i], v_part[i]) =
                    float::adamw_update(part[i], grad[start + i], m_part[i], v_part[i], &step);
            }
        });
    }
}
