//! Tenseur CPU de référence.
//!
//! Chaque opération est écrite de la façon la plus directe possible et alloue
//! son résultat : ce code fixe la sémantique des opérations et sert d'oracle
//! aux kernels GPU, il n'est pas fait pour être rapide.
//!
//! Une erreur de forme est une erreur de programmation : les opérations
//! paniquent avec un message plutôt que de renvoyer un `Result`.

use std::marker::PhantomData;

use rand::Rng;

use super::shape::{broadcast_shapes, broadcast_strides, source_index, strides};
use super::{AdamWStep, Float, Primitives, float, standard_normal};

/// Tableau à N dimensions, stocké à plat ligne par ligne (row-major) : le
/// dernier indice est celui qui varie le plus vite.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor<F> {
    data: Vec<F>,
    shape: Vec<usize>,
}

impl<F: Float> CpuTensor<F> {
    // --- Constructeurs ---

    pub fn from_vec(data: Vec<F>, shape: &[usize]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "{} éléments fournis pour la forme {shape:?}",
            data.len(),
        );
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn full(shape: &[usize], value: F) -> Self {
        Self::from_vec(vec![value; shape.iter().product()], shape)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, F::ZERO)
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, F::ONE)
    }

    /// Tenseur à un seul élément, de forme `[]`. Il s'étire à n'importe quelle
    /// forme : c'est la façon de multiplier ou diviser par une constante.
    pub fn scalar(value: F) -> Self {
        Self::from_vec(vec![value], &[])
    }

    /// `[0, 1, …, len - 1]`.
    pub fn arange(len: usize) -> Self {
        let data = (0..len).map(|i| F::from_f64(i as f64)).collect();
        Self::from_vec(data, &[len])
    }

    /// Tirages indépendants d'une loi normale centrée réduite.
    pub fn randn(shape: &[usize], rng: &mut impl Rng) -> Self {
        let len: usize = shape.iter().product();
        let data = (0..len)
            .map(|_| F::from_f64(standard_normal(rng)))
            .collect();
        Self::from_vec(data, shape)
    }

    // --- Accès ---

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[F] {
        &self.data
    }

    /// Valeur d'un tenseur à un seul élément.
    pub fn item(&self) -> F {
        assert_eq!(
            self.data.len(),
            1,
            "item() sur un tenseur de forme {:?}",
            self.shape,
        );
        self.data[0]
    }

    // --- Opérations élément par élément ---

    fn map(&self, f: impl Fn(F) -> F) -> Self {
        Self {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
        }
    }

    /// Applique `f` aux paires d'éléments correspondants, après broadcasting.
    fn zip_with(&self, other: &Self, f: impl Fn(F, F) -> F) -> Self {
        let shape = broadcast_shapes(&self.shape, &other.shape);
        let strides_a = broadcast_strides(&self.shape, &shape);
        let strides_b = broadcast_strides(&other.shape, &shape);
        let len: usize = shape.iter().product();
        let data = (0..len)
            .map(|flat| {
                let a = self.data[source_index(flat, &shape, &strides_a)];
                let b = other.data[source_index(flat, &shape, &strides_b)];
                f(a, b)
            })
            .collect();
        Self { data, shape }
    }

    pub fn add(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a / b)
    }

    pub fn neg(&self) -> Self {
        self.map(|x| -x)
    }

    pub fn exp(&self) -> Self {
        self.map(F::exp)
    }

    pub fn ln(&self) -> Self {
        self.map(F::ln)
    }

    pub fn sqrt(&self) -> Self {
        self.map(F::sqrt)
    }

    pub fn tanh(&self) -> Self {
        self.map(F::tanh)
    }

    // --- Réductions ---

    /// Somme le long de `axis`. Avec `keepdim`, l'axe réduit reste présent
    /// avec la taille 1, ce qui permet de rediffuser le résultat sur
    /// l'entrée par broadcasting.
    pub fn sum(&self, axis: usize, keepdim: bool) -> Self {
        assert!(
            axis < self.shape.len(),
            "axe {axis} hors de la forme {:?}",
            self.shape,
        );
        // On voit le tenseur comme `[outer, n, inner]` : `n` est l'axe réduit.
        let n = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();

        let mut data = vec![F::ZERO; outer * inner];
        for o in 0..outer {
            for k in 0..n {
                for i in 0..inner {
                    data[o * inner + i] += self.data[(o * n + k) * inner + i];
                }
            }
        }

        let mut shape = self.shape.clone();
        if keepdim {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        Self { data, shape }
    }

    /// Somme de tous les éléments, de forme `[]`.
    pub fn sum_all(&self) -> Self {
        let total = self.data.iter().fold(F::ZERO, |total, &x| total + x);
        Self::scalar(total)
    }

    pub fn mean(&self, axis: usize, keepdim: bool) -> Self {
        let n = F::from_f64(self.shape[axis] as f64);
        self.sum(axis, keepdim).map(|x| x / n)
    }

    // --- Changements de forme ---

    /// Même données, autre découpage. Aucun élément ne bouge.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        Self::from_vec(self.data.clone(), shape)
    }

    /// Échange deux axes. Contrairement à `reshape`, les éléments changent
    /// de place dans le stockage.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Self {
        let mut shape = self.shape.clone();
        shape.swap(axis_a, axis_b);
        // L'axe `axis_a` de la sortie parcourt l'axe `axis_b` de la source :
        // il suffit d'échanger les pas.
        let mut strides = strides(&self.shape);
        strides.swap(axis_a, axis_b);

        let data = (0..self.data.len())
            .map(|flat| self.data[source_index(flat, &shape, &strides)])
            .collect();
        Self { data, shape }
    }

    /// Étire le tenseur jusqu'à la forme `shape` en recopiant ses éléments.
    pub fn expand(&self, shape: &[usize]) -> Self {
        assert_eq!(
            broadcast_shapes(&self.shape, shape),
            shape,
            "la forme {:?} ne s'étire pas à {shape:?}",
            self.shape,
        );
        let strides = broadcast_strides(&self.shape, shape);
        let len: usize = shape.iter().product();
        let data = (0..len)
            .map(|flat| self.data[source_index(flat, shape, &strides)])
            .collect();
        Self::from_vec(data, shape)
    }

    // --- Algèbre linéaire ---

    /// Produit matriciel sur les deux dernières dimensions. Les dimensions
    /// précédentes sont des dimensions de lot, soumises au broadcasting :
    /// `[B, T, C] × [C, D] → [B, T, D]`.
    pub fn matmul(&self, other: &Self) -> Self {
        assert!(
            self.shape.len() >= 2 && other.shape.len() >= 2,
            "matmul demande au moins 2 dimensions : {:?} et {:?}",
            self.shape,
            other.shape,
        );
        let (batch_a, matrix_a) = self.shape.split_at(self.shape.len() - 2);
        let (batch_b, matrix_b) = other.shape.split_at(other.shape.len() - 2);
        let (rows, inner) = (matrix_a[0], matrix_a[1]);
        let cols = matrix_b[1];
        assert_eq!(
            inner, matrix_b[0],
            "matmul : formes incompatibles {:?} et {:?}",
            self.shape, other.shape,
        );

        // Les pas comptent ici en matrices entières, pas en éléments.
        let batch = broadcast_shapes(batch_a, batch_b);
        let strides_a = broadcast_strides(batch_a, &batch);
        let strides_b = broadcast_strides(batch_b, &batch);
        let batch_len: usize = batch.iter().product();

        let mut data = vec![F::ZERO; batch_len * rows * cols];
        for index in 0..batch_len {
            let start_a = source_index(index, &batch, &strides_a) * rows * inner;
            let start_b = source_index(index, &batch, &strides_b) * inner * cols;
            let a = &self.data[start_a..start_a + rows * inner];
            let b = &other.data[start_b..start_b + inner * cols];
            let out = &mut data[index * rows * cols..(index + 1) * rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = F::ZERO;
                    for k in 0..inner {
                        sum += a[i * inner + k] * b[k * cols + j];
                    }
                    out[i * cols + j] = sum;
                }
            }
        }

        let mut shape = batch;
        shape.extend([rows, cols]);
        Self { data, shape }
    }

    // --- Softmax ---

    /// Applique `f` à chaque ligne du dernier axe, en lui fournissant le
    /// maximum de la ligne.
    fn map_rows(&self, f: impl Fn(&mut [F], F)) -> Self {
        let width = *self
            .shape
            .last()
            .expect("softmax demande au moins une dimension");
        let mut data = self.data.clone();
        for row in data.chunks_mut(width) {
            let max = row
                .iter()
                .fold(F::NEG_INFINITY, |max, &x| if x > max { x } else { max });
            f(row, max);
        }
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// `exp(x) / Σ exp(x)` sur le dernier axe. Soustraire le maximum ne
    /// change pas le résultat, mais garantit que la plus grande exponentielle
    /// vaut `exp(0) = 1` : plus de dépassement de capacité.
    pub fn softmax(&self) -> Self {
        self.map_rows(|row, max| {
            let mut total = F::ZERO;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                total += *x;
            }
            for x in row.iter_mut() {
                *x = *x / total;
            }
        })
    }

    /// `ln(softmax(x))`, calculé sans passer par `softmax` : prendre le
    /// logarithme d'une probabilité arrondie à 0 donnerait `-∞`.
    pub fn log_softmax(&self) -> Self {
        self.map_rows(|row, max| {
            let mut total = F::ZERO;
            for &x in row.iter() {
                total += (x - max).exp();
            }
            let log_total = total.ln();
            for x in row.iter_mut() {
                *x = *x - max - log_total;
            }
        })
    }

    // --- Opérations fusionnées ---
    //
    // Ici elles sont écrites avec les primitives, de la façon la plus
    // lisible : c'est la définition à laquelle les kernels sont comparés.

    pub fn gelu(&self) -> Self {
        self.map(float::gelu)
    }

    /// `grad × gelu'(self)`
    pub fn gelu_backward(&self, grad: &Self) -> Self {
        self.zip_with(grad, |x, g| g * float::gelu_slope(x))
    }

    /// SiLU, élément par élément.
    pub fn silu(&self) -> Self {
        self.map(float::silu)
    }

    /// `grad × silu'(self)`
    pub fn silu_backward(&self, grad: &Self) -> Self {
        self.zip_with(grad, |x, g| g * float::silu_slope(x))
    }

    /// Tourne chaque paire de composantes voisines de `[…, T, D]` de l'angle
    /// dont `cos` et `sin`, de forme `[T, D/2]`, donnent le cosinus et le
    /// sinus. Avec `inverse`, la rotation se fait dans l'autre sens.
    pub fn rotate_pairs(&self, cos: &Self, sin: &Self, inverse: bool) -> Self {
        let rank = self.shape.len();
        assert!(
            rank >= 2 && self.shape[rank - 1].is_multiple_of(2),
            "rotate_pairs attend […, T, D] avec D pair, reçu {:?}",
            self.shape,
        );
        let (length, width) = (self.shape[rank - 2], self.shape[rank - 1]);
        let half = width / 2;
        assert!(
            cos.shape == [length, half] && sin.shape == [length, half],
            "rotate_pairs : angles {:?} et {:?} pour un tenseur {:?}, [{length}, {half}] attendu",
            cos.shape,
            sin.shape,
            self.shape,
        );
        let mut data = Vec::with_capacity(self.data.len());
        for (row, vector) in self.data.chunks(width).enumerate() {
            // Les lignes se suivent position après position, puis on
            // recommence pour la séquence ou la tête suivante.
            let position = row % length;
            for (pair, values) in vector.chunks(2).enumerate() {
                let c = cos.data[position * half + pair];
                let s = sin.data[position * half + pair];
                let s = if inverse { -s } else { s };
                data.push(values[0] * c - values[1] * s);
                data.push(values[0] * s + values[1] * c);
            }
        }
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    // --- Sélection ---

    /// Remplace par `value` les éléments où `mask` est non nul. Le masque
    /// s'étire à la forme du tenseur, pas l'inverse.
    pub fn masked_fill(&self, mask: &Self, value: F) -> Self {
        assert_eq!(
            broadcast_shapes(&self.shape, &mask.shape),
            self.shape,
            "le masque {:?} ne s'étire pas à la forme {:?}",
            mask.shape,
            self.shape,
        );
        self.zip_with(mask, |x, m| if m != F::ZERO { value } else { x })
    }

    /// Sélectionne des tranches le long de la première dimension :
    /// `[V, …]` et `N` indices donnent `[N, …]`. C'est la recherche
    /// d'embedding : la ligne `i` de la table pour le token `i`.
    pub fn gather_rows(&self, indices: &[u32]) -> Self {
        assert!(
            !self.shape.is_empty(),
            "gather_rows demande au moins une dimension",
        );
        let row_len: usize = self.shape[1..].iter().product();
        let mut data = Vec::with_capacity(indices.len() * row_len);
        for &index in indices {
            let index = index as usize;
            assert!(
                index < self.shape[0],
                "indice {index} hors de la première dimension ({})",
                self.shape[0],
            );
            data.extend_from_slice(&self.data[index * row_len..(index + 1) * row_len]);
        }
        let mut shape = self.shape.clone();
        shape[0] = indices.len();
        Self { data, shape }
    }

    /// Met `other` à la suite de `self` le long de la première dimension.
    /// Comme le rangement est ligne par ligne, c'est mettre les données de
    /// l'un à la suite de celles de l'autre.
    pub fn concat_rows(&self, other: &Self) -> Self {
        assert!(
            !self.shape.is_empty() && self.shape[1..] == other.shape[1..],
            "concat_rows : formes incompatibles {:?} et {:?}",
            self.shape,
            other.shape,
        );
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        let mut shape = self.shape.clone();
        shape[0] += other.shape[0];
        Self { data, shape }
    }

    /// Inverse de `gather_rows` : la ligne `r` est ajoutée à la ligne
    /// `indices[r]` d'un tenseur nul de `num_rows` lignes.
    pub fn scatter_add_rows(&self, indices: &[u32], num_rows: usize) -> Self {
        assert!(
            !self.shape.is_empty() && self.shape[0] == indices.len(),
            "scatter_add_rows : {} indices pour la forme {:?}",
            indices.len(),
            self.shape,
        );
        let row_len: usize = self.shape[1..].iter().product();
        let mut data = vec![F::ZERO; num_rows * row_len];
        for (row, &index) in indices.iter().enumerate() {
            let index = index as usize;
            assert!(
                index < num_rows,
                "indice {index} hors de la première dimension ({num_rows})",
            );
            for i in 0..row_len {
                data[index * row_len + i] += self.data[row * row_len + i];
            }
        }
        let mut shape = self.shape.clone();
        shape[0] = num_rows;
        Self { data, shape }
    }

    // --- Perte ---

    /// Vérifie les formes de l'entropie croisée et renvoie `(N, V)`.
    fn cross_entropy_dims(&self, targets: &[u32]) -> (usize, usize) {
        assert!(
            self.shape.len() == 2 && self.shape[0] == targets.len(),
            "cross_entropy : logits {:?} pour {} cibles, [N, V] attendu",
            self.shape,
            targets.len(),
        );
        let vocab = self.shape[1];
        for &target in targets {
            assert!(
                (target as usize) < vocab,
                "cible {target} hors du vocabulaire ({vocab})",
            );
        }
        (self.shape[0], vocab)
    }

    /// Moyenne de `-ln softmax(logits)[n, targets[n]]`, de forme `[]`.
    pub fn cross_entropy(&self, targets: &[u32]) -> Self {
        let (n, vocab) = self.cross_entropy_dims(targets);
        let log_probabilities = self.log_softmax();
        let mut total = F::ZERO;
        for (row, &target) in targets.iter().enumerate() {
            total += -log_probabilities.data[row * vocab + target as usize];
        }
        Self::scalar(total / F::from_f64(n as f64))
    }

    /// `(softmax(logits) - onehot(targets)) / N`.
    pub fn cross_entropy_grad(&self, targets: &[u32]) -> Self {
        let (n, vocab) = self.cross_entropy_dims(targets);
        let mut grad = self.softmax();
        let scale = F::ONE / F::from_f64(n as f64);
        for (row, &target) in targets.iter().enumerate() {
            let row = &mut grad.data[row * vocab..(row + 1) * vocab];
            row[target as usize] = row[target as usize] - F::ONE;
            for x in row.iter_mut() {
                *x = *x * scale;
            }
        }
        grad
    }

    // --- Mises à jour des paramètres ---

    /// `self -= learning_rate × grad`
    /// Écrit la ligne `r` de `rows` à la ligne `indices[r]` de ce tenseur.
    /// Les indices sont des entiers rangés dans un tenseur `[N]`.
    pub fn write_rows(&mut self, indices: &Self, rows: &Self) {
        assert!(
            !self.shape.is_empty() && !rows.shape.is_empty(),
            "write_rows demande au moins une dimension",
        );
        assert_eq!(
            self.shape[1..],
            rows.shape[1..],
            "write_rows : lignes {:?} pour une table {:?}",
            rows.shape,
            self.shape,
        );
        assert_eq!(
            indices.data.len(),
            rows.shape[0],
            "write_rows : {} indices pour {} lignes",
            indices.data.len(),
            rows.shape[0],
        );
        let row_len: usize = self.shape[1..].iter().product();
        for (row, &index) in indices.data.iter().enumerate() {
            let index = index.to_f64() as usize;
            assert!(
                index < self.shape[0],
                "indice {index} hors de la première dimension ({})",
                self.shape[0],
            );
            self.data[index * row_len..(index + 1) * row_len]
                .copy_from_slice(&rows.data[row * row_len..(row + 1) * row_len]);
        }
    }

    pub fn sgd_step(&mut self, grad: &Self, learning_rate: f64) {
        assert_eq!(self.shape, grad.shape, "gradient de forme inattendue");
        let learning_rate = F::from_f64(learning_rate);
        for (p, &g) in self.data.iter_mut().zip(&grad.data) {
            *p = *p - learning_rate * g;
        }
    }

    /// Un pas d'AdamW sur ce paramètre et ses deux moments.
    pub fn adamw_step(
        &mut self,
        grad: &Self,
        first_moment: &mut Self,
        second_moment: &mut Self,
        step: &AdamWStep,
    ) {
        assert_eq!(self.shape, grad.shape, "gradient de forme inattendue");
        assert_eq!(self.shape, first_moment.shape, "moment de forme inattendue");
        assert_eq!(
            self.shape, second_moment.shape,
            "moment de forme inattendue"
        );
        let learning_rate = F::from_f64(step.learning_rate);
        let beta1 = F::from_f64(step.beta1);
        let beta2 = F::from_f64(step.beta2);
        let epsilon = F::from_f64(step.epsilon);
        let weight_decay = F::from_f64(step.weight_decay);
        let correction1 = F::from_f64(step.bias_correction1);
        let correction2 = F::from_f64(step.bias_correction2);
        let grad_scale = F::from_f64(step.grad_scale);

        for i in 0..self.data.len() {
            let g = grad.data[i] * grad_scale;
            let m = beta1 * first_moment.data[i] + (F::ONE - beta1) * g;
            let v = beta2 * second_moment.data[i] + (F::ONE - beta2) * g * g;
            first_moment.data[i] = m;
            second_moment.data[i] = v;

            let update = (m / correction1) / ((v / correction2).sqrt() + epsilon);
            let p = self.data[i];
            self.data[i] = p - learning_rate * (update + weight_decay * p);
        }
    }
}

/// Backend CPU : il ne porte aucun état, toutes les opérations sont des
/// méthodes de [`CpuTensor`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Cpu<F>(PhantomData<F>);

impl<F> Cpu<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: Float> Primitives for Cpu<F> {
    type Scalar = F;
    type Tensor = CpuTensor<F>;

    fn from_vec(&self, data: Vec<F>, shape: &[usize]) -> CpuTensor<F> {
        CpuTensor::from_vec(data, shape)
    }

    fn to_vec(&self, tensor: &CpuTensor<F>) -> Vec<F> {
        tensor.data.clone()
    }

    fn shape<'a>(&self, tensor: &'a CpuTensor<F>) -> &'a [usize] {
        &tensor.shape
    }

    fn add(&self, a: &CpuTensor<F>, b: &CpuTensor<F>) -> CpuTensor<F> {
        a.add(b)
    }

    fn sub(&self, a: &CpuTensor<F>, b: &CpuTensor<F>) -> CpuTensor<F> {
        a.sub(b)
    }

    fn mul(&self, a: &CpuTensor<F>, b: &CpuTensor<F>) -> CpuTensor<F> {
        a.mul(b)
    }

    fn div(&self, a: &CpuTensor<F>, b: &CpuTensor<F>) -> CpuTensor<F> {
        a.div(b)
    }

    fn neg(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.neg()
    }

    fn exp(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.exp()
    }

    fn ln(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.ln()
    }

    fn sqrt(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.sqrt()
    }

    fn tanh(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.tanh()
    }

    fn sum(&self, a: &CpuTensor<F>, axis: usize, keepdim: bool) -> CpuTensor<F> {
        a.sum(axis, keepdim)
    }

    fn sum_all(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.sum_all()
    }

    fn reshape(&self, a: &CpuTensor<F>, shape: &[usize]) -> CpuTensor<F> {
        a.reshape(shape)
    }

    fn transpose(&self, a: &CpuTensor<F>, axis_a: usize, axis_b: usize) -> CpuTensor<F> {
        a.transpose(axis_a, axis_b)
    }

    fn expand(&self, a: &CpuTensor<F>, shape: &[usize]) -> CpuTensor<F> {
        a.expand(shape)
    }

    fn matmul(&self, a: &CpuTensor<F>, b: &CpuTensor<F>) -> CpuTensor<F> {
        a.matmul(b)
    }

    fn softmax(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.softmax()
    }

    fn log_softmax(&self, a: &CpuTensor<F>) -> CpuTensor<F> {
        a.log_softmax()
    }

    fn masked_fill(&self, a: &CpuTensor<F>, mask: &CpuTensor<F>, value: F) -> CpuTensor<F> {
        a.masked_fill(mask, value)
    }

    fn gather_rows(&self, a: &CpuTensor<F>, indices: &[u32]) -> CpuTensor<F> {
        a.gather_rows(indices)
    }

    fn scatter_add_rows(&self, a: &CpuTensor<F>, indices: &[u32], num_rows: usize) -> CpuTensor<F> {
        a.scatter_add_rows(indices, num_rows)
    }

    fn concat_rows(&self, a: &CpuTensor<F>, b: &CpuTensor<F>) -> CpuTensor<F> {
        a.concat_rows(b)
    }

    fn write_rows(&self, table: &mut CpuTensor<F>, indices: &CpuTensor<F>, rows: &CpuTensor<F>) {
        table.write_rows(indices, rows);
    }

    fn sgd_step(&self, param: &mut CpuTensor<F>, grad: &CpuTensor<F>, learning_rate: f64) {
        param.sgd_step(grad, learning_rate);
    }

    fn adamw_step(
        &self,
        param: &mut CpuTensor<F>,
        grad: &CpuTensor<F>,
        first_moment: &mut CpuTensor<F>,
        second_moment: &mut CpuTensor<F>,
        step: &AdamWStep,
    ) {
        param.adamw_step(grad, first_moment, second_moment, step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type T = CpuTensor<f64>;

    fn tensor(data: &[f64], shape: &[usize]) -> T {
        T::from_vec(data.to_vec(), shape)
    }

    fn assert_close(actual: &T, expected: &T) {
        assert_eq!(actual.shape(), expected.shape());
        for (a, e) in actual.data().iter().zip(expected.data()) {
            assert!(
                (a - e).abs() < 1e-12,
                "{a} != {e}\n{actual:?}\n{expected:?}"
            );
        }
    }

    // --- Constructeurs ---

    #[test]
    fn constructors() {
        assert_eq!(T::zeros(&[2, 2]).data(), &[0.0; 4]);
        assert_eq!(T::ones(&[3]).data(), &[1.0; 3]);
        assert_eq!(T::arange(4).data(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(T::scalar(7.0).shape(), &[] as &[usize]);
        assert_eq!(T::scalar(7.0).item(), 7.0);
    }

    #[test]
    #[should_panic(expected = "5 éléments fournis pour la forme [2, 3]")]
    fn from_vec_checks_length() {
        tensor(&[0.0; 5], &[2, 3]);
    }

    #[test]
    fn randn_is_standard_normal() {
        let t = T::randn(&[100_000], &mut StdRng::seed_from_u64(0));
        let n = t.data().len() as f64;
        let mean = t.data().iter().sum::<f64>() / n;
        let variance = t.data().iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.02, "moyenne {mean}");
        assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
    }

    // --- Broadcasting ---

    #[test]
    fn add_same_shape() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        assert_close(&a.add(&b), &tensor(&[11.0, 22.0, 33.0, 44.0], &[2, 2]));
    }

    #[test]
    fn add_broadcasts_a_row() {
        // Le vecteur [3] est ajouté à chaque ligne de la matrice [2, 3].
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = tensor(&[10.0, 20.0, 30.0], &[3]);
        let expected = tensor(&[11.0, 22.0, 33.0, 14.0, 25.0, 36.0], &[2, 3]);
        assert_close(&a.add(&b), &expected);
        assert_close(&b.add(&a), &expected);
    }

    #[test]
    fn add_broadcasts_both_sides() {
        // Colonne [2, 1] + ligne [1, 3] = table d'addition [2, 3].
        let column = tensor(&[10.0, 20.0], &[2, 1]);
        let row = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let expected = tensor(&[11.0, 12.0, 13.0, 21.0, 22.0, 23.0], &[2, 3]);
        assert_close(&column.add(&row), &expected);
    }

    #[test]
    fn scalar_broadcasts_everywhere() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_close(
            &a.mul(&T::scalar(10.0)),
            &tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]),
        );
    }

    #[test]
    fn elementwise_operations() {
        let a = tensor(&[1.0, 4.0], &[2]);
        let b = tensor(&[2.0, 8.0], &[2]);
        assert_close(&a.sub(&b), &tensor(&[-1.0, -4.0], &[2]));
        assert_close(&a.mul(&b), &tensor(&[2.0, 32.0], &[2]));
        assert_close(&a.div(&b), &tensor(&[0.5, 0.5], &[2]));
        assert_close(&a.neg(), &tensor(&[-1.0, -4.0], &[2]));
        assert_close(&a.sqrt(), &tensor(&[1.0, 2.0], &[2]));
        assert_close(&a.ln().exp(), &a);
        assert_close(&T::zeros(&[2]).tanh(), &T::zeros(&[2]));
        assert_close(
            &T::scalar(1.0).tanh(),
            &T::scalar((1f64.exp() - (-1f64).exp()) / (1f64.exp() + (-1f64).exp())),
        );
    }

    // --- Réductions ---

    #[test]
    fn sum_along_each_axis() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_close(&a.sum(0, false), &tensor(&[5.0, 7.0, 9.0], &[3]));
        assert_close(&a.sum(1, false), &tensor(&[6.0, 15.0], &[2]));
        assert_close(&a.sum(0, true), &tensor(&[5.0, 7.0, 9.0], &[1, 3]));
        assert_close(&a.sum(1, true), &tensor(&[6.0, 15.0], &[2, 1]));
    }

    #[test]
    fn sum_middle_axis() {
        let a = T::arange(24).reshape(&[2, 3, 4]);
        // Première tranche : lignes [0..4], [4..8], [8..12], sommées terme à terme.
        let expected = tensor(&[12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0], &[2, 4]);
        assert_close(&a.sum(1, false), &expected);
    }

    #[test]
    fn sum_to_scalar() {
        assert_eq!(T::arange(5).sum(0, false).item(), 10.0);
    }

    #[test]
    fn sum_all_gives_a_scalar() {
        let total = T::arange(24).reshape(&[2, 3, 4]).sum_all();
        assert_eq!(total.shape(), &[] as &[usize]);
        assert_eq!(total.item(), 276.0);
    }

    #[test]
    fn mean_along_axis() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_close(&a.mean(1, false), &tensor(&[2.0, 5.0], &[2]));
        assert_close(&a.mean(0, true), &tensor(&[2.5, 3.5, 4.5], &[1, 3]));
    }

    #[test]
    fn keepdim_allows_broadcasting_back() {
        // Centrer chaque ligne : x - moyenne(x), l'opération de LayerNorm.
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let centered = a.sub(&a.mean(1, true));
        assert_close(
            &centered,
            &tensor(&[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], &[2, 3]),
        );
    }

    // --- Changements de forme ---

    #[test]
    fn reshape_keeps_data_order() {
        let a = T::arange(6).reshape(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.reshape(&[3, 2]).data(), a.data());
    }

    #[test]
    fn transpose_matrix() {
        // [[0, 1, 2],      [[0, 3],
        //  [3, 4, 5]]  →    [1, 4],
        //                   [2, 5]]
        let a = T::arange(6).reshape(&[2, 3]);
        assert_close(
            &a.transpose(0, 1),
            &tensor(&[0.0, 3.0, 1.0, 4.0, 2.0, 5.0], &[3, 2]),
        );
    }

    #[test]
    fn transpose_differs_from_reshape() {
        let a = T::arange(6).reshape(&[2, 3]);
        assert_ne!(a.transpose(0, 1), a.reshape(&[3, 2]));
    }

    #[test]
    fn transpose_last_two_axes_of_a_batch() {
        let a = T::arange(12).reshape(&[2, 2, 3]);
        let t = a.transpose(1, 2);
        assert_eq!(t.shape(), &[2, 3, 2]);
        // Chaque matrice du lot est transposée indépendamment.
        assert_eq!(
            t.data(),
            &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0, 7.0, 10.0, 8.0, 11.0]
        );
    }

    #[test]
    fn transpose_outer_axes() {
        let a = T::arange(24).reshape(&[2, 3, 4]);
        let t = a.transpose(0, 2);
        assert_eq!(t.shape(), &[4, 3, 2]);
        // L'élément [i, j, k] de la sortie est l'élément [k, j, i] de l'entrée.
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    assert_eq!(t.data()[(i * 3 + j) * 2 + k], a.data()[(k * 3 + j) * 4 + i]);
                }
            }
        }
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = T::arange(24).reshape(&[2, 3, 4]);
        assert_eq!(a.transpose(0, 2).transpose(0, 2), a);
    }

    // --- Matmul ---

    #[test]
    fn matmul_two_matrices() {
        // [[1, 2, 3],   [[ 7,  8],     [[ 58,  64],
        //  [4, 5, 6]] ×  [ 9, 10],  =   [139, 154]]
        //                [11, 12]]
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        assert_close(&a.matmul(&b), &tensor(&[58.0, 64.0, 139.0, 154.0], &[2, 2]));
    }

    #[test]
    fn matmul_identity() {
        let a = T::arange(9).reshape(&[3, 3]);
        let identity = tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        assert_close(&a.matmul(&identity), &a);
        assert_close(&identity.matmul(&a), &a);
    }

    #[test]
    fn matmul_batch_times_matrix() {
        // [B, T, C] × [C, D] : la même matrice multiplie chaque élément du lot.
        let a = T::arange(12).reshape(&[2, 2, 3]);
        let b = tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let result = a.matmul(&b);
        assert_eq!(result.shape(), &[2, 2, 2]);
        for batch in 0..2 {
            let slice = tensor(&a.data()[batch * 6..(batch + 1) * 6], &[2, 3]);
            let expected = slice.matmul(&b);
            assert_eq!(&result.data()[batch * 4..(batch + 1) * 4], expected.data());
        }
    }

    #[test]
    fn matmul_batch_times_batch() {
        // [B, H, T, C] × [B, H, C, T] → [B, H, T, T], comme q · kᵀ en attention.
        let mut rng = StdRng::seed_from_u64(1);
        let a = T::randn(&[2, 3, 4, 5], &mut rng);
        let b = T::randn(&[2, 3, 5, 4], &mut rng);
        let result = a.matmul(&b);
        assert_eq!(result.shape(), &[2, 3, 4, 4]);
        for batch in 0..6 {
            let slice_a = tensor(&a.data()[batch * 20..(batch + 1) * 20], &[4, 5]);
            let slice_b = tensor(&b.data()[batch * 20..(batch + 1) * 20], &[5, 4]);
            let expected = slice_a.matmul(&slice_b);
            assert_eq!(
                &result.data()[batch * 16..(batch + 1) * 16],
                expected.data()
            );
        }
    }

    #[test]
    #[should_panic(expected = "matmul : formes incompatibles [2, 3] et [2, 3]")]
    fn matmul_rejects_mismatch() {
        T::zeros(&[2, 3]).matmul(&T::zeros(&[2, 3]));
    }

    // --- Softmax ---

    #[test]
    fn softmax_known_values() {
        // exp(0) = 1 et exp(ln 2) = 2 : les probabilités sont 1/3 et 2/3.
        let a = tensor(&[0.0, 2f64.ln()], &[2]);
        assert_close(&a.softmax(), &tensor(&[1.0 / 3.0, 2.0 / 3.0], &[2]));
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let a = T::randn(&[4, 7], &mut StdRng::seed_from_u64(2));
        assert_close(&a.softmax().sum(1, false), &T::ones(&[4]));
    }

    #[test]
    fn softmax_is_stable_for_large_values() {
        // Sans la soustraction du maximum, exp(1000) déborderait en infini.
        let a = tensor(&[1000.0, 1000.0], &[2]);
        assert_close(&a.softmax(), &tensor(&[0.5, 0.5], &[2]));
    }

    #[test]
    fn softmax_gives_zero_to_minus_infinity() {
        // C'est ainsi que le masque causal interdit de regarder le futur.
        let a = tensor(&[0.0, f64::NEG_INFINITY, 0.0], &[3]);
        assert_close(&a.softmax(), &tensor(&[0.5, 0.0, 0.5], &[3]));
    }

    #[test]
    fn log_softmax_matches_ln_of_softmax() {
        let a = T::randn(&[3, 5], &mut StdRng::seed_from_u64(3));
        assert_close(&a.log_softmax(), &a.softmax().ln());
    }

    #[test]
    fn log_softmax_stays_finite_where_softmax_underflows() {
        let a = tensor(&[0.0, -1000.0], &[2]);
        assert_eq!(a.softmax().data()[1], 0.0);
        assert_close(&a.log_softmax(), &tensor(&[0.0, -1000.0], &[2]));
    }

    // --- Sélection ---

    #[test]
    fn masked_fill_with_broadcast_mask() {
        // Masque causal [T, T] appliqué à des scores [B, T, T].
        let scores = T::ones(&[2, 2, 2]);
        let mask = tensor(&[0.0, 1.0, 0.0, 0.0], &[2, 2]);
        let filled = scores.masked_fill(&mask, f64::NEG_INFINITY);
        let inf = f64::NEG_INFINITY;
        assert_eq!(filled.data(), &[1.0, inf, 1.0, 1.0, 1.0, inf, 1.0, 1.0]);
    }

    #[test]
    fn concat_rows_appends_along_the_first_axis() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = tensor(&[5.0, 6.0], &[1, 2]);
        assert_close(
            &a.concat_rows(&b),
            &tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]),
        );
    }

    #[test]
    #[should_panic(expected = "concat_rows : formes incompatibles")]
    fn concat_rows_rejects_different_rows() {
        T::zeros(&[2, 2]).concat_rows(&T::zeros(&[1, 3]));
    }

    #[test]
    #[should_panic(expected = "ne s'étire pas à la forme")]
    fn masked_fill_rejects_larger_mask() {
        T::ones(&[2]).masked_fill(&T::zeros(&[3, 2]), 0.0);
    }

    #[test]
    fn gather_rows_looks_up_embeddings() {
        // Table de 3 tokens, embeddings de dimension 2.
        let table = tensor(&[0.0, 0.1, 1.0, 1.1, 2.0, 2.1], &[3, 2]);
        let picked = table.gather_rows(&[2, 0, 2]);
        assert_close(&picked, &tensor(&[2.0, 2.1, 0.0, 0.1, 2.0, 2.1], &[3, 2]));
    }

    #[test]
    #[should_panic(expected = "indice 3 hors de la première dimension (3)")]
    fn gather_rows_checks_bounds() {
        T::zeros(&[3, 2]).gather_rows(&[3]);
    }

    #[test]
    fn expand_repeats_elements() {
        let row = tensor(&[1.0, 2.0, 3.0], &[3]);
        assert_close(
            &row.expand(&[2, 3]),
            &tensor(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]),
        );
        let column = tensor(&[1.0, 2.0], &[2, 1]);
        assert_close(
            &column.expand(&[2, 3]),
            &tensor(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]),
        );
        assert_close(&T::scalar(7.0).expand(&[2, 2]), &T::full(&[2, 2], 7.0));
    }

    #[test]
    fn scatter_add_rows_sums_repeated_indices() {
        // Les lignes 0 et 2 visent toutes deux la ligne 2 de la sortie.
        let rows = tensor(&[1.0, 2.0, 10.0, 20.0, 100.0, 200.0], &[3, 2]);
        let scattered = rows.scatter_add_rows(&[2, 0, 2], 4);
        assert_close(
            &scattered,
            &tensor(&[10.0, 20.0, 0.0, 0.0, 101.0, 202.0, 0.0, 0.0], &[4, 2]),
        );
    }

    #[test]
    fn scatter_add_is_the_transpose_of_gather() {
        // <gather(table), y> == <table, scatter_add(y)> pour tous table et y.
        let mut rng = StdRng::seed_from_u64(4);
        let table = T::randn(&[5, 3], &mut rng);
        let y = T::randn(&[4, 3], &mut rng);
        let indices = [1, 4, 1, 0];
        let left = table.gather_rows(&indices).mul(&y).sum_all();
        let right = table.mul(&y.scatter_add_rows(&indices, 5)).sum_all();
        assert_close(&left, &right);
    }

    // --- Perte ---

    #[test]
    fn cross_entropy_of_uniform_logits_is_ln_vocab() {
        // Des logits tous égaux donnent une probabilité 1/V à chaque token.
        let logits = T::zeros(&[3, 257]);
        let loss = logits.cross_entropy(&[0, 100, 256]);
        assert!((loss.item() - 257f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn cross_entropy_known_value() {
        // Probabilités 1/3 et 2/3 ; la cible est la seconde.
        let logits = tensor(&[0.0, 2f64.ln()], &[1, 2]);
        assert!((logits.cross_entropy(&[1]).item() - 1.5f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn cross_entropy_grad_known_value() {
        // (softmax - onehot) / N, avec N = 2 lignes identiques.
        let logits = tensor(&[0.0, 2f64.ln(), 0.0, 2f64.ln()], &[2, 2]);
        let third = 1.0 / 3.0;
        assert_close(
            &logits.cross_entropy_grad(&[1, 0]),
            &tensor(
                &[
                    third / 2.0,
                    -third / 2.0,
                    -2.0 * third / 2.0,
                    2.0 * third / 2.0,
                ],
                &[2, 2],
            ),
        );
    }

    #[test]
    fn cross_entropy_grad_rows_sum_to_zero() {
        let logits = T::randn(&[4, 6], &mut StdRng::seed_from_u64(5));
        let grad = logits.cross_entropy_grad(&[0, 5, 2, 2]);
        assert_close(&grad.sum(1, false), &T::zeros(&[4]));
    }

    // --- Opérations fusionnées ---

    #[test]
    fn silu_known_values() {
        // silu(0) = 0 ; silu(1) = 1 / (1 + e⁻¹) ; les extrêmes ne débordent pas.
        let x = tensor(&[0.0, 1.0, -1.0, 100.0, -100.0], &[5]);
        let expected = [
            0.0,
            0.731_058_578_630_005,
            -0.268_941_421_369_995,
            100.0,
            0.0,
        ];
        for (a, e) in x.silu().data().iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{a} != {e}");
        }
    }

    #[test]
    fn silu_backward_is_the_slope_times_the_gradient() {
        let x = tensor(&[-2.0, -0.5, 0.0, 0.7, 3.0], &[5]);
        let grad = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let backward = x.silu_backward(&grad);
        let h = 1e-6;
        for i in 0..5 {
            let slope = (float::silu(x.data()[i] + h) - float::silu(x.data()[i] - h)) / (2.0 * h);
            let expected = grad.data()[i] * slope;
            assert!((backward.data()[i] - expected).abs() < 1e-8);
        }
    }

    /// Les tables d'angles de RoPE pour `length` positions à partir de
    /// `start` : la paire `i` tourne de `position × 10000^(-2i/D)`.
    fn angles(start: usize, length: usize, width: usize) -> (T, T) {
        let half = width / 2;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for position in start..start + length {
            for pair in 0..half {
                let angle = position as f64 * 10000f64.powf(-(2.0 * pair as f64) / width as f64);
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (
            T::from_vec(cos, &[length, half]),
            T::from_vec(sin, &[length, half]),
        )
    }

    #[test]
    fn rotate_pairs_turns_each_pair_by_its_angle() {
        // Un quart de tour : (1, 0) devient (0, 1), et (0, 1) devient (-1, 0).
        let x = tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 4]);
        let cos = tensor(&[0.0, 0.0], &[1, 2]);
        let sin = tensor(&[1.0, 1.0], &[1, 2]);
        assert_close(
            &x.rotate_pairs(&cos, &sin, false),
            &tensor(&[0.0, 1.0, -1.0, 0.0], &[1, 4]),
        );
    }

    #[test]
    fn rotate_pairs_keeps_lengths_and_can_be_undone() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let x = T::randn(&[2, 3, 5, 8], &mut rng);
        let (cos, sin) = angles(0, 5, 8);
        let rotated = x.rotate_pairs(&cos, &sin, false);
        // La position 0 ne tourne pas.
        assert_eq!(&rotated.data()[..8], &x.data()[..8]);
        assert_ne!(&rotated.data()[8..16], &x.data()[8..16]);
        // Une rotation garde la longueur de chaque vecteur…
        assert_close(
            &rotated.mul(&rotated).sum(3, false),
            &x.mul(&x).sum(3, false),
        );
        // … et la rotation inverse ramène au point de départ.
        assert_close(&rotated.rotate_pairs(&cos, &sin, true), &x);
    }

    #[test]
    fn rotated_dot_products_depend_only_on_the_distance() {
        // C'est la propriété qui fait de cette rotation un encodage de
        // position relatif : le produit scalaire d'une requête à la position
        // t et d'une clé à la position s ne dépend que de t - s.
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let query = T::randn(&[1, 8], &mut rng);
        let key = T::randn(&[1, 8], &mut rng);
        let dot = |query_position: usize, key_position: usize| {
            let (cos, sin) = angles(query_position, 1, 8);
            let q = query.rotate_pairs(&cos, &sin, false);
            let (cos, sin) = angles(key_position, 1, 8);
            let k = key.rotate_pairs(&cos, &sin, false);
            q.mul(&k).sum_all().item()
        };
        assert!((dot(7, 3) - dot(107, 103)).abs() < 1e-9);
        assert!((dot(7, 3) - dot(4, 0)).abs() < 1e-9);
        // Une autre distance donne un autre produit.
        assert!((dot(7, 3) - dot(7, 4)).abs() > 1e-3);
    }

    #[test]
    #[should_panic(expected = "rotate_pairs : angles")]
    fn rotate_pairs_checks_the_angle_tables() {
        let (cos, sin) = angles(0, 4, 8);
        T::zeros(&[5, 8]).rotate_pairs(&cos, &sin, false);
    }

    #[test]
    fn gelu_known_values() {
        let x = tensor(&[0.0, 1.0, -1.0, 10.0, -10.0], &[5]);
        let expected = [0.0, 0.841192, -0.158808, 10.0, 0.0];
        for (a, e) in x.gelu().data().iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{a} != {e}");
        }
    }

    #[test]
    fn gelu_backward_is_the_slope_times_the_gradient() {
        // Pente estimée par différences finies.
        let x = tensor(&[-2.0, -0.5, 0.0, 0.7, 3.0], &[5]);
        let grad = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let eps = T::scalar(1e-6);
        let slope = x
            .add(&eps)
            .gelu()
            .sub(&x.sub(&eps).gelu())
            .div(&T::scalar(2e-6));
        let expected = slope.mul(&grad);
        for (a, e) in x.gelu_backward(&grad).data().iter().zip(expected.data()) {
            assert!((a - e).abs() < 1e-8, "{a} != {e}");
        }
    }

    // --- Généricité ---

    #[test]
    fn works_in_f32() {
        let a = CpuTensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let product = a.matmul(&a);
        assert_eq!(product.data(), &[7.0, 10.0, 15.0, 22.0]);
    }
}
