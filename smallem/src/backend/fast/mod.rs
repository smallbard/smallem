//! Un backend pour le processeur, qui en exploite les cœurs et les
//! instructions vectorielles.
//!
//! [`Cpu`](super::cpu::Cpu) reste la référence : un seul fil, la triple
//! boucle, le calcul écrit pour qu'on le croie. [`Fast`] calcule la même
//! chose, en `f32`, aussi vite que la machine le permet :
//!
//! - les opérations se partagent entre un groupe de fils permanent
//!   ([`pool`]) ;
//! - le produit matriciel est découpé en blocs qui tiennent dans les caches,
//!   autour d'un micro-kernel AVX-512 ou AVX2 ([`gemm`], [`kernels`]) ;
//! - les calculs d'une ligne, exponentielle comprise, sont vectorisés
//!   ([`math`]) ;
//! - chaque opération a une écriture directe, souvent en une passe par
//!   ligne, au lieu de l'enchaînement de primitives de sa définition
//!   (`ops.rs`), et l'attention se calcule en tuiles (`attention.rs`) ;
//! - les poids quantifiés se lisent en octets (`quantized.rs`, `vnni.rs`) ;
//! - les tampons abandonnés retournent dans une réserve ([`Buffer`]).
//!
//! Les instructions sont détectées à l'exécution : la même compilation tourne
//! sur un processeur sans AVX-512, avec les kernels AVX2, ou sans AVX2.
//!
//! Le découpage du travail entre les fils est fixé par la forme des tenseurs,
//! jamais par le nombre de fils : avec un, deux ou seize fils, un calcul
//! donne les mêmes bits.

/// Compile une fonction écrite une fois en trois versions : AVX-512, AVX2,
/// et sans instructions vectorielles explicites. `$body` est une fonction
/// `#[inline(always)]` générique sur `FMA`, qui dit si l'on peut fusionner
/// multiplication et addition.
macro_rules! multiversion {
    (
        $(#[$attr:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty = $body:ident;
    ) => {
        $(#[$attr])*
        pub fn $name(isa: $crate::backend::fast::Isa, $($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
            unsafe fn wide($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn narrow($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            match isa {
                #[cfg(target_arch = "x86_64")]
                $crate::backend::fast::Isa::Avx512 => unsafe { wide($($arg),*) },
                #[cfg(target_arch = "x86_64")]
                $crate::backend::fast::Isa::Avx2 => unsafe { narrow($($arg),*) },
                _ => $body::<false>($($arg),*),
            }
        }
    };
}

mod attention;
mod buffer;
pub mod gemm;
pub mod kernels;
pub mod math;
mod ops;
pub mod pool;
mod quantized;
#[cfg(test)]
mod tests;
mod vnni;

use std::sync::Arc;

use super::shape::{broadcast_shapes, broadcast_strides, source_index, strides};
use super::{AdamWStep, Primitives};
pub use buffer::Buffer;
use gemm::Matrix;
use pool::{Pool, SharedMut};

/// `a × b + c`, fusionné quand le processeur sait le faire.
#[inline(always)]
pub(crate) fn fma<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// Les instructions vectorielles dont se servent les kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AVX-512 (fondation, mots et octets, longueurs variables) et FMA.
    Avx512,
    /// AVX2 et FMA.
    Avx2,
    /// Rien de plus que ce que tout processeur `x86_64` garantit.
    Scalar,
}

impl Isa {
    /// Les meilleures instructions du processeur.
    pub fn detect() -> Self {
        [Isa::Avx512, Isa::Avx2]
            .into_iter()
            .find(|isa| isa.available())
            .unwrap_or(Isa::Scalar)
    }

    /// Le processeur a-t-il ces instructions ?
    pub fn available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                Isa::Avx2.available()
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512dq")
                    && is_x86_feature_detected!("avx512vl")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

impl std::str::FromStr for Isa {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "avx512" => Ok(Isa::Avx512),
            "avx2" => Ok(Isa::Avx2),
            "scalar" => Ok(Isa::Scalar),
            other => Err(format!(
                "instructions inconnues : {other} (avx512, avx2, scalar)"
            )),
        }
    }
}

/// Le nombre de fils par défaut : un par cœur physique. Le SMT partage les
/// unités de calcul d'un cœur entre deux fils, et ne fait rien gagner au
/// produit matriciel (étape C1 de `PLAN-CPU.md`). `SMALLEM_THREADS` le
/// remplace.
pub fn default_threads() -> usize {
    if let Some(threads) = std::env::var("SMALLEM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        return threads;
    }
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    physical_cores().unwrap_or(logical).clamp(1, logical)
}

/// Le nombre de cœurs physiques, d'après `/proc/cpuinfo`.
fn physical_cores() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut cores = std::collections::BTreeSet::new();
    let mut physical = None;
    for line in info.lines() {
        let mut parts = line.splitn(2, ':').map(str::trim);
        match (parts.next(), parts.next()) {
            (Some("physical id"), Some(id)) => physical = Some(id.to_string()),
            (Some("core id"), Some(id)) => {
                cores.insert((physical.clone(), id.to_string()));
            }
            _ => {}
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// Un tenseur du processeur : des données partagées, et une forme.
///
/// Les données sont derrière un compteur de références. Cloner un tenseur ou
/// en changer la forme ne recopie rien ; une mise à jour sur place ne recopie
/// que si les données sont aussi référencées ailleurs.
#[derive(Clone, Debug)]
pub struct FastTensor {
    data: Arc<Buffer>,
    shape: Vec<usize>,
}

impl FastTensor {
    fn new(data: impl Into<Buffer>, shape: &[usize]) -> Self {
        let data = data.into();
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "{} éléments fournis pour la forme {shape:?}",
            data.len(),
        );
        Self {
            data: Arc::new(data),
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    /// Les données, modifiables : recopiées seulement si elles sont
    /// partagées.
    fn data_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }
}

/// Le backend.
#[derive(Clone)]
pub struct Fast {
    pool: Arc<Pool>,
    isa: Isa,
    /// Les matrices quantifiées, rangées pour le processeur.
    natives: Arc<quantized::Natives<quantized::Native>>,
    /// Les mêmes, rangées pour `vpdpbusd`.
    packed: Arc<quantized::Natives<vnni::Packed>>,
    /// Quantifier aussi les activations en 8 bits devant une matrice
    /// quantifiée (voir `vnni.rs`).
    int8_activations: bool,
    /// Les grands produits en `bf16`, accumulés en `f32`.
    mixed: bool,
}

impl Default for Fast {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Fast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Fast({} fils, {:?})", self.threads(), self.isa)
    }
}

/// Une tâche fait au moins ce nombre d'éléments : en dessous, partager le
/// travail coûte plus que le travail lui-même (mesuré à l'étape C1).
const CHUNK: usize = 1 << 14;

impl Fast {
    /// Un fil par cœur physique, les meilleures instructions du processeur.
    pub fn new() -> Self {
        Self::with_threads(default_threads())
    }

    pub fn with_threads(threads: usize) -> Self {
        Self {
            pool: Arc::new(Pool::new(threads)),
            isa: Isa::detect(),
            natives: Arc::default(),
            packed: Arc::default(),
            int8_activations: false,
            mixed: false,
        }
    }

    /// Les mêmes fils, avec d'autres instructions : pour comparer les
    /// kernels entre eux.
    pub fn with_isa(&self, isa: Isa) -> Self {
        assert!(
            isa.available(),
            "le processeur n'a pas les instructions {isa:?}"
        );
        Self {
            pool: Arc::clone(&self.pool),
            isa,
            natives: Arc::clone(&self.natives),
            packed: Arc::clone(&self.packed),
            int8_activations: self.int8_activations,
            mixed: self.mixed,
        }
    }

    /// La précision mixte : les opérandes des grands produits arrondis en
    /// `bf16`, leurs produits accumulés en `f32` avec `vdpbf16ps` (AVX-512
    /// BF16), deux fois plus vite qu'en `f32`. Tout le reste, poids compris,
    /// reste en `f32`, comme sur la carte. Sans ces instructions, rien ne
    /// change.
    pub fn with_mixed_precision(&self, on: bool) -> Self {
        Self {
            mixed: on,
            ..self.clone()
        }
    }

    /// `C (+)= A · B`, dans la précision du backend.
    pub(crate) fn gemm_into(&self, a: Matrix, b: Matrix, c: &mut [f32], accumulate: bool) {
        if self.mixed {
            gemm::gemm_mixed(&self.pool, self.isa, a, b, c, accumulate);
        } else {
            gemm::gemm(&self.pool, self.isa, a, b, c, accumulate);
        }
    }

    /// Devant une matrice quantifiée, quantifier aussi les activations en
    /// 8 bits, pour multiplier des octets par des octets avec `vpdpbusd`
    /// (AVX-512 VNNI). Plus rapide quand les poids sont dans les caches,
    /// mais le produit n'est plus celui des poids déquantifiés : les
    /// activations perdent leur précision (étape C5 de `PLAN-CPU.md`).
    pub fn with_int8_activations(&self, on: bool) -> Self {
        Self {
            int8_activations: on,
            ..self.clone()
        }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Remplit un nouveau tampon de `len` éléments, par tâches d'au moins
    /// `chunk` éléments : `fill(début, tranche)`.
    fn generate(
        &self,
        len: usize,
        chunk: usize,
        fill: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Buffer {
        // Chaque élément est écrit par exactement une tâche avant d'être lu.
        let mut out = Buffer::uninit(len);
        self.for_parts(&mut out, chunk, fill);
        out
    }

    /// Un tampon de zéros.
    fn zeros_buffer(&self, len: usize) -> Buffer {
        self.generate(len, 4 * CHUNK, |_, part| part.fill(0.0))
    }

    /// Applique `work(début, tranche)` à des tranches d'au moins `chunk`
    /// éléments de `data`, en parallèle.
    fn for_parts(&self, data: &mut [f32], chunk: usize, work: impl Fn(usize, &mut [f32]) + Sync) {
        let len = data.len();
        let chunk = chunk.max(1);
        let shared = SharedMut::new(data);
        self.pool.run(len.div_ceil(chunk), |task| {
            let start = task * chunk;
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
        let mut data = Buffer::uninit(a.len());
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
            let mut data = Buffer::uninit(total_rows * cols);
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
        let mut data = Buffer::uninit(batch_len * rows * cols);
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
    fn map_rows(&self, a: &FastTensor, f: impl Fn(&[f32], &mut [f32]) + Sync) -> FastTensor {
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
        assert_eq!(
            broadcast_shapes(&a.shape, shape),
            shape,
            "la forme {:?} ne s'étire pas à {shape:?}",
            a.shape,
        );
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
        assert_eq!(
            broadcast_shapes(&a.shape, &mask.shape),
            a.shape,
            "le masque {:?} ne s'étire pas à la forme {:?}",
            mask.shape,
            a.shape,
        );
        self.zip_with(a, mask, |x, m| if m != 0.0 { value } else { x })
    }

    fn gather_rows(&self, a: &FastTensor, indices: &[u32]) -> FastTensor {
        assert!(
            !a.shape.is_empty(),
            "gather_rows demande au moins une dimension"
        );
        let row_len: usize = a.shape[1..].iter().product();
        for &index in indices {
            assert!(
                (index as usize) < a.shape[0],
                "indice {index} hors de la première dimension ({})",
                a.shape[0],
            );
        }
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
        assert!(
            !a.shape.is_empty() && a.shape[0] == indices.len(),
            "scatter_add_rows : {} indices pour la forme {:?}",
            indices.len(),
            a.shape,
        );
        for &index in indices {
            assert!(
                (index as usize) < num_rows,
                "indice {index} hors de la première dimension ({num_rows})",
            );
        }
        let row_len: usize = a.shape[1..].iter().product();
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
        assert!(
            !a.shape.is_empty() && a.shape[1..] == b.shape[1..],
            "concat_rows : formes incompatibles {:?} et {:?}",
            a.shape,
            b.shape,
        );
        let mut data = Buffer::uninit(a.len() + b.len());
        data[..a.len()].copy_from_slice(a.data());
        data[a.len()..].copy_from_slice(b.data());
        let mut shape = a.shape.clone();
        shape[0] += b.shape[0];
        FastTensor::new(data, &shape)
    }

    fn write_rows(&self, table: &mut FastTensor, indices: &FastTensor, rows: &FastTensor) {
        assert!(
            !table.shape.is_empty() && !rows.shape.is_empty(),
            "write_rows demande au moins une dimension",
        );
        assert_eq!(
            table.shape[1..],
            rows.shape[1..],
            "write_rows : lignes {:?} pour une table {:?}",
            rows.shape,
            table.shape,
        );
        assert_eq!(
            indices.len(),
            rows.shape[0],
            "write_rows : {} indices pour {} lignes",
            indices.len(),
            rows.shape[0],
        );
        let row_len: usize = table.shape[1..].iter().product();
        let lines = table.shape[0];
        let target = table.data_mut();
        for (row, &index) in indices.data().iter().enumerate() {
            let index = index as usize;
            assert!(
                index < lines,
                "indice {index} hors de la première dimension ({lines})"
            );
            target[index * row_len..(index + 1) * row_len]
                .copy_from_slice(&rows.data()[row * row_len..(row + 1) * row_len]);
        }
    }

    fn sgd_step(&self, param: &mut FastTensor, grad: &FastTensor, learning_rate: f64) {
        assert_eq!(param.shape, grad.shape, "gradient de forme inattendue");
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
        assert_eq!(param.shape, grad.shape, "gradient de forme inattendue");
        assert_eq!(
            param.shape, first_moment.shape,
            "moment de forme inattendue"
        );
        assert_eq!(
            param.shape, second_moment.shape,
            "moment de forme inattendue"
        );
        let learning_rate = step.learning_rate as f32;
        let beta1 = step.beta1 as f32;
        let beta2 = step.beta2 as f32;
        let epsilon = step.epsilon as f32;
        let weight_decay = step.weight_decay as f32;
        let correction1 = step.bias_correction1 as f32;
        let correction2 = step.bias_correction2 as f32;
        let grad_scale = step.grad_scale as f32;
        let grad = grad.data();
        let first = SharedMut::new(first_moment.data_mut());
        let second = SharedMut::new(second_moment.data_mut());
        self.for_parts(param.data_mut(), CHUNK, |start, part| {
            let len = part.len();
            let (m_part, v_part) = unsafe { (first.slice(start, len), second.slice(start, len)) };
            for i in 0..len {
                let g = grad[start + i] * grad_scale;
                let m = beta1 * m_part[i] + (1.0 - beta1) * g;
                let v = beta2 * v_part[i] + (1.0 - beta2) * g * g;
                m_part[i] = m;
                v_part[i] = v;
                let update = (m / correction1) / ((v / correction2).sqrt() + epsilon);
                let p = part[i];
                part[i] = p - learning_rate * (update + weight_decay * p);
            }
        });
    }
}
