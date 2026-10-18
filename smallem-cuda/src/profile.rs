//! Un backend GPU qui chronomètre chaque opération.
//!
//! Les kernels sont lancés sans attendre leur fin : une horloge autour d'un
//! appel ne mesure que sa mise en file. [`Profiled`] attend donc le GPU
//! après chaque opération. Le total est plus lent qu'en temps normal, mais
//! la répartition du temps entre opérations est juste.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use smallem::backend::ops::{Operation, Run};
use smallem::backend::{AdamWStep, Primitives};

use crate::cuda::{Cuda, CudaTensor};

/// Nombre d'appels et temps cumulé d'une opération.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub calls: u64,
    pub total: Duration,
}

pub struct Profiled {
    inner: Cuda,
    timings: RefCell<BTreeMap<String, Timing>>,
}

impl Profiled {
    pub fn new(inner: Cuda) -> Self {
        Self {
            inner,
            timings: RefCell::new(BTreeMap::new()),
        }
    }

    /// Oublie les mesures prises jusqu'ici (les premiers pas, par exemple).
    pub fn reset(&self) {
        self.timings.borrow_mut().clear();
    }

    /// Les opérations, de la plus coûteuse à la moins coûteuse.
    pub fn report(&self) -> Vec<(String, Timing)> {
        let mut report: Vec<_> = self
            .timings
            .borrow()
            .iter()
            .map(|(name, &timing)| (name.clone(), timing))
            .collect();
        report.sort_by_key(|&(_, timing)| std::cmp::Reverse(timing.total));
        report
    }

    fn timed<T>(&self, name: &str, operation: impl FnOnce(&Cuda) -> T) -> T {
        let start = Instant::now();
        let result = operation(&self.inner);
        self.inner.synchronize();
        let mut timings = self.timings.borrow_mut();
        let timing = timings.entry(name.to_string()).or_default();
        timing.calls += 1;
        timing.total += start.elapsed();
        result
    }
}

type T = CudaTensor;

/// Écrit des méthodes de [`Primitives`] qui délèguent au backend GPU en
/// chronométrant l'appel sous le nom de la méthode.
macro_rules! timed {
    ($(fn $name:ident(&self $(, $argument:ident: $type:ty)* $(,)?) $(-> $output:ty)?;)*) => {
        $(
            fn $name(&self $(, $argument: $type)*) $(-> $output)? {
                self.timed(stringify!($name), |gpu| gpu.$name($($argument),*))
            }
        )*
    };
}

impl Primitives for Profiled {
    type Scalar = f32;
    type Tensor = CudaTensor;

    // Lire une forme ou en changer ne lance aucun kernel : rien à mesurer.
    fn shape<'a>(&self, tensor: &'a T) -> &'a [usize] {
        self.inner.shape(tensor)
    }

    fn reshape(&self, a: &T, shape: &[usize]) -> T {
        self.inner.reshape(a, shape)
    }

    timed! {
        fn from_vec(&self, data: Vec<f32>, shape: &[usize]) -> T;
        fn full(&self, shape: &[usize], value: f32) -> T;
        fn to_vec(&self, tensor: &T) -> Vec<f32>;

        fn add(&self, a: &T, b: &T) -> T;
        fn sub(&self, a: &T, b: &T) -> T;
        fn mul(&self, a: &T, b: &T) -> T;
        fn div(&self, a: &T, b: &T) -> T;
        fn neg(&self, a: &T) -> T;
        fn exp(&self, a: &T) -> T;
        fn ln(&self, a: &T) -> T;
        fn sqrt(&self, a: &T) -> T;
        fn tanh(&self, a: &T) -> T;

        fn sum(&self, a: &T, axis: usize, keepdim: bool) -> T;
        fn sum_all(&self, a: &T) -> T;
        fn transpose(&self, a: &T, axis_a: usize, axis_b: usize) -> T;
        fn expand(&self, a: &T, shape: &[usize]) -> T;
        fn matmul(&self, a: &T, b: &T) -> T;

        fn softmax(&self, a: &T) -> T;
        fn log_softmax(&self, a: &T) -> T;

        fn masked_fill(&self, a: &T, mask: &T, value: f32) -> T;
        fn gather_rows(&self, a: &T, indices: &[u32]) -> T;
        fn scatter_add_rows(&self, a: &T, indices: &[u32], num_rows: usize) -> T;
        fn concat_rows(&self, a: &T, b: &T) -> T;

        fn write_rows(&self, table: &mut T, indices: &T, rows: &T);
        fn sgd_step(&self, param: &mut T, grad: &T, learning_rate: f64);
        fn adamw_step(
            &self,
            param: &mut T,
            grad: &T,
            first_moment: &mut T,
            second_moment: &mut T,
            step: &AdamWStep,
        );
    }
}

/// Toute opération que le GPU sait exécuter est chronométrée sous son nom.
/// Le profileur n'a pas à connaître la liste des opérations.
impl<O: Operation> Run<O> for Profiled
where
    Cuda: Run<O>,
{
    fn run(&self, op: &O, inputs: &[&T]) -> T {
        self.timed(O::NAME, |gpu| gpu.run(op, inputs))
    }

    fn run_saving(&self, op: &O, inputs: &[&T]) -> (T, Option<T>) {
        self.timed(O::NAME, |gpu| gpu.run_saving(op, inputs))
    }

    fn run_backward(&self, op: &O, inputs: &[&T], output: &T, grad: &T) -> Vec<Option<T>> {
        self.run_backward_saved(op, inputs, output, None, grad)
    }

    fn run_backward_saved(
        &self,
        op: &O,
        inputs: &[&T],
        output: &T,
        saved: Option<&T>,
        grad: &T,
    ) -> Vec<Option<T>> {
        // Compté à part : le gradient d'une opération a ses propres kernels.
        let name = format!("{} (gradient)", O::NAME);
        self.timed(&name, |gpu| {
            gpu.run_backward_saved(op, inputs, output, saved, grad)
        })
    }
}
