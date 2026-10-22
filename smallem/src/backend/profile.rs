//! Un backend qui chronomètre chaque primitive et chaque opération d'un
//! autre.
//!
//! Sur le processeur, une opération est finie quand l'appel rend la main :
//! l'horloge autour de l'appel mesure son exécution. Le profileur du GPU
//! (`smallem-cuda/src/profile.rs`) doit en plus attendre la carte après
//! chaque opération ; celui-ci n'a rien à attendre, et ne ralentit pas ce
//! qu'il mesure.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rand::Rng;

use super::ops::{Operation, Run};
use super::{AdamWStep, Primitives};

/// Nombre d'appels et temps cumulé d'une opération.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub calls: u64,
    pub total: Duration,
}

/// Une mesure se range sous le nom de l'opération, et sous le fait qu'elle
/// est une rétropropagation : une clé sans allocation, même à chaque appel.
type Key = (&'static str, bool);

pub struct Profiled<B> {
    inner: B,
    timings: RefCell<BTreeMap<Key, Timing>>,
}

impl<B> Profiled<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            timings: RefCell::new(BTreeMap::new()),
        }
    }

    pub fn inner(&self) -> &B {
        &self.inner
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
            .map(|(&(name, backward), &timing)| {
                let name = if backward {
                    format!("{name} (gradient)")
                } else {
                    name.to_string()
                };
                (name, timing)
            })
            .collect();
        report.sort_by_key(|&(_, timing)| std::cmp::Reverse(timing.total));
        report
    }

    fn timed<T>(&self, key: Key, operation: impl FnOnce(&B) -> T) -> T {
        let start = Instant::now();
        let result = operation(&self.inner);
        let elapsed = start.elapsed();
        let mut timings = self.timings.borrow_mut();
        let timing = timings.entry(key).or_default();
        timing.calls += 1;
        timing.total += elapsed;
        result
    }
}

/// Écrit des méthodes de [`Primitives`] qui délèguent au backend mesuré en
/// chronométrant l'appel sous le nom de la méthode.
macro_rules! timed {
    ($(fn $name:ident(&self $(, $argument:ident: $type:ty)* $(,)?) $(-> $output:ty)?;)*) => {
        $(
            fn $name(&self $(, $argument: $type)*) $(-> $output)? {
                self.timed((stringify!($name), false), |inner| inner.$name($($argument),*))
            }
        )*
    };
}

impl<B: Primitives> Primitives for Profiled<B> {
    type Scalar = B::Scalar;
    type Tensor = B::Tensor;

    fn shape<'a>(&self, tensor: &'a B::Tensor) -> &'a [usize] {
        self.inner.shape(tensor)
    }

    fn randn(&self, shape: &[usize], rng: &mut impl Rng) -> B::Tensor {
        self.inner.randn(shape, rng)
    }

    timed! {
        fn from_vec(&self, data: Vec<B::Scalar>, shape: &[usize]) -> B::Tensor;
        fn full(&self, shape: &[usize], value: B::Scalar) -> B::Tensor;
        fn to_vec(&self, tensor: &B::Tensor) -> Vec<B::Scalar>;

        fn add(&self, a: &B::Tensor, b: &B::Tensor) -> B::Tensor;
        fn sub(&self, a: &B::Tensor, b: &B::Tensor) -> B::Tensor;
        fn mul(&self, a: &B::Tensor, b: &B::Tensor) -> B::Tensor;
        fn div(&self, a: &B::Tensor, b: &B::Tensor) -> B::Tensor;
        fn neg(&self, a: &B::Tensor) -> B::Tensor;
        fn exp(&self, a: &B::Tensor) -> B::Tensor;
        fn ln(&self, a: &B::Tensor) -> B::Tensor;
        fn sqrt(&self, a: &B::Tensor) -> B::Tensor;
        fn tanh(&self, a: &B::Tensor) -> B::Tensor;

        fn sum(&self, a: &B::Tensor, axis: usize, keepdim: bool) -> B::Tensor;
        fn sum_all(&self, a: &B::Tensor) -> B::Tensor;
        fn reshape(&self, a: &B::Tensor, shape: &[usize]) -> B::Tensor;
        fn transpose(&self, a: &B::Tensor, axis_a: usize, axis_b: usize) -> B::Tensor;
        fn expand(&self, a: &B::Tensor, shape: &[usize]) -> B::Tensor;
        fn matmul(&self, a: &B::Tensor, b: &B::Tensor) -> B::Tensor;

        fn softmax(&self, a: &B::Tensor) -> B::Tensor;
        fn log_softmax(&self, a: &B::Tensor) -> B::Tensor;

        fn masked_fill(&self, a: &B::Tensor, mask: &B::Tensor, value: B::Scalar) -> B::Tensor;
        fn gather_rows(&self, a: &B::Tensor, indices: &[u32]) -> B::Tensor;
        fn scatter_add_rows(&self, a: &B::Tensor, indices: &[u32], num_rows: usize) -> B::Tensor;
        fn concat_rows(&self, a: &B::Tensor, b: &B::Tensor) -> B::Tensor;

        fn write_rows(&self, table: &mut B::Tensor, indices: &B::Tensor, rows: &B::Tensor);
        fn sgd_step(&self, param: &mut B::Tensor, grad: &B::Tensor, learning_rate: f64);
        fn adamw_step(
            &self,
            param: &mut B::Tensor,
            grad: &B::Tensor,
            first_moment: &mut B::Tensor,
            second_moment: &mut B::Tensor,
            step: &AdamWStep,
        );
    }
}

/// Toute opération que le backend mesuré sait exécuter est chronométrée sous
/// son nom. Les primitives qu'elle appelle, dans sa définition de référence,
/// sont celles du backend mesuré : elles ne sont pas comptées à part, et le
/// temps de l'opération est compté une fois.
impl<B: Primitives + Run<O>, O: Operation> Run<O> for Profiled<B> {
    fn run(&self, op: &O, inputs: &[&B::Tensor]) -> B::Tensor {
        self.timed((O::NAME, false), |inner| inner.run(op, inputs))
    }

    fn run_saving(&self, op: &O, inputs: &[&B::Tensor]) -> (B::Tensor, Option<B::Tensor>) {
        self.timed((O::NAME, false), |inner| inner.run_saving(op, inputs))
    }

    fn run_backward(
        &self,
        op: &O,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        self.run_backward_saved(op, inputs, output, None, grad)
    }

    fn run_backward_saved(
        &self,
        op: &O,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        saved: Option<&B::Tensor>,
        grad: &B::Tensor,
    ) -> Vec<Option<B::Tensor>> {
        self.timed((O::NAME, true), |inner| {
            inner.run_backward_saved(op, inputs, output, saved, grad)
        })
    }

    fn run_backward_needed(
        &self,
        op: &O,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        saved: Option<&B::Tensor>,
        grad: &B::Tensor,
        needed: &[bool],
    ) -> Vec<Option<B::Tensor>> {
        self.timed((O::NAME, true), |inner| {
            inner.run_backward_needed(op, inputs, output, saved, grad, needed)
        })
    }
}
