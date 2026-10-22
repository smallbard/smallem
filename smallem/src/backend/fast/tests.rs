//! Chaque primitive et chaque opération de [`Fast`] est comparée à celle du
//! backend de référence en `f64`, sur les mêmes entrées aléatoires, avec les
//! tolérances des tests du GPU. Chacune est aussi exécutée avec un, deux et
//! huit fils : les résultats doivent être les mêmes au bit près.

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{Fast, Isa};
use crate::backend::cpu::Cpu;
use crate::backend::ops::{
    Affine, CachedAttention, CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu, Normalize,
    QuantizedMatmul, RmsNormalize, RotatePairs, Run, Silu, TransposedMatmul,
};
use crate::backend::{AdamWStep, Backend, Float, Primitives, standard_normal};

/// Écart relatif toléré, celui des tests du GPU : `Fast` calcule en `f32`,
/// dans un autre ordre que la référence.
const TOLERANCE: f64 = 1e-5;

type Input = (Vec<f32>, Vec<usize>);

fn random_inputs(shapes: &[&[usize]], seed: u64) -> Vec<Input> {
    let mut rng = StdRng::seed_from_u64(seed);
    shapes
        .iter()
        .map(|shape| {
            let len: usize = shape.iter().product();
            let data = (0..len).map(|_| standard_normal(&mut rng) as f32).collect();
            (data, shape.to_vec())
        })
        .collect()
}

/// Exécute `op` sur un backend et rapatrie le résultat en `f64`.
fn eval<B: Backend>(
    backend: &B,
    inputs: &[Input],
    op: impl Fn(&B, &[B::Tensor]) -> B::Tensor,
) -> (Vec<f64>, Vec<usize>) {
    let tensors: Vec<B::Tensor> = inputs
        .iter()
        .map(|(data, shape)| {
            let data = data
                .iter()
                .map(|&x| B::Scalar::from_f64(x as f64))
                .collect();
            backend.from_vec(data, shape)
        })
        .collect();
    let result = op(backend, &tensors);
    let data = backend.to_vec(&result).iter().map(|x| x.to_f64()).collect();
    (data, backend.shape(&result).to_vec())
}

/// Les backends rapides comparés : huit fils, avec chaque jeu
/// d'instructions du processeur.
fn fast_backends() -> Vec<Fast> {
    let fast = Fast::with_threads(8);
    [Isa::Avx512, Isa::Avx2, Isa::Scalar]
        .into_iter()
        .filter(|isa| isa.available())
        .map(|isa| fast.with_isa(isa))
        .collect()
}

fn assert_close(
    expected: &(Vec<f64>, Vec<usize>),
    actual: &(Vec<f64>, Vec<usize>),
    bound: Option<&[f64]>,
    what: &str,
) {
    assert_eq!(expected.1, actual.1, "{what} : formes différentes");
    for (index, (&e, &a)) in expected.0.iter().zip(&actual.0).enumerate() {
        if e == a || (e.is_nan() && a.is_nan()) {
            continue;
        }
        let scale = match bound {
            Some(bound) => bound[index].max(1e-30),
            None => e.abs().max(a.abs()).max(1.0),
        };
        assert!(
            (e - a).abs() <= TOLERANCE * scale,
            "{what}, élément {index} : référence {e}, rapide {a}",
        );
    }
}

/// Compare `Fast` à la référence, et vérifie que le nombre de fils ne change
/// rien. Avec `product`, l'écart est rapporté à `|a| · |b|`, comme pour le
/// GPU : un produit de centaines de termes peut tomber près de zéro.
macro_rules! check_with {
    ($inputs:expr, $product:expr, |$b:ident, $t:ident| $op:expr) => {{
        let inputs: Vec<Input> = $inputs;
        let expected = eval(&Cpu::<f64>::new(), &inputs, |$b, $t| $op);
        let bound = if $product {
            let absolute: Vec<Input> = inputs
                .iter()
                .map(|(data, shape)| (data.iter().map(|v| v.abs()).collect(), shape.clone()))
                .collect();
            Some(eval(&Cpu::<f64>::new(), &absolute, |$b, $t| $op).0)
        } else {
            None
        };
        for fast in fast_backends() {
            let what = format!("{:?}", fast);
            let actual = eval(&fast, &inputs, |$b, $t| $op);
            assert_close(&expected, &actual, bound.as_deref(), &what);
            for threads in [1, 2] {
                let other = Fast::with_threads(threads).with_isa(fast.isa());
                let again = eval(&other, &inputs, |$b, $t| $op);
                assert_eq!(
                    actual.0, again.0,
                    "{what} contre {threads} fil(s) : bits différents"
                );
            }
        }
    }};
}

macro_rules! check {
    ($shapes:expr, |$b:ident, $t:ident| $op:expr) => {
        check_with!(random_inputs($shapes, 0), false, |$b, $t| $op)
    };
}

macro_rules! check_product {
    ($shapes:expr, |$b:ident, $t:ident| $op:expr) => {
        check_with!(random_inputs($shapes, 0), true, |$b, $t| $op)
    };
}

const MATRIX: &[usize] = &[7, 300];
/// Plus d'éléments qu'une tâche : le travail se partage vraiment.
const LARGE: &[usize] = &[70, 1000];

// --- Élément par élément ---

#[test]
fn elementwise() {
    for shape in [MATRIX, LARGE] {
        check!(&[shape, shape], |b, t| b.add(&t[0], &t[1]));
        check!(&[shape, shape], |b, t| b.sub(&t[0], &t[1]));
        check!(&[shape, shape], |b, t| b.mul(&t[0], &t[1]));
        check!(&[shape, shape], |b, t| b.div(&t[0], &t[1]));
        check!(&[shape], |b, t| b.neg(&t[0]));
        check!(&[shape], |b, t| b.exp(&t[0]));
        check!(&[shape], |b, t| b.tanh(&t[0]));
        check!(&[shape], |b, t| {
            let positive = b.add(&b.mul(&t[0], &t[0]), &b.scalar(Float::from_f64(0.1)));
            b.add(&b.ln(&positive), &b.sqrt(&positive))
        });
    }
    check!(&[&[1]], |b, t| b.exp(&t[0]));
    check!(&[&[100]], |b, t| b
        .exp(&b.mul(&t[0], &b.scalar(Float::from_f64(40.0)))));
}

#[test]
fn broadcasting() {
    check!(&[MATRIX, &[300]], |b, t| b.add(&t[0], &t[1]));
    check!(&[LARGE, &[1000]], |b, t| b.add(&t[0], &t[1]));
    check!(&[MATRIX, &[7, 1]], |b, t| b.mul(&t[0], &t[1]));
    check!(&[&[7, 1], &[1, 300]], |b, t| b.sub(&t[0], &t[1]));
    check!(&[MATRIX], |b, t| b
        .mul(&t[0], &b.scalar(Float::from_f64(2.5))));
    check!(&[&[2, 3, 5, 7], &[5, 7]], |b, t| b.add(&t[0], &t[1]));
    check!(&[&[300], MATRIX], |b, t| b.div(&t[0], &t[1]));
    check!(&[&[4, 9, 1], &[4, 9, 11]], |b, t| b.mul(&t[0], &t[1]));
}

// --- Réductions ---

#[test]
fn reductions() {
    check!(&[MATRIX], |b, t| b.sum(&t[0], 0, false));
    check!(&[MATRIX], |b, t| b.sum(&t[0], 1, false));
    check!(&[&[4, 9, 11]], |b, t| b.sum(&t[0], 1, true));
    check!(&[&[4, 9, 11]], |b, t| b.sub(&t[0], &b.mean(&t[0], 2, true)));
    // Beaucoup de lignes, peu de colonnes : le gradient d'un biais, coupé
    // en paquets de lignes. Sur des milliers de termes, l'écart se rapporte
    // à la somme des valeurs absolues, comme pour un produit.
    check_product!(&[&[3000, 33]], |b, t| b.sum(&t[0], 0, false));
    check_product!(&[&[2, 1500, 5]], |b, t| b.sum(&t[0], 1, true));
    check!(&[&[5]], |b, t| b.sum_all(&t[0]));
    check_product!(&[&[100, 1000]], |b, t| b.sum_all(&t[0]));
}

// --- Changements de forme ---

#[test]
fn shapes() {
    check!(&[MATRIX], |b, t| b.reshape(&t[0], &[300, 7]));
    check!(&[MATRIX], |b, t| b.transpose(&t[0], 0, 1));
    check!(&[LARGE], |b, t| b.transpose(&t[0], 0, 1));
    check!(&[&[2, 5, 3, 7]], |b, t| b.transpose(&t[0], 1, 2));
    check!(&[&[4, 9, 11]], |b, t| b.transpose(&t[0], 0, 2));
    check!(&[&[3, 33, 70]], |b, t| b.transpose(&t[0], 2, 1));
    check!(&[&[7, 1]], |b, t| b.expand(&t[0], &[3, 7, 300]));
    check!(&[&[300]], |b, t| b.expand(&t[0], &[7, 300]));
}

// --- Produit matriciel ---

#[test]
fn matmul() {
    let shapes: &[(&[usize], &[usize])] = &[
        (&[17, 33], &[33, 21]),
        (&[4, 9, 33], &[33, 21]),
        (&[2, 3, 9, 8], &[2, 3, 8, 9]),
        (&[9, 8], &[5, 8, 6]),
        (&[130, 70], &[70, 150]),
        (&[3, 100, 37], &[37, 129]),
        (&[1, 512], &[512, 512]),
        (&[3, 1000], &[1000, 70]),
        (&[300, 600], &[600, 90]),
        (&[2, 2, 65, 17], &[2, 2, 17, 65]),
    ];
    for &(a, b_shape) in shapes {
        check_product!(&[a, b_shape], |b, t| b.matmul(&t[0], &t[1]));
    }
}

// --- Softmax et sélection ---

#[test]
fn softmax_and_selection() {
    check!(&[MATRIX], |b, t| b.softmax(&t[0]));
    check!(&[MATRIX], |b, t| b.log_softmax(&t[0]));
    check!(&[MATRIX], |b, t| b
        .softmax(&b.mul(&t[0], &b.scalar(Float::from_f64(500.0)))));
    check!(&[&[2, 3, 9, 9]], |b, t| b.softmax(&t[0]));
    check!(&[MATRIX, MATRIX], |b, t| b.masked_fill(
        &t[0],
        &t[1],
        Float::from_f64(-7.0)
    ));
    check!(&[&[257, 48]], |b, t| {
        let indices: Vec<u32> = (0..500).map(|i| (i * 37 + 256) % 257).collect();
        b.gather_rows(&t[0], &indices)
    });
    check!(&[&[500, 48]], |b, t| {
        let indices: Vec<u32> = (0..500).map(|i| (i * 37) % 23).collect();
        b.scatter_add_rows(&t[0], &indices, 30)
    });
    check!(&[&[5, 3, 2], &[7, 3, 2]], |b, t| b
        .concat_rows(&t[0], &t[1]));
}

// --- Mises à jour sur place ---

#[test]
fn in_place_updates() {
    let step = AdamWStep {
        learning_rate: 0.01,
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        weight_decay: 0.1,
        bias_correction1: 0.19,
        bias_correction2: 0.0975,
        grad_scale: 0.5,
    };
    let shapes: &[&[usize]] = &[LARGE, LARGE, LARGE, LARGE];
    check!(shapes, |b, t| {
        let mut param = t[0].clone();
        b.sgd_step(&mut param, &t[1], 0.1);
        param
    });
    for output in 0..3 {
        check!(shapes, |b, t| {
            let (mut param, mut m, mut v) = (t[0].clone(), t[2].clone(), b.mul(&t[3], &t[3]));
            b.adamw_step(&mut param, &t[1], &mut m, &mut v, &step);
            [param, m, v][output].clone()
        });
    }
    check!(&[&[10, 4], &[3, 4]], |b, t| {
        let mut table = t[0].clone();
        let indices = b.from_vec(
            vec![
                Float::from_f64(7.0),
                Float::from_f64(2.0),
                Float::from_f64(9.0),
            ],
            &[3],
        );
        b.write_rows(&mut table, &indices, &t[1]);
        table
    });
}

#[test]
fn updating_a_shared_tensor_copies_it() {
    let fast = Fast::with_threads(2);
    let param = fast.from_vec(vec![1.0, 2.0], &[2]);
    let mut updated = param.clone();
    fast.sgd_step(&mut updated, &fast.ones(&[2]), 0.5);
    assert_eq!(fast.to_vec(&param), [1.0, 2.0]);
    assert_eq!(fast.to_vec(&updated), [0.5, 1.5]);
}

// --- Opérations ---

/// Vérifie une opération, sa sortie puis le gradient de chacune de ses
/// entrées, la dernière forme étant celle du gradient de la sortie.
macro_rules! check_op {
    ($op:expr, $shapes:expr, $inputs:expr, $product:expr) => {{
        let shapes: &[&[usize]] = $shapes;
        let inputs = $inputs;
        let all = random_inputs(shapes, 0);
        check_with!(all.clone(), $product, |b, t| {
            let refs: Vec<_> = t[..inputs].iter().collect();
            b.run(&$op, &refs)
        });
        for input in 0..inputs {
            check_with!(all.clone(), $product, |b, t| {
                let refs: Vec<_> = t[..inputs].iter().collect();
                let output = b.run(&$op, &refs);
                b.run_backward(&$op, &refs, &output, &t[inputs])[input]
                    .clone()
                    .expect("un gradient")
            });
        }
    }};
}

#[test]
fn activations() {
    check_op!(Gelu, &[LARGE, LARGE], 1, false);
    check_op!(Silu, &[LARGE, LARGE], 1, false);
}

#[test]
fn normalizations() {
    check_op!(
        Normalize { epsilon: 1e-5 },
        &[&[4, 9, 33], &[4, 9, 33]],
        1,
        false
    );
    check_op!(
        RmsNormalize { epsilon: 1e-5 },
        &[&[70, 512], &[70, 512]],
        1,
        false
    );
}

#[test]
fn causal_softmax() {
    check_op!(
        CausalSoftmax { scale: 0.35 },
        &[&[2, 3, 9, 9], &[2, 3, 9, 9]],
        1,
        false
    );
    check_op!(
        CausalSoftmax { scale: 0.35 },
        &[&[2, 3, 3, 9], &[2, 3, 3, 9]],
        1,
        false
    );
    check_op!(
        CausalSoftmax { scale: 1.0 },
        &[&[300, 300], &[300, 300]],
        1,
        false
    );
}

#[test]
fn rotate_pairs() {
    // Des angles quelconques : les tables sont leurs cosinus et sinus.
    for (x, angles) in [
        (&[2usize, 3, 9, 16][..], &[9usize, 8][..]),
        (&[300, 6], &[300, 3]),
    ] {
        let mut inputs = random_inputs(&[x, angles, angles, x], 0);
        inputs[1].0 = inputs[1].0.iter().map(|a| a.cos()).collect();
        inputs[2].0 = inputs[2].0.iter().map(|a| a.sin()).collect();
        check_with!(inputs.clone(), false, |b, t| b
            .run(&RotatePairs, &[&t[0], &t[1], &t[2]]));
        check_with!(inputs.clone(), false, |b, t| {
            let output = b.run(&RotatePairs, &[&t[0], &t[1], &t[2]]);
            b.run_backward(&RotatePairs, &[&t[0], &t[1], &t[2]], &output, &t[3])[0]
                .clone()
                .unwrap()
        });
    }
}

#[test]
fn products() {
    check_op!(
        Affine,
        &[&[4, 9, 33], &[33, 21], &[21], &[4, 9, 21]],
        3,
        true
    );
    check_op!(
        Affine,
        &[&[1, 1, 512], &[512, 70], &[70], &[1, 1, 70]],
        3,
        true
    );
    check_op!(Affine, &[&[300, 50], &[50, 70], &[70], &[300, 70]], 3, true);
    check_op!(
        TransposedMatmul,
        &[&[9, 300], &[1000, 300], &[9, 1000]],
        2,
        true
    );
    check_op!(
        TransposedMatmul,
        &[&[1, 1, 300], &[1000, 300], &[1, 1, 1000]],
        2,
        true
    );
    check_op!(
        TransposedMatmul,
        &[&[4, 50, 64], &[257, 64], &[4, 50, 257]],
        2,
        true
    );
}

#[test]
fn backward_skips_the_gradients_nobody_needs() {
    // `Fast` et la référence : les gradients demandés sont ceux de la
    // rétropropagation complète, au bit près ; les autres ne sont pas
    // calculés.
    fn check<B: Backend + Run<O>, O: crate::backend::ops::Operation>(
        backend: &B,
        op: &O,
        inputs: &[Input],
        grad: &Input,
    ) {
        let tensor = |(data, shape): &Input| {
            let data = data
                .iter()
                .map(|&x| B::Scalar::from_f64(x as f64))
                .collect();
            backend.from_vec(data, shape)
        };
        let tensors: Vec<B::Tensor> = inputs.iter().map(tensor).collect();
        let refs: Vec<&B::Tensor> = tensors.iter().collect();
        let grad = tensor(grad);
        let output = backend.run(op, &refs);
        let full = backend.run_backward(op, &refs, &output, &grad);
        for skipped in 0..inputs.len() {
            let needed: Vec<bool> = (0..inputs.len()).map(|i| i != skipped).collect();
            let some = backend.run_backward_needed(op, &refs, &output, None, &grad, &needed);
            for (index, (all, partial)) in full.iter().zip(&some).enumerate() {
                if index == skipped {
                    assert!(partial.is_none(), "gradient {index} calculé pour rien");
                } else {
                    let (all, partial) = (all.as_ref().unwrap(), partial.as_ref().unwrap());
                    assert_eq!(backend.to_vec(all), backend.to_vec(partial));
                }
            }
        }
    }
    let affine = random_inputs(&[&[4, 9, 33], &[33, 21], &[21], &[4, 9, 21]], 2);
    let transposed = random_inputs(&[&[9, 300], &[100, 300], &[9, 100]], 3);
    for fast in fast_backends() {
        check(&fast, &Affine, &affine[..3], &affine[3]);
        check(&fast, &TransposedMatmul, &transposed[..2], &transposed[2]);
    }
    let reference = Cpu::<f64>::new();
    check(&reference, &Affine, &affine[..3], &affine[3]);
    check(
        &reference,
        &TransposedMatmul,
        &transposed[..2],
        &transposed[2],
    );
}

#[test]
fn concat() {
    check_op!(
        Concat,
        &[&[2, 3, 9, 16], &[2, 3, 9, 6], &[2, 3, 9, 22]],
        2,
        false
    );
    check_op!(Concat, &[&[300, 5], &[300, 1], &[300, 6]], 2, false);
}

#[test]
fn cross_entropy() {
    let targets: Vec<u32> = (0..70).map(|i| (i * 13) % 257).collect();
    let op = CrossEntropy { targets };
    check_with!(random_inputs(&[&[70, 257]], 0), false, |b, t| b
        .run(&op, &[&t[0]]));
    check_with!(random_inputs(&[&[70, 257]], 0), false, |b, t| {
        let output = b.run(&op, &[&t[0]]);
        let grad = b.scalar(Float::from_f64(0.7));
        b.run_backward(&op, &[&t[0]], &output, &grad)[0]
            .clone()
            .unwrap()
    });
}

/// `q`, `k`, `v` et le gradient de la sortie.
fn attention_shapes(
    heads: &[usize],
    queries: usize,
    positions: usize,
    key: usize,
    value: usize,
) -> Vec<Vec<usize>> {
    let with = |rows: usize, width: usize| [heads, &[rows, width]].concat();
    vec![
        with(queries, key),
        with(positions, key),
        with(positions, value),
        with(queries, value),
    ]
}

#[test]
fn fused_attention() {
    for (heads, queries, positions, key, value) in [
        (&[2usize, 3][..], 70, 70, 64, 64),
        (&[2, 2][..], 3, 45, 16, 16),
        (&[4][..], 1, 70, 8, 8),
        (&[3][..], 33, 33, 5, 3),
        (&[2][..], 9, 9, 96, 64),
        (&[1, 2][..], 300, 300, 32, 32),
    ] {
        let shapes = attention_shapes(heads, queries, positions, key, value);
        let shapes: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        check_op!(FusedAttention { scale: 0.3 }, &shapes, 3, false);
    }
}

#[test]
fn cached_attention() {
    // Deux séquences dans une table de 12 lignes, des lignes données dans le
    // désordre ; trois requêtes, deux têtes de requêtes par tête de clé.
    let (lines, kv_heads, heads, dim) = (12, 2, 4, 8);
    let mut inputs = random_inputs(
        &[
            &[3, heads, dim],
            &[lines, kv_heads, dim],
            &[lines, kv_heads, dim],
        ],
        0,
    );
    let slots: Vec<f32> = [3, 7, 1, 10, 0, 0, 5, 2, 11, 4, 8, 9]
        .iter()
        .map(|&x| x as f32)
        .collect();
    inputs.push((slots, vec![2, 6]));
    inputs.push((vec![0.0, 3.0, 1.0, 6.0, 0.0, 4.0], vec![3, 2]));
    for value_dim in [None, Some(5)] {
        let op = CachedAttention {
            scale: 0.4,
            value_dim,
        };
        check_with!(inputs.clone(), false, |b, t| {
            b.run(&op, &[&t[0], &t[1], &t[2], &t[3], &t[4]])
        });
    }
}

#[test]
fn quantized_matmul() {
    use crate::quantize::{Format, Quantized};
    // Peu de lignes : les codes sont lus. Beaucoup : les poids déquantifiés.
    // Des colonnes qui ne remplissent pas un bloc de 64, et une dimension
    // commune faite de plusieurs groupes et tranches.
    for (format, rows, k, n) in [
        (Format::Int8, 3, 64, 40),
        (Format::Int4 { group: 32 }, 3, 64, 40),
        (Format::Int8, 1, 512, 300),
        (Format::Int4 { group: 64 }, 6, 512, 130),
        (Format::Int4 { group: 64 }, 40, 128, 70),
    ] {
        let mut inputs = random_inputs(&[&[rows, k], &[k, n], &[n]], 1);
        let weight: Vec<f64> = inputs[1].0.iter().map(|&w| w as f64).collect();
        let quantized = Quantized::round_to_nearest(&weight, k, n, format);
        let words = (quantized.words.clone(), vec![quantized.words.len() / n, n]);
        let scales = (
            quantized.scales.clone(),
            vec![quantized.scales.len() / n, n],
        );
        let dequantized = quantized.dequantize().iter().map(|&w| w as f32).collect();
        inputs[1] = (dequantized, vec![k, n]);
        let all = vec![
            inputs[0].clone(),
            words,
            scales,
            inputs[1].clone(),
            inputs[2].clone(),
        ];
        let op = QuantizedMatmul { format };
        check_with!(all.clone(), true, |b, t| b
            .run(&op, &[&t[0], &t[1], &t[2], &t[3]]));
        check_with!(all.clone(), true, |b, t| b
            .run(&op, &[&t[0], &t[1], &t[2], &t[3], &t[4]]));
        // Le gradient de l'entrée, celui d'un produit par `Wᵀ`.
        let grad = random_inputs(&[&[rows, n]], 2).remove(0);
        let with_grad = [all, vec![grad]].concat();
        check_with!(with_grad.clone(), true, |b, t| {
            let inputs = [&t[0], &t[1], &t[2], &t[3], &t[4]];
            let output = b.run(&op, &inputs);
            b.run_backward(&op, &inputs, &output, &t[5])[0]
                .clone()
                .unwrap()
        });
    }
}
