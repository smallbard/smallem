//! Tests des opérations sur CPU en `f64` : leurs définitions de référence,
//! sur de petits exemples calculés à la main. Leurs gradients sont vérifiés
//! par différences finies dans `autograd.rs`, et leurs kernels contre ces
//! définitions dans `tests/kernels.rs`.

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::*;
use crate::backend::cpu::{Cpu, CpuTensor};

type B = Cpu<f64>;
type T = CpuTensor<f64>;

fn tensor(data: &[f64], shape: &[usize]) -> T {
    T::from_vec(data.to_vec(), shape)
}

fn assert_close(actual: &T, expected: &T) {
    assert_eq!(actual.shape(), expected.shape());
    for (a, e) in actual.data().iter().zip(expected.data()) {
        assert!((a - e).abs() < 1e-9, "{a} != {e}");
    }
}

#[test]
fn normalize_gives_zero_mean_and_unit_variance() {
    let backend = B::new();
    let x = T::randn(&[3, 7], &mut StdRng::seed_from_u64(0))
        .mul(&T::scalar(40.0))
        .add(&T::scalar(-7.0));
    let normalized = backend.run(&Normalize { epsilon: 0.0 }, &[&x]);
    assert_close(&normalized.mean(1, false), &T::zeros(&[3]));
    let variance = normalized.mul(&normalized).mean(1, false);
    assert_close(&variance, &T::ones(&[3]));
}

#[test]
fn rms_normalize_gives_unit_mean_square_without_centering() {
    let backend = B::new();
    let x = tensor(&[3.0, 4.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0], &[2, 4]);
    let normalized = backend.run(&RmsNormalize { epsilon: 0.0 }, &[&x]);
    // Première ligne : moyenne des carrés 25/4, racine 2,5.
    assert_close(
        &normalized,
        &tensor(&[1.2, 1.6, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[2, 4]),
    );
    // La moyenne n'est pas retirée : la seconde ligne, constante, reste
    // constante, là où `Normalize` la rendrait nulle.
    let centered = backend.run(&Normalize { epsilon: 1e-12 }, &[&x]);
    assert!(centered.data()[4..].iter().all(|value| value.abs() < 1e-6));
}

#[test]
fn causal_mask_hides_the_future() {
    let backend = B::new();
    assert_eq!(
        causal_mask(&backend, 3, 3).data(),
        &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    );
    // Deux requêtes face à cinq positions : elles sont aux positions 3 et
    // 4. La première ne voit pas la dernière colonne, la seconde voit tout.
    assert_eq!(
        causal_mask(&backend, 2, 5).data(),
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    // Une seule requête, la dernière position : rien n'est caché.
    assert_eq!(causal_mask(&backend, 1, 4).data(), &[0.0; 4]);
}

#[test]
fn causal_softmax_ignores_the_future() {
    let backend = B::new();
    let scores = T::randn(&[2, 4, 4], &mut StdRng::seed_from_u64(1));
    let probabilities = backend.run(&CausalSoftmax { scale: 0.5 }, &[&scores]);
    for matrix in 0..2 {
        for row in 0..4 {
            let line = &probabilities.data()[(matrix * 4 + row) * 4..][..4];
            // Rien pour le futur, et une distribution sur le reste.
            assert!(line[row + 1..].iter().all(|&p| p == 0.0));
            assert!((line.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(line[..=row].iter().all(|&p| p > 0.0));
        }
    }
    // Changer un score du futur ne change rien.
    let mut data = scores.data().to_vec();
    data[3] += 100.0;
    let changed = backend.run(
        &CausalSoftmax { scale: 0.5 },
        &[&T::from_vec(data, &[2, 4, 4])],
    );
    assert_eq!(changed, probabilities);
}

#[test]
fn causal_softmax_applies_the_scale() {
    // Sur la dernière ligne, où rien n'est masqué, c'est un softmax
    // ordinaire des scores multipliés par l'échelle.
    let backend = B::new();
    let scores = tensor(&[9.0, 9.0, 0.0, 2.0], &[2, 2]);
    let probabilities = backend.run(&CausalSoftmax { scale: 0.5 }, &[&scores]);
    let expected = tensor(&[0.0, 1.0], &[1, 2]).softmax();
    assert_close(&probabilities.gather_rows(&[1]), &expected);
    assert_eq!(probabilities.data()[..2], [1.0, 0.0]);
}

#[test]
fn causal_softmax_of_the_last_queries_matches_the_square_one() {
    // Les deux dernières lignes du softmax causal carré sont le softmax
    // causal des deux dernières requêtes.
    let backend = B::new();
    let scores = T::randn(&[5, 5], &mut StdRng::seed_from_u64(9));
    let op = CausalSoftmax { scale: 0.7 };
    let square = backend.run(&op, &[&scores]);
    let last = backend.run(&op, &[&scores.gather_rows(&[3, 4])]);
    assert_close(&last, &square.gather_rows(&[3, 4]));
}

#[test]
fn attention_of_the_first_position_is_its_value() {
    // La première position ne voit qu'elle-même : son poids vaut 1, et elle
    // reçoit sa propre valeur, quelles que soient les clés.
    let backend = B::new();
    let mut rng = StdRng::seed_from_u64(3);
    let (q, k, v) = (
        T::randn(&[3, 2], &mut rng),
        T::randn(&[3, 2], &mut rng),
        T::randn(&[3, 4], &mut rng),
    );
    let out = backend.run(&FusedAttention { scale: 0.5 }, &[&q, &k, &v]);
    assert_eq!(out.shape(), &[3, 4]);
    assert_close(&out.gather_rows(&[0]), &v.gather_rows(&[0]));
}

#[test]
fn attention_of_the_last_queries_matches_the_square_one() {
    // Comme pour le softmax causal : les dernières requêtes seules voient
    // ce que voient les dernières lignes du calcul complet.
    let backend = B::new();
    let mut rng = StdRng::seed_from_u64(4);
    let (q, k, v) = (
        T::randn(&[5, 3], &mut rng),
        T::randn(&[5, 3], &mut rng),
        T::randn(&[5, 2], &mut rng),
    );
    let op = FusedAttention { scale: 0.6 };
    let square = backend.run(&op, &[&q, &k, &v]);
    let last = backend.run(&op, &[&q.gather_rows(&[3, 4]), &k, &v]);
    assert_close(&last, &square.gather_rows(&[3, 4]));
}

#[test]
fn cross_entropy_of_uniform_logits_is_ln_vocab() {
    let backend = B::new();
    let op = CrossEntropy {
        targets: vec![0, 3, 6],
    };
    let logits = T::zeros(&[3, 7]);
    let loss = backend.run(&op, &[&logits]);
    assert!((loss.item() - 7f64.ln()).abs() < 1e-12);
    // Gradient : (1/7 - cible) / 3, donc des lignes de somme nulle.
    let one = T::scalar(1.0);
    let grad = backend
        .run_backward(&op, &[&logits], &loss, &one)
        .remove(0)
        .unwrap();
    assert_close(&grad.sum(1, false), &T::zeros(&[3]));
    assert!((grad.data()[0] - (1.0 / 7.0 - 1.0) / 3.0).abs() < 1e-12);
    assert!((grad.data()[1] - (1.0 / 7.0) / 3.0).abs() < 1e-12);
}

#[test]
fn only_the_rotated_tensor_receives_a_gradient() {
    let backend = B::new();
    let x = T::randn(&[2, 3, 4], &mut StdRng::seed_from_u64(2));
    let cos = tensor(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], &[3, 2]);
    let sin = tensor(&[0.0, 1.0, 1.0, 0.0, 0.0, -1.0], &[3, 2]);
    let rotated = backend.run(&RotatePairs, &[&x, &cos, &sin]);
    let grads = backend.run_backward(&RotatePairs, &[&x, &cos, &sin], &rotated, &rotated);
    assert_eq!(grads.len(), 3);
    assert!(grads[1].is_none() && grads[2].is_none());
    // Le gradient est la rotation inverse : appliqué à la sortie, il
    // redonne l'entrée.
    assert_close(grads[0].as_ref().unwrap(), &x);
}

#[test]
fn concat_appends_along_the_last_axis() {
    let backend = B::new();
    // Deux lignes : [1, 2] puis [3, 4], suivies de [5] puis [6].
    let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 6.0], &[2, 1]);
    let joined = backend.run(&Concat, &[&a, &b]);
    assert_close(&joined, &tensor(&[1.0, 2.0, 5.0, 3.0, 4.0, 6.0], &[2, 3]));
    // Avec des dimensions devant, et pour un simple vecteur.
    let a = T::randn(&[2, 3, 4], &mut StdRng::seed_from_u64(0));
    let b = T::randn(&[2, 3, 2], &mut StdRng::seed_from_u64(1));
    let joined = backend.run(&Concat, &[&a, &b]);
    assert_eq!(joined.shape(), &[2, 3, 6]);
    for row in 0..6 {
        assert_eq!(
            joined.data()[row * 6..row * 6 + 4],
            a.data()[row * 4..row * 4 + 4]
        );
        assert_eq!(
            joined.data()[row * 6 + 4..row * 6 + 6],
            b.data()[row * 2..row * 2 + 2]
        );
    }
    let vector = backend.run(
        &Concat,
        &[&tensor(&[1.0], &[1]), &tensor(&[2.0, 3.0], &[2])],
    );
    assert_close(&vector, &tensor(&[1.0, 2.0, 3.0], &[3]));
}

#[test]
fn concat_gradient_is_cut_back_in_two() {
    let backend = B::new();
    let a = T::zeros(&[2, 3, 4]);
    let b = T::zeros(&[2, 3, 2]);
    let grad = T::randn(&[2, 3, 6], &mut StdRng::seed_from_u64(2));
    let grads = backend.run_backward(&Concat, &[&a, &b], &grad, &grad);
    let (grad_a, grad_b) = (grads[0].as_ref().unwrap(), grads[1].as_ref().unwrap());
    assert_eq!(
        (grad_a.shape(), grad_b.shape()),
        (&[2, 3, 4][..], &[2, 3, 2][..])
    );
    // Remis bout à bout, les deux morceaux redonnent le gradient.
    assert_close(&backend.run(&Concat, &[grad_a, grad_b]), &grad);
}

#[test]
#[should_panic(expected = "concat : formes incompatibles")]
fn concat_refuses_different_leading_dimensions() {
    B::new().run(&Concat, &[&T::zeros(&[2, 3]), &T::zeros(&[3, 3])]);
}

#[test]
#[should_panic(expected = "l'opération `gelu` attend 1 entrée(s), reçu 2")]
fn wrong_number_of_inputs_is_refused() {
    let backend = B::new();
    let x = T::zeros(&[2]);
    backend.run(&Gelu, &[&x, &x]);
}

#[test]
fn operations_have_distinct_names() {
    let names = [
        Gelu::NAME,
        Silu::NAME,
        Normalize::NAME,
        RmsNormalize::NAME,
        CausalSoftmax::NAME,
        RotatePairs::NAME,
        CrossEntropy::NAME,
        Concat::NAME,
        FusedAttention::NAME,
    ];
    let distinct: std::collections::BTreeSet<_> = names.iter().collect();
    assert_eq!(distinct.len(), names.len());
}
