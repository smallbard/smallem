use super::muon::{stack, unstack};
use super::*;
use crate::autograd::Tape;
use crate::backend::Primitives;
use crate::backend::cpu::{Cpu, CpuTensor};
use rand::SeedableRng;

type B = Cpu<f64>;
type T = CpuTensor<f64>;

/// Calcule le gradient de `Σ p²`, soit `2p`, et renvoie de quoi faire
/// un pas d'optimiseur.
fn gradient_of_squares(backend: &B, parameter: &Parameter<B>) -> (u64, Gradients<B>) {
    let tape = Tape::new(backend);
    let p = parameter.var(&tape);
    let loss = tape.sum_all(tape.mul(p, p));
    (tape.id(), tape.backward(loss))
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-12, "{a} != {e}");
    }
}

#[test]
fn sgd_moves_against_the_gradient() {
    let backend = Cpu::new();
    let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
    let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
    // p - 0,1 × 2p = 0,8 p
    Sgd::new(0.1).step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
    close(parameter.value().data(), &[0.8, -1.6]);
}

#[test]
fn sgd_descends_to_the_minimum() {
    let backend = Cpu::new();
    let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
    let mut sgd = Sgd::new(0.1);
    for _ in 0..200 {
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        sgd.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
    }
    assert!(parameter.value().data().iter().all(|p| p.abs() < 1e-12));
}

#[test]
fn adamw_first_step_has_the_size_of_the_learning_rate() {
    // Au premier pas, les corrections de biais donnent m̂ = g et v̂ = g² :
    // le pas vaut lr × g / |g|, soit lr dans le sens opposé au gradient,
    // quelle que soit la taille du gradient.
    let backend = Cpu::new();
    let mut parameter = Parameter::new(T::from_vec(vec![1.0, -200.0], &[2]), true);
    let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
    let mut adamw = AdamW::new(0.01, 0.0);
    adamw.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
    let data = parameter.value().data();
    assert!((data[0] - 0.99).abs() < 1e-9, "{}", data[0]);
    assert!((data[1] + 199.99).abs() < 1e-9, "{}", data[1]);
    assert_eq!(adamw.steps(), 1);
}

#[test]
fn adamw_weight_decay_shrinks_weights_only() {
    // Gradient nul : seul le terme de décroissance agit.
    let backend = Cpu::new();
    let mut weight = Parameter::new(T::from_vec(vec![2.0], &[1]), true);
    let mut bias = Parameter::new(T::from_vec(vec![2.0], &[1]), false);

    let tape = Tape::new(&backend);
    let zero = tape.scalar(0.0);
    let loss = tape.sum_all(tape.add(
        tape.mul(weight.var(&tape), zero),
        tape.mul(bias.var(&tape), zero),
    ));
    let (tape_id, grads) = (tape.id(), tape.backward(loss));
    drop(tape);

    let mut adamw = AdamW::new(0.1, 0.5);
    adamw.step(&backend, vec![&mut weight, &mut bias], tape_id, &grads, 1.0);
    // 2 - 0,1 × 0,5 × 2 = 1,9 pour le poids ; le biais ne bouge pas.
    close(weight.value().data(), &[1.9]);
    close(bias.value().data(), &[2.0]);
}

#[test]
fn adamw_descends_to_the_minimum() {
    let backend = Cpu::new();
    let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
    let mut adamw = AdamW::new(0.05, 0.0);
    for _ in 0..500 {
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        adamw.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
    }
    assert!(
        parameter.value().data().iter().all(|p| p.abs() < 0.05),
        "{:?}",
        parameter.value().data()
    );
}

#[test]
fn parameter_without_gradient_is_left_alone() {
    let backend = Cpu::new();
    let mut used = Parameter::new(T::from_vec(vec![1.0], &[1]), true);
    let mut unused = Parameter::new(T::from_vec(vec![1.0], &[1]), true);
    let (tape_id, grads) = gradient_of_squares(&backend, &used);
    let mut adamw = AdamW::new(0.1, 0.5);
    adamw.step(&backend, vec![&mut used, &mut unused], tape_id, &grads, 1.0);
    assert_ne!(used.value().data(), &[1.0]);
    assert_eq!(unused.value().data(), &[1.0]);
}

#[test]
fn gradient_scale_shrinks_the_sgd_step() {
    let backend = Cpu::new();
    let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
    let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
    // p - 0,1 × 0,5 × 2p = 0,9 p
    Sgd::new(0.1).step(&backend, vec![&mut parameter], tape_id, &grads, 0.5);
    close(parameter.value().data(), &[0.9, -1.8]);
}

#[test]
fn gradient_scale_does_not_change_the_first_adamw_step() {
    // Adam divise le gradient par sa propre taille : au premier pas,
    // le réduire d'un facteur commun ne change rien.
    let backend = Cpu::new();
    let run = |scale: f64| {
        let mut parameter = Parameter::new(T::from_vec(vec![1.0, -200.0], &[2]), true);
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        AdamW::new(0.01, 0.0).step(&backend, vec![&mut parameter], tape_id, &grads, scale);
        parameter.value().data().to_vec()
    };
    for (a, b) in run(1.0).iter().zip(run(0.01)) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
}

#[test]
fn gradient_norm_spans_all_parameters() {
    // Gradients (2, -4) et (6) : norme √(4 + 16 + 36) = √56.
    let backend = Cpu::new();
    let first = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
    let second = Parameter::new(T::from_vec(vec![3.0], &[1]), true);
    let tape = Tape::new(&backend);
    let (a, b) = (first.var(&tape), second.var(&tape));
    let loss = tape.add(tape.sum_all(tape.mul(a, a)), tape.sum_all(tape.mul(b, b)));
    let grads = tape.backward(loss);
    let norm = gradient_norm(&backend, &[&first, &second], tape.id(), &grads);
    assert!((norm - 56f64.sqrt()).abs() < 1e-12, "{norm}");
}

#[test]
fn schedule_warms_up_then_decays() {
    let rate = |step| cosine_schedule(step, 100, 1000, 1e-3, 1e-4);
    // Montée linéaire, qui atteint le pic au dernier pas d'échauffement.
    assert!((rate(0) - 1e-5).abs() < 1e-12);
    assert!((rate(49) - 5e-4).abs() < 1e-12);
    assert!((rate(99) - 1e-3).abs() < 1e-12);
    // Le cosinus part du pic…
    assert!((rate(100) - 1e-3).abs() < 1e-12);
    // … passe à mi-hauteur à mi-parcours…
    assert!((rate(550) - 5.5e-4).abs() < 1e-12);
    // … et finit au plancher, où il reste.
    assert!((rate(1000) - 1e-4).abs() < 1e-12);
    assert!((rate(5000) - 1e-4).abs() < 1e-12);
}

#[test]
fn schedule_never_increases_after_warmup() {
    let rates: Vec<f64> = (100..1000)
        .map(|step| cosine_schedule(step, 100, 1000, 1e-3, 1e-4))
        .collect();
    assert!(rates.windows(2).all(|pair| pair[1] <= pair[0]));
}

#[test]
fn learning_rate_can_be_changed() {
    let mut adamw = AdamW::<B>::new(0.01, 0.0);
    Optimizer::<B>::set_learning_rate(&mut adamw, 0.5);
    assert_eq!(Optimizer::<B>::learning_rate(&adamw), 0.5);
}

// --- Muon ---

fn matrix(seed: u64, rows: usize, cols: usize) -> T {
    T::randn(&[rows, cols], &mut rand::rngs::StdRng::seed_from_u64(seed))
}

/// Écart maximal entre `m` et la matrice identité.
fn distance_to_identity(m: &T) -> f64 {
    let size = m.shape()[0];
    let mut largest = 0.0f64;
    for row in 0..size {
        for col in 0..size {
            let expected = if row == col { 1.0 } else { 0.0 };
            largest = largest.max((m.data()[row * size + col] - expected).abs());
        }
    }
    largest
}

#[test]
fn orthogonalize_brings_singular_values_to_one() {
    let backend = Cpu::new();
    // Une matrice haute, une large, une carrée : le plus petit des deux
    // produits `Xᵀ·X` et `X·Xᵀ` doit devenir l'identité.
    for (rows, cols) in [(7, 4), (4, 7), (5, 5)] {
        let x = orthogonalize(&backend, &matrix(1, rows, cols), &NEWTON_SCHULZ);
        assert_eq!(x.shape(), &[rows, cols]);
        let gram = if rows <= cols {
            x.matmul(&x.transpose(0, 1))
        } else {
            x.transpose(0, 1).matmul(&x)
        };
        let distance = distance_to_identity(&gram);
        assert!(distance < 1e-3, "{rows} × {cols} : écart {distance}");
    }
}

#[test]
fn orthogonalize_keeps_directions_and_forgets_sizes() {
    let backend = Cpu::new();
    // Une rotation d'un quart de tour, étirée de 5 dans une direction et
    // écrasée de 0,2 dans l'autre : il reste la rotation.
    let stretched = T::from_vec(vec![0.0, -0.2, 5.0, 0.0], &[2, 2]);
    let x = orthogonalize(&backend, &stretched, &NEWTON_SCHULZ);
    close_to(x.data(), &[0.0, -1.0, 1.0, 0.0], 1e-2);
    // Multiplier la matrice par un nombre positif ne change rien.
    let m = matrix(2, 4, 6);
    let a = orthogonalize(&backend, &m, &NEWTON_SCHULZ);
    let b = orthogonalize(&backend, &m.mul(&T::scalar(1000.0)), &NEWTON_SCHULZ);
    close_to(a.data(), b.data(), 1e-9);
}

fn close_to(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < tolerance, "{a} != {e}");
    }
}

#[test]
fn muon_first_step_is_the_orthogonalized_gradient() {
    let backend = Cpu::new();
    let start = matrix(3, 3, 5);
    let grad = matrix(4, 3, 5);
    let mut weight = Parameter::new(start.clone(), true);
    let mut muon = Muon::<B>::new(0.01, 0.0);
    muon.apply(&backend, vec![&mut weight], &[Some(&grad)], 1.0);
    // Au premier pas, la moyenne est le gradient, et « un pas plus
    // loin » en est un multiple : la direction est celle du gradient
    // orthogonalisé. Le pas vaut `lr × γ × √max(3, 5)`.
    let step = 0.01 * 0.18 * 5f64.sqrt();
    let expected = start.sub(&orthogonalize(&backend, &grad, &NEWTON_SCHULZ).mul(&T::scalar(step)));
    close_to(weight.value().data(), expected.data(), 1e-12);
    // La taille du gradient ne compte pas.
    let mut again = Parameter::new(start, true);
    let mut muon = Muon::<B>::new(0.01, 0.0);
    let larger = grad.mul(&T::scalar(50.0));
    muon.apply(&backend, vec![&mut again], &[Some(&larger)], 1.0);
    close_to(again.value().data(), weight.value().data(), 1e-9);
}

#[test]
fn muon_leaves_everything_but_weight_matrices_to_adamw() {
    let backend = Cpu::new();
    let build = || {
        (
            Parameter::<B>::new(matrix(5, 3, 4), true),
            // Une table d'embeddings : deux dimensions, sans décroissance.
            Parameter::<B>::new(matrix(6, 3, 4), false),
            Parameter::<B>::new(T::from_vec(vec![1.0, -2.0], &[2]), false),
        )
    };
    let grads = [matrix(7, 3, 4), matrix(8, 3, 4), T::ones(&[2])];
    let grads: Vec<Option<&T>> = grads.iter().map(Some).collect();

    let (mut weight, mut table, mut bias) = build();
    assert!(Muon::handles(&backend, &weight));
    assert!(!Muon::handles(&backend, &table) && !Muon::handles(&backend, &bias));
    let mut muon = Muon::<B>::new(0.01, 0.1);
    muon.apply(
        &backend,
        vec![&mut weight, &mut table, &mut bias],
        &grads,
        1.0,
    );

    let (mut plain_weight, mut plain_table, mut plain_bias) = build();
    let mut adamw = AdamW::<B>::new(0.01, 0.1);
    let all = vec![&mut plain_weight, &mut plain_table, &mut plain_bias];
    adamw.apply(&backend, all, &grads, 1.0);

    assert_eq!(table.value().data(), plain_table.value().data());
    assert_eq!(bias.value().data(), plain_bias.value().data());
    assert_ne!(weight.value().data(), plain_weight.value().data());
    // Un seul état de moyenne glissante, pour la matrice de poids.
    let state = muon.state();
    let (_, momenta) = state.slots.last().unwrap();
    assert_eq!(
        momenta.iter().map(Option::is_some).collect::<Vec<_>>(),
        [true, false, false]
    );
}

#[test]
fn muon_weight_decay_shrinks_the_matrix() {
    // Un gradient nul s'orthogonalise en zéro : seule la décroissance
    // agit, de `lr × décroissance`.
    let backend = Cpu::new();
    let start = matrix(9, 2, 3);
    let mut weight = Parameter::new(start.clone(), true);
    let mut muon = Muon::<B>::new(0.1, 0.5);
    muon.apply(
        &backend,
        vec![&mut weight],
        &[Some(&T::zeros(&[2, 3]))],
        1.0,
    );
    close_to(
        weight.value().data(),
        start.mul(&T::scalar(0.95)).data(),
        1e-9,
    );
}

#[test]
fn muon_descends_to_the_minimum() {
    // `Σ w²` sur une matrice : chaque pas a une taille fixée par le
    // taux, qu'on fait décroître pour finir près de zéro.
    let backend = Cpu::new();
    let mut parameter = Parameter::new(matrix(10, 3, 3), true);
    let mut muon = Muon::<B>::new(0.2, 0.0);
    for step in 0..200 {
        muon.set_learning_rate(0.2 * (1.0 - step as f64 / 200.0));
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        muon.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
    }
    let largest = parameter
        .value()
        .data()
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(largest < 0.05, "plus grand poids : {largest}");
}

#[test]
fn muon_state_can_be_saved_and_restored() {
    let backend = Cpu::new();
    let grad = matrix(11, 3, 4);
    let mut weight = Parameter::new(matrix(12, 3, 4), true);
    let mut muon = Muon::<B>::new(0.01, 0.0);
    muon.apply(&backend, vec![&mut weight], &[Some(&grad)], 1.0);

    // Une copie de l'état, comme le ferait un point de reprise.
    let state = muon.state();
    let copy = OptimizerState {
        counters: state.counters.clone(),
        slots: state
            .slots
            .iter()
            .map(|(name, tensors)| (name.clone(), tensors.iter().map(|t| t.cloned()).collect()))
            .collect(),
    };
    let (counters, slots) = muon.state_layout();
    assert_eq!(counters, ["adamw.steps"]);
    assert_eq!(slots, ["adamw.m.", "adamw.v.", "muon.m."]);

    let mut restored = Muon::<B>::new(0.01, 0.0);
    restored.restore_state(copy).unwrap();
    let mut twin = Parameter::new(weight.value().clone(), true);
    muon.apply(&backend, vec![&mut weight], &[Some(&grad)], 1.0);
    restored.apply(&backend, vec![&mut twin], &[Some(&grad)], 1.0);
    assert_eq!(weight.value().data(), twin.value().data());
    // Sans la série de Muon, l'état est refusé.
    let without = OptimizerState::<T> {
        counters: BTreeMap::from([("adamw.steps".to_string(), "1".to_string())]),
        slots: vec![
            ("adamw.m.".to_string(), vec![None]),
            ("adamw.v.".to_string(), vec![None]),
        ],
    };
    let error = Muon::<B>::new(0.01, 0.0).restore_state(without);
    assert!(error.unwrap_err().contains("muon.m."));
}

#[test]
fn a_refused_variant_is_an_error_for_try_with_variant() {
    let variant = MuonVariant {
        period: 0,
        ..MuonVariant::default()
    };
    assert!(Muon::<B>::try_with_variant(1e-2, 0.1, variant).is_err());
    assert!(Muon::<B>::try_with_variant(1e-2, 0.1, MuonVariant::default()).is_ok());
}

#[test]
fn optimizers_are_chosen_by_name() {
    let variants = [
        MuonVariant::default(),
        MuonVariant {
            short: true,
            period: 3,
            row_scale: 0.2,
            ..MuonVariant::default()
        },
        MuonVariant {
            neuron_norm: true,
            ..MuonVariant::default()
        },
        MuonVariant {
            tail_boost: 5.0,
            ..MuonVariant::default()
        },
    ];
    let mut kinds = vec![OptimizerKind::AdamW];
    kinds.extend(variants.map(OptimizerKind::Muon));
    for kind in kinds {
        assert_eq!(kind.name().parse::<OptimizerKind>(), Ok(kind));
    }
    // Le Muon de DeepSeek-V4 garde son nom nu : les points de reprise
    // de l'étape 10 restent lisibles.
    assert_eq!(OptimizerKind::Muon(MuonVariant::default()).name(), "muon");
    assert_eq!(
        OptimizerKind::Muon(variants[1]).name(),
        "muon:short,period=3,row=0.2"
    );
    for (text, message) in [
        ("sgd", "adamw, muon"),
        ("adamw:short", "adamw, muon"),
        ("muon:fast", "`fast` inconnu"),
        ("muon:period=0", "au moins 1"),
        ("muon:period=x", "attend un entier"),
        ("muon:boost=0.5", "au moins 1"),
        ("muon:norm,boost=5", "ne se combinent pas"),
    ] {
        let error = text.parse::<OptimizerKind>().unwrap_err();
        assert!(error.contains(message), "{text} : {error}");
    }
    let muon = AnyOptimizer::<B>::new(OptimizerKind::Muon(variants[0]), 0.5, 0.1);
    assert_eq!(muon.learning_rate(), 0.5);
    assert_eq!(muon.state_layout().1.len(), 3);
    // NorMuon a une série de plus à enregistrer.
    let normuon = AnyOptimizer::<B>::new(OptimizerKind::Muon(variants[2]), 0.5, 0.1);
    assert_eq!(normuon.state_layout().1.len(), 4);
    let adamw = AnyOptimizer::<B>::new(OptimizerKind::AdamW, 0.5, 0.1);
    assert_eq!(adamw.state_layout().1.len(), 2);
}

#[test]
fn a_stack_of_matrices_is_orthogonalized_one_by_one() {
    let backend = Cpu::new();
    let matrices = [matrix(1, 4, 7), matrix(2, 4, 7), matrix(3, 4, 7)];
    // Des tailles très différentes : chacune a sa propre norme.
    let matrices = [
        matrices[0].clone(),
        matrices[1].mul(&T::scalar(1e3)),
        matrices[2].mul(&T::scalar(1e-3)),
    ];
    let stacked = stack(&backend, &matrices.iter().collect::<Vec<_>>());
    assert_eq!(stacked.shape(), &[3, 4, 7]);
    let together = orthogonalize(&backend, &stacked, &NEWTON_SCHULZ);
    for (index, single) in matrices.iter().enumerate() {
        let alone = orthogonalize(&backend, single, &NEWTON_SCHULZ);
        close_to(
            unstack(&backend, &together, index).data(),
            alone.data(),
            1e-9,
        );
    }
}

#[test]
fn muon_treats_each_matrix_as_if_it_were_alone() {
    // Trois matrices, dont deux de même forme : les empiler pour les
    // orthogonaliser ne doit rien changer à leurs mises à jour.
    let backend = Cpu::new();
    let shapes = [(3, 5), (4, 4), (3, 5)];
    let grads: Vec<T> = (0..3)
        .map(|i| matrix(10 + i as u64, shapes[i].0, shapes[i].1))
        .collect();
    let build = || -> Vec<Parameter<B>> {
        (0..3)
            .map(|i| Parameter::new(matrix(20 + i as u64, shapes[i].0, shapes[i].1), true))
            .collect()
    };
    let mut together = build();
    let mut muon = Muon::<B>::new(0.01, 0.1);
    for _ in 0..3 {
        let all: Vec<Option<&T>> = grads.iter().map(Some).collect();
        muon.apply(&backend, together.iter_mut().collect(), &all, 1.0);
    }
    let mut alone = build();
    for (parameter, grad) in alone.iter_mut().zip(&grads) {
        let mut muon = Muon::<B>::new(0.01, 0.1);
        for _ in 0..3 {
            muon.apply(&backend, vec![&mut *parameter], &[Some(grad)], 1.0);
        }
    }
    for (a, b) in together.iter().zip(&alone) {
        close_to(a.value().data(), b.value().data(), 1e-12);
    }
}

#[test]
fn short_newton_schulz_orthogonalizes_too() {
    let backend = Cpu::new();
    for (rows, cols) in [(7, 4), (4, 7), (5, 5)] {
        let x = orthogonalize(&backend, &matrix(1, rows, cols), &NEWTON_SCHULZ_SHORT);
        let gram = if rows <= cols {
            x.matmul(&x.transpose(0, 1))
        } else {
            x.transpose(0, 1).matmul(&x)
        };
        // Quatre itérations : l'article annonce une erreur de 0,007 sur
        // chaque valeur singulière, donc du double sur son carré.
        let distance = distance_to_identity(&gram);
        assert!(distance < 3e-2, "{rows} × {cols} : écart {distance}");
    }
}

#[test]
fn normalize_rows_gives_unit_rows_along_the_long_side() {
    let backend = Cpu::new();
    // Large : chaque ligne. Haute : chaque colonne.
    let wide = normalize_rows(&backend, &matrix(1, 3, 5));
    for row in 0..3 {
        let norm: f64 = (0..5)
            .map(|col| wide.data()[row * 5 + col] * wide.data()[row * 5 + col])
            .sum();
        assert!((norm - 1.0).abs() < 1e-6, "ligne {row} : {norm}");
    }
    let tall = normalize_rows(&backend, &matrix(2, 5, 3));
    for col in 0..3 {
        let norm: f64 = (0..5)
            .map(|row| tall.data()[row * 3 + col] * tall.data()[row * 3 + col])
            .sum();
        assert!((norm - 1.0).abs() < 1e-6, "colonne {col} : {norm}");
    }
    // La direction de chaque ligne est gardée.
    let source = matrix(1, 3, 5);
    let ratio = wide.data()[0] / source.data()[0];
    assert!(ratio > 0.0);
    assert!((wide.data()[1] / source.data()[1] - ratio).abs() < 1e-9);
}

/// `3·u₁·v₁ᵀ + u₂·v₂ᵀ`, avec des vecteurs choisis perpendiculaires :
/// une matrice dont on connaît la direction dominante, `u₁·v₁ᵀ`.
fn two_directions() -> (T, T) {
    let u1 = T::from_vec(vec![0.6, 0.8, 0.0], &[3, 1]);
    let u2 = T::from_vec(vec![0.0, 0.0, 1.0], &[3, 1]);
    let v1 = T::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[1, 4]);
    let v2 = T::from_vec(vec![0.5, -0.5, 0.5, -0.5], &[1, 4]);
    let leading = u1.matmul(&v1);
    let m = leading.mul(&T::scalar(3.0)).add(&u2.matmul(&v2));
    (m, leading)
}

#[test]
fn leading_direction_finds_the_largest_singular_pair() {
    let backend = Cpu::new();
    let (m, leading) = two_directions();
    close_to(
        leading_direction(&backend, &m, 10).data(),
        leading.data(),
        1e-6,
    );
    // Haute plutôt que large : même résultat, transposé.
    let found = leading_direction(&backend, &m.transpose(0, 1), 10);
    close_to(found.transpose(0, 1).data(), leading.data(), 1e-6);
}

/// Un pas de Muon sur une matrice, avec la variante et l'avancement
/// donnés ; renvoie le déplacement `W₀ − W₁`.
fn muon_step(variant: MuonVariant, progress: f64, grads: &[&T]) -> Vec<T> {
    let backend = Cpu::new();
    let mut weight = Parameter::<B>::new(T::zeros(grads[0].shape()), true);
    let mut muon = Muon::<B>::with_variant(1.0, 0.0, variant);
    muon.set_progress(progress);
    grads
        .iter()
        .map(|grad| {
            let before: T = weight.value().clone();
            muon.apply(&backend, vec![&mut weight], &[Some(grad)], 1.0);
            before.sub(weight.value())
        })
        .collect()
}

#[test]
fn periodic_muon_orthogonalizes_one_step_out_of_k() {
    let backend = Cpu::new();
    let variant = MuonVariant {
        period: 3,
        row_scale: 0.25,
        ..MuonVariant::default()
    };
    let grad = matrix(4, 3, 5);
    let moves = muon_step(variant, 0.0, &[&grad; 4]);
    let scale = 0.18 * 5f64.sqrt();
    // Pas 0 et 3 : orthogonalisés. La moyenne est un multiple du
    // gradient à chaque pas, puisque le gradient ne change pas.
    let orthogonal = orthogonalize(&backend, &grad, &NEWTON_SCHULZ).mul(&T::scalar(scale));
    close_to(moves[0].data(), orthogonal.data(), 1e-7);
    close_to(moves[3].data(), orthogonal.data(), 1e-7);
    // Pas 1 et 2 : lignes normalisées, à l'échelle réduite.
    let rows = normalize_rows(&backend, &grad).mul(&T::scalar(scale * 0.25));
    close_to(moves[1].data(), rows.data(), 1e-7);
    close_to(moves[2].data(), rows.data(), 1e-7);
}

#[test]
fn neuron_norm_gives_every_neuron_the_same_update_size() {
    let (rows, cols) = (4, 6);
    // Dix itérations laissent des colonnes de normes inégales sur une
    // matrice mal conditionnée ; la normalisation les égalise.
    let mut data = matrix(4, rows, cols);
    data = data.mul(&T::from_vec(
        vec![1.0, 1.0, 1.0, 1.0, 1e-3, 1e-3],
        &[1, cols],
    ));
    let variant = MuonVariant {
        neuron_norm: true,
        ..MuonVariant::default()
    };
    let update = &muon_step(variant, 0.0, &[&data])[0];
    let column = |col: usize| -> f64 {
        (0..rows)
            .map(|row| update.data()[row * cols + col] * update.data()[row * cols + col])
            .sum()
    };
    for col in 1..cols {
        assert!(
            (column(col) - column(0)).abs() < 1e-3 * column(0),
            "colonne {col} : {} contre {}",
            column(col),
            column(0)
        );
    }
    // Les éléments valent `γ` en moyenne quadratique.
    let mean_square: f64 = update.data().iter().map(|x| x * x).sum::<f64>() / (rows * cols) as f64;
    assert!((mean_square.sqrt() - 0.18).abs() < 1e-6, "{mean_square}");
}

#[test]
fn tail_boost_amplifies_everything_but_the_leading_direction() {
    let backend = Cpu::new();
    let (m, leading) = two_directions();
    let variant = MuonVariant {
        tail_boost: 5.0,
        ..MuonVariant::default()
    };
    let plain = &muon_step(MuonVariant::default(), 1.0, &[&m])[0];
    // Au début de l'entraînement, l'amplification est éteinte.
    let cold = &muon_step(variant, 0.0, &[&m])[0];
    close_to(cold.data(), plain.data(), 1e-12);
    // À mi-montée, elle vaut la moitié de son excès : 1 + 4 / 2 = 3.
    let scale = 0.18 * 4f64.sqrt();
    let lead = leading.mul(&T::scalar(scale));
    let expected = |boost: f64| {
        plain
            .mul(&T::scalar(boost))
            .sub(&lead.mul(&T::scalar(boost - 1.0)))
    };
    let warm = &muon_step(variant, 0.15, &[&m])[0];
    close_to(warm.data(), expected(3.0).data(), 1e-6);
    let hot = &muon_step(variant, 0.5, &[&m])[0];
    close_to(hot.data(), expected(5.0).data(), 1e-6);
    // Projetée sur la direction dominante, la mise à jour a la taille
    // de celle de Muon ; sur l'autre, cinq fois plus. La tolérance est
    // celle de l'orthogonalisation, dont l'erreur est ici amplifiée.
    let along = |update: &T, direction: &T| -> f64 {
        let product = backend.mul(update, direction);
        product.data().iter().sum()
    };
    let other = orthogonalize(&backend, &m, &NEWTON_SCHULZ).sub(&leading);
    assert!((along(hot, &leading) / along(plain, &leading) - 1.0).abs() < 5e-2);
    assert!((along(hot, &other) / along(plain, &other) - 5.0).abs() < 5e-2);
}
