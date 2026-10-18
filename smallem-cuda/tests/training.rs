//! L'entraînement sur GPU est comparé à l'entraînement sur CPU : mêmes
//! poids de départ, mêmes lots, donc mêmes pertes.
//!
//! `cargo oxide test -- --test training`

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::cpu::Cpu;
use smallem::backend::{Backend, Float};
use smallem::data::Dataset;
use smallem::mlp::{Mlp, MlpConfig, next_token_examples};
use smallem::nn::{Module, Parameter};
use smallem::optim::{
    AdamW, Muon, MuonVariant, NEWTON_SCHULZ, NEWTON_SCHULZ_SHORT, Optimizer, Sgd, orthogonalize,
};
use smallem::training::{evaluate, train_step};
use smallem_cuda::Cuda;

const CONFIG: MlpConfig = MlpConfig {
    vocab: 257,
    context: 8,
    embed: 16,
    hidden: 64,
};

/// Un corpus synthétique avec de la structure à apprendre.
fn dataset() -> Dataset {
    Dataset::from_tokens(
        (0..5000u32)
            .map(|i| ((i * i + i / 3) % 97) as u16)
            .collect(),
    )
}

/// Entraîne un modèle neuf pendant `steps` pas et renvoie la perte de
/// chaque pas. Tout l'aléa vient de `SEED` : deux backends reçoivent les
/// mêmes poids de départ et les mêmes lots.
fn loss_curve<B: Backend>(backend: &B, steps: usize) -> Vec<f64> {
    const SEED: u64 = 5;
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut model = Mlp::new(backend, CONFIG, &mut rng);
    let mut optimizer = AdamW::new(1e-2, 0.1);
    let dataset = dataset();

    (0..steps)
        .map(|_| {
            let batch = dataset.sample_batch(&mut rng, 64, CONFIG.context);
            let (contexts, targets) = next_token_examples(&batch);
            let loss = train_step(backend, &mut model, &mut optimizer, |m, t| {
                m.loss(t, &contexts, &targets)
            });
            backend.item(&loss).to_f64()
        })
        .collect()
}

#[test]
fn gpu_follows_the_same_loss_curve_as_cpu() {
    const STEPS: usize = 30;
    let cpu = loss_curve(&Cpu::<f32>::new(), STEPS);
    let gpu = loss_curve(&Cuda::new(0).unwrap(), STEPS);

    // La perte doit avoir vraiment baissé, sinon la comparaison ne dit rien.
    assert!(
        cpu[STEPS - 1] < cpu[0] - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        // Les écarts d'arrondi se propagent d'un pas au suivant : la
        // tolérance est plus large qu'entre deux opérations isolées.
        assert!(
            (c - g).abs() < 1e-3,
            "pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}",
        );
    }
}

#[test]
fn initial_loss_on_gpu_is_ln_vocab() {
    let gpu = Cuda::new(0).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let model = Mlp::new(&gpu, CONFIG, &mut rng);
    let batch = dataset().sample_batch(&mut rng, 64, CONFIG.context);
    let (contexts, targets) = next_token_examples(&batch);
    let loss = evaluate(&gpu, &model, |m, t| m.loss(t, &contexts, &targets));
    assert!((loss - 257f32.ln()).abs() < 0.01, "perte initiale {loss}");
}

#[test]
fn gpu_overfits_a_single_batch() {
    let gpu = Cuda::new(0).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Mlp::new(&gpu, CONFIG, &mut rng);
    let mut optimizer = AdamW::new(1e-2, 0.0);
    let batch = dataset().sample_batch(&mut rng, 32, CONFIG.context);
    let (contexts, targets) = next_token_examples(&batch);

    for _ in 0..300 {
        train_step(&gpu, &mut model, &mut optimizer, |m, t| {
            m.loss(t, &contexts, &targets)
        });
    }
    let loss = evaluate(&gpu, &model, |m, t| m.loss(t, &contexts, &targets));
    assert!(loss < 0.01, "perte après sur-apprentissage : {loss}");
}

/// Un pas de SGD sur la perte `Σ p²`, dont le gradient est `2p`.
fn sgd_once<B: Backend<Scalar = f32>>(backend: &B, values: &[f32]) -> Vec<f32> {
    let mut parameter = Parameter::new(backend.from_vec(values.to_vec(), &[values.len()]), true);
    let (tape_id, grads) = {
        let tape = Tape::new(backend);
        let p = parameter.var(&tape);
        let loss = tape.sum_all(tape.mul(p, p));
        (tape.id(), tape.backward(loss))
    };
    Sgd::new(0.1).step(backend, vec![&mut parameter], tape_id, &grads, 1.0);
    backend.to_vec(parameter.value())
}

#[test]
fn sgd_step_matches_cpu() {
    let values = [1.0f32, -2.0, 3.0, 0.5];
    let on_cpu = sgd_once(&Cpu::<f32>::new(), &values);
    let on_gpu = sgd_once(&Cuda::new(0).unwrap(), &values);
    for (c, g) in on_cpu.iter().zip(&on_gpu) {
        assert!((c - g).abs() < 1e-6, "CPU {on_cpu:?}, GPU {on_gpu:?}");
    }
    // p - 0,1 × 2p = 0,8 p
    assert!((on_gpu[0] - 0.8).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "encore partagé")]
fn updating_a_parameter_still_on_a_tape_panics() {
    let backend = Cuda::new(0).unwrap();
    let mut model = Mlp::new(&backend, CONFIG, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    let loss = model.loss(&tape, &[0; 8], &[1]);
    let grads = tape.backward(loss);
    // La tape est encore vivante : elle référence les paramètres.
    Sgd::new(0.1).step(&backend, model.parameters_mut(), tape.id(), &grads, 1.0);
}

/// Orthogonalise une matrice aléatoire et rend le résultat, puis l'écart
/// maximal entre `X·Xᵀ` et l'identité.
fn orthogonalized<B: Backend<Scalar = f32>>(
    backend: &B,
    coefficients: &[(f64, f64, f64)],
) -> (Vec<f32>, f32) {
    const ROWS: usize = 24;
    let matrix = backend.randn(&[ROWS, 40], &mut StdRng::seed_from_u64(3));
    let x = orthogonalize(backend, &matrix, coefficients);
    let gram = backend.to_vec(&backend.matmul(&x, &backend.transpose(&x, 0, 1)));
    let mut largest = 0.0f32;
    for row in 0..ROWS {
        for col in 0..ROWS {
            let expected = if row == col { 1.0 } else { 0.0 };
            largest = largest.max((gram[row * ROWS + col] - expected).abs());
        }
    }
    (backend.to_vec(&x), largest)
}

#[test]
fn orthogonalization_matches_cpu() {
    // Dix itérations de produits matriciels : les écarts d'arrondi
    // s'accumulent, d'où une tolérance plus large que pour un kernel seul.
    // Les quatre itérations courtes visent une erreur de 0,007 par valeur
    // singulière, donc un peu plus sur `X·Xᵀ`.
    for (coefficients, tolerance) in [(&NEWTON_SCHULZ[..], 1e-2), (&NEWTON_SCHULZ_SHORT[..], 3e-2)]
    {
        let (cpu, cpu_distance) = orthogonalized(&Cpu::<f32>::new(), coefficients);
        let (gpu, gpu_distance) = orthogonalized(&Cuda::new(0).unwrap(), coefficients);
        assert!(cpu_distance < tolerance, "CPU : écart {cpu_distance}");
        assert!(gpu_distance < tolerance, "GPU : écart {gpu_distance}");
        for (index, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            assert!((c - g).abs() < 1e-3, "élément {index} : CPU {c}, GPU {g}");
        }
    }
}

#[test]
fn orthogonalization_survives_mixed_precision() {
    // Sur Tensor Cores, les produits se font en `bf16`, qui ne garde que
    // trois chiffres : le résultat reste orthogonal à quelques centièmes
    // près, ce qui suffit à une direction de descente.
    let mixed = smallem_cuda::gpu(true).unwrap();
    let (_, distance) = orthogonalized(&mixed, &NEWTON_SCHULZ);
    assert!(distance < 0.1, "écart {distance}");
}

/// La courbe de perte du MLP entraîné avec Muon.
fn muon_loss_curve<B: Backend>(backend: &B, steps: usize, variant: MuonVariant) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(5);
    let mut model = Mlp::new(backend, CONFIG, &mut rng);
    let mut optimizer = Muon::with_variant(1e-2, 0.1, variant);
    // Au-delà de la montée : l'amplification de SAMuon-lite est entière.
    optimizer.set_progress(0.5);
    let dataset = dataset();
    (0..steps)
        .map(|_| {
            let batch = dataset.sample_batch(&mut rng, 64, CONFIG.context);
            let (contexts, targets) = next_token_examples(&batch);
            let loss = train_step(backend, &mut model, &mut optimizer, |m, t| {
                m.loss(t, &contexts, &targets)
            });
            backend.item(&loss).to_f64()
        })
        .collect()
}

#[test]
fn muon_follows_the_cpu_loss_curve() {
    const STEPS: usize = 30;
    let variants = [
        ("muon", MuonVariant::default()),
        (
            "court et périodique",
            MuonVariant {
                short: true,
                period: 3,
                ..MuonVariant::default()
            },
        ),
        (
            "par neurone",
            MuonVariant {
                neuron_norm: true,
                ..MuonVariant::default()
            },
        ),
        (
            "amplifié",
            MuonVariant {
                tail_boost: 3.0,
                ..MuonVariant::default()
            },
        ),
    ];
    for (name, variant) in variants {
        let cpu = muon_loss_curve(&Cpu::<f32>::new(), STEPS, variant);
        let gpu = muon_loss_curve(&Cuda::new(0).unwrap(), STEPS, variant);
        // La variante périodique avance moins par pas : deux pas sur trois
        // sont des pas par lignes, sept fois plus courts.
        assert!(
            cpu[STEPS - 1] < cpu[0] - 0.3,
            "{name} : le CPU n'apprend pas : {cpu:?}"
        );
        for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            assert!(
                (c - g).abs() < 5e-3,
                "{name}, pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}"
            );
        }
    }
}
