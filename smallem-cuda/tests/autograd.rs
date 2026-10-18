//! Les gradients calculés sur GPU sont comparés à ceux du CPU en `f32` sur
//! le même calcul. Les règles de rétropropagation sont déjà vérifiées par
//! différences finies dans `src/autograd.rs` ; ces tests valident les
//! kernels que la rétropropagation utilise.
//!
//! `cargo oxide test -- --test autograd`

mod common;

use common::Shortcuts;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::{Tape, Var};
use smallem::backend::cpu::Cpu;
use smallem::backend::{Backend, Primitives, standard_normal};
use smallem_cuda::Cuda;

/// Plus large que pour une opération seule : un gradient traverse une
/// dizaine d'opérations, et les erreurs d'arrondi se cumulent.
const TOLERANCE: f32 = 1e-4;

type Host = (Vec<f32>, Vec<usize>);

fn random(shapes: &[&[usize]]) -> Vec<Host> {
    let mut rng = StdRng::seed_from_u64(11);
    shapes
        .iter()
        .map(|shape| {
            let len: usize = shape.iter().product();
            let data = (0..len).map(|_| standard_normal(&mut rng) as f32).collect();
            (data, shape.to_vec())
        })
        .collect()
}

fn assert_close(what: &str, cpu: &Host, gpu: &Host) {
    assert_eq!(cpu.1, gpu.1, "{what} : formes différentes");
    for (index, (&c, &g)) in cpu.0.iter().zip(&gpu.0).enumerate() {
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!(
            (c - g).abs() <= TOLERANCE * scale,
            "{what}, élément {index} : CPU {c}, GPU {g}",
        );
    }
}

fn to_host<B: Backend<Scalar = f32>>(backend: &B, tensor: &B::Tensor) -> Host {
    (backend.to_vec(tensor), backend.shape(tensor).to_vec())
}

/// Construit la perte avec `build`, rétropropage, et rapatrie la perte
/// suivie du gradient de chaque entrée.
fn loss_and_gradients<B: Backend<Scalar = f32>>(
    backend: &B,
    inputs: &[Host],
    build: impl Fn(&Tape<B>, &[Var]) -> Var,
) -> Vec<Host> {
    let tape = Tape::new(backend);
    let vars: Vec<Var> = inputs
        .iter()
        .map(|(data, shape)| tape.leaf(backend.from_vec(data.clone(), shape)))
        .collect();
    let loss = build(&tape, &vars);
    let grads = tape.backward(loss);

    let mut results = vec![to_host(backend, &tape.value(loss))];
    for var in &vars {
        results.push(to_host(backend, grads.get(*var).expect("gradient absent")));
    }
    results
}

/// Compare perte et gradients entre CPU et GPU. Le corps est écrit une
/// fois et compilé deux fois, une par backend.
macro_rules! check {
    ($shapes:expr, |$t:ident, $v:ident| $build:expr) => {{
        let inputs = random($shapes);
        let cpu = loss_and_gradients(&Cpu::<f32>::new(), &inputs, |$t, $v| $build);
        let gpu = loss_and_gradients(&Cuda::new(0).unwrap(), &inputs, |$t, $v| $build);
        assert_close("perte", &cpu[0], &gpu[0]);
        for index in 1..cpu.len() {
            assert_close(&format!("gradient {}", index - 1), &cpu[index], &gpu[index]);
        }
    }};
}

// --- Les opérations propres à la rétropropagation ---

#[test]
fn expand() {
    let inputs = random(&[&[7, 1, 30]]);
    let run = |host: &Host| {
        let (cpu, gpu) = (Cpu::<f32>::new(), Cuda::new(0).unwrap());
        let on_cpu = cpu.expand(&cpu.from_vec(host.0.clone(), &host.1), &[4, 7, 9, 30]);
        let on_gpu = gpu.expand(&gpu.from_vec(host.0.clone(), &host.1), &[4, 7, 9, 30]);
        assert_close("expand", &to_host(&cpu, &on_cpu), &to_host(&gpu, &on_gpu));
    };
    run(&inputs[0]);
}

#[test]
fn scatter_add_rows_with_heavy_collisions() {
    // 3 000 lignes dispersées dans seulement 5 : chaque case de la sortie
    // reçoit environ 600 additions concurrentes. Sans addition atomique,
    // la plupart seraient perdues.
    const ROWS: usize = 3000;
    let inputs = random(&[&[ROWS, 48]]);
    let indices: Vec<u32> = (0..ROWS as u32).map(|i| (i * 7) % 5).collect();

    let (cpu, gpu) = (Cpu::<f32>::new(), Cuda::new(0).unwrap());
    let (data, shape) = &inputs[0];
    let on_cpu = cpu.scatter_add_rows(&cpu.from_vec(data.clone(), shape), &indices, 5);
    let on_gpu = gpu.scatter_add_rows(&gpu.from_vec(data.clone(), shape), &indices, 5);
    assert_close(
        "scatter_add_rows",
        &to_host(&cpu, &on_cpu),
        &to_host(&gpu, &on_gpu),
    );
}

#[test]
fn scatter_add_rows_leaves_untouched_rows_at_zero() {
    let gpu = Cuda::new(0).unwrap();
    let rows = gpu.ones(&[2, 3]);
    let scattered = gpu.scatter_add_rows(&rows, &[4, 4], 6);
    let mut expected = vec![0.0; 6 * 3];
    expected[4 * 3..5 * 3].fill(2.0);
    assert_eq!(gpu.to_vec(&scattered), expected);
}

#[test]
fn cross_entropy_loss_and_gradient() {
    const N: usize = 300;
    const VOCAB: usize = 257;
    let inputs = random(&[&[N, VOCAB]]);
    let targets: Vec<u32> = (0..N as u32).map(|i| (i * 31) % VOCAB as u32).collect();

    let (cpu, gpu) = (Cpu::<f32>::new(), Cuda::new(0).unwrap());
    let (data, shape) = &inputs[0];
    let (cpu_logits, gpu_logits) = (
        cpu.from_vec(data.clone(), shape),
        gpu.from_vec(data.clone(), shape),
    );
    assert_close(
        "cross_entropy",
        &to_host(&cpu, &cpu.cross_entropy(&cpu_logits, &targets)),
        &to_host(&gpu, &gpu.cross_entropy(&gpu_logits, &targets)),
    );
    assert_close(
        "cross_entropy_grad",
        &to_host(&cpu, &cpu.cross_entropy_grad(&cpu_logits, &targets)),
        &to_host(&gpu, &gpu.cross_entropy_grad(&gpu_logits, &targets)),
    );
}

#[test]
fn cross_entropy_of_uniform_logits_is_ln_vocab() {
    let gpu = Cuda::new(0).unwrap();
    let loss = gpu.cross_entropy(&gpu.zeros(&[4, 257]), &[0, 1, 2, 256]);
    assert!((gpu.item(&loss) - 257f32.ln()).abs() < 1e-5);
}

// --- Rétropropagation complète ---

#[test]
fn mean_of_tanh_of_affine() {
    // mean(tanh(A·x + b)), l'exemple du plan.
    check!(&[&[30, 40], &[40, 20], &[30, 1]], |t, v| {
        let affine = t.add(t.matmul(v[0], v[1]), v[2]);
        t.sum_all(t.mean(t.tanh(affine), 1, false))
    });
}

#[test]
fn elementwise_chain_with_broadcasting() {
    check!(&[&[9, 31], &[31], &[9, 1]], |t, v| {
        let positive = t.add(t.mul(v[0], v[0]), t.scalar(0.5));
        let mixed = t.div(t.exp(v[1]), t.sqrt(positive));
        t.sum_all(t.mul(t.ln(positive), t.sub(mixed, v[2])))
    });
}

#[test]
fn layer_norm() {
    check!(&[&[4, 9, 32], &[32], &[32]], |t, v| {
        let centered = t.sub(v[0], t.mean(v[0], 2, true));
        let variance = t.mean(t.mul(centered, centered), 2, true);
        let std = t.sqrt(t.add(variance, t.scalar(1e-5)));
        let normalized = t.add(t.mul(t.div(centered, std), v[1]), v[2]);
        t.sum_all(t.mul(normalized, normalized))
    });
}

#[test]
fn causal_self_attention() {
    // Une tête d'attention complète : q · kᵀ / √d, masque causal, softmax,
    // puis poids · v.
    const T: usize = 9;
    const D: usize = 16;
    check!(&[&[3, T, D], &[D, D], &[D, D], &[D, D]], |t, v| {
        // Des poids d'écart-type 1 donneraient des scores énormes et un
        // softmax saturé, dont le gradient est une différence de grands
        // nombres presque égaux : CPU et GPU s'écarteraient par simple
        // arrondi. On les réduit, comme le fera l'initialisation du modèle.
        let small = |weights: Var| t.mul(weights, t.scalar(0.25));
        let q = t.matmul(v[0], small(v[1]));
        let k = t.matmul(v[0], small(v[2]));
        let values = t.matmul(v[0], small(v[3]));
        let scores = t.div(
            t.matmul(q, t.transpose(k, 1, 2)),
            t.scalar((D as f32).sqrt()),
        );

        let mut upper = vec![0.0; T * T];
        for row in 0..T {
            for col in row + 1..T {
                upper[row * T + col] = 1.0;
            }
        }
        let mask = t.constant(t.backend().from_vec(upper, &[T, T]));
        let weights = t.softmax(t.masked_fill(scores, mask, f32::NEG_INFINITY));
        let out = t.matmul(weights, values);
        t.sum_all(t.mul(out, out))
    });
}

#[test]
fn language_model() {
    // Embedding → couche cachée → logits → entropie croisée, sur un lot de
    // 512 tokens tirés d'un vocabulaire de 257 : beaucoup de répétitions,
    // donc beaucoup de collisions dans le scatter-add.
    const VOCAB: usize = 257;
    const N: usize = 512;
    check!(
        &[&[VOCAB, 24], &[24, 48], &[48], &[48, VOCAB], &[VOCAB]],
        |t, v| {
            let tokens: Vec<u32> = (0..N as u32).map(|i| (i * i + 3) % 40).collect();
            let targets: Vec<u32> = (0..N as u32).map(|i| (i * 17 + 1) % VOCAB as u32).collect();
            let embedded = t.gather_rows(v[0], &tokens);
            let hidden = t.tanh(t.add(t.matmul(embedded, v[1]), v[2]));
            let logits = t.add(t.matmul(hidden, v[3]), v[4]);
            t.cross_entropy(logits, &targets)
        }
    );
}
