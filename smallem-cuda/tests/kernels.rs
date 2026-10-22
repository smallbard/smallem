//! Chaque opération du backend CUDA est comparée à celle du backend CPU sur
//! les mêmes entrées aléatoires.
//!
//! Ces tests ont besoin d'un GPU et des kernels compilés :
//! `cargo oxide test -- --test kernels`

mod common;

use common::Shortcuts;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::backend::cpu::Cpu;
use smallem::backend::{AdamWStep, Backend, Primitives, standard_normal};
use smallem_cuda::Cuda;

/// Écart relatif toléré entre CPU et GPU. Les deux calculent en `f32`, mais
/// pas dans le même ordre ni avec les mêmes instructions (le GPU fusionne
/// multiplication et addition), donc pas au bit près.
const TOLERANCE: f32 = 1e-5;

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

/// Exécute `op` sur un backend et rapatrie le résultat.
fn eval<B: Backend<Scalar = f32>>(
    backend: &B,
    inputs: &[Input],
    op: impl Fn(&B, &[B::Tensor]) -> B::Tensor,
) -> Input {
    let tensors: Vec<B::Tensor> = inputs
        .iter()
        .map(|(data, shape)| backend.from_vec(data.clone(), shape))
        .collect();
    let result = op(backend, &tensors);
    (backend.to_vec(&result), backend.shape(&result).to_vec())
}

fn assert_close(cpu: &Input, gpu: &Input) {
    assert_eq!(cpu.1, gpu.1, "formes différentes");
    for (index, (&c, &g)) in cpu.0.iter().zip(&gpu.0).enumerate() {
        if c == g {
            continue; // couvre aussi les infinis de même signe
        }
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!(
            (c - g).abs() <= TOLERANCE * scale,
            "élément {index} : CPU {c}, GPU {g}",
        );
    }
}

/// Compare les deux backends sur une opération. Le corps est écrit une
/// fois et compilé deux fois, une par backend.
macro_rules! check {
    ($shapes:expr, |$b:ident, $t:ident| $op:expr) => {{
        let inputs = random_inputs($shapes, 0);
        let cpu = eval(&Cpu::<f32>::new(), &inputs, |$b, $t| $op);
        let gpu = eval(&Cuda::new(0).unwrap(), &inputs, |$b, $t| $op);
        assert_close(&cpu, &gpu);
    }};
}

// 7 × 300 = 2100 éléments : plusieurs blocs de 256 threads, dont un dernier
// incomplet.
const MATRIX: &[usize] = &[7, 300];

// --- Élément par élément ---

#[test]
fn add() {
    check!(&[MATRIX, MATRIX], |b, t| b.add(&t[0], &t[1]));
}

#[test]
fn sub() {
    check!(&[MATRIX, MATRIX], |b, t| b.sub(&t[0], &t[1]));
}

#[test]
fn mul() {
    check!(&[MATRIX, MATRIX], |b, t| b.mul(&t[0], &t[1]));
}

#[test]
fn div() {
    check!(&[MATRIX, MATRIX], |b, t| b.div(&t[0], &t[1]));
}

#[test]
fn neg() {
    check!(&[MATRIX], |b, t| b.neg(&t[0]));
}

#[test]
fn exp() {
    check!(&[MATRIX], |b, t| b.exp(&t[0]));
}

#[test]
fn tanh() {
    check!(&[MATRIX], |b, t| b.tanh(&t[0]));
}

#[test]
fn ln_and_sqrt_of_positive_values() {
    // x² + 0,1 est strictement positif.
    check!(&[MATRIX], |b, t| {
        let positive = b.add(&b.mul(&t[0], &t[0]), &b.scalar(0.1));
        b.add(&b.ln(&positive), &b.sqrt(&positive))
    });
}

#[test]
fn single_element() {
    check!(&[&[1]], |b, t| b.exp(&t[0]));
}

// --- Broadcasting ---

#[test]
fn broadcast_row() {
    check!(&[MATRIX, &[300]], |b, t| b.add(&t[0], &t[1]));
}

#[test]
fn broadcast_column() {
    check!(&[MATRIX, &[7, 1]], |b, t| b.mul(&t[0], &t[1]));
}

#[test]
fn broadcast_both_sides() {
    check!(&[&[7, 1], &[1, 300]], |b, t| b.sub(&t[0], &t[1]));
}

#[test]
fn broadcast_scalar() {
    check!(&[MATRIX], |b, t| b.mul(&t[0], &b.scalar(2.5)));
}

#[test]
fn broadcast_four_dimensions() {
    // Positions [T, C] ajoutées à des activations [B, H, T, C].
    check!(&[&[2, 3, 5, 7], &[5, 7]], |b, t| b.add(&t[0], &t[1]));
}

#[test]
fn broadcast_left_operand() {
    check!(&[&[300], MATRIX], |b, t| b.div(&t[0], &t[1]));
}

// --- Réductions ---

#[test]
fn sum_first_axis() {
    check!(&[MATRIX], |b, t| b.sum(&t[0], 0, false));
}

#[test]
fn sum_last_axis() {
    check!(&[MATRIX], |b, t| b.sum(&t[0], 1, false));
}

#[test]
fn sum_middle_axis_keepdim() {
    check!(&[&[4, 9, 11]], |b, t| b.sum(&t[0], 1, true));
}

#[test]
fn mean_then_broadcast_back() {
    // Le centrage de LayerNorm : x - moyenne(x) sur le dernier axe.
    check!(&[&[4, 9, 11]], |b, t| b.sub(&t[0], &b.mean(&t[0], 2, true)));
}

#[test]
fn sum_all_small() {
    check!(&[&[5]], |b, t| b.sum_all(&t[0]));
}

#[test]
fn sum_all_single_pass() {
    check!(&[&[200]], |b, t| b.sum_all(&t[0]));
}

#[test]
fn sum_all_several_passes() {
    // 100 000 éléments : trois passes, avec des paquets incomplets.
    //
    // La référence est calculée en f64 et non par le backend CPU en f32 :
    // additionner 100 000 termes un par un en f32 accumule une erreur
    // d'arrondi supérieure à la tolérance. La somme par paquets du GPU,
    // elle, additionne des nombres de tailles comparables et reste précise.
    let (values, shape) = random_inputs(&[&[100, 1000]], 0).remove(0);
    let squares: Vec<f32> = values.iter().map(|x| x * x).collect();
    let expected: f64 = squares.iter().map(|&x| x as f64).sum();

    let gpu = Cuda::new(0).unwrap();
    let total = gpu.sum_all(&gpu.from_vec(squares, &shape));
    assert_eq!(gpu.shape(&total), &[] as &[usize]);
    let actual = gpu.item(&total) as f64;
    assert!(
        (actual - expected).abs() <= TOLERANCE as f64 * expected,
        "GPU {actual}, attendu {expected}",
    );
}

#[test]
fn sum_all_of_one_element() {
    check!(&[&[1, 1]], |b, t| b.sum_all(&t[0]));
}

// --- Changements de forme ---

#[test]
fn reshape() {
    check!(&[MATRIX], |b, t| b.reshape(&t[0], &[300, 7]));
}

#[test]
fn reshape_shares_data_with_later_operations() {
    check!(&[MATRIX], |b, t| {
        let reshaped = b.reshape(&t[0], &[2, 1050]);
        b.add(&reshaped, &reshaped)
    });
}

#[test]
fn transpose_matrix() {
    check!(&[MATRIX], |b, t| b.transpose(&t[0], 0, 1));
}

#[test]
fn transpose_inner_axes() {
    // [B, T, H, D] → [B, H, T, D], la séparation des têtes d'attention.
    check!(&[&[2, 5, 3, 7]], |b, t| b.transpose(&t[0], 1, 2));
}

#[test]
fn transpose_outer_axes() {
    check!(&[&[4, 9, 11]], |b, t| b.transpose(&t[0], 0, 2));
}

// --- Matmul ---

#[test]
fn matmul_two_matrices() {
    check!(&[&[17, 33], &[33, 21]], |b, t| b.matmul(&t[0], &t[1]));
}

#[test]
fn matmul_batch_times_matrix() {
    // [B, T, C] × [C, D] : une couche linéaire.
    check!(&[&[4, 9, 33], &[33, 21]], |b, t| b.matmul(&t[0], &t[1]));
}

#[test]
fn matmul_batch_times_batch() {
    // [B, H, T, D] × [B, H, D, T] : les scores d'attention q · kᵀ.
    check!(&[&[2, 3, 9, 8], &[2, 3, 8, 9]], |b, t| b
        .matmul(&t[0], &t[1]));
}

#[test]
fn matmul_matrix_times_batch() {
    check!(&[&[9, 8], &[5, 8, 6]], |b, t| b.matmul(&t[0], &t[1]));
}

#[test]
fn matmul_spanning_several_tiles() {
    // Plus d'une tuile de 64 dans chaque direction, et aucune dimension
    // multiple de 64 ni de 16 : tous les bords sont incomplets.
    check!(&[&[130, 70], &[70, 150]], |b, t| b.matmul(&t[0], &t[1]));
    check!(&[&[3, 100, 37], &[37, 129]], |b, t| b.matmul(&t[0], &t[1]));
    check!(&[&[2, 2, 65, 17], &[2, 2, 17, 65]], |b, t| b
        .matmul(&t[0], &t[1]));
}

/// Comme `check!`, pour un produit de deux tenseurs : l'écart est rapporté
/// à `|a| · |b|`, l'ordre de grandeur des termes additionnés.
///
/// Les sommes du GPU ne se font pas dans le même ordre que sur CPU. Sur
/// des centaines de termes, des résultats proches de zéro peuvent alors
/// différer de plus de 1e-5 en relatif, sans que rien ne soit faux.
macro_rules! check_product {
    ($shapes:expr, |$b:ident, $t:ident| $op:expr) => {{ check_product_of!(random_inputs($shapes, 0), |$b, $t| $op) }};
}

/// Comme `check_product!`, sur des entrées données.
macro_rules! check_product_of {
    ($inputs:expr, |$b:ident, $t:ident| $op:expr) => {
        check_product_of!(Cuda::new(0).unwrap(); $inputs, |$b, $t| $op)
    };
    ($gpu:expr; $inputs:expr, |$b:ident, $t:ident| $op:expr) => {{
        let inputs: Vec<Input> = $inputs;
        let absolute: Vec<Input> = inputs
            .iter()
            .map(|(data, shape)| (data.iter().map(|v| v.abs()).collect(), shape.clone()))
            .collect();
        let cpu = eval(&Cpu::<f32>::new(), &inputs, |$b, $t| $op);
        let bound = eval(&Cpu::<f32>::new(), &absolute, |$b, $t| $op);
        let gpu = eval(&$gpu, &inputs, |$b, $t| $op);
        assert_eq!(cpu.1, gpu.1, "formes différentes");
        let shapes: Vec<&Vec<usize>> = inputs.iter().map(|(_, shape)| shape).collect();
        for ((&c, &g), &scale) in cpu.0.iter().zip(&gpu.0).zip(&bound.0) {
            assert!(
                (c - g).abs() <= TOLERANCE * scale,
                "{shapes:?} : CPU {c}, GPU {g}",
            );
        }
    }};
}

#[test]
fn matmul_of_few_rows() {
    // Au plus 16 lignes : le kernel des matrices étroites. Un vecteur face
    // à une couche de 512, découpé en tranches le long de la dimension
    // commune ; une dimension commune qui ne tombe juste sur aucune
    // tranche ; 16 et 9 lignes, soit deux paquets de 8 dont un incomplet ;
    // des lots des deux côtés.
    let shapes: &[(&[usize], &[usize])] = &[
        (&[1, 512], &[512, 512]),
        (&[5, 1000], &[1000, 70]),
        (&[16, 33], &[33, 21]),
        (&[2, 3, 9, 37], &[37, 41]),
        (&[11, 300], &[2, 300, 45]),
        (&[2, 4, 1, 64], &[2, 4, 64, 100]),
    ];
    for &(shape_a, shape_b) in shapes {
        check_product!(&[shape_a, shape_b], |b, t| b.matmul(&t[0], &t[1]));
    }
}

#[test]
fn matmul_of_a_batch_of_tokens() {
    // Entre 17 et 256 lignes, une grille de tuiles trop petite : la
    // dimension commune est découpée en tranches.
    check_product!(&[&[40, 1368], &[1368, 512]], |b, t| b.matmul(&t[0], &t[1]));
    check_product!(&[&[2, 100, 300][..], &[300, 40]], |b, t| b
        .matmul(&t[0], &t[1]));
}

#[test]
fn affine_of_few_and_many_rows() {
    use smallem::backend::ops::{Affine, Run};
    // Un token, puis neuf (deux paquets de 8), avec une dimension commune
    // découpée en tranches ou non ; puis vingt lignes, qui passent par le
    // produit puis l'addition.
    for (shape_x, inner) in [
        (&[1, 1, 512][..], 512),
        (&[9, 40][..], 40),
        (&[4, 5, 300][..], 300),
        (&[64, 1368][..], 1368),
        (&[300, 50][..], 50),
    ] {
        check_product!(&[shape_x, &[inner, 70], &[70]], |b, t| b
            .run(&Affine, &[&t[0], &t[1], &t[2]]));
    }
}

#[test]
fn transposed_matmul_of_few_and_many_rows() {
    use smallem::backend::ops::{Run, TransposedMatmul};
    // Une ligne face à une table de 1000, neuf lignes (deux paquets de 8),
    // puis vingt, qui passent par la transposée et le produit ordinaire.
    for shape_a in [&[1, 1, 300][..], &[9, 300], &[4, 5, 300]] {
        check_product!(&[shape_a, &[1000, 300]], |b, t| b
            .run(&TransposedMatmul, &[&t[0], &t[1]]));
    }
}

// --- Matmul sur Tensor Cores (précision mixte) ---

/// Les formes sur lesquelles la matmul en précision mixte est vérifiée :
/// petites, à cheval sur plusieurs tuiles, et avec des lots des deux côtés.
const MIXED_SHAPES: &[(&[usize], &[usize])] = &[
    (&[1, 1], &[1, 1]),
    (&[17, 33], &[33, 21]),
    (&[64, 16], &[16, 64]),
    (&[130, 70], &[70, 150]),
    (&[3, 100, 37], &[37, 129]),
    (&[2, 3, 9, 8], &[2, 3, 8, 9]),
    (&[2, 2, 65, 17], &[2, 2, 17, 65]),
    (&[9, 8], &[5, 8, 6]),
    // Des tuiles 128 × 128 entières, donc le chemin de chargement sans
    // test, avec ce que l'entraînement lui fait subir : un lot de plusieurs
    // matrices face à une matrice partagée (une couche linéaire), une
    // dimension commune impaire, un lot des deux côtés, et le lot à droite.
    (&[3, 130, 33], &[33, 150]),
    (&[2, 256, 48], &[48, 256]),
    (&[2, 2, 130, 20], &[2, 2, 20, 140]),
    (&[140, 17], &[3, 17, 130]),
    // Peu de lignes face à une grande dimension commune : la génération
    // par lots, où la dimension commune est découpée en tranches.
    (&[40, 1368], &[1368, 512]),
    (&[2, 100, 300], &[300, 40]),
];

#[test]
fn mixed_matmul_is_exact_on_small_integers() {
    // Des entiers entre -8 et 8 sont exacts en bf16, et les sommes restent
    // exactes en f32 : le résultat doit être celui du CPU, au bit près.
    // Une erreur de disposition des fragments donnerait d'autres nombres.
    let gpu = Cuda::new(0).unwrap().with_mixed_precision();
    let cpu = Cpu::<f32>::new();
    for (index, &(shape_a, shape_b)) in MIXED_SHAPES.iter().enumerate() {
        let mut inputs = random_inputs(&[shape_a, shape_b], index as u64);
        for (data, _) in &mut inputs {
            for value in data {
                *value = (*value * 4.0).round().clamp(-8.0, 8.0);
            }
        }
        let expected = eval(&cpu, &inputs, |b, t| b.matmul(&t[0], &t[1]));
        let actual = eval(&gpu, &inputs, |b, t| b.matmul(&t[0], &t[1]));
        assert_eq!(expected, actual, "formes {shape_a:?} × {shape_b:?}");
    }
}

#[test]
fn mixed_matmul_is_close_on_real_values() {
    // Chaque opérande est arrondi à 8 bits de mantisse, soit une erreur
    // relative d'au plus 2⁻⁹. L'erreur d'un élément du produit est donc
    // bornée par environ 2⁻⁸ × Σ |a| × |b|, qu'on calcule sur CPU.
    let gpu = Cuda::new(0).unwrap().with_mixed_precision();
    let cpu = Cpu::<f32>::new();
    let mut largest_error = 0.0f32;
    for (index, &(shape_a, shape_b)) in MIXED_SHAPES.iter().enumerate() {
        let inputs = random_inputs(&[shape_a, shape_b], 100 + index as u64);
        let absolute: Vec<Input> = inputs
            .iter()
            .map(|(data, shape)| (data.iter().map(|v| v.abs()).collect(), shape.clone()))
            .collect();
        let expected = eval(&cpu, &inputs, |b, t| b.matmul(&t[0], &t[1]));
        let bound = eval(&cpu, &absolute, |b, t| b.matmul(&t[0], &t[1]));
        let actual = eval(&gpu, &inputs, |b, t| b.matmul(&t[0], &t[1]));
        assert_eq!(expected.1, actual.1);
        for ((&e, &a), &scale) in expected.0.iter().zip(&actual.0).zip(&bound.0) {
            let error = (e - a).abs();
            assert!(
                error <= scale / 256.0 + 1e-6,
                "{shape_a:?} × {shape_b:?} : CPU {e}, GPU {a}, borne {}",
                scale / 256.0,
            );
            largest_error = largest_error.max(error / scale.max(1e-6));
        }
    }
    // La précision mixte doit vraiment arrondir : une erreur nulle voudrait
    // dire que ce test n'a pas exercé le bon kernel.
    assert!(largest_error > 1e-5, "erreur maximale {largest_error}");
}

#[test]
fn mixed_precision_only_changes_matmul() {
    // Toutes les autres opérations restent en f32.
    let gpu = Cuda::new(0).unwrap().with_mixed_precision();
    let inputs = random_inputs(&[MATRIX, MATRIX], 0);
    let op = |b: &Cuda, t: &[<Cuda as Primitives>::Tensor]| b.gelu(&b.mul(&t[0], &t[1]));
    let exact = eval(&Cuda::new(0).unwrap(), &inputs, op);
    assert_eq!(exact, eval(&gpu, &inputs, op));
}

// --- Softmax ---

#[test]
fn softmax() {
    check!(&[MATRIX], |b, t| b.softmax(&t[0]));
}

#[test]
fn log_softmax() {
    check!(&[MATRIX], |b, t| b.log_softmax(&t[0]));
}

#[test]
fn softmax_of_large_values() {
    check!(&[MATRIX], |b, t| b.softmax(&b.mul(&t[0], &b.scalar(500.0))));
}

#[test]
fn softmax_four_dimensions() {
    check!(&[&[2, 3, 9, 9]], |b, t| b.softmax(&t[0]));
}

// --- Sélection ---

#[test]
fn masked_fill() {
    // Le masque est non nul là où le tirage aléatoire est positif.
    check!(&[MATRIX, MATRIX], |b, t| {
        let mask = b.add(&t[1], &b.scalar(0.0));
        b.masked_fill(&t[0], &mask, -7.0)
    });
}

#[test]
fn causal_mask_then_softmax() {
    // Masque triangulaire [T, T] étiré sur des scores [B, H, T, T], rempli
    // de -∞, puis softmax : l'enchaînement exact de l'attention.
    const T: usize = 9;
    check!(&[&[2, 3, T, T]], |b, t| {
        let mut upper = vec![0.0; T * T];
        for row in 0..T {
            for col in row + 1..T {
                upper[row * T + col] = 1.0;
            }
        }
        let mask = b.from_vec(upper, &[T, T]);
        b.softmax(&b.masked_fill(&t[0], &mask, f32::NEG_INFINITY))
    });
}

#[test]
fn gather_rows() {
    // Table d'embeddings [V, C], indices répétés et dans le désordre.
    check!(&[&[257, 48]], |b, t| {
        let indices: Vec<u32> = (0..500).map(|i| (i * 37 + 256) % 257).collect();
        b.gather_rows(&t[0], &indices)
    });
}

#[test]
fn gather_rows_then_reshape() {
    // [B × T] indices → [B, T, C], la forme d'entrée du modèle.
    check!(&[&[257, 48]], |b, t| {
        let indices: Vec<u32> = (0..24).map(|i| (i * 5) % 257).collect();
        b.reshape(&b.gather_rows(&t[0], &indices), &[4, 6, 48])
    });
}

// --- Opérations fusionnées ---

#[test]
fn gelu() {
    check!(&[MATRIX], |b, t| b.gelu(&b.mul(&t[0], &b.scalar(3.0))));
}

#[test]
fn gelu_backward() {
    check!(&[MATRIX, MATRIX], |b, t| b.gelu_backward(&t[0], &t[1]));
}

#[test]
fn normalize() {
    check!(&[&[4, 9, 33]], |b, t| b.normalize(&t[0], 1e-5));
}

#[test]
fn normalize_with_offset_and_scale() {
    check!(&[&[7, 300]], |b, t| {
        let shifted = b.add(&b.mul(&t[0], &b.scalar(40.0)), &b.scalar(-7.0));
        b.normalize(&shifted, 1e-5)
    });
}

#[test]
fn normalize_backward() {
    check!(&[&[4, 9, 33], &[4, 9, 33]], |b, t| {
        let normalized = b.normalize(&t[0], 1e-5);
        b.normalize_backward(&t[0], &normalized, &t[1], 1e-5)
    });
}

#[test]
fn silu() {
    check!(&[MATRIX], |b, t| b.silu(&b.mul(&t[0], &b.scalar(3.0))));
}

#[test]
fn silu_backward() {
    check!(&[MATRIX, MATRIX], |b, t| b.silu_backward(&t[0], &t[1]));
}

#[test]
fn rms_normalize() {
    check!(&[&[4, 9, 33]], |b, t| b.rms_normalize(&t[0], 1e-5));
    // Une ligne plus large qu'un warp, avec un décalage : la moyenne ne doit
    // pas être retirée.
    check!(&[&[7, 300]], |b, t| {
        let shifted = b.add(&b.mul(&t[0], &b.scalar(40.0)), &b.scalar(-7.0));
        b.rms_normalize(&shifted, 1e-5)
    });
}

#[test]
fn rms_normalize_backward() {
    check!(&[&[4, 9, 33], &[4, 9, 33]], |b, t| {
        let normalized = b.rms_normalize(&t[0], 1e-5);
        b.rms_normalize_backward(&t[0], &normalized, &t[1], 1e-5)
    });
}

/// Fait d'un tenseur d'angles quelconques ses tables de cosinus et de sinus.
fn tables<B: Backend<Scalar = f32>>(backend: &B, angles: &B::Tensor) -> (B::Tensor, B::Tensor) {
    let shape = backend.shape(angles).to_vec();
    let values = backend.to_vec(angles);
    let cos = values.iter().map(|a| a.cos()).collect();
    let sin = values.iter().map(|a| a.sin()).collect();
    (backend.from_vec(cos, &shape), backend.from_vec(sin, &shape))
}

#[test]
fn rotate_pairs() {
    // [B, H, T, D] comme dans l'attention, et plusieurs blocs de threads.
    check!(&[&[2, 3, 9, 16], &[9, 8]], |b, t| {
        let (cos, sin) = tables(b, &t[1]);
        b.rotate_pairs(&t[0], &cos, &sin, false)
    });
    check!(&[&[300, 6], &[300, 3]], |b, t| {
        let (cos, sin) = tables(b, &t[1]);
        b.rotate_pairs(&t[0], &cos, &sin, true)
    });
}

#[test]
fn sum_along_a_long_axis() {
    // Les lignes d'une matrice haute : le chemin par transposition.
    check!(&[&[700, 33]], |b, t| b.sum(&t[0], 0, false));
    check!(&[&[700, 33]], |b, t| b.sum(&t[0], 0, true));
    check!(&[&[3, 300, 5]], |b, t| b.sum(&t[0], 1, false));
    check!(&[&[3, 300, 5]], |b, t| b.sum(&t[0], 1, true));
    // Juste sous le seuil : le chemin ordinaire.
    check!(&[&[100, 33]], |b, t| b.sum(&t[0], 0, false));
}

#[test]
fn concat() {
    // [B, H, T, D] et [B, H, T, d_r] comme dans l'attention latente, puis
    // des lignes plus nombreuses que les threads d'un bloc.
    check!(&[&[2, 3, 9, 16], &[2, 3, 9, 6]], |b, t| b
        .concat(&t[0], &t[1]));
    check!(&[&[300, 5], &[300, 1]], |b, t| b.concat(&t[0], &t[1]));
    check!(&[&[7], &[300]], |b, t| b.concat(&t[0], &t[1]));
}

#[test]
fn concat_backward() {
    let shapes: &[&[usize]] = &[&[2, 3, 9, 16], &[2, 3, 9, 6], &[2, 3, 9, 22]];
    check!(shapes, |b, t| b.concat_backward(&t[0], &t[1], &t[2]).0);
    check!(shapes, |b, t| b.concat_backward(&t[0], &t[1], &t[2]).1);
}

#[test]
fn concat_can_be_undone_on_gpu() {
    let gpu = Cuda::new(0).unwrap();
    let inputs = random_inputs(&[&[300, 5], &[300, 3]], 6);
    let a = gpu.from_vec(inputs[0].0.clone(), &inputs[0].1);
    let b = gpu.from_vec(inputs[1].0.clone(), &inputs[1].1);
    let joined = gpu.concat(&a, &b);
    assert_eq!(gpu.shape(&joined), &[300, 8]);
    let (back_a, back_b) = gpu.concat_backward(&a, &b, &joined);
    assert_eq!(gpu.to_vec(&back_a), inputs[0].0);
    assert_eq!(gpu.to_vec(&back_b), inputs[1].0);
}

#[test]
fn rotate_pairs_can_be_undone_on_gpu() {
    let gpu = Cuda::new(0).unwrap();
    let inputs = random_inputs(&[&[2, 3, 9, 16], &[9, 8]], 5);
    let x = gpu.from_vec(inputs[0].0.clone(), &inputs[0].1);
    let (cos, sin) = tables(&gpu, &gpu.from_vec(inputs[1].0.clone(), &inputs[1].1));
    let rotated = gpu.rotate_pairs(&x, &cos, &sin, false);
    let back = gpu.to_vec(&gpu.rotate_pairs(&rotated, &cos, &sin, true));
    assert_ne!(gpu.to_vec(&rotated), inputs[0].0);
    for (b, x) in back.iter().zip(&inputs[0].0) {
        assert!((b - x).abs() < 1e-5, "{b} != {x}");
    }
}

#[test]
fn causal_softmax() {
    check!(&[&[2, 3, 9, 9]], |b, t| b.causal_softmax(&t[0], 0.35));
}

#[test]
fn causal_softmax_of_the_last_queries() {
    // Les requêtes ne sont que les dernières positions : trois sur neuf,
    // puis une seule, comme en génération.
    check!(&[&[2, 3, 3, 9]], |b, t| b.causal_softmax(&t[0], 0.35));
    check!(&[&[4, 1, 70]], |b, t| b.causal_softmax(&t[0], 0.35));
}

#[test]
fn softmax_backward_of_the_last_queries() {
    check!(&[&[2, 3, 3, 9], &[2, 3, 3, 9]], |b, t| {
        let probabilities = b.causal_softmax(&t[0], 0.35);
        b.softmax_backward(&probabilities, &t[1], 0.35)
    });
}

#[test]
fn causal_softmax_of_one_large_matrix() {
    // 300 lignes : plus d'un bloc de threads pour les réductions.
    check!(&[&[300, 300]], |b, t| b.causal_softmax(&t[0], 1.0));
}

#[test]
fn causal_softmax_matches_mask_then_softmax() {
    // Le kernel fusionné sur GPU contre la composition de primitives sur GPU.
    const T: usize = 9;
    let gpu = Cuda::new(0).unwrap();
    let (data, shape) = random_inputs(&[&[2, 3, T, T]], 0).remove(0);
    let scores = gpu.from_vec(data, &shape);

    let mut upper = vec![0.0; T * T];
    for row in 0..T {
        for col in row + 1..T {
            upper[row * T + col] = 1.0;
        }
    }
    let mask = gpu.from_vec(upper, &[T, T]);
    let scaled = gpu.mul(&scores, &gpu.scalar(0.35));
    let composed = gpu.softmax(&gpu.masked_fill(&scaled, &mask, f32::NEG_INFINITY));
    let fused = gpu.causal_softmax(&scores, 0.35);
    assert_close(
        &(gpu.to_vec(&composed), shape.clone()),
        &(gpu.to_vec(&fused), shape),
    );
}

#[test]
fn softmax_backward() {
    check!(&[&[2, 3, 9, 9], &[2, 3, 9, 9]], |b, t| {
        let probabilities = b.causal_softmax(&t[0], 0.35);
        b.softmax_backward(&probabilities, &t[1], 0.35)
    });
}

// --- Attention fusionnée ---

/// `q`, `k`, `v` et le gradient de la sortie pour `heads` têtes, `queries`
/// requêtes face à `positions` positions, des têtes de `key_dim` et
/// `value_dim`.
fn attention_shapes(
    heads: &[usize],
    queries: usize,
    positions: usize,
    key_dim: usize,
    value_dim: usize,
) -> Vec<Vec<usize>> {
    let with = |rows: usize, width: usize| [heads, &[rows, width]].concat();
    vec![
        with(queries, key_dim),
        with(positions, key_dim),
        with(positions, value_dim),
        with(queries, value_dim),
    ]
}

/// La sortie et les trois gradients de l'attention fusionnée, CPU contre
/// GPU, pour une forme donnée.
fn check_attention(
    heads: &[usize],
    queries: usize,
    positions: usize,
    key_dim: usize,
    value_dim: usize,
) {
    let shapes = attention_shapes(heads, queries, positions, key_dim, value_dim);
    let shapes: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    check!(&shapes, |b, t| b.attention(&t[0], &t[1], &t[2], 0.3));
    for input in 0..3 {
        check!(&shapes, |b, t| b
            .attention_backward(&t[0], &t[1], &t[2], &t[3], 0.3, input));
    }
}

#[test]
fn attention_of_a_whole_sequence() {
    // 70 positions : deux tuiles pleines et une incomplète, et des têtes
    // de la taille du modèle `xl`.
    check_attention(&[2, 3], 70, 70, 64, 64);
}

#[test]
fn attention_of_the_last_queries() {
    // Trois requêtes face à 45 positions, puis une seule : la génération.
    check_attention(&[2, 2], 3, 45, 16, 16);
    check_attention(&[4], 1, 70, 8, 8);
}

#[test]
fn attention_with_small_and_unequal_heads() {
    // Clés et valeurs de largeurs différentes, comme dans l'attention
    // latente, et qui ne remplissent pas les tuiles.
    check_attention(&[3], 33, 33, 5, 3);
}

#[test]
fn attention_with_wide_heads_uses_the_primitives() {
    // 96 composantes : au-delà des kernels, le calcul passe par la
    // définition de référence, et donne le même résultat.
    check_attention(&[2], 9, 9, 96, 64);
}

#[test]
fn mixed_attention_is_close() {
    // Sur les Tensor Cores, requêtes, clés, valeurs, poids et gradients des
    // scores sont arrondis en `bf16`, à 2⁻⁹ près : on ne retrouve le CPU qu'à
    // une erreur relative de l'ordre du pour cent, mesurée sur tout le
    // tenseur.
    let cpu = Cpu::<f32>::new();
    let gpu = Cuda::new(0).unwrap().with_mixed_precision();
    let mut largest_error = 0.0f32;
    for (index, &(heads, queries, positions, key_dim, value_dim)) in [
        (&[2usize, 3][..], 130, 130, 64, 64),
        (&[2, 2][..], 3, 70, 16, 16),
        (&[3][..], 1, 65, 8, 8),
        (&[2][..], 33, 33, 5, 3),
    ]
    .iter()
    .enumerate()
    {
        let shapes = attention_shapes(heads, queries, positions, key_dim, value_dim);
        let shapes: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let inputs = random_inputs(&shapes, 200 + index as u64);
        for output in 0..4 {
            let expected = if output == 0 {
                eval(&cpu, &inputs, |b, t| b.attention(&t[0], &t[1], &t[2], 0.3))
            } else {
                eval(&cpu, &inputs, |b, t| {
                    b.attention_backward(&t[0], &t[1], &t[2], &t[3], 0.3, output - 1)
                })
            };
            let actual = if output == 0 {
                eval(&gpu, &inputs, |b, t| b.attention(&t[0], &t[1], &t[2], 0.3))
            } else {
                eval(&gpu, &inputs, |b, t| {
                    b.attention_backward(&t[0], &t[1], &t[2], &t[3], 0.3, output - 1)
                })
            };
            assert_eq!(expected.1, actual.1);
            let largest = expected.0.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let error = expected
                .0
                .iter()
                .zip(&actual.0)
                .fold(0.0f32, |m, (e, a)| m.max((e - a).abs()))
                / largest;
            assert!(
                error < 2e-2,
                "{heads:?}, {queries} requêtes, {positions} positions, sortie {output} : \
                 erreur relative {error}",
            );
            largest_error = largest_error.max(error);
        }
    }
    // Une erreur nulle voudrait dire que ce test n'a pas exercé les kernels
    // en précision mixte.
    assert!(largest_error > 1e-4, "erreur maximale {largest_error}");
}

#[test]
fn fused_attention_matches_the_primitives_on_the_gpu() {
    // Les kernels contre la composition de primitives, sur la même carte.
    let shapes = attention_shapes(&[2, 2], 40, 40, 32, 32);
    let shapes: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let inputs = random_inputs(&shapes, 1);
    let fused = eval(&Cuda::new(0).unwrap(), &inputs, |b, t| {
        b.attention_backward(&t[0], &t[1], &t[2], &t[3], 0.3, 0)
    });
    let primitives = eval(
        &Cuda::new(0).unwrap().without_fused_attention(),
        &inputs,
        |b, t| b.attention_backward(&t[0], &t[1], &t[2], &t[3], 0.3, 0),
    );
    assert_close(&primitives, &fused);
}

#[test]
fn reused_buffers_do_not_leak_old_values() {
    // La réserve rend des tampons au contenu quelconque. On la remplit de
    // tampons pleins de 7, puis on vérifie qu'un scatter-add, qui suppose un
    // tampon nul, ne les voit pas.
    let gpu = Cuda::new(0).unwrap();
    for _ in 0..4 {
        drop(gpu.full(&[6, 3], 7.0));
    }
    let scattered = gpu.scatter_add_rows(&gpu.ones(&[2, 3]), &[4, 4], 6);
    let mut expected = vec![0.0; 6 * 3];
    expected[4 * 3..5 * 3].fill(2.0);
    assert_eq!(gpu.to_vec(&scattered), expected);
}

// --- Enchaînements ---

#[test]
fn linear_layer() {
    // x · W + biais
    check!(&[&[4, 9, 33], &[33, 21], &[21]], |b, t| {
        b.add(&b.matmul(&t[0], &t[1]), &t[2])
    });
}

#[test]
fn intermediate_results_are_freed_safely() {
    // Chaque résultat intermédiaire est libéré alors que le kernel suivant,
    // qui le lit, vient seulement d'être mis en file.
    check!(&[MATRIX], |b, t| {
        let mut x = b.tanh(&t[0]);
        for _ in 0..20 {
            x = b.tanh(&b.add(&x, &t[0]));
        }
        x
    });
}

#[test]
fn concat_rows() {
    check!(&[&[5, 3, 40], &[2, 3, 40]], |b, t| b
        .concat_rows(&t[0], &t[1]));
    check!(&[&[1, 7], &[300, 7]], |b, t| b.concat_rows(&t[0], &t[1]));
}

// --- Optimiseur ---

/// Trois pas d'AdamW sur un paramètre, avec des gradients différents à
/// chaque pas. Renvoie le paramètre et ses deux moments.
fn adamw_steps<B: Backend<Scalar = f32>>(
    backend: &B,
    weight_decay: f64,
    grad_scale: f64,
) -> [Vec<f32>; 3] {
    // Plusieurs blocs de 256 threads, dont un dernier incomplet.
    let inputs = random_inputs(&[&[1000], &[1000], &[1000], &[1000]], 21);
    let tensor = |index: usize| backend.from_vec(inputs[index].0.clone(), &[1000]);
    let mut param = tensor(0);
    let mut first_moment = backend.zeros(&[1000]);
    let mut second_moment = backend.zeros(&[1000]);
    for step in 1..=3 {
        let settings = AdamWStep {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay,
            bias_correction1: 1.0 - 0.9f64.powi(step),
            bias_correction2: 1.0 - 0.95f64.powi(step),
            grad_scale,
        };
        backend.adamw_step(
            &mut param,
            &tensor(step as usize),
            &mut first_moment,
            &mut second_moment,
            &settings,
        );
    }
    [param, first_moment, second_moment].map(|t| backend.to_vec(&t))
}

#[test]
fn adamw_step_matches_cpu() {
    let gpu = Cuda::new(0).unwrap();
    let cpu = Cpu::<f32>::new();
    for (weight_decay, grad_scale) in [(0.0, 1.0), (0.1, 1.0), (0.1, 0.25)] {
        let expected = adamw_steps(&cpu, weight_decay, grad_scale);
        let actual = adamw_steps(&gpu, weight_decay, grad_scale);
        for (name, (e, a)) in ["paramètre", "premier moment", "second moment"]
            .iter()
            .zip(expected.iter().zip(&actual))
        {
            for (index, (e, a)) in e.iter().zip(a).enumerate() {
                assert!(
                    (e - a).abs() <= 1e-6 * e.abs().max(1.0),
                    "{name} {index} (décroissance {weight_decay}, échelle {grad_scale}) : \
                     CPU {e}, GPU {a}",
                );
            }
        }
    }
}

#[test]
fn adamw_step_uses_every_setting() {
    // La comparaison avec le CPU ne vaut que si les réglages comptent : la
    // décroissance des poids et le facteur d'écrêtage doivent changer le
    // résultat bien au-delà de la tolérance de cette comparaison.
    let gpu = Cuda::new(0).unwrap();
    let largest_gap = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let [plain, _, plain_second] = adamw_steps(&gpu, 0.0, 1.0);
    let [decayed, _, _] = adamw_steps(&gpu, 0.1, 1.0);
    let [_, _, scaled_second] = adamw_steps(&gpu, 0.0, 0.25);
    assert!(largest_gap(&plain, &decayed) > 1e-4);
    assert!(largest_gap(&plain_second, &scaled_second) > 1e-4);
}

// --- Erreurs ---

#[test]
#[should_panic(expected = "formes incompatibles pour le broadcasting")]
fn incompatible_shapes_panic() {
    let gpu = Cuda::new(0).unwrap();
    gpu.add(&gpu.zeros(&[2, 3]), &gpu.zeros(&[2]));
}

#[test]
#[should_panic(expected = "au plus 4 dimensions")]
fn too_many_dimensions_panic() {
    let gpu = Cuda::new(0).unwrap();
    gpu.zeros(&[1, 2, 3, 4, 5]);
}

#[test]
#[should_panic(expected = "indice 3 hors de la première dimension (3)")]
fn gather_rows_out_of_bounds_panics() {
    let gpu = Cuda::new(0).unwrap();
    gpu.gather_rows(&gpu.zeros(&[3, 2]), &[3]);
}

#[test]
#[should_panic(expected = "ne gère pas les tenseurs vides")]
fn empty_tensors_are_refused() {
    let gpu = Cuda::new(0).unwrap();
    gpu.zeros(&[0, 3]);
}

#[test]
#[should_panic(expected = "ne gère pas les tenseurs vides")]
fn gathering_no_row_is_refused() {
    let gpu = Cuda::new(0).unwrap();
    gpu.gather_rows(&gpu.zeros(&[3, 2]), &[]);
}

// --- Le cache de génération ---

#[test]
fn write_rows_changes_only_the_given_rows() {
    let inputs = random_inputs(&[&[6, 2, 3], &[2, 2, 3]], 0);
    let indices = vec![4.0, 1.0];
    let run = |backend: &dyn Fn(&Input, &Input, &[f32]) -> Input| {
        backend(&inputs[0], &inputs[1], &indices)
    };
    let cpu = run(&|table, rows, indices| {
        let b = Cpu::<f32>::new();
        let mut table = b.from_vec(table.0.clone(), &table.1);
        b.write_rows(
            &mut table,
            &b.from_vec(indices.to_vec(), &[indices.len()]),
            &b.from_vec(rows.0.clone(), &rows.1),
        );
        (b.to_vec(&table), b.shape(&table).to_vec())
    });
    let gpu = run(&|table, rows, indices| {
        let b = Cuda::new(0).unwrap();
        let mut table = b.from_vec(table.0.clone(), &table.1);
        b.write_rows(
            &mut table,
            &b.from_vec(indices.to_vec(), &[indices.len()]),
            &b.from_vec(rows.0.clone(), &rows.1),
        );
        (b.to_vec(&table), b.shape(&table).to_vec())
    });
    assert_eq!(cpu, gpu);
    // Les lignes 4 et 1 sont celles de `rows`, les autres n'ont pas bougé.
    let (table, rows) = (&inputs[0].0, &inputs[1].0);
    for line in 0..6 {
        let expected = match line {
            4 => &rows[..6],
            1 => &rows[6..],
            _ => &table[line * 6..(line + 1) * 6],
        };
        assert_eq!(&gpu.0[line * 6..(line + 1) * 6], expected, "ligne {line}");
    }
}

/// L'attention sur le cache, CPU contre GPU. Les séquences de longueurs
/// `lengths` occupent des lignes dispersées d'une table commune ; chaque
/// requête est `(séquence, positions visibles)`.
fn check_cached_attention(
    heads: usize,
    kv_heads: usize,
    key_dim: usize,
    value_dim: usize,
    lengths: &[usize],
    queries: &[(usize, usize)],
) {
    check_cached_attention_of(heads, kv_heads, key_dim, value_dim, None, lengths, queries);
}

/// Comme [`check_cached_attention`], où seules les `value_dim` premières
/// composantes des lignes de valeurs servent, si `value_dim` est donné.
fn check_cached_attention_of(
    heads: usize,
    kv_heads: usize,
    key_dim: usize,
    stride: usize,
    value_dim: Option<usize>,
    lengths: &[usize],
    queries: &[(usize, usize)],
) {
    use rand::seq::SliceRandom;
    use smallem::backend::ops::{CachedAttention, Run};

    let lines: usize = lengths.iter().sum::<usize>() + 7;
    let max_len = *lengths.iter().max().unwrap();
    let mut rng = StdRng::seed_from_u64(lines as u64);
    let mut free: Vec<usize> = (0..lines).collect();
    free.shuffle(&mut rng);
    let mut slots = vec![0.0f32; lengths.len() * max_len];
    for (sequence, &length) in lengths.iter().enumerate() {
        for position in 0..length {
            slots[sequence * max_len + position] = free.pop().unwrap() as f32;
        }
    }
    let rows: Vec<f32> = queries
        .iter()
        .flat_map(|&(sequence, count)| [sequence as f32, count as f32])
        .collect();
    let mut inputs = random_inputs(
        &[
            &[queries.len(), heads, key_dim],
            &[lines, kv_heads, key_dim],
            &[lines, kv_heads, stride],
        ],
        heads as u64,
    );
    inputs.push((slots, vec![lengths.len(), max_len]));
    inputs.push((rows, vec![queries.len(), 2]));
    let op = CachedAttention {
        scale: 0.3,
        value_dim,
    };
    let cpu = eval(&Cpu::<f32>::new(), &inputs, |b, t| {
        b.run(&op, &t.iter().collect::<Vec<_>>())
    });
    let gpu = eval(&Cuda::new(0).unwrap(), &inputs, |b, t| {
        b.run(&op, &t.iter().collect::<Vec<_>>())
    });
    assert_close(&cpu, &gpu);
}

#[test]
fn cached_attention_of_one_token_per_sequence() {
    // La génération par lots : quatre séquences de longueurs très
    // différentes, dont une assez longue pour être découpée en tranches.
    let lengths = [500, 37, 1, 129];
    let queries: Vec<_> = lengths.iter().copied().enumerate().collect();
    check_cached_attention(8, 8, 64, 64, &lengths, &queries);
}

#[test]
fn cached_attention_of_a_prompt() {
    // Les dix derniers tokens d'une amorce de 70 : chacun voit une
    // position de plus que le précédent.
    let queries: Vec<_> = (61..=70).map(|count| (0, count)).collect();
    check_cached_attention(2, 2, 16, 16, &[70], &queries);
}

#[test]
fn cached_attention_with_shared_and_unequal_heads() {
    // Trois têtes de requêtes par tête de clés, des têtes de tailles
    // différentes et qui ne tombent pas juste sur 8.
    check_cached_attention(6, 2, 13, 40, &[33, 90], &[(1, 90), (0, 20), (1, 3)]);
}

#[test]
fn cached_attention_of_part_of_the_values() {
    // L'attention latente : seules les premières composantes des lignes
    // de valeurs sont des valeurs.
    check_cached_attention_of(4, 1, 40, 48, Some(32), &[50, 9], &[(0, 50), (1, 9)]);
}

#[test]
fn cached_attention_with_wide_heads() {
    // Un warp par clé : des têtes de 96 et 80, puis celles de l'attention
    // latente (latent de 128 suivi de 32 composantes tournées, une seule
    // tête de clés pour 8 de requêtes), puis au-delà de 512, la référence.
    check_cached_attention(2, 1, 96, 80, &[20, 5], &[(0, 20), (1, 5)]);
    check_cached_attention_of(8, 1, 160, 160, Some(128), &[300, 7], &[(0, 300), (1, 7)]);
    check_cached_attention(2, 1, 600, 8, &[9], &[(0, 9)]);
}

// --- Poids quantifiés ---

#[test]
fn quantized_matmul_reads_the_codes_like_the_reference() {
    use smallem::backend::ops::{QuantizedMatmul, Run};
    use smallem::quantize::{Format, Quantized};
    // Un token, neuf, quarante (tuiles découpées) et trois cents lignes
    // (forme déquantifiée), en int8 et en int4, avec et sans biais, et des
    // dimensions communes qui ne tombent juste ni sur les mots ni sur les
    // groupes.
    for format in [Format::Int8, Format::INT4, Format::Int4 { group: 32 }] {
        for (rows, inner, cols) in [(1, 512, 512), (9, 100, 70), (40, 1368, 96), (300, 130, 50)] {
            let mut inputs = random_inputs(&[&[rows, inner], &[inner, cols], &[cols]], rows as u64);
            let bias = inputs.pop().unwrap();
            let weight = inputs.pop().unwrap();
            let weight: Vec<f64> = weight.0.iter().map(|&w| f64::from(w)).collect();
            let q = Quantized::round_to_nearest(&weight, inner, cols, format);
            let (_, groups, per_group) = Quantized::layout(format, inner);
            let dequantized: Vec<f32> = q.dequantize().iter().map(|&w| w as f32).collect();
            inputs.push((q.words.clone(), vec![groups * per_group, cols]));
            inputs.push((q.scales.clone(), vec![groups, cols]));
            inputs.push((dequantized, vec![inner, cols]));
            let op = QuantizedMatmul { format };
            check_product_of!(inputs.clone(), |b, t| b
                .run(&op, &[&t[0], &t[1], &t[2], &t[3]]));
            inputs.push(bias);
            check_product_of!(inputs, |b, t| b
                .run(&op, &[&t[0], &t[1], &t[2], &t[3], &t[4]]));
        }
    }
}

#[test]
fn write_rows_ignores_rows_outside_the_table() {
    // Le cache ne donne que des lignes de la table ; un indice faux n'écrit
    // rien plutôt que d'écrire ailleurs sur la carte.
    let inputs = random_inputs(&[&[4, 3], &[4, 3]], 0);
    let b = Cuda::new(0).unwrap();
    let mut table = b.from_vec(inputs[0].0.clone(), &inputs[0].1);
    let indices = b.from_vec(vec![2.0, 4.0, 1e9, -1.0], &[4]);
    b.write_rows(
        &mut table,
        &indices,
        &b.from_vec(inputs[1].0.clone(), &inputs[1].1),
    );
    let (table, before, rows) = (b.to_vec(&table), &inputs[0].0, &inputs[1].0);
    // -1 devient la ligne 0, comme toute valeur négative.
    assert_eq!(&table[..3], &rows[9..]);
    assert_eq!(&table[3..6], &before[3..6]);
    assert_eq!(&table[6..9], &rows[..3]);
    assert_eq!(&table[9..], &before[9..]);
}

#[test]
fn transposed_matmul_of_many_rows_by_the_narrow_kernel() {
    use smallem::backend::ops::{Run, TransposedMatmul};
    // Avec la frontière repoussée, le kernel des matrices étroites reçoit
    // 40 lignes : plus que `NARROW_ROWS`, et pas un multiple de 8.
    let inputs = random_inputs(&[&[2, 20, 70], &[33, 70]], 0);
    check_product_of!(Cuda::new(0).unwrap().with_narrow_rows(usize::MAX); inputs, |b, t| b
        .run(&TransposedMatmul, &[&t[0], &t[1]]));
}

#[test]
#[should_panic(expected = "échelles de forme inattendue")]
fn quantized_matmul_checks_the_scales() {
    use smallem::backend::ops::{QuantizedMatmul, Run};
    use smallem::quantize::{Format, Quantized};
    let (inner, cols) = (64, 8);
    let (_, groups, per_group) = Quantized::layout(Format::Int8, inner);
    let b = Cuda::new(0).unwrap();
    let x = b.ones(&[1, inner]);
    let words = b.zeros(&[groups * per_group, cols]);
    let scales = b.ones(&[groups + 1, cols]);
    let dequantized = b.zeros(&[inner, cols]);
    b.run(
        &QuantizedMatmul {
            format: Format::Int8,
        },
        &[&x, &words, &scales, &dequantized],
    );
}

#[test]
#[should_panic(expected = "affine : [3] · [3]")]
fn affine_checks_the_weight_before_reading_it() {
    // Un poids à une dimension : sa forme n'a pas de seconde dimension à
    // lire.
    let b = Cuda::new(0).unwrap();
    b.affine(&b.ones(&[3]), &b.ones(&[3]), &b.ones(&[3]));
}
