//! Tests des couches, sur CPU en `f64`.

use super::*;
use crate::autograd::{Tape, Var};
use crate::backend::cpu::{Cpu, CpuTensor};
use crate::cache::{AttentionCache, CacheStep, KvCache};
use rand::SeedableRng;
use rand::rngs::StdRng;

type B = Cpu<f64>;

#[test]
fn parameter_is_bound_once_per_tape() {
    let backend = Cpu::new();
    let parameter = Parameter::<B>::zeros(&backend, &[2]);
    let tape = Tape::new(&backend);
    assert_eq!(parameter.var(&tape), parameter.var(&tape));
}

#[test]
fn parameter_is_rebound_on_a_new_tape() {
    let backend = Cpu::new();
    let parameter = Parameter::<B>::zeros(&backend, &[2]);
    let first = Tape::new(&backend);
    first.leaf(CpuTensor::zeros(&[1])); // décale les indices
    let old = parameter.var(&first);
    let second = Tape::new(&backend);
    assert_ne!(parameter.var(&second), old);
}

#[test]
fn shared_parameter_accumulates_both_uses() {
    // perte = Σ (w + w) : le gradient de w vaut 2 partout.
    let backend = Cpu::new();
    let parameter = Parameter::<B>::zeros(&backend, &[3]);
    let tape = Tape::new(&backend);
    let loss = tape.sum_all(tape.add(parameter.var(&tape), parameter.var(&tape)));
    let grads = tape.backward(loss);
    let grad = parameter.grad(tape.id(), &grads).unwrap();
    assert_eq!(grad.data(), &[2.0, 2.0, 2.0]);
}

#[test]
fn grad_is_none_for_another_tape() {
    let backend = Cpu::new();
    let parameter = Parameter::<B>::zeros(&backend, &[3]);
    let tape = Tape::new(&backend);
    let loss = tape.sum_all(parameter.var(&tape));
    let grads = tape.backward(loss);
    assert!(parameter.grad(tape.id() + 1, &grads).is_none());
}

#[test]
fn initialization_has_the_requested_spread() {
    let backend = Cpu::new();
    let parameter =
        Parameter::<B>::normal(&backend, &[200, 200], 0.02, &mut StdRng::seed_from_u64(0));
    let data = parameter.value().data();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    assert!(mean.abs() < 1e-3, "moyenne {mean}");
    assert!((std - 0.02).abs() < 1e-3, "écart-type {std}");
}

#[test]
fn embedding_looks_up_rows() {
    let backend = Cpu::new();
    let embedding = Embedding::<B>::new(&backend, 5, 3, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    let out = embedding.forward(&tape, &[4, 0, 4]);
    assert_eq!(tape.shape(out), vec![3, 3]);
    let table = embedding.weight.value().data();
    assert_eq!(&tape.value(out).data()[0..3], &table[12..15]);
    assert_eq!(&tape.value(out).data()[6..9], &table[12..15]);
}

#[test]
fn linear_maps_the_last_dimension() {
    let backend = Cpu::new();
    let linear = Linear::<B>::new(&backend, 4, 7, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    let x = tape.constant(CpuTensor::ones(&[2, 3, 4]));
    assert_eq!(tape.shape(linear.forward(&tape, x)), vec![2, 3, 7]);
    assert_eq!(linear.parameter_count(&backend), 4 * 7 + 7);
}

#[test]
fn weights_decay_but_biases_do_not() {
    let backend = Cpu::new();
    let linear = Linear::<B>::new(&backend, 4, 7, &mut StdRng::seed_from_u64(0));
    assert!(linear.weight.decay());
    assert!(!linear.bias.decay());
}

#[test]
fn embeddings_do_not_decay() {
    let backend = Cpu::new();
    let embedding = Embedding::<B>::new(&backend, 5, 3, &mut StdRng::seed_from_u64(0));
    assert!(!embedding.weight.decay());
}

#[test]
fn layer_norm_gives_zero_mean_and_unit_variance() {
    let backend = Cpu::new();
    let norm = LayerNorm::<B>::new(&backend, 6);
    let tape = Tape::new(&backend);
    let x = tape.constant(CpuTensor::randn(&[2, 3, 6], &mut StdRng::seed_from_u64(0)));
    // Décalage et échelle arbitraires : la normalisation doit les effacer.
    let x = tape.add(tape.mul(x, tape.scalar(50.0)), tape.scalar(-7.0));
    let out = norm.forward(&tape, x);
    assert_eq!(tape.shape(out), vec![2, 3, 6]);
    for row in tape.value(out).data().chunks(6) {
        let mean = row.iter().sum::<f64>() / 6.0;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 6.0;
        assert!(mean.abs() < 1e-9, "moyenne {mean}");
        assert!((variance - 1.0).abs() < 1e-5, "variance {variance}");
    }
}

/// Sortie de l'attention sur une entrée `[1, T, C]` donnée.
fn attend(attention: &CausalSelfAttention<B>, input: &CpuTensor<f64>) -> Vec<f64> {
    let backend = Cpu::new();
    let tape = Tape::new(&backend);
    let out = attention.forward(&tape, tape.constant(input.clone()), None);
    assert_eq!(tape.shape(out), input.shape());
    tape.value(out).data().to_vec()
}

#[test]
fn attention_cannot_see_the_future() {
    // Modifier la position 2 ne doit rien changer aux positions 0 et 1.
    const T: usize = 4;
    const C: usize = 6;
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    // Poids d'écart-type 1 : avec 0,02 les sorties seraient si petites
    // qu'une fuite passerait inaperçue.
    let mut attention = CausalSelfAttention::<B>::new(&backend, C, 2, 1.0, &mut rng);
    for parameter in attention.parameters_mut() {
        let shape = parameter.value().shape().to_vec();
        *parameter.value_mut() = CpuTensor::randn(&shape, &mut rng);
    }

    let input = CpuTensor::randn(&[1, T, C], &mut rng);
    let mut changed = input.data().to_vec();
    for value in &mut changed[2 * C..3 * C] {
        *value += 1.0;
    }
    let changed = CpuTensor::from_vec(changed, &[1, T, C]);

    let before = attend(&attention, &input);
    let after = attend(&attention, &changed);
    assert_eq!(&before[..2 * C], &after[..2 * C], "le passé a changé");
    assert_ne!(
        &before[2 * C..],
        &after[2 * C..],
        "le présent n'a pas changé"
    );
}

#[test]
fn attention_with_one_head_matches_several_in_shape() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let input = CpuTensor::randn(&[2, 5, 8], &mut rng);
    for heads in [1, 2, 4, 8] {
        let attention = CausalSelfAttention::<B>::new(&backend, 8, heads, 0.02, &mut rng);
        assert_eq!(attend_batch(&attention, &input), vec![2, 5, 8]);
    }
}

fn attend_batch(attention: &CausalSelfAttention<B>, input: &CpuTensor<f64>) -> Vec<usize> {
    let backend = Cpu::new();
    let tape = Tape::new(&backend);
    tape.shape(attention.forward(&tape, tape.constant(input.clone()), None))
}

#[test]
#[should_panic(expected = "n'est pas divisible par 3 têtes")]
fn attention_rejects_uneven_heads() {
    let backend = Cpu::new();
    CausalSelfAttention::<B>::new(&backend, 8, 3, 0.02, &mut StdRng::seed_from_u64(0));
}

#[test]
fn block_keeps_the_shape_and_counts_its_parameters() {
    const C: usize = 8;
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let block = Block::<B>::new(&backend, C, 2, 1, Architecture::GPT2, false, &mut rng);
    let tape = Tape::new(&backend);
    let x = tape.constant(CpuTensor::randn(&[2, 5, C], &mut rng));
    assert_eq!(
        tape.shape(block.forward(&tape, x, None, None, &mut None)),
        vec![2, 5, C]
    );

    let attention = 4 * (C * C + C);
    let feed_forward = (C * 4 * C + 4 * C) + (4 * C * C + C);
    let norms = 2 * 2 * C;
    assert_eq!(
        block.parameter_count(&backend),
        attention + feed_forward + norms
    );
}

#[test]
fn residual_projections_start_smaller() {
    // Avec 8 couches, l'écart-type des projections est 0,02 / √16.
    let backend = Cpu::new();
    let block = Block::<B>::new(
        &backend,
        64,
        4,
        8,
        Architecture::GPT2,
        false,
        &mut StdRng::seed_from_u64(0),
    );
    let spread = |parameter: &Parameter<B>| {
        let data = parameter.value().data();
        (data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64).sqrt()
    };
    let Attention::Standard(attention) = &block.attention else {
        panic!("attention classique attendue");
    };
    assert!((spread(&attention.query.weight) - 0.02).abs() < 2e-3);
    assert!((spread(&block.attention.proj().weight) - 0.005).abs() < 5e-4);
    let FeedForwardLayer::Gelu(feed_forward) = &block.feed_forward else {
        panic!("feed-forward de GPT-2 attendu");
    };
    assert!((spread(&feed_forward.proj.weight) - 0.005).abs() < 5e-4);
}

#[test]
fn names_follow_the_order_of_parameters() {
    let backend = Cpu::new();
    let block = Block::<B>::new(
        &backend,
        8,
        2,
        1,
        Architecture::GPT2,
        false,
        &mut StdRng::seed_from_u64(0),
    );
    let names = block.parameter_names();
    assert_eq!(names.len(), block.parameters().len());
    assert_eq!(names[0], "ln1.gain");
    assert_eq!(names[2], "attention.query.weight");
    assert_eq!(names[9], "attention.proj.bias");
    assert_eq!(names.last().unwrap(), "feed_forward.proj.bias");
    // Chaque nom désigne bien le paramètre de même rang : un biais a une
    // dimension, un poids en a deux.
    for (name, parameter) in names.iter().zip(block.parameters()) {
        let rank = parameter.value().shape().len();
        assert_eq!(rank == 2, name.ends_with("weight"), "{name}");
    }
}

#[test]
fn rms_norm_gives_unit_mean_square() {
    let backend = Cpu::new();
    let norm = RmsNorm::<B>::new(&backend, 6);
    let tape = Tape::new(&backend);
    let x = CpuTensor::randn(&[4, 6], &mut StdRng::seed_from_u64(0));
    let x = x.mul(&CpuTensor::scalar(30.0)).add(&CpuTensor::scalar(5.0));
    let out = norm.forward(&tape, tape.constant(x));
    let out = tape.value(out).clone();
    for row in out.mul(&out).mean(1, false).data() {
        assert!((row - 1.0).abs() < 1e-6, "moyenne des carrés {row}");
    }
    // Un gain, pas de biais.
    assert_eq!(norm.parameter_names(), ["gain"]);
    assert!(!norm.gain.decay());
}

#[test]
fn swiglu_maps_back_to_the_input_dimension() {
    let backend = Cpu::new();
    let layer = SwiGluFeedForward::<B>::new(&backend, 12, 0.02, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    let x = tape.constant(CpuTensor::randn(&[2, 5, 12], &mut StdRng::seed_from_u64(1)));
    assert_eq!(tape.shape(layer.forward(&tape, x)), vec![2, 5, 12]);
    assert_eq!(SwiGluFeedForward::<B>::hidden(12), 32);
    assert_eq!(SwiGluFeedForward::<B>::hidden(512), 1368);
    assert_eq!(
        layer.parameter_names(),
        [
            "gate.weight",
            "gate.bias",
            "up.weight",
            "up.bias",
            "proj.weight",
            "proj.bias"
        ]
    );
}

#[test]
fn rope_tables_follow_the_positions() {
    let backend = Cpu::new();
    let tape = Tape::<B>::new(&backend);
    let rope = Rope::new(&tape, 3, 2, 4);
    let cos = tape.value(rope.cos).clone();
    let sin = tape.value(rope.sin).clone();
    assert_eq!(cos.shape(), &[2, 2]);
    // Positions 3 et 4 ; la première paire tourne d'un radian par
    // position, la seconde cent fois moins vite (10000^(-2/4) = 0,01).
    let expected = [3.0f64, 0.03, 4.0, 0.04];
    for (index, angle) in expected.iter().enumerate() {
        assert!((cos.data()[index] - angle.cos()).abs() < 1e-12);
        assert!((sin.data()[index] - angle.sin()).abs() < 1e-12);
    }
}

#[test]
fn architecture_choices_have_names() {
    for kind in [NormKind::Layer, NormKind::Rms] {
        assert_eq!(kind.name().parse::<NormKind>(), Ok(kind));
    }
    for kind in [FeedForwardKind::Gelu, FeedForwardKind::SwiGlu] {
        assert_eq!(kind.name().parse::<FeedForwardKind>(), Ok(kind));
    }
    for kind in [PositionKind::Learned, PositionKind::Rotary] {
        assert_eq!(kind.name().parse::<PositionKind>(), Ok(kind));
    }
    let error = "sinus".parse::<PositionKind>().unwrap_err();
    assert!(error.contains("learned, rope"), "{error}");
}

/// Une attention à deux têtes aux poids agrandis, et de quoi la nourrir.
fn attention_and_input(positions: usize) -> (CausalSelfAttention<B>, CpuTensor<f64>) {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(8);
    let mut attention = CausalSelfAttention::<B>::new(&backend, 8, 2, 0.02, &mut rng);
    for parameter in attention.parameters_mut() {
        let shape = parameter.value().shape().to_vec();
        *parameter.value_mut() = CpuTensor::randn(&shape, &mut rng);
    }
    (attention, CpuTensor::randn(&[1, positions, 8], &mut rng))
}

/// Fait passer `input`, `[1, 6, C]`, par une couche d'attention en trois
/// morceaux de 2, 1 et 3 positions, à travers un cache d'une séquence.
/// `rope` donne la taille des vecteurs tournés et le nombre de têtes, si la
/// couche utilise RoPE. Rend les sorties des morceaux mises bout à bout, et
/// le cache.
fn piece_by_piece(
    input: &CpuTensor<f64>,
    rope: Option<(usize, usize)>,
    mut layer: impl FnMut(&Tape<B>, Var, &mut AttentionCache<B>, &CacheStep) -> Var,
) -> (Vec<f64>, KvCache<B>) {
    let backend = Cpu::new();
    let mut cache = KvCache::new(1, 2, 4);
    let sequence = cache.add_sequence();
    let mut pieces = Vec::new();
    for (start, length) in [(0usize, 2usize), (2, 1), (3, 3)] {
        // Une tape par morceau, comme en génération : la table du cache ne
        // s'écrit que si plus aucune tape ne la tient.
        let tape = Tape::new(&backend);
        let (mut step, positions) = cache.begin(&tape, &[(sequence, length)], 6);
        assert_eq!(positions, (start..start + length).collect::<Vec<_>>());
        if let Some((dim, heads)) = rope {
            step.rope = Some(Rope::at(&tape, &positions, dim, heads));
            step.rope_single = Some(Rope::at(&tape, &positions, dim, 1));
        }
        let rows: Vec<u32> = (start as u32..(start + length) as u32).collect();
        let width = input.shape()[2];
        let piece = input
            .reshape(&[6, width])
            .gather_rows(&rows)
            .reshape(&[1, length, width]);
        let out = layer(&tape, tape.constant(piece), cache.stage_mut(0), &step);
        pieces.extend_from_slice(tape.value(out).data());
        assert_eq!(cache.len(sequence), start + length);
    }
    (pieces, cache)
}

#[test]
fn attention_gives_the_same_result_whole_or_piece_by_piece() {
    // La séquence d'un coup, puis en trois morceaux qui passent par le
    // cache : 2 positions, puis 1, puis 3. Avec et sans RoPE.
    let backend = Cpu::new();
    let (attention, input) = attention_and_input(6);
    for with_rope in [false, true] {
        let tape = Tape::new(&backend);
        let rope = with_rope.then(|| Rope::new(&tape, 0, 6, 4));
        let whole = attention.forward(&tape, tape.constant(input.clone()), rope);
        let whole = tape.value(whole).clone();

        let (pieces, _) = piece_by_piece(
            &input,
            with_rope.then_some((4, 2)),
            |tape, x, cache, step| attention.forward_cached(tape, x, cache, step).0,
        );
        for (piece, whole) in pieces.iter().zip(whole.data()) {
            assert!(
                (piece - whole).abs() < 1e-12,
                "RoPE {with_rope} : {piece} != {whole}"
            );
        }
    }
}

#[test]
fn modules_are_walked_in_field_order_with_nested_names() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut blocks = vec![
        Block::<B>::new(&backend, 8, 2, 2, Architecture::GPT2, false, &mut rng),
        Block::<B>::new(&backend, 8, 2, 2, Architecture::MODERN, false, &mut rng),
    ];
    let names = blocks.parameter_names();
    assert_eq!(names[0], "0.ln1.gain");
    assert_eq!(names[1], "0.ln1.bias");
    assert_eq!(names[2], "0.attention.query.weight");
    // Le second bloc est moderne : RMSNorm n'a pas de biais, et SwiGLU a
    // trois couches.
    assert!(names.contains(&"1.ln1.gain".to_string()));
    assert!(!names.contains(&"1.ln1.bias".to_string()));
    assert!(names.contains(&"1.feed_forward.gate.weight".to_string()));
    assert_eq!(names.last().unwrap(), "1.feed_forward.proj.bias");

    // Les trois vues d'un module parcourent les mêmes paramètres, dans le
    // même ordre : elles viennent d'un seul et même parcours.
    assert_eq!(names.len(), blocks.parameters().len());
    let addresses: Vec<*const Parameter<B>> =
        blocks.parameters().iter().map(|p| *p as *const _).collect();
    let mutable: Vec<*const Parameter<B>> = blocks
        .parameters_mut()
        .into_iter()
        .map(|p| p as *const _)
        .collect();
    assert_eq!(addresses, mutable);

    // Un module absent n'a aucun paramètre ; un paramètre seul porte le nom
    // qu'on lui donne.
    assert!(None::<Linear<B>>.parameter_names().is_empty());
    let mut seen = Vec::new();
    Parameter::<B>::zeros(&backend, &[1]).walk("seul", &mut |name, _| seen.push(name));
    assert_eq!(seen, ["seul"]);
}

// --- Attention latente ---

/// Tailles de l'attention latente des tests : `C = 8`, deux têtes de 4.
const LATENT: LatentDims = LatentDims {
    latent: 3,
    query_latent: 5,
    rope: 2,
};

/// Une attention latente à deux têtes aux poids agrandis, et de quoi la
/// nourrir.
fn latent_attention_and_input(positions: usize) -> (LatentAttention<B>, CpuTensor<f64>) {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(8);
    let mut attention = LatentAttention::<B>::new(&backend, 8, 2, LATENT, 0.02, &mut rng);
    for parameter in attention.parameters_mut() {
        let shape = parameter.value().shape().to_vec();
        *parameter.value_mut() = CpuTensor::randn(&shape, &mut rng);
    }
    (attention, CpuTensor::randn(&[1, positions, 8], &mut rng))
}

/// L'attention latente de `input`, une séquence dont la première position
/// est `start`.
fn latent_attend(attention: &LatentAttention<B>, input: &CpuTensor<f64>, start: usize) -> Vec<f64> {
    let backend = Cpu::new();
    let tape = Tape::new(&backend);
    let rope = Rope::new(&tape, start, input.shape()[1], LATENT.rope);
    let out = attention.forward(&tape, tape.constant(input.clone()), rope);
    tape.value(out).data().to_vec()
}

/// L'attention latente d'une séquence, écrite d'après les formules du
/// rapport de DeepSeek-V2, vecteur par vecteur et sans aucun tenseur : les
/// clés et les requêtes y sont réellement concaténées.
fn latent_attention_by_hand(attention: &LatentAttention<B>, input: &CpuTensor<f64>) -> Vec<f64> {
    const C: usize = 8;
    const HEADS: usize = 2;
    const D: usize = C / HEADS;
    let rope_dim = LATENT.rope;
    // `x · W`, pour un seul vecteur.
    let project = |x: &[f64], weight: &Parameter<B>| -> Vec<f64> {
        let (rows, cols) = (weight.value().shape()[0], weight.value().shape()[1]);
        assert_eq!(x.len(), rows);
        (0..cols)
            .map(|j| {
                (0..rows)
                    .map(|i| x[i] * weight.value().data()[i * cols + j])
                    .sum()
            })
            .collect()
    };
    let rms = |x: Vec<f64>, gain: &Parameter<B>| -> Vec<f64> {
        let mean_square = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let scale = (mean_square + RmsNorm::<B>::EPSILON).sqrt();
        x.iter()
            .zip(gain.value().data())
            .map(|(v, g)| v / scale * g)
            .collect()
    };
    let rotate = |x: &[f64], position: usize| -> Vec<f64> {
        let mut out = Vec::new();
        for (pair, values) in x.chunks(2).enumerate() {
            let frequency = Rope::BASE.powf(-(2.0 * pair as f64) / x.len() as f64);
            let (sin, cos) = (position as f64 * frequency).sin_cos();
            out.push(values[0] * cos - values[1] * sin);
            out.push(values[0] * sin + values[1] * cos);
        }
        out
    };

    let length = input.shape()[1];
    // Pour chaque position et chaque tête : requête et clé concaténées, et
    // valeur.
    let mut queries = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (position, x) in input.data().chunks(C).enumerate() {
        let query_latent = rms(
            project(x, &attention.query_down.weight),
            &attention.query_norm.gain,
        );
        let q = project(&query_latent, &attention.query_up.weight);
        let q_rope = project(&query_latent, &attention.query_rope.weight);
        let latent = rms(
            project(x, &attention.latent_down.weight),
            &attention.latent_norm.gain,
        );
        let k = project(&latent, &attention.key_up.weight);
        let v = project(&latent, &attention.value_up.weight);
        let k_rope = rotate(&project(x, &attention.key_rope.weight), position);
        for head in 0..HEADS {
            let mut query = q[head * D..(head + 1) * D].to_vec();
            query.extend(rotate(
                &q_rope[head * rope_dim..(head + 1) * rope_dim],
                position,
            ));
            let mut key = k[head * D..(head + 1) * D].to_vec();
            key.extend(&k_rope);
            queries.push(query);
            keys.push(key);
            values.push(v[head * D..(head + 1) * D].to_vec());
        }
    }

    let mut output = Vec::new();
    for position in 0..length {
        let mut merged = Vec::new();
        for head in 0..HEADS {
            let query = &queries[position * HEADS + head];
            let scores: Vec<f64> = (0..=position)
                .map(|past| {
                    let key = &keys[past * HEADS + head];
                    let dot: f64 = query.iter().zip(key).map(|(q, k)| q * k).sum();
                    dot / ((D + rope_dim) as f64).sqrt()
                })
                .collect();
            let largest = scores.iter().cloned().fold(f64::MIN, f64::max);
            let total: f64 = scores.iter().map(|s| (s - largest).exp()).sum();
            let mut mixed = vec![0.0; D];
            for (past, score) in scores.iter().enumerate() {
                let weight = (score - largest).exp() / total;
                for (m, v) in mixed.iter_mut().zip(&values[past * HEADS + head]) {
                    *m += weight * v;
                }
            }
            merged.extend(mixed);
        }
        let projected = project(&merged, &attention.proj.weight);
        output.extend(
            projected
                .iter()
                .zip(attention.proj.bias.value().data())
                .map(|(p, b)| p + b),
        );
    }
    output
}

#[test]
fn latent_attention_matches_the_formulas_written_by_hand() {
    let (attention, input) = latent_attention_and_input(5);
    let layer = latent_attend(&attention, &input, 0);
    let by_hand = latent_attention_by_hand(&attention, &input);
    assert_eq!(layer.len(), by_hand.len());
    let mut largest = 0.0f64;
    for (index, (a, b)) in layer.iter().zip(&by_hand).enumerate() {
        assert!((a - b).abs() < 1e-11, "élément {index} : {a} != {b}");
        largest = largest.max(a.abs());
    }
    assert!(largest > 0.1, "sorties trop petites : max {largest}");
}

#[test]
fn latent_attention_cannot_see_the_future() {
    let (attention, input) = latent_attention_and_input(4);
    let before = latent_attend(&attention, &input, 0);
    for changed_position in 0..4 {
        let mut data = input.data().to_vec();
        for value in &mut data[changed_position * 8..(changed_position + 1) * 8] {
            *value += 1.0;
        }
        let after = latent_attend(&attention, &CpuTensor::from_vec(data, &[1, 4, 8]), 0);
        let split = changed_position * 8;
        assert_eq!(&before[..split], &after[..split]);
        assert_ne!(&before[split..split + 8], &after[split..split + 8]);
    }
}

#[test]
fn latent_attention_gives_the_same_result_whole_or_piece_by_piece() {
    // La séquence d'un coup, puis en trois morceaux qui passent par le
    // cache : 2 positions, puis 1, puis 3. Le chemin de la génération
    // absorbe les matrices des clés et des valeurs dans la requête et la
    // sortie : le calcul est réorganisé, le résultat est le même.
    let (attention, input) = latent_attention_and_input(6);
    let whole = latent_attend(&attention, &input, 0);
    let (pieces, _) = piece_by_piece(&input, Some((LATENT.rope, 2)), |tape, x, cache, step| {
        attention.forward_cached(tape, x, cache, step)
    });
    for (piece, whole) in pieces.iter().zip(&whole) {
        assert!((piece - whole).abs() < 1e-12, "{piece} != {whole}");
    }
}

#[test]
fn latent_attention_caches_only_latents_and_rotated_keys() {
    let backend = Cpu::new();
    let (attention, input) = latent_attention_and_input(6);
    let (_, cache) = piece_by_piece(&input, Some((LATENT.rope, 2)), |tape, x, cache, step| {
        attention.forward_cached(tape, x, cache, step)
    });
    // Une seule table, d'une « tête » : le latent suivi de la clé tournée.
    // `d_c + d_r` nombres par position, contre `2 × C` pour l'attention
    // classique.
    assert_eq!(cache.numbers_per_position(&backend), 3 + 2);
    let (standard, _) = attention_and_input(6);
    let (_, full) = piece_by_piece(&input, None, |tape, x, cache, step| {
        standard.forward_cached(tape, x, cache, step).0
    });
    assert_eq!(full.numbers_per_position(&backend), 2 * 8);
}

#[test]
fn latent_attention_scores_depend_only_on_distances() {
    // La propriété de RoPE, conservée par la partie tournée : décaler
    // toutes les positions du même nombre ne change rien.
    let (attention, input) = latent_attention_and_input(4);
    let at_zero = latent_attend(&attention, &input, 0);
    let shifted = latent_attend(&attention, &input, 37);
    for (a, b) in at_zero.iter().zip(&shifted) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn latent_attention_counts_its_parameters() {
    const C: usize = 8;
    let backend = Cpu::new();
    let (attention, _) = latent_attention_and_input(1);
    let LatentDims {
        latent,
        query_latent,
        rope,
    } = LATENT;
    let queries = C * query_latent + query_latent + query_latent * C + query_latent * 2 * rope;
    let keys_and_values = C * latent + latent + 2 * latent * C + C * rope;
    let proj = C * C + C;
    assert_eq!(
        attention.parameter_count(&backend),
        queries + keys_and_values + proj
    );
    assert_eq!(
        attention.parameter_names(),
        [
            "query_down.weight",
            "query_norm.gain",
            "query_up.weight",
            "query_rope.weight",
            "latent_down.weight",
            "latent_norm.gain",
            "key_up.weight",
            "value_up.weight",
            "key_rope.weight",
            "proj.weight",
            "proj.bias",
        ]
    );
}

#[test]
fn default_latent_sizes_follow_the_model() {
    // La configuration `xl` : `C = 512`, 8 têtes de 64.
    assert_eq!(
        LatentDims::for_model(512, 8),
        LatentDims {
            latent: 128,
            query_latent: 384,
            rope: 32,
        }
    );
    // Une demi-tête impaire est arrondie à la taille paire suivante.
    assert_eq!(LatentDims::for_model(60, 10).rope, 4);
    assert_eq!(AttentionKind::Standard.name(), "standard");
    assert_eq!(AttentionKind::Latent(LATENT).name(), "mla");
}

#[test]
#[should_panic(expected = "RoPE demande une taille paire")]
fn latent_attention_rejects_an_odd_rotary_size() {
    let dims = LatentDims { rope: 3, ..LATENT };
    LatentAttention::<B>::new(&Cpu::new(), 8, 2, dims, 0.02, &mut StdRng::seed_from_u64(0));
}

#[test]
fn projection_has_no_bias() {
    let backend = Cpu::new();
    let projection = Projection::<B>::new(&backend, 4, 3, &mut StdRng::seed_from_u64(0));
    assert_eq!(projection.parameter_names(), ["weight"]);
    assert!(projection.weight.decay());
    let tape = Tape::new(&backend);
    let x = tape.constant(CpuTensor::zeros(&[2, 5, 4]));
    let out = projection.forward(&tape, x);
    assert_eq!(tape.shape(out), vec![2, 5, 3]);
    assert!(tape.value(out).data().iter().all(|&v| v == 0.0));
}

#[test]
fn a_block_can_use_latent_attention() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let architecture = Architecture::MODERN.with_attention(AttentionKind::Latent(LATENT));
    let block = Block::<B>::new(&backend, 8, 2, 2, architecture, false, &mut rng);
    let names = block.parameter_names();
    assert!(names.contains(&"attention.latent_down.weight".to_string()));
    assert!(!names.iter().any(|name| name.starts_with("attention.key.")));
    let tape = Tape::new(&backend);
    let x = tape.constant(CpuTensor::randn(&[2, 5, 8], &mut rng));
    let rope = Rope::new(&tape, 0, 5, LATENT.rope);
    assert_eq!(
        tape.shape(block.forward(&tape, x, Some(rope), None, &mut None)),
        vec![2, 5, 8]
    );
}

// --- Mélange d'experts ---

/// Tailles du mélange d'experts des tests : `C = 8`, un expert partagé et
/// quatre aiguillés dont deux actifs.
const EXPERTS: ExpertDims = ExpertDims {
    shared: 1,
    routed: 4,
    active: 2,
    hidden: 8,
    balance: Balance::None,
};

/// Un mélange d'experts aux poids agrandis, et de quoi le nourrir.
fn experts_and_input(dims: ExpertDims, tokens: usize) -> (MixtureOfExperts<B>, CpuTensor<f64>) {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(3);
    let mut layer = MixtureOfExperts::<B>::new(&backend, 8, dims, 0.02, &mut rng);
    for (name, parameter) in layer.parameter_names().iter().zip(layer.parameters_mut()) {
        if name != "routing_bias" {
            let shape = parameter.value().shape().to_vec();
            *parameter.value_mut() = CpuTensor::randn(&shape, &mut rng);
        }
    }
    (layer, CpuTensor::randn(&[2, tokens / 2, 8], &mut rng))
}

fn run_experts(layer: &MixtureOfExperts<B>, input: &CpuTensor<f64>) -> Vec<f64> {
    let backend = Cpu::new();
    let tape = Tape::new(&backend);
    let out = layer.forward(&tape, tape.constant(input.clone()));
    assert_eq!(tape.shape(out), input.shape());
    tape.value(out).data().to_vec()
}

#[test]
fn routing_keeps_the_best_experts_of_each_token() {
    // Deux tokens, quatre experts, deux actifs.
    let scores = [0.1, 0.9, 0.5, 0.2, 0.7, 0.7, 0.7, 0.1];
    let routing = route(&scores, &[0.0; 4], 2);
    assert_eq!(routing.groups, [vec![1], vec![0, 1], vec![0], vec![]]);
    // À égalité, le plus petit rang l'emporte : le second token garde les
    // experts 0 et 1, pas le 2.
    assert_eq!(
        routing.mask,
        [false, true, true, false, true, true, false, false]
    );
    // Le biais change le choix : l'expert 3 passe devant.
    let biased = route(&scores, &[0.0, 0.0, 0.0, 1.0], 2);
    assert_eq!(biased.groups, [vec![1], vec![0], vec![], vec![0, 1]]);
    // Chaque token a toujours ses deux experts.
    for row in biased.mask.chunks(4) {
        assert_eq!(row.iter().filter(|&&kept| kept).count(), 2);
    }
}

#[test]
fn experts_match_the_formula_written_by_hand() {
    let backend = Cpu::new();
    let (layer, input) = experts_and_input(EXPERTS, 6);
    let output = run_experts(&layer, &input);

    // Un expert appliqué à un seul token.
    let apply = |expert: &SwiGluFeedForward<B>, token: &[f64]| -> Vec<f64> {
        let tape = Tape::new(&backend);
        let x = tape.constant(CpuTensor::from_vec(token.to_vec(), &[1, 8]));
        tape.value(expert.forward(&tape, x)).data().to_vec()
    };
    let router = layer.router.weight.value().data();
    for (index, token) in input.data().chunks(8).enumerate() {
        let affinity: Vec<f64> = (0..4)
            .map(|expert| {
                let dot: f64 = (0..8).map(|i| token[i] * router[i * 4 + expert]).sum();
                1.0 / (1.0 + (-dot).exp())
            })
            .collect();
        let mut order = [0usize, 1, 2, 3];
        order.sort_by(|&a, &b| affinity[b].total_cmp(&affinity[a]));
        let kept = &order[..2];
        let total: f64 = kept.iter().map(|&expert| affinity[expert]).sum();

        let mut expected = apply(&layer.shared[0], token);
        for &expert in kept {
            let gate = affinity[expert] / total;
            for (e, value) in expected
                .iter_mut()
                .zip(apply(&layer.experts[expert], token))
            {
                *e += gate * value;
            }
        }
        for (a, e) in output[index * 8..(index + 1) * 8].iter().zip(&expected) {
            assert!((a - e).abs() < 1e-11, "token {index} : {a} != {e}");
        }
    }
}

#[test]
fn sparse_experts_match_dense_experts() {
    let (mut layer, input) = experts_and_input(EXPERTS, 12);
    let sparse = run_experts(&layer, &input);
    layer.dense = true;
    let dense = run_experts(&layer, &input);
    let mut largest = 0.0f64;
    for (s, d) in sparse.iter().zip(&dense) {
        assert!((s - d).abs() < 1e-12, "{s} != {d}");
        largest = largest.max(s.abs());
    }
    assert!(largest > 0.1, "sorties trop petites : max {largest}");
}

#[test]
fn sparse_and_dense_experts_have_the_same_gradients() {
    let backend = Cpu::new();
    let (mut layer, input) = experts_and_input(EXPERTS, 12);
    let mut gradients = |dense: bool| -> Vec<Vec<f64>> {
        layer.dense = dense;
        let tape = Tape::new(&backend);
        let out = layer.forward(&tape, tape.constant(input.clone()));
        let grads = tape.backward(tape.sum_all(tape.mul(out, out)));
        layer
            .parameters()
            .iter()
            .map(|p| {
                p.grad(tape.id(), &grads)
                    .map_or(Vec::new(), |grad| grad.data().to_vec())
            })
            .collect()
    };
    let (sparse, dense) = (gradients(false), gradients(true));
    for (index, (s, d)) in sparse.iter().zip(&dense).enumerate() {
        assert_eq!(s.len(), d.len(), "paramètre {index}");
        for (s, d) in s.iter().zip(d) {
            assert!((s - d).abs() < 1e-10, "paramètre {index} : {s} != {d}");
        }
    }
    // L'aiguilleur apprend : son gradient passe par les portes.
    let names = layer.parameter_names();
    let router = names.iter().position(|n| n == "router.weight").unwrap();
    assert!(sparse[router].iter().any(|g| g.abs() > 1e-3));
    // Le biais d'aiguillage, lui, n'est pas appris par gradient.
    let bias = names.iter().position(|n| n == "routing_bias").unwrap();
    assert!(sparse[bias].is_empty());
}

#[test]
fn an_expert_nobody_chose_is_skipped() {
    // Un biais énorme contre l'expert 2 : personne ne le choisit.
    let backend = Cpu::new();
    let (mut layer, input) = experts_and_input(EXPERTS, 12);
    *layer.routing_bias.value_mut() = CpuTensor::from_vec(vec![0.0, 0.0, -10.0, 0.0], &[4]);
    let tape = Tape::new(&backend);
    let out = layer.forward(&tape, tape.constant(input.clone()));
    let grads = tape.backward(tape.sum_all(out));
    assert_eq!(layer.take_load()[2], 0);
    // Il n'a pas servi : aucun gradient, donc aucun pas d'optimiseur.
    assert!(
        layer.experts[2]
            .proj
            .weight
            .grad(tape.id(), &grads)
            .is_none()
    );
    assert!(
        layer.experts[0]
            .proj
            .weight
            .grad(tape.id(), &grads)
            .is_some()
    );
    // Les portes ne tiennent pas compte du biais : la version dense, qui
    // calcule tout, donne le même résultat.
    let sparse = tape.value(out).data().to_vec();
    layer.dense = true;
    for (s, d) in sparse.iter().zip(&run_experts(&layer, &input)) {
        assert!((s - d).abs() < 1e-12, "{s} != {d}");
    }
}

#[test]
fn borrowed_attention_reads_the_keys_it_is_lent() {
    // Avec les requêtes et la sortie de la couche à qui elle emprunte, une
    // attention sans clés calcule exactement ce que calcule cette couche.
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(3);
    let owner = CausalSelfAttention::<B>::new(&backend, 8, 2, 0.02, &mut rng);
    let mut borrower = BorrowedAttention::<B>::new(&backend, 8, 2, 0.02, &mut rng);
    *borrower.query.weight.value_mut() = owner.query.weight.value().clone();
    *borrower.query.bias.value_mut() = owner.query.bias.value().clone();
    *borrower.proj.weight.value_mut() = owner.proj.weight.value().clone();
    *borrower.proj.bias.value_mut() = owner.proj.bias.value().clone();

    let input = CpuTensor::<f64>::randn(&[2, 5, 8], &mut rng);
    // Renvoie la sortie de la couche complète, puis celle de l'emprunteuse.
    let outputs = |borrower: &BorrowedAttention<B>| {
        let tape = Tape::new(&backend);
        let x = tape.constant(input.clone());
        let expected = owner.forward(&tape, x, None);
        let lent = owner.keys_and_values(&tape, x, None);
        let borrowed = borrower.forward(&tape, x, None, lent);
        (
            tape.value(expected).data().to_vec(),
            tape.value(borrowed).data().to_vec(),
        )
    };
    let (expected, borrowed) = outputs(&borrower);
    assert_eq!(borrowed, expected);

    // Avec d'autres requêtes, elle regarde ailleurs.
    *borrower.query.weight.value_mut() = CpuTensor::<f64>::randn(&[8, 8], &mut rng);
    let (expected, borrowed) = outputs(&borrower);
    assert_ne!(borrowed, expected);
}

#[test]
fn expert_load_is_counted_and_handed_over_once() {
    let (layer, input) = experts_and_input(EXPERTS, 12);
    run_experts(&layer, &input);
    let load = layer.take_load();
    // Douze tokens, deux experts chacun.
    assert_eq!(load.iter().sum::<u64>(), 24);
    assert!(load.iter().all(|&count| count <= 12));
    assert_eq!(layer.take_load(), [0; 4]);
    run_experts(&layer, &input);
    layer.reset_load();
    assert_eq!(layer.take_load(), [0; 4]);
}

#[test]
fn routing_bias_moves_against_the_load() {
    let backend = Cpu::new();
    let dims = ExpertDims {
        balance: Balance::Bias { rate: 0.25 },
        ..EXPERTS
    };
    let (mut layer, input) = experts_and_input(dims, 12);
    run_experts(&layer, &input);
    let load = layer.take_load();
    layer.rebalance(&backend);
    let bias = layer.routing_bias.value().data().to_vec();
    // Moyenne : 24 choix pour 4 experts, soit 6.
    for (count, bias) in load.iter().zip(&bias) {
        let expected = match count.cmp(&6) {
            std::cmp::Ordering::Greater => -0.25,
            std::cmp::Ordering::Less => 0.25,
            std::cmp::Ordering::Equal => 0.0,
        };
        assert_eq!(*bias, expected, "charge {count}");
    }
    assert!(bias.iter().any(|&b| b != 0.0), "charges {load:?}");
    // Sans nouveaux tokens, un second appel ne change rien.
    layer.rebalance(&backend);
    assert_eq!(layer.routing_bias.value().data(), &bias[..]);

    // Répété, le réglage finit par répartir la charge : aucun expert ne
    // reçoit plus tous les tokens, aucun n'est oublié.
    for _ in 0..200 {
        run_experts(&layer, &input);
        layer.rebalance(&backend);
    }
    layer.take_load();
    run_experts(&layer, &input);
    let balanced = layer.take_load();
    assert!(
        balanced.iter().all(|&count| (3..=9).contains(&count)),
        "charges après équilibrage : {balanced:?}, au départ {load:?}"
    );

    // Sans équilibrage par biais, rien ne bouge.
    let (mut plain, input) = experts_and_input(EXPERTS, 12);
    run_experts(&plain, &input);
    plain.rebalance(&backend);
    assert_eq!(plain.routing_bias.value().data(), &[0.0; 4]);
}

#[test]
fn balance_penalty_is_its_weight_when_affinities_are_uniform() {
    let backend = Cpu::new();
    let dims = ExpertDims {
        balance: Balance::Loss { weight: 0.5 },
        ..EXPERTS
    };
    let (mut layer, input) = experts_and_input(dims, 12);
    // Un aiguilleur nul : toutes les affinités valent 1/2, donc `P_i = 1/E`
    // et `Σ f_i × P_i = 1` quelle que soit la répartition.
    *layer.router.weight.value_mut() = CpuTensor::zeros(&[8, 4]);
    let tape = Tape::new(&backend);
    layer.forward(&tape, tape.constant(input.clone()));
    let penalty = layer.penalty(&tape).expect("pénalité calculée");
    assert!((tape.value(penalty).item() - 0.5).abs() < 1e-12);
    // Elle appartient à sa tape, et n'existe pas sans ce réglage.
    assert!(layer.penalty(&Tape::new(&backend)).is_none());
    let (plain, _) = experts_and_input(EXPERTS, 12);
    let tape = Tape::new(&backend);
    plain.forward(&tape, tape.constant(input));
    assert!(plain.penalty(&tape).is_none());
}

#[test]
fn balance_penalty_grows_with_the_imbalance() {
    // Un aiguilleur qui préfère nettement deux experts donne une pénalité
    // plus forte qu'un aiguilleur indifférent.
    let backend = Cpu::new();
    let dims = ExpertDims {
        balance: Balance::Loss { weight: 1.0 },
        ..EXPERTS
    };
    let (mut layer, input) = experts_and_input(dims, 12);
    let positive = input.mul(&input);
    let mut weights = vec![0.0; 32];
    for row in 0..8 {
        weights[row * 4] = 1.0;
        weights[row * 4 + 1] = 1.0;
        weights[row * 4 + 2] = -1.0;
        weights[row * 4 + 3] = -1.0;
    }
    *layer.router.weight.value_mut() = CpuTensor::from_vec(weights, &[8, 4]);
    let tape = Tape::new(&backend);
    layer.forward(&tape, tape.constant(positive));
    let penalty = tape.value(layer.penalty(&tape).unwrap()).item();
    assert_eq!(layer.take_load(), [12, 12, 0, 0]);
    assert!(penalty > 1.5, "pénalité {penalty}");
}

#[test]
fn default_expert_sizes_keep_the_active_parameters() {
    // `C = 512` : le feed-forward de référence a une largeur de 1 368.
    let dims = ExpertDims::for_model(512);
    assert_eq!((dims.shared, dims.routed, dims.active), (1, 8, 2));
    assert_eq!(dims.hidden, 456);
    assert_eq!(dims.hidden * (dims.shared + dims.active), 1368);
    assert_eq!(dims.validate(), Ok(()));
    for refused in [
        ExpertDims { active: 9, ..dims },
        ExpertDims { routed: 0, ..dims },
        ExpertDims {
            balance: Balance::Bias { rate: 0.0 },
            ..dims
        },
        ExpertDims {
            balance: Balance::Loss { weight: f64::NAN },
            ..dims
        },
    ] {
        assert!(refused.validate().is_err(), "{refused:?}");
    }
}

#[test]
fn expert_parameters_have_nested_names() {
    let backend = Cpu::new();
    let (layer, _) = experts_and_input(EXPERTS, 2);
    let names = layer.parameter_names();
    assert_eq!(names[0], "shared.0.gate.weight");
    assert!(names.contains(&"experts.3.proj.bias".to_string()));
    assert_eq!(&names[names.len() - 2..], ["router.weight", "routing_bias"]);
    let expert = 3 * 8 * 8 + 2 * 8 + 8;
    assert_eq!(layer.parameter_count(&backend), 5 * expert + 8 * 4 + 4);
}

#[test]
fn expert_groups_are_padded_to_round_sizes() {
    // 64 tokens : les groupes sont arrondis au multiple de 2 supérieur,
    // avec des tokens dont la porte est nulle.
    let mut rng = StdRng::seed_from_u64(5);
    let scores: Vec<f64> = CpuTensor::<f64>::randn(&[64, 4], &mut rng).data().to_vec();
    let routing = route(&scores, &[0.0; 4], 2);
    let mut padded_somewhere = false;
    for expert in 0..4 {
        let group = &routing.groups[expert];
        let padded = routing.padded_group(expert);
        assert_eq!(padded.len(), group.len().next_multiple_of(2));
        assert_eq!(&padded[..group.len()], &group[..]);
        for &extra in &padded[group.len()..] {
            assert!(!routing.mask[extra as usize * 4 + expert]);
            padded_somewhere = true;
        }
    }
    assert!(padded_somewhere, "aucun groupe de taille impaire");
    // Un groupe qui contient tout le monde n'a personne pour le compléter.
    let everyone = route(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[0.0; 2], 1);
    assert_eq!(everyone.padded_group(0), [0, 1, 2]);

    // Et le résultat ne change pas : la version dense donne les mêmes
    // sorties sur 64 tokens.
    let (mut layer, input) = experts_and_input(EXPERTS, 64);
    let sparse = run_experts(&layer, &input);
    layer.dense = true;
    for (s, d) in sparse.iter().zip(&run_experts(&layer, &input)) {
        assert!((s - d).abs() < 1e-12, "{s} != {d}");
    }
}
