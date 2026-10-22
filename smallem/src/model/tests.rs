use super::*;
use crate::backend::Float;
use crate::backend::cpu::{Cpu, CpuTensor};
use crate::data::Dataset;
use std::collections::BTreeMap;

use serde_json::Value;

use crate::nn::{
    Architecture, Balance, ExpertDims, FeedForwardKind, LatentDims, MemoryDims, MemoryStorage,
    Module, MultiToken, NormKind,
};
use crate::optim::AdamW;
use crate::training::{evaluate, train_step};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

type B = Cpu<f64>;

/// Le modèle minuscule des tests : une couche, `C = 8`.
pub(crate) const TINY: GptConfig = GptConfig {
    vocab: 11,
    context: 4,
    layers: 1,
    heads: 2,
    embed: 8,
    architecture: Architecture::GPT2,
};

/// Remplace tous les paramètres par des tirages d'écart-type 0,5. Avec
/// l'initialisation normale, à 0,02, les sorties et les gradients
/// seraient si petits qu'une erreur passerait sous les tolérances.
pub(crate) fn randomize(model: &mut Gpt<B>, rng: &mut StdRng) {
    for parameter in model.parameters_mut() {
        let shape = parameter.value().shape().to_vec();
        let noise = CpuTensor::randn(&shape, rng);
        *parameter.value_mut() = noise.mul(&CpuTensor::scalar(0.5));
    }
}

/// Les tailles de l'attention latente pour le modèle minuscule : deux
/// têtes de 4, un latent de 4, une partie tournée de 2.
pub(crate) const TINY_LATENT: LatentDims = LatentDims {
    latent: 4,
    query_latent: 6,
    rope: 2,
};

/// Le socle moderne avec l'attention latente.
pub(crate) const LATENT: Architecture =
    Architecture::MODERN.with_attention(AttentionKind::Latent(TINY_LATENT));

/// Les tailles du mélange d'experts pour le modèle minuscule.
pub(crate) const TINY_EXPERTS: ExpertDims = ExpertDims {
    shared: 1,
    routed: 4,
    active: 2,
    hidden: 8,
    balance: Balance::None,
};

/// Le socle moderne avec un mélange d'experts.
pub(crate) const EXPERTS: Architecture = Architecture::MODERN.with_experts(TINY_EXPERTS);

/// Le socle moderne avec le module de prédiction multi-token.
pub(crate) const MULTI_TOKEN: Architecture =
    Architecture::MODERN.with_multi_token(MultiToken::DEEPSEEK);

/// Le socle moderne dont la pile de blocs est traversée deux fois.
pub(crate) const LOOPED: Architecture = Architecture::MODERN.with_passes(2);

/// Trois couches, dont la dernière emprunte ses clés et ses valeurs à
/// la deuxième.
pub(crate) const SHARED_KEYS: GptConfig = GptConfig {
    layers: 3,
    architecture: Architecture::MODERN.with_own_keys(2),
    ..TINY
};

/// GPT-2, puis chaque pièce moderne seule, puis les trois ensemble,
/// puis l'attention latente, le mélange d'experts, la prédiction
/// multi-token ou la double traversée par-dessus.
pub(crate) const ARCHITECTURES: [Architecture; 9] = [
    Architecture::GPT2,
    Architecture {
        norm: NormKind::Rms,
        ..Architecture::GPT2
    },
    Architecture {
        feed_forward: FeedForwardKind::SwiGlu,
        ..Architecture::GPT2
    },
    Architecture {
        positions: PositionKind::Rotary,
        ..Architecture::GPT2
    },
    Architecture::MODERN,
    LATENT,
    EXPERTS,
    MULTI_TOKEN,
    LOOPED,
];

fn logits_of(model: &Gpt<B>, tokens: &[u32], batch: usize) -> Vec<f64> {
    let backend = Cpu::new();
    let tape = Tape::new(&backend);
    let logits = model.logits(&tape, tokens, batch);
    tape.value(logits).data().to_vec()
}

#[test]
fn a_quantized_output_changes_logits_little_and_alike_with_or_without_cache() {
    use crate::generation::Generator;
    let backend = Cpu::new();
    let tokens = [1, 2, 3, 4];
    let mut model = Gpt::<B>::new(&backend, TINY, &mut StdRng::seed_from_u64(0));
    let exact = logits_of(&model, &tokens, 1);
    model.quantize_output(&backend, Format::Int8);
    assert!(model.token_embedding.output.is_some());
    let quantized = logits_of(&model, &tokens, 1);
    let largest = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for (e, q) in exact.iter().zip(&quantized) {
        assert!((e - q).abs() < 0.02 * largest, "{e} contre {q}");
    }
    assert_ne!(exact, quantized);
    // La génération, qui passe par le cache, lit la même sortie.
    let mut generator = Generator::new(&backend, &model, &tokens);
    let last = generator.logits().to_vec();
    let vocab = TINY.vocab;
    for (g, q) in last.iter().zip(&quantized[(tokens.len() - 1) * vocab..]) {
        assert!((g - q).abs() < 1e-9, "{g} contre {q}");
    }
}

/// Une mémoire minuscule, avec tout : porte et convolution.
fn memory_config(storage: MemoryStorage) -> GptConfig {
    GptConfig {
        context: 8,
        layers: 2,
        architecture: Architecture::MODERN.with_memory(MemoryDims {
            heads: 2,
            storage,
            ..MemoryDims::engram(8, 17)
        }),
        ..TINY
    }
}

#[test]
fn a_memory_reads_the_same_with_or_without_cache() {
    let backend = Cpu::new();
    let tokens = [1, 2, 3, 2, 3, 4, 2, 3];
    for storage in [MemoryStorage::Dense, MemoryStorage::Sparse] {
        let config = memory_config(storage);
        let mut rng = StdRng::seed_from_u64(4);
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        let full = logits_of(&model, &tokens, 1);
        // La convolution compte : sans elle, les logits changent.
        let without_conv = {
            let mut model = Gpt::<B>::new(&backend, config, &mut StdRng::seed_from_u64(4));
            randomize(&mut model, &mut StdRng::seed_from_u64(4));
            let memory = model.memory.as_mut().expect("mémoire");
            memory.conv = Some(Parameter::zeros(&backend, &[4, 8]));
            logits_of(&model, &tokens, 1)
        };
        assert_ne!(without_conv, full);
        // Par morceaux, à travers le cache : trois, un, puis quatre.
        let mut cache = model.cache(1);
        let id = cache.add_sequence();
        let mut cached = Vec::new();
        for chunk in [&tokens[..3], &tokens[3..4], &tokens[4..]] {
            let tape = Tape::new(&backend);
            let logits = model.forward_cached(&tape, &mut cache, &[(id, chunk)]);
            cached.extend_from_slice(tape.value(logits).data());
        }
        for (a, b) in full.iter().zip(&cached) {
            assert!((a - b).abs() < 1e-9, "{storage:?} : {a} contre {b}");
        }
        // Deux séquences à la fois, par morceaux de tailles différentes.
        let other = [4, 4, 1, 2, 3, 1, 4, 2];
        let both: Vec<u32> = tokens.iter().chain(&other).copied().collect();
        let full_both = logits_of(&model, &both, 2);
        let mut cache = model.cache(2);
        let (a, b) = (cache.add_sequence(), cache.add_sequence());
        let (mut first, mut second) = (Vec::new(), Vec::new());
        for (x, y) in [(0..3, 0..5), (3..4, 5..6), (4..8, 6..8)] {
            let tape = Tape::new(&backend);
            let logits = model.forward_cached(
                &tape,
                &mut cache,
                &[(a, &tokens[x.clone()]), (b, &other[y.clone()])],
            );
            let values = tape.value(logits).data().to_vec();
            let split = x.len() * config.vocab;
            first.extend_from_slice(&values[..split]);
            second.extend_from_slice(&values[split..]);
        }
        let cached_both: Vec<f64> = first.into_iter().chain(second).collect();
        for (a, b) in full_both.iter().zip(&cached_both) {
            assert!(
                (a - b).abs() < 1e-9,
                "{storage:?}, deux séquences : {a} contre {b}"
            );
        }
        // La table déplacée en mémoire centrale lit les mêmes lignes.
        if storage == MemoryStorage::Sparse {
            let mut moved = Gpt::<B>::new(&backend, config, &mut StdRng::seed_from_u64(4));
            randomize(&mut moved, &mut StdRng::seed_from_u64(4));
            let before = logits_of(&moved, &tokens, 1);
            let memory = moved.memory_mut().expect("mémoire");
            memory.move_to_host(&backend).expect("table creuse");
            for (a, b) in before.iter().zip(&logits_of(&moved, &tokens, 1)) {
                assert!((a - b).abs() < 1e-6, "en mémoire centrale : {a} contre {b}");
            }
        }
        // Sans la mémoire, les logits ne sont plus les mêmes.
        let mut plain = config;
        plain.architecture.memory = None;
        let without = Gpt::<B>::new(&backend, plain, &mut StdRng::seed_from_u64(4));
        assert_ne!(logits_of(&without, &tokens, 1), full);
    }
}

#[test]
fn a_memory_config_survives_the_file() {
    let config = memory_config(MemoryStorage::Host);
    assert_eq!(GptConfig::from_metadata(&config.to_metadata()), Ok(config));
}

#[test]
fn logits_have_one_score_per_token_and_position() {
    let backend = Cpu::new();
    let model = Gpt::<B>::new(&backend, TINY, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    let logits = model.logits(&tape, &[1, 2, 3, 4, 5, 6, 7, 8], 2);
    assert_eq!(tape.shape(logits), vec![2, 4, 11]);
}

#[test]
fn shorter_sequences_are_accepted() {
    let backend = Cpu::new();
    let model = Gpt::<B>::new(&backend, TINY, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    assert_eq!(
        tape.shape(model.logits(&tape, &[1, 2, 3], 1)),
        vec![1, 3, 11]
    );
}

#[test]
#[should_panic(expected = "le contexte maximal est 4")]
fn longer_sequences_are_rejected() {
    let backend = Cpu::new();
    let model = Gpt::<B>::new(&backend, TINY, &mut StdRng::seed_from_u64(0));
    let tape = Tape::new(&backend);
    model.logits(&tape, &[1, 2, 3, 4, 5], 1);
}

#[test]
fn parameter_count_matches_the_architecture() {
    let backend = Cpu::<f32>::new();
    let count = |config: GptConfig| {
        Gpt::new(&backend, config, &mut StdRng::seed_from_u64(0)).parameter_count(&backend)
    };
    let expected = |c: GptConfig| {
        let (dim, layers) = (c.embed, c.layers);
        let attention = 4 * (dim * dim + dim);
        let feed_forward = (dim * 4 * dim + 4 * dim) + (4 * dim * dim + dim);
        let block = attention + feed_forward + 2 * 2 * dim;
        // Pas de matrice de sortie : elle est partagée avec les embeddings.
        c.vocab * dim + c.context * dim + layers * block + 2 * dim
    };
    assert_eq!(count(TINY), expected(TINY));
    let small = GptConfig::small(257);
    assert_eq!(count(small), expected(small));
    assert_eq!(count(small), 834_432);
}

#[test]
fn large_configuration_has_about_eleven_million_parameters() {
    // Calculé sans construire le modèle, qui pèserait 40 Mo.
    let c = GptConfig::large(257);
    let dim = c.embed;
    let block = 4 * (dim * dim + dim) + (8 * dim * dim + 5 * dim) + 4 * dim;
    let total = c.vocab * dim + c.context * dim + c.layers * block + 2 * dim;
    assert_eq!(total, 10_844_544);
}

#[test]
fn xl_configuration_has_about_thirty_million_parameters() {
    let c = GptConfig::xl(8192);
    let dim = c.embed;
    let block = 4 * (dim * dim + dim) + (8 * dim * dim + 5 * dim) + 4 * dim;
    let total = c.vocab * dim + c.context * dim + c.layers * block + 2 * dim;
    assert_eq!(total, 29_676_544);
}

#[test]
fn model_cannot_see_the_future() {
    // Modifier le token à la position t ne doit changer aucun logit aux
    // positions avant t. C'est le test qui attrape une fuite du futur,
    // un bug qui donne une perte trop belle pour être vraie.
    const V: usize = 11;
    let backend = Cpu::new();
    for architecture in ARCHITECTURES {
        let mut rng = StdRng::seed_from_u64(0);
        let config = TINY.with_architecture(architecture);
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);

        let tokens = [3u32, 7, 1, 9];
        let before = logits_of(&model, &tokens, 1);
        for changed_position in 0..4 {
            let mut changed = tokens;
            changed[changed_position] = 5;
            let after = logits_of(&model, &changed, 1);
            let split = changed_position * V;
            assert_eq!(
                &before[..split],
                &after[..split],
                "{architecture:?} : changer la position {changed_position} a modifié le passé",
            );
            assert_ne!(
                &before[split..split + V],
                &after[split..split + V],
                "{architecture:?} : changer la position {changed_position} n'a rien modifié",
            );
        }
    }
}

#[test]
fn sequences_of_a_batch_do_not_mix() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::<B>::new(&backend, TINY, &mut rng);
    randomize(&mut model, &mut rng);

    let alone = logits_of(&model, &[3, 7, 1, 9], 1);
    let together = logits_of(&model, &[3, 7, 1, 9, 0, 0, 2, 4], 2);
    for (a, b) in alone.iter().zip(&together) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
}

#[test]
fn gradients_match_finite_differences() {
    // Tout le modèle d'un coup : embeddings partagés avec la sortie,
    // attention à deux têtes, LayerNorm, GELU, connexions résiduelles,
    // entropie croisée.
    check_gradients(TINY);
}

#[test]
fn gradients_match_finite_differences_for_every_architecture() {
    // Chaque pièce moderne seule, puis les trois ensemble.
    for architecture in &ARCHITECTURES[1..] {
        check_gradients(TINY.with_architecture(*architecture));
    }
}

/// Compare le gradient de chaque paramètre du modèle à
/// `(perte(p + ε) - perte(p - ε)) / 2ε`.
fn check_gradients(config: GptConfig) {
    const EPSILON: f64 = 1e-5;
    const TOLERANCE: f64 = 1e-6;
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(1);
    let mut model = Gpt::<B>::new(&backend, config, &mut rng);
    randomize(&mut model, &mut rng);

    let tokens = [3u32, 7, 1, 9, 0, 3, 3, 10];
    let targets = [7u32, 1, 9, 2, 3, 3, 10, 5];
    // La perte d'entraînement : celle du modèle, plus les termes
    // d'appoint des architectures qui en ont.
    let loss_of = |model: &Gpt<B>| {
        evaluate(&backend, model, |m, t| {
            m.training_loss(t, &tokens, &targets, 2)
        })
    };

    let analytic: Vec<Vec<f64>> = {
        let tape = Tape::new(&backend);
        let loss = model.training_loss(&tape, &tokens, &targets, 2);
        let grads = tape.backward(loss);
        model
            .parameters()
            .iter()
            .map(|p| match p.grad(tape.id(), &grads) {
                Some(grad) => grad.data().to_vec(),
                // Un paramètre sans gradient, comme un biais
                // d'aiguillage : sa dérivée doit être nulle.
                None => vec![0.0; p.value().data().len()],
            })
            .collect()
    };

    let mut checked = 0;
    for index in 0..analytic.len() {
        let original = model.parameters()[index].value().clone();
        for element in 0..original.data().len() {
            let mut perturbed = |delta: f64| {
                let mut data = original.data().to_vec();
                data[element] += delta;
                *model.parameters_mut()[index].value_mut() =
                    CpuTensor::from_vec(data, original.shape());
                loss_of(&model)
            };
            let numeric = (perturbed(EPSILON) - perturbed(-EPSILON)) / (2.0 * EPSILON);
            let analytic = analytic[index][element];
            let scale = analytic.abs().max(numeric.abs()).max(1.0);
            assert!(
                (analytic - numeric).abs() <= TOLERANCE * scale,
                "{:?}, paramètre {index}, élément {element} : \
                 analytique {analytic}, numérique {numeric}",
                config.architecture,
            );
            checked += 1;
        }
        *model.parameters_mut()[index].value_mut() = original;
    }
    assert_eq!(checked, model.parameter_count(&backend));
}

#[test]
fn gradients_are_not_all_tiny() {
    // Garde-fou du test précédent : si tous les gradients étaient
    // proches de zéro, la comparaison passerait sans rien vérifier.
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(1);
    let mut model = Gpt::<B>::new(&backend, TINY, &mut rng);
    randomize(&mut model, &mut rng);
    let tape = Tape::new(&backend);
    let loss = model.loss(&tape, &[3, 7, 1, 9], &[7, 1, 9, 2], 1);
    let grads = tape.backward(loss);
    // Ordre des paramètres : embeddings (0, 1), ln1 (2, 3), requête
    // (4, 5), clé (6, 7), …
    const KEY_BIAS: usize = 7;
    for (index, parameter) in model.parameters().iter().enumerate() {
        let grad = parameter.grad(tape.id(), &grads).unwrap();
        let largest = grad.data().iter().fold(0.0f64, |m, g| m.max(g.abs()));
        if index == KEY_BIAS {
            // Le biais des clés n'a aucun effet : il ajoute la même
            // quantité à tous les scores d'une même requête, et le
            // softmax est insensible à un décalage commun. Son gradient
            // est donc exactement nul, par construction.
            assert!(largest < 1e-12, "biais des clés : gradient max {largest}");
        } else {
            assert!(largest > 1e-3, "paramètre {index} : gradient max {largest}");
        }
    }
}

#[test]
fn initial_loss_is_close_to_ln_vocab() {
    // Légèrement au-dessus de ln(257) ≈ 5,549 : les logits ne sont pas
    // exactement nuls, ils ont un petit écart-type hérité des embeddings.
    let backend = Cpu::<f32>::new();
    let mut rng = StdRng::seed_from_u64(0);
    let config = GptConfig::small(257);
    let model = Gpt::new(&backend, config, &mut rng);
    let dataset = Dataset::from_tokens((0..2000).map(|i| (i * 7 % 257) as u16).collect());
    let (tokens, targets) = dataset.sample_batch(&mut rng, 2, 16).examples();
    let loss = evaluate(&backend, &model, |m, t| m.loss(t, &tokens, &targets, 2));
    assert!(
        (loss.to_f64() - 257f64.ln()).abs() < 0.1,
        "perte initiale {loss}"
    );
}

#[test]
fn overfits_a_single_batch() {
    // Avec l'attention de GPT-2, puis avec l'attention latente.
    let backend = Cpu::<f64>::new();
    for architecture in [Architecture::GPT2, LATENT, EXPERTS, MULTI_TOKEN] {
        let mut rng = StdRng::seed_from_u64(0);
        let config = TINY.with_architecture(architecture);
        let mut model = Gpt::new(&backend, config, &mut rng);
        let mut optimizer = AdamW::new(0.01, 0.0);
        // Deux séquences sans régularité : il faut les apprendre par cœur.
        let tokens = [3u32, 7, 1, 9, 0, 3, 3, 10];
        let targets = [7u32, 1, 9, 2, 3, 3, 10, 5];

        for _ in 0..400 {
            train_step(&backend, &mut model, &mut optimizer, |m, t| {
                m.training_loss(t, &tokens, &targets, 2)
            });
        }
        let loss = evaluate(&backend, &model, |m, t| m.loss(t, &tokens, &targets, 2));
        assert!(
            loss < 0.05,
            "{architecture:?} : perte après sur-apprentissage {loss}"
        );
    }
}

#[test]
fn configuration_survives_metadata() {
    let config = GptConfig::large(257);
    let restored = GptConfig::from_metadata(&config.to_metadata()).unwrap();
    assert_eq!(format!("{restored:?}"), format!("{config:?}"));
    assert!(GptConfig::from_metadata(&BTreeMap::new()).is_err());
}

#[test]
fn a_looped_model_is_a_deeper_model_with_tied_layers() {
    // Une couche traversée deux fois calcule ce que calculent deux
    // couches aux poids identiques.
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut looped = Gpt::<B>::new(&backend, TINY.with_architecture(LOOPED), &mut rng);
    randomize(&mut looped, &mut rng);
    let deep = GptConfig {
        layers: 2,
        ..TINY.with_architecture(Architecture::MODERN)
    };
    let mut deep = Gpt::<B>::new(&backend, deep, &mut rng);
    let shared: BTreeMap<String, CpuTensor<f64>> = looped
        .parameter_names()
        .into_iter()
        .zip(looped.parameters())
        .map(|(name, parameter)| (name, parameter.value().clone()))
        .collect();
    let names = deep.parameter_names();
    for (name, parameter) in names.iter().zip(deep.parameters_mut()) {
        let source = name.replace("blocks.1.", "blocks.0.");
        *parameter.value_mut() = shared[&source].clone();
    }
    let tokens = [3u32, 7, 1, 9, 2, 2, 10, 0];
    let (a, b) = (logits_of(&looped, &tokens, 2), logits_of(&deep, &tokens, 2));
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-12, "{x} != {y}");
    }
    // Deux fois plus de calcul, pas un paramètre de plus.
    let single = Gpt::<B>::new(
        &backend,
        TINY.with_architecture(Architecture::MODERN),
        &mut rng,
    );
    assert_eq!(
        looped.parameter_count(&backend),
        single.parameter_count(&backend)
    );
}

#[test]
fn layers_that_borrow_keys_have_none_of_their_own() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let shared = Gpt::<B>::new(&backend, SHARED_KEYS, &mut rng);
    let full = GptConfig {
        architecture: Architecture::MODERN,
        ..SHARED_KEYS
    };
    let full = Gpt::<B>::new(&backend, full, &mut rng);
    // La troisième couche n'a ni clés ni valeurs : deux couches
    // affines de `C × C + C` en moins.
    let dim = SHARED_KEYS.embed;
    assert_eq!(
        full.parameter_count(&backend) - shared.parameter_count(&backend),
        2 * (dim * dim + dim)
    );
    let names = shared.parameter_names();
    assert!(
        names
            .iter()
            .any(|name| name == "blocks.1.attention.key.weight")
    );
    assert!(
        names
            .iter()
            .any(|name| name == "blocks.2.attention.query.weight")
    );
    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("blocks.2.attention.key"))
    );
    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("blocks.2.attention.value"))
    );
}

#[test]
fn a_model_with_shared_keys_cannot_see_the_future() {
    const V: usize = 11;
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::<B>::new(&backend, SHARED_KEYS, &mut rng);
    randomize(&mut model, &mut rng);
    let tokens = [3u32, 7, 1, 9];
    let before = logits_of(&model, &tokens, 1);
    for changed_position in 0..4 {
        let mut changed = tokens;
        changed[changed_position] = 5;
        let after = logits_of(&model, &changed, 1);
        let split = changed_position * V;
        assert_eq!(&before[..split], &after[..split]);
        assert_ne!(&before[split..split + V], &after[split..split + V]);
    }
}

#[test]
fn gradients_match_finite_differences_with_shared_keys() {
    // Les clés de la deuxième couche reçoivent le gradient de deux
    // attentions : la sienne et celle qui les lui emprunte.
    check_gradients(SHARED_KEYS);
}

#[test]
fn passes_and_shared_keys_travel_with_the_configuration() {
    let looped = TINY.with_architecture(LOOPED);
    for config in [looped, SHARED_KEYS] {
        let restored = GptConfig::from_metadata(&config.to_metadata()).unwrap();
        assert_eq!(restored.architecture, config.architecture);
    }
    // Un modèle ordinaire n'en parle pas : ses fichiers ne changent pas.
    let plain = TINY.with_architecture(Architecture::MODERN).to_metadata();
    assert!(!plain["config"].contains("passes"));
    assert!(!plain["config"].contains("own_keys"));

    let refused = [
        (Architecture::MODERN.with_passes(0), "au moins une fois"),
        (Architecture::MODERN.with_own_keys(0), "entre 1 et 1"),
        (Architecture::MODERN.with_own_keys(2), "entre 1 et 1"),
        (LATENT.with_own_keys(1), "l'attention `standard`"),
    ];
    for (architecture, message) in refused {
        let error = TINY.with_architecture(architecture).validate().unwrap_err();
        assert!(error.contains(message), "{error}");
    }
}

#[test]
fn architecture_survives_the_metadata_round_trip() {
    for architecture in ARCHITECTURES {
        let config = TINY.with_architecture(architecture);
        let restored = GptConfig::from_metadata(&config.to_metadata()).unwrap();
        assert_eq!(restored.architecture, architecture);
    }
}

#[test]
fn configuration_travels_as_one_json_field() {
    let config = TINY.with_architecture(Architecture::MODERN);
    let metadata = config.to_metadata();
    assert_eq!(metadata.len(), 1);
    let json: Value = serde_json::from_str(&metadata["config"]).unwrap();
    assert_eq!(json["layers"], 1);
    assert_eq!(json["architecture"]["positions"], "rope");
    assert_eq!(GptConfig::from_metadata(&metadata).unwrap(), config);
}

#[test]
fn missing_pieces_are_those_of_gpt2() {
    let metadata = |json: &str| BTreeMap::from([("config".to_string(), json.to_string())]);
    let sizes = r#""vocab":11,"context":4,"layers":1,"heads":2,"embed":8"#;
    // Pas d'architecture du tout, puis une seule pièce nommée.
    let plain = GptConfig::from_metadata(&metadata(&format!("{{{sizes}}}"))).unwrap();
    assert_eq!(plain, TINY);
    let partial = format!(r#"{{{sizes},"architecture":{{"norm":"rms"}}}}"#);
    let partial = GptConfig::from_metadata(&metadata(&partial)).unwrap();
    assert_eq!(partial.architecture.norm, NormKind::Rms);
    assert_eq!(partial.architecture.positions, PositionKind::Learned);
}

#[test]
fn unreadable_configurations_are_errors() {
    let metadata = |json: &str| BTreeMap::from([("config".to_string(), json.to_string())]);
    let sizes = r#""vocab":11,"context":4,"layers":1,"heads":2,"embed":8"#;
    for (json, expected) in [
        ("{".to_string(), "illisible"),
        (r#"{"vocab":11}"#.to_string(), "context"),
        (
            format!(r#"{{{sizes},"architecture":{{"norm":"batch"}}}}"#),
            "rms",
        ),
        (
            format!(r#"{{{sizes},"architecture":{{"norm":3}}}}"#),
            "norm",
        ),
    ] {
        let error = GptConfig::from_metadata(&metadata(&json)).unwrap_err();
        assert!(error.contains(expected), "{json} : {error}");
    }
    assert!(GptConfig::from_metadata(&BTreeMap::new()).is_err());
}

#[test]
fn flat_metadata_of_early_files_are_still_read() {
    // Un champ par réglage, sans architecture : c'était alors GPT-2.
    let mut metadata: BTreeMap<String, String> = [
        ("vocab", "11"),
        ("context", "4"),
        ("layers", "1"),
        ("heads", "2"),
        ("embed", "8"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect();
    assert_eq!(GptConfig::from_metadata(&metadata).unwrap(), TINY);
    // Les pièces, quand elles sont nommées, sont lues.
    metadata.insert("positions".into(), "rope".into());
    let config = GptConfig::from_metadata(&metadata).unwrap();
    assert_eq!(config.architecture.positions, PositionKind::Rotary);
    metadata.insert("norm".into(), "batch".into());
    assert!(
        GptConfig::from_metadata(&metadata)
            .unwrap_err()
            .contains("rms")
    );
}

#[test]
fn latent_attention_travels_with_its_sizes() {
    let config = TINY.with_architecture(LATENT);
    let metadata = config.to_metadata();
    let json: Value = serde_json::from_str(&metadata["config"]).unwrap();
    let attention = &json["architecture"]["attention"];
    assert_eq!(attention["kind"], "mla");
    assert_eq!(attention["latent"], 4);
    assert_eq!(attention["query_latent"], 6);
    assert_eq!(attention["rope"], 2);
    assert_eq!(GptConfig::from_metadata(&metadata).unwrap(), config);
    // L'attention classique se nomme aussi, et son absence la désigne.
    let json: Value = serde_json::from_str(&TINY.to_metadata()["config"]).unwrap();
    assert_eq!(json["architecture"]["attention"]["kind"], "standard");
}

#[test]
fn unreadable_attentions_are_errors() {
    let metadata = |attention: &str| {
        let json = format!(
            r#"{{"vocab":11,"context":4,"layers":1,"heads":2,"embed":8,
                "architecture":{{"positions":"rope","attention":{attention}}}}}"#
        );
        BTreeMap::from([("config".to_string(), json)])
    };
    for (attention, expected) in [
        (r#"{"kind":"flash"}"#, "standard, mla"),
        (r#"{"latent":4}"#, "standard, mla"),
        (r#"{"kind":"mla","latent":4,"rope":2}"#, "query_latent"),
        (
            r#"{"kind":"mla","latent":4,"query_latent":6,"rope":3}"#,
            "paire",
        ),
        (
            r#"{"kind":"mla","latent":0,"query_latent":6,"rope":2}"#,
            "nulles",
        ),
    ] {
        let error = GptConfig::from_metadata(&metadata(attention)).unwrap_err();
        assert!(error.contains(expected), "{attention} : {error}");
    }
    let valid = r#"{"kind":"mla","latent":4,"query_latent":6,"rope":2}"#;
    assert!(GptConfig::from_metadata(&metadata(valid)).is_ok());
}

#[test]
fn latent_attention_needs_rotary_positions() {
    let learned = Architecture::GPT2.with_attention(AttentionKind::Latent(TINY_LATENT));
    let error = TINY.with_architecture(learned).validate().unwrap_err();
    assert!(error.contains("rope"), "{error}");
    assert!(TINY.with_architecture(LATENT).validate().is_ok());
    for architecture in ARCHITECTURES {
        assert_eq!(TINY.with_architecture(architecture).validate(), Ok(()));
    }
    // La partie tournée a sa propre taille, qui n'est pas celle des têtes.
    assert_eq!(TINY.with_architecture(LATENT).rotary_dim(), 2);
    assert_eq!(TINY.with_architecture(Architecture::MODERN).rotary_dim(), 4);
}

#[test]
#[should_panic(expected = "demande les positions `rope`")]
fn latent_model_without_rope_is_refused() {
    let learned = Architecture::GPT2.with_attention(AttentionKind::Latent(TINY_LATENT));
    Gpt::<B>::new(
        &Cpu::new(),
        TINY.with_architecture(learned),
        &mut StdRng::seed_from_u64(0),
    );
}

#[test]
fn latent_attention_has_slightly_fewer_parameters() {
    // La configuration `xl`, calculée sans construire les modèles.
    let c = GptConfig::xl(8192);
    let (dim, heads) = (c.embed, c.heads);
    let dims = LatentDims::for_model(dim, heads);
    let standard = 4 * (dim * dim + dim);
    let latent = (dim * dims.query_latent + dims.query_latent)
        + dims.query_latent * dim
        + dims.query_latent * heads * dims.rope
        + (dim * dims.latent + dims.latent)
        + 2 * dims.latent * dim
        + dim * dims.rope
        + (dim * dim + dim);
    assert_eq!(standard, 1_050_624);
    assert_eq!(latent, 967_680);
    // Et sur un modèle construit, la différence est bien celle-là.
    let backend = Cpu::<f32>::new();
    let small = GptConfig {
        context: 8,
        layers: 2,
        ..TINY
    };
    let count = |architecture| {
        Gpt::new(
            &backend,
            small.with_architecture(architecture),
            &mut StdRng::seed_from_u64(0),
        )
        .parameter_count(&backend)
    };
    let per_layer = |d: LatentDims| {
        (8 * d.query_latent + d.query_latent)
            + d.query_latent * 8
            + d.query_latent * 2 * d.rope
            + (8 * d.latent + d.latent)
            + 2 * d.latent * 8
            + 8 * d.rope
            + (8 * 8 + 8)
    };
    assert_eq!(
        count(Architecture::MODERN) as isize - count(LATENT) as isize,
        2 * (4 * (8 * 8 + 8) as isize - per_layer(TINY_LATENT) as isize),
    );
}

#[test]
fn experts_travel_with_their_sizes_and_balance() {
    for balance in [
        Balance::None,
        Balance::Bias { rate: 0.001 },
        Balance::Loss { weight: 0.01 },
    ] {
        let dims = ExpertDims {
            balance,
            ..TINY_EXPERTS
        };
        let config = TINY.with_architecture(Architecture::MODERN.with_experts(dims));
        let metadata = config.to_metadata();
        let json: Value = serde_json::from_str(&metadata["config"]).unwrap();
        let experts = &json["architecture"]["experts"];
        assert_eq!(experts["routed"], 4);
        assert_eq!(experts["balance"]["kind"], balance.name());
        assert_eq!(GptConfig::from_metadata(&metadata).unwrap(), config);
    }
    // Un modèle sans experts n'en parle pas.
    let json: Value = serde_json::from_str(&TINY.to_metadata()["config"]).unwrap();
    assert!(json["architecture"]["experts"].is_null());
}

#[test]
fn unreadable_experts_are_errors() {
    let metadata = |experts: &str| {
        let json = format!(
            r#"{{"vocab":11,"context":4,"layers":1,"heads":2,"embed":8,
                "architecture":{{"feed_forward":"swiglu","experts":{experts}}}}}"#
        );
        BTreeMap::from([("config".to_string(), json)])
    };
    let sizes = r#""shared":1,"routed":4,"active":2,"hidden":8"#;
    for (experts, expected) in [
        (r#"{"shared":1}"#.to_string(), "routed"),
        (
            format!(r#"{{{sizes},"balance":{{"kind":"hasard"}}}}"#),
            "none, bias, loss",
        ),
        (
            format!(r#"{{{sizes},"balance":{{"kind":"bias"}}}}"#),
            "rate",
        ),
        (
            r#"{"shared":1,"routed":2,"active":3,"hidden":8}"#.to_string(),
            "actifs",
        ),
    ] {
        let error = GptConfig::from_metadata(&metadata(&experts)).unwrap_err();
        assert!(error.contains(expected), "{experts} : {error}");
    }
    assert!(GptConfig::from_metadata(&metadata(&format!("{{{sizes}}}"))).is_ok());
    // Chaque expert est un SwiGLU : le dire autrement est une erreur.
    let gelu = Architecture::GPT2.with_experts(TINY_EXPERTS);
    let error = TINY.with_architecture(gelu).validate().unwrap_err();
    assert!(error.contains("swiglu"), "{error}");
}

#[test]
fn experts_multiply_the_parameters_but_not_the_active_ones() {
    // La configuration `xl`, calculée sans construire les modèles.
    let dim = 512;
    let dims = ExpertDims::for_model(dim);
    let swiglu = |hidden: usize| 3 * dim * hidden + 2 * hidden + dim;
    let reference = swiglu(1368);
    let expert = swiglu(dims.hidden);
    let active = (dims.shared + dims.active) * expert;
    let total = (dims.shared + dims.routed) * expert + dim * dims.routed + dims.routed;
    // Un token traverse autant de paramètres qu'avant, à 0,1 % près…
    assert!((active as f64 / reference as f64 - 1.0).abs() < 1e-3);
    // … mais la couche en possède trois fois plus.
    assert!((total as f64 / reference as f64 - 3.0).abs() < 0.01);
}

#[test]
fn training_loss_adds_the_balance_penalty() {
    let backend = Cpu::new();
    let tokens = [3u32, 7, 1, 9];
    let targets = [7u32, 1, 9, 2];
    let losses = |balance: Balance| {
        let dims = ExpertDims {
            balance,
            ..TINY_EXPERTS
        };
        let config = TINY.with_architecture(Architecture::MODERN.with_experts(dims));
        let mut rng = StdRng::seed_from_u64(0);
        let mut model = Gpt::<B>::new(&backend, config, &mut rng);
        randomize(&mut model, &mut rng);
        let tape = Tape::new(&backend);
        let plain = tape.value(model.loss(&tape, &tokens, &targets, 1)).item();
        let tape = Tape::new(&backend);
        let training = model.training_loss(&tape, &tokens, &targets, 1);
        (plain, tape.value(training).item())
    };
    let (plain, training) = losses(Balance::None);
    assert_eq!(plain, training);
    let (plain, training) = losses(Balance::Loss { weight: 0.5 });
    // `Σ f_i × P_i` vaut au moins 1 : la pénalité vaut au moins son poids.
    assert!(training - plain >= 0.5 - 1e-9, "{plain} puis {training}");
    assert!(training - plain < 1.5, "{plain} puis {training}");
}

#[test]
fn expert_load_is_reported_by_layer() {
    let backend = Cpu::new();
    let config = GptConfig {
        layers: 2,
        ..TINY.with_architecture(EXPERTS)
    };
    let mut model = Gpt::<B>::new(&backend, config, &mut StdRng::seed_from_u64(0));
    logits_of(&model, &[3, 7, 1, 9, 0, 3, 3, 10], 2);
    let load = model.take_expert_load();
    assert_eq!(load.len(), 2);
    for layer in &load {
        // Huit tokens, deux experts chacun.
        assert_eq!(layer.iter().sum::<u64>(), 16);
    }
    logits_of(&model, &[3, 7, 1, 9], 1);
    model.reset_expert_load();
    assert!(model.take_expert_load().iter().flatten().all(|&c| c == 0));
    // La version dense donne les mêmes logits.
    let sparse = logits_of(&model, &[3, 7, 1, 9], 1);
    model.set_dense_experts(true);
    let dense = logits_of(&model, &[3, 7, 1, 9], 1);
    for (s, d) in sparse.iter().zip(&dense) {
        assert!((s - d).abs() < 1e-12, "{s} != {d}");
    }
    // Un modèle sans experts n'a rien à rapporter.
    let plain = Gpt::<B>::new(&backend, TINY, &mut StdRng::seed_from_u64(0));
    assert!(plain.take_expert_load().is_empty());
}

/// Un modèle minuscule de contexte 4 avec prédiction multi-token, aux
/// poids agrandis.
fn multi_token_model(weights: MultiToken) -> Gpt<B> {
    let mut rng = StdRng::seed_from_u64(2);
    let config = TINY.with_architecture(Architecture::MODERN.with_multi_token(weights));
    let mut model = Gpt::<B>::new(&Cpu::new(), config, &mut rng);
    randomize(&mut model, &mut rng);
    model
}

/// Perte du modèle, puis perte d'entraînement.
fn both_losses(model: &Gpt<B>, tokens: &[u32], targets: &[u32], batch: usize) -> (f64, f64) {
    let backend = Cpu::new();
    let plain = evaluate(&backend, model, |m, t| m.loss(t, tokens, targets, batch));
    let training = evaluate(&backend, model, |m, t| {
        m.training_loss(t, tokens, targets, batch)
    });
    (plain, training)
}

#[test]
fn multi_token_loss_is_added_with_its_weight() {
    let tokens = [3u32, 7, 1, 9];
    let targets = [7u32, 1, 9, 2];
    let model = multi_token_model(MultiToken::DEEPSEEK);
    let (plain, early) = both_losses(&model, &tokens, &targets, 1);
    let extra = (early - plain) / 0.3;
    // Une perte d'entropie croisée, de l'ordre de ln(11) ou plus avec
    // des poids agrandis : nettement positive.
    assert!(extra > 0.5, "perte du module : {extra}");
    // Passé les deux tiers de l'entraînement, le poids tombe à 0,1.
    model.set_training_progress(0.5);
    assert_eq!(both_losses(&model, &tokens, &targets, 1).1, early);
    model.set_training_progress(0.7);
    let (_, late) = both_losses(&model, &tokens, &targets, 1);
    assert!(((late - plain) - 0.1 * extra).abs() < 1e-12);
    // Un poids nul, ou pas de module : la perte d'entraînement est
    // celle du modèle.
    let silent = multi_token_model(MultiToken {
        early_weight: 0.0,
        late_weight: 0.0,
    });
    let (plain, training) = both_losses(&silent, &tokens, &targets, 1);
    assert_eq!(plain, training);
}

#[test]
fn multi_token_targets_stay_within_their_sequence() {
    // La perte du module sur deux séquences est la moyenne de ses
    // pertes sur chacune : la dernière position d'une séquence ne va
    // pas chercher sa cible dans la suivante.
    let model = multi_token_model(MultiToken::DEEPSEEK);
    let (first, second) = ([3u32, 7, 1, 9], [0u32, 3, 3, 10]);
    let (first_targets, second_targets) = ([7u32, 1, 9, 2], [3u32, 3, 10, 5]);
    let extra = |tokens: &[u32], targets: &[u32], batch| {
        let (plain, training) = both_losses(&model, tokens, targets, batch);
        training - plain
    };
    let together = extra(
        &[first, second].concat(),
        &[first_targets, second_targets].concat(),
        2,
    );
    let apart = (extra(&first, &first_targets, 1) + extra(&second, &second_targets, 1)) / 2.0;
    assert!((together - apart).abs() < 1e-12, "{together} != {apart}");
    // Elle dépend de la cible lointaine : changer la dernière cible, que
    // le modèle principal ne voit qu'à la dernière position, la modifie.
    let changed = extra(&first, &[7, 1, 9, 4], 1);
    assert!((changed - extra(&first, &first_targets, 1)).abs() > 1e-6);
    // Une séquence d'un seul token n'a pas de cible lointaine.
    assert_eq!(extra(&[3], &[7], 1), 0.0);
}

#[test]
fn multi_token_module_is_left_out_for_inference() {
    let backend = Cpu::new();
    let model = multi_token_model(MultiToken::DEEPSEEK);
    let config = model.config();
    assert!(
        model
            .parameter_names()
            .contains(&"multi_token.proj.weight".to_string())
    );
    // La configuration voyage avec ses poids…
    let restored = GptConfig::from_metadata(&config.to_metadata()).unwrap();
    assert_eq!(restored, config);
    assert_eq!(
        restored.architecture.multi_token,
        Some(MultiToken::DEEPSEEK)
    );
    // … et le fichier de poids se lit sans le module.
    let dir = std::env::temp_dir().join(format!("smallem-mtp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    crate::io::save_model(&path, &backend, &model, config.to_metadata()).unwrap();
    let file = crate::io::SafeTensors::read(&path).unwrap();
    let inference = GptConfig::from_metadata(&file.metadata)
        .unwrap()
        .for_inference();
    assert_eq!(inference.architecture, Architecture::MODERN);
    let mut lean = Gpt::<B>::new(&backend, inference, &mut StdRng::seed_from_u64(9));
    crate::io::load_parameters(&file, &backend, &mut lean).unwrap();
    assert!(
        !lean
            .parameter_names()
            .iter()
            .any(|name| name.starts_with("multi_token"))
    );
    // Le fichier range les poids en `f32` : l'égalité vaut à cette
    // précision.
    let (lean, full) = (
        logits_of(&lean, &[3, 7, 1, 9], 1),
        logits_of(&model, &[3, 7, 1, 9], 1),
    );
    for (l, f) in lean.iter().zip(&full) {
        assert!((l - f).abs() < 1e-5, "{l} != {f}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn unreadable_multi_token_weights_are_errors() {
    let metadata = |multi_token: &str| {
        let json = format!(
            r#"{{"vocab":11,"context":4,"layers":1,"heads":2,"embed":8,
                "architecture":{{"multi_token":{multi_token}}}}}"#
        );
        BTreeMap::from([("config".to_string(), json)])
    };
    let error = GptConfig::from_metadata(&metadata(r#"{"early_weight":0.3}"#)).unwrap_err();
    assert!(error.contains("late_weight"), "{error}");
    let negative = r#"{"early_weight":-1.0,"late_weight":0.1}"#;
    assert!(GptConfig::from_metadata(&metadata(negative)).is_err());
    let valid = r#"{"early_weight":0.3,"late_weight":0.1}"#;
    assert!(GptConfig::from_metadata(&metadata(valid)).is_ok());
    assert_eq!(MultiToken::DEEPSEEK.weight_at(0.0), 0.3);
    assert_eq!(MultiToken::DEEPSEEK.weight_at(0.9), 0.1);
}

#[test]
fn rotary_model_has_no_position_table() {
    let backend = Cpu::new();
    let build = |architecture| {
        Gpt::<B>::new(
            &backend,
            TINY.with_architecture(architecture),
            &mut StdRng::seed_from_u64(0),
        )
    };
    let learned = build(Architecture::GPT2);
    let rotary = build(ARCHITECTURES[3]);
    assert!(
        learned
            .parameter_names()
            .contains(&"position_embedding.weight".to_string())
    );
    assert!(
        !rotary
            .parameter_names()
            .iter()
            .any(|name| name.contains("position"))
    );
    // Exactement la table des positions en moins : contexte × dimension.
    assert_eq!(
        learned.parameter_count(&backend) - rotary.parameter_count(&backend),
        TINY.context * TINY.embed,
    );
    assert_eq!(rotary.parameters().len(), rotary.parameter_names().len());
}

#[test]
fn swiglu_keeps_about_as_many_parameters_as_gelu() {
    // Calculé sans construire les modèles : par bloc, le feed-forward
    // GELU a 8C² + 5C paramètres, SwiGLU 3Ch + 2h + C avec h ≈ 8C/3.
    for dim in [128usize, 384, 512] {
        let gelu = 8 * dim * dim + 5 * dim;
        let hidden = crate::nn::swiglu_hidden(dim);
        let swiglu = 3 * dim * hidden + 2 * hidden + dim;
        let ratio = swiglu as f64 / gelu as f64;
        assert!((ratio - 1.0).abs() < 0.01, "C = {dim} : rapport {ratio}");
    }
}

#[test]
fn positions_matter_with_rope() {
    // Sans information de position, permuter deux tokens du passé ne
    // changerait pas le logit de la dernière position : l'attention
    // seule ne voit qu'un ensemble. Avec RoPE, il doit changer.
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::<B>::new(&backend, TINY.with_architecture(ARCHITECTURES[3]), &mut rng);
    randomize(&mut model, &mut rng);
    let last = |tokens: &[u32]| logits_of(&model, tokens, 1)[3 * TINY.vocab..].to_vec();
    assert_ne!(last(&[3, 7, 1, 9]), last(&[7, 3, 1, 9]));
}

#[test]
fn growing_the_vocabulary_keeps_the_old_logits() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(2);
    let mut model = Gpt::<B>::new(&backend, TINY, &mut rng);
    randomize(&mut model, &mut rng);
    let tokens = [3u32, 7, 1, 9];
    let logits = |model: &Gpt<B>| {
        let tape = Tape::new(&backend);
        tape.value(model.logits(&tape, &tokens, 1)).data().to_vec()
    };
    let before = logits(&model);
    model.grow_vocab(&backend, TINY.vocab + 3);
    let after = logits(&model);
    let vocab = TINY.vocab;
    for (position, row) in after.chunks(vocab + 3).enumerate() {
        for (a, b) in row[..vocab].iter().zip(&before[position * vocab..]) {
            assert!((a - b).abs() < 1e-12);
        }
        // Le logit d'un nouveau token est la moyenne des anciens.
        let mean: f64 = row[..vocab].iter().sum::<f64>() / vocab as f64;
        assert!((row[vocab] - mean).abs() < 1e-12);
    }
}

#[test]
#[should_panic(expected = "avant les adaptateurs")]
fn the_vocabulary_does_not_grow_under_adapters() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(2);
    let mut model = Gpt::<B>::new(&backend, TINY, &mut rng);
    model.add_adapters(&backend, 2, &mut rng);
    model.grow_vocab(&backend, TINY.vocab + 3);
}

#[test]
fn a_refused_configuration_is_an_error_for_try_new() {
    let learned = Architecture::GPT2.with_attention(AttentionKind::Latent(TINY_LATENT));
    let config = TINY.with_architecture(learned);
    let error = Gpt::<B>::try_new(&Cpu::new(), config, &mut StdRng::seed_from_u64(0))
        .err()
        .expect("configuration refusée");
    assert!(error.contains("rope"), "{error}");
    assert!(Gpt::<B>::try_new(&Cpu::new(), TINY, &mut StdRng::seed_from_u64(0)).is_ok());
}

#[test]
fn the_masked_loss_ignores_the_targets_it_does_not_score() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(2);
    let mut model = Gpt::<B>::new(&backend, TINY, &mut rng);
    randomize(&mut model, &mut rng);
    let tokens = [3u32, 7, 1, 9];
    let scored = [false, true, false, true];
    let loss = |tokens: &[u32], targets: &[u32]| {
        let tape = Tape::new(&backend);
        tape.value(model.masked_loss(&tape, tokens, targets, &scored, 1))
            .item()
    };
    let base = loss(&tokens, &[7, 1, 9, 2]);
    // Une autre cible là où l'on ne note pas : même perte.
    assert_eq!(base, loss(&tokens, &[0, 1, 5, 2]));
    // Une autre cible notée : autre perte.
    assert_ne!(base, loss(&tokens, &[7, 4, 9, 2]));
    // Un autre token lu en entrée, même non noté : autre perte, car
    // les positions suivantes le lisent.
    assert_ne!(base, loss(&[3, 7, 4, 9], &[7, 1, 9, 2]));
    // Et la perte est la moyenne des positions notées.
    let tape = Tape::new(&backend);
    let all = model.loss(&tape, &tokens, &[7, 1, 9, 2], 1);
    assert!(tape.value(all).item() != base);
}

#[test]
fn adapters_start_neutral_learn_alone_and_merge_exactly() {
    let backend = Cpu::new();
    let mut rng = StdRng::seed_from_u64(2);
    let config = GptConfig {
        layers: 2,
        ..TINY.with_architecture(Architecture::MODERN)
    };
    let mut model = Gpt::<B>::new(&backend, config, &mut rng);
    randomize(&mut model, &mut rng);
    let tokens = [3u32, 7, 1, 9];
    let logits = |model: &Gpt<B>| {
        let tape = Tape::new(&backend);
        tape.value(model.logits(&tape, &tokens, 1)).data().to_vec()
    };
    let before = logits(&model);
    let base: Vec<Vec<f64>> = model
        .parameters()
        .iter()
        .map(|p| p.value().data().to_vec())
        .collect();
    let trained = model.add_adapters(&backend, 2, &mut rng);
    // La table `[11, 8]` et 2 × 7 matrices, chacune `r × (entrée + sortie)`.
    let hidden = crate::nn::swiglu_hidden(8);
    let per_block = 4 * 2 * 16 + 3 * 2 * (8 + hidden);
    assert_eq!(trained, 2 * (11 + 8) + 2 * per_block);
    assert_eq!(logits(&model), before);

    let mut optimizer = AdamW::new(0.01, 0.0);
    for _ in 0..3 {
        train_step(&backend, &mut model, &mut optimizer, |m, t| {
            m.loss(t, &tokens, &[7, 1, 9, 2], 1)
        });
    }
    let adapted = logits(&model);
    assert_ne!(adapted, before);
    // Les paramètres d'origine n'ont pas bougé.
    let mut originals = model
        .parameters()
        .into_iter()
        .filter(|p| p.is_frozen())
        .map(|p| p.value().data().to_vec());
    for value in &base {
        assert_eq!(&originals.next().unwrap(), value);
    }
    model.merge_adapters(&backend);
    assert_eq!(model.parameters().len(), base.len());
    for (a, b) in logits(&model).iter().zip(&adapted) {
        assert!((a - b).abs() < 1e-12);
    }
}

/// Le changement moyen des logits après trois pas d'AdamW, pour un
/// modèle de largeur `width`, en paramétrisation standard ou sous μP
/// (de base 16).
fn logit_change(width: usize, mup: bool) -> f64 {
    let backend = Cpu::new();
    let config = GptConfig {
        vocab: 32,
        context: 8,
        layers: 2,
        heads: width / 8,
        embed: width,
        architecture: Architecture::MODERN,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::<B>::new(&backend, config, &mut rng);
    if mup {
        model.apply_mup(&backend, 16);
    }
    let tokens: Vec<u32> = (0..32).map(|_| rng.random_range(0..32)).collect();
    let targets: Vec<u32> = (0..32).map(|_| rng.random_range(0..32)).collect();
    let logits = |model: &Gpt<B>| {
        let tape = Tape::new(&backend);
        tape.value(model.logits(&tape, &tokens, 4)).data().to_vec()
    };
    let before = logits(&model);
    let mut optimizer = AdamW::new(1e-2, 0.0);
    for _ in 0..3 {
        train_step(&backend, &mut model, &mut optimizer, |m, t| {
            m.loss(t, &tokens, &targets, 4)
        });
    }
    let after = logits(&model);
    before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / before.len() as f64
}

#[test]
fn under_mup_the_logits_move_as_much_at_every_width() {
    // Le test des coordonnées de μP : sous la paramétrisation standard,
    // un même taux fait bouger les logits d'autant plus que le modèle
    // est large ; sous μP, autant à toute largeur.
    let standard: Vec<f64> = [16, 64, 256]
        .iter()
        .map(|&w| logit_change(w, false))
        .collect();
    let mup: Vec<f64> = [16, 64, 256]
        .iter()
        .map(|&w| logit_change(w, true))
        .collect();
    eprintln!("standard {standard:?}, μP {mup:?}");
    assert_eq!(standard[0], mup[0], "à la largeur de base, rien ne change");
    assert!(standard[2] / standard[0] > 3.0, "{standard:?}");
    let (low, high) = mup
        .iter()
        .fold((f64::MAX, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
    assert!(high / low < 1.5, "{mup:?}");
}
