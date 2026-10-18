//! Le Transformer sur GPU est comparé au Transformer sur CPU : mêmes poids,
//! mêmes entrées, donc mêmes sorties.
//!
//! `cargo oxide test -- --test gpt`

use rand::SeedableRng;
use rand::rngs::StdRng;
use smallem::autograd::Tape;
use smallem::backend::cpu::Cpu;
use smallem::backend::{Backend, Float, Primitives};
use smallem::data::Dataset;
use smallem::generation::{Generator, generate};
use smallem::model::{Gpt, GptConfig};
use smallem::nn::{
    Architecture, AttentionKind, Balance, ExpertDims, LatentDims, Module, MultiToken,
};
use smallem::optim::AdamW;
use smallem::sampling::Sampling;
use smallem::training::{evaluate, train_step, train_step_accumulated};
use smallem_cuda::Cuda;

const VOCAB: usize = 257;

/// Un corpus synthétique avec de la structure à apprendre.
fn dataset() -> Dataset {
    Dataset::from_tokens(
        (0..20_000u32)
            .map(|i| ((i * i + i / 3) % 97) as u16)
            .collect(),
    )
}

/// Logits d'un modèle neuf sur un lot fixe. La graine détermine à la fois
/// les poids et le lot : deux backends calculent exactement la même chose.
fn logits<B: Backend<Scalar = f32>>(backend: &B, config: GptConfig) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(3);
    let model = Gpt::new(backend, config, &mut rng);
    let batch = dataset().sample_batch(&mut rng, 4, config.context);
    let (tokens, _) = batch.examples();
    let tape = Tape::new(backend);
    let out = model.logits(&tape, &tokens, 4);
    assert_eq!(tape.shape(out), vec![4, config.context, config.vocab]);
    backend.to_vec(&tape.value(out))
}

#[test]
fn logits_match_cpu_on_the_small_configuration() {
    let config = GptConfig::small(VOCAB);
    let cpu = logits(&Cpu::<f32>::new(), config);
    let gpu = logits(&Cuda::new(0).unwrap(), config);

    let mut largest = 0.0f32;
    for (index, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!(
            (c - g).abs() <= 1e-4 * scale,
            "logit {index} : CPU {c}, GPU {g}",
        );
        largest = largest.max(c.abs());
    }
    // Les logits ne doivent pas être tous nuls, sinon la comparaison ne dit rien.
    assert!(largest > 0.1, "logits trop petits : max {largest}");
}

/// Entraîne un modèle neuf et renvoie la perte de chaque pas.
fn loss_curve<B: Backend<Scalar = f32>>(backend: &B, steps: usize) -> Vec<f64> {
    let config = GptConfig {
        vocab: VOCAB,
        context: 16,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture: Architecture::GPT2,
    };
    let mut rng = StdRng::seed_from_u64(5);
    let mut model = Gpt::new(backend, config, &mut rng);
    let mut optimizer = AdamW::new(3e-3, 0.1);
    let dataset = dataset();

    (0..steps)
        .map(|_| {
            let batch = dataset.sample_batch(&mut rng, 8, config.context);
            let (tokens, targets) = batch.examples();
            let loss = train_step(backend, &mut model, &mut optimizer, |m, t| {
                m.loss(t, &tokens, &targets, 8)
            });
            backend.item(&loss).to_f64()
        })
        .collect()
}

#[test]
fn gpu_follows_the_same_loss_curve_as_cpu() {
    const STEPS: usize = 20;
    let cpu = loss_curve(&Cpu::<f32>::new(), STEPS);
    let gpu = loss_curve(&Cuda::new(0).unwrap(), STEPS);

    assert!(
        cpu[STEPS - 1] < cpu[0] - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c - g).abs() < 2e-3,
            "pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}",
        );
    }
}

#[test]
fn mixed_precision_follows_the_exact_loss_curve() {
    // Les produits matriciels arrondis en bf16 ne donnent pas les mêmes
    // nombres, mais l'entraînement doit suivre la même trajectoire.
    const STEPS: usize = 30;
    let exact = loss_curve(&Cuda::new(0).unwrap(), STEPS);
    let mixed = loss_curve(&Cuda::new(0).unwrap().with_mixed_precision(), STEPS);
    assert!(
        mixed[STEPS - 1] < mixed[0] - 0.5,
        "la précision mixte n'apprend pas : {mixed:?}"
    );
    let mut largest = 0.0f64;
    for (step, (e, m)) in exact.iter().zip(&mixed).enumerate() {
        assert!(
            (e - m).abs() < 0.05,
            "pas {step} : f32 {e}, bf16 {m}\nf32 {exact:?}\nbf16 {mixed:?}",
        );
        largest = largest.max((e - m).abs());
    }
    // Des courbes identiques voudraient dire que le bf16 n'a pas servi.
    assert!(largest > 0.0, "courbes identiques");
}

/// Entraîne en accumulant trois lots par pas ; renvoie la perte et la norme
/// du gradient de chaque pas.
fn accumulated_curve<B: Backend<Scalar = f32>>(backend: &B, steps: usize) -> Vec<(f64, f64)> {
    let config = GptConfig {
        vocab: VOCAB,
        context: 16,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture: Architecture::GPT2,
    };
    let mut rng = StdRng::seed_from_u64(9);
    let mut model = Gpt::new(backend, config, &mut rng);
    let mut optimizer = AdamW::new(3e-3, 0.1);
    let dataset = dataset();

    (0..steps)
        .map(|_| {
            let batches: Vec<_> = (0..3)
                .map(|_| dataset.sample_batch(&mut rng, 4, config.context).examples())
                .collect();
            let (loss, norm) = train_step_accumulated(
                backend,
                &mut model,
                &mut optimizer,
                1.0,
                3,
                |m, t, index| m.loss(t, &batches[index].0, &batches[index].1, 4),
            );
            (backend.item(&loss).to_f64(), norm)
        })
        .collect()
}

#[test]
fn gpu_accumulates_gradients_like_cpu() {
    const STEPS: usize = 15;
    let cpu = accumulated_curve(&Cpu::<f32>::new(), STEPS);
    let gpu = accumulated_curve(&Cuda::new(0).unwrap(), STEPS);

    assert!(
        cpu[STEPS - 1].0 < cpu[0].0 - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    // L'écrêtage doit avoir servi, sinon il n'est pas comparé.
    assert!(cpu.iter().any(|&(_, norm)| norm > 1.0), "{cpu:?}");
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c.0 - g.0).abs() < 2e-3 && (c.1 - g.1).abs() < 2e-3 * c.1.max(1.0),
            "pas {step} : CPU {c:?}, GPU {g:?}",
        );
    }
}

#[test]
fn initial_loss_on_gpu_is_close_to_ln_vocab() {
    let gpu = Cuda::new(0).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let config = GptConfig::small(VOCAB);
    let model = Gpt::new(&gpu, config, &mut rng);
    let batch = dataset().sample_batch(&mut rng, 8, config.context);
    let (tokens, targets) = batch.examples();
    let loss = evaluate(&gpu, &model, |m, t| m.loss(t, &tokens, &targets, 8));
    assert!(
        (loss - (VOCAB as f32).ln()).abs() < 0.1,
        "perte initiale {loss}"
    );
}

#[test]
fn gpu_overfits_a_single_batch() {
    let gpu = Cuda::new(0).unwrap();
    let config = GptConfig {
        vocab: VOCAB,
        context: 16,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture: Architecture::GPT2,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(&gpu, config, &mut rng);
    let mut optimizer = AdamW::new(3e-3, 0.0);
    let batch = dataset().sample_batch(&mut rng, 4, config.context);
    let (tokens, targets) = batch.examples();

    for _ in 0..400 {
        train_step(&gpu, &mut model, &mut optimizer, |m, t| {
            m.loss(t, &tokens, &targets, 4)
        });
    }
    let loss = evaluate(&gpu, &model, |m, t| m.loss(t, &tokens, &targets, 4));
    assert!(loss < 0.05, "perte après sur-apprentissage : {loss}");
}

#[test]
fn gpu_model_cannot_see_the_future() {
    let gpu = Cuda::new(0).unwrap();
    let config = GptConfig {
        vocab: VOCAB,
        context: 8,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture: Architecture::GPT2,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(&gpu, config, &mut rng);
    // Poids plus grands que l'initialisation : une fuite serait visible.
    for parameter in model.parameters_mut() {
        let shape = gpu.shape(parameter.value()).to_vec();
        *parameter.value_mut() = gpu.mul(&gpu.randn(&shape, &mut rng), &gpu.scalar(0.5));
    }

    let run = |tokens: &[u32]| {
        let tape = Tape::new(&gpu);
        let out = model.logits(&tape, tokens, 1);
        gpu.to_vec(&tape.value(out))
    };
    let tokens = [3u32, 7, 1, 9, 200, 0, 42, 256];
    let before = run(&tokens);
    let mut changed = tokens;
    changed[5] = 99;
    let after = run(&changed);

    let split = 5 * VOCAB;
    assert_eq!(&before[..split], &after[..split], "le passé a changé");
    assert_ne!(
        &before[split..],
        &after[split..],
        "le présent n'a pas changé"
    );
}

#[test]
fn gpu_generates_the_same_tokens_with_and_without_cache() {
    let gpu = Cuda::new(0).unwrap();
    let config = GptConfig {
        vocab: VOCAB,
        context: 24,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture: Architecture::GPT2,
    };
    let mut rng = StdRng::seed_from_u64(0);
    let mut model = Gpt::new(&gpu, config, &mut rng);
    // Des poids plus grands que l'initialisation, pour des logits bien
    // séparés : le token le plus probable ne dépend pas des arrondis.
    for parameter in model.parameters_mut() {
        let shape = gpu.shape(parameter.value()).to_vec();
        *parameter.value_mut() = gpu.mul(&gpu.randn(&shape, &mut rng), &gpu.scalar(0.3));
    }
    let greedy = Sampling {
        temperature: 0.0,
        ..Sampling::default()
    };
    // 40 tokens pour un contexte de 24 : le cache sert, puis est abandonné.
    let cached = generate(&gpu, &model, &[3, 7, 1], 40, &greedy, &mut rng);
    let uncached = generate_uncached(&gpu, &model, &[3, 7, 1], 40, &greedy, &mut rng);
    assert_eq!(cached, uncached);
    // Les deux générations ne sont pas une simple répétition d'un token.
    let distinct: std::collections::BTreeSet<_> = cached.iter().collect();
    assert!(distinct.len() > 3, "{cached:?}");
}

/// Prolonge `prompt` de `length` tokens sans cache.
fn generate_uncached(
    gpu: &Cuda,
    model: &Gpt<Cuda>,
    prompt: &[u32],
    length: usize,
    sampling: &Sampling,
    rng: &mut StdRng,
) -> Vec<u32> {
    let mut generator = Generator::without_cache(gpu, model, prompt);
    generator.extend(length, sampling, rng);
    generator.into_tokens()
}

/// Le socle moderne avec l'attention latente, pour des têtes de 16.
const LATENT: Architecture =
    Architecture::MODERN.with_attention(AttentionKind::Latent(LatentDims {
        latent: 8,
        query_latent: 24,
        rope: 8,
    }));

/// Un petit modèle à l'architecture moderne, aux poids agrandis.
fn modern_model<B: Backend<Scalar = f32>>(backend: &B) -> (Gpt<B>, GptConfig) {
    model_with(backend, Architecture::MODERN)
}

/// Un petit modèle aux poids agrandis, de l'architecture demandée.
fn model_with<B: Backend<Scalar = f32>>(
    backend: &B,
    architecture: Architecture,
) -> (Gpt<B>, GptConfig) {
    let config = GptConfig {
        vocab: VOCAB,
        context: 24,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture,
    };
    let mut rng = StdRng::seed_from_u64(11);
    let mut model = Gpt::new(backend, config, &mut rng);
    for parameter in model.parameters_mut() {
        let shape = backend.shape(parameter.value()).to_vec();
        let noise: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| 0.3 * smallem::backend::standard_normal(&mut rng) as f32)
            .collect();
        *parameter.value_mut() = backend.from_vec(noise, &shape);
    }
    (model, config)
}

#[test]
fn modern_architecture_logits_match_cpu() {
    // RMSNorm, SwiGLU et RoPE d'un coup, avec les mêmes poids des deux côtés.
    let tokens: Vec<u32> = (0..48u32).map(|i| (i * i + 7 * i) % 251).collect();
    let run = |logits: Vec<f32>| logits;
    let (cpu_model, _) = modern_model(&Cpu::<f32>::new());
    let cpu_backend = Cpu::<f32>::new();
    let tape = Tape::new(&cpu_backend);
    let cpu = run(cpu_backend.to_vec(&tape.value(cpu_model.logits(&tape, &tokens, 2))));

    let gpu_backend = Cuda::new(0).unwrap();
    let (gpu_model, _) = modern_model(&gpu_backend);
    let tape = Tape::new(&gpu_backend);
    let gpu = run(gpu_backend.to_vec(&tape.value(gpu_model.logits(&tape, &tokens, 2))));

    let mut largest = 0.0f32;
    for (index, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!(
            (c - g).abs() <= 2e-4 * scale,
            "logit {index} : CPU {c}, GPU {g}"
        );
        largest = largest.max(c.abs());
    }
    assert!(largest > 1.0, "logits trop petits : max {largest}");
}

/// Entraîne un modèle moderne neuf et renvoie la perte de chaque pas.
fn modern_loss_curve<B: Backend<Scalar = f32>>(backend: &B, steps: usize) -> Vec<f64> {
    loss_curve_with(backend, Architecture::MODERN, steps)
}

/// Entraîne un modèle neuf de l'architecture demandée et renvoie la perte
/// de chaque pas.
fn loss_curve_with<B: Backend<Scalar = f32>>(
    backend: &B,
    architecture: Architecture,
    steps: usize,
) -> Vec<f64> {
    let config = GptConfig {
        vocab: VOCAB,
        context: 16,
        layers: 2,
        heads: 2,
        embed: 32,
        architecture,
    };
    let mut rng = StdRng::seed_from_u64(5);
    let mut model = Gpt::new(backend, config, &mut rng);
    let mut optimizer = AdamW::new(3e-3, 0.1);
    let dataset = dataset();
    (0..steps)
        .map(|_| {
            let batch = dataset.sample_batch(&mut rng, 8, config.context);
            let (tokens, targets) = batch.examples();
            let loss = train_step(backend, &mut model, &mut optimizer, |m, t| {
                m.training_loss(t, &tokens, &targets, 8)
            });
            backend.item(&loss).to_f64()
        })
        .collect()
}

#[test]
fn modern_architecture_follows_the_cpu_loss_curve() {
    const STEPS: usize = 20;
    let cpu = modern_loss_curve(&Cpu::<f32>::new(), STEPS);
    let gpu = modern_loss_curve(&Cuda::new(0).unwrap(), STEPS);
    // Perte initiale proche de ln(vocabulaire), puis apprentissage.
    assert!((cpu[0] - (VOCAB as f64).ln()).abs() < 0.2, "{cpu:?}");
    assert!(
        cpu[STEPS - 1] < cpu[0] - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c - g).abs() < 2e-3,
            "pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}"
        );
    }
}

#[test]
fn modern_generation_with_cache_matches_without_on_gpu() {
    let gpu = Cuda::new(0).unwrap();
    let (model, config) = modern_model(&gpu);
    let greedy = Sampling {
        temperature: 0.0,
        ..Sampling::default()
    };
    let mut rng = StdRng::seed_from_u64(0);
    // Dans le contexte (24) : avec et sans cache, mêmes tokens.
    let length = config.context - 3;
    let cached = generate(&gpu, &model, &[3, 7, 1], length, &greedy, &mut rng);
    let uncached = generate_uncached(&gpu, &model, &[3, 7, 1], length, &greedy, &mut rng);
    assert_eq!(cached, uncached);
    // Au-delà : le cache glisse sans erreur et produit le bon nombre de tokens.
    let long = generate(&gpu, &model, &[3, 7, 1], 60, &greedy, &mut rng);
    assert_eq!(long.len(), 63);
    assert_eq!(long[..cached.len()], cached[..]);
}

#[test]
fn looped_and_shared_key_logits_match_cpu() {
    // Ni la double traversée ni les clés empruntées n'ont de kernel à
    // elles : ce test vérifie le câblage, cache de génération compris.
    let tokens: Vec<u32> = (0..48u32).map(|i| (i * i + 7 * i) % 251).collect();
    let architectures = [
        ("deux traversées", Architecture::MODERN.with_passes(2)),
        ("clés partagées", Architecture::MODERN.with_own_keys(1)),
    ];
    for (name, architecture) in architectures {
        let cpu_backend = Cpu::<f32>::new();
        let (cpu_model, _) = model_with(&cpu_backend, architecture);
        let tape = Tape::new(&cpu_backend);
        let cpu = cpu_backend.to_vec(&tape.value(cpu_model.logits(&tape, &tokens, 2)));

        let gpu_backend = Cuda::new(0).unwrap();
        let (gpu_model, _) = model_with(&gpu_backend, architecture);
        let tape = Tape::new(&gpu_backend);
        let gpu = gpu_backend.to_vec(&tape.value(gpu_model.logits(&tape, &tokens, 2)));
        for (index, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            let scale = c.abs().max(g.abs()).max(1.0);
            assert!(
                (c - g).abs() <= 5e-4 * scale,
                "{name}, logit {index} : CPU {c}, GPU {g}"
            );
        }

        // Sur la carte, avec et sans cache : mêmes logits au dernier token.
        let sequence = &tokens[..20];
        let mut cached = Generator::new(&gpu_backend, &gpu_model, sequence);
        let mut full = Generator::without_cache(&gpu_backend, &gpu_model, sequence);
        for (index, (c, f)) in cached.logits().iter().zip(full.logits()).enumerate() {
            let scale = c.abs().max(f.abs()).max(1.0);
            assert!(
                (c - f).abs() <= 5e-4 * scale,
                "{name}, logit {index} : avec cache {c}, sans {f}"
            );
        }
    }
}

#[test]
fn latent_attention_logits_match_cpu() {
    // L'attention latente n'a pas de kernel à elle : ce test vérifie que
    // ceux qu'elle emploie s'accordent, dont le produit d'une matrice par
    // tête avec une seule matrice partagée.
    let tokens: Vec<u32> = (0..48u32).map(|i| (i * i + 7 * i) % 251).collect();
    let cpu_backend = Cpu::<f32>::new();
    let (cpu_model, _) = model_with(&cpu_backend, LATENT);
    let tape = Tape::new(&cpu_backend);
    let cpu = cpu_backend.to_vec(&tape.value(cpu_model.logits(&tape, &tokens, 2)));

    let gpu_backend = Cuda::new(0).unwrap();
    let (gpu_model, _) = model_with(&gpu_backend, LATENT);
    let tape = Tape::new(&gpu_backend);
    let gpu = gpu_backend.to_vec(&tape.value(gpu_model.logits(&tape, &tokens, 2)));

    let mut largest = 0.0f32;
    for (index, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!(
            (c - g).abs() <= 2e-4 * scale,
            "logit {index} : CPU {c}, GPU {g}"
        );
        largest = largest.max(c.abs());
    }
    assert!(largest > 1.0, "logits trop petits : max {largest}");
}

#[test]
fn latent_attention_follows_the_cpu_loss_curve() {
    const STEPS: usize = 20;
    let cpu = loss_curve_with(&Cpu::<f32>::new(), LATENT, STEPS);
    let gpu = loss_curve_with(&Cuda::new(0).unwrap(), LATENT, STEPS);
    assert!((cpu[0] - (VOCAB as f64).ln()).abs() < 0.2, "{cpu:?}");
    assert!(
        cpu[STEPS - 1] < cpu[0] - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c - g).abs() < 2e-3,
            "pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}"
        );
    }
}

#[test]
fn latent_generation_with_cache_matches_without_on_gpu() {
    let gpu = Cuda::new(0).unwrap();
    let (model, config) = model_with(&gpu, LATENT);
    let greedy = Sampling {
        temperature: 0.0,
        ..Sampling::default()
    };
    let mut rng = StdRng::seed_from_u64(0);
    // Dans le contexte (24) : avec et sans cache, mêmes tokens.
    let length = config.context - 3;
    let cached = generate(&gpu, &model, &[3, 7, 1], length, &greedy, &mut rng);
    let uncached = generate_uncached(&gpu, &model, &[3, 7, 1], length, &greedy, &mut rng);
    assert_eq!(cached, uncached);
    // Au-delà : le cache glisse sans erreur et produit le bon nombre de tokens.
    let long = generate(&gpu, &model, &[3, 7, 1], 60, &greedy, &mut rng);
    assert_eq!(long.len(), 63);
    assert_eq!(long[..cached.len()], cached[..]);

    // Le cache ne garde que le latent et la clé tournée : 8 + 8 nombres par
    // position et par couche, contre 2 × 32 pour l'attention classique.
    let per_position = |model: &Gpt<Cuda>| {
        let mut generator = Generator::new(&gpu, model, &[3, 7, 1]);
        generator.logits();
        generator.cache().unwrap().0.numbers_per_position(&gpu)
    };
    assert_eq!(per_position(&model), 2 * (8 + 8));
    assert_eq!(per_position(&modern_model(&gpu).0), 2 * 64);
}

/// Le socle moderne avec un mélange d'experts : un partagé, quatre
/// aiguillés dont deux actifs.
const EXPERTS: Architecture = Architecture::MODERN.with_experts(ExpertDims {
    shared: 1,
    routed: 4,
    active: 2,
    hidden: 32,
    balance: Balance::None,
});

fn logits_with<B: Backend<Scalar = f32>>(
    backend: &B,
    architecture: Architecture,
    dense: bool,
) -> (Vec<f32>, Vec<Vec<u64>>) {
    let tokens: Vec<u32> = (0..48u32).map(|i| (i * i + 7 * i) % 251).collect();
    let (mut model, _) = model_with(backend, architecture);
    model.set_dense_experts(dense);
    let tape = Tape::new(backend);
    let logits = backend.to_vec(&tape.value(model.logits(&tape, &tokens, 2)));
    (logits, model.take_expert_load())
}

#[test]
fn experts_logits_match_cpu() {
    // Le choix des experts se fait sur l'hôte à partir de nombres calculés
    // sur la carte : les deux backends doivent aiguiller de la même façon,
    // puis calculer la même chose. Les groupes ont des tailles quelconques.
    let (cpu, cpu_load) = logits_with(&Cpu::<f32>::new(), EXPERTS, false);
    let gpu_backend = Cuda::new(0).unwrap();
    let (gpu, gpu_load) = logits_with(&gpu_backend, EXPERTS, false);
    assert_eq!(cpu_load, gpu_load);
    // 48 tokens, deux experts chacun, dans chacune des deux couches. Un
    // expert peut n'avoir reçu aucun token : il est alors sauté.
    for layer in &cpu_load {
        assert_eq!(layer.iter().sum::<u64>(), 96);
    }

    let mut largest = 0.0f32;
    for (index, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let scale = c.abs().max(g.abs()).max(1.0);
        assert!(
            (c - g).abs() <= 2e-4 * scale,
            "logit {index} : CPU {c}, GPU {g}"
        );
        largest = largest.max(c.abs());
    }
    assert!(largest > 1.0, "logits trop petits : max {largest}");

    // Sur la carte aussi, la version clairsemée vaut la version dense.
    let (dense, _) = logits_with(&gpu_backend, EXPERTS, true);
    for (index, (s, d)) in gpu.iter().zip(&dense).enumerate() {
        let scale = s.abs().max(d.abs()).max(1.0);
        assert!(
            (s - d).abs() <= 2e-4 * scale,
            "logit {index} : clairsemé {s}, dense {d}"
        );
    }
}

#[test]
fn experts_follow_the_cpu_loss_curve() {
    const STEPS: usize = 20;
    let cpu = loss_curve_with(&Cpu::<f32>::new(), EXPERTS, STEPS);
    let gpu = loss_curve_with(&Cuda::new(0).unwrap(), EXPERTS, STEPS);
    assert!((cpu[0] - (VOCAB as f64).ln()).abs() < 0.2, "{cpu:?}");
    assert!(
        cpu[STEPS - 1] < cpu[0] - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    // Un token dont deux affinités sont presque égales peut être aiguillé
    // autrement sur les deux backends : la tolérance est plus large que
    // pour les autres architectures.
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c - g).abs() < 1e-2,
            "pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}"
        );
    }
}

#[test]
fn multi_token_prediction_follows_the_cpu_loss_curve() {
    // La perte d'entraînement, avec son terme de prédiction multi-token :
    // la concaténation, le second passage par la sortie du modèle et la
    // sélection des positions s'accordent sur les deux backends.
    const STEPS: usize = 20;
    let architecture = Architecture::MODERN.with_multi_token(MultiToken::DEEPSEEK);
    let cpu = loss_curve_with(&Cpu::<f32>::new(), architecture, STEPS);
    let gpu = loss_curve_with(&Cuda::new(0).unwrap(), architecture, STEPS);
    // Au départ, les deux pertes valent ln(vocabulaire) : 1,3 fois en tout.
    assert!((cpu[0] - 1.3 * (VOCAB as f64).ln()).abs() < 0.3, "{cpu:?}");
    assert!(
        cpu[STEPS - 1] < cpu[0] - 0.5,
        "le CPU n'apprend pas : {cpu:?}"
    );
    for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c - g).abs() < 2e-3,
            "pas {step} : CPU {c}, GPU {g}\nCPU {cpu:?}\nGPU {gpu:?}"
        );
    }
}
