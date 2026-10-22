//! Chargement des poids de GPT-2 tels que les publie OpenAI.
//!
//! L'architecture de [`Gpt`](crate::model::Gpt) est celle de GPT-2 : il
//! suffit de renommer les tenseurs, à une exception près. GPT-2 range les
//! trois projections de l'attention (requête, clé, valeur) dans une seule
//! matrice `[C, 3C]`, que ce modèle garde en trois matrices `[C, C]`.

use crate::io::{HostTensor, SafeTensors};
use crate::model::GptConfig;
use crate::nn::Architecture;

/// GPT-2 small, le plus petit des quatre modèles publiés : 124 millions de
/// paramètres.
pub const GPT2_SMALL: GptConfig = GptConfig {
    vocab: 50257,
    context: 1024,
    layers: 12,
    heads: 12,
    embed: 768,
    architecture: Architecture::GPT2,
};

/// Découpe un tenseur en `parts` morceaux égaux le long de son dernier axe.
/// Une matrice `[C, 3C]` donne trois matrices `[C, C]`, faites des colonnes
/// `0..C`, `C..2C` et `2C..3C`.
fn split_last_axis(tensor: &HostTensor, parts: usize) -> Vec<HostTensor> {
    let width = *tensor.shape.last().expect("tenseur sans dimension");
    assert!(
        width.is_multiple_of(parts),
        "largeur {width} non divisible par {parts}"
    );
    let part_width = width / parts;
    let mut shape = tensor.shape.clone();
    *shape.last_mut().unwrap() = part_width;

    (0..parts)
        .map(|part| HostTensor {
            shape: shape.clone(),
            data: tensor
                .data
                .chunks(width)
                .flat_map(|row| {
                    row[part * part_width..(part + 1) * part_width]
                        .iter()
                        .copied()
                })
                .collect(),
        })
        .collect()
}

/// Traduit un fichier de poids GPT-2 dans les noms et les formes de
/// [`Gpt`](crate::model::Gpt).
///
/// Aucune transposition n'est nécessaire : GPT-2 range ses matrices en
/// `[entrée, sortie]`, comme [`Linear`](crate::nn::Linear). Ce serait
/// différent avec la convention habituelle de PyTorch, `[sortie, entrée]`.
pub fn convert(file: &SafeTensors, config: GptConfig) -> Result<SafeTensors, String> {
    if config.architecture != Architecture::GPT2 {
        return Err(format!(
            "les poids de GPT-2 demandent l'architecture de GPT-2, pas {:?}",
            config.architecture
        ));
    }
    let get = |name: &str| {
        file.tensors
            .get(name)
            .ok_or_else(|| format!("tenseur `{name}` absent du fichier GPT-2"))
    };
    let mut result = SafeTensors {
        metadata: config.to_metadata(),
        ..Default::default()
    };
    let mut put = |name: String, tensor: HostTensor| {
        result.tensors.insert(name, tensor);
    };

    put("token_embedding.weight".into(), get("wte.weight")?.clone());
    put(
        "position_embedding.weight".into(),
        get("wpe.weight")?.clone(),
    );
    put("final_norm.gain".into(), get("ln_f.weight")?.clone());
    put("final_norm.bias".into(), get("ln_f.bias")?.clone());

    for layer in 0..config.layers {
        let from = |suffix: &str| format!("h.{layer}.{suffix}");
        let to = |suffix: &str| format!("blocks.{layer}.{suffix}");

        // Renommages directs.
        for (source, target) in [
            ("ln_1.weight", "ln1.gain"),
            ("ln_1.bias", "ln1.bias"),
            ("attn.c_proj.weight", "attention.proj.weight"),
            ("attn.c_proj.bias", "attention.proj.bias"),
            ("ln_2.weight", "ln2.gain"),
            ("ln_2.bias", "ln2.bias"),
            ("mlp.c_fc.weight", "feed_forward.expand.weight"),
            ("mlp.c_fc.bias", "feed_forward.expand.bias"),
            ("mlp.c_proj.weight", "feed_forward.proj.weight"),
            ("mlp.c_proj.bias", "feed_forward.proj.bias"),
        ] {
            put(to(target), get(&from(source))?.clone());
        }

        // `c_attn` contient requête, clé et valeur côte à côte.
        let weights = split_last_axis(get(&from("attn.c_attn.weight"))?, 3);
        let biases = split_last_axis(get(&from("attn.c_attn.bias"))?, 3);
        for ((name, weight), bias) in ["query", "key", "value"].iter().zip(weights).zip(biases) {
            put(to(&format!("attention.{name}.weight")), weight);
            put(to(&format!("attention.{name}.bias")), bias);
        }
        // `h.N.attn.bias` n'est pas un paramètre : c'est le masque causal,
        // que ce modèle n'a pas besoin de stocker.
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tape;
    use crate::backend::Primitives;
    use crate::backend::cpu::Cpu;
    use crate::io::load_parameters;
    use crate::model::Gpt;
    use crate::nn::Module;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Le modèle minuscule des tests, avec deux blocs.
    const TINY: GptConfig = GptConfig {
        layers: 2,
        ..crate::model::tests::TINY
    };

    #[test]
    fn split_takes_columns_not_rows() {
        // [[0, 1, 2, 3, 4, 5],
        //  [6, 7, 8, 9, 10, 11]]
        let tensor = HostTensor {
            shape: vec![2, 6],
            data: (0..12).map(|i| i as f32).collect(),
        };
        let parts = split_last_axis(&tensor, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape, vec![2, 2]);
        assert_eq!(parts[0].data, vec![0.0, 1.0, 6.0, 7.0]);
        assert_eq!(parts[1].data, vec![2.0, 3.0, 8.0, 9.0]);
        assert_eq!(parts[2].data, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn split_of_a_vector() {
        let tensor = HostTensor {
            shape: vec![6],
            data: (0..6).map(|i| i as f32).collect(),
        };
        let parts = split_last_axis(&tensor, 3);
        assert_eq!(parts[1].shape, vec![2]);
        assert_eq!(parts[1].data, vec![2.0, 3.0]);
    }

    /// Écrit un modèle dans la disposition de GPT-2 : l'inverse de `convert`.
    fn to_gpt2_layout(backend: &Cpu<f32>, model: &Gpt<Cpu<f32>>) -> SafeTensors {
        let ours: std::collections::BTreeMap<String, HostTensor> = model
            .parameter_names()
            .into_iter()
            .zip(model.parameters())
            .map(|(name, parameter)| {
                let tensor = HostTensor {
                    shape: backend.shape(parameter.value()).to_vec(),
                    data: backend.to_vec(parameter.value()),
                };
                (name, tensor)
            })
            .collect();

        // Recolle des tenseurs côte à côte le long du dernier axe.
        let join = |names: [String; 3]| {
            let parts: Vec<&HostTensor> = names.iter().map(|name| &ours[name]).collect();
            let width = *parts[0].shape.last().unwrap();
            let rows = parts[0].data.len() / width;
            let mut data = Vec::new();
            for row in 0..rows {
                for part in &parts {
                    data.extend_from_slice(&part.data[row * width..(row + 1) * width]);
                }
            }
            let mut shape = parts[0].shape.clone();
            *shape.last_mut().unwrap() = 3 * width;
            HostTensor { shape, data }
        };

        let mut file = SafeTensors::default();
        let mut put = |name: String, tensor: HostTensor| {
            file.tensors.insert(name, tensor);
        };
        put("wte.weight".into(), ours["token_embedding.weight"].clone());
        put(
            "wpe.weight".into(),
            ours["position_embedding.weight"].clone(),
        );
        put("ln_f.weight".into(), ours["final_norm.gain"].clone());
        put("ln_f.bias".into(), ours["final_norm.bias"].clone());
        for layer in 0..TINY.layers {
            let ours_name = |suffix: &str| format!("blocks.{layer}.{suffix}");
            for (target, source) in [
                ("ln_1.weight", "ln1.gain"),
                ("ln_1.bias", "ln1.bias"),
                ("attn.c_proj.weight", "attention.proj.weight"),
                ("attn.c_proj.bias", "attention.proj.bias"),
                ("ln_2.weight", "ln2.gain"),
                ("ln_2.bias", "ln2.bias"),
                ("mlp.c_fc.weight", "feed_forward.expand.weight"),
                ("mlp.c_fc.bias", "feed_forward.expand.bias"),
                ("mlp.c_proj.weight", "feed_forward.proj.weight"),
                ("mlp.c_proj.bias", "feed_forward.proj.bias"),
            ] {
                put(
                    format!("h.{layer}.{target}"),
                    ours[&ours_name(source)].clone(),
                );
            }
            for kind in ["weight", "bias"] {
                let joined = join(
                    ["query", "key", "value"]
                        .map(|name| ours_name(&format!("attention.{name}.{kind}"))),
                );
                put(format!("h.{layer}.attn.c_attn.{kind}"), joined);
            }
        }
        file
    }

    #[test]
    fn conversion_restores_the_model() {
        // Un modèle écrit dans la disposition de GPT-2 puis converti doit
        // redonner exactement les mêmes logits.
        let backend = Cpu::<f32>::new();
        let mut rng = StdRng::seed_from_u64(0);
        let mut original = Gpt::new(&backend, TINY, &mut rng);
        // Poids et biais tous différents, pour qu'un mélange se voie.
        for parameter in original.parameters_mut() {
            let shape = backend.shape(parameter.value()).to_vec();
            *parameter.value_mut() = backend.randn(&shape, &mut rng);
        }

        let converted = convert(&to_gpt2_layout(&backend, &original), TINY).unwrap();
        let mut restored = Gpt::new(&backend, TINY, &mut rng);
        load_parameters(&converted, &backend, &mut restored).unwrap();

        let logits = |model: &Gpt<Cpu<f32>>| {
            let tape = Tape::new(&backend);
            let out = model.logits(&tape, &[3, 7, 1, 9], 1);
            backend.to_vec(&tape.value(out))
        };
        assert_eq!(logits(&original), logits(&restored));
    }

    #[test]
    fn missing_tensor_is_reported_by_name() {
        let error = convert(&SafeTensors::default(), TINY).unwrap_err();
        assert!(error.contains("wte.weight"), "{error}");
    }

    #[test]
    fn another_architecture_is_refused() {
        let modern = TINY.with_architecture(Architecture::MODERN);
        let error = convert(&SafeTensors::default(), modern).unwrap_err();
        assert!(error.contains("GPT-2"), "{error}");
    }
}
