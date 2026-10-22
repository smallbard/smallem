//! Lecture et écriture de poids au format safetensors.
//!
//! Le format est volontairement simple :
//!
//! ```text
//! 8 octets   taille N de l'en-tête, entier petit-boutiste
//! N octets   en-tête JSON : pour chaque tenseur, son type, sa forme et
//!            l'intervalle d'octets qu'il occupe dans la suite du fichier
//! le reste   les données brutes des tenseurs, bout à bout
//! ```
//!
//! C'est aussi le format dans lequel sont publiés les poids de GPT-2.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use serde_json::{Map, Value, json};

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::backend::{Backend, Float};
use crate::model::{Gpt, GptConfig};
use crate::nn::Module;
use crate::optim::{Optimizer, OptimizerState};

/// Clé de l'en-tête réservée aux informations libres (ici, la configuration
/// du modèle).
const METADATA_KEY: &str = "__metadata__";

/// Un tenseur lu dans un fichier, encore sur l'hôte.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Le contenu d'un fichier safetensors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SafeTensors {
    pub tensors: BTreeMap<String, HostTensor>,
    pub metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let bytes = fs::read(path)?;
        if bytes.len() < 8 {
            return Err("fichier trop court pour être un safetensors".into());
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into()?);
        // Un fichier abîmé peut annoncer n'importe quelle taille : tous les
        // calculs de position sont vérifiés, pour renvoyer une erreur et
        // non paniquer.
        let data_start = usize::try_from(header_len)
            .ok()
            .and_then(|len| len.checked_add(8))
            .filter(|&start| start <= bytes.len())
            .ok_or("en-tête plus long que le fichier")?;
        let header: Map<String, Value> = serde_json::from_slice(&bytes[8..data_start])?;
        let data = &bytes[data_start..];

        let mut result = Self::default();
        for (name, entry) in &header {
            if name == METADATA_KEY {
                for (key, value) in entry.as_object().ok_or("métadonnées mal formées")? {
                    let value = value.as_str().ok_or("métadonnée qui n'est pas du texte")?;
                    result.metadata.insert(key.clone(), value.to_string());
                }
                continue;
            }
            let dtype = entry["dtype"].as_str().ok_or("type absent")?;
            if dtype != "F32" {
                return Err(format!("tenseur `{name}` de type {dtype}, seul F32 est géré").into());
            }
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .ok_or("forme absente")?
                .iter()
                .map(|dim| dim.as_u64().map(|dim| dim as usize))
                .collect::<Option<_>>()
                .ok_or("forme mal formée")?;
            let offsets: Vec<usize> = entry["data_offsets"]
                .as_array()
                .ok_or("position absente")?
                .iter()
                .map(|offset| {
                    offset
                        .as_u64()
                        .and_then(|offset| usize::try_from(offset).ok())
                })
                .collect::<Option<_>>()
                .ok_or("position mal formée")?;
            let &[begin, end] = offsets.as_slice() else {
                return Err(format!("tenseur `{name}` : deux positions attendues").into());
            };

            let expected = shape
                .iter()
                .try_fold(4usize, |bytes, &dim| bytes.checked_mul(dim));
            if end < begin || end > data.len() || Some(end - begin) != expected {
                return Err(format!("tenseur `{name}` : position ou taille incohérente").into());
            }
            let (values, _) = data[begin..end].as_chunks::<4>();
            let values = values
                .iter()
                .map(|&four| f32::from_le_bytes(four))
                .collect();
            result.tensors.insert(
                name.clone(),
                HostTensor {
                    shape,
                    data: values,
                },
            );
        }
        Ok(result)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let mut header = Map::new();
        if !self.metadata.is_empty() {
            header.insert(METADATA_KEY.into(), json!(self.metadata));
        }
        let mut offset = 0;
        for (name, tensor) in &self.tensors {
            let end = offset + tensor.data.len() * 4;
            header.insert(
                name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": tensor.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&header)?;
        // Le format demande que les données commencent sur un multiple de
        // 8 octets : on complète l'en-tête avec des espaces.
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut bytes = Vec::with_capacity(8 + header.len() + offset);
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(header);
        for tensor in self.tensors.values() {
            for value in &tensor.data {
                bytes.extend(value.to_le_bytes());
            }
        }
        Ok(write_atomically(path, &bytes)?)
    }
}

/// Écrit `bytes` à côté de `path`, puis renomme : une sauvegarde
/// interrompue ne détruit pas le fichier précédent.
pub(crate) fn write_atomically(path: impl AsRef<Path>, bytes: &[u8]) -> std::io::Result<()> {
    let path = path.as_ref();
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    let temporary = path.with_file_name(name);
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)
}

/// Rapatrie un tenseur sur l'hôte.
fn to_host<B: Backend>(backend: &B, tensor: &B::Tensor) -> HostTensor {
    HostTensor {
        shape: backend.shape(tensor).to_vec(),
        data: backend
            .to_vec(tensor)
            .iter()
            .map(|value| value.to_f64() as f32)
            .collect(),
    }
}

/// Le tenseur `name` de `file`, envoyé sur le backend. Il doit avoir la
/// forme `expected`.
fn from_file<B: Backend>(
    file: &SafeTensors,
    backend: &B,
    name: &str,
    expected: &[usize],
) -> Result<B::Tensor, Box<dyn Error>> {
    read_tensor(file, backend, name, expected, false)
}

/// Comme [`from_file`]. Avec `reduced`, une dimension du tenseur peut aussi
/// valoir 1 là où `expected` en attend davantage : l'état d'un optimiseur
/// résume parfois un paramètre par ligne ou par colonne.
fn read_tensor<B: Backend>(
    file: &SafeTensors,
    backend: &B,
    name: &str,
    expected: &[usize],
    reduced: bool,
) -> Result<B::Tensor, Box<dyn Error>> {
    let tensor = file
        .tensors
        .get(name)
        .ok_or_else(|| format!("tenseur `{name}` absent du fichier"))?;
    let fits = tensor.shape.len() == expected.len()
        && tensor
            .shape
            .iter()
            .zip(expected)
            .all(|(&found, &wanted)| found == wanted || (reduced && found == 1));
    if !fits {
        return Err(format!(
            "tenseur `{name}` : forme {:?} dans le fichier, {expected:?} dans le modèle",
            tensor.shape,
        )
        .into());
    }
    let data = tensor
        .data
        .iter()
        .map(|&value| B::Scalar::from_f64(value as f64))
        .collect();
    Ok(backend.from_vec(data, &tensor.shape))
}

/// Les paramètres d'un modèle, rapatriés sur l'hôte sous leurs noms.
fn model_tensors<B: Backend, M: Module<B>>(
    backend: &B,
    model: &M,
    metadata: BTreeMap<String, String>,
) -> SafeTensors {
    let mut file = SafeTensors {
        metadata,
        ..Default::default()
    };
    for (name, parameter) in model.parameter_names().into_iter().zip(model.parameters()) {
        file.tensors
            .insert(name, to_host(backend, parameter.value()));
    }
    file
}

/// Rapatrie les paramètres d'un modèle sur l'hôte et les enregistre, avec
/// des métadonnées libres.
pub fn save_model<B: Backend, M: Module<B>>(
    path: impl AsRef<Path>,
    backend: &B,
    model: &M,
    metadata: BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    model_tensors(backend, model, metadata).write(path)
}

/// Remplace les paramètres d'un modèle par ceux de `file`. Chaque paramètre
/// doit s'y trouver sous son nom, avec la même forme. Les autres tenseurs
/// du fichier sont ignorés.
pub fn load_parameters<B: Backend, M: Module<B>>(
    file: &SafeTensors,
    backend: &B,
    model: &mut M,
) -> Result<(), Box<dyn Error>> {
    let names = model.parameter_names();
    for (name, parameter) in names.into_iter().zip(model.parameters_mut()) {
        let shape = backend.shape(parameter.value()).to_vec();
        *parameter.value_mut() = from_file(file, backend, &name, &shape)?;
    }
    Ok(())
}

/// Le modèle enregistré dans `file` : sa configuration, lue dans les
/// métadonnées, puis ses poids. Avec `inference`, sans ce qui ne sert qu'à
/// l'entraînement (le module de prédiction multi-token, que le fichier peut
/// contenir : [`GptConfig::for_inference`]).
///
/// Les poids tirés à la construction sont aussitôt remplacés : ils ont leur
/// propre générateur, à graine fixe, pour que rien d'autre n'en dépende.
pub fn gpt_from_file<B: Backend>(
    backend: &B,
    file: &SafeTensors,
    inference: bool,
) -> Result<Gpt<B>, Box<dyn Error>> {
    let mut config = GptConfig::from_metadata(&file.metadata)?;
    if inference {
        config = config.for_inference();
    }
    let mut model = Gpt::try_new(backend, config, &mut StdRng::seed_from_u64(0))?;
    load_parameters(file, backend, &mut model)?;
    Ok(model)
}

/// Lit le fichier `path` et en fait un modèle : voir [`gpt_from_file`].
pub fn load_gpt<B: Backend>(
    backend: &B,
    path: impl AsRef<Path>,
    inference: bool,
) -> Result<Gpt<B>, Box<dyn Error>> {
    let path = path.as_ref();
    let file = SafeTensors::read(path).map_err(|error| format!("{} : {error}", path.display()))?;
    gpt_from_file(backend, &file, inference)
}

/// Enregistre un point de reprise : les paramètres, comme [`save_model`],
/// plus l'état de l'optimiseur. Sans ses moyennes glissantes, un
/// entraînement repris ferait de mauvais premiers pas.
///
/// Ce module ne sait rien des optimiseurs : chacun décrit son état, des
/// compteurs et des tenseurs rangés par paramètre
/// ([`Optimizer::state`]). Un tenseur est enregistré sous le nom de son
/// paramètre, précédé du préfixe de sa série : `adamw.m.blocks.0.ln1.gain`.
///
/// Le fichier reste un fichier de poids ordinaire : [`load_parameters`] le
/// lit en ignorant ce qui concerne l'optimiseur.
pub(crate) fn save_checkpoint<B: Backend, M: Module<B>>(
    path: impl AsRef<Path>,
    backend: &B,
    model: &M,
    optimizer: &impl Optimizer<B>,
    mut metadata: BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let state = optimizer.state();
    metadata.extend(state.counters);
    let mut file = model_tensors(backend, model, metadata);
    let names = model.parameter_names();
    for (prefix, tensors) in &state.slots {
        for (name, tensor) in names.iter().zip(tensors) {
            if let Some(tensor) = tensor {
                file.tensors
                    .insert(format!("{prefix}{name}"), to_host(backend, tensor));
            }
        }
    }
    file.write(path)
}

/// Recharge un point de reprise dans un modèle et un optimiseur.
pub(crate) fn load_checkpoint<B: Backend, M: Module<B>>(
    file: &SafeTensors,
    backend: &B,
    model: &mut M,
    optimizer: &mut impl Optimizer<B>,
) -> Result<(), Box<dyn Error>> {
    load_parameters(file, backend, model)?;
    let (counter_names, prefixes) = optimizer.state_layout();
    let mut counters = BTreeMap::new();
    for name in counter_names {
        let value = file.metadata.get(name).ok_or_else(|| {
            format!("ce fichier n'est pas un point de reprise de cet optimiseur : `{name}` absent")
        })?;
        counters.insert(name.to_string(), value.clone());
    }
    let names = model.parameter_names();
    let parameters = model.parameters();
    let mut slots = Vec::new();
    for prefix in prefixes {
        let mut tensors = Vec::with_capacity(names.len());
        for (name, parameter) in names.iter().zip(&parameters) {
            let key = format!("{prefix}{name}");
            tensors.push(if file.tensors.contains_key(&key) {
                let shape = backend.shape(parameter.value()).to_vec();
                Some(read_tensor(file, backend, &key, &shape, true)?)
            } else {
                None
            });
        }
        slots.push((prefix.to_string(), tensors));
    }
    optimizer.restore_state(OptimizerState { counters, slots })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::Cpu;
    use crate::optim::AdamW;
    use crate::training::train_step;

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("smallem-{name}-{}.safetensors", std::process::id()))
    }

    /// Le modèle minuscule des tests, avec deux couches : les noms des
    /// paramètres de plusieurs blocs se distinguent.
    const TINY: GptConfig = GptConfig {
        layers: 2,
        ..crate::model::tests::TINY
    };

    #[test]
    fn tensors_and_metadata_survive_a_round_trip() {
        let mut file = SafeTensors::default();
        file.metadata.insert("clé".into(), "valeur".into());
        file.tensors.insert(
            "matrice".into(),
            HostTensor {
                shape: vec![2, 3],
                data: vec![1.0, -2.5, 3.0, f32::MIN_POSITIVE, 1e30, 0.0],
            },
        );
        file.tensors.insert(
            "scalaire".into(),
            HostTensor {
                shape: vec![],
                data: vec![7.0],
            },
        );
        let path = temporary("round-trip");
        file.write(&path).unwrap();
        let read = SafeTensors::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(read, file);
    }

    #[test]
    fn header_follows_the_format() {
        let mut file = SafeTensors::default();
        file.tensors.insert(
            "a".into(),
            HostTensor {
                shape: vec![2],
                data: vec![1.0, 2.0],
            },
        );
        let path = temporary("header");
        file.write(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert!(header_len.is_multiple_of(8));
        let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        assert_eq!(header["a"]["dtype"], "F32");
        assert_eq!(header["a"]["shape"], json!([2]));
        assert_eq!(header["a"]["data_offsets"], json!([0, 8]));
        // Les données suivent l'en-tête : deux f32 petit-boutistes.
        assert_eq!(&bytes[8 + header_len..], &[0, 0, 128, 63, 0, 0, 0, 64]);
    }

    #[test]
    fn model_survives_a_round_trip() {
        let backend = Cpu::<f32>::new();
        let model = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(1));
        let path = temporary("model");
        save_model(&path, &backend, &model, TINY.to_metadata()).unwrap();

        let file = SafeTensors::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let config = GptConfig::from_metadata(&file.metadata).unwrap();
        // Graine différente : tous les poids diffèrent avant le chargement.
        let mut loaded = Gpt::new(&backend, config, &mut StdRng::seed_from_u64(2));
        load_parameters(&file, &backend, &mut loaded).unwrap();

        assert_eq!(model.parameters().len(), loaded.parameters().len());
        for (a, b) in model.parameters().iter().zip(loaded.parameters()) {
            assert_eq!(a.value(), b.value());
        }
    }

    #[test]
    fn loading_into_a_different_architecture_fails() {
        let backend = Cpu::<f32>::new();
        let model = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(1));
        let path = temporary("mismatch");
        save_model(&path, &backend, &model, BTreeMap::new()).unwrap();
        let file = SafeTensors::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        let wider = GptConfig { embed: 16, ..TINY };
        let mut other = Gpt::new(&backend, wider, &mut StdRng::seed_from_u64(1));
        let error = load_parameters(&file, &backend, &mut other).unwrap_err();
        assert!(error.to_string().contains("forme"), "{error}");

        let deeper = GptConfig { layers: 3, ..TINY };
        let mut other = Gpt::new(&backend, deeper, &mut StdRng::seed_from_u64(1));
        let error = load_parameters(&file, &backend, &mut other).unwrap_err();
        assert!(error.to_string().contains("absent"), "{error}");
    }

    /// Un fichier dont l'en-tête est `header`, suivi de `data_len` octets.
    fn file_with_header(name: &str, header: &str, data_len: usize) -> std::path::PathBuf {
        let path = temporary(name);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(vec![0u8; data_len]);
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn malformed_headers_are_errors_not_panics() {
        let headers = [
            // Positions absentes, trop peu nombreuses, trop nombreuses.
            r#"{"a":{"dtype":"F32","shape":[2]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8,16]}}"#,
            // Positions qui ne sont pas des entiers positifs, ou à l'envers.
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[-8,0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":["0","8"]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
            // Taille qui ne correspond pas à la forme, ou qui dépasse le fichier.
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
            // Forme dont le nombre d'éléments déborde.
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,8]}}"#,
            // Type non géré, forme mal formée, JSON invalide.
            r#"{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":"2","data_offsets":[0,8]}}"#,
            r#"{"a":"#,
        ];
        for (index, header) in headers.iter().enumerate() {
            let path = file_with_header(&format!("malformed-{index}"), header, 8);
            let result = SafeTensors::read(&path);
            fs::remove_file(&path).unwrap();
            assert!(result.is_err(), "en-tête accepté : {header}");
        }

        // Le même en-tête, bien formé, est accepté : le test ne passe pas
        // seulement parce que tout est refusé.
        let path = file_with_header(
            "well-formed",
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            8,
        );
        let file = SafeTensors::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(file.tensors["a"].data, [0.0, 0.0]);
    }

    #[test]
    fn absurd_header_length_is_rejected() {
        // Une taille d'en-tête proche du maximum : `8 + taille` déborderait.
        for header_len in [u64::MAX, u64::MAX - 7, 1 << 40] {
            let path = temporary(&format!("header-len-{header_len}"));
            let mut bytes = header_len.to_le_bytes().to_vec();
            bytes.extend_from_slice(b"{}");
            fs::write(&path, bytes).unwrap();
            let result = SafeTensors::read(&path);
            fs::remove_file(&path).unwrap();
            assert!(result.is_err(), "taille {header_len} acceptée");
        }
    }

    #[test]
    fn truncated_file_is_rejected() {
        let path = temporary("truncated");
        fs::write(&path, [200, 0, 0, 0, 0, 0, 0, 0, b'{']).unwrap();
        let result = SafeTensors::read(&path);
        fs::remove_file(&path).unwrap();
        assert!(result.is_err());
    }

    /// Un pas d'entraînement dont le lot ne dépend que de son rang.
    fn training_step(
        backend: &Cpu<f32>,
        model: &mut Gpt<Cpu<f32>>,
        optimizer: &mut AdamW<Cpu<f32>>,
        step: u32,
    ) {
        let tokens: Vec<u32> = (0..8).map(|i| (i * 3 + step) % 11).collect();
        let targets: Vec<u32> = (0..8).map(|i| (i * 5 + step + 1) % 11).collect();
        train_step(backend, model, optimizer, |m, t| {
            m.loss(t, &tokens, &targets, 2)
        });
    }

    #[test]
    fn resumed_training_continues_exactly() {
        let backend = Cpu::<f32>::new();
        let mut model = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(1));
        let mut optimizer = AdamW::new(1e-2, 0.1);
        for step in 0..3 {
            training_step(&backend, &mut model, &mut optimizer, step);
        }
        let path = temporary("checkpoint");
        save_checkpoint(&path, &backend, &model, &optimizer, TINY.to_metadata()).unwrap();
        for step in 3..6 {
            training_step(&backend, &mut model, &mut optimizer, step);
        }

        // Reprise dans un modèle et un optimiseur neufs.
        let file = SafeTensors::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let mut resumed = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(2));
        let mut resumed_optimizer = AdamW::new(1e-2, 0.1);
        load_checkpoint(&file, &backend, &mut resumed, &mut resumed_optimizer).unwrap();
        assert_eq!(resumed_optimizer.steps(), 3);
        for step in 3..6 {
            training_step(&backend, &mut resumed, &mut resumed_optimizer, step);
        }
        for (a, b) in model.parameters().iter().zip(resumed.parameters()) {
            assert_eq!(a.value(), b.value());
        }

        // Sans l'état de l'optimiseur, la suite n'est pas la même : c'est
        // bien lui qui rend la reprise exacte.
        let mut forgetful = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(2));
        load_parameters(&file, &backend, &mut forgetful).unwrap();
        let mut fresh_optimizer = AdamW::new(1e-2, 0.1);
        for step in 3..6 {
            training_step(&backend, &mut forgetful, &mut fresh_optimizer, step);
        }
        assert!(
            model
                .parameters()
                .iter()
                .zip(forgetful.parameters())
                .any(|(a, b)| a.value() != b.value())
        );
    }

    #[test]
    fn plain_model_file_is_not_a_checkpoint() {
        let backend = Cpu::<f32>::new();
        let mut model = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(1));
        let path = temporary("not-a-checkpoint");
        save_model(&path, &backend, &model, TINY.to_metadata()).unwrap();
        let file = SafeTensors::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let error =
            load_checkpoint(&file, &backend, &mut model, &mut AdamW::new(1e-2, 0.1)).unwrap_err();
        assert!(error.to_string().contains("point de reprise"), "{error}");
    }

    #[test]
    fn a_saved_model_comes_back_whole() {
        let backend = Cpu::<f32>::new();
        let model = Gpt::new(&backend, TINY, &mut StdRng::seed_from_u64(1));
        let path = temporary("load-gpt");
        save_model(&path, &backend, &model, TINY.to_metadata()).unwrap();
        let loaded = load_gpt(&backend, &path, true);
        fs::remove_file(&path).unwrap();
        let loaded = loaded.unwrap();
        assert_eq!(loaded.config(), model.config());
        assert!(
            model
                .parameters()
                .iter()
                .zip(loaded.parameters())
                .all(|(a, b)| a.value() == b.value())
        );
        let Err(missing) = load_gpt(&backend, temporary("absent"), true) else {
            panic!("un fichier absent ne donne pas de modèle");
        };
        assert!(missing.to_string().contains("absent"), "{missing}");
    }
}
