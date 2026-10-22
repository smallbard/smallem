//! La configuration d'un GPT, et sa forme dans l'en-tête d'un fichier de
//! poids.

use std::collections::BTreeMap;
use std::str::FromStr;

use serde_json::{Value, json};

use crate::nn::{
    Architecture, AttentionKind, Balance, BlockShape, ExpertDims, FeedForwardKind, LatentDims,
    MemoryDims, MultiToken, PositionKind,
};

/// Nom du champ des métadonnées qui porte la configuration.
const CONFIG_KEY: &str = "config";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GptConfig {
    pub vocab: usize,
    /// Longueur maximale d'une séquence.
    pub context: usize,
    pub layers: usize,
    pub heads: usize,
    /// Dimension `C` des vecteurs qui circulent dans le modèle.
    pub embed: usize,
    /// Les pièces dont les blocs sont faits.
    pub architecture: Architecture,
}

impl GptConfig {
    /// Petite configuration pour la mise au point : elle tourne aussi sur
    /// le backend CPU.
    pub fn small(vocab: usize) -> Self {
        Self {
            vocab,
            context: 64,
            layers: 4,
            heads: 4,
            embed: 128,
            architecture: Architecture::GPT2,
        }
    }

    /// Configuration d'entraînement sur GPU.
    pub fn large(vocab: usize) -> Self {
        Self {
            vocab,
            context: 256,
            layers: 6,
            heads: 6,
            embed: 384,
            architecture: Architecture::GPT2,
        }
    }

    /// Configuration de l'entraînement à l'échelle, sur des tokens BPE :
    /// environ 30 millions de paramètres avec un vocabulaire de 8 192.
    pub fn xl(vocab: usize) -> Self {
        Self {
            vocab,
            context: 512,
            layers: 8,
            heads: 8,
            embed: 512,
            architecture: Architecture::GPT2,
        }
    }

    /// La même configuration, avec d'autres pièces dans les blocs.
    pub fn with_architecture(self, architecture: Architecture) -> Self {
        Self {
            architecture,
            ..self
        }
    }

    /// Nombre de nombres que le cache de génération garde par position,
    /// toutes couches confondues : `2 × C` par couche d'attention classique
    /// qui calcule ses clés, `d_c + d_r` par couche d'attention latente,
    /// rien pour une couche qui emprunte ses clés, et autant d'étages que de
    /// traversées de la pile.
    pub fn cache_numbers_per_position(&self) -> usize {
        let per_layer = |layer: usize| match self.architecture.attention {
            AttentionKind::Latent(dims) => dims.latent + dims.rope,
            AttentionKind::Standard if self.architecture.borrows_keys(layer) => 0,
            AttentionKind::Standard => 2 * self.embed,
        };
        let layers: usize = (0..self.layers).map(per_layer).sum();
        layers * self.architecture.passes
    }

    /// La forme de chaque bloc du modèle.
    pub fn block_shape(&self) -> BlockShape {
        BlockShape {
            dim: self.embed,
            heads: self.heads,
            layers: self.layers,
            architecture: self.architecture,
        }
    }

    /// Dimension de chaque tête d'attention.
    pub fn head_dim(&self) -> usize {
        self.embed / self.heads
    }

    /// Taille des vecteurs que RoPE fait tourner : les requêtes et les clés
    /// entières dans l'attention classique, leur seule partie réservée aux
    /// positions dans l'attention latente.
    pub fn rotary_dim(&self) -> usize {
        match self.architecture.attention {
            AttentionKind::Standard => self.head_dim(),
            AttentionKind::Latent(dims) => dims.rope,
        }
    }

    /// La configuration du même modèle, sans ce qui ne sert qu'à
    /// l'entraînement. Un modèle construit ainsi lit le fichier de poids
    /// d'un modèle entraîné avec ces pièces, et ignore les leurs.
    pub fn for_inference(self) -> Self {
        Self {
            architecture: Architecture {
                multi_token: None,
                ..self.architecture
            },
            ..self
        }
    }

    /// Vérifie que les pièces choisies vont ensemble.
    pub fn validate(&self) -> Result<(), String> {
        if !self.embed.is_multiple_of(self.heads.max(1)) || self.heads == 0 {
            return Err(format!(
                "la dimension {} n'est pas divisible par {} têtes",
                self.embed, self.heads
            ));
        }
        if let AttentionKind::Latent(dims) = self.architecture.attention {
            if self.architecture.positions != PositionKind::Rotary {
                return Err("l'attention latente (mla) demande les positions `rope` : \
                     la position n'y entre que par la partie tournée des clés"
                    .to_string());
            }
            if dims.latent == 0 || dims.query_latent == 0 || dims.rope == 0 {
                return Err(format!("attention latente : tailles nulles dans {dims:?}"));
            }
        }
        if let Some(multi_token) = self.architecture.multi_token {
            multi_token.validate()?;
            if self.context < 2 {
                return Err(
                    "la prédiction multi-token demande un contexte d'au moins 2".to_string()
                );
            }
        }
        if let Some(experts) = self.architecture.experts {
            experts.validate()?;
            if self.architecture.feed_forward != FeedForwardKind::SwiGlu {
                return Err("le mélange d'experts demande le feed-forward `swiglu` : \
                     chaque expert en est un"
                    .to_string());
            }
        }
        if let Some(memory) = self.architecture.memory {
            memory.validate(self.layers)?;
        }
        if self.architecture.passes == 0 {
            return Err("la pile de blocs doit être traversée au moins une fois".to_string());
        }
        if let Some(own_keys) = self.architecture.own_keys {
            if self.architecture.attention != AttentionKind::Standard {
                return Err(
                    "le partage des clés entre couches demande l'attention `standard`".to_string(),
                );
            }
            if own_keys == 0 || own_keys > self.layers {
                return Err(format!(
                    "{own_keys} couches à clés propres : il en faut entre 1 et {}",
                    self.layers
                ));
            }
        }
        if self.architecture.positions == PositionKind::Rotary
            && !self.rotary_dim().is_multiple_of(2)
        {
            return Err(format!(
                "RoPE demande une taille paire, reçu {}",
                self.rotary_dim()
            ));
        }
        Ok(())
    }

    /// La configuration telle qu'elle voyage dans l'en-tête d'un fichier de
    /// poids : un seul champ, `config`, qui contient un texte JSON.
    ///
    /// ```text
    /// {"vocab":8192,"context":512,"layers":8,"heads":8,"embed":512,
    ///  "architecture":{"norm":"rms","feed_forward":"swiglu","positions":"rope",
    ///    "attention":{"kind":"mla","latent":128,"query_latent":384,"rope":32}}}
    /// ```
    ///
    /// Les métadonnées d'un fichier de poids ne sont que des paires de
    /// textes. Un texte JSON y range une description aussi imbriquée qu'il
    /// le faut : une pièce d'architecture qui a ses propres réglages y
    /// trouve sa place sans qu'on invente une convention de nommage.
    pub fn to_metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::from([(CONFIG_KEY.to_string(), self.to_json().to_string())])
    }

    pub fn from_metadata(metadata: &BTreeMap<String, String>) -> Result<Self, String> {
        let Some(text) = metadata.get(CONFIG_KEY) else {
            return Self::from_flat_metadata(metadata);
        };
        let json = serde_json::from_str(text)
            .map_err(|error| format!("configuration illisible : {error}"))?;
        Self::from_json(&json)
    }

    /// Lit la configuration d'un fichier dont chaque réglage est un champ à
    /// part des métadonnées (`layers`, `heads`…), comme les écrivaient les
    /// premières versions de ce projet.
    fn from_flat_metadata(metadata: &BTreeMap<String, String>) -> Result<Self, String> {
        let mut config = serde_json::Map::new();
        for key in ["vocab", "context", "layers", "heads", "embed"] {
            let value: u64 = metadata
                .get(key)
                .ok_or_else(|| format!("configuration : champ `{key}` absent"))?
                .parse()
                .map_err(|_| format!("configuration : champ `{key}` illisible"))?;
            config.insert(key.to_string(), json!(value));
        }
        let mut architecture = serde_json::Map::new();
        for key in ["norm", "feed_forward", "positions"] {
            if let Some(value) = metadata.get(key) {
                architecture.insert(key.to_string(), json!(value));
            }
        }
        config.insert("architecture".to_string(), Value::Object(architecture));
        Self::from_json(&Value::Object(config))
    }

    fn to_json(self) -> Value {
        let architecture = self.architecture;
        let mut json = json!({
            "vocab": self.vocab,
            "context": self.context,
            "layers": self.layers,
            "heads": self.heads,
            "embed": self.embed,
            "architecture": {
                "norm": architecture.norm.name(),
                "feed_forward": architecture.feed_forward.name(),
                "positions": architecture.positions.name(),
                "attention": attention_to_json(architecture.attention),
            },
        });
        // Une pièce absente du modèle est absente du texte.
        let pieces = &mut json["architecture"];
        if let Some(experts) = architecture.experts {
            pieces["experts"] = experts_to_json(&experts);
        }
        if let Some(multi_token) = architecture.multi_token {
            pieces["multi_token"] = multi_token_to_json(&multi_token);
        }
        if architecture.passes != 1 {
            pieces["passes"] = json!(architecture.passes);
        }
        if let Some(own_keys) = architecture.own_keys {
            pieces["own_keys"] = json!(own_keys);
        }
        if let Some(memory) = architecture.memory {
            pieces["memory"] = memory_to_json(&memory);
        }
        json
    }

    fn from_json(json: &Value) -> Result<Self, String> {
        let fields = Fields::new(json, "");
        let architecture = Fields::new(&json["architecture"], "");
        // Une pièce absente est celle de GPT-2 : c'est le modèle de départ,
        // et un fichier n'a à nommer que ce qui s'en écarte.
        let gpt2 = Architecture::GPT2;
        let config = Self {
            vocab: fields.size("vocab")?,
            context: fields.size("context")?,
            layers: fields.size("layers")?,
            heads: fields.size("heads")?,
            embed: fields.size("embed")?,
            architecture: Architecture {
                norm: architecture.choice("norm", Some(gpt2.norm))?,
                feed_forward: architecture.choice("feed_forward", Some(gpt2.feed_forward))?,
                positions: architecture.choice("positions", Some(gpt2.positions))?,
                attention: attention_from_json(&json["architecture"]["attention"])?,
                experts: optional(&json["architecture"]["experts"], experts_from_json)?,
                multi_token: optional(&json["architecture"]["multi_token"], multi_token_from_json)?,
                passes: architecture.optional_size("passes")?.unwrap_or(1),
                own_keys: architecture.optional_size("own_keys")?,
                memory: optional(&json["architecture"]["memory"], memory_from_json)?,
            },
        };
        config
            .validate()
            .map_err(|error| format!("configuration : {error}"))?;
        Ok(config)
    }
}

/// Un objet JSON de la configuration, lu champ par champ. `section` nomme
/// l'objet dans les messages d'erreur ; vide pour le niveau principal.
#[derive(Clone, Copy)]
struct Fields<'a> {
    json: &'a Value,
    section: &'static str,
}

impl<'a> Fields<'a> {
    fn new(json: &'a Value, section: &'static str) -> Self {
        Self { json, section }
    }

    /// Le début des messages d'erreur : `configuration : experts : `.
    fn context(&self) -> String {
        if self.section.is_empty() {
            "configuration : ".to_string()
        } else {
            format!("configuration : {} : ", self.section)
        }
    }

    fn missing(&self, key: &str) -> String {
        format!("{}champ `{key}` absent ou illisible", self.context())
    }

    fn size(&self, key: &str) -> Result<usize, String> {
        self.json[key]
            .as_u64()
            .map(|value| value as usize)
            .ok_or_else(|| self.missing(key))
    }

    fn number(&self, key: &str) -> Result<f64, String> {
        self.json[key].as_f64().ok_or_else(|| self.missing(key))
    }

    fn flag(&self, key: &str) -> Result<bool, String> {
        self.json[key].as_bool().ok_or_else(|| self.missing(key))
    }

    /// Un entier facultatif : absent, il vaut `None`.
    fn optional_size(&self, key: &str) -> Result<Option<usize>, String> {
        match &self.json[key] {
            Value::Null => Ok(None),
            value => value
                .as_u64()
                .map(|value| Some(value as usize))
                .ok_or_else(|| format!("{}`{key}` illisible", self.context())),
        }
    }

    /// Un choix écrit par son nom. Absent, il vaut `default` s'il y en a
    /// un.
    fn choice<T: FromStr<Err = String>>(&self, key: &str, default: Option<T>) -> Result<T, String> {
        match (&self.json[key], default) {
            (Value::Null, Some(default)) => Ok(default),
            (Value::String(text), _) => text
                .parse()
                .map_err(|error| format!("{}champ `{key}` : {error}", self.context())),
            (Value::Null, None) => Err(self.missing(key)),
            (other, _) => Err(format!(
                "{}champ `{key}` : `{other}` inattendu",
                self.context()
            )),
        }
    }
}

/// Une pièce facultative : absente du texte, elle est absente du modèle.
fn optional<T>(
    json: &Value,
    read: impl FnOnce(&Value) -> Result<T, String>,
) -> Result<Option<T>, String> {
    if json.is_null() {
        Ok(None)
    } else {
        read(json).map(Some)
    }
}

// Chaque pièce qui a ses propres réglages s'écrit en un objet, et se relit
// par la fonction qui suit celle qui l'écrit. Les variantes d'une pièce sont
// désignées par leur nom (`name()`), dans le champ `kind`.

fn attention_to_json(attention: AttentionKind) -> Value {
    match attention {
        AttentionKind::Standard => json!({ "kind": attention.name() }),
        AttentionKind::Latent(dims) => json!({
            "kind": attention.name(),
            "latent": dims.latent,
            "query_latent": dims.query_latent,
            "rope": dims.rope,
        }),
    }
}

fn attention_from_json(json: &Value) -> Result<AttentionKind, String> {
    let fields = Fields::new(json, "attention");
    match json["kind"].as_str() {
        None if json.is_null() => Ok(AttentionKind::Standard),
        Some(AttentionKind::STANDARD) => Ok(AttentionKind::Standard),
        Some(AttentionKind::LATENT) => Ok(AttentionKind::Latent(LatentDims {
            latent: fields.size("latent")?,
            query_latent: fields.size("query_latent")?,
            rope: fields.size("rope")?,
        })),
        _ => Err(format!(
            "configuration : attention `{}` inconnue, choix possibles : {}",
            json["kind"],
            AttentionKind::NAMES.join(", ")
        )),
    }
}

fn experts_to_json(experts: &ExpertDims) -> Value {
    let balance = experts.balance;
    json!({
        "shared": experts.shared,
        "routed": experts.routed,
        "active": experts.active,
        "hidden": experts.hidden,
        "balance": match balance {
            Balance::None => json!({ "kind": balance.name() }),
            Balance::Bias { rate } => json!({ "kind": balance.name(), "rate": rate }),
            Balance::Loss { weight } => json!({ "kind": balance.name(), "weight": weight }),
        },
    })
}

fn experts_from_json(json: &Value) -> Result<ExpertDims, String> {
    let fields = Fields::new(json, "experts");
    let balance = &json["balance"];
    let settings = Fields::new(balance, "experts");
    let balance = match balance["kind"].as_str() {
        None if balance.is_null() => Balance::None,
        Some(Balance::NONE) => Balance::None,
        Some(Balance::BIAS) => Balance::Bias {
            rate: settings.number("rate")?,
        },
        Some(Balance::LOSS) => Balance::Loss {
            weight: settings.number("weight")?,
        },
        _ => {
            return Err(format!(
                "configuration : équilibrage `{}` inconnu, choix possibles : {}",
                balance["kind"],
                Balance::NAMES.join(", ")
            ));
        }
    };
    Ok(ExpertDims {
        shared: fields.size("shared")?,
        routed: fields.size("routed")?,
        active: fields.size("active")?,
        hidden: fields.size("hidden")?,
        balance,
    })
}

fn multi_token_to_json(multi_token: &MultiToken) -> Value {
    json!({
        "early_weight": multi_token.early_weight,
        "late_weight": multi_token.late_weight,
    })
}

fn multi_token_from_json(json: &Value) -> Result<MultiToken, String> {
    let fields = Fields::new(json, "multi_token");
    Ok(MultiToken {
        early_weight: fields.number("early_weight")?,
        late_weight: fields.number("late_weight")?,
    })
}

fn memory_to_json(memory: &MemoryDims) -> Value {
    json!({
        "rows": memory.rows,
        "heads": memory.heads,
        "max_order": memory.max_order,
        "dim": memory.dim,
        "layer": memory.layer,
        "gate": memory.gate,
        "conv": memory.conv,
        "storage": memory.storage.name(),
        "table_std": memory.table_std,
        "lr_scale": memory.lr_scale,
    })
}

fn memory_from_json(json: &Value) -> Result<MemoryDims, String> {
    let fields = Fields::new(json, "mémoire");
    Ok(MemoryDims {
        rows: fields.size("rows")?,
        heads: fields.size("heads")?,
        max_order: fields.size("max_order")?,
        dim: fields.size("dim")?,
        layer: fields.size("layer")?,
        gate: fields.flag("gate")?,
        conv: fields.flag("conv")?,
        storage: fields.choice("storage", None)?,
        table_std: fields.number("table_std")?,
        lr_scale: fields.number("lr_scale")?,
    })
}
