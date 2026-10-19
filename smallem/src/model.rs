//! GPT : un Transformer « decoder-only ».
//!
//! ```text
//! tokens [B, T] → embeddings de tokens + embeddings de positions [B, T, C]
//!               → N blocs (attention causale, puis feed-forward)
//!               → LayerNorm finale
//!               → logits [B, T, V]
//! ```
//!
//! À chaque position, le modèle prédit le token suivant en ne voyant que
//! les positions précédentes. Une séquence de `T` tokens fournit donc `T`
//! exemples d'entraînement d'un coup.

use std::cell::Cell;
use std::collections::BTreeMap;

use rand::Rng;
use serde_json::{Value, json};

use crate::autograd::{Gradients, Tape, Var};
use crate::backend::{Backend, Float};
use crate::cache::{CacheStep, KvCache, PAGE_SIZE, SequenceId};
use crate::nn::{
    Adapter, Architecture, AttentionKind, Balance, Block, Calibration, Embedding, ExpertDims,
    FeedForwardKind, LatentDims, Memory, MemoryDims, MemoryStorage, MixtureOfExperts, Module,
    MultiToken, MultiTokenHead, Norm, Parameter, PositionKind, Rope, module,
};
use crate::quantize::{Format, Quantized};

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
        let mut json = json!({
            "vocab": self.vocab,
            "context": self.context,
            "layers": self.layers,
            "heads": self.heads,
            "embed": self.embed,
            "architecture": {
                "norm": self.architecture.norm.name(),
                "feed_forward": self.architecture.feed_forward.name(),
                "positions": self.architecture.positions.name(),
                "attention": match self.architecture.attention {
                    AttentionKind::Standard => json!({ "kind": "standard" }),
                    AttentionKind::Latent(dims) => json!({
                        "kind": "mla",
                        "latent": dims.latent,
                        "query_latent": dims.query_latent,
                        "rope": dims.rope,
                    }),
                },
            },
        });
        // Un modèle sans mélange d'experts n'en parle pas.
        if let Some(experts) = self.architecture.experts {
            json["architecture"]["experts"] = json!({
                "shared": experts.shared,
                "routed": experts.routed,
                "active": experts.active,
                "hidden": experts.hidden,
                "balance": match experts.balance {
                    Balance::None => json!({ "kind": "none" }),
                    Balance::Bias { rate } => json!({ "kind": "bias", "rate": rate }),
                    Balance::Loss { weight } => json!({ "kind": "loss", "weight": weight }),
                },
            });
        }
        if let Some(multi_token) = self.architecture.multi_token {
            json["architecture"]["multi_token"] = json!({
                "early_weight": multi_token.early_weight,
                "late_weight": multi_token.late_weight,
            });
        }
        if self.architecture.passes != 1 {
            json["architecture"]["passes"] = json!(self.architecture.passes);
        }
        if let Some(own_keys) = self.architecture.own_keys {
            json["architecture"]["own_keys"] = json!(own_keys);
        }
        if let Some(memory) = self.architecture.memory {
            json["architecture"]["memory"] = json!({
                "rows": memory.rows,
                "heads": memory.heads,
                "max_order": memory.max_order,
                "dim": memory.dim,
                "layer": memory.layer,
                "gate": memory.gate,
                "conv": memory.conv,
                "storage": match memory.storage {
                    MemoryStorage::Dense => "dense",
                    MemoryStorage::Sparse => "sparse",
                    MemoryStorage::Host => "host",
                },
                "table_std": memory.table_std,
                "lr_scale": memory.lr_scale,
            });
        }
        json
    }

    fn from_json(json: &Value) -> Result<Self, String> {
        let size = |key: &str| -> Result<usize, String> {
            json[key]
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| format!("configuration : champ `{key}` absent ou illisible"))
        };
        // Une pièce absente est celle de GPT-2 : c'est le modèle de départ,
        // et un fichier n'a à nommer que ce qui s'en écarte.
        fn choice<T: std::str::FromStr<Err = String>>(
            architecture: &Value,
            key: &str,
            default: T,
        ) -> Result<T, String> {
            match &architecture[key] {
                Value::Null => Ok(default),
                Value::String(text) => text
                    .parse()
                    .map_err(|error| format!("configuration : champ `{key}` : {error}")),
                other => Err(format!(
                    "configuration : champ `{key}` : `{other}` inattendu"
                )),
            }
        }
        // L'attention a ses propres tailles : c'est un objet, dont `kind`
        // dit lequel.
        fn attention(json: &Value) -> Result<AttentionKind, String> {
            let dim = |key: &str| -> Result<usize, String> {
                json[key]
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| {
                        format!("configuration : attention : champ `{key}` absent ou illisible")
                    })
            };
            match &json["kind"] {
                Value::Null if json.is_null() => Ok(AttentionKind::Standard),
                Value::String(kind) if kind == "standard" => Ok(AttentionKind::Standard),
                Value::String(kind) if kind == "mla" => Ok(AttentionKind::Latent(LatentDims {
                    latent: dim("latent")?,
                    query_latent: dim("query_latent")?,
                    rope: dim("rope")?,
                })),
                other => Err(format!(
                    "configuration : attention `{other}` inconnue, choix possibles : standard, mla"
                )),
            }
        }
        fn experts(json: &Value) -> Result<Option<ExpertDims>, String> {
            if json.is_null() {
                return Ok(None);
            }
            let field =
                |key: &str| format!("configuration : experts : champ `{key}` absent ou illisible");
            let size = |key: &str| -> Result<usize, String> {
                json[key]
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| field(key))
            };
            let balance = &json["balance"];
            let number = |key: &str| balance[key].as_f64().ok_or_else(|| field(key));
            let balance = match balance["kind"].as_str() {
                None if balance.is_null() => Balance::None,
                Some("none") => Balance::None,
                Some("bias") => Balance::Bias {
                    rate: number("rate")?,
                },
                Some("loss") => Balance::Loss {
                    weight: number("weight")?,
                },
                _ => {
                    return Err(format!(
                        "configuration : équilibrage `{}` inconnu, choix possibles : none, bias, loss",
                        balance["kind"]
                    ));
                }
            };
            Ok(Some(ExpertDims {
                shared: size("shared")?,
                routed: size("routed")?,
                active: size("active")?,
                hidden: size("hidden")?,
                balance,
            }))
        }
        fn multi_token(json: &Value) -> Result<Option<MultiToken>, String> {
            if json.is_null() {
                return Ok(None);
            }
            let weight = |key: &str| {
                json[key].as_f64().ok_or_else(|| {
                    format!("configuration : multi_token : champ `{key}` absent ou illisible")
                })
            };
            Ok(Some(MultiToken {
                early_weight: weight("early_weight")?,
                late_weight: weight("late_weight")?,
            }))
        }
        fn memory(json: &Value) -> Result<Option<MemoryDims>, String> {
            if json.is_null() {
                return Ok(None);
            }
            let field =
                |key: &str| format!("configuration : mémoire : champ `{key}` absent ou illisible");
            let size = |key: &str| -> Result<usize, String> {
                json[key]
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| field(key))
            };
            let flag = |key: &str| json[key].as_bool().ok_or_else(|| field(key));
            let number = |key: &str| json[key].as_f64().ok_or_else(|| field(key));
            Ok(Some(MemoryDims {
                rows: size("rows")?,
                heads: size("heads")?,
                max_order: size("max_order")?,
                dim: size("dim")?,
                layer: size("layer")?,
                gate: flag("gate")?,
                conv: flag("conv")?,
                storage: match json["storage"].as_str() {
                    Some("dense") => MemoryStorage::Dense,
                    Some("sparse") => MemoryStorage::Sparse,
                    Some("host") => MemoryStorage::Host,
                    _ => return Err(field("storage")),
                },
                table_std: number("table_std")?,
                lr_scale: number("lr_scale")?,
            }))
        }
        /// Un entier facultatif : absent, il vaut `None`.
        fn count(json: &Value, name: &str) -> Result<Option<usize>, String> {
            if json.is_null() {
                return Ok(None);
            }
            json.as_u64()
                .map(|value| Some(value as usize))
                .ok_or_else(|| format!("configuration : `{name}` illisible"))
        }
        let architecture = &json["architecture"];
        let gpt2 = Architecture::GPT2;
        let config = Self {
            vocab: size("vocab")?,
            context: size("context")?,
            layers: size("layers")?,
            heads: size("heads")?,
            embed: size("embed")?,
            architecture: Architecture {
                norm: choice(architecture, "norm", gpt2.norm)?,
                feed_forward: choice(architecture, "feed_forward", gpt2.feed_forward)?,
                positions: choice(architecture, "positions", gpt2.positions)?,
                attention: attention(&architecture["attention"])?,
                experts: experts(&architecture["experts"])?,
                multi_token: multi_token(&architecture["multi_token"])?,
                passes: count(&architecture["passes"], "passes")?.unwrap_or(1),
                own_keys: count(&architecture["own_keys"], "own_keys")?,
                memory: memory(&architecture["memory"])?,
            },
        };
        config
            .validate()
            .map_err(|error| format!("configuration : {error}"))?;
        Ok(config)
    }
}

pub struct Gpt<B: Backend> {
    config: GptConfig,
    /// Un vecteur par token du vocabulaire. Cette même table sert, à la
    /// sortie, à transformer les vecteurs en logits.
    token_embedding: Embedding<B>,
    /// Un vecteur par position : l'attention ne connaît pas l'ordre des
    /// tokens, c'est cette table qui le lui apprend. Absente avec RoPE, où
    /// la position entre dans l'attention elle-même.
    position_embedding: Option<Embedding<B>>,
    blocks: Vec<Block<B>>,
    final_norm: Norm<B>,
    /// Le module de prédiction multi-token, s'il y en a un. Il ne sert qu'à
    /// [`Gpt::training_loss`].
    multi_token: Option<MultiTokenHead<B>>,
    /// La mémoire de n-grammes, s'il y en a une.
    memory: Option<Memory<B>>,
    /// Part de l'entraînement déjà faite, entre 0 et 1 : le poids de la
    /// prédiction multi-token en dépend.
    progress: Cell<f64>,
    /// Facteur des logits : 1, sauf sous μP (voir [`Gpt::apply_mup`]). Il
    /// n'est pas enregistré avec les poids.
    output_scale: f64,
}

impl<B: Backend> Gpt<B> {
    pub fn new(backend: &B, config: GptConfig, rng: &mut impl Rng) -> Self {
        if let Err(error) = config.validate() {
            panic!("{error}");
        }
        Self {
            config,
            token_embedding: Embedding::new(backend, config.vocab, config.embed, rng),
            position_embedding: match config.architecture.positions {
                PositionKind::Learned => {
                    Some(Embedding::new(backend, config.context, config.embed, rng))
                }
                PositionKind::Rotary => None,
            },
            blocks: (0..config.layers)
                .map(|layer| {
                    Block::new(
                        backend,
                        config.embed,
                        config.heads,
                        config.layers,
                        config.architecture,
                        config.architecture.borrows_keys(layer),
                        rng,
                    )
                })
                .collect(),
            final_norm: Norm::new(backend, config.architecture.norm, config.embed),
            multi_token: config.architecture.multi_token.map(|_| {
                MultiTokenHead::new(
                    backend,
                    config.embed,
                    config.heads,
                    config.layers,
                    config.architecture,
                    rng,
                )
            }),
            memory: config
                .architecture
                .memory
                .map(|dims| Memory::new(backend, dims, config.embed, rng)),
            progress: Cell::new(0.0),
            output_scale: 1.0,
        }
    }

    pub fn config(&self) -> GptConfig {
        self.config
    }

    /// `tokens` contient `batch` séquences de même longueur `T`, l'une après
    /// l'autre. Renvoie les logits `[batch, T, vocab]` : à chaque position,
    /// un score par token possible pour la position suivante.
    pub fn logits(&self, tape: &Tape<B>, tokens: &[u32], batch: usize) -> Var {
        self.head(tape, self.hidden(tape, tokens, batch))
    }

    /// Les états cachés à la sortie du dernier bloc, `[batch, T, C]` : tout
    /// le modèle sauf sa sortie. Les arguments sont ceux de [`Gpt::logits`].
    fn hidden(&self, tape: &Tape<B>, tokens: &[u32], batch: usize) -> Var {
        assert!(
            batch > 0 && tokens.len().is_multiple_of(batch),
            "{} tokens ne forment pas {batch} séquences de même longueur",
            tokens.len(),
        );
        let length = tokens.len() / batch;
        let context = self.config.context;
        assert!(
            length <= context,
            "séquence de {length} tokens, le contexte maximal est {context}"
        );
        let dim = self.config.embed;

        // `[B × T, C] → [B, T, C]`, plus les positions `[T, C]` qui
        // s'étirent sur le lot : toutes les séquences partagent les mêmes.
        let embedded = self.token_embedding.forward(tape, tokens);
        let mut x = tape.reshape(embedded, &[batch, length, dim]);
        if let Some(position_embedding) = &self.position_embedding {
            let positions: Vec<u32> = (0..length as u32).collect();
            x = tape.add(x, position_embedding.forward(tape, &positions));
        }
        // Avec RoPE, rien n'est ajouté à l'entrée : les angles des positions
        // sont calculés une fois ici, et chaque bloc s'en sert dans son
        // attention.
        let rope = self.rope(tape, 0, length);
        // Les lignes de la mémoire ne dépendent que des tokens : elles se
        // calculent avant tout le reste.
        let memory = self
            .memory
            .as_ref()
            .map(|memory| (memory, memory.hash().indices(tokens, batch)));

        // La pile est traversée une fois, ou plusieurs : les mêmes blocs
        // resservent alors, sur d'autres états cachés.
        for pass in 0..self.config.architecture.passes {
            // Les clés et les valeurs du dernier bloc qui en a calculé,
            // pour ceux qui empruntent les leurs.
            let mut lent = None;
            for (index, block) in self.blocks.iter().enumerate() {
                if let Some((memory, indices)) = &memory
                    && pass == 0
                    && index == memory.dims().layer
                {
                    x = memory.forward(tape, x, indices, length);
                }
                x = block.forward(tape, x, rope, None, &mut lent);
            }
        }
        if let Some((memory, indices)) = &memory
            && memory.dims().layer == self.blocks.len()
        {
            x = memory.forward(tape, x, indices, length);
        }
        x
    }

    /// La passe de la génération. `entries` donne, pour une ou plusieurs
    /// séquences suivies par `cache`, les tokens qui s'y ajoutent : toute
    /// l'amorce quand on la lit, un seul token par séquence quand on
    /// génère. Renvoie les logits `[R, vocab]` de ces `R` tokens, dans
    /// l'ordre des entrées.
    ///
    /// Seuls les nouveaux tokens traversent le modèle ; ce qu'ils ont
    /// besoin de savoir du passé, les clés et les valeurs de chaque couche,
    /// sort du cache, qui est complété au passage. C'est le calcul de
    /// [`Gpt::logits`] : à cause du masque causal, une position ne dépend
    /// jamais de celles qui la suivent, et traiter une séquence d'un coup ou
    /// morceau par morceau donne les mêmes nombres.
    ///
    /// Les tokens de toutes les séquences forment une seule suite `[1, R,
    /// C]`. Tout ce qui ne mélange pas les positions, c'est-à-dire tout sauf
    /// l'attention, les traite comme une séquence ordinaire ; l'attention
    /// sait, par le pas du cache, à quelle séquence appartient chacun.
    ///
    /// Le modèle ne voit jamais plus de `context` positions d'une séquence.
    /// Quand la suite ferait déborder une séquence, un modèle à positions
    /// relatives oublie ses plus anciennes positions ; avec des positions
    /// apprises, c'est une erreur.
    pub fn forward_cached(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        entries: &[(SequenceId, &[u32])],
    ) -> Var {
        let hidden = self.hidden_cached(tape, cache, entries);
        self.head_cached(tape, hidden)
    }

    /// Comme [`Gpt::forward_cached`], et rend aussi les états cachés à la
    /// sortie du dernier bloc, `[1, R, C]` : ce dont le module de prédiction
    /// multi-token a besoin.
    pub fn forward_cached_with_hidden(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        entries: &[(SequenceId, &[u32])],
    ) -> (Var, Var) {
        let hidden = self.hidden_cached(tape, cache, entries);
        (hidden, self.head_cached(tape, hidden))
    }

    /// Les états cachés de la passe de [`Gpt::forward_cached`], `[1, R, C]`.
    fn hidden_cached(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        entries: &[(SequenceId, &[u32])],
    ) -> Var {
        let context = self.config.context;
        let dim = self.config.embed;
        if !self.has_relative_positions() {
            for (id, tokens) in entries {
                let total = cache.len(*id) + tokens.len();
                assert!(
                    total <= context,
                    "séquence de {total} tokens, le contexte maximal est {context}"
                );
            }
        }
        let counts: Vec<(SequenceId, usize)> = entries
            .iter()
            .map(|(id, tokens)| (*id, tokens.len()))
            .collect();
        let (mut step, positions) = cache.begin(tape, &counts, context);
        let tokens: Vec<u32> = entries
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().copied())
            .collect();
        let rows = tokens.len();

        let embedded = self.token_embedding.forward(tape, &tokens);
        let mut x = tape.reshape(embedded, &[1, rows, dim]);
        if let Some(position_embedding) = &self.position_embedding {
            let positions: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let embedded = position_embedding.forward(tape, &positions);
            x = tape.add(x, tape.reshape(embedded, &[1, rows, dim]));
        }
        self.set_rope(tape, &mut step, &positions);
        // Les n-grammes des nouveaux tokens ont besoin des derniers tokens
        // de chaque séquence : le cache les garde.
        let memory = self.memory.as_ref().map(|memory| {
            let hash = memory.hash();
            let keep = hash.max_order() - 1;
            let mut indices = Vec::with_capacity(rows * hash.tables());
            let mut gram = vec![0u32; hash.max_order()];
            for (id, tokens) in entries {
                let mut combined = cache.recent(*id).to_vec();
                let past = combined.len();
                combined.extend_from_slice(tokens);
                for t in past..combined.len() {
                    crate::ngram::suffix(&combined, t, &mut gram);
                    indices.extend(hash.position_indices(&gram));
                }
                cache.remember(*id, tokens, keep);
            }
            (memory, indices)
        });

        let layers = self.blocks.len();
        for pass in 0..self.config.architecture.passes {
            let mut lent = None;
            for (index, block) in self.blocks.iter().enumerate() {
                if let Some((memory, indices)) = &memory
                    && pass == 0
                    && index == memory.dims().layer
                {
                    x = memory.forward_cached(tape, x, indices, cache, &counts);
                }
                let stage = cache.stage_mut(pass * layers + index);
                x = block.forward(tape, x, None, Some((stage, &step)), &mut lent);
            }
        }
        if let Some((memory, indices)) = &memory
            && memory.dims().layer == self.blocks.len()
        {
            x = memory.forward_cached(tape, x, indices, cache, &counts);
        }
        x
    }

    /// Les angles de RoPE des tokens d'une passe de génération, s'il y en
    /// a, dans le pas du cache.
    fn set_rope(&self, tape: &Tape<B>, step: &mut CacheStep, positions: &[usize]) {
        if self.config.architecture.positions == PositionKind::Rotary {
            let rotary = self.config.rotary_dim();
            step.rope = Some(Rope::at(tape, positions, rotary, self.config.heads));
            if let AttentionKind::Latent(_) = self.config.architecture.attention {
                step.rope_single = Some(Rope::at(tape, positions, rotary, 1));
            }
        }
    }

    /// La sortie du modèle en génération, de `[1, R, C]` à `[R, V]` : le
    /// calcul de [`Gpt::head`], par `a · bᵀ`, pour ne pas recopier la table
    /// d'embedding transposée à chaque token.
    fn head_cached(&self, tape: &Tape<B>, hidden: Var) -> Var {
        let rows = tape.shape(hidden)[1];
        let x = self.final_norm.forward(tape, hidden);
        let logits = self.token_embedding.unembed(tape, x);
        let logits = self.scale_output(tape, logits);
        tape.reshape(logits, &[rows, self.config.vocab])
    }

    /// Le module de prédiction multi-token en génération. `hidden` contient
    /// les états cachés `h_i` du modèle, `[n, C]`, de positions consécutives
    /// d'une séquence que suit `cache`, un cache d'un seul étage, et `next`
    /// les tokens `t_{i+1}` qui les suivent. Rend `[n, V]` : à chaque
    /// position, les logits du token `t_{i+2}`.
    ///
    /// Le bloc du module a sa propre attention, sur les positions déjà
    /// passées par lui : d'où son cache.
    pub fn multi_token_cached(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        id: SequenceId,
        hidden: Var,
        next: &[u32],
    ) -> Var {
        let module = self
            .multi_token
            .as_ref()
            .expect("ce modèle n'a pas de module de prédiction multi-token");
        let (rows, dim) = (next.len(), self.config.embed);
        let (mut step, positions) = cache.begin(tape, &[(id, rows)], self.config.context);
        self.set_rope(tape, &mut step, &positions);
        let hidden = tape.reshape(hidden, &[1, rows, dim]);
        let next = tape.reshape(self.token_embedding.forward(tape, next), &[1, rows, dim]);
        let out = module.forward_cached(tape, hidden, next, cache.stage_mut(0), &step);
        self.head_cached(tape, out)
    }

    /// Un cache pour le module de prédiction multi-token : un seul étage,
    /// une séquence.
    pub fn multi_token_cache(&self) -> KvCache<B> {
        KvCache::new(1, self.config.context.div_ceil(PAGE_SIZE) + 1, PAGE_SIZE)
    }

    /// Passe le modèle, juste construit, en paramétrisation μP (Yang et al.,
    /// 2022), relativement à la largeur `base` : à cette largeur, rien ne
    /// change. Pour une largeur `C = m × base` :
    ///
    /// - les matrices cachées, celles des blocs, sont initialisées
    ///   `√m` fois plus petites, et apprennent avec un taux `m` fois plus
    ///   petit ;
    /// - les logits sont divisés par `m` ;
    /// - la table d'embedding, les gains et les biais ne changent pas.
    ///
    /// L'échelle de l'attention, `1/√D`, ne change pas non plus : dans
    /// notre famille de modèles, les têtes gardent la même taille et c'est
    /// leur nombre qui grandit.
    ///
    /// L'idée : sous la paramétrisation standard, une mise à jour d'Adam
    /// change chaque poids d'environ le taux, et une couche de largeur `C`
    /// additionne `C` de ces changements. Ses sorties bougent donc `m` fois
    /// plus quand le modèle est `m` fois plus large, et le meilleur taux
    /// baisse avec la largeur. Diviser le taux des matrices cachées par `m`
    /// rend ce mouvement indépendant de la largeur.
    pub fn apply_mup(&mut self, backend: &B, base: usize) {
        let m = self.config.embed as f64 / base as f64;
        let shrink = backend.scalar(B::Scalar::from_f64(1.0 / m.sqrt()));
        for block in &mut self.blocks {
            block.walk_mut("", &mut |_, parameter| {
                if backend.shape(parameter.value()).len() == 2 {
                    let value = backend.mul(parameter.value(), &shrink);
                    *parameter.value_mut() = value;
                    parameter.set_lr_scale(1.0 / m);
                }
            });
        }
        self.output_scale = 1.0 / m;
    }

    /// Ajoute un adaptateur LoRA de rang `rank` à chaque couche linéaire des
    /// blocs, et à la table d'embedding, puis gèle tous les autres
    /// paramètres : seuls les adaptateurs apprendront. Rend le nombre de
    /// paramètres entraînés.
    ///
    /// La table d'embedding a son adaptateur, `[V, r] · [r, C]`, parce
    /// qu'elle porte les tokens de rôle, nés identiques : gelée, elle ne
    /// pourrait pas apprendre à les distinguer.
    pub fn add_adapters(&mut self, backend: &B, rank: usize, rng: &mut impl Rng) -> usize {
        for parameter in self.parameters_mut() {
            parameter.set_frozen(true);
        }
        let mut count = 0;
        let (vocab, dim) = (self.config.vocab, self.config.embed);
        let adapter = Adapter::new(backend, vocab, dim, rank, rng);
        count += adapter.len(backend);
        self.token_embedding.adapter = Some(Box::new(adapter));
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                let shape = backend.shape(linear.weight.value()).to_vec();
                let adapter = Adapter::new(backend, shape[0], shape[1], rank, rng);
                count += adapter.len(backend);
                linear.adapter = Some(Box::new(adapter));
            }
        }
        count
    }

    /// Ajoute chaque correction à sa matrice, `W ← W + A · B`, retire les
    /// adaptateurs et dégèle tout : le modèle redevient un modèle ordinaire,
    /// qui donne les mêmes logits.
    pub fn merge_adapters(&mut self, backend: &B) {
        if let Some(adapter) = self.token_embedding.adapter.take() {
            adapter.merge_into(backend, &mut self.token_embedding.weight);
        }
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                if let Some(adapter) = linear.adapter.take() {
                    adapter.merge_into(backend, &mut linear.weight);
                }
            }
        }
        for parameter in self.parameters_mut() {
            parameter.set_frozen(false);
        }
    }

    /// Agrandit le vocabulaire à `vocab` tokens : la table d'embedding,
    /// qui sert aussi à la sortie, reçoit de nouvelles lignes. Chacune est
    /// la moyenne des lignes existantes : un token nouveau commence au
    /// centre de ceux que le modèle connaît, et son logit, à la moyenne des
    /// autres. Les logits des anciens tokens ne changent pas.
    pub fn grow_vocab(&mut self, backend: &B, vocab: usize) {
        let (old, dim) = (self.config.vocab, self.config.embed);
        assert!(vocab >= old, "le vocabulaire ne peut que grandir");
        let mut table = backend.to_vec(self.token_embedding.weight.value());
        let mut mean = vec![0.0; dim];
        for row in table.chunks(dim) {
            for (m, value) in mean.iter_mut().zip(row) {
                *m += value.to_f64() / old as f64;
            }
        }
        for _ in old..vocab {
            table.extend(mean.iter().map(|&m| B::Scalar::from_f64(m)));
        }
        *self.token_embedding.weight.value_mut() = backend.from_vec(table, &[vocab, dim]);
        self.config.vocab = vocab;
    }

    /// L'entropie croisée moyenne sur les seules positions de `scored` :
    /// celles dont la cible est un token écrit par l'assistant. Les autres
    /// positions sont lues, et leur contenu change donc la perte, mais leur
    /// cible n'y entre pas.
    pub fn masked_loss(
        &self,
        tape: &Tape<B>,
        tokens: &[u32],
        targets: &[u32],
        scored: &[bool],
        batch: usize,
    ) -> Var {
        assert!(
            tokens.len() == targets.len() && tokens.len() == scored.len(),
            "une cible et un masque par token sont attendus"
        );
        let rows: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&i| scored[i as usize])
            .collect();
        assert!(!rows.is_empty(), "aucune position notée");
        let logits = self.logits(tape, tokens, batch);
        let flat = tape.reshape(logits, &[tokens.len(), self.config.vocab]);
        let kept: Vec<u32> = rows.iter().map(|&row| targets[row as usize]).collect();
        tape.cross_entropy(tape.gather_rows(flat, &rows), &kept)
    }

    /// Quantifie au format `format` les matrices de toutes les couches
    /// linéaires : celles des attentions et des feed-forward. La table
    /// d'embedding, qui sert aussi à la sortie, reste en `f32`. Rend, pour
    /// chaque matrice, son nom et sa forme quantifiée.
    pub fn quantize(&mut self, backend: &B, format: Format) -> Vec<(String, Quantized)> {
        self.quantize_with(backend, |_, weight, rows, cols| {
            Quantized::round_to_nearest(weight, rows, cols, format)
        })
    }

    /// Comme [`Gpt::quantize`], chaque matrice étant quantifiée par
    /// `quantizer`, qui reçoit son nom, ses poids `[rows, cols]` et leurs
    /// dimensions.
    pub fn quantize_with(
        &mut self,
        backend: &B,
        mut quantizer: impl FnMut(&str, &[f64], usize, usize) -> Quantized,
    ) -> Vec<(String, Quantized)> {
        let mut done = Vec::new();
        self.walk_mut("", &mut |name, parameter| {
            if Self::is_quantizable(backend, &name, parameter) {
                let shape = backend.shape(parameter.value()).to_vec();
                let weight: Vec<f64> = backend
                    .to_vec(parameter.value())
                    .iter()
                    .map(|v| v.to_f64())
                    .collect();
                let quantized = quantizer(&name, &weight, shape[0], shape[1]);
                parameter.set_quantized(backend, &quantized);
                done.push((name, quantized));
            }
        });
        done
    }

    /// Quantifie aussi la table d'embedding, pour la sortie du modèle (voir
    /// [`Embedding::quantize_output`](crate::nn::Embedding::quantize_output)).
    pub fn quantize_output(&mut self, backend: &B, format: Format) -> Quantized {
        self.token_embedding.quantize_output(backend, format)
    }

    /// Les matrices que [`Gpt::quantize`] quantifie : celles des couches
    /// linéaires, pas les tables d'embedding.
    fn is_quantizable(backend: &B, name: &str, parameter: &Parameter<B>) -> bool {
        let matrix = backend.shape(parameter.value()).len() == 2;
        matrix && name.ends_with(".weight") && !name.ends_with("embedding.weight")
    }

    /// Commence à observer les entrées de toutes les matrices que
    /// [`Gpt::quantize`] quantifierait (voir [`Parameter::observe`]).
    pub fn start_calibration(&self, backend: &B, host: bool) {
        self.walk("", &mut |name, parameter| {
            if Self::is_quantizable(backend, &name, parameter) {
                parameter.start_calibration(host);
            }
        });
    }

    /// Arrête l'observation, et rend ce qu'elle a appris de chaque matrice.
    pub fn take_calibration(&self) -> Vec<(String, Calibration<B>)> {
        let mut taken = Vec::new();
        self.walk("", &mut |name, parameter| {
            if let Some(calibration) = parameter.take_calibration() {
                taken.push((name, calibration));
            }
        });
        taken
    }

    /// Un cache vide pour ce modèle, de quoi suivre `sequences` séquences
    /// de `context` positions.
    pub fn cache(&self, sequences: usize) -> KvCache<B> {
        KvCache::for_model(&self.config, sequences)
    }

    /// La sortie du modèle : des états cachés `[…, C]` aux logits `[…, V]`.
    fn head(&self, tape: &Tape<B>, x: Var) -> Var {
        let x = self.final_norm.forward(tape, x);

        // Partage des poids : la table `[V, C]` qui a transformé les tokens
        // en vecteurs sert, transposée en `[C, V]`, à transformer les
        // vecteurs en scores de tokens. Le logit d'un token est le produit
        // scalaire entre le vecteur de sortie et l'embedding de ce token.
        if self.token_embedding.output.is_some() {
            return self.scale_output(tape, self.token_embedding.unembed(tape, x));
        }
        let unembedding = tape.transpose(self.token_embedding.table(tape), 0, 1);
        self.scale_output(tape, tape.matmul(x, unembedding))
    }

    /// Les logits multipliés par le facteur de sortie, s'il n'est pas 1.
    fn scale_output(&self, tape: &Tape<B>, logits: Var) -> Var {
        if self.output_scale == 1.0 {
            logits
        } else {
            tape.mul(logits, tape.scalar(B::Scalar::from_f64(self.output_scale)))
        }
    }

    /// Entropie croisée moyenne sur toutes les positions de toutes les
    /// séquences. `targets[i]` est le token qui suit `tokens[i]`.
    pub fn loss(&self, tape: &Tape<B>, tokens: &[u32], targets: &[u32], batch: usize) -> Var {
        assert_eq!(
            tokens.len(),
            targets.len(),
            "une cible par token est attendue"
        );
        let logits = self.logits(tape, tokens, batch);
        // L'entropie croisée attend `[N, V]` : toutes les positions de
        // toutes les séquences deviennent des exemples indépendants.
        let flat = tape.reshape(logits, &[tokens.len(), self.config.vocab]);
        tape.cross_entropy(flat, targets)
    }

    /// La perte à minimiser pendant l'entraînement : celle de
    /// [`Gpt::loss`], plus les termes qui ne servent qu'à guider
    /// l'apprentissage : la pénalité d'équilibrage des mélanges d'experts
    /// qui en ont une, et la perte de la prédiction multi-token.
    ///
    /// La perte de validation reste [`Gpt::loss`] : c'est elle qui se
    /// compare d'un modèle à l'autre.
    pub fn training_loss(
        &self,
        tape: &Tape<B>,
        tokens: &[u32],
        targets: &[u32],
        batch: usize,
    ) -> Var {
        assert_eq!(
            tokens.len(),
            targets.len(),
            "une cible par token est attendue"
        );
        let hidden = self.hidden(tape, tokens, batch);
        let logits = self.head(tape, hidden);
        let flat = tape.reshape(logits, &[tokens.len(), self.config.vocab]);
        let mut loss = tape.cross_entropy(flat, targets);

        if let Some(extra) = self.multi_token_loss(tape, hidden, targets, batch) {
            loss = tape.add(loss, extra);
        }
        for experts in self.experts() {
            if let Some(penalty) = experts.penalty(tape) {
                loss = tape.add(loss, penalty);
            }
        }
        loss
    }

    /// La perte de la prédiction multi-token, déjà multipliée par son
    /// poids. `None` si le modèle n'a pas ce module, ou si son poids est
    /// nul.
    ///
    /// À la position `i`, la cible `targets[i]` est le token `t_{i+1}` :
    /// le module le reçoit en entrée, et doit prédire `targets[i + 1]`,
    /// soit `t_{i+2}`. La dernière position de chaque séquence n'a pas de
    /// cible aussi lointaine : elle est laissée de côté.
    fn multi_token_loss(
        &self,
        tape: &Tape<B>,
        hidden: Var,
        targets: &[u32],
        batch: usize,
    ) -> Option<Var> {
        let module = self.multi_token.as_ref()?;
        let weight = self
            .config
            .architecture
            .multi_token?
            .weight_at(self.progress.get());
        let length = targets.len() / batch;
        if weight == 0.0 || length < 2 {
            return None;
        }
        let dim = self.config.embed;
        // Toutes les positions sauf la dernière de chaque séquence.
        let rows: Vec<u32> = (0..batch)
            .flat_map(|sequence| (0..length - 1).map(move |t| (sequence * length + t) as u32))
            .collect();
        let next: Vec<u32> = rows.iter().map(|&row| targets[row as usize]).collect();
        let after: Vec<u32> = rows.iter().map(|&row| targets[row as usize + 1]).collect();

        let shape = [batch, length - 1, dim];
        let hidden = tape.reshape(hidden, &[batch * length, dim]);
        let hidden = tape.reshape(tape.gather_rows(hidden, &rows), &shape);
        let next = tape.reshape(self.token_embedding.forward(tape, &next), &shape);
        let rope = self.rope(tape, 0, length - 1);
        let logits = self.head(tape, module.forward(tape, hidden, next, rope));
        let flat = tape.reshape(logits, &[rows.len(), self.config.vocab]);
        let loss = tape.cross_entropy(flat, &after);
        Some(tape.mul(loss, tape.scalar(B::Scalar::from_f64(weight))))
    }

    /// Dit au modèle quelle part de l'entraînement est faite, entre 0 et
    /// 1 : le poids de la prédiction multi-token en dépend.
    /// La mémoire de n-grammes, s'il y en a une.
    pub fn memory(&self) -> Option<&Memory<B>> {
        self.memory.as_ref()
    }

    pub fn memory_mut(&mut self) -> Option<&mut Memory<B>> {
        self.memory.as_mut()
    }

    /// Garde le gradient des lignes que la mémoire a lues pendant la passe
    /// de la tape `tape` : voir [`Memory::collect`].
    pub fn collect_memory_gradient(&mut self, tape: u64, grads: &Gradients<B>) {
        if let Some(memory) = &mut self.memory {
            memory.collect(tape, grads);
        }
    }

    /// Le pas d'AdamW des lignes lues de la mémoire : voir [`Memory::apply`].
    pub fn apply_memory_gradient(&mut self, backend: &B, learning_rate: f64, grad_scale: f64) {
        if let Some(memory) = &mut self.memory {
            memory.apply(backend, learning_rate, grad_scale);
        }
    }

    pub fn set_training_progress(&self, progress: f64) {
        self.progress.set(progress);
    }

    /// Les mélanges d'experts du modèle, un par bloc qui en a un. Celui du
    /// module de prédiction multi-token vient en dernier.
    pub fn experts(&self) -> impl Iterator<Item = &MixtureOfExperts<B>> {
        self.blocks
            .iter()
            .chain(self.multi_token.iter().map(|module| &module.block))
            .filter_map(|block| block.feed_forward.experts())
    }

    fn experts_mut(&mut self) -> impl Iterator<Item = &mut MixtureOfExperts<B>> {
        self.blocks
            .iter_mut()
            .chain(self.multi_token.iter_mut().map(|module| &mut module.block))
            .filter_map(|block| block.feed_forward.experts_mut())
    }

    /// Pour chaque mélange d'experts, le nombre de tokens reçus par chacun
    /// de ses experts aiguillés depuis le dernier appel.
    pub fn take_expert_load(&self) -> Vec<Vec<u64>> {
        self.experts().map(MixtureOfExperts::take_load).collect()
    }

    /// Oublie les comptes de tokens en cours : à appeler après une passe
    /// qui n'était pas de l'entraînement.
    pub fn reset_expert_load(&self) {
        self.experts().for_each(MixtureOfExperts::reset_load);
    }

    /// Règle les biais d'aiguillage après un pas d'entraînement (voir
    /// [`MixtureOfExperts::rebalance`]).
    pub fn rebalance_experts(&mut self, backend: &B) {
        for experts in self.experts_mut() {
            experts.rebalance(backend);
        }
    }

    /// Fait calculer chaque expert sur tous les tokens (`true`) ou sur les
    /// seuls tokens qui l'ont choisi (`false`, le réglage normal).
    pub fn set_dense_experts(&mut self, dense: bool) {
        for experts in self.experts_mut() {
            experts.dense = dense;
        }
    }

    /// Les angles de RoPE des positions `start..start + length`, si le
    /// modèle utilise RoPE.
    fn rope(&self, tape: &Tape<B>, start: usize, length: usize) -> Option<Rope> {
        match self.config.architecture.positions {
            PositionKind::Learned => None,
            PositionKind::Rotary => Some(Rope::new(tape, start, length, self.config.rotary_dim())),
        }
    }

    /// Vrai si le modèle ne connaît que les distances entre positions
    /// (RoPE), et non leur rang dans la séquence. Une fenêtre de tokens
    /// peut alors glisser sans que rien ne soit à recalculer.
    pub fn has_relative_positions(&self) -> bool {
        self.position_embedding.is_none()
    }
}

module!(Gpt {
    token_embedding,
    position_embedding,
    blocks,
    final_norm,
    multi_token,
    memory
});

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::Float;
    use crate::backend::cpu::{Cpu, CpuTensor};
    use crate::data::Dataset;
    use crate::nn::{FeedForwardKind, Module, NormKind};
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
            let hidden = crate::nn::SwiGluFeedForward::<B>::hidden(dim);
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
        let hidden = crate::nn::SwiGluFeedForward::<B>::hidden(8);
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
}
