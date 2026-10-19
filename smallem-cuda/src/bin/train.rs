//! Entraîne un GPT sur FineWeb-Edu et enregistre le meilleur modèle.
//!
//! Usage : `cargo oxide run --bin train -- [options]`
//!
//! ```text
//! --config small|large|xl  architecture (défaut : small)
//! --steps N                nombre de pas (défaut : 5000)
//! --batch N                séquences par lot (défaut : 32)
//! --accumulate N           lots accumulés par pas (défaut : 1)
//! --lr X                   taux d'apprentissage au pic (défaut : 1e-3)
//! --warmup N               pas de montée du taux (défaut : 200)
//! --optimizer adamw|muon   optimiseur (défaut : adamw) ; muon traite les matrices de poids,
//!                          adamw gardant les embeddings, les gains et les biais.
//!                          Variantes de muon, à combiner après deux-points, comme dans
//!                          `muon:short,period=3` :
//!                            short      4 itérations de Newton-Schulz au lieu de 10
//!                            period=K   orthogonalise un pas sur K, normalise les lignes sinon
//!                            row=X      taille relative des pas par lignes (défaut : 0.15)
//!                            norm       NorMuon : normalisation par neurone
//!                            boost=X    SAMuon-lite : amplifie tout sauf la direction dominante
//! --weight-decay X         décroissance des poids (défaut : 0.1)
//! --clip X                 norme à laquelle le gradient est écrêté (défaut : 1)
//! --eval-every N           pas entre deux évaluations (défaut : 500)
//! --out FICHIER            où enregistrer le modèle (défaut : <data>/gpt-<config>.safetensors)
//! --data DOSSIER           où trouver les fichiers de tokens (défaut : ../data)
//! --seed N                 graine du hasard (défaut : 42)
//! --arch gpt2|modern       pièces des blocs (défaut : gpt2) ; modern = rms + swiglu + rope
//! --norm layer|rms         normalisation, seule
//! --ffn gelu|swiglu        feed-forward, seul
//! --positions learned|rope positions, seules
//! --attention standard|mla attention, seule ; mla = attention latente, qui demande rope
//! --latent N               avec mla : taille du latent des clés et valeurs (défaut : C/4)
//! --query-latent N         avec mla : taille du latent des requêtes (défaut : 3C/4)
//! --rope-dim N             avec mla : taille de la partie tournée (défaut : une demi-tête)
//! --experts                feed-forward en mélange d'experts, qui demande swiglu
//! --experts-shared N       experts traversés par tous les tokens (défaut : 1)
//! --experts-routed N       experts aiguillés (défaut : 8)
//! --experts-active N       experts aiguillés actifs par token (défaut : 2)
//! --experts-hidden N       largeur de chaque expert (défaut : celle du feed-forward / 3)
//! --balance none|bias|loss équilibrage de la charge des experts (défaut : none)
//! --balance-rate X         pas du biais (défaut : 0.001) ou poids de la perte (défaut : 0.01)
//! --dense-experts          calculer chaque expert sur tous les tokens (lent : la référence)
//! --layers N               nombre de couches, à la place de celui de la configuration
//! --embed N                largeur des vecteurs, à la place de celle de la configuration
//! --heads N                nombre de têtes, à la place de celui de la configuration
//! --budget X               calcul total, en opérations (par exemple 3e15) : le nombre de
//!                          pas s'en déduit, à la place de --steps
//! --mup BASE               paramétrisation μP, relative à la largeur BASE
//! --context N              longueur des séquences, à la place de celle de la configuration
//! --passes N               traversées de la pile de couches (défaut : 1) ; avec 2, le modèle
//!                          est deux fois plus profond sans paramètre de plus
//! --own-keys N             seules les N premières couches calculent des clés et des valeurs ;
//!                          les suivantes empruntent celles de la couche N
//! --mtp                    prédiction multi-token pendant l'entraînement (poids 0.3 puis 0.1)
//! --mtp-weights X,Y        ses poids : X pendant les deux premiers tiers, Y ensuite
//! --memory LIGNES          mémoire de n-grammes (Engram) : LIGNES visées par table ; par
//!                          défaut 8 têtes par ordre, bigrammes et trigrammes, d_mém = C,
//!                          avant la couche 1, porte et convolution, table creuse
//! --memory-heads N         têtes de hachage par ordre
//! --memory-order N         ordre le plus grand (défaut : 3)
//! --memory-dim N           composantes du vecteur lu (défaut : C)
//! --memory-layer N         la mémoire s'ajoute avant la couche N (défaut : 1)
//! --memory-storage dense|sparse|host
//!                          table dense (gradient de toute la table), creuse (seules les
//!                          lignes lues, sur la carte) ou en mémoire centrale
//! --memory-no-gate         sans la porte de contexte
//! --memory-no-conv         sans la convolution
//! --memory-std X           écart-type initial des lignes (défaut : 0,02, celui des embeddings)
//! --memory-lr-scale X      facteur du taux de la table (défaut : 5)
//! --resume                 reprendre au dernier point de reprise
//! --mixed                  produits matriciels en bf16 : sur les Tensor Cores, ou avec
//!                          --cpu, par vdpbf16ps (AVX-512 BF16)
//! --cpu [reference]        sur le processeur, sans la carte : le backend `Fast`,
//!                          ou avec `reference`, le backend de référence `Cpu<f32>`
//! ```
//!
//! Le dossier des données contient les fichiers `*train.bin` et `*val.bin`
//! écrits par `prepare`. S'il contient aussi `tokenizer.bpe`, les tokens
//! sont des sous-mots BPE ; sinon, des octets.
//!
//! À chaque évaluation, un point de reprise (`….checkpoint.safetensors`)
//! est écrit, avec l'état de l'optimiseur et le numéro de pas ; le modèle
//! lui-même n'est enregistré que si sa perte de validation s'est améliorée.
//!
//! Pour reprendre un entraînement interrompu, relancer la même commande
//! avec `--resume`. Les réglages qui déterminent la suite (architecture,
//! nombre de pas, lot, taux, décroissance, écrêtage, graine, précision)
//! doivent être ceux du point de reprise : toute différence est refusée.
//!
//! Si la perte ou le gradient cesse d'être un nombre fini, l'entraînement
//! s'arrête sans toucher au dernier point de reprise.
//!
//! La boucle elle-même est dans la bibliothèque (`smallem::training`) : ce
//! programme lit les options, ouvre le corpus et affiche ce qu'elle rapporte.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use smallem::backend::Backend;
use smallem::backend::cpu::Cpu;
use smallem::backend::fast::Fast;
use smallem::cli::Args;
use smallem::data::{Shards, bytes_per_token};
use smallem::flops;
use smallem::generation::generate;
use smallem::model::{Gpt, GptConfig};
use smallem::nn::{
    Architecture, AttentionKind, Balance, ExpertDims, LatentDims, MemoryDims, MemoryStorage,
    Module, MultiToken,
};
use smallem::sampling::Sampling;
use smallem::tokenizer::Bpe;
use smallem::training::{Evaluation, Event, Settings, Trainer, step_rng};

use rand::SeedableRng;
use rand::rngs::StdRng;

const GENERATED_TOKENS: usize = 400;
/// Nom du fichier de fusions cherché dans le dossier des données.
const TOKENIZER_FILE: &str = "tokenizer.bpe";

struct Options {
    config_name: String,
    /// Les réglages de la boucle d'entraînement.
    settings: Settings,
    out: Option<PathBuf>,
    data_dir: PathBuf,
    /// Les pièces des blocs. L'attention n'y est pas encore : les tailles
    /// de l'attention latente dépendent de la configuration.
    architecture: Architecture,
    /// `--attention mla`.
    latent_attention: bool,
    /// Les tailles de l'attention latente données sur la ligne de commande.
    latent: Option<usize>,
    query_latent: Option<usize>,
    rope_dim: Option<usize>,
    /// `--experts`, et ce qui s'écarte des tailles par défaut.
    experts: bool,
    experts_shared: Option<usize>,
    experts_routed: Option<usize>,
    experts_active: Option<usize>,
    experts_hidden: Option<usize>,
    balance: String,
    balance_rate: Option<f64>,
    dense_experts: bool,
    /// `--mtp` ou `--mtp-weights`.
    multi_token: Option<MultiToken>,
    layers: Option<usize>,
    embed: Option<usize>,
    heads: Option<usize>,
    /// `--budget`, en opérations.
    budget: Option<f64>,
    /// `--mup`, la largeur de base.
    mup: Option<usize>,
    context: Option<usize>,
    passes: usize,
    own_keys: Option<usize>,
    /// `--memory` et ses réglages.
    memory: Option<MemoryOptions>,
    resume: bool,
    mixed: bool,
    cpu: Option<Processor>,
}

/// La mémoire de n-grammes demandée : ce qui s'écarte des choix d'Engram.
#[derive(Clone, Copy, Debug, Default)]
struct MemoryOptions {
    rows: usize,
    heads: Option<usize>,
    max_order: Option<usize>,
    dim: Option<usize>,
    layer: Option<usize>,
    storage: Option<MemoryStorage>,
    no_gate: bool,
    no_conv: bool,
    table_std: Option<f64>,
    lr_scale: Option<f64>,
}

impl MemoryOptions {
    fn dims(&self, width: usize) -> MemoryDims {
        let engram = MemoryDims::engram(width, self.rows);
        MemoryDims {
            heads: self.heads.unwrap_or(engram.heads),
            max_order: self.max_order.unwrap_or(engram.max_order),
            dim: self.dim.unwrap_or(engram.dim),
            layer: self.layer.unwrap_or(engram.layer),
            storage: self.storage.unwrap_or(engram.storage),
            gate: !self.no_gate,
            conv: !self.no_conv,
            table_std: self.table_std.unwrap_or(engram.table_std),
            lr_scale: self.lr_scale.unwrap_or(engram.lr_scale),
            ..engram
        }
    }
}

/// Le backend du processeur choisi par `--cpu`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Processor {
    Fast,
    Reference,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        Self::parse_from(Args::from_env())
    }

    fn parse_from(mut args: Args) -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            config_name: "small".into(),
            settings: Settings::default(),
            out: None,
            data_dir: "../data".into(),
            architecture: Architecture::GPT2,
            latent_attention: false,
            latent: None,
            query_latent: None,
            rope_dim: None,
            experts: false,
            experts_shared: None,
            experts_routed: None,
            experts_active: None,
            experts_hidden: None,
            balance: "none".into(),
            balance_rate: None,
            dense_experts: false,
            multi_token: None,
            layers: None,
            embed: None,
            heads: None,
            budget: None,
            mup: None,
            context: None,
            passes: 1,
            own_keys: None,
            memory: None,
            resume: false,
            mixed: false,
            cpu: None,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--config" => options.config_name = args.value(&flag)?,
                "--steps" => options.settings.steps = args.value(&flag)?,
                "--batch" => options.settings.batch_size = args.value(&flag)?,
                "--accumulate" => options.settings.accumulate = args.value(&flag)?,
                "--lr" => options.settings.learning_rate = args.value(&flag)?,
                "--warmup" => options.settings.warmup_steps = args.value(&flag)?,
                "--optimizer" => options.settings.optimizer = args.value(&flag)?,
                "--weight-decay" => options.settings.weight_decay = args.value(&flag)?,
                "--clip" => options.settings.max_grad_norm = args.value(&flag)?,
                "--eval-every" => options.settings.eval_every = args.value(&flag)?,
                "--out" => options.out = Some(args.value(&flag)?),
                "--data" => options.data_dir = args.value(&flag)?,
                "--seed" => options.settings.seed = args.value(&flag)?,
                "--arch" => {
                    options.architecture = match args.value::<String>(&flag)?.as_str() {
                        "gpt2" => Architecture::GPT2,
                        "modern" => Architecture::MODERN,
                        other => {
                            return Err(
                                format!("--arch : `{other}` inconnu (gpt2 ou modern)").into()
                            );
                        }
                    }
                }
                "--norm" => options.architecture.norm = args.value(&flag)?,
                "--ffn" => options.architecture.feed_forward = args.value(&flag)?,
                "--positions" => options.architecture.positions = args.value(&flag)?,
                "--attention" => {
                    options.latent_attention = match args.value::<String>(&flag)?.as_str() {
                        "standard" => false,
                        "mla" => true,
                        other => {
                            return Err(format!(
                                "--attention : `{other}` inconnu (standard ou mla)"
                            )
                            .into());
                        }
                    }
                }
                "--latent" => options.latent = Some(args.value(&flag)?),
                "--query-latent" => options.query_latent = Some(args.value(&flag)?),
                "--rope-dim" => options.rope_dim = Some(args.value(&flag)?),
                "--experts" => options.experts = true,
                "--experts-shared" => options.experts_shared = Some(args.value(&flag)?),
                "--experts-routed" => options.experts_routed = Some(args.value(&flag)?),
                "--experts-active" => options.experts_active = Some(args.value(&flag)?),
                "--experts-hidden" => options.experts_hidden = Some(args.value(&flag)?),
                "--balance" => options.balance = args.value(&flag)?,
                "--balance-rate" => options.balance_rate = Some(args.value(&flag)?),
                "--dense-experts" => options.dense_experts = true,
                "--layers" => options.layers = Some(args.value(&flag)?),
                "--embed" => options.embed = Some(args.value(&flag)?),
                "--heads" => options.heads = Some(args.value(&flag)?),
                "--budget" => options.budget = Some(args.value(&flag)?),
                "--mup" => options.mup = Some(args.value(&flag)?),
                "--context" => options.context = Some(args.value(&flag)?),
                "--passes" => options.passes = args.value(&flag)?,
                "--own-keys" => options.own_keys = Some(args.value(&flag)?),
                "--mtp" => {
                    options.multi_token.get_or_insert(MultiToken::DEEPSEEK);
                }
                "--mtp-weights" => {
                    let text: String = args.value(&flag)?;
                    let weights = text
                        .split_once(',')
                        .and_then(|(early, late)| Some((early.parse().ok()?, late.parse().ok()?)));
                    let Some((early_weight, late_weight)) = weights else {
                        return Err(
                            format!("--mtp-weights : `{text}` illisible (X,Y attendu)").into()
                        );
                    };
                    options.multi_token = Some(MultiToken {
                        early_weight,
                        late_weight,
                    });
                }
                "--memory" => options.memory.get_or_insert_default().rows = args.value(&flag)?,
                "--memory-heads" => {
                    options.memory.get_or_insert_default().heads = Some(args.value(&flag)?)
                }
                "--memory-order" => {
                    options.memory.get_or_insert_default().max_order = Some(args.value(&flag)?)
                }
                "--memory-dim" => {
                    options.memory.get_or_insert_default().dim = Some(args.value(&flag)?)
                }
                "--memory-layer" => {
                    options.memory.get_or_insert_default().layer = Some(args.value(&flag)?)
                }
                "--memory-storage" => {
                    options.memory.get_or_insert_default().storage =
                        Some(match args.value::<String>(&flag)?.as_str() {
                            "dense" => MemoryStorage::Dense,
                            "sparse" => MemoryStorage::Sparse,
                            "host" => MemoryStorage::Host,
                            other => {
                                return Err(format!(
                                    "--memory-storage : `{other}` inconnu (dense, sparse ou host)"
                                )
                                .into());
                            }
                        })
                }
                "--memory-no-gate" => options.memory.get_or_insert_default().no_gate = true,
                "--memory-no-conv" => options.memory.get_or_insert_default().no_conv = true,
                "--memory-std" => {
                    options.memory.get_or_insert_default().table_std = Some(args.value(&flag)?)
                }
                "--memory-lr-scale" => {
                    options.memory.get_or_insert_default().lr_scale = Some(args.value(&flag)?)
                }
                "--resume" => options.resume = true,
                "--mixed" => options.mixed = true,
                "--cpu" if args.accept("reference") => options.cpu = Some(Processor::Reference),
                "--cpu" => options.cpu = Some(Processor::Fast),
                other => return Err(format!("option inconnue : {other}").into()),
            }
        }
        for (flag, value) in [
            ("--steps", options.settings.steps),
            ("--batch", options.settings.batch_size),
            ("--accumulate", options.settings.accumulate),
            ("--eval-every", options.settings.eval_every),
        ] {
            if value == 0 {
                return Err(format!("{flag} doit valoir au moins 1").into());
            }
        }
        let (decay, clip) = (
            options.settings.weight_decay,
            options.settings.max_grad_norm,
        );
        if !(decay.is_finite() && decay >= 0.0) {
            return Err(format!("--weight-decay : valeur {decay} refusée").into());
        }
        if !(clip.is_finite() && clip > 0.0) {
            return Err(format!("--clip : valeur {clip} refusée").into());
        }
        if !options.latent_attention {
            for (flag, value) in [
                ("--latent", options.latent),
                ("--query-latent", options.query_latent),
                ("--rope-dim", options.rope_dim),
            ] {
                if value.is_some() {
                    return Err(format!("{flag} n'a de sens qu'avec --attention mla").into());
                }
            }
        }
        if !options.experts {
            let given = [
                ("--experts-shared", options.experts_shared.is_some()),
                ("--experts-routed", options.experts_routed.is_some()),
                ("--experts-active", options.experts_active.is_some()),
                ("--experts-hidden", options.experts_hidden.is_some()),
                ("--balance", options.balance != "none"),
                ("--balance-rate", options.balance_rate.is_some()),
                ("--dense-experts", options.dense_experts),
            ];
            if let Some((flag, _)) = given.iter().find(|(_, given)| *given) {
                return Err(format!("{flag} n'a de sens qu'avec --experts").into());
            }
        }
        Ok(options)
    }

    /// L'équilibrage demandé, avec son réglage par défaut.
    fn balance(&self) -> Result<Balance, Box<dyn Error>> {
        match (self.balance.as_str(), self.balance_rate) {
            ("none", None) => Ok(Balance::None),
            ("none", Some(_)) => Err("--balance-rate demande --balance bias ou loss".into()),
            ("bias", rate) => Ok(Balance::Bias {
                rate: rate.unwrap_or(0.001),
            }),
            ("loss", weight) => Ok(Balance::Loss {
                weight: weight.unwrap_or(0.01),
            }),
            (other, _) => Err(format!("--balance : `{other}` inconnu (none, bias ou loss)").into()),
        }
    }

    fn config(&self, vocab: usize) -> Result<GptConfig, Box<dyn Error>> {
        let config = match self.config_name.as_str() {
            "small" => GptConfig::small(vocab),
            "large" => GptConfig::large(vocab),
            "xl" => GptConfig::xl(vocab),
            other => {
                return Err(
                    format!("configuration inconnue : {other} (small, large ou xl)").into(),
                );
            }
        };
        let mut architecture = self.architecture;
        if self.latent_attention {
            let defaults = LatentDims::for_model(config.embed, config.heads);
            architecture.attention = AttentionKind::Latent(LatentDims {
                latent: self.latent.unwrap_or(defaults.latent),
                query_latent: self.query_latent.unwrap_or(defaults.query_latent),
                rope: self.rope_dim.unwrap_or(defaults.rope),
            });
        }
        if self.experts {
            let defaults = ExpertDims::for_model(config.embed);
            architecture.experts = Some(ExpertDims {
                shared: self.experts_shared.unwrap_or(defaults.shared),
                routed: self.experts_routed.unwrap_or(defaults.routed),
                active: self.experts_active.unwrap_or(defaults.active),
                hidden: self.experts_hidden.unwrap_or(defaults.hidden),
                balance: self.balance()?,
            });
        }
        architecture.multi_token = self.multi_token;
        architecture.passes = self.passes;
        architecture.own_keys = self.own_keys;
        let mut config = config.with_architecture(architecture);
        if let Some(layers) = self.layers {
            config.layers = layers;
        }
        if let Some(embed) = self.embed {
            config.embed = embed;
        }
        if let Some(heads) = self.heads {
            config.heads = heads;
        }
        if let Some(context) = self.context {
            config.context = context;
        }
        if let Some(memory) = &self.memory {
            if memory.rows == 0 {
                return Err("les options --memory-… demandent --memory LIGNES".into());
            }
            config.architecture.memory = Some(memory.dims(config.embed));
        }
        config.validate()?;
        Ok(config)
    }
}

impl Options {
    /// Ce qu'une reprise doit retrouver à l'identique, en plus des réglages
    /// de la boucle : ce qui décide du modèle et de la précision du calcul.
    fn extra_settings(&self, config: &GptConfig) -> BTreeMap<String, String> {
        let architecture = config.architecture;
        [
            ("run.config", self.config_name.clone()),
            ("run.mixed", self.mixed.to_string()),
            ("run.norm", architecture.norm.name().to_string()),
            ("run.ffn", architecture.feed_forward.name().to_string()),
            ("run.positions", architecture.positions.name().to_string()),
            ("run.attention", describe_attention(architecture.attention)),
            ("run.experts", describe_experts(architecture.experts)),
            ("run.layers", config.layers.to_string()),
            ("run.embed", config.embed.to_string()),
            (
                "run.mup",
                self.mup.map_or("non".to_string(), |base| base.to_string()),
            ),
            ("run.heads", config.heads.to_string()),
            ("run.context", config.context.to_string()),
            ("run.passes", architecture.passes.to_string()),
            (
                "run.own_keys",
                architecture
                    .own_keys
                    .map_or("toutes".to_string(), |own| own.to_string()),
            ),
            (
                "run.mtp",
                architecture.multi_token.map_or("aucun".to_string(), |m| {
                    format!("{} puis {}", m.early_weight, m.late_weight)
                }),
            ),
            (
                "run.memory",
                architecture
                    .memory
                    .map_or("aucune".to_string(), |m| format!("{m:?}")),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }
}

/// L'attention et ses tailles, en un mot : `standard`, ou
/// `mla(latent 128, requêtes 384, rope 32)`.
fn describe_attention(attention: AttentionKind) -> String {
    match attention {
        AttentionKind::Standard => attention.name().to_string(),
        AttentionKind::Latent(dims) => format!(
            "{}(latent {}, requêtes {}, rope {})",
            attention.name(),
            dims.latent,
            dims.query_latent,
            dims.rope,
        ),
    }
}

/// Le mélange d'experts et ses réglages, en un mot : `aucun`, ou
/// `1 partagé + 8 aiguillés dont 2 actifs, largeur 456, équilibrage bias 0.001`.
fn describe_experts(experts: Option<ExpertDims>) -> String {
    let Some(dims) = experts else {
        return "aucun".to_string();
    };
    let balance = match dims.balance {
        Balance::None => dims.balance.name().to_string(),
        Balance::Bias { rate: value } | Balance::Loss { weight: value } => {
            format!("{} {value}", dims.balance.name())
        }
    };
    format!(
        "{} partagé + {} aiguillés dont {} actifs, largeur {}, équilibrage {balance}",
        dims.shared, dims.routed, dims.active, dims.hidden,
    )
}

/// La charge des experts, couche par couche : pour chacune, la part de
/// l'expert le plus chargé rapportée à la part moyenne (1 si tout est
/// équilibré), puis le nombre d'experts qui n'ont reçu aucun token.
fn describe_load(load: &[Vec<u64>]) -> Option<String> {
    if load.is_empty() {
        return None;
    }
    let mut dead = 0;
    let ratios: Vec<String> = load
        .iter()
        .map(|layer| {
            let total: u64 = layer.iter().sum();
            let largest = layer.iter().copied().max().unwrap_or(0);
            dead += layer.iter().filter(|&&count| count == 0).count();
            let mean = total as f64 / layer.len().max(1) as f64;
            format!(
                "{:.2}",
                if total == 0 {
                    0.0
                } else {
                    largest as f64 / mean
                }
            )
        })
        .collect();
    Some(format!(
        "charge max/moyenne par couche : {} ; experts sans token : {dead}",
        ratios.join(" ")
    ))
}

/// Le corpus tokenisé et de quoi le relire.
struct Corpus {
    train: Shards,
    val: Shards,
    tokenizer: Bpe,
    /// Octets de texte par token, mesuré sur la validation : il convertit
    /// une perte par token en perte par octet, la seule qui se compare
    /// d'un tokeniseur à l'autre.
    bytes_per_token: f64,
}

impl Corpus {
    fn open(dir: &Path) -> Result<Self, Box<dyn Error>> {
        let tokenizer_path = dir.join(TOKENIZER_FILE);
        let tokenizer = if tokenizer_path.exists() {
            Bpe::load(tokenizer_path)?
        } else {
            Bpe::bytes()
        };
        let train = Shards::open_dir(dir, "train.bin")?;
        let val = Shards::open_dir(dir, "val.bin")?;

        let bytes_per_token = bytes_per_token(&val, &tokenizer).map_err(|error| {
            format!(
                "{} : {error} ; le fichier `{TOKENIZER_FILE}` de ces données est-il dans ce \
                 dossier ?",
                dir.display(),
            )
        })?;
        Ok(Self {
            train,
            val,
            tokenizer,
            bytes_per_token,
        })
    }
}

fn run<B: Backend>(backend: &B, options: &Options, corpus: &Corpus) -> Result<(), Box<dyn Error>> {
    let config = options.config(corpus.tokenizer.vocab_size())?;
    let mut settings = options.settings.clone();
    let sequences = settings.batch_size * settings.accumulate;
    let step_flops = flops::training_flops(&config, sequences, config.context);
    if let Some(budget) = options.budget {
        let per_step =
            step_flops.ok_or("--budget : le calcul de cette architecture n'est pas compté")?;
        settings.steps = ((budget / per_step as f64).round() as usize).max(1);
    }
    let settings = &settings;
    let trainer = Trainer {
        backend,
        settings: settings.clone(),
        train: &corpus.train,
        validation: &corpus.val,
        model_path: options.out.clone().unwrap_or_else(|| {
            options
                .data_dir
                .join(format!("gpt-{}.safetensors", options.config_name))
        }),
        extra_settings: options.extra_settings(&config),
    };
    let tokens_per_step = settings.batch_size * settings.accumulate * config.context;
    let nats_per_byte = |loss: f64| loss / corpus.bytes_per_token;
    let mut model = Gpt::new(backend, config, &mut StdRng::seed_from_u64(settings.seed));
    model.set_dense_experts(options.dense_experts);
    if let Some(base) = options.mup {
        model.apply_mup(backend, base);
        println!("paramétrisation μP, largeur de base {base}");
    }

    println!(
        "architecture : normalisation {}, feed-forward {}, positions {}, attention {}",
        config.architecture.norm.name(),
        config.architecture.feed_forward.name(),
        config.architecture.positions.name(),
        describe_attention(config.architecture.attention),
    );
    if config.architecture.experts.is_some() {
        println!(
            "experts : {}",
            describe_experts(config.architecture.experts)
        );
    }
    if config.architecture.passes != 1 {
        println!(
            "{} couches traversées {} fois",
            config.layers, config.architecture.passes
        );
    }
    if let Some(own_keys) = config.architecture.own_keys {
        println!(
            "clés et valeurs calculées par les {own_keys} premières couches sur {}",
            config.layers
        );
    }
    if let Some(multi_token) = config.architecture.multi_token {
        println!(
            "prédiction multi-token : poids {} puis {}",
            multi_token.early_weight, multi_token.late_weight
        );
    }
    // Les paramètres du réseau, ceux qui calculent ; la table de la
    // mémoire est comptée à part.
    let mut parameters = model.parameter_count(backend);
    if let Some(memory) = model.memory() {
        let dims = memory.dims();
        if dims.storage != MemoryStorage::Host {
            parameters -= memory.table_parameters();
        }
        println!(
            "mémoire de n-grammes : {} paramètres de table, {} tables de {} lignes et plus, \
             ordres 2 à {}, d_mém {}, avant la couche {}{}{}, table {:?}",
            memory.table_parameters(),
            memory.hash().tables(),
            dims.rows,
            dims.max_order,
            dims.dim,
            dims.layer,
            if dims.gate { ", porte" } else { "" },
            if dims.conv { ", convolution" } else { "" },
            dims.storage,
        );
    }
    println!(
        "{} paramètres, vocabulaire de {} tokens ({:.3} octets par token)",
        parameters, config.vocab, corpus.bytes_per_token,
    );
    println!(
        "{} tokens d'entraînement dans {} fichier(s), {} de validation",
        corpus.train.len(),
        corpus.train.shard_count(),
        corpus.val.len(),
    );
    println!(
        "{} pas de {} × {} × {} tokens, soit {:.2} passage(s) sur le corpus",
        settings.steps,
        settings.accumulate,
        settings.batch_size,
        config.context,
        (settings.steps * tokens_per_step) as f64 / corpus.train.len() as f64,
    );
    if let Some(per_step) = step_flops {
        println!(
            "{:.3e} opérations par pas, {:.3e} en tout",
            per_step as f64,
            (per_step * settings.steps as u64) as f64,
        );
    }
    println!(
        "optimiseur {}, décroissance des poids {}, gradient écrêté à {}",
        settings.optimizer.name(),
        settings.weight_decay,
        settings.max_grad_norm,
    );

    let summary = trainer.run(&mut model, options.resume, |event| match event {
        Event::Started { val_loss } => println!(
            "perte de validation initiale : {val_loss:.4} (uniforme : {:.4})",
            (config.vocab as f64).ln(),
        ),
        Event::Resumed {
            step,
            best_val_loss,
            unverified,
        } => {
            for key in unverified {
                println!("attention : `{key}` absent du point de reprise, non vérifié");
            }
            println!(
                "reprise de {} au pas {step}, meilleure perte de validation {best_val_loss:.4}",
                trainer.checkpoint_path().display()
            );
        }
        Event::Evaluated(Evaluation {
            step,
            train_loss,
            val_loss,
            learning_rate,
            grad_norm_max,
            steps,
            seconds,
            improved,
            expert_load,
        }) => {
            let memory =
                smallem_cuda::gpu_memory_mib().map_or(String::new(), |mib| format!("  {mib} Mio"));
            println!(
                "pas {step:>6}  train {train_loss:.4}  val {val_loss:.4} ({:.4} nats/octet)  \
                 lr {learning_rate:.2e}  norme max {grad_norm_max:>5.2}  \
                 {:>7.0} tokens/s{memory}{}",
                nats_per_byte(val_loss),
                (steps * tokens_per_step) as f64 / seconds,
                if improved { "  enregistré" } else { "" },
            );
            if let Some(load) = describe_load(&expert_load) {
                println!("            {load}");
            }
        }
    })?;
    println!(
        "meilleure perte de validation : {:.4} par token, {:.4} nats par octet ; modèle dans {}",
        summary.best_val_loss,
        nats_per_byte(summary.best_val_loss),
        trainer.model_path.display()
    );
    println!(
        "perte de validation finale, sur {} lots : {:.4} par token, {:.4} nats par octet",
        settings.final_eval_batches,
        summary.final_val_loss,
        nats_per_byte(summary.final_val_loss),
    );

    // Le début du premier document de validation sert d'amorce.
    // Le premier fichier de validation qui n'est pas vide : `Corpus::open`
    // garantit qu'il y en a un.
    let shard = (0..corpus.val.shard_count())
        .find(|&shard| corpus.val.shard_len(shard) > 0)
        .expect("validation non vide");
    let prompt: Vec<u32> = corpus
        .val
        .read(shard, 0, 32.min(corpus.val.shard_len(shard)))?
        .iter()
        .map(|&t| u32::from(t))
        .collect();
    let generated = generate(
        backend,
        &model,
        &prompt,
        GENERATED_TOKENS,
        &Sampling::default(),
        &mut step_rng(settings.seed, settings.steps),
    );
    let generated: Vec<u16> = generated.iter().map(|&t| t as u16).collect();
    println!(
        "\n--- texte généré ---\n{}",
        corpus.tokenizer.decode(&generated)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let corpus = Corpus::open(&options.data_dir)?;
    if let Some(processor) = options.cpu {
        match processor {
            Processor::Fast => run(
                &Fast::new().with_mixed_precision(options.mixed),
                &options,
                &corpus,
            ),
            Processor::Reference => run(&Cpu::<f32>::new(), &options, &corpus),
        }
    } else {
        let gpu = smallem_cuda::gpu(options.mixed)?;
        run(&gpu, &options, &corpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallem::training::check_resume;

    fn options(args: &[&str]) -> Result<Options, Box<dyn Error>> {
        Options::parse_from(Args::new(
            args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>(),
        ))
    }

    /// Tout ce qu'un point de reprise retient d'une ligne de commande.
    fn recorded(args: &[&str]) -> BTreeMap<String, String> {
        let options = options(args).unwrap();
        let mut settings = options.settings.metadata();
        settings.extend(options.extra_settings(&options.config(8192).unwrap()));
        settings
    }

    #[test]
    fn options_are_read_from_the_command_line() {
        let options =
            options(&["--config", "xl", "--steps", "10", "--mixed", "--lr", "6e-4"]).unwrap();
        assert_eq!(options.config_name, "xl");
        assert_eq!(options.settings.steps, 10);
        assert_eq!(options.settings.learning_rate, 6e-4);
        assert!(options.mixed && !options.resume && options.cpu.is_none());
        // Les autres gardent leur valeur par défaut.
        let settings = &options.settings;
        assert_eq!((settings.batch_size, settings.accumulate), (32, 1));
        assert_eq!((settings.weight_decay, settings.max_grad_norm), (0.1, 1.0));
    }

    #[test]
    fn decay_and_clipping_are_options() {
        let options = options(&["--weight-decay", "0", "--clip", "0.5"]).unwrap();
        assert_eq!(options.settings.weight_decay, 0.0);
        assert_eq!(options.settings.max_grad_norm, 0.5);
        for refused in [
            &["--weight-decay", "-1"][..],
            &["--clip", "0"],
            &["--clip", "nan"],
            &["--clip", "inf"],
        ] {
            let error = self::options(refused).err().expect(refused[1]).to_string();
            assert!(error.contains(refused[0]), "{error}");
        }
    }

    #[test]
    fn zero_is_rejected_where_it_would_divide_or_do_nothing() {
        for flag in ["--steps", "--batch", "--accumulate", "--eval-every"] {
            let error = options(&[flag, "0"]).err().expect(flag).to_string();
            assert!(error.contains(flag), "{error}");
        }
    }

    #[test]
    fn unknown_and_incomplete_options_are_rejected() {
        assert!(options(&["--nope"]).is_err());
        assert!(options(&["--steps"]).is_err());
        assert!(options(&["--steps", "beaucoup"]).is_err());
    }

    #[test]
    fn resume_accepts_the_same_command_line() {
        let settings = recorded(&["--config", "large", "--seed", "7"]);
        assert_eq!(check_resume(&settings, &settings), Ok(Vec::new()));
    }

    #[test]
    fn resume_rejects_each_changed_option() {
        let saved = recorded(&[]);
        for change in [
            &["--config", "large"][..],
            &["--steps", "9"],
            &["--batch", "8"],
            &["--accumulate", "2"],
            &["--lr", "5e-4"],
            &["--warmup", "3"],
            &["--weight-decay", "0.2"],
            &["--optimizer", "muon"],
            &["--clip", "2"],
            &["--seed", "1"],
            &["--mixed"],
            &["--arch", "modern"],
            &["--norm", "rms"],
            &["--ffn", "swiglu"],
            &["--positions", "rope"],
            &["--positions", "rope", "--attention", "mla"],
        ] {
            let error = check_resume(&saved, &recorded(change)).expect_err(change[0]);
            assert!(error.contains("enregistré"), "{error}");
        }
        // Une taille de l'attention latente compte autant que son choix.
        let latent = recorded(&["--arch", "modern", "--attention", "mla"]);
        let narrower = recorded(&["--arch", "modern", "--attention", "mla", "--latent", "64"]);
        assert!(check_resume(&latent, &narrower).is_err());
        // Ce qui ne change pas la suite du calcul reste libre.
        let current = recorded(&["--eval-every", "7", "--out", "ailleurs"]);
        assert_eq!(check_resume(&saved, &current), Ok(Vec::new()));
    }

    #[test]
    fn architecture_is_chosen_whole_or_piece_by_piece() {
        assert_eq!(options(&[]).unwrap().architecture, Architecture::GPT2);
        assert_eq!(
            options(&["--arch", "modern"]).unwrap().architecture,
            Architecture::MODERN
        );
        let rope_only = options(&["--positions", "rope"]).unwrap();
        assert_eq!(rope_only.architecture.positions.name(), "rope");
        assert_eq!(rope_only.architecture.norm.name(), "layer");
        assert_eq!(
            rope_only.config(8192).unwrap().architecture,
            rope_only.architecture
        );
        // Une pièce peut revenir en arrière après `--arch modern`.
        let mixed = options(&["--arch", "modern", "--ffn", "gelu"]).unwrap();
        assert_eq!(mixed.architecture.feed_forward.name(), "gelu");
        assert_eq!(mixed.architecture.norm.name(), "rms");
        assert!(options(&["--arch", "futur"]).is_err());
        assert!(options(&["--norm", "batch"]).is_err());
    }

    #[test]
    fn latent_attention_takes_its_sizes_from_the_configuration() {
        let config = |args: &[&str]| options(args).unwrap().config(8192);
        let attention = |args: &[&str]| config(args).unwrap().architecture.attention;
        assert_eq!(attention(&["--arch", "modern"]), AttentionKind::Standard);
        // `xl` : C = 512 et 8 têtes de 64.
        let mla = ["--config", "xl", "--arch", "modern", "--attention", "mla"];
        assert_eq!(
            attention(&mla),
            AttentionKind::Latent(LatentDims {
                latent: 128,
                query_latent: 384,
                rope: 32,
            })
        );
        let sized = [&mla[..], &["--latent", "64", "--rope-dim", "16"]].concat();
        assert_eq!(
            attention(&sized),
            AttentionKind::Latent(LatentDims {
                latent: 64,
                query_latent: 384,
                rope: 16,
            })
        );
        assert_eq!(
            describe_attention(attention(&mla)),
            "mla(latent 128, requêtes 384, rope 32)"
        );
        // Sans RoPE, pas d'attention latente ; et une taille sans elle
        // n'a pas de sens.
        let error = config(&["--attention", "mla"]).unwrap_err().to_string();
        assert!(error.contains("rope"), "{error}");
        assert!(config(&[&mla[..], &["--rope-dim", "3"]].concat()).is_err());
        assert!(options(&["--latent", "64"]).is_err());
        assert!(options(&["--attention", "flash"]).is_err());
    }

    #[test]
    fn experts_take_their_sizes_from_the_configuration() {
        let config = |args: &[&str]| options(args).unwrap().config(8192);
        let experts = |args: &[&str]| config(args).unwrap().architecture.experts;
        assert_eq!(experts(&["--arch", "modern"]), None);
        let moe = ["--config", "xl", "--arch", "modern", "--experts"];
        assert_eq!(experts(&moe), Some(ExpertDims::for_model(512)));
        let tuned = [
            &moe[..],
            &["--experts-routed", "16", "--experts-active", "4"],
            &["--balance", "bias"],
        ]
        .concat();
        let dims = experts(&tuned).unwrap();
        assert_eq!((dims.routed, dims.active), (16, 4));
        assert_eq!(dims.balance, Balance::Bias { rate: 0.001 });
        let loss = [&moe[..], &["--balance", "loss", "--balance-rate", "0.1"]].concat();
        assert_eq!(
            experts(&loss).unwrap().balance,
            Balance::Loss { weight: 0.1 }
        );
        assert_eq!(
            describe_experts(experts(&tuned)),
            "1 partagé + 16 aiguillés dont 4 actifs, largeur 456, équilibrage bias 0.001"
        );
        // Chaque expert est un SwiGLU ; et les réglages n'ont de sens
        // qu'avec des experts.
        assert!(config(&["--experts"]).is_err());
        assert!(config(&[&moe[..], &["--balance", "hasard"]].concat()).is_err());
        assert!(config(&[&moe[..], &["--balance-rate", "0.1"]].concat()).is_err());
        assert!(config(&[&moe[..], &["--experts-active", "9"]].concat()).is_err());
        assert!(options(&["--balance", "bias"]).is_err());
        assert!(options(&["--dense-experts"]).is_err());
        // L'équilibrage décide de la suite du calcul : une reprise le vérifie.
        let plain = recorded(&["--arch", "modern", "--experts"]);
        let biased = recorded(&["--arch", "modern", "--experts", "--balance", "bias"]);
        assert!(check_resume(&plain, &biased).is_err());
    }

    #[test]
    fn expert_load_is_summarized_by_layer() {
        assert_eq!(describe_load(&[]), None);
        // Première couche équilibrée ; dans la seconde, un expert reçoit la
        // moitié des tokens et un autre aucun.
        let load = [vec![10, 10, 10, 10], vec![20, 10, 10, 0]];
        assert_eq!(
            describe_load(&load).unwrap(),
            "charge max/moyenne par couche : 1.00 2.00 ; experts sans token : 1"
        );
    }

    #[test]
    fn depth_passes_and_shared_keys_are_options() {
        let config = |args: &[&str]| options(args).unwrap().config(8192).unwrap();
        let plain = config(&["--config", "xl", "--arch", "modern"]);
        assert_eq!((plain.layers, plain.architecture.passes), (8, 1));
        assert_eq!(plain.architecture.own_keys, None);

        let deep = config(&["--config", "xl", "--arch", "modern", "--layers", "16"]);
        assert_eq!(deep.layers, 16);
        let long = config(&["--config", "xl", "--arch", "modern", "--context", "4096"]);
        assert_eq!(long.context, 4096);
        let looped = config(&["--config", "xl", "--arch", "modern", "--passes", "2"]);
        assert_eq!((looped.layers, looped.architecture.passes), (8, 2));
        let shared = config(&["--config", "xl", "--arch", "modern", "--own-keys", "4"]);
        assert_eq!(shared.architecture.own_keys, Some(4));

        // Plus de couches à clés que de couches, ou aucune traversée.
        let refused = |args: &[&str]| options(args).unwrap().config(8192).is_err();
        assert!(refused(&[
            "--config",
            "xl",
            "--arch",
            "modern",
            "--own-keys",
            "9"
        ]));
        assert!(refused(&[
            "--config", "xl", "--arch", "modern", "--passes", "0"
        ]));
        assert!(refused(&[
            "--attention",
            "mla",
            "--arch",
            "modern",
            "--own-keys",
            "2"
        ]));

        for change in [
            &["--layers", "3"][..],
            &["--context", "100"],
            &["--passes", "2"],
            &["--own-keys", "1"],
        ] {
            let error = check_resume(&recorded(&[]), &recorded(change)).unwrap_err();
            assert!(error.contains("run."), "{error}");
        }
    }

    #[test]
    fn multi_token_prediction_is_an_option() {
        let multi_token = |args: &[&str]| {
            let config = options(args).unwrap().config(8192).unwrap();
            config.architecture.multi_token
        };
        assert_eq!(multi_token(&["--arch", "modern"]), None);
        assert_eq!(
            multi_token(&["--arch", "modern", "--mtp"]),
            Some(MultiToken::DEEPSEEK)
        );
        let custom = Some(MultiToken {
            early_weight: 0.5,
            late_weight: 0.0,
        });
        assert_eq!(multi_token(&["--mtp-weights", "0.5,0"]), custom);
        assert_eq!(multi_token(&["--mtp-weights", "0.5,0", "--mtp"]), custom);
        assert!(options(&["--mtp-weights", "0.5"]).is_err());
        assert!(options(&["--mtp-weights", "a,b"]).is_err());
        let negative = options(&["--mtp-weights", "-1,0"]).unwrap();
        assert!(negative.config(8192).is_err());
        // Le module change la suite du calcul : une reprise le vérifie.
        let error = check_resume(&recorded(&[]), &recorded(&["--mtp"])).unwrap_err();
        assert!(error.contains("run.mtp"), "{error}");
    }
}
