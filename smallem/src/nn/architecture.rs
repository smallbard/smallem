//! Les choix d'architecture d'un bloc : quelle normalisation, quel
//! feed-forward, quelle attention, quelle façon de connaître les positions.

use super::MemoryDims;

/// Les choix d'architecture d'un bloc. Chaque variante remplace une pièce du
/// GPT-2 d'origine par celle qu'utilisent les modèles récents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Architecture {
    pub norm: NormKind,
    pub feed_forward: FeedForwardKind,
    pub positions: PositionKind,
    pub attention: AttentionKind,
    /// Avec `Some`, le feed-forward de chaque bloc est un mélange
    /// d'experts.
    pub experts: Option<ExpertDims>,
    /// Avec `Some`, un module d'appoint prédit pendant l'entraînement le
    /// token d'après le suivant.
    pub multi_token: Option<MultiToken>,
    /// Nombre de traversées de la pile de blocs. Avec 2, les états cachés
    /// sortis du dernier bloc repassent par le premier : le modèle est deux
    /// fois plus profond sans un paramètre de plus (Nanbeige 4.2).
    pub passes: usize,
    /// Avec `Some(n)`, seuls les `n` premiers blocs calculent des clés et
    /// des valeurs. Les suivants gardent leurs propres requêtes, mais les
    /// comparent aux clés du bloc `n`, dont ils lisent aussi les valeurs :
    /// le cache de génération n'a que `n` étages (Gemma 4).
    pub own_keys: Option<usize>,
    /// Avec `Some`, une mémoire de n-grammes s'ajoute au flux avant l'un des
    /// blocs (Engram).
    pub memory: Option<MemoryDims>,
}

impl Architecture {
    /// L'architecture de GPT-2 : LayerNorm, feed-forward à GELU, positions
    /// apprises.
    pub const GPT2: Self = Self {
        norm: NormKind::Layer,
        feed_forward: FeedForwardKind::Gelu,
        positions: PositionKind::Learned,
        attention: AttentionKind::Standard,
        experts: None,
        multi_token: None,
        passes: 1,
        own_keys: None,
        memory: None,
    };

    /// Le socle des modèles récents : RMSNorm, SwiGLU et RoPE.
    pub const MODERN: Self = Self {
        norm: NormKind::Rms,
        feed_forward: FeedForwardKind::SwiGlu,
        positions: PositionKind::Rotary,
        attention: AttentionKind::Standard,
        experts: None,
        multi_token: None,
        passes: 1,
        own_keys: None,
        memory: None,
    };

    /// La même architecture, dont la pile de blocs est traversée `passes`
    /// fois.
    pub const fn with_passes(self, passes: usize) -> Self {
        Self { passes, ..self }
    }

    /// La même architecture, où seuls les `own_keys` premiers blocs
    /// calculent des clés et des valeurs.
    pub const fn with_own_keys(self, own_keys: usize) -> Self {
        Self {
            own_keys: Some(own_keys),
            ..self
        }
    }

    /// Vrai si le bloc `layer` emprunte ses clés et ses valeurs à un bloc
    /// qui le précède.
    pub fn borrows_keys(&self, layer: usize) -> bool {
        self.own_keys.is_some_and(|own| layer >= own)
    }

    /// La même architecture, avec une autre attention.
    pub const fn with_attention(self, attention: AttentionKind) -> Self {
        Self { attention, ..self }
    }

    /// La même architecture, avec le module de prédiction multi-token.
    pub const fn with_multi_token(self, multi_token: MultiToken) -> Self {
        Self {
            multi_token: Some(multi_token),
            ..self
        }
    }

    /// La même architecture, avec une mémoire de n-grammes.
    pub const fn with_memory(self, memory: MemoryDims) -> Self {
        Self {
            memory: Some(memory),
            ..self
        }
    }

    /// La même architecture, avec un mélange d'experts pour feed-forward.
    pub const fn with_experts(self, experts: ExpertDims) -> Self {
        Self {
            experts: Some(experts),
            ..self
        }
    }
}

/// Le poids de la prédiction multi-token dans la perte d'entraînement :
/// `perte = perte principale + poids × perte du module`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiToken {
    /// Poids pendant les deux premiers tiers de l'entraînement.
    pub early_weight: f64,
    /// Poids pendant le dernier tiers.
    pub late_weight: f64,
}

impl MultiToken {
    /// Les poids de DeepSeek-V3 : 0,3 puis 0,1.
    pub const DEEPSEEK: Self = Self {
        early_weight: 0.3,
        late_weight: 0.1,
    };

    /// Part de l'entraînement après laquelle le poids baisse. DeepSeek-V3
    /// le baisse après 10 des 14,8 milliers de milliards de tokens de son
    /// entraînement.
    pub const SWITCH: f64 = 2.0 / 3.0;

    /// Le poids quand la part `progress` de l'entraînement est faite.
    pub fn weight_at(&self, progress: f64) -> f64 {
        if progress < Self::SWITCH {
            self.early_weight
        } else {
            self.late_weight
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let valid = |weight: f64| weight.is_finite() && weight >= 0.0;
        if valid(self.early_weight) && valid(self.late_weight) {
            Ok(())
        } else {
            Err(format!(
                "prédiction multi-token : poids refusés dans {self:?}"
            ))
        }
    }
}

/// Les tailles d'un mélange d'experts, et sa façon d'équilibrer leur charge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertDims {
    /// Experts traversés par tous les tokens, sans aiguillage.
    pub shared: usize,
    /// Experts entre lesquels l'aiguilleur choisit.
    pub routed: usize,
    /// Experts aiguillés activés pour chaque token.
    pub active: usize,
    /// Largeur cachée de chaque expert.
    pub hidden: usize,
    pub balance: Balance,
}

impl ExpertDims {
    /// Les tailles par défaut d'un modèle de dimension `embed` : un expert
    /// partagé et huit aiguillés dont deux actifs. Chacun est trois fois
    /// plus étroit que le feed-forward qu'ils remplacent : un token
    /// traverse trois experts, donc autant de paramètres qu'avant.
    pub fn for_model(embed: usize) -> Self {
        let (shared, active) = (1, 2);
        let reference = super::swiglu_hidden(embed);
        Self {
            shared,
            routed: 8,
            active,
            hidden: (reference / (shared + active)).next_multiple_of(8),
            balance: Balance::None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.routed == 0 || self.active == 0 || self.hidden == 0 {
            return Err(format!("mélange d'experts : tailles nulles dans {self:?}"));
        }
        if self.active > self.routed {
            return Err(format!(
                "mélange d'experts : {} experts actifs parmi {} seulement",
                self.active, self.routed
            ));
        }
        match self.balance {
            Balance::Bias { rate } | Balance::Loss { weight: rate }
                if !(rate.is_finite() && rate > 0.0) =>
            {
                Err(format!(
                    "mélange d'experts : réglage d'équilibrage {rate} refusé"
                ))
            }
            _ => Ok(()),
        }
    }
}

/// La façon d'empêcher l'aiguilleur d'envoyer tous les tokens aux mêmes
/// experts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Balance {
    /// Rien : l'aiguilleur fait ce qu'il veut.
    None,
    /// Un biais par expert, ajouté à son affinité pour le seul choix des
    /// experts actifs. Après chaque pas, il baisse de `rate` pour un expert
    /// surchargé et monte d'autant pour un expert sous-chargé. C'est la
    /// méthode de DeepSeek-V3.
    Bias { rate: f64 },
    /// Un terme ajouté à la perte, de poids `weight`, qui pénalise le
    /// déséquilibre. C'est la méthode classique.
    Loss { weight: f64 },
}

impl Balance {
    /// Les noms des variantes, sur la ligne de commande et dans un fichier
    /// de poids.
    pub const NONE: &str = "none";
    pub const BIAS: &str = "bias";
    pub const LOSS: &str = "loss";
    pub const NAMES: [&str; 3] = [Self::NONE, Self::BIAS, Self::LOSS];

    pub fn name(self) -> &'static str {
        match self {
            Self::None => Self::NONE,
            Self::Bias { .. } => Self::BIAS,
            Self::Loss { .. } => Self::LOSS,
        }
    }
}

/// L'attention de chaque bloc.
///
/// Ce n'est pas un simple nom comme les autres choix : l'attention latente
/// a ses propres tailles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Une clé et une valeur complètes par tête et par position.
    Standard,
    /// MLA : clés et valeurs recalculées à partir d'un petit vecteur latent.
    Latent(LatentDims),
}

impl AttentionKind {
    /// Les noms des variantes, sur la ligne de commande et dans un fichier
    /// de poids.
    pub const STANDARD: &str = "standard";
    pub const LATENT: &str = "mla";
    pub const NAMES: [&str; 2] = [Self::STANDARD, Self::LATENT];

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => Self::STANDARD,
            Self::Latent(_) => Self::LATENT,
        }
    }
}

/// Les tailles propres à l'attention latente.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentDims {
    /// `d_c` : taille du vecteur latent dont sortent les clés et les
    /// valeurs de toutes les têtes. C'est lui que garde le cache.
    pub latent: usize,
    /// Taille du vecteur latent dont sortent les requêtes.
    pub query_latent: usize,
    /// Taille de la partie des requêtes et des clés qui porte la rotation
    /// de RoPE.
    pub rope: usize,
}

impl LatentDims {
    /// Les tailles par défaut d'un modèle de dimension `embed` à `heads`
    /// têtes : un latent de `C/4`, celui des requêtes trois fois plus grand
    /// (le rapport de DeepSeek-V3), et une partie tournée d'une demi-tête.
    pub fn for_model(embed: usize, heads: usize) -> Self {
        let head_dim = embed / heads;
        Self {
            latent: embed / 4,
            query_latent: 3 * embed / 4,
            // Une rotation agit sur des paires : la taille doit être paire.
            rope: (head_dim / 2).next_multiple_of(2),
        }
    }
}

/// Déclare une énumération de choix d'architecture, avec le nom sous lequel
/// chaque variante s'écrit sur la ligne de commande et dans un fichier de
/// poids.
macro_rules! named_choices {
    ($(#[$doc:meta])* $name:ident { $($(#[$variant_doc:meta])* $variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $($(#[$variant_doc])* $variant),+
        }

        impl $name {
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }

        impl std::str::FromStr for $name {
            type Err = String;

            fn from_str(text: &str) -> Result<Self, String> {
                match text {
                    $($text => Ok(Self::$variant),)+
                    other => Err(format!(
                        "`{other}` inconnu, choix possibles : {}",
                        [$($text),+].join(", ")
                    )),
                }
            }
        }
    };
}

pub(crate) use named_choices;

named_choices! {
    /// La normalisation à l'entrée de chaque sous-couche.
    NormKind {
        /// LayerNorm : moyenne retirée, gain et biais.
        Layer => "layer",
        /// RMSNorm : moyenne conservée, gain seul.
        Rms => "rms",
    }
}

named_choices! {
    /// Le petit réseau de chaque bloc.
    FeedForwardKind {
        /// Deux couches et GELU, comme GPT-2.
        Gelu => "gelu",
        /// Trois couches dont une porte : SwiGLU.
        SwiGlu => "swiglu",
    }
}

named_choices! {
    /// La façon dont le modèle connaît l'ordre des tokens.
    PositionKind {
        /// Un vecteur appris par position, ajouté à l'entrée.
        Learned => "learned",
        /// RoPE : une rotation des requêtes et des clés dans l'attention.
        Rotary => "rope",
    }
}
