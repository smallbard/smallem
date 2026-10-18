//! Les optimiseurs : comment un gradient devient une modification des
//! paramètres.
//!
//! Un pas d'optimiseur modifie les paramètres sur place. La tape qui a
//! servi à calculer les gradients doit donc être libérée avant, sans quoi
//! elle référence encore les tenseurs à modifier.

use std::collections::BTreeMap;

use crate::autograd::Gradients;
use crate::backend::{AdamWStep, Backend, Float, Primitives};
use crate::nn::Parameter;

/// L'état d'un optimiseur, sous la forme qu'un point de reprise sait
/// ranger : des compteurs, et des séries de tenseurs.
///
/// Chaque série a un préfixe (`adamw.m.`) et un tenseur par paramètre du
/// modèle, dans l'ordre des paramètres ; `None` pour un paramètre auquel
/// l'optimiseur n'a encore rien associé.
pub struct OptimizerState<T> {
    pub counters: BTreeMap<String, String>,
    pub slots: Vec<(String, Vec<Option<T>>)>,
}

impl<T> Default for OptimizerState<T> {
    fn default() -> Self {
        Self {
            counters: BTreeMap::new(),
            slots: Vec::new(),
        }
    }
}

/// Ce que le code d'entraînement attend d'un optimiseur : savoir faire un
/// pas. Tout le reste (taux d'apprentissage, moments, compteurs) est l'état
/// interne de chaque implémentation.
pub trait Optimizer<B: Backend> {
    /// Met à jour, sur place, tous les paramètres qui ont un gradient sur la
    /// tape `tape_id`. La liste doit être la même, dans le même ordre, à
    /// chaque appel : un optimiseur peut ranger ses données par position.
    ///
    /// `grad_scale` multiplie chaque gradient avant usage : 1 en temps
    /// normal, moins quand la norme du gradient est écrêtée.
    fn step(
        &mut self,
        backend: &B,
        parameters: Vec<&mut Parameter<B>>,
        tape_id: u64,
        grads: &Gradients<B>,
        grad_scale: f64,
    ) {
        let grads: Vec<Option<&B::Tensor>> = parameters
            .iter()
            .map(|parameter| parameter.grad(tape_id, grads))
            .collect();
        self.apply(backend, parameters, &grads, grad_scale);
    }

    /// Le pas lui-même, avec un gradient par paramètre, dans le même ordre
    /// (`None` : paramètre laissé tel quel). [`Optimizer::step`] va chercher
    /// ces gradients sur une tape ; les donner directement permet d'utiliser
    /// des gradients venus d'ailleurs, par exemple accumulés sur plusieurs
    /// lots.
    fn apply(
        &mut self,
        backend: &B,
        parameters: Vec<&mut Parameter<B>>,
        grads: &[Option<&B::Tensor>],
        grad_scale: f64,
    );

    fn learning_rate(&self) -> f64;

    /// Change le taux d'apprentissage des prochains pas : c'est ainsi
    /// qu'un planificateur le fait varier pendant l'entraînement.
    fn set_learning_rate(&mut self, learning_rate: f64);

    /// Indique où en est l'entraînement, de 0 au premier pas à 1 à la fin.
    /// Sans effet par défaut : seul un optimiseur dont un réglage évolue au
    /// fil de l'entraînement s'en sert.
    fn set_progress(&mut self, _progress: f64) {}

    /// Ce qu'il faut enregistrer pour reprendre un entraînement là où il en
    /// était. Rien par défaut : un optimiseur sans mémoire n'a pas d'état.
    fn state(&self) -> OptimizerState<&B::Tensor> {
        OptimizerState::default()
    }

    /// Les noms des compteurs et les préfixes des séries de
    /// [`Optimizer::state`] : ce qu'un point de reprise doit aller chercher.
    fn state_layout(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        (Vec::new(), Vec::new())
    }

    /// Remet l'optimiseur dans un état enregistré.
    fn restore_state(&mut self, _state: OptimizerState<B::Tensor>) -> Result<(), String> {
        Ok(())
    }
}

/// Taux d'apprentissage au pas `step` (compté à partir de 0) d'un
/// entraînement de `total_steps` pas :
///
/// - montée linéaire de 0 à `peak` pendant `warmup_steps` pas. Au début, les
///   moments d'Adam ne sont pas encore fiables et les poids sont aléatoires :
///   de grands pas feraient diverger le modèle ;
/// - puis descente en demi-cosinus de `peak` à `floor`. De grands pas
///   dégrossissent, de petits pas affinent.
pub fn cosine_schedule(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    peak: f64,
    floor: f64,
) -> f64 {
    if step < warmup_steps {
        return peak * (step + 1) as f64 / warmup_steps as f64;
    }
    if step >= total_steps {
        return floor;
    }
    let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
    // Le cosinus va de 1 à -1 : on le ramène entre 1 et 0.
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    floor + (peak - floor) * cosine
}

/// Norme du gradient de tous les paramètres mis bout à bout : `√(Σ g²)`.
/// Elle mesure la taille du pas que le gradient réclame.
///
/// Le calcul se fait sur le GPU ; seul le résultat est lu, ce qui force
/// une synchronisation.
pub fn gradient_norm<B: Backend>(
    backend: &B,
    parameters: &[&Parameter<B>],
    tape_id: u64,
    grads: &Gradients<B>,
) -> f64 {
    let grads: Vec<Option<&B::Tensor>> = parameters
        .iter()
        .map(|parameter| parameter.grad(tape_id, grads))
        .collect();
    tensors_norm(backend, &grads)
}

/// Norme de plusieurs tenseurs mis bout à bout : `√(Σ g²)`.
pub fn tensors_norm<B: Backend>(backend: &B, tensors: &[Option<&B::Tensor>]) -> f64 {
    let mut total: Option<B::Tensor> = None;
    for tensor in tensors.iter().flatten() {
        let squares = backend.sum_all(&backend.mul(tensor, tensor));
        total = Some(match total {
            Some(total) => backend.add(&total, &squares),
            None => squares,
        });
    }
    match total {
        Some(total) => backend.item(&total).to_f64().sqrt(),
        None => 0.0,
    }
}

/// Stochastic Gradient Descent (Descente de gradient ordinaire) : chaque paramètre recule d'un pas
/// proportionnel à son gradient.
pub struct Sgd {
    pub learning_rate: f64,
}

impl Sgd {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl<B: Backend> Optimizer<B> for Sgd {
    fn apply(
        &mut self,
        backend: &B,
        parameters: Vec<&mut Parameter<B>>,
        grads: &[Option<&B::Tensor>],
        grad_scale: f64,
    ) {
        for (parameter, grad) in parameters.into_iter().zip(grads) {
            let Some(grad) = grad else {
                continue;
            };
            // Réduire le gradient ou réduire le pas revient au même.
            let learning_rate = self.learning_rate * parameter.lr_scale() * grad_scale;
            backend.sgd_step(parameter.value_mut(), grad, learning_rate);
        }
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
}

/// Moyennes glissantes du gradient d'un paramètre et de son carré.
pub type Moments<B> = (<B as Primitives>::Tensor, <B as Primitives>::Tensor);

/// Adam avec décroissance des poids découplée.
///
/// Là où SGD applique le même pas à tous les paramètres, Adam adapte le pas
/// de chaque nombre : il divise la moyenne récente de son gradient par la
/// racine de la moyenne récente de son carré. Un paramètre dont le gradient
/// est petit mais régulier avance alors aussi vite qu'un autre.
pub struct AdamW<B: Backend> {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    /// Nombre de pas déjà faits.
    steps: u32,
    /// Pour chaque paramètre, dans l'ordre de `Module::parameters_mut` :
    /// moyennes glissantes du gradient et de son carré.
    moments: Vec<Option<Moments<B>>>,
}

impl<B: Backend> AdamW<B> {
    /// Les réglages de nanoGPT : `β₁ = 0,9`, `β₂ = 0,95`, `ε = 1e-8`.
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay,
            steps: 0,
            moments: Vec::new(),
        }
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Les moments de chaque paramètre, dans l'ordre des paramètres ; `None`
    /// pour un paramètre qui n'a encore jamais reçu de gradient.
    pub fn moments(&self) -> &[Option<Moments<B>>] {
        &self.moments
    }

    /// Remet l'optimiseur dans un état enregistré : avec les moments et le
    /// nombre de pas, un entraînement interrompu reprend exactement où il
    /// en était.
    pub fn restore(&mut self, steps: u32, moments: Vec<Option<Moments<B>>>) {
        self.steps = steps;
        self.moments = moments;
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn apply(
        &mut self,
        backend: &B,
        parameters: Vec<&mut Parameter<B>>,
        grads: &[Option<&B::Tensor>],
        grad_scale: f64,
    ) {
        assert_eq!(
            parameters.len(),
            grads.len(),
            "un gradient par paramètre est attendu"
        );
        self.steps += 1;
        // Les moments partent de zéro et sous-estiment donc les vraies
        // moyennes pendant les premiers pas. Diviser par `1 - βᵗ` compense
        // exactement ; ce facteur tend vers 1 quand `t` grandit.
        let bias_correction1 = 1.0 - self.beta1.powi(self.steps as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.steps as i32);

        if self.moments.len() < parameters.len() {
            self.moments.resize_with(parameters.len(), || None);
        }

        for ((parameter, grad), moments) in parameters.into_iter().zip(grads).zip(&mut self.moments)
        {
            let Some(grad) = grad else {
                continue;
            };
            let (first_moment, second_moment) = moments.get_or_insert_with(|| {
                let shape = backend.shape(parameter.value()).to_vec();
                (backend.zeros(&shape), backend.zeros(&shape))
            });
            let step = AdamWStep {
                learning_rate: self.learning_rate * parameter.lr_scale(),
                beta1: self.beta1,
                beta2: self.beta2,
                epsilon: self.epsilon,
                weight_decay: if parameter.decay() {
                    self.weight_decay
                } else {
                    0.0
                },
                bias_correction1,
                bias_correction2,
                grad_scale,
            };
            backend.adamw_step(
                parameter.value_mut(),
                grad,
                first_moment,
                second_moment,
                &step,
            );
        }
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    fn state(&self) -> OptimizerState<&B::Tensor> {
        let first = self.moments.iter().map(|m| m.as_ref().map(|m| &m.0));
        let second = self.moments.iter().map(|m| m.as_ref().map(|m| &m.1));
        OptimizerState {
            counters: BTreeMap::from([(ADAMW_STEPS.to_string(), self.steps.to_string())]),
            slots: vec![
                (ADAMW_FIRST_MOMENT.to_string(), first.collect()),
                (ADAMW_SECOND_MOMENT.to_string(), second.collect()),
            ],
        }
    }

    fn state_layout(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        (
            vec![ADAMW_STEPS],
            vec![ADAMW_FIRST_MOMENT, ADAMW_SECOND_MOMENT],
        )
    }

    fn restore_state(&mut self, mut state: OptimizerState<B::Tensor>) -> Result<(), String> {
        let steps = state
            .counters
            .get(ADAMW_STEPS)
            .and_then(|steps| steps.parse().ok())
            .ok_or_else(|| format!("état d'AdamW : `{ADAMW_STEPS}` absent ou illisible"))?;
        let mut take = |prefix: &str| {
            let index = state.slots.iter().position(|(name, _)| name == prefix);
            index
                .map(|index| state.slots.swap_remove(index).1)
                .ok_or_else(|| format!("état d'AdamW : série `{prefix}` absente"))
        };
        let (first, second) = (take(ADAMW_FIRST_MOMENT)?, take(ADAMW_SECOND_MOMENT)?);
        let moments = first
            .into_iter()
            .zip(second)
            .map(|pair| match pair {
                (Some(first), Some(second)) => Ok(Some((first, second))),
                (None, None) => Ok(None),
                _ => Err("état d'AdamW : un moment sans l'autre".to_string()),
            })
            .collect::<Result<_, _>>()?;
        self.restore(steps, moments);
        Ok(())
    }
}

/// Nom du compteur de pas d'AdamW dans un point de reprise, et préfixes de
/// ses deux séries de moments.
const ADAMW_STEPS: &str = "adamw.steps";
const ADAMW_FIRST_MOMENT: &str = "adamw.m.";
const ADAMW_SECOND_MOMENT: &str = "adamw.v.";

/// Muon, l'optimiseur des matrices de poids de DeepSeek-V4, avec AdamW pour
/// tout le reste.
///
/// AdamW règle le pas de chaque nombre séparément. Muon regarde une matrice
/// de poids comme un tout. Il accumule le gradient dans une moyenne
/// glissante `M`, puis remplace cette matrice par la matrice **orthogonale**
/// la plus proche : celle qui garde ses directions mais ramène toutes ses
/// valeurs singulières à 1.
///
/// ```text
/// M = μ·M + G
/// O = orthogonaliser(μ·M + G) × √max(lignes, colonnes) × γ
/// W = W × (1 - η·λ) - η·O
/// ```
///
/// Une moyenne de gradients est dominée par quelques directions, celles où
/// la perte varie le plus. L'orthogonaliser donne le même poids à toutes :
/// les directions rares mais utiles avancent aussi vite que les autres.
///
/// Une matrice orthogonale a des éléments de taille `1/√max(lignes,
/// colonnes)`. Le facteur `√max(…) × γ` ramène ceux de la mise à jour à la
/// taille `γ`, proche de celle d'un pas d'AdamW : le même taux
/// d'apprentissage sert alors aux deux optimiseurs.
///
/// Muon ne s'applique qu'aux matrices de poids des couches. Les embeddings,
/// les gains et les biais restent à AdamW, comme dans le rapport.
pub struct Muon<B: Backend> {
    /// L'optimiseur des paramètres qui ne sont pas des matrices de poids.
    /// Il porte aussi le taux d'apprentissage et la décroissance des poids,
    /// communs aux deux.
    pub adamw: AdamW<B>,
    /// `μ`
    pub momentum: f64,
    /// `γ` : taille des éléments de la mise à jour avant le taux
    /// d'apprentissage.
    pub update_scale: f64,
    /// Les variantes, toutes éteintes par défaut.
    pub variant: MuonVariant,
    /// Pour chaque paramètre : la moyenne glissante de son gradient, s'il
    /// est traité par Muon et a déjà reçu un gradient.
    momenta: Vec<Option<B::Tensor>>,
    /// Pour chaque paramètre, avec [`MuonVariant::neuron_norm`] : la
    /// moyenne glissante du carré de la mise à jour de chaque neurone.
    neuron_moments: Vec<Option<B::Tensor>>,
    /// Où en est l'entraînement, de 0 à 1.
    progress: f64,
}

/// Les coefficients `(a, b, c)` de chaque itération de Newton-Schulz : huit
/// itérations qui font grandir vite les petites valeurs singulières, puis
/// deux qui les amènent précisément à 1. Ce sont ceux de DeepSeek-V4.
pub const NEWTON_SCHULZ: [(f64, f64, f64); 10] = {
    let fast = (3.4445, -4.7750, 2.0315);
    let exact = (2.0, -1.5, 0.5);
    [fast, fast, fast, fast, fast, fast, fast, fast, exact, exact]
};

/// Quatre itérations aux coefficients tous différents, réglés pour que la
/// réponse soit proche de 1 dès qu'une valeur singulière dépasse 0,02 de la
/// norme : douze produits matriciels au lieu de trente. Ce sont ceux de
/// l'article SAMuon (arXiv 2608.25990, table 4).
pub const NEWTON_SCHULZ_SHORT: [(f64, f64, f64); 4] = [
    (5.30697775, -9.73226547, 4.52926445),
    (3.99123669, -4.20899105, 1.18235242),
    (2.66316843, -2.16650701, 0.56325209),
    (1.93040931, -1.31219244, 0.38289258),
];

/// Les variantes de Muon étudiées à l'étape 11. Chacune vient d'un article
/// et se règle séparément ; par défaut, Muon est celui de DeepSeek-V4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonVariant {
    /// Quatre itérations de Newton-Schulz ([`NEWTON_SCHULZ_SHORT`]) au lieu
    /// de dix.
    pub short: bool,
    /// Muon périodique par lignes (arXiv 2608.20818) : l'orthogonalisation
    /// n'a lieu qu'un pas sur `period`. Les autres pas se contentent de
    /// ramener chaque ligne de la moyenne à la norme 1, ce qui ne coûte
    /// presque rien. 1 : orthogonalisation à chaque pas.
    pub period: usize,
    /// Taille d'un pas par lignes rapportée à celle d'un pas orthogonalisé.
    /// Les deux mises à jour ont la même norme, mais pas la même géométrie :
    /// l'article retient 0,15.
    pub row_scale: f64,
    /// NorMuon (arXiv 2510.05491) : après l'orthogonalisation, la mise à
    /// jour de chaque neurone est divisée par la racine de la moyenne
    /// glissante de son carré, comme le fait Adam nombre par nombre.
    pub neuron_norm: bool,
    /// SAMuon-lite (arXiv 2608.25990) : toutes les directions de la mise à
    /// jour sont multipliées par ce facteur, sauf la première, la plus
    /// instable, qui garde la taille que Muon lui donne. 1 : éteint.
    pub tail_boost: f64,
}

impl Default for MuonVariant {
    fn default() -> Self {
        Self {
            short: false,
            period: 1,
            row_scale: 0.15,
            neuron_norm: false,
            tail_boost: 1.0,
        }
    }
}

impl MuonVariant {
    /// Les coefficients de Newton-Schulz de cette variante.
    pub fn newton_schulz(&self) -> &'static [(f64, f64, f64)] {
        if self.short {
            &NEWTON_SCHULZ_SHORT
        } else {
            &NEWTON_SCHULZ
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.period == 0 {
            return Err("muon : `period` doit valoir au moins 1".to_string());
        }
        if self.row_scale <= 0.0 || !self.row_scale.is_finite() {
            return Err("muon : `row` doit être positif".to_string());
        }
        if self.tail_boost < 1.0 || !self.tail_boost.is_finite() {
            return Err("muon : `boost` doit valoir au moins 1".to_string());
        }
        if self.neuron_norm && self.tail_boost != 1.0 {
            // La normalisation par neurone redonne à la mise à jour une
            // taille fixe : elle défait ce que l'amplification a réglé.
            return Err("muon : `norm` et `boost` ne se combinent pas".to_string());
        }
        Ok(())
    }
}

/// `β₂` et `ε` de la normalisation par neurone, ceux de l'article NorMuon.
const NEURON_BETA: f64 = 0.95;
const NEURON_EPSILON: f64 = 1e-8;
/// SAMuon-lite : l'amplification monte de 1 à sa valeur pendant cette part
/// de l'entraînement, en demi-cosinus. Au début, la direction dominante
/// n'est pas encore détachée des autres.
const BOOST_WARMUP: f64 = 0.3;
/// SAMuon-lite : nombre d'itérations de la puissance itérée qui cherche la
/// direction dominante (10 pour le plus petit modèle de l'article).
const POWER_ITERATIONS: usize = 10;

/// La matrice orthogonale la plus proche de `matrix`, par itérations de
/// Newton-Schulz.
///
/// Toute matrice s'écrit `U·S·Vᵀ`, où `S` porte ses valeurs singulières ;
/// la matrice cherchée est `U·Vᵀ`. La calculer exactement demande une
/// décomposition coûteuse. L'itération y arrive avec des produits
/// matriciels seulement :
///
/// ```text
/// X ← a·X + b·(X·Xᵀ)·X + c·(X·Xᵀ)²·X
/// ```
///
/// Elle ne touche ni à `U` ni à `V` : elle applique à chaque valeur
/// singulière `s` le polynôme `a·s + b·s³ + c·s⁵`, qui la pousse vers 1.
/// La matrice est d'abord divisée par sa norme, pour que toutes ses valeurs
/// singulières soient entre 0 et 1.
///
/// `coefficients` donne `(a, b, c)` pour chaque itération :
/// [`NEWTON_SCHULZ`] ou [`NEWTON_SCHULZ_SHORT`].
///
/// `matrix` peut aussi être une pile de matrices de même forme,
/// `[K, lignes, colonnes]` : chacune est orthogonalisée séparément, mais
/// chaque produit matriciel les traite toutes d'un coup. Sur GPU, un seul
/// gros produit coûte bien moins que `K` petits.
pub fn orthogonalize<B: Backend>(
    backend: &B,
    matrix: &B::Tensor,
    coefficients: &[(f64, f64, f64)],
) -> B::Tensor {
    let shape = backend.shape(matrix).to_vec();
    let rank = shape.len();
    assert!(
        rank == 2 || rank == 3,
        "orthogonalize attend une matrice ou une pile de matrices, reçu {shape:?}"
    );
    let scalar = |value: f64| backend.scalar(B::Scalar::from_f64(value));
    let (rows, cols) = (rank - 2, rank - 1);
    let mut x = normalize(backend, matrix);
    // `X·Xᵀ` ou `Xᵀ·X` : on choisit le plus petit des deux carrés.
    let wide = shape[rows] <= shape[cols];
    for &(a, b, c) in coefficients {
        let transposed = backend.transpose(&x, rows, cols);
        let gram = if wide {
            backend.matmul(&x, &transposed)
        } else {
            backend.matmul(&transposed, &x)
        };
        // `b·A + c·A²`, puis son produit avec `X` du bon côté.
        let polynomial = backend.add(
            &backend.mul(&gram, &scalar(b)),
            &backend.mul(&backend.matmul(&gram, &gram), &scalar(c)),
        );
        let correction = if wide {
            backend.matmul(&polynomial, &x)
        } else {
            backend.matmul(&x, &polynomial)
        };
        x = backend.add(&backend.mul(&x, &scalar(a)), &correction);
    }
    x
}

/// `X / ‖X‖`, où `‖X‖ = √(Σ x²)`, pour une matrice ou pour chaque matrice
/// d'une pile `[K, m, n]`. Calculé sur le backend : rien n'est lu sur
/// l'hôte.
fn normalize<B: Backend>(backend: &B, tensor: &B::Tensor) -> B::Tensor {
    let rank = backend.shape(tensor).len();
    let squares = backend.mul(tensor, tensor);
    // La somme sur les deux derniers axes : une norme par matrice.
    let sums = backend.sum(&backend.sum(&squares, rank - 1, true), rank - 2, true);
    let epsilon = backend.scalar(B::Scalar::from_f64(1e-7));
    backend.div(tensor, &backend.add(&backend.sqrt(&sums), &epsilon))
}

/// Ramène à la norme 1 chaque ligne de `matrix`, ou chaque colonne si elle
/// est plus haute que large : les « lignes » sont prises le long de la plus
/// grande dimension, comme pour l'orthogonalisation.
///
/// Une matrice orthogonalisée a elle aussi des lignes de norme 1, mais en
/// plus toutes perpendiculaires entre elles. Cette opération n'en garde
/// que la première moitié, la moins chère.
pub fn normalize_rows<B: Backend>(backend: &B, matrix: &B::Tensor) -> B::Tensor {
    let shape = backend.shape(matrix);
    assert_eq!(
        shape.len(),
        2,
        "normalize_rows attend une matrice, reçu {shape:?}"
    );
    let along = if shape[0] <= shape[1] { 1 } else { 0 };
    let norms = backend.sqrt(&backend.sum(&backend.mul(matrix, matrix), along, true));
    let epsilon = backend.scalar(B::Scalar::from_f64(1e-7));
    backend.div(matrix, &backend.add(&norms, &epsilon))
}

/// La direction dominante de `matrix` : `u·vᵀ`, où `u` et `v` sont ses
/// premiers vecteurs singuliers, ceux de sa plus grande valeur singulière.
///
/// Elle est trouvée par la puissance itérée : multiplier un vecteur par
/// `Mᵀ·M` encore et encore l'aligne sur `v`, parce que cette direction est
/// celle que la matrice amplifie le plus. Chaque itération ne coûte que
/// deux produits d'une matrice par un vecteur.
///
/// Comme [`orthogonalize`], accepte une pile de matrices `[K, m, n]`.
pub fn leading_direction<B: Backend>(
    backend: &B,
    matrix: &B::Tensor,
    iterations: usize,
) -> B::Tensor {
    let shape = backend.shape(matrix).to_vec();
    let rank = shape.len();
    assert!(
        rank == 2 || rank == 3,
        "leading_direction attend une matrice ou une pile de matrices, reçu {shape:?}"
    );
    let (rows, cols) = (rank - 2, rank - 1);
    // Normalisée, la matrice a toutes ses valeurs singulières sous 1 : le
    // vecteur ne peut que rétrécir, et il est renormalisé assez souvent
    // pour ne jamais s'éteindre.
    let matrix = normalize(backend, matrix);
    let transposed = backend.transpose(&matrix, rows, cols);
    // Un vecteur colonne par matrice : `[n, 1]` ou `[K, n, 1]`.
    let mut column = shape.clone();
    column[rows] = shape[cols];
    column[cols] = 1;
    let mut v = normalize(backend, &backend.ones(&column));
    for iteration in 0..iterations {
        v = backend.matmul(&transposed, &backend.matmul(&matrix, &v));
        if iteration % 4 == 3 || iteration + 1 == iterations {
            v = normalize(backend, &v);
        }
    }
    let u = normalize(backend, &backend.matmul(&matrix, &v));
    backend.matmul(&u, &backend.transpose(&v, rows, cols))
}

/// Empile des matrices de même forme `[m, n]` en un tenseur `[K, m, n]`.
///
/// Les matrices sont réunies deux par deux, puis les paires entre elles :
/// chaque nombre n'est recopié que `log₂ K` fois.
fn stack<B: Backend>(backend: &B, matrices: &[&B::Tensor]) -> B::Tensor {
    let shape = backend.shape(matrices[0]).to_vec();
    let mut level: Vec<B::Tensor> = matrices
        .iter()
        .map(|matrix| backend.reshape(matrix, &[1, shape[0], shape[1]]))
        .collect();
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [a, b] => backend.concat_rows(a, b),
                [a] => a.clone(),
                _ => unreachable!("chunks(2) rend une ou deux matrices"),
            })
            .collect();
    }
    level.remove(0)
}

/// La matrice `index` d'une pile `[K, m, n]`.
fn unstack<B: Backend>(backend: &B, stacked: &B::Tensor, index: usize) -> B::Tensor {
    let shape = backend.shape(stacked).to_vec();
    let one = backend.gather_rows(stacked, &[index as u32]);
    backend.reshape(&one, &[shape[1], shape[2]])
}

impl<B: Backend> Muon<B> {
    /// Les réglages de DeepSeek-V4 : `μ = 0,95` et `γ = 0,18`.
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self::with_variant(learning_rate, weight_decay, MuonVariant::default())
    }

    /// Muon avec une ou plusieurs des variantes de l'étape 11.
    pub fn with_variant(learning_rate: f64, weight_decay: f64, variant: MuonVariant) -> Self {
        if let Err(message) = variant.validate() {
            panic!("{message}");
        }
        Self {
            adamw: AdamW::new(learning_rate, weight_decay),
            momentum: 0.95,
            update_scale: 0.18,
            variant,
            momenta: Vec::new(),
            neuron_moments: Vec::new(),
            progress: 0.0,
        }
    }

    /// Vrai si Muon traite ce paramètre : une matrice de poids, c'est-à-dire
    /// un tenseur à deux dimensions soumis à la décroissance. Une table
    /// d'embeddings a deux dimensions aussi, mais elle n'est pas une
    /// transformation : elle y échappe, comme elle échappe à la
    /// décroissance.
    pub fn handles(backend: &B, parameter: &Parameter<B>) -> bool {
        parameter.decay() && backend.shape(parameter.value()).len() == 2
    }

    /// L'amplification de SAMuon-lite au point où en est l'entraînement.
    fn tail_boost(&self) -> f64 {
        let warmup = (self.progress / BOOST_WARMUP).clamp(0.0, 1.0);
        let weight = 0.5 * (1.0 - (std::f64::consts::PI * warmup).cos());
        1.0 + (self.variant.tail_boost - 1.0) * weight
    }

    /// La mise à jour d'une matrice de poids, avant le taux
    /// d'apprentissage, à l'échelle où ses éléments valent `γ` en moyenne.
    /// `orthogonal` est la moyenne `ahead` orthogonalisée, et amplifiée si
    /// la variante le demande ; sans elle, le pas est un pas par lignes.
    fn update(
        &mut self,
        backend: &B,
        index: usize,
        ahead: &B::Tensor,
        orthogonal: Option<B::Tensor>,
    ) -> B::Tensor {
        let scalar = |value: f64| backend.scalar(B::Scalar::from_f64(value));
        let shape = backend.shape(ahead).to_vec();
        let size = shape[0].max(shape[1]) as f64;
        let scale = self.update_scale * size.sqrt();
        let Some(direction) = orthogonal else {
            let rows = normalize_rows(backend, ahead);
            return backend.mul(&rows, &scalar(scale * self.variant.row_scale));
        };
        if !self.variant.neuron_norm {
            return backend.mul(&direction, &scalar(scale));
        }
        // Un neurone est une colonne : les poids sont rangés
        // `[entrée, sortie]`.
        let squares = backend.mean(&backend.mul(&direction, &direction), 0, true);
        let squares = backend.mul(&squares, &scalar(1.0 - NEURON_BETA));
        let moment = match &self.neuron_moments[index] {
            Some(previous) => backend.add(&backend.mul(previous, &scalar(NEURON_BETA)), &squares),
            None => squares,
        };
        let normalized = backend.div(
            &direction,
            &backend.add(&backend.sqrt(&moment), &scalar(NEURON_EPSILON)),
        );
        self.neuron_moments[index] = Some(moment);
        // La division a changé la taille de la mise à jour : on la ramène
        // à celle où ses éléments valent `γ` en moyenne.
        let elements = (shape[0] * shape[1]) as f64;
        let unit = normalize(backend, &normalized);
        backend.mul(&unit, &scalar(self.update_scale * elements.sqrt()))
    }
}

impl<B: Backend> Optimizer<B> for Muon<B> {
    fn apply(
        &mut self,
        backend: &B,
        parameters: Vec<&mut Parameter<B>>,
        grads: &[Option<&B::Tensor>],
        grad_scale: f64,
    ) {
        assert_eq!(
            parameters.len(),
            grads.len(),
            "un gradient par paramètre est attendu"
        );
        if self.momenta.len() < parameters.len() {
            self.momenta.resize_with(parameters.len(), || None);
            self.neuron_moments.resize_with(parameters.len(), || None);
        }
        let scalar = |value: f64| backend.scalar(B::Scalar::from_f64(value));
        let learning_rate = self.adamw.learning_rate;
        // Le compteur d'AdamW compte les pas déjà faits : le premier pas
        // orthogonalise toujours.
        let orthogonal = (self.adamw.steps() as usize).is_multiple_of(self.variant.period);

        // Les matrices de poids d'abord. Leur gradient est ensuite retiré
        // de la liste : AdamW ne verra que le reste.
        let mut rest: Vec<Option<&B::Tensor>> = grads.to_vec();
        let mut parameters = parameters;
        // La moyenne « un pas plus loin » de chaque matrice traitée, et les
        // matrices rangées par forme : celles d'une même forme seront
        // orthogonalisées ensemble.
        let mut aheads: Vec<(usize, B::Tensor)> = Vec::new();
        let mut by_shape: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
        for (index, parameter) in parameters.iter().enumerate() {
            let Some(grad) = grads[index] else {
                continue;
            };
            if !Self::handles(backend, parameter) {
                continue;
            }
            rest[index] = None;
            let grad = backend.mul(grad, &scalar(grad_scale));
            let momentum = match &self.momenta[index] {
                Some(previous) => {
                    backend.add(&backend.mul(previous, &scalar(self.momentum)), &grad)
                }
                None => grad.clone(),
            };
            // L'astuce de Nesterov : on regarde un pas plus loin que la
            // moyenne, en lui ajoutant une fois de plus le gradient du jour.
            let ahead = backend.add(&backend.mul(&momentum, &scalar(self.momentum)), &grad);
            self.momenta[index] = Some(momentum);
            by_shape
                .entry(backend.shape(&ahead).to_vec())
                .or_default()
                .push(aheads.len());
            aheads.push((index, ahead));
        }

        // Une pile par forme : trente produits matriciels pour trente-deux
        // matrices d'attention, au lieu de neuf cent soixante.
        let mut directions: Vec<Option<B::Tensor>> = aheads.iter().map(|_| None).collect();
        let boost = self.tail_boost();
        if orthogonal {
            for group in by_shape.values() {
                let matrices: Vec<&B::Tensor> = group.iter().map(|&at| &aheads[at].1).collect();
                let stacked = stack(backend, &matrices);
                let mut result = orthogonalize(backend, &stacked, self.variant.newton_schulz());
                if boost > 1.0 {
                    // Tout est amplifié, puis la direction dominante est
                    // ramenée à sa taille de départ.
                    let leading = leading_direction(backend, &stacked, POWER_ITERATIONS);
                    result = backend.sub(
                        &backend.mul(&result, &scalar(boost)),
                        &backend.mul(&leading, &scalar(boost - 1.0)),
                    );
                }
                for (position, &at) in group.iter().enumerate() {
                    directions[at] = Some(unstack(backend, &result, position));
                }
            }
        }

        for ((index, ahead), direction) in aheads.iter().zip(directions) {
            let update = self.update(backend, *index, ahead, direction);
            let parameter = &mut parameters[*index];
            let scale = parameter.lr_scale();
            let update = backend.mul(&update, &scalar(learning_rate * scale));
            let decay = 1.0 - learning_rate * scale * self.adamw.weight_decay;
            let decayed = backend.mul(parameter.value(), &scalar(decay));
            *parameter.value_mut() = backend.sub(&decayed, &update);
        }
        self.adamw.apply(backend, parameters, &rest, grad_scale);
    }

    fn learning_rate(&self) -> f64 {
        self.adamw.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.adamw.learning_rate = learning_rate;
    }

    fn set_progress(&mut self, progress: f64) {
        self.progress = progress;
    }

    fn state(&self) -> OptimizerState<&B::Tensor> {
        let mut state = self.adamw.state();
        state.slots.push((
            MUON_MOMENTUM.to_string(),
            self.momenta.iter().map(Option::as_ref).collect(),
        ));
        if self.variant.neuron_norm {
            state.slots.push((
                MUON_NEURON_MOMENT.to_string(),
                self.neuron_moments.iter().map(Option::as_ref).collect(),
            ));
        }
        state
    }

    fn state_layout(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        let (counters, mut slots) = self.adamw.state_layout();
        slots.push(MUON_MOMENTUM);
        if self.variant.neuron_norm {
            slots.push(MUON_NEURON_MOMENT);
        }
        (counters, slots)
    }

    fn restore_state(&mut self, mut state: OptimizerState<B::Tensor>) -> Result<(), String> {
        let mut take = |prefix: &str| {
            let index = state.slots.iter().position(|(name, _)| name == prefix);
            index
                .map(|index| state.slots.swap_remove(index).1)
                .ok_or_else(|| format!("état de Muon : série `{prefix}` absente"))
        };
        self.momenta = take(MUON_MOMENTUM)?;
        self.neuron_moments = if self.variant.neuron_norm {
            take(MUON_NEURON_MOMENT)?
        } else {
            self.momenta.iter().map(|_| None).collect()
        };
        self.adamw.restore_state(state)
    }
}

/// Préfixe des moyennes glissantes de Muon dans un point de reprise.
const MUON_MOMENTUM: &str = "muon.m.";
/// Préfixe des moyennes par neurone de la variante NorMuon.
const MUON_NEURON_MOMENT: &str = "muon.v.";

/// L'optimiseur d'un entraînement, tel qu'on le nomme sur une ligne de
/// commande : `adamw`, `muon`, ou Muon suivi de ses variantes, comme
/// `muon:short,period=3,row=0.15`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerKind {
    AdamW,
    /// Muon pour les matrices de poids, AdamW pour le reste.
    Muon(MuonVariant),
}

impl OptimizerKind {
    /// Le texte qui désigne cet optimiseur : celui que
    /// [`std::str::FromStr`] relit. Les réglages à leur valeur par défaut
    /// n'y figurent pas, si bien que le Muon de DeepSeek-V4 reste `muon`.
    pub fn name(self) -> String {
        let variant = match self {
            Self::AdamW => return "adamw".to_string(),
            Self::Muon(variant) => variant,
        };
        let mut items = Vec::new();
        if variant.short {
            items.push("short".to_string());
        }
        if variant.period != 1 {
            items.push(format!("period={}", variant.period));
            items.push(format!("row={}", variant.row_scale));
        }
        if variant.neuron_norm {
            items.push("norm".to_string());
        }
        if variant.tail_boost != 1.0 {
            items.push(format!("boost={}", variant.tail_boost));
        }
        if items.is_empty() {
            "muon".to_string()
        } else {
            format!("muon:{}", items.join(","))
        }
    }
}

impl std::str::FromStr for OptimizerKind {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let (name, items) = text.split_once(':').unwrap_or((text, ""));
        match name {
            "adamw" if items.is_empty() => return Ok(Self::AdamW),
            "muon" => {}
            _ => {
                return Err(format!(
                    "`{text}` inconnu, choix possibles : adamw, muon, \
                     muon:short,period=K,row=X,norm,boost=X"
                ));
            }
        }
        let mut variant = MuonVariant::default();
        for item in items.split(',').filter(|item| !item.is_empty()) {
            let (key, value) = item.split_once('=').unwrap_or((item, ""));
            let number = || {
                value
                    .parse::<f64>()
                    .map_err(|_| format!("muon : `{item}` attend un nombre"))
            };
            match (key, value.is_empty()) {
                ("short", true) => variant.short = true,
                ("norm", true) => variant.neuron_norm = true,
                ("period", false) => {
                    variant.period = value
                        .parse()
                        .map_err(|_| format!("muon : `{item}` attend un entier"))?;
                }
                ("row", false) => variant.row_scale = number()?,
                ("boost", false) => variant.tail_boost = number()?,
                _ => return Err(format!("muon : réglage `{item}` inconnu")),
            }
        }
        variant.validate()?;
        Ok(Self::Muon(variant))
    }
}

/// L'optimiseur d'un entraînement, choisi par son nom.
pub enum AnyOptimizer<B: Backend> {
    AdamW(AdamW<B>),
    Muon(Muon<B>),
}

impl<B: Backend> AnyOptimizer<B> {
    pub fn new(kind: OptimizerKind, learning_rate: f64, weight_decay: f64) -> Self {
        match kind {
            OptimizerKind::AdamW => Self::AdamW(AdamW::new(learning_rate, weight_decay)),
            OptimizerKind::Muon(variant) => {
                Self::Muon(Muon::with_variant(learning_rate, weight_decay, variant))
            }
        }
    }

    fn inner(&self) -> &dyn Optimizer<B> {
        match self {
            Self::AdamW(optimizer) => optimizer,
            Self::Muon(optimizer) => optimizer,
        }
    }

    fn inner_mut(&mut self) -> &mut dyn Optimizer<B> {
        match self {
            Self::AdamW(optimizer) => optimizer,
            Self::Muon(optimizer) => optimizer,
        }
    }
}

impl<B: Backend> Optimizer<B> for AnyOptimizer<B> {
    fn apply(
        &mut self,
        backend: &B,
        parameters: Vec<&mut Parameter<B>>,
        grads: &[Option<&B::Tensor>],
        grad_scale: f64,
    ) {
        self.inner_mut()
            .apply(backend, parameters, grads, grad_scale);
    }

    fn learning_rate(&self) -> f64 {
        self.inner().learning_rate()
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.inner_mut().set_learning_rate(learning_rate);
    }

    fn set_progress(&mut self, progress: f64) {
        self.inner_mut().set_progress(progress);
    }

    fn state(&self) -> OptimizerState<&B::Tensor> {
        self.inner().state()
    }

    fn state_layout(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        self.inner().state_layout()
    }

    fn restore_state(&mut self, state: OptimizerState<B::Tensor>) -> Result<(), String> {
        self.inner_mut().restore_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tape;
    use crate::backend::cpu::{Cpu, CpuTensor};
    use rand::SeedableRng;

    type B = Cpu<f64>;
    type T = CpuTensor<f64>;

    /// Calcule le gradient de `Σ p²`, soit `2p`, et renvoie de quoi faire
    /// un pas d'optimiseur.
    fn gradient_of_squares(backend: &B, parameter: &Parameter<B>) -> (u64, Gradients<B>) {
        let tape = Tape::new(backend);
        let p = parameter.var(&tape);
        let loss = tape.sum_all(tape.mul(p, p));
        (tape.id(), tape.backward(loss))
    }

    fn close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{a} != {e}");
        }
    }

    #[test]
    fn sgd_moves_against_the_gradient() {
        let backend = Cpu::new();
        let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        // p - 0,1 × 2p = 0,8 p
        Sgd::new(0.1).step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
        close(parameter.value().data(), &[0.8, -1.6]);
    }

    #[test]
    fn sgd_descends_to_the_minimum() {
        let backend = Cpu::new();
        let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
        let mut sgd = Sgd::new(0.1);
        for _ in 0..200 {
            let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
            sgd.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
        }
        assert!(parameter.value().data().iter().all(|p| p.abs() < 1e-12));
    }

    #[test]
    fn adamw_first_step_has_the_size_of_the_learning_rate() {
        // Au premier pas, les corrections de biais donnent m̂ = g et v̂ = g² :
        // le pas vaut lr × g / |g|, soit lr dans le sens opposé au gradient,
        // quelle que soit la taille du gradient.
        let backend = Cpu::new();
        let mut parameter = Parameter::new(T::from_vec(vec![1.0, -200.0], &[2]), true);
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        let mut adamw = AdamW::new(0.01, 0.0);
        adamw.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
        let data = parameter.value().data();
        assert!((data[0] - 0.99).abs() < 1e-9, "{}", data[0]);
        assert!((data[1] + 199.99).abs() < 1e-9, "{}", data[1]);
        assert_eq!(adamw.steps(), 1);
    }

    #[test]
    fn adamw_weight_decay_shrinks_weights_only() {
        // Gradient nul : seul le terme de décroissance agit.
        let backend = Cpu::new();
        let mut weight = Parameter::new(T::from_vec(vec![2.0], &[1]), true);
        let mut bias = Parameter::new(T::from_vec(vec![2.0], &[1]), false);

        let tape = Tape::new(&backend);
        let zero = tape.scalar(0.0);
        let loss = tape.sum_all(tape.add(
            tape.mul(weight.var(&tape), zero),
            tape.mul(bias.var(&tape), zero),
        ));
        let (tape_id, grads) = (tape.id(), tape.backward(loss));
        drop(tape);

        let mut adamw = AdamW::new(0.1, 0.5);
        adamw.step(&backend, vec![&mut weight, &mut bias], tape_id, &grads, 1.0);
        // 2 - 0,1 × 0,5 × 2 = 1,9 pour le poids ; le biais ne bouge pas.
        close(weight.value().data(), &[1.9]);
        close(bias.value().data(), &[2.0]);
    }

    #[test]
    fn adamw_descends_to_the_minimum() {
        let backend = Cpu::new();
        let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
        let mut adamw = AdamW::new(0.05, 0.0);
        for _ in 0..500 {
            let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
            adamw.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
        }
        assert!(
            parameter.value().data().iter().all(|p| p.abs() < 0.05),
            "{:?}",
            parameter.value().data()
        );
    }

    #[test]
    fn parameter_without_gradient_is_left_alone() {
        let backend = Cpu::new();
        let mut used = Parameter::new(T::from_vec(vec![1.0], &[1]), true);
        let mut unused = Parameter::new(T::from_vec(vec![1.0], &[1]), true);
        let (tape_id, grads) = gradient_of_squares(&backend, &used);
        let mut adamw = AdamW::new(0.1, 0.5);
        adamw.step(&backend, vec![&mut used, &mut unused], tape_id, &grads, 1.0);
        assert_ne!(used.value().data(), &[1.0]);
        assert_eq!(unused.value().data(), &[1.0]);
    }

    #[test]
    fn gradient_scale_shrinks_the_sgd_step() {
        let backend = Cpu::new();
        let mut parameter = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
        let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
        // p - 0,1 × 0,5 × 2p = 0,9 p
        Sgd::new(0.1).step(&backend, vec![&mut parameter], tape_id, &grads, 0.5);
        close(parameter.value().data(), &[0.9, -1.8]);
    }

    #[test]
    fn gradient_scale_does_not_change_the_first_adamw_step() {
        // Adam divise le gradient par sa propre taille : au premier pas,
        // le réduire d'un facteur commun ne change rien.
        let backend = Cpu::new();
        let run = |scale: f64| {
            let mut parameter = Parameter::new(T::from_vec(vec![1.0, -200.0], &[2]), true);
            let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
            AdamW::new(0.01, 0.0).step(&backend, vec![&mut parameter], tape_id, &grads, scale);
            parameter.value().data().to_vec()
        };
        for (a, b) in run(1.0).iter().zip(run(0.01)) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn gradient_norm_spans_all_parameters() {
        // Gradients (2, -4) et (6) : norme √(4 + 16 + 36) = √56.
        let backend = Cpu::new();
        let first = Parameter::new(T::from_vec(vec![1.0, -2.0], &[2]), true);
        let second = Parameter::new(T::from_vec(vec![3.0], &[1]), true);
        let tape = Tape::new(&backend);
        let (a, b) = (first.var(&tape), second.var(&tape));
        let loss = tape.add(tape.sum_all(tape.mul(a, a)), tape.sum_all(tape.mul(b, b)));
        let grads = tape.backward(loss);
        let norm = gradient_norm(&backend, &[&first, &second], tape.id(), &grads);
        assert!((norm - 56f64.sqrt()).abs() < 1e-12, "{norm}");
    }

    #[test]
    fn schedule_warms_up_then_decays() {
        let rate = |step| cosine_schedule(step, 100, 1000, 1e-3, 1e-4);
        // Montée linéaire, qui atteint le pic au dernier pas d'échauffement.
        assert!((rate(0) - 1e-5).abs() < 1e-12);
        assert!((rate(49) - 5e-4).abs() < 1e-12);
        assert!((rate(99) - 1e-3).abs() < 1e-12);
        // Le cosinus part du pic…
        assert!((rate(100) - 1e-3).abs() < 1e-12);
        // … passe à mi-hauteur à mi-parcours…
        assert!((rate(550) - 5.5e-4).abs() < 1e-12);
        // … et finit au plancher, où il reste.
        assert!((rate(1000) - 1e-4).abs() < 1e-12);
        assert!((rate(5000) - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn schedule_never_increases_after_warmup() {
        let rates: Vec<f64> = (100..1000)
            .map(|step| cosine_schedule(step, 100, 1000, 1e-3, 1e-4))
            .collect();
        assert!(rates.windows(2).all(|pair| pair[1] <= pair[0]));
    }

    #[test]
    fn learning_rate_can_be_changed() {
        let mut adamw = AdamW::<B>::new(0.01, 0.0);
        Optimizer::<B>::set_learning_rate(&mut adamw, 0.5);
        assert_eq!(Optimizer::<B>::learning_rate(&adamw), 0.5);
    }

    // --- Muon ---

    fn matrix(seed: u64, rows: usize, cols: usize) -> T {
        T::randn(&[rows, cols], &mut rand::rngs::StdRng::seed_from_u64(seed))
    }

    /// Écart maximal entre `m` et la matrice identité.
    fn distance_to_identity(m: &T) -> f64 {
        let size = m.shape()[0];
        let mut largest = 0.0f64;
        for row in 0..size {
            for col in 0..size {
                let expected = if row == col { 1.0 } else { 0.0 };
                largest = largest.max((m.data()[row * size + col] - expected).abs());
            }
        }
        largest
    }

    #[test]
    fn orthogonalize_brings_singular_values_to_one() {
        let backend = Cpu::new();
        // Une matrice haute, une large, une carrée : le plus petit des deux
        // produits `Xᵀ·X` et `X·Xᵀ` doit devenir l'identité.
        for (rows, cols) in [(7, 4), (4, 7), (5, 5)] {
            let x = orthogonalize(&backend, &matrix(1, rows, cols), &NEWTON_SCHULZ);
            assert_eq!(x.shape(), &[rows, cols]);
            let gram = if rows <= cols {
                x.matmul(&x.transpose(0, 1))
            } else {
                x.transpose(0, 1).matmul(&x)
            };
            let distance = distance_to_identity(&gram);
            assert!(distance < 1e-3, "{rows} × {cols} : écart {distance}");
        }
    }

    #[test]
    fn orthogonalize_keeps_directions_and_forgets_sizes() {
        let backend = Cpu::new();
        // Une rotation d'un quart de tour, étirée de 5 dans une direction et
        // écrasée de 0,2 dans l'autre : il reste la rotation.
        let stretched = T::from_vec(vec![0.0, -0.2, 5.0, 0.0], &[2, 2]);
        let x = orthogonalize(&backend, &stretched, &NEWTON_SCHULZ);
        close_to(x.data(), &[0.0, -1.0, 1.0, 0.0], 1e-2);
        // Multiplier la matrice par un nombre positif ne change rien.
        let m = matrix(2, 4, 6);
        let a = orthogonalize(&backend, &m, &NEWTON_SCHULZ);
        let b = orthogonalize(&backend, &m.mul(&T::scalar(1000.0)), &NEWTON_SCHULZ);
        close_to(a.data(), b.data(), 1e-9);
    }

    fn close_to(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < tolerance, "{a} != {e}");
        }
    }

    #[test]
    fn muon_first_step_is_the_orthogonalized_gradient() {
        let backend = Cpu::new();
        let start = matrix(3, 3, 5);
        let grad = matrix(4, 3, 5);
        let mut weight = Parameter::new(start.clone(), true);
        let mut muon = Muon::<B>::new(0.01, 0.0);
        muon.apply(&backend, vec![&mut weight], &[Some(&grad)], 1.0);
        // Au premier pas, la moyenne est le gradient, et « un pas plus
        // loin » en est un multiple : la direction est celle du gradient
        // orthogonalisé. Le pas vaut `lr × γ × √max(3, 5)`.
        let step = 0.01 * 0.18 * 5f64.sqrt();
        let expected =
            start.sub(&orthogonalize(&backend, &grad, &NEWTON_SCHULZ).mul(&T::scalar(step)));
        close_to(weight.value().data(), expected.data(), 1e-12);
        // La taille du gradient ne compte pas.
        let mut again = Parameter::new(start, true);
        let mut muon = Muon::<B>::new(0.01, 0.0);
        let larger = grad.mul(&T::scalar(50.0));
        muon.apply(&backend, vec![&mut again], &[Some(&larger)], 1.0);
        close_to(again.value().data(), weight.value().data(), 1e-9);
    }

    #[test]
    fn muon_leaves_everything_but_weight_matrices_to_adamw() {
        let backend = Cpu::new();
        let build = || {
            (
                Parameter::<B>::new(matrix(5, 3, 4), true),
                // Une table d'embeddings : deux dimensions, sans décroissance.
                Parameter::<B>::new(matrix(6, 3, 4), false),
                Parameter::<B>::new(T::from_vec(vec![1.0, -2.0], &[2]), false),
            )
        };
        let grads = [matrix(7, 3, 4), matrix(8, 3, 4), T::ones(&[2])];
        let grads: Vec<Option<&T>> = grads.iter().map(Some).collect();

        let (mut weight, mut table, mut bias) = build();
        assert!(Muon::handles(&backend, &weight));
        assert!(!Muon::handles(&backend, &table) && !Muon::handles(&backend, &bias));
        let mut muon = Muon::<B>::new(0.01, 0.1);
        muon.apply(
            &backend,
            vec![&mut weight, &mut table, &mut bias],
            &grads,
            1.0,
        );

        let (mut plain_weight, mut plain_table, mut plain_bias) = build();
        let mut adamw = AdamW::<B>::new(0.01, 0.1);
        let all = vec![&mut plain_weight, &mut plain_table, &mut plain_bias];
        adamw.apply(&backend, all, &grads, 1.0);

        assert_eq!(table.value().data(), plain_table.value().data());
        assert_eq!(bias.value().data(), plain_bias.value().data());
        assert_ne!(weight.value().data(), plain_weight.value().data());
        // Un seul état de moyenne glissante, pour la matrice de poids.
        let state = muon.state();
        let (_, momenta) = state.slots.last().unwrap();
        assert_eq!(
            momenta.iter().map(Option::is_some).collect::<Vec<_>>(),
            [true, false, false]
        );
    }

    #[test]
    fn muon_weight_decay_shrinks_the_matrix() {
        // Un gradient nul s'orthogonalise en zéro : seule la décroissance
        // agit, de `lr × décroissance`.
        let backend = Cpu::new();
        let start = matrix(9, 2, 3);
        let mut weight = Parameter::new(start.clone(), true);
        let mut muon = Muon::<B>::new(0.1, 0.5);
        muon.apply(
            &backend,
            vec![&mut weight],
            &[Some(&T::zeros(&[2, 3]))],
            1.0,
        );
        close_to(
            weight.value().data(),
            start.mul(&T::scalar(0.95)).data(),
            1e-9,
        );
    }

    #[test]
    fn muon_descends_to_the_minimum() {
        // `Σ w²` sur une matrice : chaque pas a une taille fixée par le
        // taux, qu'on fait décroître pour finir près de zéro.
        let backend = Cpu::new();
        let mut parameter = Parameter::new(matrix(10, 3, 3), true);
        let mut muon = Muon::<B>::new(0.2, 0.0);
        for step in 0..200 {
            muon.set_learning_rate(0.2 * (1.0 - step as f64 / 200.0));
            let (tape_id, grads) = gradient_of_squares(&backend, &parameter);
            muon.step(&backend, vec![&mut parameter], tape_id, &grads, 1.0);
        }
        let largest = parameter
            .value()
            .data()
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(largest < 0.05, "plus grand poids : {largest}");
    }

    #[test]
    fn muon_state_can_be_saved_and_restored() {
        let backend = Cpu::new();
        let grad = matrix(11, 3, 4);
        let mut weight = Parameter::new(matrix(12, 3, 4), true);
        let mut muon = Muon::<B>::new(0.01, 0.0);
        muon.apply(&backend, vec![&mut weight], &[Some(&grad)], 1.0);

        // Une copie de l'état, comme le ferait un point de reprise.
        let state = muon.state();
        let copy = OptimizerState {
            counters: state.counters.clone(),
            slots: state
                .slots
                .iter()
                .map(|(name, tensors)| (name.clone(), tensors.iter().map(|t| t.cloned()).collect()))
                .collect(),
        };
        let (counters, slots) = muon.state_layout();
        assert_eq!(counters, ["adamw.steps"]);
        assert_eq!(slots, ["adamw.m.", "adamw.v.", "muon.m."]);

        let mut restored = Muon::<B>::new(0.01, 0.0);
        restored.restore_state(copy).unwrap();
        let mut twin = Parameter::new(weight.value().clone(), true);
        muon.apply(&backend, vec![&mut weight], &[Some(&grad)], 1.0);
        restored.apply(&backend, vec![&mut twin], &[Some(&grad)], 1.0);
        assert_eq!(weight.value().data(), twin.value().data());
        // Sans la série de Muon, l'état est refusé.
        let without = OptimizerState::<T> {
            counters: BTreeMap::from([("adamw.steps".to_string(), "1".to_string())]),
            slots: vec![
                ("adamw.m.".to_string(), vec![None]),
                ("adamw.v.".to_string(), vec![None]),
            ],
        };
        let error = Muon::<B>::new(0.01, 0.0).restore_state(without);
        assert!(error.unwrap_err().contains("muon.m."));
    }

    #[test]
    fn optimizers_are_chosen_by_name() {
        let variants = [
            MuonVariant::default(),
            MuonVariant {
                short: true,
                period: 3,
                row_scale: 0.2,
                ..MuonVariant::default()
            },
            MuonVariant {
                neuron_norm: true,
                ..MuonVariant::default()
            },
            MuonVariant {
                tail_boost: 5.0,
                ..MuonVariant::default()
            },
        ];
        let mut kinds = vec![OptimizerKind::AdamW];
        kinds.extend(variants.map(OptimizerKind::Muon));
        for kind in kinds {
            assert_eq!(kind.name().parse::<OptimizerKind>(), Ok(kind));
        }
        // Le Muon de DeepSeek-V4 garde son nom nu : les points de reprise
        // de l'étape 10 restent lisibles.
        assert_eq!(OptimizerKind::Muon(MuonVariant::default()).name(), "muon");
        assert_eq!(
            OptimizerKind::Muon(variants[1]).name(),
            "muon:short,period=3,row=0.2"
        );
        for (text, message) in [
            ("sgd", "adamw, muon"),
            ("adamw:short", "adamw, muon"),
            ("muon:fast", "`fast` inconnu"),
            ("muon:period=0", "au moins 1"),
            ("muon:period=x", "attend un entier"),
            ("muon:boost=0.5", "au moins 1"),
            ("muon:norm,boost=5", "ne se combinent pas"),
        ] {
            let error = text.parse::<OptimizerKind>().unwrap_err();
            assert!(error.contains(message), "{text} : {error}");
        }
        let muon = AnyOptimizer::<B>::new(OptimizerKind::Muon(variants[0]), 0.5, 0.1);
        assert_eq!(muon.learning_rate(), 0.5);
        assert_eq!(muon.state_layout().1.len(), 3);
        // NorMuon a une série de plus à enregistrer.
        let normuon = AnyOptimizer::<B>::new(OptimizerKind::Muon(variants[2]), 0.5, 0.1);
        assert_eq!(normuon.state_layout().1.len(), 4);
        let adamw = AnyOptimizer::<B>::new(OptimizerKind::AdamW, 0.5, 0.1);
        assert_eq!(adamw.state_layout().1.len(), 2);
    }

    #[test]
    fn a_stack_of_matrices_is_orthogonalized_one_by_one() {
        let backend = Cpu::new();
        let matrices = [matrix(1, 4, 7), matrix(2, 4, 7), matrix(3, 4, 7)];
        // Des tailles très différentes : chacune a sa propre norme.
        let matrices = [
            matrices[0].clone(),
            matrices[1].mul(&T::scalar(1e3)),
            matrices[2].mul(&T::scalar(1e-3)),
        ];
        let stacked = stack(&backend, &matrices.iter().collect::<Vec<_>>());
        assert_eq!(stacked.shape(), &[3, 4, 7]);
        let together = orthogonalize(&backend, &stacked, &NEWTON_SCHULZ);
        for (index, single) in matrices.iter().enumerate() {
            let alone = orthogonalize(&backend, single, &NEWTON_SCHULZ);
            close_to(
                unstack(&backend, &together, index).data(),
                alone.data(),
                1e-9,
            );
        }
    }

    #[test]
    fn muon_treats_each_matrix_as_if_it_were_alone() {
        // Trois matrices, dont deux de même forme : les empiler pour les
        // orthogonaliser ne doit rien changer à leurs mises à jour.
        let backend = Cpu::new();
        let shapes = [(3, 5), (4, 4), (3, 5)];
        let grads: Vec<T> = (0..3)
            .map(|i| matrix(10 + i as u64, shapes[i].0, shapes[i].1))
            .collect();
        let build = || -> Vec<Parameter<B>> {
            (0..3)
                .map(|i| Parameter::new(matrix(20 + i as u64, shapes[i].0, shapes[i].1), true))
                .collect()
        };
        let mut together = build();
        let mut muon = Muon::<B>::new(0.01, 0.1);
        for _ in 0..3 {
            let all: Vec<Option<&T>> = grads.iter().map(Some).collect();
            muon.apply(&backend, together.iter_mut().collect(), &all, 1.0);
        }
        let mut alone = build();
        for (parameter, grad) in alone.iter_mut().zip(&grads) {
            let mut muon = Muon::<B>::new(0.01, 0.1);
            for _ in 0..3 {
                muon.apply(&backend, vec![&mut *parameter], &[Some(grad)], 1.0);
            }
        }
        for (a, b) in together.iter().zip(&alone) {
            close_to(a.value().data(), b.value().data(), 1e-12);
        }
    }

    #[test]
    fn short_newton_schulz_orthogonalizes_too() {
        let backend = Cpu::new();
        for (rows, cols) in [(7, 4), (4, 7), (5, 5)] {
            let x = orthogonalize(&backend, &matrix(1, rows, cols), &NEWTON_SCHULZ_SHORT);
            let gram = if rows <= cols {
                x.matmul(&x.transpose(0, 1))
            } else {
                x.transpose(0, 1).matmul(&x)
            };
            // Quatre itérations : l'article annonce une erreur de 0,007 sur
            // chaque valeur singulière, donc du double sur son carré.
            let distance = distance_to_identity(&gram);
            assert!(distance < 3e-2, "{rows} × {cols} : écart {distance}");
        }
    }

    #[test]
    fn normalize_rows_gives_unit_rows_along_the_long_side() {
        let backend = Cpu::new();
        // Large : chaque ligne. Haute : chaque colonne.
        let wide = normalize_rows(&backend, &matrix(1, 3, 5));
        for row in 0..3 {
            let norm: f64 = (0..5)
                .map(|col| wide.data()[row * 5 + col] * wide.data()[row * 5 + col])
                .sum();
            assert!((norm - 1.0).abs() < 1e-6, "ligne {row} : {norm}");
        }
        let tall = normalize_rows(&backend, &matrix(2, 5, 3));
        for col in 0..3 {
            let norm: f64 = (0..5)
                .map(|row| tall.data()[row * 3 + col] * tall.data()[row * 3 + col])
                .sum();
            assert!((norm - 1.0).abs() < 1e-6, "colonne {col} : {norm}");
        }
        // La direction de chaque ligne est gardée.
        let source = matrix(1, 3, 5);
        let ratio = wide.data()[0] / source.data()[0];
        assert!(ratio > 0.0);
        assert!((wide.data()[1] / source.data()[1] - ratio).abs() < 1e-9);
    }

    /// `3·u₁·v₁ᵀ + u₂·v₂ᵀ`, avec des vecteurs choisis perpendiculaires :
    /// une matrice dont on connaît la direction dominante, `u₁·v₁ᵀ`.
    fn two_directions() -> (T, T) {
        let u1 = T::from_vec(vec![0.6, 0.8, 0.0], &[3, 1]);
        let u2 = T::from_vec(vec![0.0, 0.0, 1.0], &[3, 1]);
        let v1 = T::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[1, 4]);
        let v2 = T::from_vec(vec![0.5, -0.5, 0.5, -0.5], &[1, 4]);
        let leading = u1.matmul(&v1);
        let m = leading.mul(&T::scalar(3.0)).add(&u2.matmul(&v2));
        (m, leading)
    }

    #[test]
    fn leading_direction_finds_the_largest_singular_pair() {
        let backend = Cpu::new();
        let (m, leading) = two_directions();
        close_to(
            leading_direction(&backend, &m, 10).data(),
            leading.data(),
            1e-6,
        );
        // Haute plutôt que large : même résultat, transposé.
        let found = leading_direction(&backend, &m.transpose(0, 1), 10);
        close_to(found.transpose(0, 1).data(), leading.data(), 1e-6);
    }

    /// Un pas de Muon sur une matrice, avec la variante et l'avancement
    /// donnés ; renvoie le déplacement `W₀ − W₁`.
    fn muon_step(variant: MuonVariant, progress: f64, grads: &[&T]) -> Vec<T> {
        let backend = Cpu::new();
        let mut weight = Parameter::<B>::new(T::zeros(grads[0].shape()), true);
        let mut muon = Muon::<B>::with_variant(1.0, 0.0, variant);
        muon.set_progress(progress);
        grads
            .iter()
            .map(|grad| {
                let before: T = weight.value().clone();
                muon.apply(&backend, vec![&mut weight], &[Some(grad)], 1.0);
                before.sub(weight.value())
            })
            .collect()
    }

    #[test]
    fn periodic_muon_orthogonalizes_one_step_out_of_k() {
        let backend = Cpu::new();
        let variant = MuonVariant {
            period: 3,
            row_scale: 0.25,
            ..MuonVariant::default()
        };
        let grad = matrix(4, 3, 5);
        let moves = muon_step(variant, 0.0, &[&grad; 4]);
        let scale = 0.18 * 5f64.sqrt();
        // Pas 0 et 3 : orthogonalisés. La moyenne est un multiple du
        // gradient à chaque pas, puisque le gradient ne change pas.
        let orthogonal = orthogonalize(&backend, &grad, &NEWTON_SCHULZ).mul(&T::scalar(scale));
        close_to(moves[0].data(), orthogonal.data(), 1e-7);
        close_to(moves[3].data(), orthogonal.data(), 1e-7);
        // Pas 1 et 2 : lignes normalisées, à l'échelle réduite.
        let rows = normalize_rows(&backend, &grad).mul(&T::scalar(scale * 0.25));
        close_to(moves[1].data(), rows.data(), 1e-7);
        close_to(moves[2].data(), rows.data(), 1e-7);
    }

    #[test]
    fn neuron_norm_gives_every_neuron_the_same_update_size() {
        let (rows, cols) = (4, 6);
        // Dix itérations laissent des colonnes de normes inégales sur une
        // matrice mal conditionnée ; la normalisation les égalise.
        let mut data = matrix(4, rows, cols);
        data = data.mul(&T::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 1e-3, 1e-3],
            &[1, cols],
        ));
        let variant = MuonVariant {
            neuron_norm: true,
            ..MuonVariant::default()
        };
        let update = &muon_step(variant, 0.0, &[&data])[0];
        let column = |col: usize| -> f64 {
            (0..rows)
                .map(|row| update.data()[row * cols + col] * update.data()[row * cols + col])
                .sum()
        };
        for col in 1..cols {
            assert!(
                (column(col) - column(0)).abs() < 1e-3 * column(0),
                "colonne {col} : {} contre {}",
                column(col),
                column(0)
            );
        }
        // Les éléments valent `γ` en moyenne quadratique.
        let mean_square: f64 =
            update.data().iter().map(|x| x * x).sum::<f64>() / (rows * cols) as f64;
        assert!((mean_square.sqrt() - 0.18).abs() < 1e-6, "{mean_square}");
    }

    #[test]
    fn tail_boost_amplifies_everything_but_the_leading_direction() {
        let backend = Cpu::new();
        let (m, leading) = two_directions();
        let variant = MuonVariant {
            tail_boost: 5.0,
            ..MuonVariant::default()
        };
        let plain = &muon_step(MuonVariant::default(), 1.0, &[&m])[0];
        // Au début de l'entraînement, l'amplification est éteinte.
        let cold = &muon_step(variant, 0.0, &[&m])[0];
        close_to(cold.data(), plain.data(), 1e-12);
        // À mi-montée, elle vaut la moitié de son excès : 1 + 4 / 2 = 3.
        let scale = 0.18 * 4f64.sqrt();
        let lead = leading.mul(&T::scalar(scale));
        let expected = |boost: f64| {
            plain
                .mul(&T::scalar(boost))
                .sub(&lead.mul(&T::scalar(boost - 1.0)))
        };
        let warm = &muon_step(variant, 0.15, &[&m])[0];
        close_to(warm.data(), expected(3.0).data(), 1e-6);
        let hot = &muon_step(variant, 0.5, &[&m])[0];
        close_to(hot.data(), expected(5.0).data(), 1e-6);
        // Projetée sur la direction dominante, la mise à jour a la taille
        // de celle de Muon ; sur l'autre, cinq fois plus. La tolérance est
        // celle de l'orthogonalisation, dont l'erreur est ici amplifiée.
        let along = |update: &T, direction: &T| -> f64 {
            let product = backend.mul(update, direction);
            product.data().iter().sum()
        };
        let other = orthogonalize(&backend, &m, &NEWTON_SCHULZ).sub(&leading);
        assert!((along(hot, &leading) / along(plain, &leading) - 1.0).abs() < 5e-2);
        assert!((along(hot, &other) / along(plain, &other) - 5.0).abs() < 5e-2);
    }
}
