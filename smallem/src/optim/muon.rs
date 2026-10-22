//! Muon, l'optimiseur des matrices de poids, et ses variantes.

use std::collections::BTreeMap;

use super::{AdamW, Optimizer, OptimizerState};
use crate::backend::{Backend, Float};
use crate::nn::Parameter;

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
pub(super) fn stack<B: Backend>(backend: &B, matrices: &[&B::Tensor]) -> B::Tensor {
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
pub(super) fn unstack<B: Backend>(backend: &B, stacked: &B::Tensor, index: usize) -> B::Tensor {
    let shape = backend.shape(stacked).to_vec();
    let one = backend.gather_rows(stacked, &[index as u32]);
    backend.reshape(&one, &[shape[1], shape[2]])
}

impl<B: Backend> Muon<B> {
    /// Les réglages de DeepSeek-V4 : `μ = 0,95` et `γ = 0,18`.
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self::with_variant(learning_rate, weight_decay, MuonVariant::default())
    }

    /// Muon avec une ou plusieurs des variantes de l'étape 11. Une variante
    /// refusée par [`MuonVariant::validate`] arrête le programme : voir
    /// [`Muon::try_with_variant`].
    pub fn with_variant(learning_rate: f64, weight_decay: f64, variant: MuonVariant) -> Self {
        Self::try_with_variant(learning_rate, weight_decay, variant)
            .unwrap_or_else(|message| panic!("{message}"))
    }

    /// Comme [`Muon::with_variant`], une variante refusée étant une erreur.
    pub fn try_with_variant(
        learning_rate: f64,
        weight_decay: f64,
        variant: MuonVariant,
    ) -> Result<Self, String> {
        variant.validate()?;
        Ok(Self {
            adamw: AdamW::new(learning_rate, weight_decay),
            momentum: 0.95,
            update_scale: 0.18,
            variant,
            momenta: Vec::new(),
            neuron_moments: Vec::new(),
            progress: 0.0,
        })
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
        self.momenta = state.take_slot(MUON_MOMENTUM, "de Muon")?;
        self.neuron_moments = if self.variant.neuron_norm {
            state.take_slot(MUON_NEURON_MOMENT, "de Muon")?
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
