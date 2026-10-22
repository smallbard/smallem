//! Les optimiseurs : comment un gradient devient une modification des
//! paramètres.
//!
//! Un pas d'optimiseur modifie les paramètres sur place. La tape qui a
//! servi à calculer les gradients doit donc être libérée avant, sans quoi
//! elle référence encore les tenseurs à modifier.

use std::collections::BTreeMap;

use crate::autograd::Gradients;
use crate::backend::{Backend, Float};
use crate::nn::Parameter;

mod adamw;
mod muon;
#[cfg(test)]
mod tests;

pub use adamw::{AdamW, AdamWSettings, Moments};
pub use muon::{
    Muon, MuonVariant, NEWTON_SCHULZ, NEWTON_SCHULZ_SHORT, leading_direction, normalize_rows,
    orthogonalize,
};

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

impl<T> OptimizerState<T> {
    /// Retire la série `prefix`, ou dit son absence au nom de `owner`.
    pub fn take_slot(&mut self, prefix: &str, owner: &str) -> Result<Vec<Option<T>>, String> {
        let index = self.slots.iter().position(|(name, _)| name == prefix);
        index
            .map(|index| self.slots.swap_remove(index).1)
            .ok_or_else(|| format!("état {owner} : série `{prefix}` absente"))
    }

    /// Ajoute à cet état les compteurs et les séries de `other`.
    pub fn extend(&mut self, other: Self) {
        self.counters.extend(other.counters);
        self.slots.extend(other.slots);
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
