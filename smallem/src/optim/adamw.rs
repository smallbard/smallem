//! AdamW, et ses réglages.

use std::collections::BTreeMap;

use super::{Optimizer, OptimizerState};
use crate::backend::{AdamWStep, Backend, Primitives};
use crate::nn::Parameter;

/// Moyennes glissantes du gradient d'un paramètre et de son carré.
pub type Moments<B> = (<B as Primitives>::Tensor, <B as Primitives>::Tensor);

/// Les réglages d'AdamW qui ne changent pas au fil de l'entraînement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWSettings {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl AdamWSettings {
    /// Les réglages de nanoGPT : `β₁ = 0,9`, `β₂ = 0,95`, `ε = 1e-8`.
    pub const NANOGPT: Self = Self {
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
    };

    /// Le pas numéro `step`, compté à partir de 1, tel que le backend le
    /// reçoit.
    ///
    /// Les moments partent de zéro et sous-estiment donc les vraies
    /// moyennes pendant les premiers pas. Diviser par `1 - βᵗ` compense
    /// exactement ; ce facteur tend vers 1 quand `t` grandit.
    pub fn step(
        &self,
        step: u32,
        learning_rate: f64,
        weight_decay: f64,
        grad_scale: f64,
    ) -> AdamWStep {
        AdamWStep {
            learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay,
            bias_correction1: 1.0 - self.beta1.powi(step as i32),
            bias_correction2: 1.0 - self.beta2.powi(step as i32),
            grad_scale,
        }
    }
}

/// Adam avec décroissance des poids découplée.
///
/// Là où SGD applique le même pas à tous les paramètres, Adam adapte le pas
/// de chaque nombre : il divise la moyenne récente de son gradient par la
/// racine de la moyenne récente de son carré. Un paramètre dont le gradient
/// est petit mais régulier avance alors aussi vite qu'un autre.
pub struct AdamW<B: Backend> {
    pub learning_rate: f64,
    pub settings: AdamWSettings,
    pub weight_decay: f64,
    /// Nombre de pas déjà faits.
    steps: u32,
    /// Pour chaque paramètre, dans l'ordre de `Module::parameters_mut` :
    /// moyennes glissantes du gradient et de son carré.
    moments: Vec<Option<Moments<B>>>,
}

impl<B: Backend> AdamW<B> {
    /// Avec les réglages de nanoGPT ([`AdamWSettings::NANOGPT`]).
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            settings: AdamWSettings::NANOGPT,
            weight_decay,
            steps: 0,
            moments: Vec::new(),
        }
    }

    pub fn steps(&self) -> u32 {
        self.steps
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
            let weight_decay = if parameter.decay() {
                self.weight_decay
            } else {
                0.0
            };
            let step = self.settings.step(
                self.steps,
                self.learning_rate * parameter.lr_scale(),
                weight_decay,
                grad_scale,
            );
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
        let first = state.take_slot(ADAMW_FIRST_MOMENT, "d'AdamW")?;
        let second = state.take_slot(ADAMW_SECOND_MOMENT, "d'AdamW")?;
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
