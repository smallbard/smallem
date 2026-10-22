//! L'entraînement : un pas, puis la boucle qui les enchaîne.
//!
//! - [`step`] : un pas d'entraînement, indépendant du modèle et de
//!   l'optimiseur ;
//! - [`trainer`] : la boucle complète d'un GPT, avec sa planification, ses
//!   évaluations et ses points de reprise.

pub mod step;
pub mod trainer;

pub use step::{evaluate, train_step, train_step_accumulated, train_step_clipped};
pub use trainer::{
    Evaluation, Event, Settings, Summary, Trainer, check_resume, step_rng, validation_loss,
};
