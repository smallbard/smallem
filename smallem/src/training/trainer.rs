//! La boucle d'entraînement d'un GPT.
//!
//! Un [`Trainer`] réunit ce qu'un entraînement ne change pas en cours de
//! route : les réglages, les données, l'endroit où écrire. Sa méthode
//! [`Trainer::run`] enchaîne les pas, et à intervalles réguliers :
//!
//! - mesure la perte de validation ;
//! - enregistre le modèle si elle s'est améliorée ;
//! - écrit un point de reprise, avec l'état de l'optimiseur, le numéro de
//!   pas et les réglages.
//!
//! La boucle n'affiche rien. Ce qui se passe est rapporté à l'appelant par
//! des [`Event`], qu'il présente comme il veut.
//!
//! Si la perte ou le gradient cesse d'être un nombre fini, l'entraînement
//! s'arrête sans toucher au dernier point de reprise.

use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::step::{clip_scale, evaluate, train_step_observed};
use crate::backend::{Backend, Float};
use crate::data::Batches;
use crate::io::{SafeTensors, load_checkpoint, save_checkpoint, save_model};
use crate::model::Gpt;
use crate::nn::MemoryStorage;
use crate::optim::{AnyOptimizer, Optimizer, OptimizerKind, cosine_schedule};

/// Graine des lots de validation. Elle ne dépend pas de la graine de
/// l'entraînement : deux entraînements, même de graines différentes, sont
/// évalués sur les mêmes lots.
const EVAL_SEED: u64 = 0x5EED;

/// Les réglages d'un entraînement.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Nombre total de pas.
    pub steps: usize,
    /// Séquences par lot.
    pub batch_size: usize,
    /// Lots accumulés par pas : le gradient d'un pas est la moyenne de
    /// ceux de `accumulate` lots.
    pub accumulate: usize,
    /// Taux d'apprentissage au pic.
    pub learning_rate: f64,
    /// Nombre de pas de montée du taux.
    pub warmup_steps: usize,
    /// Le taux d'apprentissage finit à cette fraction de son pic.
    pub floor_ratio: f64,
    /// L'optimiseur.
    pub optimizer: OptimizerKind,
    /// Décroissance des poids.
    pub weight_decay: f64,
    /// Norme à laquelle le gradient est écrêté.
    pub max_grad_norm: f64,
    /// Graine des lots d'entraînement.
    pub seed: u64,
    /// Nombre de pas entre deux évaluations.
    pub eval_every: usize,
    /// Nombre de lots de validation moyennés à chaque évaluation.
    pub eval_batches: usize,
    /// Nombre de lots de l'évaluation finale. Elle sert à comparer des
    /// entraînements entre eux : il faut que son propre bruit soit petit
    /// devant les écarts qu'on cherche.
    pub final_eval_batches: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            steps: 5000,
            batch_size: 32,
            accumulate: 1,
            learning_rate: 1e-3,
            warmup_steps: 200,
            floor_ratio: 0.1,
            optimizer: OptimizerKind::AdamW,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
            seed: 42,
            eval_every: 500,
            eval_batches: 20,
            final_eval_batches: 200,
        }
    }
}

impl Settings {
    /// Le taux d'apprentissage du pas `step` : montée linéaire, puis
    /// descente en demi-cosinus jusqu'au plancher.
    pub fn learning_rate_at(&self, step: usize) -> f64 {
        cosine_schedule(
            step,
            self.warmup_steps,
            self.steps,
            self.learning_rate,
            self.learning_rate * self.floor_ratio,
        )
    }

    /// Les réglages dont dépend la suite d'un entraînement. Ils voyagent
    /// dans le point de reprise, pour vérifier qu'une reprise les respecte.
    ///
    /// Le rythme des évaluations n'en fait pas partie : il ne change rien
    /// aux pas eux-mêmes.
    pub fn metadata(&self) -> BTreeMap<String, String> {
        [
            ("run.steps", self.steps.to_string()),
            ("run.batch", self.batch_size.to_string()),
            ("run.accumulate", self.accumulate.to_string()),
            ("run.lr", self.learning_rate.to_string()),
            ("run.warmup", self.warmup_steps.to_string()),
            ("run.floor", self.floor_ratio.to_string()),
            ("run.optimizer", self.optimizer.name()),
            ("run.weight_decay", self.weight_decay.to_string()),
            ("run.max_grad_norm", self.max_grad_norm.to_string()),
            ("run.seed", self.seed.to_string()),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }
}

/// Vérifie qu'un entraînement repris a les réglages de celui qui a écrit le
/// point de reprise. Avec une autre graine ou un autre lot, les lots tirés
/// ne seraient plus les mêmes ; avec un autre nombre de pas, le taux
/// d'apprentissage ferait un saut.
///
/// Renvoie les réglages que le point de reprise ne contient pas (il a été
/// écrit par une version plus ancienne) : ceux-là ne peuvent pas être
/// vérifiés.
pub fn check_resume(
    saved: &BTreeMap<String, String>,
    current: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    let mut unknown = Vec::new();
    let mut differences = Vec::new();
    for (key, value) in current {
        match saved.get(key) {
            None => unknown.push(key.clone()),
            Some(saved_value) if saved_value != value => {
                differences.push(format!("{key} : {saved_value} enregistré, {value} demandé"));
            }
            Some(_) => {}
        }
    }
    if differences.is_empty() {
        Ok(unknown)
    } else {
        Err(format!(
            "les réglages ne sont pas ceux du point de reprise ({})",
            differences.join(" ; ")
        ))
    }
}

/// Le hasard des lots du pas `step`. Il ne dépend que de la graine et du
/// numéro de pas : un entraînement repris tire donc exactement les lots
/// qu'il aurait tirés sans interruption.
pub fn step_rng(seed: u64, step: usize) -> StdRng {
    StdRng::seed_from_u64(
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(step as u64),
    )
}

/// Ce que la boucle rapporte à l'appelant.
#[derive(Debug)]
pub enum Event<'a> {
    /// L'entraînement part de zéro. `val_loss` est la perte de validation
    /// du modèle tel qu'il a été initialisé.
    Started { val_loss: f64 },
    /// L'entraînement repart d'un point de reprise, après `step` pas.
    Resumed {
        step: usize,
        best_val_loss: f64,
        /// Les réglages absents du point de reprise, qui n'ont pas pu être
        /// vérifiés.
        unverified: &'a [String],
    },
    /// Une évaluation vient d'avoir lieu, et le point de reprise est écrit.
    Evaluated(Evaluation),
}

/// L'état d'un entraînement à une évaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Nombre de pas faits depuis le début de l'entraînement.
    pub step: usize,
    /// Perte d'entraînement moyenne depuis l'évaluation précédente.
    pub train_loss: f64,
    pub val_loss: f64,
    /// Taux d'apprentissage du dernier pas.
    pub learning_rate: f64,
    /// Plus grande norme de gradient depuis l'évaluation précédente, avant
    /// écrêtage.
    pub grad_norm_max: f64,
    /// Nombre de pas depuis l'évaluation précédente, et leur durée.
    pub steps: usize,
    pub seconds: f64,
    /// Vrai si la perte de validation est la meilleure jusqu'ici : le
    /// modèle vient d'être enregistré.
    pub improved: bool,
    /// Pour chaque mélange d'experts du modèle, le nombre de tokens
    /// d'entraînement reçus par chacun de ses experts aiguillés depuis
    /// l'évaluation précédente. Vide pour un modèle sans experts.
    pub expert_load: Vec<Vec<u64>>,
}

/// Le bilan d'un entraînement mené à son terme.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// Meilleure perte de validation : celle du modèle enregistré.
    pub best_val_loss: f64,
    /// Perte de validation du modèle final, sur `final_eval_batches` lots.
    pub final_val_loss: f64,
}

/// Perte moyenne de `model` sur `batches` lots de validation de
/// `batch_size` séquences. La graine est fixe : chaque évaluation voit les
/// mêmes lots, donc les valeurs sont comparables d'un modèle à l'autre, et
/// d'un programme à l'autre.
pub fn validation_loss<B: Backend>(
    backend: &B,
    model: &Gpt<B>,
    validation: &impl Batches,
    batches: usize,
    batch_size: usize,
) -> io::Result<f64> {
    let mut rng = StdRng::seed_from_u64(EVAL_SEED);
    let mut total = 0.0;
    for _ in 0..batches {
        let batch = validation.batch(&mut rng, batch_size, model.config().context)?;
        let (tokens, targets) = batch.examples();
        let loss = evaluate(backend, model, |m, t| {
            m.loss(t, &tokens, &targets, batch_size)
        });
        total += loss.to_f64();
    }
    // Ces tokens-là ne sont pas de l'entraînement : ils ne comptent ni
    // dans la charge des experts, ni dans son équilibrage.
    model.reset_expert_load();
    Ok(total / batches as f64)
}

/// Un entraînement : ses réglages, ses données, et où il écrit.
pub struct Trainer<'a, B: Backend, D: Batches> {
    pub backend: &'a B,
    pub settings: Settings,
    pub train: &'a D,
    pub validation: &'a D,
    /// Où enregistrer le meilleur modèle.
    pub model_path: PathBuf,
    /// Ce que l'appelant veut voir vérifié à une reprise, en plus des
    /// réglages : par exemple l'architecture, ou la précision du calcul.
    pub extra_settings: BTreeMap<String, String>,
}

impl<B: Backend, D: Batches> Trainer<'_, B, D> {
    /// Le point de reprise est écrit à côté du modèle.
    pub fn checkpoint_path(&self) -> PathBuf {
        self.model_path.with_extension("checkpoint.safetensors")
    }

    /// Tout ce qu'une reprise doit retrouver à l'identique.
    pub fn resume_settings(&self) -> BTreeMap<String, String> {
        let mut settings = self.settings.metadata();
        settings.extend(self.extra_settings.clone());
        settings
    }

    /// Perte moyenne sur `batches` lots de validation. La graine est fixe :
    /// chaque évaluation voit les mêmes lots, donc les valeurs sont
    /// comparables.
    pub fn validation_loss(&self, model: &Gpt<B>, batches: usize) -> io::Result<f64> {
        validation_loss(
            self.backend,
            model,
            self.validation,
            batches,
            self.settings.batch_size,
        )
    }

    /// Entraîne `model` jusqu'au dernier pas.
    ///
    /// Avec `resume`, le modèle et l'optimiseur sont d'abord relus du point
    /// de reprise, et l'entraînement repart du pas où il s'était arrêté.
    /// Les réglages doivent être ceux du point de reprise : toute
    /// différence est refusée.
    pub fn run(
        &self,
        model: &mut Gpt<B>,
        resume: bool,
        mut journal: impl FnMut(Event<'_>),
    ) -> Result<Summary, Box<dyn Error>> {
        let settings = &self.settings;
        for (name, value) in [
            ("steps", settings.steps),
            ("batch_size", settings.batch_size),
            ("accumulate", settings.accumulate),
            ("eval_every", settings.eval_every),
            ("eval_batches", settings.eval_batches),
            ("final_eval_batches", settings.final_eval_batches),
        ] {
            if value == 0 {
                return Err(format!("`{name}` doit valoir au moins 1").into());
            }
        }
        let backend = self.backend;
        let config = model.config();
        let checkpoint = self.checkpoint_path();
        let mut optimizer = AnyOptimizer::new(
            settings.optimizer,
            settings.learning_rate,
            settings.weight_decay,
        );
        let mut first_step = 0;
        let mut best = f64::INFINITY;

        if let Some(memory) = model.memory()
            && memory.dims().storage != MemoryStorage::Dense
        {
            if resume {
                return Err(
                    "une mémoire creuse ne reprend pas : ses moments ne sont pas \
                            enregistrés"
                        .into(),
                );
            }
            if settings.accumulate > 1 {
                return Err(
                    "une mémoire creuse ne cumule pas les gradients de plusieurs lots".into(),
                );
            }
        }
        if resume {
            let file = SafeTensors::read(&checkpoint)?;
            // Les réglages d'abord : un point de reprise qui n'est pas le
            // bon ne doit rien changer au modèle.
            let unverified = check_resume(&file.metadata, &self.resume_settings())?;
            let field = |key: &str| file.metadata.get(key).ok_or(format!("`{key}` absent"));
            first_step = field("step")?.parse()?;
            best = field("best_val_loss")?.parse()?;
            if first_step >= settings.steps {
                return Err(format!(
                    "cet entraînement est déjà terminé : pas {first_step} sur {}",
                    settings.steps
                )
                .into());
            }
            load_checkpoint(&file, backend, model, &mut optimizer)?;
            journal(Event::Resumed {
                step: first_step,
                best_val_loss: best,
                unverified: &unverified,
            });
        } else {
            journal(Event::Started {
                val_loss: self.validation_loss(model, settings.eval_batches)?,
            });
        }

        let stopped = |reason: String| -> Box<dyn Error> {
            format!(
                "{reason}. Entraînement arrêté ; le point de reprise {} n'a pas été modifié",
                checkpoint.display(),
            )
            .into()
        };
        let mut train_loss_sum = 0.0;
        let mut grad_norm_max: f64 = 0.0;
        let mut steps_since_eval = 0;
        let mut last_eval = Instant::now();

        for step in first_step..settings.steps {
            let learning_rate = settings.learning_rate_at(step);
            optimizer.set_learning_rate(learning_rate);
            let progress = step as f64 / settings.steps as f64;
            model.set_training_progress(progress);
            optimizer.set_progress(progress);

            let mut rng = step_rng(settings.seed, step);
            let mut batches = Vec::with_capacity(settings.accumulate);
            for _ in 0..settings.accumulate {
                let batch = self
                    .train
                    .batch(&mut rng, settings.batch_size, config.context)?;
                batches.push(batch.examples());
            }
            let (loss, grad_norm) = train_step_observed(
                backend,
                model,
                &mut optimizer,
                settings.max_grad_norm,
                settings.accumulate,
                |m, t, index| {
                    let (tokens, targets) = &batches[index];
                    m.training_loss(t, tokens, targets, settings.batch_size)
                },
                |m, tape, grads| m.collect_memory_gradient(tape, grads),
            );
            // Les lignes lues de la mémoire font leur pas à part, écrêtées
            // comme le reste ; leur gradient n'entre pas dans la norme.
            if let Some(scale) = clip_scale(grad_norm, settings.max_grad_norm) {
                model.apply_memory_gradient(backend, learning_rate, scale);
            }
            // L'écrêtage a déjà synchronisé le GPU pour lire la norme : lire
            // la perte en plus ne coûte presque rien.
            let loss = backend.item(&loss).to_f64();
            let done = step + 1;
            if !loss.is_finite() || !grad_norm.is_finite() {
                // Le pas n'a pas été appliqué. Continuer ne servirait à
                // rien, et écrire un point de reprise écraserait le dernier
                // qui vaut quelque chose.
                return Err(stopped(format!(
                    "pas {done} : perte {loss}, norme du gradient {grad_norm}"
                )));
            }
            model.rebalance_experts(backend);
            train_loss_sum += loss;
            grad_norm_max = grad_norm_max.max(grad_norm);
            steps_since_eval += 1;

            if done % settings.eval_every == 0 || done == settings.steps {
                let seconds = last_eval.elapsed().as_secs_f64();
                let expert_load = model.take_expert_load();
                let val_loss = self.validation_loss(model, settings.eval_batches)?;
                if !val_loss.is_finite() {
                    return Err(stopped(format!(
                        "pas {done} : perte de validation {val_loss}"
                    )));
                }
                let improved = val_loss < best;
                if improved {
                    best = val_loss;
                    save_model(&self.model_path, backend, model, config.to_metadata())?;
                }
                let mut metadata = config.to_metadata();
                metadata.extend(self.resume_settings());
                metadata.insert("step".into(), done.to_string());
                metadata.insert("best_val_loss".into(), best.to_string());
                save_checkpoint(&checkpoint, backend, model, &optimizer, metadata)?;
                journal(Event::Evaluated(Evaluation {
                    step: done,
                    train_loss: train_loss_sum / steps_since_eval as f64,
                    val_loss,
                    learning_rate,
                    grad_norm_max,
                    steps: steps_since_eval,
                    seconds,
                    improved,
                    expert_load,
                }));
                train_loss_sum = 0.0;
                grad_norm_max = 0.0;
                steps_since_eval = 0;
                last_eval = Instant::now();
            }
        }
        Ok(Summary {
            best_val_loss: best,
            final_val_loss: self.validation_loss(model, settings.final_eval_batches)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::Cpu;
    use crate::data::Dataset;
    use crate::io::load_parameters;
    use crate::model::GptConfig;
    use crate::nn::{Architecture, Module};
    use std::fs;

    type B = Cpu<f32>;

    const CONFIG: GptConfig = GptConfig {
        vocab: 11,
        context: 8,
        layers: 1,
        heads: 2,
        embed: 16,
        architecture: Architecture::MODERN,
    };

    /// Un corpus avec une régularité à apprendre : une période de 7 tokens.
    fn dataset() -> Dataset {
        Dataset::from_tokens((0..600u32).map(|i| (i * 3 % 7) as u16).collect())
    }

    /// Des réglages à l'échelle d'un test.
    fn settings() -> Settings {
        Settings {
            steps: 6,
            batch_size: 4,
            learning_rate: 1e-2,
            warmup_steps: 2,
            eval_every: 3,
            eval_batches: 2,
            final_eval_batches: 4,
            ..Settings::default()
        }
    }

    /// Un dossier vide, propre à un test.
    fn directory(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("smallem-trainer-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn trainer<'a>(
        backend: &'a B,
        data: &'a Dataset,
        settings: Settings,
        dir: &std::path::Path,
    ) -> Trainer<'a, B, Dataset> {
        Trainer {
            backend,
            settings,
            train: data,
            validation: data,
            model_path: dir.join("model.safetensors"),
            extra_settings: BTreeMap::from([("run.config".to_string(), "test".to_string())]),
        }
    }

    fn model(backend: &B, seed: u64) -> Gpt<B> {
        Gpt::new(backend, CONFIG, &mut StdRng::seed_from_u64(seed))
    }

    fn values(model: &Gpt<B>) -> Vec<f32> {
        model
            .parameters()
            .iter()
            .flat_map(|p| p.value().data().to_vec())
            .collect()
    }

    /// Mène un entraînement et renvoie ses évaluations.
    fn evaluations(
        trainer: &Trainer<'_, B, Dataset>,
        model: &mut Gpt<B>,
        resume: bool,
    ) -> (Summary, Vec<Evaluation>) {
        let mut evaluations = Vec::new();
        let summary = trainer
            .run(model, resume, |event| {
                if let Event::Evaluated(evaluation) = event {
                    evaluations.push(evaluation);
                }
            })
            .unwrap();
        (summary, evaluations)
    }

    #[test]
    fn evaluations_come_at_the_chosen_pace_and_at_the_last_step() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("pace"));
        let settings = Settings {
            steps: 7,
            accumulate: 2,
            ..settings()
        };
        let trainer = trainer(&backend, &data, settings, &dir);
        let mut model = model(&backend, 0);
        let mut started = None;
        let mut evaluations = Vec::new();
        let summary = trainer
            .run(&mut model, false, |event| match event {
                Event::Started { val_loss } => started = Some(val_loss),
                Event::Evaluated(evaluation) => evaluations.push(evaluation),
                Event::Resumed { .. } => panic!("rien n'a été repris"),
            })
            .unwrap();

        let steps: Vec<_> = evaluations.iter().map(|e| (e.step, e.steps)).collect();
        assert_eq!(steps, vec![(3, 3), (6, 3), (7, 1)]);
        // La perte initiale est celle d'un tirage uniforme, et elle baisse.
        let initial = started.expect("perte initiale rapportée");
        assert!((initial - 11f64.ln()).abs() < 0.1, "{initial}");
        let last = evaluations.last().unwrap();
        assert!(
            last.val_loss < initial - 0.2,
            "{} → {}",
            initial,
            last.val_loss
        );
        assert!(evaluations.iter().all(|e| e.grad_norm_max > 0.0));
        // Le taux rapporté est celui du dernier pas : la fin de la descente.
        assert_eq!(last.learning_rate, trainer.settings.learning_rate_at(6));
        let best = evaluations
            .iter()
            .map(|e| e.val_loss)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(summary.best_val_loss, best);
        assert!(summary.final_val_loss.is_finite());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_saved_model_is_the_best_one() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("best"));
        let trainer = trainer(&backend, &data, settings(), &dir);
        let mut trained = model(&backend, 0);
        let (summary, evaluations) = evaluations(&trainer, &mut trained, false);
        assert!(evaluations.iter().any(|e| e.improved));

        let file = SafeTensors::read(&trainer.model_path).unwrap();
        assert_eq!(GptConfig::from_metadata(&file.metadata).unwrap(), CONFIG);
        let mut saved = model(&backend, 9);
        load_parameters(&file, &backend, &mut saved).unwrap();
        let loss = trainer.validation_loss(&saved, 2).unwrap();
        assert!((loss - summary.best_val_loss).abs() < 1e-6, "{loss}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn validation_always_sees_the_same_batches() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("validation"));
        let model = model(&backend, 0);
        let first = trainer(&backend, &data, settings(), &dir);
        // Une autre graine d'entraînement ne change pas les lots évalués.
        let other_seed = Settings {
            seed: 7,
            ..settings()
        };
        let second = trainer(&backend, &data, other_seed, &dir);
        let loss = first.validation_loss(&model, 3).unwrap();
        assert_eq!(loss, first.validation_loss(&model, 3).unwrap());
        assert_eq!(loss, second.validation_loss(&model, 3).unwrap());
        assert_ne!(loss, first.validation_loss(&model, 4).unwrap());
    }

    #[test]
    fn a_resumed_training_ends_with_the_same_model() {
        let (backend, data) = (Cpu::new(), dataset());
        let (whole_dir, resumed_dir) = (directory("whole"), directory("resumed"));

        // Un entraînement d'une traite. Son point de reprise du pas 3 est
        // mis de côté au moment où il est écrit.
        let whole = trainer(&backend, &data, settings(), &whole_dir);
        let resumed = trainer(&backend, &data, settings(), &resumed_dir);
        let mut straight = model(&backend, 0);
        let mut reference = Vec::new();
        whole
            .run(&mut straight, false, |event| {
                if let Event::Evaluated(evaluation) = event {
                    if evaluation.step == 3 {
                        fs::copy(whole.checkpoint_path(), resumed.checkpoint_path()).unwrap();
                    }
                    reference.push(evaluation);
                }
            })
            .unwrap();

        // La reprise part d'un modèle quelconque : tout vient du fichier.
        let mut restarted = model(&backend, 5);
        let mut resumed_at = None;
        let mut evaluations = Vec::new();
        resumed
            .run(&mut restarted, true, |event| match event {
                Event::Resumed {
                    step, unverified, ..
                } => {
                    assert!(unverified.is_empty(), "{unverified:?}");
                    resumed_at = Some(step);
                }
                Event::Evaluated(evaluation) => evaluations.push(evaluation),
                Event::Started { .. } => panic!("la reprise ne repart pas de zéro"),
            })
            .unwrap();

        assert_eq!(resumed_at, Some(3));
        assert_eq!(evaluations.len(), 1);
        let (last, expected) = (&evaluations[0], &reference[1]);
        assert_eq!((last.step, last.steps), (6, 3));
        assert_eq!(last.train_loss, expected.train_loss);
        assert_eq!(last.val_loss, expected.val_loss);
        assert_eq!(values(&restarted), values(&straight));
        fs::remove_dir_all(whole_dir).unwrap();
        fs::remove_dir_all(resumed_dir).unwrap();
    }

    /// La configuration des tests, avec un mélange d'experts équilibré par
    /// ses biais.
    fn experts_config() -> GptConfig {
        use crate::nn::{Balance, ExpertDims};
        let dims = ExpertDims {
            shared: 1,
            routed: 4,
            active: 2,
            hidden: 8,
            balance: Balance::Bias { rate: 0.01 },
        };
        CONFIG.with_architecture(Architecture::MODERN.with_experts(dims))
    }

    #[test]
    fn expert_load_is_reported_and_biases_are_tuned() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("experts"));
        let trainer = trainer(&backend, &data, settings(), &dir);
        let mut model = Gpt::new(&backend, experts_config(), &mut StdRng::seed_from_u64(0));
        let mut loads = Vec::new();
        trainer
            .run(&mut model, false, |event| {
                if let Event::Evaluated(evaluation) = event {
                    loads.push(evaluation.expert_load);
                }
            })
            .unwrap();
        // Entre deux évaluations : 3 pas de 4 séquences de 8 tokens, deux
        // experts par token. Les lots de validation ne sont pas comptés.
        assert_eq!(loads.len(), 2);
        for load in &loads {
            assert_eq!(load.len(), 1);
            assert_eq!(load[0].iter().sum::<u64>(), 3 * 4 * 8 * 2);
        }
        // Les biais ont bougé, par crans de 0,01, et ils sont enregistrés
        // avec le modèle.
        let saved = SafeTensors::read(&trainer.model_path).unwrap();
        let mut reloaded = Gpt::new(&backend, experts_config(), &mut StdRng::seed_from_u64(9));
        load_parameters(&saved, &backend, &mut reloaded).unwrap();
        let bias = |model: &Gpt<B>| {
            let layer = model.experts().next().unwrap();
            layer.routing_bias.value().data().to_vec()
        };
        assert!(bias(&model).iter().any(|&b| b != 0.0), "{:?}", bias(&model));
        for value in bias(&model) {
            let steps = f64::from(value) / 0.01;
            assert!((steps - steps.round()).abs() < 1e-3 && steps.abs() <= 6.5);
        }
        assert_eq!(bias(&model), bias(&reloaded));
    }

    #[test]
    fn a_resumed_training_with_experts_ends_with_the_same_model() {
        // Les biais d'aiguillage ne sont pas dans l'optimiseur : ils doivent
        // revenir du point de reprise comme les autres paramètres.
        let (backend, data) = (Cpu::new(), dataset());
        let (whole_dir, resumed_dir) = (directory("experts-whole"), directory("experts-resumed"));
        let whole = trainer(&backend, &data, settings(), &whole_dir);
        let resumed = trainer(&backend, &data, settings(), &resumed_dir);
        let build = |seed| Gpt::new(&backend, experts_config(), &mut StdRng::seed_from_u64(seed));
        let mut straight = build(0);
        whole
            .run(&mut straight, false, |event| {
                if let Event::Evaluated(evaluation) = event
                    && evaluation.step == 3
                {
                    fs::copy(whole.checkpoint_path(), resumed.checkpoint_path()).unwrap();
                }
            })
            .unwrap();
        let mut restarted = build(5);
        resumed.run(&mut restarted, true, |_| {}).unwrap();
        assert_eq!(values(&restarted), values(&straight));
    }

    /// Chaque variante de Muon a son état : pas périodiques comptés,
    /// moyennes par neurone, avancement de l'entraînement. Toutes doivent
    /// reprendre à l'identique.
    #[test]
    fn a_resumed_training_with_a_muon_variant_ends_with_the_same_model() {
        let (backend, data) = (Cpu::new(), dataset());
        for (index, name) in ["muon:short,period=2", "muon:norm", "muon:boost=3"]
            .into_iter()
            .enumerate()
        {
            let settings = Settings {
                optimizer: name.parse().unwrap(),
                ..settings()
            };
            let whole_dir = directory(&format!("variant-whole-{index}"));
            let resumed_dir = directory(&format!("variant-resumed-{index}"));
            let whole = trainer(&backend, &data, settings.clone(), &whole_dir);
            let resumed = trainer(&backend, &data, settings, &resumed_dir);
            let mut straight = model(&backend, 0);
            whole
                .run(&mut straight, false, |event| {
                    if let Event::Evaluated(evaluation) = event
                        && evaluation.step == 3
                    {
                        fs::copy(whole.checkpoint_path(), resumed.checkpoint_path()).unwrap();
                    }
                })
                .unwrap();
            let mut restarted = model(&backend, 5);
            resumed.run(&mut restarted, true, |_| {}).unwrap();
            assert_eq!(values(&restarted), values(&straight), "{name}");
        }
    }

    #[test]
    fn a_resumed_training_with_muon_ends_with_the_same_model() {
        let (backend, data) = (Cpu::new(), dataset());
        let (whole_dir, resumed_dir) = (directory("muon-whole"), directory("muon-resumed"));
        let settings = Settings {
            optimizer: OptimizerKind::Muon(Default::default()),
            ..settings()
        };
        let whole = trainer(&backend, &data, settings.clone(), &whole_dir);
        let resumed = trainer(&backend, &data, settings, &resumed_dir);
        let mut straight = model(&backend, 0);
        let mut losses = Vec::new();
        whole
            .run(&mut straight, false, |event| {
                if let Event::Evaluated(evaluation) = event {
                    if evaluation.step == 3 {
                        fs::copy(whole.checkpoint_path(), resumed.checkpoint_path()).unwrap();
                    }
                    losses.push(evaluation.val_loss);
                }
            })
            .unwrap();
        // Muon apprend : la perte de validation baisse.
        assert!(losses[1] < losses[0], "{losses:?}");
        let mut restarted = model(&backend, 5);
        resumed.run(&mut restarted, true, |_| {}).unwrap();
        assert_eq!(values(&restarted), values(&straight));

        // Un point de reprise d'AdamW ne se reprend pas avec Muon.
        let adamw_dir = directory("muon-from-adamw");
        let adamw = trainer(&backend, &data, self::settings(), &adamw_dir);
        adamw.run(&mut model(&backend, 0), false, |_| {}).unwrap();
        let muon = Settings {
            optimizer: OptimizerKind::Muon(Default::default()),
            steps: 9,
            ..self::settings()
        };
        let error = trainer(&backend, &data, muon, &adamw_dir)
            .run(&mut model(&backend, 0), true, |_| {})
            .unwrap_err();
        assert!(error.to_string().contains("run.optimizer"), "{error}");
    }

    #[test]
    fn a_resume_with_other_settings_is_refused() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("refused"));
        let settings = Settings {
            steps: 3,
            ..settings()
        };
        let trained = trainer(&backend, &data, settings.clone(), &dir);
        trained.run(&mut model(&backend, 0), false, |_| {}).unwrap();

        let refusal = |trainer: &Trainer<'_, B, Dataset>| {
            let mut model = model(&backend, 1);
            let before = values(&model);
            let error = trainer.run(&mut model, true, |_| {}).unwrap_err();
            // Un refus ne touche pas au modèle.
            assert_eq!(values(&model), before);
            error.to_string()
        };
        for (changed, key) in [
            (
                Settings {
                    learning_rate: 5e-3,
                    ..settings.clone()
                },
                "run.lr",
            ),
            (
                Settings {
                    weight_decay: 0.0,
                    ..settings.clone()
                },
                "run.weight_decay",
            ),
            (
                Settings {
                    max_grad_norm: 0.5,
                    ..settings.clone()
                },
                "run.max_grad_norm",
            ),
            (
                Settings {
                    steps: 9,
                    ..settings.clone()
                },
                "run.steps",
            ),
        ] {
            let error = refusal(&trainer(&backend, &data, changed, &dir));
            assert!(
                error.contains(key) && error.contains("enregistré"),
                "{error}"
            );
        }
        // Un réglage de l'appelant est vérifié comme les autres.
        let mut other = trainer(&backend, &data, settings.clone(), &dir);
        other
            .extra_settings
            .insert("run.config".into(), "autre".into());
        assert!(refusal(&other).contains("run.config"));
        // Les mêmes réglages : l'entraînement est simplement fini.
        let error = refusal(&trained);
        assert!(error.contains("déjà terminé"), "{error}");
        // Le rythme des évaluations est libre.
        let paced = Settings {
            eval_every: 1,
            ..settings
        };
        assert!(refusal(&trainer(&backend, &data, paced, &dir)).contains("déjà terminé"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_resume_without_checkpoint_is_an_error() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("missing"));
        let trainer = trainer(&backend, &data, settings(), &dir);
        assert!(trainer.run(&mut model(&backend, 0), true, |_| {}).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn training_stops_when_the_loss_is_not_a_number() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("nan"));
        // Un taux infini : le premier pas détruit les paramètres.
        let settings = Settings {
            learning_rate: f64::INFINITY,
            eval_every: 5,
            steps: 10,
            ..settings()
        };
        let trainer = trainer(&backend, &data, settings, &dir);
        let mut evaluated = 0;
        let error = trainer
            .run(&mut model(&backend, 0), false, |event| {
                evaluated += usize::from(matches!(event, Event::Evaluated(_)));
            })
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("pas 2 :") && error.contains("arrêté"),
            "{error}"
        );
        // Rien n'a été écrit : ni modèle, ni point de reprise.
        assert_eq!(evaluated, 0);
        assert!(!trainer.checkpoint_path().exists() && !trainer.model_path.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_bad_evaluation_keeps_the_last_checkpoint() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("kept"));
        let trainer = trainer(&backend, &data, settings(), &dir);
        trainer.run(&mut model(&backend, 0), false, |_| {}).unwrap();
        let before = fs::read(trainer.checkpoint_path()).unwrap();

        // Le même dossier, avec un taux qui fait tout diverger.
        let diverging = Settings {
            learning_rate: f64::INFINITY,
            ..settings()
        };
        let trainer = self::trainer(&backend, &data, diverging, &dir);
        assert!(trainer.run(&mut model(&backend, 0), false, |_| {}).is_err());
        assert_eq!(fs::read(trainer.checkpoint_path()).unwrap(), before);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_is_refused_where_it_would_divide_or_do_nothing() {
        let (backend, data, dir) = (Cpu::new(), dataset(), directory("zero"));
        let zero = Settings {
            eval_every: 0,
            ..settings()
        };
        let trainer = trainer(&backend, &data, zero, &dir);
        let error = trainer
            .run(&mut model(&backend, 0), false, |_| {})
            .unwrap_err();
        assert!(error.to_string().contains("eval_every"), "{error}");
    }

    #[test]
    fn learning_rate_rises_then_falls_to_its_floor() {
        let settings = Settings {
            steps: 100,
            warmup_steps: 10,
            learning_rate: 1e-3,
            floor_ratio: 0.1,
            ..Settings::default()
        };
        assert!(settings.learning_rate_at(0) < settings.learning_rate_at(5));
        assert_eq!(settings.learning_rate_at(9), 1e-3);
        assert!(settings.learning_rate_at(50) < 1e-3);
        assert!((settings.learning_rate_at(100) - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn every_setting_that_shapes_the_steps_is_recorded() {
        let saved = Settings::default().metadata();
        let changes: [fn(&mut Settings); 10] = [
            |s| s.optimizer = OptimizerKind::Muon(Default::default()),
            |s| s.steps = 9,
            |s| s.batch_size = 8,
            |s| s.accumulate = 2,
            |s| s.learning_rate = 5e-4,
            |s| s.warmup_steps = 3,
            |s| s.floor_ratio = 0.5,
            |s| s.weight_decay = 0.0,
            |s| s.max_grad_norm = 2.0,
            |s| s.seed = 1,
        ];
        assert_eq!(changes.len(), saved.len());
        for (index, change) in changes.iter().enumerate() {
            let mut settings = Settings::default();
            change(&mut settings);
            let error = check_resume(&saved, &settings.metadata()).expect_err("changement");
            assert!(error.contains("enregistré"), "changement {index} : {error}");
        }
        // Ce qui ne change pas la suite du calcul reste libre.
        let paced = Settings {
            eval_every: 7,
            eval_batches: 3,
            final_eval_batches: 5,
            ..Settings::default()
        };
        assert_eq!(check_resume(&saved, &paced.metadata()), Ok(Vec::new()));
    }

    #[test]
    fn resume_reports_settings_missing_from_an_old_checkpoint() {
        let current = Settings::default().metadata();
        let mut saved = current.clone();
        saved.remove("run.seed");
        assert_eq!(
            check_resume(&saved, &current),
            Ok(vec!["run.seed".to_string()])
        );
    }

    #[test]
    fn batches_of_a_step_do_not_depend_on_the_steps_before() {
        use rand::RngExt;
        let draw = |step| step_rng(42, step).random::<u64>();
        assert_eq!(draw(3000), draw(3000));
        assert_ne!(draw(3000), draw(3001));
        assert_ne!(
            step_rng(42, 0).random::<u64>(),
            step_rng(43, 0).random::<u64>()
        );
    }
}
