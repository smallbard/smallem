//! Un pas d'entraînement, indépendant du modèle et de l'optimiseur.

use crate::autograd::{Gradients, Tape, Var};
use crate::backend::{Backend, Float};
use crate::nn::Module;
use crate::optim::{Optimizer, gradient_norm, tensors_norm};

/// Fait un pas d'entraînement et renvoie la perte d'avant la mise à jour.
///
/// `loss` construit la perte du modèle sur la tape qu'on lui donne. La perte
/// est renvoyée sous forme de tenseur, sans être lue : sur GPU, la lire
/// rapatrie une valeur sur l'hôte, ce qui n'est utile que de temps en temps,
/// pour l'affichage.
pub fn train_step<B: Backend, M: Module<B>>(
    backend: &B,
    model: &mut M,
    optimizer: &mut impl Optimizer<B>,
    loss: impl FnOnce(&M, &Tape<B>) -> Var,
) -> B::Tensor {
    let (loss_value, tape_id, grads) = loss_and_gradients(backend, model, loss);
    // Les gradients sont recalculés de zéro à chaque pas : il n'y a pas de
    // « remise à zéro » à faire, la tape et ses gradients sont jetés.
    optimizer.step(backend, model.parameters_mut(), tape_id, &grads, 1.0);
    loss_value
}

/// Comme [`train_step`], en écrêtant la norme du gradient à `max_norm`.
/// Renvoie aussi la norme mesurée avant écrêtage.
///
/// De temps en temps, un lot inhabituel produit un gradient énorme, et le
/// pas qui en découle détruit ce que le modèle a appris. Ramener la norme à
/// `max_norm` garde la direction du gradient mais borne la taille du pas.
///
/// Mesurer la norme oblige à lire un nombre sur le GPU, donc à le
/// synchroniser à chaque pas.
///
/// Si la norme n'est pas un nombre fini, le modèle n'est pas modifié.
pub fn train_step_clipped<B: Backend, M: Module<B>>(
    backend: &B,
    model: &mut M,
    optimizer: &mut impl Optimizer<B>,
    max_norm: f64,
    loss: impl FnOnce(&M, &Tape<B>) -> Var,
) -> (B::Tensor, f64) {
    let (loss_value, tape_id, grads) = loss_and_gradients(backend, model, loss);
    let norm = gradient_norm(backend, &model.parameters(), tape_id, &grads);
    if let Some(grad_scale) = clip_scale(norm, max_norm) {
        optimizer.step(backend, model.parameters_mut(), tape_id, &grads, grad_scale);
    }
    (loss_value, norm)
}

/// Facteur à appliquer au gradient pour ramener sa norme à `max_norm` au
/// plus, ou `None` si la norme n'est pas un nombre fini.
///
/// Un gradient infini ou NaN ne doit jamais atteindre l'optimiseur : en un
/// pas, il rendrait NaN tous les paramètres et tous les moments d'Adam,
/// sans retour possible. Le pas est alors sauté, et c'est à l'appelant de
/// réagir en lisant la norme renvoyée.
fn clip_scale(norm: f64, max_norm: f64) -> Option<f64> {
    if !norm.is_finite() {
        None
    } else if norm > max_norm {
        Some(max_norm / norm)
    } else {
        Some(1.0)
    }
}

/// Un pas d'entraînement sur `micro_batches` lots traités l'un après
/// l'autre : leurs gradients sont additionnés, et la mise à jour se fait une
/// seule fois, avec leur moyenne. Renvoie la perte moyenne et la norme du
/// gradient moyen avant écrêtage.
///
/// C'est le moyen d'entraîner avec un grand lot quand il ne tient pas dans
/// la mémoire du GPU : comme la perte est une moyenne sur les exemples, la
/// moyenne des gradients de `n` lots de taille `B` est le gradient d'un lot
/// de taille `n × B`. La mémoire occupée reste celle d'un seul petit lot.
///
/// `loss` reçoit le rang du lot, de 0 à `micro_batches - 1`.
///
/// Si la norme n'est pas un nombre fini, le modèle n'est pas modifié.
pub fn train_step_accumulated<B: Backend, M: Module<B>>(
    backend: &B,
    model: &mut M,
    optimizer: &mut impl Optimizer<B>,
    max_norm: f64,
    micro_batches: usize,
    mut loss: impl FnMut(&M, &Tape<B>, usize) -> Var,
) -> (B::Tensor, f64) {
    assert!(micro_batches > 0, "il faut au moins un lot");
    let mut grad_sums: Vec<Option<B::Tensor>> = vec![None; model.parameters().len()];
    let mut loss_sum: Option<B::Tensor> = None;
    for index in 0..micro_batches {
        let (loss_value, tape_id, grads) =
            loss_and_gradients(backend, model, |m, t| loss(m, t, index));
        for (sum, parameter) in grad_sums.iter_mut().zip(model.parameters()) {
            let Some(grad) = parameter.grad(tape_id, &grads) else {
                continue;
            };
            *sum = Some(match sum.take() {
                Some(sum) => backend.add(&sum, grad),
                None => grad.clone(),
            });
        }
        loss_sum = Some(match loss_sum {
            Some(sum) => backend.add(&sum, &loss_value),
            None => loss_value,
        });
    }

    // Diviser la somme par le nombre de lots donne la moyenne. La division
    // n'est pas faite tenseur par tenseur : elle passe par le facteur que
    // l'optimiseur applique déjà à chaque gradient.
    let mean = 1.0 / micro_batches as f64;
    let grad_sums: Vec<Option<&B::Tensor>> = grad_sums.iter().map(Option::as_ref).collect();
    let norm = tensors_norm(backend, &grad_sums) * mean;
    if let Some(clip) = clip_scale(norm, max_norm) {
        optimizer.apply(backend, model.parameters_mut(), &grad_sums, clip * mean);
    }

    let loss_sum = loss_sum.expect("au moins un lot");
    let mean_loss = backend.mul(&loss_sum, &backend.scalar(B::Scalar::from_f64(mean)));
    (mean_loss, norm)
}

/// Passe avant puis rétropropagation, sur une tape qui est jetée ensuite.
fn loss_and_gradients<B: Backend, M>(
    backend: &B,
    model: &M,
    loss: impl FnOnce(&M, &Tape<B>) -> Var,
) -> (B::Tensor, u64, Gradients<B>) {
    // La tape référence les paramètres : elle doit avoir disparu quand
    // l'optimiseur les modifie sur place.
    let tape = Tape::new(backend);
    let loss = loss(model, &tape);
    let grads = tape.backward(loss);
    (tape.value(loss).clone(), tape.id(), grads)
}

/// Calcule une perte sans rien apprendre : pas de rétropropagation, pas de
/// mise à jour.
pub fn evaluate<B: Backend, M>(
    backend: &B,
    model: &M,
    loss: impl FnOnce(&M, &Tape<B>) -> Var,
) -> B::Scalar {
    let tape = Tape::new(backend);
    let loss = loss(model, &tape);
    backend.item(&tape.value(loss))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Primitives;
    use crate::backend::cpu::Cpu;
    use crate::model::{Gpt, GptConfig};
    use crate::nn::Architecture;
    use crate::optim::AdamW;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Cpu<f64>;

    const TINY: GptConfig = GptConfig {
        vocab: 11,
        context: 4,
        layers: 1,
        heads: 2,
        embed: 8,
        architecture: Architecture::GPT2,
    };

    /// Quatre séquences de quatre tokens et leurs cibles.
    const TOKENS: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 3, 5, 7, 9];
    const TARGETS: [u32; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 5, 7, 9, 0];

    fn model(backend: &B) -> Gpt<B> {
        Gpt::new(backend, TINY, &mut StdRng::seed_from_u64(0))
    }

    fn values(model: &Gpt<B>) -> Vec<f64> {
        model
            .parameters()
            .iter()
            .flat_map(|p| p.value().data().to_vec())
            .collect()
    }

    /// Deux pas d'entraînement : en un lot de quatre séquences, ou en
    /// `micro_batches` lots égaux accumulés.
    fn two_steps(micro_batches: usize, max_norm: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let backend = Cpu::new();
        let mut model = model(&backend);
        let mut optimizer = AdamW::new(1e-2, 0.1);
        let per_batch = 16 / micro_batches;
        let (mut losses, mut norms) = (Vec::new(), Vec::new());
        for _ in 0..2 {
            let (loss, norm) = train_step_accumulated(
                &backend,
                &mut model,
                &mut optimizer,
                max_norm,
                micro_batches,
                |m, t, index| {
                    let range = index * per_batch..(index + 1) * per_batch;
                    m.loss(t, &TOKENS[range.clone()], &TARGETS[range], per_batch / 4)
                },
            );
            losses.push(backend.item(&loss));
            norms.push(norm);
        }
        (values(&model), losses, norms)
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-10, "{a} != {e}");
        }
    }

    #[test]
    fn accumulated_small_batches_equal_one_large_batch() {
        // Sans écrêtage (norme maximale énorme), puis avec.
        for max_norm in [1e9, 0.05] {
            let (whole, whole_losses, whole_norms) = two_steps(1, max_norm);
            for micro_batches in [2, 4] {
                let (split, losses, norms) = two_steps(micro_batches, max_norm);
                assert_close(&split, &whole);
                assert_close(&losses, &whole_losses);
                assert_close(&norms, &whole_norms);
            }
            assert!(whole_norms[0] > 0.05, "l'écrêtage n'est pas exercé");
        }
    }

    #[test]
    fn one_accumulated_batch_is_a_clipped_step() {
        let backend = Cpu::new();
        let mut reference = model(&backend);
        let mut optimizer = AdamW::new(1e-2, 0.1);
        let (loss, norm) =
            train_step_clipped(&backend, &mut reference, &mut optimizer, 0.05, |m, t| {
                m.loss(t, &TOKENS, &TARGETS, 4)
            });

        let mut accumulated = model(&backend);
        let mut optimizer = AdamW::new(1e-2, 0.1);
        let (same_loss, same_norm) = train_step_accumulated(
            &backend,
            &mut accumulated,
            &mut optimizer,
            0.05,
            1,
            |m, t, _| m.loss(t, &TOKENS, &TARGETS, 4),
        );
        assert_close(&values(&accumulated), &values(&reference));
        assert_close(&[backend.item(&same_loss)], &[backend.item(&loss)]);
        assert_close(&[same_norm], &[norm]);
    }

    #[test]
    fn non_finite_gradient_leaves_the_model_untouched() {
        // `ln` d'un nombre négatif donne NaN. En multipliant la perte par
        // ce NaN, tous les gradients le deviennent aussi.
        let backend = Cpu::new();
        let poisoned = |m: &Gpt<B>, t: &Tape<B>| {
            let loss = m.loss(t, &TOKENS, &TARGETS, 4);
            t.mul(loss, t.ln(t.scalar(-1.0)))
        };

        let mut clipped = model(&backend);
        let before = values(&clipped);
        let mut optimizer = AdamW::new(1e-2, 0.1);
        let (loss, norm) =
            train_step_clipped(&backend, &mut clipped, &mut optimizer, 1.0, poisoned);
        assert!(backend.item(&loss).is_nan() && norm.is_nan());
        assert_eq!(values(&clipped), before);
        // L'optimiseur n'a pas compté ce pas.
        assert_eq!(optimizer.steps(), 0);

        let mut accumulated = model(&backend);
        let (_, norm) = train_step_accumulated(
            &backend,
            &mut accumulated,
            &mut AdamW::new(1e-2, 0.1),
            1.0,
            2,
            |m, t, index| {
                if index == 1 {
                    poisoned(m, t)
                } else {
                    m.loss(t, &TOKENS, &TARGETS, 4)
                }
            },
        );
        assert!(norm.is_nan());
        assert_eq!(values(&accumulated), before);
    }

    #[test]
    fn clip_scale_bounds_the_norm() {
        assert_eq!(clip_scale(0.5, 1.0), Some(1.0));
        assert_eq!(clip_scale(4.0, 1.0), Some(0.25));
        assert_eq!(clip_scale(f64::NAN, 1.0), None);
        assert_eq!(clip_scale(f64::INFINITY, 1.0), None);
    }

    #[test]
    fn accumulation_calls_the_loss_once_per_batch_in_order() {
        let backend = Cpu::new();
        let mut model = model(&backend);
        let mut seen = Vec::new();
        train_step_accumulated(
            &backend,
            &mut model,
            &mut AdamW::new(1e-2, 0.0),
            1.0,
            3,
            |m, t, index| {
                seen.push(index);
                m.loss(t, &TOKENS[..4], &TARGETS[..4], 1)
            },
        );
        assert_eq!(seen, [0, 1, 2]);
    }
}
