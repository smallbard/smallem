//! Tirage d'un token selon les probabilités données par un modèle.

use rand::{Rng, RngExt};

/// Tire un indice selon une distribution de probabilités : on tire un nombre
/// dans `[0, 1)` et on avance dans les probabilités cumulées jusqu'à le
/// dépasser.
pub fn sample(rng: &mut impl Rng, probabilities: &[f64]) -> usize {
    let mut remaining: f64 = rng.random();
    for (index, &probability) in probabilities.iter().enumerate() {
        if remaining < probability {
            return index;
        }
        remaining -= probability;
    }
    // Les erreurs d'arrondi peuvent laisser un reste infime après la boucle.
    probabilities.len() - 1
}

/// Réglages du tirage d'un token à partir des logits du modèle.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// Divise les logits avant le softmax. En dessous de 1, la distribution
    /// se resserre sur les tokens probables et le texte devient plus sûr et
    /// plus répétitif ; au-dessus, elle s'aplatit et le texte devient plus
    /// varié et plus décousu. À 0, on prend toujours le token le plus
    /// probable.
    pub temperature: f64,
    /// Ne garde que les `k` tokens les plus probables.
    pub top_k: Option<usize>,
    /// Ne garde que le plus petit ensemble de tokens dont la probabilité
    /// cumulée atteint `p`. Contrairement à `top_k`, le nombre de tokens
    /// gardés s'adapte : peu quand le modèle est sûr de lui, beaucoup quand
    /// il hésite.
    pub top_p: Option<f64>,
}

impl Default for Sampling {
    /// Le tirage fidèle à la distribution du modèle.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}

/// Indice du plus grand élément ; le premier, en cas d'égalité, comme
/// pour le tirage glouton.
pub(crate) fn argmax<T: PartialOrd>(values: &[T]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Transforme des logits en probabilités de tirage : température, puis
/// softmax, puis filtres `top_k` et `top_p`, puis renormalisation.
pub fn probabilities(logits: &[f64], sampling: &Sampling) -> Vec<f64> {
    if sampling.temperature <= 0.0 {
        let mut result = vec![0.0; logits.len()];
        result[argmax(logits)] = 1.0;
        return result;
    }

    // Softmax de `logits / température`, en soustrayant le maximum.
    let max = logits[argmax(logits)];
    let mut result: Vec<f64> = logits
        .iter()
        .map(|&logit| ((logit - max) / sampling.temperature).exp())
        .collect();
    let total: f64 = result.iter().sum();
    for probability in &mut result {
        *probability /= total;
    }

    if sampling.top_k.is_none() && sampling.top_p.is_none() {
        return result;
    }

    // Les indices, du token le plus probable au moins probable.
    let mut order: Vec<usize> = (0..result.len()).collect();
    order.sort_by(|&a, &b| result[b].total_cmp(&result[a]));

    let mut kept = order.len();
    if let Some(top_k) = sampling.top_k {
        kept = kept.min(top_k.max(1));
    }
    if let Some(top_p) = sampling.top_p {
        let mut cumulative = 0.0;
        for (rank, &index) in order.iter().enumerate() {
            cumulative += result[index];
            if cumulative >= top_p {
                kept = kept.min(rank + 1);
                break;
            }
        }
    }
    for &index in &order[kept..] {
        result[index] = 0.0;
    }

    let total: f64 = result.iter().sum();
    for probability in &mut result {
        *probability /= total;
    }
    result
}

/// Tire un token à partir des logits du modèle.
pub(crate) fn sample_logits(rng: &mut impl Rng, logits: &[f64], sampling: &Sampling) -> usize {
    sample(rng, &probabilities(logits, sampling))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn certain_outcome_is_always_drawn() {
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            assert_eq!(sample(&mut rng, &[0.0, 1.0, 0.0]), 1);
        }
    }

    #[test]
    fn frequencies_follow_probabilities() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut counts = [0usize; 3];
        for _ in 0..100_000 {
            counts[sample(&mut rng, &[0.2, 0.5, 0.3])] += 1;
        }
        for (count, expected) in counts.iter().zip([0.2, 0.5, 0.3]) {
            let frequency = *count as f64 / 100_000.0;
            assert!(
                (frequency - expected).abs() < 0.01,
                "{frequency} != {expected}"
            );
        }
    }

    fn close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    /// Logits dont le softmax vaut exactement `[0,1 ; 0,2 ; 0,3 ; 0,4]`.
    fn logits() -> Vec<f64> {
        [0.1f64, 0.2, 0.3, 0.4].iter().map(|p| p.ln()).collect()
    }

    #[test]
    fn default_sampling_is_the_plain_softmax() {
        close(
            &probabilities(&logits(), &Sampling::default()),
            &[0.1, 0.2, 0.3, 0.4],
        );
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let sampling = Sampling {
            temperature: 0.0,
            ..Default::default()
        };
        close(&probabilities(&logits(), &sampling), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn low_temperature_sharpens_and_high_temperature_flattens() {
        let at = |temperature| {
            probabilities(
                &logits(),
                &Sampling {
                    temperature,
                    ..Default::default()
                },
            )
        };
        // p^(1/T), renormalisé : à T = 0,5 les probabilités sont au carré.
        close(&at(0.5), &[1.0 / 30.0, 4.0 / 30.0, 9.0 / 30.0, 16.0 / 30.0]);
        assert!(at(0.5)[3] > 0.4 && at(2.0)[3] < 0.4);
        assert!(at(100.0).iter().all(|p| (p - 0.25).abs() < 0.01));
    }

    #[test]
    fn top_k_keeps_the_most_likely_tokens() {
        let sampling = Sampling {
            top_k: Some(2),
            ..Default::default()
        };
        close(
            &probabilities(&logits(), &sampling),
            &[0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0],
        );
    }

    #[test]
    fn top_p_keeps_the_smallest_sufficient_set() {
        let at = |top_p| {
            probabilities(
                &logits(),
                &Sampling {
                    top_p: Some(top_p),
                    ..Default::default()
                },
            )
        };
        // 0,4 suffit pour 0,3 ; il faut 0,4 + 0,3 pour 0,5.
        close(&at(0.3), &[0.0, 0.0, 0.0, 1.0]);
        close(&at(0.5), &[0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0]);
        close(&at(1.0), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn filters_combine() {
        // top_k en garde 3, top_p n'en demande que 2 : le plus strict gagne.
        let sampling = Sampling {
            temperature: 1.0,
            top_k: Some(3),
            top_p: Some(0.5),
        };
        close(
            &probabilities(&logits(), &sampling),
            &[0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0],
        );
    }

    #[test]
    fn filtered_tokens_are_never_drawn() {
        let mut rng = StdRng::seed_from_u64(0);
        let sampling = Sampling {
            top_k: Some(2),
            ..Default::default()
        };
        for _ in 0..1000 {
            assert!(sample_logits(&mut rng, &logits(), &sampling) >= 2);
        }
    }

    #[test]
    fn huge_logits_do_not_overflow() {
        let result = probabilities(&[1000.0, 1000.0, -1000.0], &Sampling::default());
        close(&result, &[0.5, 0.5, 0.0]);
    }
}
