//! Ajuster des lois d'échelle : des paraboles, des lois de puissance, et la
//! formule de Chinchilla pour la perte.

use rand::{Rng, RngExt};

/// La parabole `y = a·x² + b·x + c` des moindres carrés sur des points
/// `(x, y)`.
pub fn fit_parabola(points: &[(f64, f64)]) -> (f64, f64, f64) {
    assert!(points.len() >= 3, "trois points au moins pour une parabole");
    // Les équations normales, `[Σx⁴ Σx³ Σx²; Σx³ Σx² Σx; Σx² Σx n]`.
    let mut m = [[0.0; 3]; 3];
    let mut v = [0.0; 3];
    for &(x, y) in points {
        let powers = [x * x, x, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += powers[i] * powers[j];
            }
            v[i] += powers[i] * y;
        }
    }
    let solution = solve3(m, v);
    (solution[0], solution[1], solution[2])
}

/// Résout un système 3 × 3 par la méthode de Gauss, avec pivot.
fn solve3(mut m: [[f64; 3]; 3], mut v: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .expect("une ligne");
        m.swap(col, pivot);
        v.swap(col, pivot);
        for row in col + 1..3 {
            let factor = m[row][col] / m[col][col];
            let pivot_row = m[col];
            for (value, pivot) in m[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot;
            }
            v[row] -= factor * v[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let rest: f64 = (row + 1..3).map(|k| m[row][k] * x[k]).sum();
        x[row] = (v[row] - rest) / m[row][row];
    }
    x
}

/// La droite des moindres carrés `y = pente · x + origine`.
pub fn fit_line(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let covariance: f64 = points.iter().map(|&(x, y)| (x - mx) * (y - my)).sum();
    let variance: f64 = points.iter().map(|&(x, _)| (x - mx).powi(2)).sum();
    let slope = covariance / variance;
    (slope, my - slope * mx)
}

/// Une loi de puissance `y = k · x^exposant`, ajustée en échelle
/// logarithmique : `(exposant, k)`.
pub fn fit_power_law(points: &[(f64, f64)]) -> (f64, f64) {
    let logs: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x.ln(), y.ln())).collect();
    let (slope, intercept) = fit_line(&logs);
    (slope, intercept.exp())
}

/// La formule de Chinchilla : `L(N, D) = E + A / N^α + B / D^β`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossLaw {
    pub e: f64,
    pub a: f64,
    pub alpha: f64,
    pub b: f64,
    pub beta: f64,
}

impl LossLaw {
    pub fn predict(&self, parameters: f64, tokens: f64) -> f64 {
        self.e + self.a / parameters.powf(self.alpha) + self.b / tokens.powf(self.beta)
    }

    /// La taille et les tokens qui minimisent la perte pour un calcul `c`,
    /// avec `c = flops_per_parameter_token × N × D`. Rend `(N, D)`.
    ///
    /// Le minimum de `A/N^α + B/D^β` sous `N · D = c / k` est, comme dans
    /// l'article, `N = G · (c/k)^(β/(α+β))` avec
    /// `G = (α·A / (β·B))^(1/(α+β))`.
    pub fn optimum(&self, compute: f64, flops_per_parameter_token: f64) -> (f64, f64) {
        let budget = compute / flops_per_parameter_token;
        let g = (self.alpha * self.a / (self.beta * self.b)).powf(1.0 / (self.alpha + self.beta));
        let parameters = g * budget.powf(self.beta / (self.alpha + self.beta));
        (parameters, budget / parameters)
    }
}

/// Un point d'ajustement : une taille, des tokens, la perte mesurée.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub parameters: f64,
    pub tokens: f64,
    pub loss: f64,
}

/// La perte de Huber de seuil `delta`.
fn huber(r: f64, delta: f64) -> (f64, f64) {
    if r.abs() <= delta {
        (0.5 * r * r, r)
    } else {
        (delta * (r.abs() - 0.5 * delta), delta * r.signum())
    }
}

/// L'objectif de Chinchilla et son gradient, pour les paramètres
/// `θ = (a, b, e, α, β)` où `A = eᵃ`, `B = eᵇ`, `E = eᵉ` :
///
/// ```text
/// Σ Huber( LSE(a − α ln N, b − β ln D, e) − ln L )
/// ```
fn objective(theta: &[f64; 5], runs: &[Run], delta: f64) -> (f64, [f64; 5]) {
    let [a, b, e, alpha, beta] = *theta;
    let mut value = 0.0;
    let mut grad = [0.0; 5];
    for run in runs {
        let (ln_n, ln_d) = (run.parameters.ln(), run.tokens.ln());
        let terms = [a - alpha * ln_n, b - beta * ln_d, e];
        let max = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = terms.iter().map(|t| (t - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let lse = max + total.ln();
        let (h, dh) = huber(lse - run.loss.ln(), delta);
        value += h;
        let soft: Vec<f64> = weights.iter().map(|w| w / total).collect();
        grad[0] += dh * soft[0];
        grad[1] += dh * soft[1];
        grad[2] += dh * soft[2];
        grad[3] += dh * soft[0] * -ln_n;
        grad[4] += dh * soft[1] * -ln_d;
    }
    (value, grad)
}

/// Minimise l'objectif par BFGS, à partir de `start`. Rend le point
/// trouvé et la valeur de l'objectif.
fn bfgs(start: [f64; 5], runs: &[Run], delta: f64) -> ([f64; 5], f64) {
    let mut x = start;
    let (mut f, mut g) = objective(&x, runs, delta);
    // L'inverse approché du hessien, d'abord l'identité.
    let mut h = [[0.0; 5]; 5];
    for (i, row) in h.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..500 {
        let direction: [f64; 5] =
            std::array::from_fn(|i| -(0..5).map(|j| h[i][j] * g[j]).sum::<f64>());
        let slope: f64 = (0..5).map(|i| direction[i] * g[i]).sum();
        if slope >= 0.0 {
            // Une direction qui ne descend pas : on repart de l'identité.
            for (i, row) in h.iter_mut().enumerate() {
                *row = [0.0; 5];
                row[i] = 1.0;
            }
            continue;
        }
        // Recherche linéaire d'Armijo.
        let mut step = 1.0;
        let (mut next, mut f_next, mut g_next);
        loop {
            next = std::array::from_fn(|i| x[i] + step * direction[i]);
            (f_next, g_next) = objective(&next, runs, delta);
            if f_next <= f + 1e-4 * step * slope || step < 1e-12 {
                break;
            }
            step *= 0.5;
        }
        let s: [f64; 5] = std::array::from_fn(|i| next[i] - x[i]);
        let y: [f64; 5] = std::array::from_fn(|i| g_next[i] - g[i]);
        let sy: f64 = (0..5).map(|i| s[i] * y[i]).sum();
        let converged = (f - f_next).abs() < 1e-14 * (1.0 + f.abs());
        x = next;
        f = f_next;
        g = g_next;
        if converged {
            break;
        }
        if sy > 1e-12 {
            // La mise à jour de BFGS de l'inverse du hessien.
            let hy: [f64; 5] = std::array::from_fn(|i| (0..5).map(|j| h[i][j] * y[j]).sum());
            let yhy: f64 = (0..5).map(|i| y[i] * hy[i]).sum();
            for i in 0..5 {
                for j in 0..5 {
                    h[i][j] +=
                        (sy + yhy) * s[i] * s[j] / (sy * sy) - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                }
            }
        }
    }
    (x, f)
}

/// Ajuste la formule de Chinchilla sur `runs`, comme l'article : la perte de
/// Huber de seuil 10⁻³ sur les logarithmes, minimisée depuis une grille de
/// points de départ, dont on garde le meilleur.
pub fn fit_loss_law(runs: &[Run]) -> LossLaw {
    let delta = 1e-3;
    let mut best: Option<([f64; 5], f64)> = None;
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0] {
        for beta in [0.0, 0.5, 1.0, 1.5, 2.0] {
            for e in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                for a in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
                    for b in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
                        let (x, f) = bfgs([a, b, e, alpha, beta], runs, delta);
                        if f.is_finite() && best.is_none_or(|(_, g)| f < g) {
                            best = Some((x, f));
                        }
                    }
                }
            }
        }
    }
    let ([a, b, e, alpha, beta], _) = best.expect("un ajustement");
    LossLaw {
        e: e.exp(),
        a: a.exp(),
        alpha,
        b: b.exp(),
        beta,
    }
}

/// Refait l'ajustement sur `rounds` tirages avec remise des points, et rend
/// les lois obtenues : de quoi donner un intervalle à chaque constante.
pub fn bootstrap(runs: &[Run], rounds: usize, rng: &mut impl Rng) -> Vec<LossLaw> {
    (0..rounds)
        .map(|_| {
            let sample: Vec<Run> = (0..runs.len())
                .map(|_| runs[rng.random_range(0..runs.len())])
                .collect();
            fit_loss_law(&sample)
        })
        .collect()
}

/// Le quantile `q` de `values`.
pub fn quantile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn the_parabola_through_three_points_is_exact() {
        let (a, b, c) = fit_parabola(&[(0.0, 1.0), (1.0, 0.0), (2.0, 1.0)]);
        assert!((a - 1.0).abs() < 1e-12 && (b + 2.0).abs() < 1e-12 && (c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_power_law_is_a_line_in_logarithms() {
        let points: Vec<(f64, f64)> = [1e15f64, 1e16, 1e17]
            .iter()
            .map(|&c| (c, 3.0 * c.powf(0.5)))
            .collect();
        let (exponent, k) = fit_power_law(&points);
        assert!((exponent - 0.5).abs() < 1e-12 && (k / 3.0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_loss_law_is_found_from_synthetic_runs() {
        // Des points tirés d'une loi connue, avec un peu de bruit.
        let truth = LossLaw {
            e: 0.7,
            a: 40.0,
            alpha: 0.35,
            b: 300.0,
            beta: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let mut runs = Vec::new();
        for n in [1e6, 3e6, 1e7, 3e7, 1e8] {
            for d in [1e7, 3e7, 1e8, 3e8, 1e9] {
                let noise = 1.0 + 0.002 * (rng.random::<f64>() - 0.5);
                runs.push(Run {
                    parameters: n,
                    tokens: d,
                    loss: truth.predict(n, d) * noise,
                });
            }
        }
        let fitted = fit_loss_law(&runs);
        for run in &runs {
            let error = (fitted.predict(run.parameters, run.tokens)
                / truth.predict(run.parameters, run.tokens)
                - 1.0)
                .abs();
            assert!(error < 0.003, "{fitted:?}");
        }
        assert!((fitted.alpha - truth.alpha).abs() < 0.05, "{fitted:?}");
        assert!((fitted.beta - truth.beta).abs() < 0.05, "{fitted:?}");
        // L'optimum suit la formule fermée : à calcul fixé, la perte
        // prédite est minimale en `N`.
        let (n, d) = truth.optimum(1e18, 6.0);
        let best = truth.predict(n, d);
        for factor in [0.8, 1.25] {
            assert!(truth.predict(n * factor, d / factor) > best);
        }
    }
}
