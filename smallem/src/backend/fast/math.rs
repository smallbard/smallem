//! Les calculs d'une ligne : produit scalaire, somme, maximum, exponentielle.
//!
//! Chaque fonction est écrite une fois, en Rust ordinaire, par paquets de
//! seize éléments, puis compilée trois fois : pour AVX-512, pour AVX2 et
//! sans instructions vectorielles explicites (voir `multiversion!`). Le
//! compilateur vectorise les paquets avec les registres de chaque version.
//!
//! Les sommes se font dans un ordre fixé par la longueur de la ligne : seize
//! sommes partielles, réunies ensuite deux à deux. Deux appels sur la même
//! ligne donnent les mêmes bits.

#[cfg(test)]
use super::Isa;
use super::fma;

/// La largeur des paquets.
const LANES: usize = 16;

/// Réunit des sommes partielles deux à deux, toujours dans le même ordre.
#[inline(always)]
fn reduce(mut acc: [f32; LANES]) -> f32 {
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] += acc[i + width];
        }
    }
    acc[0]
}

#[inline(always)]
fn dot_body<const FMA: bool>(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = [[0.0f32; LANES]; 4];
    let (chunks_a, rest_a) = a.as_chunks::<{ 4 * LANES }>();
    let (chunks_b, rest_b) = b.as_chunks::<{ 4 * LANES }>();
    for (x, y) in chunks_a.iter().zip(chunks_b) {
        for (part, lanes) in acc.iter_mut().enumerate() {
            let (x, y) = (&x[part * LANES..][..LANES], &y[part * LANES..][..LANES]);
            for ((sum, &x), &y) in lanes.iter_mut().zip(x).zip(y) {
                *sum = fma::<FMA>(x, y, *sum);
            }
        }
    }
    let mut total = [0.0f32; LANES];
    for (lane, sum) in total.iter_mut().enumerate() {
        *sum = (acc[0][lane] + acc[1][lane]) + (acc[2][lane] + acc[3][lane]);
    }
    for (index, (&x, &y)) in rest_a.iter().zip(rest_b).enumerate() {
        total[index % LANES] = fma::<FMA>(x, y, total[index % LANES]);
    }
    reduce(total)
}

multiversion! {
    /// `Σ a[i] × b[i]`.
    pub fn dot(a: &[f32], b: &[f32]) -> f32 = dot_body;
}

#[inline(always)]
fn sum_body<const FMA: bool>(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (chunks, rest) = x.as_chunks::<LANES>();
    for chunk in chunks {
        for (sum, &value) in acc.iter_mut().zip(chunk) {
            *sum += value;
        }
    }
    for (sum, &value) in acc.iter_mut().zip(rest) {
        *sum += value;
    }
    reduce(acc)
}

multiversion! {
    /// `Σ x[i]`.
    pub fn sum(x: &[f32]) -> f32 = sum_body;
}

#[inline(always)]
fn max_body<const FMA: bool>(x: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; LANES];
    let (chunks, rest) = x.as_chunks::<LANES>();
    for chunk in chunks {
        for (max, &value) in acc.iter_mut().zip(chunk) {
            *max = if value > *max { value } else { *max };
        }
    }
    for (max, &value) in acc.iter_mut().zip(rest) {
        if value > *max {
            *max = value;
        }
    }
    acc.into_iter()
        .fold(f32::NEG_INFINITY, |max, x| if x > max { x } else { max })
}

multiversion! {
    /// Le plus grand élément, `-∞` pour une ligne vide.
    pub fn max(x: &[f32]) -> f32 = max_body;
}

// --- Exponentielle ---

/// En dessous, `exp(x)` est sous le plus petit `f32` normal : 0.
const EXP_LOW: f32 = -87.336_54;
/// Au-dessus, `exp(x)` dépasse le plus grand `f32` : l'infini.
const EXP_HIGH: f32 = 88.376_26;

/// `exp(x)`, à environ un ulp près, en opérations que le compilateur sait
/// vectoriser : c'est l'algorithme de Cephes.
///
/// On écrit `x = n·ln 2 + r`, avec `n` entier et `|r| ≤ ln 2 / 2`. Alors
/// `exp(x) = 2ⁿ · exp(r)` : `2ⁿ` se construit directement dans l'exposant du
/// `f32`, et `exp(r)` est un polynôme de degré 7 sur un petit intervalle.
/// `ln 2` est coupé en deux constantes, pour que `n·ln 2` reste exact.
#[inline(always)]
pub fn exp_lane<const FMA: bool>(x: f32) -> f32 {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    const LN2_HIGH: f32 = 0.693_359_4;
    const LN2_LOW: f32 = -2.121_944_4e-4;
    let clamped = x.clamp(EXP_LOW, EXP_HIGH);
    let n = (clamped * LOG2_E).round_ties_even();
    let r = fma::<FMA>(n, -LN2_HIGH, clamped);
    let r = fma::<FMA>(n, -LN2_LOW, r);
    let mut p = 1.987_569_1e-4;
    p = fma::<FMA>(p, r, 1.398_199_9e-3);
    p = fma::<FMA>(p, r, 8.333_452e-3);
    p = fma::<FMA>(p, r, 4.166_579_6e-2);
    p = fma::<FMA>(p, r, 1.666_666_5e-1);
    // 5,000 000 1 × 10⁻¹ chez Cephes, qui vaut 0,5 en `f32`.
    p = fma::<FMA>(p, r, 0.5);
    let y = fma::<FMA>(p, r * r, r) + 1.0;
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    let value = y * scale;
    if x < EXP_LOW {
        0.0
    } else if x > EXP_HIGH {
        f32::INFINITY
    } else {
        value
    }
}

#[inline(always)]
fn exp_body<const FMA: bool>(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = exp_lane::<FMA>(*value);
    }
}

multiversion! {
    /// `x[i] ← exp(x[i])`.
    pub fn exp_in_place(x: &mut [f32]) -> () = exp_body;
}

#[inline(always)]
fn exp_shifted_sum_body<const FMA: bool>(x: &mut [f32], shift: f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (chunks, rest) = x.as_chunks_mut::<LANES>();
    for chunk in chunks {
        for (sum, value) in acc.iter_mut().zip(chunk.iter_mut()) {
            *value = exp_lane::<FMA>(*value - shift);
            *sum += *value;
        }
    }
    for (sum, value) in acc.iter_mut().zip(rest.iter_mut()) {
        *value = exp_lane::<FMA>(*value - shift);
        *sum += *value;
    }
    reduce(acc)
}

multiversion! {
    /// `x[i] ← exp(x[i] - shift)`, et rend la somme des nouvelles valeurs :
    /// le cœur du softmax.
    pub fn exp_shifted_sum(x: &mut [f32], shift: f32) -> f32 = exp_shifted_sum_body;
}

#[inline(always)]
fn exp_shifted_total_body<const FMA: bool>(x: &[f32], shift: f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (chunks, rest) = x.as_chunks::<LANES>();
    for chunk in chunks {
        for (sum, &value) in acc.iter_mut().zip(chunk) {
            *sum += exp_lane::<FMA>(value - shift);
        }
    }
    for (sum, &value) in acc.iter_mut().zip(rest) {
        *sum += exp_lane::<FMA>(value - shift);
    }
    reduce(acc)
}

multiversion! {
    /// `Σ exp(x[i] - shift)`, sans rien écrire : le dénominateur du
    /// logarithme du softmax.
    pub fn exp_shifted_total(x: &[f32], shift: f32) -> f32 = exp_shifted_total_body;
}

#[inline(always)]
fn scale_body<const FMA: bool>(x: &mut [f32], factor: f32) {
    for value in x.iter_mut() {
        *value *= factor;
    }
}

multiversion! {
    /// `x[i] ← x[i] × factor`.
    pub fn scale(x: &mut [f32], factor: f32) -> () = scale_body;
}

#[inline(always)]
fn axpy_body<const FMA: bool>(factor: f32, x: &[f32], y: &mut [f32]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y = fma::<FMA>(factor, x, *y);
    }
}

multiversion! {
    /// `y[i] ← y[i] + factor × x[i]`.
    pub fn axpy(factor: f32, x: &[f32], y: &mut [f32]) -> () = axpy_body;
}

// --- Activations ---

/// `tanh(u) = 1 - 2 / (exp(2u) + 1)` : une exponentielle et une division,
/// que le compilateur vectorise. Aux deux bouts, `exp` donne `0` ou `∞`, et
/// la formule donne `-1` ou `1`.
#[inline(always)]
fn tanh_lane<const FMA: bool>(u: f32) -> f32 {
    1.0 - 2.0 / (exp_lane::<FMA>(2.0 * u) + 1.0)
}

/// `√(2/π)` et le coefficient du terme cubique de GELU (voir
/// [`float::gelu`](crate::backend::float::gelu)).
const GELU_SCALE: f32 = 0.797_884_6;
const GELU_CUBIC: f32 = 0.044_715;

#[inline(always)]
fn gelu_body<const FMA: bool>(x: &[f32], out: &mut [f32]) {
    for (y, &x) in out.iter_mut().zip(x) {
        let t = tanh_lane::<FMA>(GELU_SCALE * (x + GELU_CUBIC * x * x * x));
        *y = 0.5 * x * (1.0 + t);
    }
}

multiversion! {
    /// `out[i] = gelu(x[i])`.
    pub fn gelu(x: &[f32], out: &mut [f32]) -> () = gelu_body;
}

#[inline(always)]
fn gelu_backward_body<const FMA: bool>(x: &[f32], grad: &[f32], out: &mut [f32]) {
    for ((y, &x), &g) in out.iter_mut().zip(x).zip(grad) {
        let t = tanh_lane::<FMA>(GELU_SCALE * (x + GELU_CUBIC * x * x * x));
        let inner = GELU_SCALE * (1.0 + 3.0 * GELU_CUBIC * x * x);
        *y = g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * inner);
    }
}

multiversion! {
    /// `out[i] = grad[i] × gelu'(x[i])`.
    pub fn gelu_backward(x: &[f32], grad: &[f32], out: &mut [f32]) -> () = gelu_backward_body;
}

#[inline(always)]
fn silu_body<const FMA: bool>(x: &[f32], out: &mut [f32]) {
    for (y, &x) in out.iter_mut().zip(x) {
        *y = x / (1.0 + exp_lane::<FMA>(-x));
    }
}

multiversion! {
    /// `out[i] = silu(x[i]) = x / (1 + exp(-x))`.
    pub fn silu(x: &[f32], out: &mut [f32]) -> () = silu_body;
}

#[inline(always)]
fn silu_backward_body<const FMA: bool>(x: &[f32], grad: &[f32], out: &mut [f32]) {
    for ((y, &x), &g) in out.iter_mut().zip(x).zip(grad) {
        let s = 1.0 / (1.0 + exp_lane::<FMA>(-x));
        *y = g * s * (1.0 + x * (1.0 - s));
    }
}

multiversion! {
    /// `out[i] = grad[i] × silu'(x[i])`.
    pub fn silu_backward(x: &[f32], grad: &[f32], out: &mut [f32]) -> () = silu_backward_body;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isas() -> Vec<Isa> {
        [Isa::Scalar, Isa::Avx2, Isa::Avx512]
            .into_iter()
            .filter(|isa| isa.available())
            .collect()
    }

    #[test]
    fn exp_is_within_two_ulps() {
        for isa in isas() {
            let mut x: Vec<f32> = (-20_000..=20_000).map(|i| i as f32 * 4.35e-3).collect();
            let original = x.clone();
            exp_in_place(isa, &mut x);
            for (&input, &got) in original.iter().zip(&x) {
                let want = (input as f64).exp();
                let ulp = (want as f32).abs() * f32::EPSILON;
                assert!(
                    (got as f64 - want).abs() <= 2.0 * ulp as f64,
                    "{isa:?} exp({input}) = {got}, attendu {want}",
                );
            }
        }
    }

    #[test]
    fn exp_handles_the_extremes() {
        for isa in isas() {
            let mut x = vec![
                f32::NEG_INFINITY,
                -1000.0,
                -87.5,
                0.0,
                88.0,
                89.0,
                f32::INFINITY,
            ];
            exp_in_place(isa, &mut x);
            assert_eq!(x[0], 0.0);
            assert_eq!(x[1], 0.0);
            assert_eq!(x[2], 0.0);
            assert_eq!(x[3], 1.0);
            assert!(x[4].is_finite() && x[4] > 1e38);
            assert_eq!(x[5], f32::INFINITY);
            assert_eq!(x[6], f32::INFINITY);
            let mut nan = vec![f32::NAN];
            exp_in_place(isa, &mut nan);
            assert!(nan[0].is_nan());
        }
    }

    #[test]
    fn sums_are_exact_on_integers_and_stable() {
        for isa in isas() {
            for len in [0, 1, 15, 16, 17, 100, 1000] {
                let x: Vec<f32> = (0..len).map(|i| (i % 7) as f32).collect();
                let want: f32 = x.iter().sum();
                assert_eq!(sum(isa, &x), want);
                assert_eq!(dot(isa, &x, &x), x.iter().map(|v| v * v).sum::<f32>());
                let expected_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert_eq!(max(isa, &x), expected_max);
            }
        }
    }
}
