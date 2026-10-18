//! Abstraction minimale sur `f32` et `f64`.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Le strict nécessaire pour écrire les opérations une seule fois pour `f32`
/// (comparaison avec le GPU) et `f64` (vérification des gradients).
pub trait Float:
    Copy
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_INFINITY: Self;

    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
}

macro_rules! impl_float {
    ($type:ty) => {
        impl Float for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const NEG_INFINITY: Self = <$type>::NEG_INFINITY;

            fn from_f64(value: f64) -> Self {
                value as $type
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn exp(self) -> Self {
                <$type>::exp(self)
            }
            fn ln(self) -> Self {
                <$type>::ln(self)
            }
            fn sqrt(self) -> Self {
                <$type>::sqrt(self)
            }
            fn tanh(self) -> Self {
                <$type>::tanh(self)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

/// `√(2/π)`, le facteur de l'approximation de GELU.
const GELU_SCALE: f64 = 0.797_884_560_802_865_4;
/// Coefficient du terme cubique de l'approximation de GELU.
const GELU_CUBIC: f64 = 0.044715;

/// GELU, la fonction d'activation de GPT, dans l'approximation en `tanh`
/// qu'utilise GPT-2 :
///
/// ```text
/// gelu(x) = 0,5 · x · (1 + tanh(√(2/π) · (x + 0,044715 · x³)))
/// ```
///
/// Elle ressemble à `max(0, x)` en plus lisse : presque nulle pour les
/// valeurs négatives, presque `x` pour les grandes valeurs positives.
///
/// Cette fonction est compilée pour l'hôte et pour le GPU : les deux
/// backends appliquent exactement la même formule.
pub fn gelu<F: Float>(x: F) -> F {
    let inner = F::from_f64(GELU_SCALE) * (x + F::from_f64(GELU_CUBIC) * x * x * x);
    F::from_f64(0.5) * x * (F::ONE + inner.tanh())
}

/// Dérivée de [`gelu`]. En notant `u = √(2/π) · (x + 0,044715 · x³)` et
/// `t = tanh(u)` :
///
/// ```text
/// gelu'(x) = 0,5 · (1 + t) + 0,5 · x · (1 - t²) · √(2/π) · (1 + 3 · 0,044715 · x²)
/// ```
pub fn gelu_slope<F: Float>(x: F) -> F {
    let scale = F::from_f64(GELU_SCALE);
    let cubic = F::from_f64(GELU_CUBIC);
    let half = F::from_f64(0.5);
    let t = (scale * (x + cubic * x * x * x)).tanh();
    let inner_slope = scale * (F::ONE + F::from_f64(3.0) * cubic * x * x);
    half * (F::ONE + t) + half * x * (F::ONE - t * t) * inner_slope
}

/// La sigmoïde, `1 / (1 + exp(-x))` : elle ramène tout nombre entre 0 et 1.
pub fn sigmoid<F: Float>(x: F) -> F {
    F::ONE / (F::ONE + (-x).exp())
}

/// SiLU (ou *swish*), la fonction d'activation de SwiGLU :
///
/// ```text
/// silu(x) = x · sigmoïde(x)
/// ```
///
/// Comme GELU, à laquelle elle ressemble beaucoup : presque nulle pour les
/// valeurs très négatives, presque `x` pour les grandes valeurs positives.
pub fn silu<F: Float>(x: F) -> F {
    x * sigmoid(x)
}

/// Dérivée de [`silu`]. En notant `s = sigmoïde(x)`, dont la dérivée est
/// `s · (1 - s)` :
///
/// ```text
/// silu'(x) = s + x · s · (1 - s) = s · (1 + x · (1 - s))
/// ```
pub fn silu_slope<F: Float>(x: F) -> F {
    let s = sigmoid(x);
    s * (F::ONE + x * (F::ONE - s))
}
