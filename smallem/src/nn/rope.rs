//! RoPE : la position d'un token entre dans l'attention par une rotation
//! de ses requêtes et de ses clés.

use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};

/// Les angles de RoPE pour une suite de positions, sous la forme de leurs
/// cosinus et de leurs sinus, `[positions, D/2]` chacun.
#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub(super) cos: Var,
    pub(super) sin: Var,
}

impl Rope {
    /// La fréquence la plus lente fait un tour en `2π × 10 000` positions.
    pub const BASE: f64 = 10_000.0;

    /// Les angles des positions `start..start + length`, pour des têtes de
    /// dimension `head_dim`.
    ///
    /// La paire de composantes `i` tourne de `position × BASE^(-2i/D)`. Les
    /// premières paires tournent vite et distinguent des positions voisines,
    /// les dernières tournent lentement et distinguent des positions
    /// éloignées, comme les aiguilles d'une horloge.
    pub fn new<B: Backend>(tape: &Tape<B>, start: usize, length: usize, head_dim: usize) -> Self {
        let positions: Vec<usize> = (start..start + length).collect();
        Self::at(tape, &positions, head_dim, 1)
    }

    /// Les angles de positions quelconques, chacune répétée `repeat` fois
    /// de suite : `[positions × repeat, D/2]`.
    ///
    /// En génération, les tokens d'une passe viennent de séquences
    /// différentes, et chacun a sa position. Leurs requêtes et leurs clés
    /// sont tournées sous la forme `[tokens × têtes, D]`, une ligne par
    /// token et par tête : la table répète donc chaque position pour toutes
    /// les têtes.
    pub fn at<B: Backend>(
        tape: &Tape<B>,
        positions: &[usize],
        head_dim: usize,
        repeat: usize,
    ) -> Self {
        assert!(
            head_dim.is_multiple_of(2),
            "RoPE demande des têtes de dimension paire, reçu {head_dim}"
        );
        let half = head_dim / 2;
        let frequencies: Vec<f64> = (0..half)
            .map(|pair| Self::BASE.powf(-(2.0 * pair as f64) / head_dim as f64))
            .collect();
        let rows = positions.len() * repeat;
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        for &position in positions {
            let start = cos.len();
            for frequency in &frequencies {
                let (s, c) = (position as f64 * frequency).sin_cos();
                cos.push(B::Scalar::from_f64(c));
                sin.push(B::Scalar::from_f64(s));
            }
            // Les répétitions sont des copies de la première ligne.
            for _ in 1..repeat {
                cos.extend_from_within(start..start + half);
                sin.extend_from_within(start..start + half);
            }
        }
        let backend = tape.backend();
        Self {
            cos: tape.constant(backend.from_vec(cos, &[rows, half])),
            sin: tape.constant(backend.from_vec(sin, &[rows, half])),
        }
    }

    /// Tourne `x`, de forme `[…, positions, D]`.
    pub fn rotate<B: Backend>(&self, tape: &Tape<B>, x: Var) -> Var {
        tape.rotate_pairs(x, self.cos, self.sin)
    }
}
