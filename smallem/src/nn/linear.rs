//! Les couches de base : la table d'embeddings, la couche affine et la
//! projection.

use rand::Rng;

use super::{Adapter, INIT_STD, Parameter, QuantizedWeight, module};
use crate::autograd::{Tape, Var};
use crate::backend::Backend;
use crate::backend::Float;
use crate::quantize::{Format, Quantized};

/// Table qui associe un vecteur appris à chaque token.
pub struct Embedding<B: Backend> {
    /// `[vocabulaire, dimension]`
    pub weight: Parameter<B>,
    /// Une correction de rang faible de la table, avec LoRA.
    pub adapter: Option<Box<Adapter<B>>>,
    /// La table quantifiée pour la sortie du modèle, si elle l'a été (voir
    /// [`Embedding::quantize_output`]).
    pub output: Option<Box<QuantizedOutput<B>>>,
}

/// La table d'une sortie quantifiée : `Eᵀ` `[dimension, vocabulaire]`, dont
/// chaque colonne, celle d'un token, a ses échelles.
pub struct QuantizedOutput<B: Backend> {
    pub weight: QuantizedWeight<B>,
    /// `Eᵀ` déquantifiée, la définition de référence du produit.
    pub transposed: B::Tensor,
}

impl<B: Backend> Embedding<B> {
    pub fn new(backend: &B, vocab: usize, dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            // Une table d'embeddings n'est pas une matrice de transformation :
            // la décroissance des poids ne s'y applique pas.
            weight: Parameter::normal(backend, &[vocab, dim], INIT_STD, rng).without_decay(),
            adapter: None,
            output: None,
        }
    }

    /// Quantifie la table pour qu'elle serve, aussi, de sortie quantifiée :
    /// `Eᵀ` `[C, V]` est quantifiée comme une matrice de poids, des groupes
    /// le long de `C` dans la colonne de chaque token. La table elle-même
    /// devient la table déquantifiée : l'entrée et la sortie lisent les
    /// mêmes poids. Rend la matrice quantifiée.
    pub fn quantize_output(&mut self, backend: &B, format: Format) -> Quantized {
        let shape = backend.shape(self.weight.value()).to_vec();
        let (vocab, dim) = (shape[0], shape[1]);
        let table = backend.to_vec(self.weight.value());
        let mut transposed = vec![0.0f64; vocab * dim];
        for v in 0..vocab {
            for c in 0..dim {
                transposed[c * vocab + v] = table[v * dim + c].to_f64();
            }
        }
        let quantized = Quantized::round_to_nearest(&transposed, dim, vocab, format);
        let dequantized = quantized.dequantize();
        let mut back = vec![B::Scalar::from_f64(0.0); vocab * dim];
        for c in 0..dim {
            for v in 0..vocab {
                back[v * dim + c] = B::Scalar::from_f64(dequantized[c * vocab + v]);
            }
        }
        self.weight.set_value(backend.from_vec(back, &[vocab, dim]));
        let dequantized = dequantized.into_iter().map(B::Scalar::from_f64).collect();
        self.output = Some(Box::new(QuantizedOutput {
            weight: QuantizedWeight::new(backend, &quantized),
            transposed: backend.from_vec(dequantized, &[dim, vocab]),
        }));
        quantized
    }

    /// `x · Eᵀ`, `[…, C]` donne `[…, V]` : avec les poids quantifiés de la
    /// sortie s'il y en a, sinon le produit par la table transposée.
    pub fn unembed(&self, tape: &Tape<B>, x: Var) -> Var {
        match &self.output {
            Some(output) => {
                let dequantized = tape.constant(output.transposed.clone());
                tape.quantized_matmul(x, &output.weight, dequantized, None)
            }
            None => tape.transposed_matmul(x, self.table(tape)),
        }
    }

    /// La table, `[vocabulaire, dimension]`, corrigée par son adaptateur
    /// s'il y en a un.
    pub fn table(&self, tape: &Tape<B>) -> Var {
        let weight = self.weight.var(tape);
        match &self.adapter {
            Some(adapter) => tape.add(weight, adapter.delta(tape)),
            None => weight,
        }
    }

    /// `N` tokens donnent `[N, dimension]` : une ligne de la table par token.
    pub fn forward(&self, tape: &Tape<B>, tokens: &[u32]) -> Var {
        tape.gather_rows(self.table(tape), tokens)
    }
}

module!(Embedding { weight, adapter });

/// Couche affine : `x · W + b`.
pub struct Linear<B: Backend> {
    /// `[entrée, sortie]`
    pub weight: Parameter<B>,
    /// `[sortie]`
    pub bias: Parameter<B>,
    /// Une correction de rang faible de `W`, avec LoRA.
    pub adapter: Option<Box<Adapter<B>>>,
}

impl<B: Backend> Linear<B> {
    pub fn new(backend: &B, input: usize, output: usize, rng: &mut impl Rng) -> Self {
        Self::with_std(backend, input, output, INIT_STD, rng)
    }

    /// Comme [`Linear::new`], avec un écart-type d'initialisation choisi.
    pub fn with_std(
        backend: &B,
        input: usize,
        output: usize,
        std: f64,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            weight: Parameter::normal(backend, &[input, output], std, rng),
            bias: Parameter::zeros(backend, &[output]),
            adapter: None,
        }
    }

    /// `[…, entrée]` donne `[…, sortie]`. Le biais s'étire sur toutes les
    /// dimensions qui précèdent.
    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        self.weight.observe(tape, x);
        let bias = self.bias.var(tape);
        let out = match self.weight.quantized() {
            Some(quantized) => {
                tape.quantized_matmul(x, quantized, self.weight.var(tape), Some(bias))
            }
            None => tape.affine(x, self.weight.var(tape), bias),
        };
        match &self.adapter {
            Some(adapter) => tape.add(out, adapter.forward(tape, x)),
            None => out,
        }
    }
}

module!(Linear {
    weight,
    bias,
    adapter
});

/// Projection : `x · W`, une couche affine sans biais.
///
/// Elle sert là où un biais ne changerait rien, ou presque : devant une
/// normalisation, ou entre deux matrices qui n'en forment qu'une.
pub struct Projection<B: Backend> {
    /// `[entrée, sortie]`
    pub weight: Parameter<B>,
}

impl<B: Backend> Projection<B> {
    pub fn new(backend: &B, input: usize, output: usize, rng: &mut impl Rng) -> Self {
        Self {
            weight: Parameter::normal(backend, &[input, output], INIT_STD, rng),
        }
    }

    /// `[…, entrée]` donne `[…, sortie]`.
    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        self.weight.observe(tape, x);
        match self.weight.quantized() {
            Some(quantized) => tape.quantized_matmul(x, quantized, self.weight.var(tape), None),
            None => tape.matmul(x, self.weight.var(tape)),
        }
    }
}

module!(Projection { weight });
