//! Compter le calcul : les multiplications-additions d'une passe du modèle,
//! et les opérations d'un pas d'entraînement.
//!
//! Presque tout le calcul d'un Transformer est dans ses produits
//! matriciels : un produit `[M, K] · [K, N]` fait `M × K × N`
//! multiplications-additions, soit deux fois plus d'opérations. Les
//! normalisations, les activations et le softmax font quelques opérations
//! par élément : à côté, c'est négligeable, et on ne les compte pas.
//!
//! On compte le calcul **du modèle**, celui que sa définition demande, et
//! non celui que font nos kernels. L'attention fusionnée saute les tuiles
//! entièrement dans le futur, et refait les scores dans sa passe arrière :
//! le premier fait moins, le second plus. Le compte d'ici est celui de la
//! définition de référence, avec la matrice des scores entière.
//!
//! La passe arrière fait deux produits pour chaque produit de la passe
//! avant : le gradient par rapport à chacun des deux facteurs. Un pas
//! d'entraînement coûte donc trois passes avant.

use crate::model::GptConfig;
use crate::nn::{AttentionKind, FeedForwardKind, SwiGluFeedForward};

/// Les multiplications-additions d'une passe avant sur `batch` séquences
/// de `length` tokens, ou `None` pour les pièces que ce compte ne connaît
/// pas : le mélange d'experts et la prédiction multi-token.
pub fn forward_macs(config: &GptConfig, batch: usize, length: usize) -> Option<u64> {
    let architecture = config.architecture;
    if architecture.experts.is_some() || architecture.multi_token.is_some() {
        return None;
    }
    let (c, h, v) = (
        config.embed as u64,
        config.heads as u64,
        config.vocab as u64,
    );
    let tokens = (batch * length) as u64;
    let t = length as u64;
    let mut macs = 0u64;
    for _ in 0..architecture.passes {
        for layer in 0..config.layers {
            macs += match architecture.attention {
                AttentionKind::Standard => {
                    // Requêtes et sortie, plus clés et valeurs si la couche
                    // ne les emprunte pas. Puis les scores `[T, T]` de
                    // chaque tête, et le mélange des valeurs : `T² × C`
                    // chacun par séquence.
                    let projections = if architecture.borrows_keys(layer) {
                        2
                    } else {
                        4
                    };
                    projections * tokens * c * c + 2 * batch as u64 * t * t * c
                }
                AttentionKind::Latent(dims) => {
                    let (dq, dc, dr) = (
                        dims.query_latent as u64,
                        dims.latent as u64,
                        dims.rope as u64,
                    );
                    let per_token =
                        c * dq + dq * c + dq * h * dr + c * dc + 2 * dc * c + c * dr + c * c;
                    let head = c / h;
                    tokens * per_token + batch as u64 * h * t * t * (2 * head + dr)
                }
            };
            macs += match architecture.feed_forward {
                FeedForwardKind::Gelu => 8 * tokens * c * c,
                FeedForwardKind::SwiGlu => {
                    3 * tokens
                        * c
                        * SwiGluFeedForward::<crate::backend::cpu::Cpu<f32>>::hidden(config.embed)
                            as u64
                }
            };
        }
    }
    // La sortie : chaque état caché face à toute la table d'embedding.
    Some(macs + tokens * c * v)
}

/// Les opérations d'un pas d'entraînement : trois passes avant, deux
/// opérations par multiplication-addition.
pub fn training_flops(config: &GptConfig, batch: usize, length: usize) -> Option<u64> {
    forward_macs(config, batch, length).map(|macs| 6 * macs)
}

/// L'approximation habituelle, `6 × N × D` : `N` paramètres, `D` tokens. Elle
/// oublie l'attention, qui croît avec le contexte, et compte comme du calcul
/// la table d'embedding de l'entrée, qui n'est qu'une lecture.
pub fn six_n_d(parameters: u64, tokens: u64) -> u64 {
    6 * parameters * tokens
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::backend::cpu::{Cpu, CpuTensor};
    use crate::backend::ops::{
        Affine, CachedAttention, CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu,
        Normalize, QuantizedMatmul, Reference, RmsNormalize, RotatePairs, Run, Silu,
        TransposedMatmul,
    };
    use crate::backend::{AdamWStep, Primitives};
    use crate::model::Gpt;
    use crate::model::tests::TINY;
    use crate::nn::{Architecture, LatentDims};
    use crate::optim::Sgd;
    use crate::training::train_step;

    /// Le backend CPU, qui compte les multiplications-additions de ses
    /// produits matriciels. Les opérations écrites avec les primitives
    /// passent par lui, et leurs produits sont comptés aussi.
    struct Counting {
        inner: Cpu<f64>,
        macs: Cell<u64>,
    }

    type T = CpuTensor<f64>;

    macro_rules! delegate {
        ($(fn $name:ident(&self $(, $argument:ident: $type:ty)* $(,)?) $(-> $output:ty)?;)*) => {
            $(
                fn $name(&self $(, $argument: $type)*) $(-> $output)? {
                    self.inner.$name($($argument),*)
                }
            )*
        };
    }

    impl Primitives for Counting {
        type Scalar = f64;
        type Tensor = T;

        fn shape<'a>(&self, tensor: &'a T) -> &'a [usize] {
            self.inner.shape(tensor)
        }

        fn matmul(&self, a: &T, b: &T) -> T {
            let out = self.inner.matmul(a, b);
            let shape = out.shape();
            let inner = a.shape()[a.shape().len() - 1];
            let count: usize = shape.iter().product::<usize>() * inner;
            self.macs.set(self.macs.get() + count as u64);
            out
        }

        delegate! {
            fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> T;
            fn to_vec(&self, tensor: &T) -> Vec<f64>;
            fn add(&self, a: &T, b: &T) -> T;
            fn sub(&self, a: &T, b: &T) -> T;
            fn mul(&self, a: &T, b: &T) -> T;
            fn div(&self, a: &T, b: &T) -> T;
            fn neg(&self, a: &T) -> T;
            fn exp(&self, a: &T) -> T;
            fn ln(&self, a: &T) -> T;
            fn sqrt(&self, a: &T) -> T;
            fn tanh(&self, a: &T) -> T;
            fn sum(&self, a: &T, axis: usize, keepdim: bool) -> T;
            fn sum_all(&self, a: &T) -> T;
            fn reshape(&self, a: &T, shape: &[usize]) -> T;
            fn transpose(&self, a: &T, axis_a: usize, axis_b: usize) -> T;
            fn expand(&self, a: &T, shape: &[usize]) -> T;
            fn softmax(&self, a: &T) -> T;
            fn log_softmax(&self, a: &T) -> T;
            fn masked_fill(&self, a: &T, mask: &T, value: f64) -> T;
            fn gather_rows(&self, a: &T, indices: &[u32]) -> T;
            fn scatter_add_rows(&self, a: &T, indices: &[u32], num_rows: usize) -> T;
            fn concat_rows(&self, a: &T, b: &T) -> T;
            fn write_rows(&self, table: &mut T, indices: &T, rows: &T);
            fn sgd_step(&self, param: &mut T, grad: &T, learning_rate: f64);
            fn adamw_step(
                &self,
                param: &mut T,
                grad: &T,
                first_moment: &mut T,
                second_moment: &mut T,
                step: &AdamWStep,
            );
        }
    }

    /// Les opérations écrites avec les primitives : leurs produits passent
    /// par `Counting`.
    macro_rules! by_reference {
        ($($op:ty),*) => {$(
            impl Run<$op> for Counting {
                fn run(&self, op: &$op, inputs: &[&T]) -> T {
                    op.reference(self, inputs)
                }
                fn run_backward(&self, op: &$op, inputs: &[&T], output: &T, grad: &T) -> Vec<Option<T>> {
                    op.reference_backward(self, inputs, output, grad)
                }
            }
        )*};
    }

    /// Les opérations élément par élément, sans produit.
    macro_rules! elementwise {
        ($($op:ty),*) => {$(
            impl Run<$op> for Counting {
                fn run(&self, op: &$op, inputs: &[&T]) -> T {
                    self.inner.run(op, inputs)
                }
                fn run_backward(&self, op: &$op, inputs: &[&T], output: &T, grad: &T) -> Vec<Option<T>> {
                    self.inner.run_backward(op, inputs, output, grad)
                }
            }
        )*};
    }

    by_reference!(
        Affine,
        FusedAttention,
        CachedAttention,
        CausalSoftmax,
        Concat,
        Normalize,
        RmsNormalize,
        TransposedMatmul,
        QuantizedMatmul
    );
    elementwise!(Gelu, Silu, RotatePairs, CrossEntropy);

    /// Compte les multiplications-additions d'un pas d'entraînement réel.
    fn counted_step(config: GptConfig, batch: usize) -> u64 {
        let backend = Counting {
            inner: Cpu::new(),
            macs: Cell::new(0),
        };
        let mut model = Gpt::new(&backend, config, &mut StdRng::seed_from_u64(0));
        let tokens: Vec<u32> = (0..batch * config.context)
            .map(|i| (i % config.vocab) as u32)
            .collect();
        let mut optimizer = Sgd::new(0.1);
        backend.macs.set(0);
        train_step(&backend, &mut model, &mut optimizer, |m, t| {
            m.training_loss(t, &tokens, &tokens, batch)
        });
        backend.macs.get()
    }

    #[test]
    fn the_count_is_that_of_a_real_training_step() {
        let config = GptConfig {
            context: 6,
            layers: 2,
            ..TINY
        };
        let latent = Architecture::MODERN.with_attention(AttentionKind::Latent(LatentDims {
            latent: 3,
            query_latent: 5,
            rope: 2,
        }));
        for architecture in [
            Architecture::GPT2,
            Architecture::MODERN,
            Architecture::MODERN.with_own_keys(1),
            Architecture::MODERN.with_passes(2),
            latent,
        ] {
            let config = config.with_architecture(architecture);
            let counted = counted_step(config, 3);
            // La passe arrière de l'attention recalcule ses poids, comme le
            // fait le GPU : un produit des scores de plus par couche, que
            // la définition du modèle ne demande pas.
            let key_dim = match architecture.attention {
                AttentionKind::Standard => config.head_dim(),
                AttentionKind::Latent(dims) => config.head_dim() + dims.rope,
            };
            let t = config.context;
            let recomputed =
                config.layers * architecture.passes * 3 * config.heads * t * t * key_dim;
            let computed = 3 * forward_macs(&config, 3, t).unwrap() + recomputed as u64;
            assert_eq!(counted, computed, "{architecture:?}");
        }
    }

    #[test]
    fn the_rule_of_thumb_forgets_the_attention() {
        // Le modèle `xl` : 29 millions de paramètres, des séquences de 512.
        let config = GptConfig::xl(8192).with_architecture(Architecture::MODERN);
        let exact = training_flops(&config, 1, 512).unwrap() as f64;
        let parameters = 29_443_968u64;
        let rule = six_n_d(parameters, 512) as f64;
        // La table d'embedding compte bien : elle sert aussi à la sortie.
        // Mais l'attention ajoute `12 × L × T × C` opérations par token,
        // 14 % de plus à 512 positions.
        let attention = 12.0 * 8.0 * 512.0 * 512.0 * 512.0;
        assert!(
            (exact - rule - attention).abs() < 0.01 * exact,
            "{exact} contre {rule}"
        );
        assert!((1.13..1.15).contains(&(exact / rule)));
    }
}
