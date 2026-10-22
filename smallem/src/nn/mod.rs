//! Les briques d'un réseau : paramètres et couches.

mod architecture;
mod attention;
mod block;
mod experts;
mod feed_forward;
mod latent_attention;
mod linear;
mod lora;
mod memory;
mod multi_token;
mod norm;
mod parameter;
mod rope;

#[cfg(test)]
mod tests;

pub use architecture::{
    Architecture, AttentionKind, Balance, ExpertDims, FeedForwardKind, LatentDims, MultiToken,
    NormKind, PositionKind,
};
pub use attention::{Attention, BorrowedAttention, CausalSelfAttention, Lent};
pub use block::{Block, BlockShape};
pub use experts::{MixtureOfExperts, Routing, route};
pub use feed_forward::{FeedForward, FeedForwardLayer, SwiGluFeedForward, swiglu_hidden};
pub use latent_attention::LatentAttention;
pub use linear::{Embedding, Linear, Projection};
pub use lora::Adapter;
pub use memory::{HASH_SEED, MEMORY_STEPS, Memory, MemoryDims, MemoryStorage, MemoryTable};
pub use multi_token::MultiTokenHead;
pub use norm::{LayerNorm, Norm, RmsNorm};
pub(crate) use parameter::module;
pub use parameter::{Calibration, Module, Parameter, QuantizedWeight, child_name};
pub use rope::Rope;

/// Écart-type des poids à l'initialisation, comme dans GPT-2.
pub const INIT_STD: f64 = 0.02;

/// `1 / (1 + e^(−x))`, élément par élément.
pub(crate) fn sigmoid<B: crate::backend::Backend>(
    tape: &crate::autograd::Tape<B>,
    x: crate::autograd::Var,
) -> crate::autograd::Var {
    use crate::backend::Float;
    let one = tape.scalar(B::Scalar::ONE);
    tape.div(one, tape.add(one, tape.exp(tape.neg(x))))
}
