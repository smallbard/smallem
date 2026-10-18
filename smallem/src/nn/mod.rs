//! Les briques d'un réseau : paramètres et couches.

mod architecture;
mod attention;
mod block;
mod experts;
mod feed_forward;
mod latent_attention;
mod linear;
mod lora;
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
pub use block::Block;
pub use experts::{MixtureOfExperts, Routing, route};
pub use feed_forward::{FeedForward, FeedForwardLayer, SwiGluFeedForward};
pub use latent_attention::LatentAttention;
pub use linear::{Embedding, Linear, Projection};
pub use lora::Adapter;
pub use multi_token::MultiTokenHead;
pub use norm::{LayerNorm, Norm, RmsNorm};
pub(crate) use parameter::module;
pub use parameter::{Calibration, Module, Parameter, QuantizedWeight, child_name};
pub use rope::Rope;

/// Écart-type des poids à l'initialisation, comme dans GPT-2.
pub const INIT_STD: f64 = 0.02;
