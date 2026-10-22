//! Un modèle de langage de type GPT, écrit à partir de zéro.
//!
//! Les modules, du texte au modèle entraîné :
//!
//! - le texte : [`corpus`], [`tokenizer`], [`data`], [`ngram`] ;
//! - le calcul : [`backend`], [`autograd`], [`flops`] ;
//! - les modèles : [`nn`], [`model`], [`mlp`], [`gpt2`], [`io`] ;
//! - l'entraînement : [`optim`], [`training`], [`scaling`] ;
//! - la génération : [`cache`], [`generation`], [`sampling`],
//!   [`speculative`], [`quantize`] ;
//! - le post-entraînement : [`chat`], [`tasks`], [`post_training`] ;
//! - les programmes : [`cli`].

pub mod autograd;
pub mod backend;
pub mod cache;
pub mod chat;
pub mod cli;
pub mod corpus;
pub mod data;
pub mod flops;
pub mod generation;
pub mod gpt2;
pub mod io;
pub mod mlp;
pub mod model;
pub mod ngram;
pub mod nn;
pub mod optim;
pub mod post_training;
pub mod quantize;
pub mod sampling;
pub mod scaling;
pub mod speculative;
pub mod tasks;
pub mod tokenizer;
pub mod training;
