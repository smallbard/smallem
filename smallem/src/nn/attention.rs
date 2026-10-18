//! L'attention causale à plusieurs têtes, et celle qui emprunte ses clés.

use rand::Rng;

use super::{AttentionKind, LatentAttention, Linear, Rope, module};
use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};
use crate::cache::{AttentionCache, CacheStep};

/// Self-attention causale à plusieurs têtes : le seul endroit du modèle où
/// les positions d'une séquence échangent de l'information.
///
/// Chaque position émet une requête (`query`), une clé (`key`) et une valeur
/// (`value`). Le score entre la requête d'une position et la clé d'une autre
/// dit combien la première doit écouter la seconde ; la sortie d'une
/// position est la moyenne des valeurs, pondérée par ces scores.
pub struct CausalSelfAttention<B: Backend> {
    heads: usize,
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    /// Recombine les sorties des têtes.
    pub proj: Linear<B>,
}

impl<B: Backend> CausalSelfAttention<B> {
    /// `proj_std` est l'écart-type d'initialisation de la projection de
    /// sortie, plus petit que les autres (voir [`Block::new`](super::Block::new)).
    pub fn new(backend: &B, dim: usize, heads: usize, proj_std: f64, rng: &mut impl Rng) -> Self {
        assert!(
            dim.is_multiple_of(heads),
            "la dimension {dim} n'est pas divisible par {heads} têtes",
        );
        Self {
            heads,
            query: Linear::new(backend, dim, dim, rng),
            key: Linear::new(backend, dim, dim, rng),
            value: Linear::new(backend, dim, dim, rng),
            proj: Linear::with_std(backend, dim, dim, proj_std, rng),
        }
    }

    /// `x` contient les vecteurs de `N` positions d'une séquence entière,
    /// `[B, N, C]`. Renvoie `[B, N, C]`. Chaque position compare sa requête
    /// aux clés de celles qui ne la suivent pas.
    ///
    /// Avec `rope`, qui donne les angles des `N` positions, les requêtes et
    /// les clés sont tournées selon leur position avant d'être comparées. Le
    /// score d'une requête et d'une clé ne dépend alors que de la distance
    /// entre leurs positions : c'est par là, et non par un vecteur ajouté à
    /// l'entrée, que le modèle connaît l'ordre des tokens. Les valeurs ne
    /// sont pas tournées.
    pub fn forward(&self, tape: &Tape<B>, x: Var, rope: Option<Rope>) -> Var {
        let lent = self.keys_and_values(tape, x, rope);
        attend(tape, self.heads, &self.query, &self.proj, x, rope, lent)
    }

    /// Les clés et les valeurs de toutes les positions, chacune
    /// `[B, H, T, D]` : ce que cette couche offre à ses requêtes, et à
    /// celles des couches qui lui empruntent les siennes.
    pub fn keys_and_values(&self, tape: &Tape<B>, x: Var, rope: Option<Rope>) -> Lent {
        let mut keys = split_heads(tape, self.key.forward(tape, x), self.heads);
        let values = split_heads(tape, self.value.forward(tape, x), self.heads);
        if let Some(rope) = rope {
            keys = rope.rotate(tape, keys);
        }
        Lent { keys, values }
    }

    /// La même attention en génération : `x` contient les vecteurs de `R`
    /// nouveaux tokens, `[1, R, C]`, de séquences quelconques, que décrit
    /// `step`. Leurs clés et leurs valeurs sont écrites dans les tables de
    /// `cache`, puis chaque requête est comparée aux positions de sa
    /// séquence. Renvoie `[1, R, C]`, et les tables, pour les couches qui
    /// les empruntent.
    ///
    /// Les vecteurs restent rangés une ligne par token, `[R, H, D]` : c'est
    /// la forme des lignes des tables, et aucune transposition n'est
    /// nécessaire. Les clés y entrent déjà tournées : leur rotation ne
    /// dépend que de leur propre position.
    pub fn forward_cached(
        &self,
        tape: &Tape<B>,
        x: Var,
        cache: &mut AttentionCache<B>,
        step: &CacheStep,
    ) -> (Var, Lent) {
        let rows = |v: Var| per_token(tape, v, self.heads, step.rope);
        let keys = rows(self.key.forward(tape, x));
        let values = per_token(tape, self.value.forward(tape, x), self.heads, None);
        let lent = Lent {
            keys: cache.keys.write(tape, step, keys),
            values: cache.values.write(tape, step, values),
        };
        let out = attend_cached(tape, self.heads, &self.query, &self.proj, x, step, lent);
        (out, lent)
    }
}

module!(CausalSelfAttention {
    query,
    key,
    value,
    proj
});

/// Les clés et les valeurs qu'une couche d'attention a calculées, telles
/// qu'une couche suivante peut les lui emprunter : `[B, H, T, D]` chacune
/// sur une séquence entière, les tables du cache en génération.
#[derive(Clone, Copy)]
pub struct Lent {
    pub keys: Var,
    pub values: Var,
}

/// Découpe `C` en `H` têtes de taille `D`, puis amène l'axe des têtes
/// devant celui des positions : `[B, N, C] → [B, N, H, D] → [B, H, N, D]`.
/// Chaque tête devient une matrice `[N, D]` traitée indépendamment.
fn split_heads<B: Backend>(tape: &Tape<B>, v: Var, heads: usize) -> Var {
    let shape = tape.shape(v);
    let (batch, length, dim) = (shape[0], shape[1], shape[2]);
    tape.transpose(tape.reshape(v, &[batch, length, heads, dim / heads]), 1, 2)
}

/// L'attention proprement dite : les requêtes des `N` positions de `x`,
/// comparées aux clés de `lent`, mélangent ses valeurs. Renvoie `[B, N, C]`.
fn attend<B: Backend>(
    tape: &Tape<B>,
    heads: usize,
    query: &Linear<B>,
    proj: &Linear<B>,
    x: Var,
    rope: Option<Rope>,
    lent: Lent,
) -> Var {
    let shape = tape.shape(x);
    let (batch, length, dim) = (shape[0], shape[1], shape[2]);
    let head_dim = dim / heads;
    let mut q = split_heads(tape, query.forward(tape, x), heads);
    if let Some(rope) = rope {
        q = rope.rotate(tape, q);
    }

    // Le score de chaque requête face à chaque position connue, divisé par
    // `√D` : des scores d'ordre 1 quelle que soit la taille des têtes, sinon
    // le softmax saturerait. Le softmax causal donne ensuite un poids nul
    // aux positions qui suivent chaque requête, le futur est interdit, et
    // les poids mélangent les valeurs : `[B, H, N, D]`. Une seule opération
    // pour les trois étapes, que le GPU calcule sans ranger les scores.
    let scale = B::Scalar::from_f64(1.0 / (head_dim as f64).sqrt());
    let mixed = tape.attention(q, lent.keys, lent.values, scale);

    // Le chemin inverse du découpage pour revenir à `[B, N, C]`.
    let merged = tape.reshape(tape.transpose(mixed, 1, 2), &[batch, length, dim]);
    proj.forward(tape, merged)
}

/// Les vecteurs de `R` tokens, `[1, R, H × D]`, rangés une ligne par
/// token et par tête, `[R, H, D]`, et tournés par `rope` s'il y en a un.
/// La rotation se fait sur `[R × H, D]`, avec une table d'angles qui répète
/// chaque position pour toutes les têtes.
fn per_token<B: Backend>(tape: &Tape<B>, v: Var, heads: usize, rope: Option<Rope>) -> Var {
    let shape = tape.shape(v);
    let (tokens, dim) = (shape[1], shape[2]);
    let mut v = tape.reshape(v, &[tokens * heads, dim / heads]);
    if let Some(rope) = rope {
        v = rope.rotate(tape, v);
    }
    tape.reshape(v, &[tokens, heads, dim / heads])
}

/// L'attention de [`attend`] en génération : les requêtes des `R` tokens
/// de `x`, comparées aux positions de leurs séquences dans les tables de
/// `lent`. Renvoie `[1, R, C]`.
fn attend_cached<B: Backend>(
    tape: &Tape<B>,
    heads: usize,
    query: &Linear<B>,
    proj: &Linear<B>,
    x: Var,
    step: &CacheStep,
    lent: Lent,
) -> Var {
    let shape = tape.shape(x);
    let (tokens, dim) = (shape[1], shape[2]);
    let head_dim = dim / heads;
    let q = per_token(tape, query.forward(tape, x), heads, step.rope);
    let scale = B::Scalar::from_f64(1.0 / (head_dim as f64).sqrt());
    let mixed = tape.cached_attention(q, lent.keys, lent.values, step, scale, None);
    proj.forward(tape, tape.reshape(mixed, &[1, tokens, dim]))
}

/// Une attention sans clés ni valeurs à elle : elle pose ses propres
/// requêtes aux clés d'une couche précédente, et mélange ses valeurs.
///
/// Elle peut ainsi regarder ailleurs que la couche à qui elle emprunte,
/// mais parmi ce que cette couche a choisi d'offrir. En échange, elle n'a
/// que la moitié des paramètres d'une attention complète, et rien à
/// garder dans le cache de génération.
pub struct BorrowedAttention<B: Backend> {
    heads: usize,
    pub query: Linear<B>,
    /// Recombine les sorties des têtes.
    pub proj: Linear<B>,
}

impl<B: Backend> BorrowedAttention<B> {
    pub fn new(backend: &B, dim: usize, heads: usize, proj_std: f64, rng: &mut impl Rng) -> Self {
        assert!(
            dim.is_multiple_of(heads),
            "la dimension {dim} n'est pas divisible par {heads} têtes",
        );
        Self {
            heads,
            query: Linear::new(backend, dim, dim, rng),
            proj: Linear::with_std(backend, dim, dim, proj_std, rng),
        }
    }

    /// `x` contient les vecteurs de `N` positions, `[B, N, C]` ; `lent`,
    /// les clés et les valeurs de toutes les positions connues.
    pub fn forward(&self, tape: &Tape<B>, x: Var, rope: Option<Rope>, lent: Lent) -> Var {
        attend(tape, self.heads, &self.query, &self.proj, x, rope, lent)
    }

    /// La même attention en génération : `lent` porte les tables du cache
    /// de la couche prêteuse.
    pub fn forward_cached(&self, tape: &Tape<B>, x: Var, step: &CacheStep, lent: Lent) -> Var {
        attend_cached(tape, self.heads, &self.query, &self.proj, x, step, lent)
    }
}

module!(BorrowedAttention { query, proj });

/// L'attention d'un bloc, selon l'architecture choisie.
pub enum Attention<B: Backend> {
    Standard(CausalSelfAttention<B>),
    Latent(LatentAttention<B>),
    Borrowed(BorrowedAttention<B>),
}

impl<B: Backend> Attention<B> {
    /// Avec `borrows`, la couche n'a pas de clés ni de valeurs à elle :
    /// elle prend celles de la dernière couche qui en a calculé.
    pub fn new(
        backend: &B,
        kind: AttentionKind,
        borrows: bool,
        dim: usize,
        heads: usize,
        proj_std: f64,
        rng: &mut impl Rng,
    ) -> Self {
        match kind {
            AttentionKind::Standard if borrows => {
                Self::Borrowed(BorrowedAttention::new(backend, dim, heads, proj_std, rng))
            }
            AttentionKind::Standard => {
                Self::Standard(CausalSelfAttention::new(backend, dim, heads, proj_std, rng))
            }
            AttentionKind::Latent(dims) => Self::Latent(LatentAttention::new(
                backend, dim, heads, dims, proj_std, rng,
            )),
        }
    }

    /// `lent` passe de couche en couche : une attention classique y laisse
    /// ses clés et ses valeurs, une attention qui emprunte les y prend.
    ///
    /// Avec `cache`, c'est la génération : `x` contient les nouveaux tokens
    /// que décrit le pas, et ce que la couche garde du passé est dans les
    /// tables.
    pub fn forward(
        &self,
        tape: &Tape<B>,
        x: Var,
        rope: Option<Rope>,
        cache: Option<(&mut AttentionCache<B>, &CacheStep)>,
        lent: &mut Option<Lent>,
    ) -> Var {
        match (self, cache) {
            (Self::Standard(attention), None) => {
                let own = attention.keys_and_values(tape, x, rope);
                *lent = Some(own);
                attend(
                    tape,
                    attention.heads,
                    &attention.query,
                    &attention.proj,
                    x,
                    rope,
                    own,
                )
            }
            (Self::Standard(attention), Some((cache, step))) => {
                let (out, own) = attention.forward_cached(tape, x, cache, step);
                *lent = Some(own);
                out
            }
            (Self::Latent(attention), None) => {
                let rope = rope.expect("l'attention latente demande les angles de RoPE");
                attention.forward(tape, x, rope)
            }
            (Self::Latent(attention), Some((cache, step))) => {
                attention.forward_cached(tape, x, cache, step)
            }
            (Self::Borrowed(attention), cache) => {
                let lent = lent.expect("aucune couche précédente n'a calculé de clés");
                match cache {
                    None => attention.forward(tape, x, rope, lent),
                    Some((_, step)) => attention.forward_cached(tape, x, step, lent),
                }
            }
        }
    }

    /// Les couches linéaires de l'attention.
    pub fn linears_mut(&mut self) -> Vec<&mut Linear<B>> {
        match self {
            Self::Standard(a) => vec![&mut a.query, &mut a.key, &mut a.value, &mut a.proj],
            Self::Latent(a) => vec![&mut a.proj],
            Self::Borrowed(a) => vec![&mut a.query, &mut a.proj],
        }
    }

    /// La couche qui écrit dans le flux résiduel.
    pub fn proj(&self) -> &Linear<B> {
        match self {
            Self::Standard(attention) => &attention.proj,
            Self::Latent(attention) => &attention.proj,
            Self::Borrowed(attention) => &attention.proj,
        }
    }
}

module!(
    enum Attention {
        Standard,
        Latent,
        Borrowed,
    }
);
