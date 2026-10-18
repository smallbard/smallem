//! L'attention latente (MLA, *Multi-head Latent Attention*) de DeepSeek :
//! une attention dont le cache de génération est beaucoup plus petit.

use rand::Rng;

use super::{LatentDims, Linear, Projection, RmsNorm, Rope, module};
use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};
use crate::cache::{AttentionCache, CacheStep};

/// Self-attention causale dont les clés et les valeurs de toutes les têtes
/// sortent d'un seul petit vecteur par position, le *latent*.
///
/// Dans l'attention classique, le cache de génération garde pour chaque
/// position une clé et une valeur par tête : `2 × C` nombres par couche.
/// Ici, chaque position est d'abord compressée en un vecteur de taille
/// `d_c`, bien plus petit que `C`, et c'est lui seul qu'on garde :
///
/// ```text
/// c    = rms(x · W_latent)        [d_c]      le latent, gardé dans le cache
/// k    = c · W_clé                [H × D]    les clés de toutes les têtes
/// v    = c · W_valeur             [H × D]    les valeurs de toutes les têtes
/// ```
///
/// Les requêtes passent par leur propre latent, plus large. Elles ne sont
/// jamais gardées : la compression ne sert ici qu'à économiser des
/// paramètres.
///
/// ```text
/// c_q  = rms(x · W_latent_requête)
/// q    = c_q · W_requête          [H × D]
/// ```
///
/// **RoPE découplé.** La rotation de RoPE dépend de la position, et elle ne
/// commute pas avec `W_clé` : tourner les clés obligerait à les garder
/// tournées, donc à garder les clés elles-mêmes, et le latent ne servirait
/// plus à rien. La position passe donc par un canal à part. Chaque clé et
/// chaque requête reçoit une seconde partie, de taille `d_r`, qui porte
/// seule la rotation :
///
/// ```text
/// k_r  = rotation(x · W_clé_tournée)          [d_r]      une pour toutes les têtes
/// q_r  = rotation(c_q · W_requête_tournée)    [H × d_r]  une par tête
/// ```
///
/// La clé d'une tête est la concaténation `[k ; k_r]`, sa requête
/// `[q ; q_r]`. Le produit scalaire de deux vecteurs concaténés est la
/// somme des produits de leurs parties :
///
/// ```text
/// score = [q ; q_r] · [k ; k_r] / √(D + d_r)
///       = (q · k + q_r · k_r) / √(D + d_r)
/// ```
///
/// Le premier terme compare les contenus, le second les positions. Le cache
/// garde `c` et `k_r` : `d_c + d_r` nombres par position.
///
/// Les deux écritures du score donnent le même résultat. Celle qui
/// concatène est la plus rapide sur GPU : un seul produit matriciel au lieu
/// de deux, et pas de somme de deux tableaux de scores `[B, H, N, T]`, les
/// plus gros tenseurs de la couche.
pub struct LatentAttention<B: Backend> {
    heads: usize,
    dims: LatentDims,
    /// `C → d_q`
    pub query_down: Projection<B>,
    pub query_norm: RmsNorm<B>,
    /// `d_q → C`
    pub query_up: Projection<B>,
    /// `d_q → H × d_r`
    pub query_rope: Projection<B>,
    /// `C → d_c`
    pub latent_down: Projection<B>,
    pub latent_norm: RmsNorm<B>,
    /// `d_c → C`
    pub key_up: Projection<B>,
    /// `d_c → C`
    pub value_up: Projection<B>,
    /// `C → d_r`
    pub key_rope: Projection<B>,
    /// Recombine les sorties des têtes.
    pub proj: Linear<B>,
}

impl<B: Backend> LatentAttention<B> {
    /// `proj_std` est l'écart-type d'initialisation de la projection de
    /// sortie, plus petit que les autres (voir [`Block::new`](super::Block::new)).
    pub fn new(
        backend: &B,
        dim: usize,
        heads: usize,
        dims: LatentDims,
        proj_std: f64,
        rng: &mut impl Rng,
    ) -> Self {
        assert!(
            dim.is_multiple_of(heads),
            "la dimension {dim} n'est pas divisible par {heads} têtes",
        );
        assert!(
            dims.latent > 0 && dims.query_latent > 0 && dims.rope > 0,
            "attention latente : tailles nulles dans {dims:?}",
        );
        assert!(
            dims.rope.is_multiple_of(2),
            "RoPE demande une taille paire, reçu {}",
            dims.rope,
        );
        Self {
            heads,
            dims,
            query_down: Projection::new(backend, dim, dims.query_latent, rng),
            query_norm: RmsNorm::new(backend, dims.query_latent),
            query_up: Projection::new(backend, dims.query_latent, dim, rng),
            query_rope: Projection::new(backend, dims.query_latent, heads * dims.rope, rng),
            latent_down: Projection::new(backend, dim, dims.latent, rng),
            latent_norm: RmsNorm::new(backend, dims.latent),
            key_up: Projection::new(backend, dims.latent, dim, rng),
            value_up: Projection::new(backend, dims.latent, dim, rng),
            key_rope: Projection::new(backend, dim, dims.rope, rng),
            proj: Linear::with_std(backend, dim, dim, proj_std, rng),
        }
    }

    /// `x` contient les vecteurs de `N` positions d'une séquence entière,
    /// `[B, N, C]`. Renvoie `[B, N, C]`. `rope` donne les angles de ces
    /// positions, pour des vecteurs de taille `d_r`.
    pub fn forward(&self, tape: &Tape<B>, x: Var, rope: Rope) -> Var {
        let shape = tape.shape(x);
        let (batch, length, dim) = (shape[0], shape[1], shape[2]);
        let heads = self.heads;
        let head_dim = dim / heads;
        let LatentDims { rope: rope_dim, .. } = self.dims;

        // `[B, n, H × d] → [B, n, H, d] → [B, H, n, d]` : une matrice par
        // tête, comme dans l'attention classique.
        let split = |v: Var, positions: usize, size: usize| {
            tape.transpose(tape.reshape(v, &[batch, positions, heads, size]), 1, 2)
        };

        // Les requêtes des `N` positions, en deux parties.
        let query_latent = self
            .query_norm
            .forward(tape, self.query_down.forward(tape, x));
        let q = split(self.query_up.forward(tape, query_latent), length, head_dim);
        let q_rope = split(
            self.query_rope.forward(tape, query_latent),
            length,
            rope_dim,
        );
        let q_rope = rope.rotate(tape, q_rope);

        // Ce que ces positions offrent aux suivantes : leur latent, et la
        // partie tournée de leur clé. Celle-ci est la même pour toutes les
        // têtes : `[B, 1, N, d_r]`.
        let latent = self
            .latent_norm
            .forward(tape, self.latent_down.forward(tape, x));
        let k_rope = tape.reshape(
            self.key_rope.forward(tape, x),
            &[batch, 1, length, rope_dim],
        );
        let k_rope = rope.rotate(tape, k_rope);

        // Les clés et les valeurs de toutes les têtes :
        // `[B, N, d_c] → [B, N, C] → [B, H, N, D]`.
        let k = split(self.key_up.forward(tape, latent), length, head_dim);
        let v = split(self.value_up.forward(tape, latent), length, head_dim);

        // La clé complète d'une tête est sa partie « contenu » suivie de la
        // partie tournée, recopiée pour chaque tête ; de même la requête.
        // `[B, H, T, D + d_r]` et `[B, H, N, D + d_r]`.
        let k_rope = tape.expand(k_rope, &[batch, heads, length, rope_dim]);
        let k = tape.concat(k, k_rope);
        let q = tape.concat(q, q_rope);

        // L'attention elle-même, comme dans `CausalSelfAttention`. Les
        // vecteurs comparés ont `D + d_r` composantes en tout, les valeurs
        // `D` : `[B, H, N, D]`, puis retour à `[B, N, C]`.
        let scale = B::Scalar::from_f64(1.0 / ((head_dim + rope_dim) as f64).sqrt());
        let mixed = tape.attention(q, k, v, scale);
        let merged = tape.reshape(tape.transpose(mixed, 1, 2), &[batch, length, dim]);
        self.proj.forward(tape, merged)
    }

    /// La même attention en génération, sur les `R` nouveaux tokens de
    /// `x`, `[1, R, C]`, que décrit `step`.
    ///
    /// Recalculer les clés et les valeurs de toutes les positions à partir
    /// de leurs latents, à chaque token, coûterait un produit matriciel par
    /// position gardée. On s'en passe en **absorbant** les matrices dans la
    /// requête et dans la sortie. Pour la tête `h`, avec `W_k` et `W_v` ses
    /// parts de `W_clé` et de `W_valeur` :
    ///
    /// ```text
    /// q · k   = q · (c · W_k) = (q · W_kᵀ) · c
    /// Σ p · v = Σ p · (c · W_v) = (Σ p · c) · W_v
    /// ```
    ///
    /// La requête passe dans l'espace des latents, `q · W_kᵀ` de taille
    /// `d_c`, et l'attention compare directement les requêtes aux latents
    /// gardés, puis mélange ces latents. Une seule tête de clés, le latent
    /// suivi de la partie tournée, sert toutes les têtes de requêtes ; et
    /// le latent sert aussi de valeur, d'où une seule table. C'est ainsi
    /// que DeepSeek fait tourner l'attention latente en génération.
    pub fn forward_cached(
        &self,
        tape: &Tape<B>,
        x: Var,
        cache: &mut AttentionCache<B>,
        step: &CacheStep,
    ) -> Var {
        let shape = tape.shape(x);
        let (tokens, dim) = (shape[1], shape[2]);
        let heads = self.heads;
        let head_dim = dim / heads;
        let LatentDims {
            latent: latent_dim,
            rope: rope_dim,
            ..
        } = self.dims;
        let rope = step
            .rope
            .expect("l'attention latente demande les angles de RoPE");
        let rope_single = step
            .rope_single
            .expect("l'attention latente demande les angles de RoPE");

        // Une matrice `[C', H × D]` vue comme une matrice `[d', D]` par
        // tête : `[H, d', D]`.
        let per_head = |weight: &Projection<B>, rows: usize| {
            let weight = tape.reshape(weight.weight.var(tape), &[rows, heads, head_dim]);
            tape.transpose(weight, 0, 1)
        };

        // Les requêtes, `[R, H, d_c + d_r]` : la partie « contenu »
        // absorbée par `W_clé`, puis la partie tournée.
        let query_latent = self
            .query_norm
            .forward(tape, self.query_down.forward(tape, x));
        let q = tape.reshape(
            self.query_up.forward(tape, query_latent),
            &[tokens, heads, head_dim],
        );
        let key_up = per_head(&self.key_up, latent_dim);
        let absorbed = tape.matmul(tape.transpose(q, 0, 1), tape.transpose(key_up, 1, 2));
        let absorbed = tape.transpose(absorbed, 0, 1);
        let q_rope = tape.reshape(
            self.query_rope.forward(tape, query_latent),
            &[tokens * heads, rope_dim],
        );
        let q_rope = tape.reshape(rope.rotate(tape, q_rope), &[tokens, heads, rope_dim]);
        let q = tape.concat(absorbed, q_rope);

        // Ce que chaque token offre : son latent suivi de la partie tournée
        // de sa clé, une seule « tête », `[R, 1, d_c + d_r]`.
        let latent = self
            .latent_norm
            .forward(tape, self.latent_down.forward(tape, x));
        let latent = tape.reshape(latent, &[tokens, latent_dim]);
        let k_rope = tape.reshape(self.key_rope.forward(tape, x), &[tokens, rope_dim]);
        let k_rope = rope_single.rotate(tape, k_rope);
        let line = tape.reshape(
            tape.concat(latent, k_rope),
            &[tokens, 1, latent_dim + rope_dim],
        );
        let table = cache.latents.write(tape, step, line);

        // L'attention mélange les latents, `[R, H, d_c]`, que `W_valeur`
        // ramène aux valeurs de chaque tête, `[R, H, D]`.
        let scale = B::Scalar::from_f64(1.0 / ((head_dim + rope_dim) as f64).sqrt());
        let mixed = tape.cached_attention(q, table, table, step, scale, Some(latent_dim));
        let value_up = per_head(&self.value_up, latent_dim);
        let mixed = tape.matmul(tape.transpose(mixed, 0, 1), value_up);
        let merged = tape.reshape(tape.transpose(mixed, 0, 1), &[1, tokens, dim]);
        self.proj.forward(tape, merged)
    }
}

module!(LatentAttention {
    query_down,
    query_norm,
    query_up,
    query_rope,
    latent_down,
    latent_norm,
    key_up,
    value_up,
    key_rope,
    proj
});
