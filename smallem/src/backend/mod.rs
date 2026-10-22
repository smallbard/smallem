//! Les opérations primitives du modèle, et leurs implémentations.
//!
//! [`Backend`] est le contrat commun : tout ce qui est écrit au-dessus
//! (autograd, couches, modèle) ne passe que par lui et tourne donc aussi
//! bien sur CPU que sur GPU.

pub mod cpu;
pub mod fast;
pub mod float;
pub mod ops;
pub mod profile;
pub mod shape;

use rand::{Rng, RngExt};

pub use float::Float;
use ops::{
    Affine, CachedAttention, CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu, Normalize,
    QuantizedMatmul, RmsNormalize, RotatePairs, Run, Silu, TransposedMatmul,
};

/// Tire un nombre selon la loi normale centrée réduite, par la méthode de
/// Box-Muller : deux nombres uniformes donnent un nombre gaussien.
pub fn standard_normal(rng: &mut impl Rng) -> f64 {
    // `1 - u` est dans `(0, 1]` : cela évite `ln(0)`.
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Les réglages d'un pas d'AdamW, tels que les backends les reçoivent.
///
/// Les corrections de biais dépendent du numéro du pas : l'optimiseur les
/// calcule sur l'hôte, une fois pour tous les paramètres.
#[derive(Clone, Copy, Debug)]
pub struct AdamWStep {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    /// 0 pour les paramètres qu'on ne veut pas faire décroître.
    pub weight_decay: f64,
    /// `1 - β₁ᵗ`
    pub bias_correction1: f64,
    /// `1 - β₂ᵗ`
    pub bias_correction2: f64,
    /// Facteur appliqué au gradient avant tout calcul : moins de 1 quand
    /// l'écrêtage de la norme est actif, 1 sinon.
    pub grad_scale: f64,
}

impl AdamWStep {
    /// Les mêmes réglages, convertis dans le type des éléments.
    pub fn scalars<F: Float>(&self) -> float::AdamWScalars<F> {
        float::AdamWScalars {
            learning_rate: F::from_f64(self.learning_rate),
            beta1: F::from_f64(self.beta1),
            beta2: F::from_f64(self.beta2),
            epsilon: F::from_f64(self.epsilon),
            weight_decay: F::from_f64(self.weight_decay),
            correction1: F::from_f64(self.bias_correction1),
            correction2: F::from_f64(self.bias_correction2),
            grad_scale: F::from_f64(self.grad_scale),
        }
    }
}

/// Un endroit où vivent des tenseurs, et les opérations élémentaires qu'on
/// peut leur appliquer : les primitives.
///
/// Chaque primitive renvoie un nouveau tenseur sans toucher à ses opérandes,
/// à trois exceptions près, tout à la fin du trait : l'écriture dans le
/// cache de génération et les pas d'optimiseur modifient un tenseur sur
/// place.
///
/// La liste est courte et stable. Tout ce qui se construit au-dessus (GELU,
/// les normalisations, le softmax causal, la perte) est une *opération*, et
/// vit dans [`ops`].
///
/// Une erreur de forme est une erreur de programmation : les primitives
/// paniquent avec un message plutôt que de renvoyer un `Result`.
pub trait Primitives {
    type Scalar: Float;
    /// Cloner un tenseur doit rester raisonnable : l'autograd le fait pour
    /// propager un même gradient vers plusieurs parents.
    type Tensor: Clone;

    // --- Échanges avec l'hôte ---

    // Le nom suit `Vec::from`, mais c'est le backend qui décide où vont les
    // données, d'où le `&self`.
    //
    /// Crée un tenseur à partir de données de l'hôte. Sur GPU, c'est un
    /// transfert, et il attend que la carte ait vidé sa file. Tous les
    /// constructeurs ci-dessous passent par là.
    #[allow(clippy::wrong_self_convention)]
    fn from_vec(&self, data: Vec<Self::Scalar>, shape: &[usize]) -> Self::Tensor;

    /// Copie les données vers l'hôte. Sur GPU, c'est un transfert et une
    /// synchronisation : à réserver à la lecture des résultats.
    fn to_vec(&self, tensor: &Self::Tensor) -> Vec<Self::Scalar>;

    fn shape<'a>(&self, tensor: &'a Self::Tensor) -> &'a [usize];

    // --- Constructeurs, tous bâtis sur `from_vec` ---

    fn full(&self, shape: &[usize], value: Self::Scalar) -> Self::Tensor {
        self.from_vec(vec![value; shape.iter().product()], shape)
    }

    fn zeros(&self, shape: &[usize]) -> Self::Tensor {
        self.full(shape, Self::Scalar::ZERO)
    }

    fn ones(&self, shape: &[usize]) -> Self::Tensor {
        self.full(shape, Self::Scalar::ONE)
    }

    /// Tenseur à un seul élément, de forme `[]`, qui s'étire à toute forme.
    fn scalar(&self, value: Self::Scalar) -> Self::Tensor {
        self.from_vec(vec![value], &[])
    }

    /// `[0, 1, …, len - 1]`.
    fn arange(&self, len: usize) -> Self::Tensor {
        let data = (0..len).map(|i| Self::Scalar::from_f64(i as f64)).collect();
        self.from_vec(data, &[len])
    }

    /// Tirages d'une loi normale centrée réduite. Ils sont faits sur l'hôte :
    /// avec la même graine, tous les backends reçoivent les mêmes valeurs.
    fn randn(&self, shape: &[usize], rng: &mut impl Rng) -> Self::Tensor
    where
        Self: Sized,
    {
        let len: usize = shape.iter().product();
        let data = (0..len)
            .map(|_| Self::Scalar::from_f64(standard_normal(rng)))
            .collect();
        self.from_vec(data, shape)
    }

    /// Valeur d'un tenseur à un seul élément.
    fn item(&self, tensor: &Self::Tensor) -> Self::Scalar {
        let data = self.to_vec(tensor);
        assert_eq!(
            data.len(),
            1,
            "item() sur un tenseur de forme {:?}",
            self.shape(tensor),
        );
        data[0]
    }

    // --- Élément par élément, avec broadcasting ---

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn sub(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn div(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    fn neg(&self, a: &Self::Tensor) -> Self::Tensor;
    fn exp(&self, a: &Self::Tensor) -> Self::Tensor;
    fn ln(&self, a: &Self::Tensor) -> Self::Tensor;
    fn sqrt(&self, a: &Self::Tensor) -> Self::Tensor;
    fn tanh(&self, a: &Self::Tensor) -> Self::Tensor;

    // --- Réductions ---

    /// Somme le long de `axis`. Avec `keepdim`, l'axe réduit reste présent
    /// avec la taille 1.
    fn sum(&self, a: &Self::Tensor, axis: usize, keepdim: bool) -> Self::Tensor;

    /// Somme de tous les éléments, de forme `[]`.
    fn sum_all(&self, a: &Self::Tensor) -> Self::Tensor;

    fn mean(&self, a: &Self::Tensor, axis: usize, keepdim: bool) -> Self::Tensor {
        let n = self.shape(a)[axis];
        let inverse = self.scalar(Self::Scalar::from_f64(1.0 / n as f64));
        self.mul(&self.sum(a, axis, keepdim), &inverse)
    }

    // --- Changements de forme ---

    fn reshape(&self, a: &Self::Tensor, shape: &[usize]) -> Self::Tensor;
    fn transpose(&self, a: &Self::Tensor, axis_a: usize, axis_b: usize) -> Self::Tensor;

    /// Étire `a` jusqu'à la forme `shape` en recopiant ses éléments, selon
    /// les règles du broadcasting. C'est l'inverse d'une somme sur un axe.
    fn expand(&self, a: &Self::Tensor, shape: &[usize]) -> Self::Tensor;

    // --- Algèbre linéaire ---

    /// Produit matriciel sur les deux dernières dimensions ; les précédentes
    /// sont des dimensions de lot, soumises au broadcasting.
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    // --- Softmax, sur le dernier axe ---

    fn softmax(&self, a: &Self::Tensor) -> Self::Tensor;
    fn log_softmax(&self, a: &Self::Tensor) -> Self::Tensor;

    // --- Sélection ---

    /// Remplace par `value` les éléments où `mask` est non nul. Le masque
    /// s'étire à la forme de `a`, pas l'inverse.
    fn masked_fill(
        &self,
        a: &Self::Tensor,
        mask: &Self::Tensor,
        value: Self::Scalar,
    ) -> Self::Tensor;

    /// Sélectionne des tranches le long de la première dimension :
    /// `[V, …]` et `N` indices donnent `[N, …]`.
    fn gather_rows(&self, a: &Self::Tensor, indices: &[u32]) -> Self::Tensor;

    /// Inverse de [`Primitives::gather_rows`] : part d'un tenseur nul de
    /// `num_rows` lignes et ajoute la ligne `r` de `a` à sa ligne
    /// `indices[r]`. Un indice répété reçoit la somme de toutes ses lignes.
    fn scatter_add_rows(&self, a: &Self::Tensor, indices: &[u32], num_rows: usize) -> Self::Tensor;

    /// Met `b` à la suite de `a` le long de la première dimension :
    /// `[M, …]` et `[N, …]` donnent `[M + N, …]`. Les autres dimensions
    /// doivent être les mêmes.
    fn concat_rows(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    // --- Mises à jour sur place ---
    //
    // Les seules opérations qui modifient un tenseur existant : l'écriture
    // dans le cache de génération, et les pas d'optimiseur, qui remplacent
    // les paramètres au lieu d'en créer de nouveaux. Le tenseur modifié ne
    // doit être référencé nulle part ailleurs, en particulier plus par une
    // tape.

    /// Écrit la ligne `r` de `rows` à la ligne `indices[r]` de `table`, sur
    /// place : `[N, …]` dans `[V, …]`. Les autres lignes de `table` ne
    /// changent pas. C'est ainsi que le cache de génération reçoit les clés
    /// et les valeurs des nouvelles positions, sans recopier les anciennes.
    ///
    /// Les indices sont des entiers rangés dans un tenseur ordinaire, `[N]` :
    /// ils servent à toutes les couches d'une même passe, et ne traversent
    /// ainsi qu'une fois vers la carte. Un `f32` représente exactement tout
    /// entier jusqu'à 2²⁴, soit seize millions de lignes.
    fn write_rows(&self, table: &mut Self::Tensor, indices: &Self::Tensor, rows: &Self::Tensor);

    /// Descente de gradient : `param -= learning_rate × grad`.
    fn sgd_step(&self, param: &mut Self::Tensor, grad: &Self::Tensor, learning_rate: f64);

    /// Un pas d'AdamW, qui met à jour le paramètre et ses deux moments,
    /// élément par élément : voir [`float::adamw_update`].
    fn adamw_step(
        &self,
        param: &mut Self::Tensor,
        grad: &Self::Tensor,
        first_moment: &mut Self::Tensor,
        second_moment: &mut Self::Tensor,
        step: &AdamWStep,
    );
}

/// Un backend complet : les primitives, et toutes les opérations dont un
/// modèle a besoin.
///
/// C'est ce trait que demandent l'autograd, les couches et les modèles. Il
/// n'a rien à implémenter : un type qui fournit les primitives et sait
/// exécuter chaque opération de la liste est un `Backend`.
pub trait Backend:
    Primitives
    + Run<Affine>
    + Run<FusedAttention>
    + Run<CachedAttention>
    + Run<Gelu>
    + Run<Silu>
    + Run<Normalize>
    + Run<RmsNormalize>
    + Run<CausalSoftmax>
    + Run<RotatePairs>
    + Run<Concat>
    + Run<CrossEntropy>
    + Run<TransposedMatmul>
    + Run<QuantizedMatmul>
{
}

impl<B> Backend for B where
    B: Primitives
        + Run<Affine>
        + Run<FusedAttention>
        + Run<CachedAttention>
        + Run<Gelu>
        + Run<Silu>
        + Run<Normalize>
        + Run<RmsNormalize>
        + Run<CausalSoftmax>
        + Run<RotatePairs>
        + Run<Concat>
        + Run<CrossEntropy>
        + Run<TransposedMatmul>
        + Run<QuantizedMatmul>
{
}
