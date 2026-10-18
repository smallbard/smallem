//! Les paramètres d'un réseau, et le moyen de les parcourir.
//!
//! Un paramètre vit en dehors des tapes, d'un pas d'entraînement à l'autre.
//! À chaque pas, les couches l'inscrivent comme feuille sur la tape du
//! moment ; après la rétropropagation, l'optimiseur retrouve son gradient
//! par la même tape.

use std::cell::{Cell, RefCell};

use rand::Rng;

use crate::autograd::{Gradients, Tape, Var};
use crate::backend::{Backend, Float, standard_normal};
use crate::quantize::{Format, Quantized};

/// Un tenseur appris : son gradient est calculé à chaque pas, et
/// l'optimiseur le modifie.
pub struct Parameter<B: Backend> {
    value: B::Tensor,
    /// Faux pour les biais et les gains de normalisation, que la
    /// décroissance des poids ne doit pas toucher.
    decay: bool,
    /// La feuille de ce paramètre sur la dernière tape où il a servi, avec
    /// le numéro de cette tape.
    binding: Cell<Option<(u64, Var)>>,
    /// Les poids quantifiés, pour une matrice qui l'a été (voir
    /// [`Parameter::quantize`]).
    quantized: Option<Box<QuantizedWeight<B>>>,
    /// Ce qu'on apprend des entrées de cette matrice pendant un étalonnage
    /// (voir [`Parameter::start_calibration`]).
    calibration: RefCell<Option<Box<Calibration<B>>>>,
    /// Vrai pour un paramètre qui n'apprend plus : il entre sur les tapes
    /// comme une constante, sans gradient.
    frozen: bool,
    /// Facteur du taux d'apprentissage de ce paramètre : 1, sauf sous μP
    /// (voir [`Gpt::apply_mup`](crate::model::Gpt::apply_mup)).
    lr_scale: f64,
}

/// Ce qu'un étalonnage retient des entrées `x` `[…, K]` d'une matrice de
/// poids, sur toutes les passes qui l'ont traversée.
pub struct Calibration<B: Backend> {
    /// `Σ xᵀ · x`, `[K, K]` : la matrice que GPTQ appelle le hessien de la
    /// couche, au facteur 2 près.
    pub hessian: Option<B::Tensor>,
    /// Nombre de vecteurs `x` vus.
    pub rows: usize,
    /// Avec `host`, pour chaque composante de `x` : sa plus grande valeur
    /// absolue, et combien de fois elle a dépassé 6 en valeur absolue.
    pub largest: Vec<f64>,
    pub beyond_six: Vec<u64>,
    /// Vrai pour rapatrier chaque `x` sur l'hôte et en tenir les
    /// statistiques ci-dessus : lent, pour une analyse.
    pub host: bool,
}

/// Une matrice de poids quantifiée, telle que les couches la passent à
/// l'opération [`QuantizedMatmul`](crate::backend::ops::QuantizedMatmul).
pub struct QuantizedWeight<B: Backend> {
    pub format: Format,
    /// `[groupes × mots par groupe, N]`.
    pub words: B::Tensor,
    /// `[groupes, N]`.
    pub scales: B::Tensor,
}

impl<B: Backend> QuantizedWeight<B> {
    /// Les codes et les échelles de `quantized`, dans des tenseurs du
    /// backend.
    pub fn new(backend: &B, quantized: &Quantized) -> Self {
        let (_, groups, per_group) = Quantized::layout(quantized.format, quantized.rows);
        let scalars = |values: &[f32]| {
            values
                .iter()
                .map(|&v| B::Scalar::from_f64(f64::from(v)))
                .collect()
        };
        Self {
            format: quantized.format,
            words: backend.from_vec(
                scalars(&quantized.words),
                &[groups * per_group, quantized.cols],
            ),
            scales: backend.from_vec(scalars(&quantized.scales), &[groups, quantized.cols]),
        }
    }
}

impl<B: Backend> Parameter<B> {
    pub fn new(value: B::Tensor, decay: bool) -> Self {
        Self {
            value,
            decay,
            binding: Cell::new(None),
            quantized: None,
            calibration: RefCell::new(None),
            frozen: false,
            lr_scale: 1.0,
        }
    }

    /// Poids tirés d'une loi normale d'écart-type `std`, soumis à la
    /// décroissance des poids.
    pub fn normal(backend: &B, shape: &[usize], std: f64, rng: &mut impl Rng) -> Self {
        let len: usize = shape.iter().product();
        let data = (0..len)
            .map(|_| B::Scalar::from_f64(std * standard_normal(rng)))
            .collect();
        Self::new(backend.from_vec(data, shape), true)
    }

    /// Paramètre nul, hors décroissance des poids : un biais.
    pub fn zeros(backend: &B, shape: &[usize]) -> Self {
        Self::new(backend.zeros(shape), false)
    }

    /// Paramètre rempli de 1, hors décroissance des poids : un gain.
    pub fn ones(backend: &B, shape: &[usize]) -> Self {
        Self::new(backend.ones(shape), false)
    }

    /// Exclut ce paramètre de la décroissance des poids.
    pub fn without_decay(mut self) -> Self {
        self.decay = false;
        self
    }

    pub fn value(&self) -> &B::Tensor {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut B::Tensor {
        &mut self.value
    }

    pub fn decay(&self) -> bool {
        self.decay
    }

    /// Gèle ou dégèle ce paramètre (voir le champ `frozen`).
    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
        self.binding.set(None);
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Le facteur du taux d'apprentissage de ce paramètre.
    pub fn lr_scale(&self) -> f64 {
        self.lr_scale
    }

    pub fn set_lr_scale(&mut self, scale: f64) {
        self.lr_scale = scale;
    }

    /// Quantifie cette matrice `[K, N]` au format `format`, en arrondissant
    /// chaque poids au code le plus proche. Sa valeur devient la valeur
    /// déquantifiée : tout calcul qui la lit voit les poids quantifiés.
    /// Rend la matrice quantifiée.
    pub fn quantize(&mut self, backend: &B, format: Format) -> Quantized {
        let shape = backend.shape(&self.value).to_vec();
        assert_eq!(
            shape.len(),
            2,
            "seule une matrice se quantifie, reçu {shape:?}"
        );
        let values: Vec<f64> = backend
            .to_vec(&self.value)
            .iter()
            .map(|v| v.to_f64())
            .collect();
        let quantized = Quantized::round_to_nearest(&values, shape[0], shape[1], format);
        self.set_quantized(backend, &quantized);
        quantized
    }

    /// Remplace cette matrice par des poids déjà quantifiés.
    pub fn set_quantized(&mut self, backend: &B, quantized: &Quantized) {
        let dequantized = quantized
            .dequantize()
            .into_iter()
            .map(B::Scalar::from_f64)
            .collect();
        self.value = backend.from_vec(dequantized, &[quantized.rows, quantized.cols]);
        self.quantized = Some(Box::new(QuantizedWeight::new(backend, quantized)));
        self.binding.set(None);
    }

    /// Remplace la valeur de ce paramètre.
    pub fn set_value(&mut self, value: B::Tensor) {
        self.value = value;
        self.binding.set(None);
    }

    /// Commence à observer les entrées de cette matrice : chaque appel à
    /// [`Parameter::observe`] les comptera. Avec `host`, les statistiques
    /// par composante sont tenues aussi.
    pub fn start_calibration(&self, host: bool) {
        *self.calibration.borrow_mut() = Some(Box::new(Calibration {
            hessian: None,
            rows: 0,
            largest: Vec::new(),
            beyond_six: Vec::new(),
            host,
        }));
    }

    /// Arrête l'observation et rend ce qu'elle a appris.
    pub fn take_calibration(&self) -> Option<Calibration<B>> {
        self.calibration
            .borrow_mut()
            .take()
            .map(|calibration| *calibration)
    }

    /// Compte les entrées `x` `[…, K]` de cette matrice, si un étalonnage
    /// est en cours. Les couches l'appellent avant leur produit.
    pub fn observe(&self, tape: &Tape<B>, x: Var) {
        let mut calibration = self.calibration.borrow_mut();
        let Some(calibration) = calibration.as_mut() else {
            return;
        };
        let backend = tape.backend();
        let x = tape.value(x);
        let shape = backend.shape(&x).to_vec();
        let width = *shape.last().expect("au moins une dimension");
        let rows = shape.iter().product::<usize>() / width;
        let flat = backend.reshape(&x, &[rows, width]);
        let product = backend.matmul(&backend.transpose(&flat, 0, 1), &flat);
        calibration.hessian = Some(match calibration.hessian.take() {
            Some(sum) => backend.add(&sum, &product),
            None => product,
        });
        calibration.rows += rows;
        if calibration.host {
            if calibration.largest.is_empty() {
                calibration.largest = vec![0.0; width];
                calibration.beyond_six = vec![0; width];
            }
            for row in backend.to_vec(&flat).chunks(width) {
                for (k, value) in row.iter().enumerate() {
                    let value = value.to_f64().abs();
                    calibration.largest[k] = calibration.largest[k].max(value);
                    calibration.beyond_six[k] += u64::from(value >= 6.0);
                }
            }
        }
    }

    /// Les poids quantifiés, si cette matrice l'a été.
    pub fn quantized(&self) -> Option<&QuantizedWeight<B>> {
        self.quantized.as_deref()
    }

    /// La feuille de ce paramètre sur `tape`. Elle est créée au premier
    /// appel, puis réutilisée : un paramètre employé à deux endroits d'un
    /// même calcul doit y être un seul et même nœud, pour que ses gradients
    /// s'additionnent.
    pub fn var(&self, tape: &Tape<B>) -> Var {
        if let Some((id, var)) = self.binding.get()
            && id == tape.id()
        {
            return var;
        }
        let var = if self.frozen {
            tape.constant(self.value.clone())
        } else {
            tape.leaf(self.value.clone())
        };
        self.binding.set(Some((tape.id(), var)));
        var
    }

    /// Gradient de ce paramètre après `tape.backward(…)`, ou `None` s'il
    /// n'a pas servi au calcul de la perte sur cette tape.
    pub fn grad<'g>(&self, tape_id: u64, grads: &'g Gradients<B>) -> Option<&'g B::Tensor> {
        let (id, var) = self.binding.get()?;
        if id != tape_id {
            return None;
        }
        grads.get(var)
    }
}

/// Tout ce qui contient des paramètres : une couche, un bloc, un modèle, et
/// le paramètre lui-même.
///
/// Un module ne décrit qu'une chose : comment parcourir ses paramètres, un
/// par un, chacun avec son nom. Tout le reste s'en déduit. L'ordre du
/// parcours est stable d'un appel à l'autre : l'optimiseur range ses propres
/// données dans cet ordre, et les noms sont ceux du fichier de poids.
///
/// Une couche n'écrit pas ce parcours à la main : la macro `module!` le
/// fabrique à partir de la liste de ses champs.
pub trait Module<B: Backend> {
    /// Présente à `visitor` chaque paramètre, précédé de son nom. `prefix`
    /// est le nom de ce module dans celui qui le contient, ou une chaîne
    /// vide tout en haut.
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>));

    /// Comme [`Module::walk`], pour modifier les paramètres.
    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    );

    fn parameters(&self) -> Vec<&Parameter<B>> {
        let mut parameters = Vec::new();
        self.walk("", &mut |_, parameter| parameters.push(parameter));
        parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<B>> {
        let mut parameters = Vec::new();
        self.walk_mut("", &mut |_, parameter| parameters.push(parameter));
        parameters
    }

    /// Un nom par paramètre, dans l'ordre de [`Module::parameters`] :
    /// `blocks.0.attention.query.weight`, par exemple. C'est sous ces noms
    /// que les poids sont enregistrés dans un fichier.
    fn parameter_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.walk("", &mut |name, _| names.push(name));
        names
    }

    /// Nombre total de nombres appris.
    fn parameter_count(&self, backend: &B) -> usize {
        self.parameters()
            .iter()
            .map(|p| backend.shape(p.value()).iter().product::<usize>())
            .sum()
    }
}

/// Nom d'un champ `name` dans un module nommé `prefix`.
pub fn child_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Un paramètre est le module le plus simple : il ne contient que lui-même.
impl<B: Backend> Module<B> for Parameter<B> {
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>)) {
        visitor(prefix.to_string(), self);
    }

    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    ) {
        visitor(prefix.to_string(), self);
    }
}

/// Une suite de modules : chacun porte son rang pour nom, `blocks.0`,
/// `blocks.1`…
impl<B: Backend, M: Module<B>> Module<B> for Vec<M> {
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>)) {
        for (index, module) in self.iter().enumerate() {
            module.walk(&child_name(prefix, &index.to_string()), visitor);
        }
    }

    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    ) {
        for (index, module) in self.iter_mut().enumerate() {
            module.walk_mut(&child_name(prefix, &index.to_string()), visitor);
        }
    }
}

/// Un module facultatif : absent, il n'a aucun paramètre.
impl<B: Backend, M: Module<B>> Module<B> for Option<M> {
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>)) {
        if let Some(module) = self {
            module.walk(prefix, visitor);
        }
    }

    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    ) {
        if let Some(module) = self {
            module.walk_mut(prefix, visitor);
        }
    }
}

/// Un module en boîte se parcourt comme le module lui-même.
impl<B: Backend, M: Module<B>> Module<B> for Box<M> {
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>)) {
        (**self).walk(prefix, visitor);
    }

    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    ) {
        (**self).walk_mut(prefix, visitor);
    }
}

/// Fait d'un type un [`Module`].
///
/// Pour une structure, on donne les champs à parcourir, dans l'ordre. Chacun
/// est lui-même un module (un paramètre, une couche, une suite de blocs), et
/// son nom de champ devient son nom dans le fichier de poids :
///
/// ```ignore
/// module!(Linear { weight, bias });
/// ```
///
/// Pour une énumération dont chaque variante enveloppe un module, on donne
/// les variantes. L'enveloppe est transparente : elle n'ajoute rien au nom.
///
/// ```ignore
/// module!(enum Norm { Layer, Rms });
/// ```
macro_rules! module {
    ($type:ident { $($field:ident),+ $(,)? }) => {
        impl<B: $crate::backend::Backend> $crate::nn::Module<B> for $type<B> {
            fn walk<'a>(
                &'a self,
                prefix: &str,
                visitor: &mut dyn FnMut(String, &'a $crate::nn::Parameter<B>),
            ) {
                $(
                    let name = $crate::nn::child_name(prefix, stringify!($field));
                    $crate::nn::Module::walk(&self.$field, &name, visitor);
                )+
            }

            fn walk_mut<'a>(
                &'a mut self,
                prefix: &str,
                visitor: &mut dyn FnMut(String, &'a mut $crate::nn::Parameter<B>),
            ) {
                $(
                    let name = $crate::nn::child_name(prefix, stringify!($field));
                    $crate::nn::Module::walk_mut(&mut self.$field, &name, visitor);
                )+
            }
        }
    };
    (enum $type:ident { $($variant:ident),+ $(,)? }) => {
        impl<B: $crate::backend::Backend> $crate::nn::Module<B> for $type<B> {
            fn walk<'a>(
                &'a self,
                prefix: &str,
                visitor: &mut dyn FnMut(String, &'a $crate::nn::Parameter<B>),
            ) {
                match self {
                    $(Self::$variant(inner) => $crate::nn::Module::walk(inner, prefix, visitor),)+
                }
            }

            fn walk_mut<'a>(
                &'a mut self,
                prefix: &str,
                visitor: &mut dyn FnMut(String, &'a mut $crate::nn::Parameter<B>),
            ) {
                match self {
                    $(Self::$variant(inner) => {
                        $crate::nn::Module::walk_mut(inner, prefix, visitor)
                    })+
                }
            }
        }
    };
}
pub(crate) use module;
