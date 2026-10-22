//! Différentiation automatique en mode inverse.
//!
//! Une [`Tape`] enregistre chaque opération au moment où elle est calculée.
//! [`Tape::backward`] relit ensuite cet enregistrement à l'envers pour
//! obtenir le gradient d'une valeur scalaire, la perte, par rapport à toutes
//! les feuilles.
//!
//! La tape ne manipule les tenseurs qu'à travers le trait [`Backend`] : le
//! même code calcule les gradients sur CPU et sur GPU.

use std::cell::{Ref, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::ops::{
    Affine, CachedAttention, CausalSoftmax, Concat, CrossEntropy, FusedAttention, Gelu, Normalize,
    Operation, QuantizedMatmul, RmsNormalize, RotatePairs, Run, Silu, TransposedMatmul,
};
use crate::backend::{Backend, Float};
use crate::cache::CacheStep;
use crate::nn::QuantizedWeight;

/// Poignée vers un nœud de la tape. Elle ne contient qu'un indice : la
/// copier ne coûte rien, et elle n'emprunte pas la tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

/// Ce qui a produit un nœud, avec ses parents. C'est tout ce dont la
/// rétropropagation a besoin : les valeurs sont dans les nœuds.
///
/// Les primitives du backend sont énumérées ici, chacune avec sa règle de
/// gradient dans [`Tape::backward`] : il y en a peu, et la liste ne bouge
/// pas. Tout le reste passe par la dernière variante, qui range une
/// [opération](crate::backend::ops) telle quelle. La tape n'a donc pas à
/// connaître les opérations : en ajouter une ne change rien ici.
enum Op<B: Backend> {
    /// Paramètre ou entrée dont on veut le gradient.
    Leaf,
    /// Valeur fixe : aucun gradient ne lui est propagé.
    Constant,
    Add(Var, Var),
    Sub(Var, Var),
    Mul(Var, Var),
    Div(Var, Var),
    Neg(Var),
    Exp(Var),
    Ln(Var),
    Sqrt(Var),
    Tanh(Var),
    Sum {
        input: Var,
        axis: usize,
        keepdim: bool,
    },
    SumAll(Var),
    Reshape(Var),
    Transpose(Var, usize, usize),
    Expand(Var),
    MatMul(Var, Var),
    Softmax(Var),
    LogSoftmax(Var),
    MaskedFill {
        input: Var,
        mask: Var,
    },
    GatherRows {
        input: Var,
        indices: Vec<u32>,
    },
    ScatterAddRows {
        input: Var,
        indices: Vec<u32>,
    },
    /// Une opération, avec ses entrées dans l'ordre où elle les attend.
    Operation {
        rule: Box<dyn Rule<B>>,
        inputs: Vec<Var>,
        /// Ce que l'opération a gardé pour sa rétropropagation.
        saved: Option<B::Tensor>,
    },
}

/// La règle de rétropropagation d'une opération, sous une forme que la tape
/// peut ranger sans connaître le type de l'opération.
trait Rule<B: Backend> {
    fn backward(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        saved: Option<&B::Tensor>,
        grad: &B::Tensor,
        needed: &[bool],
    ) -> Vec<Option<B::Tensor>>;
}

/// Toute opération que le backend sait exécuter a une règle : celle que le
/// backend lui-même fournit.
impl<B: Backend + Run<O>, O: Operation> Rule<B> for O {
    fn backward(
        &self,
        backend: &B,
        inputs: &[&B::Tensor],
        output: &B::Tensor,
        saved: Option<&B::Tensor>,
        grad: &B::Tensor,
        needed: &[bool],
    ) -> Vec<Option<B::Tensor>> {
        backend.run_backward_needed(self, inputs, output, saved, grad, needed)
    }
}

struct Node<B: Backend> {
    value: B::Tensor,
    op: Op<B>,
    /// Faux si le nœud ne dépend d'aucune feuille : la rétropropagation
    /// l'ignore, ainsi que tout ce qui est en amont.
    requires_grad: bool,
}

/// L'enregistrement d'un calcul.
///
/// Les nœuds sont ajoutés dans l'ordre où ils sont calculés : un nœud vient
/// toujours après ses parents. La liste est donc déjà triée, et la
/// rétropropagation n'a qu'à la parcourir de la fin vers le début.
///
/// Une tape sert à un seul calcul : on en crée une neuve à chaque pas
/// d'entraînement.
pub struct Tape<'b, B: Backend> {
    /// Numéro unique, pour reconnaître à quelle tape appartient une `Var`.
    id: u64,
    backend: &'b B,
    // `RefCell` pour que les opérations prennent `&self` : on peut alors
    // les imbriquer, `tape.add(tape.mul(a, b), c)`.
    nodes: RefCell<Vec<Node<B>>>,
}

/// Les gradients calculés par [`Tape::backward`], un par feuille atteinte.
pub struct Gradients<B: Backend> {
    grads: Vec<Option<B::Tensor>>,
}

impl<B: Backend> Gradients<B> {
    /// Gradient de la perte par rapport à `var`, ou `None` si `var` n'est
    /// pas une feuille ou n'a pas servi au calcul de la perte.
    pub fn get(&self, var: Var) -> Option<&B::Tensor> {
        self.grads.get(var.0)?.as_ref()
    }
}

/// Ramène un gradient à la forme de l'opérande qui avait été étiré par
/// broadcasting : étirer une valeur revient à l'utiliser plusieurs fois,
/// donc son gradient est la somme des gradients de toutes ses copies.
fn unbroadcast<B: Backend>(backend: &B, grad: B::Tensor, target: &[usize]) -> B::Tensor {
    let mut grad = grad;
    // Dimensions ajoutées à gauche.
    while backend.shape(&grad).len() > target.len() {
        grad = backend.sum(&grad, 0, false);
    }
    // Dimensions de taille 1 qui ont été étirées.
    for (dim, &size) in target.iter().enumerate() {
        if size == 1 && backend.shape(&grad)[dim] != 1 {
            grad = backend.sum(&grad, dim, true);
        }
    }
    grad
}

impl<'b, B: Backend> Tape<'b, B> {
    pub fn new(backend: &'b B) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            backend,
            nodes: RefCell::new(Vec::new()),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn backend(&self) -> &'b B {
        self.backend
    }

    fn push(&self, value: B::Tensor, op: Op<B>, requires_grad: bool) -> Var {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(Node {
            value,
            op,
            requires_grad,
        });
        Var(nodes.len() - 1)
    }

    /// Enregistre le résultat d'une opération à un parent.
    fn unary(&self, a: Var, op: Op<B>, f: impl FnOnce(&B, &B::Tensor) -> B::Tensor) -> Var {
        let (value, requires_grad) = {
            let nodes = self.nodes.borrow();
            (f(self.backend, &nodes[a.0].value), nodes[a.0].requires_grad)
        };
        self.push(value, op, requires_grad)
    }

    /// Enregistre le résultat d'une opération à deux parents.
    fn binary(
        &self,
        a: Var,
        b: Var,
        op: Op<B>,
        f: impl FnOnce(&B, &B::Tensor, &B::Tensor) -> B::Tensor,
    ) -> Var {
        let (value, requires_grad) = {
            let nodes = self.nodes.borrow();
            (
                f(self.backend, &nodes[a.0].value, &nodes[b.0].value),
                nodes[a.0].requires_grad || nodes[b.0].requires_grad,
            )
        };
        self.push(value, op, requires_grad)
    }

    /// Exécute une opération et l'enregistre. Les `inputs` sont ses entrées,
    /// dans l'ordre que documente sa structure.
    ///
    /// C'est le seul point d'entrée des opérations sur la tape : les
    /// méthodes comme [`Tape::gelu`] ne sont que des raccourcis vers lui.
    pub fn apply<O: Operation>(&self, op: O, inputs: &[Var]) -> Var
    where
        B: Run<O>,
    {
        let ((value, saved), requires_grad) = {
            let nodes = self.nodes.borrow();
            let values: Vec<&B::Tensor> = inputs.iter().map(|var| &nodes[var.0].value).collect();
            let requires_grad = inputs.iter().any(|var| nodes[var.0].requires_grad);
            (self.backend.run_saving(&op, &values), requires_grad)
        };
        let op = Op::Operation {
            rule: Box::new(op),
            inputs: inputs.to_vec(),
            saved,
        };
        self.push(value, op, requires_grad)
    }

    // --- Entrées ---

    /// Ajoute un tenseur dont on voudra le gradient : un paramètre du modèle.
    pub fn leaf(&self, value: B::Tensor) -> Var {
        self.push(value, Op::Leaf, true)
    }

    /// Ajoute un tenseur fixe : un masque, une constante.
    pub fn constant(&self, value: B::Tensor) -> Var {
        self.push(value, Op::Constant, false)
    }

    /// Constante à un seul élément, qui s'étire à toute forme.
    pub fn scalar(&self, value: B::Scalar) -> Var {
        self.constant(self.backend.scalar(value))
    }

    // --- Lecture ---

    pub fn value(&self, var: Var) -> Ref<'_, B::Tensor> {
        Ref::map(self.nodes.borrow(), |nodes| &nodes[var.0].value)
    }

    pub fn shape(&self, var: Var) -> Vec<usize> {
        self.backend.shape(&self.value(var)).to_vec()
    }

    // --- Élément par élément ---

    pub fn add(&self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Add(a, b), |backend, x, y| backend.add(x, y))
    }

    pub fn sub(&self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Sub(a, b), |backend, x, y| backend.sub(x, y))
    }

    pub fn mul(&self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Mul(a, b), |backend, x, y| backend.mul(x, y))
    }

    pub fn div(&self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::Div(a, b), |backend, x, y| backend.div(x, y))
    }

    pub fn neg(&self, a: Var) -> Var {
        self.unary(a, Op::Neg(a), |backend, x| backend.neg(x))
    }

    pub fn exp(&self, a: Var) -> Var {
        self.unary(a, Op::Exp(a), |backend, x| backend.exp(x))
    }

    pub fn ln(&self, a: Var) -> Var {
        self.unary(a, Op::Ln(a), |backend, x| backend.ln(x))
    }

    pub fn sqrt(&self, a: Var) -> Var {
        self.unary(a, Op::Sqrt(a), |backend, x| backend.sqrt(x))
    }

    pub fn tanh(&self, a: Var) -> Var {
        self.unary(a, Op::Tanh(a), |backend, x| backend.tanh(x))
    }

    /// GELU, la fonction d'activation de GPT.
    pub fn gelu(&self, a: Var) -> Var {
        self.apply(Gelu, &[a])
    }

    // --- Réductions ---

    pub fn sum(&self, a: Var, axis: usize, keepdim: bool) -> Var {
        let op = Op::Sum {
            input: a,
            axis,
            keepdim,
        };
        self.unary(a, op, |backend, x| backend.sum(x, axis, keepdim))
    }

    pub fn sum_all(&self, a: Var) -> Var {
        self.unary(a, Op::SumAll(a), |backend, x| backend.sum_all(x))
    }

    /// Moyenne le long d'un axe. Elle est composée de `sum` et `mul` : son
    /// gradient s'en déduit tout seul.
    pub fn mean(&self, a: Var, axis: usize, keepdim: bool) -> Var {
        let n = self.shape(a)[axis];
        let inverse = self.scalar(B::Scalar::from_f64(1.0 / n as f64));
        self.mul(self.sum(a, axis, keepdim), inverse)
    }

    // --- Changements de forme ---

    pub fn reshape(&self, a: Var, shape: &[usize]) -> Var {
        self.unary(a, Op::Reshape(a), |backend, x| backend.reshape(x, shape))
    }

    pub fn transpose(&self, a: Var, axis_a: usize, axis_b: usize) -> Var {
        self.unary(a, Op::Transpose(a, axis_a, axis_b), |backend, x| {
            backend.transpose(x, axis_a, axis_b)
        })
    }

    /// Étire `a` jusqu'à la forme `shape` en recopiant ses éléments, selon
    /// les règles du broadcasting.
    pub fn expand(&self, a: Var, shape: &[usize]) -> Var {
        self.unary(a, Op::Expand(a), |backend, x| backend.expand(x, shape))
    }

    // --- Algèbre linéaire ---

    pub fn matmul(&self, a: Var, b: Var) -> Var {
        self.binary(a, b, Op::MatMul(a, b), |backend, x, y| backend.matmul(x, y))
    }

    // --- Softmax ---

    pub fn softmax(&self, a: Var) -> Var {
        self.unary(a, Op::Softmax(a), |backend, x| backend.softmax(x))
    }

    pub fn log_softmax(&self, a: Var) -> Var {
        self.unary(a, Op::LogSoftmax(a), |backend, x| backend.log_softmax(x))
    }

    /// Normalise chaque vecteur du dernier axe : moyenne nulle, variance 1.
    pub fn normalize(&self, a: Var, epsilon: B::Scalar) -> Var {
        let epsilon = epsilon.to_f64();
        self.apply(Normalize { epsilon }, &[a])
    }

    /// SiLU, la fonction d'activation de SwiGLU.
    pub fn silu(&self, a: Var) -> Var {
        self.apply(Silu, &[a])
    }

    /// Divise chaque vecteur du dernier axe par la racine de la moyenne de
    /// ses carrés : la normalisation de RMSNorm.
    pub fn rms_normalize(&self, a: Var, epsilon: B::Scalar) -> Var {
        let epsilon = epsilon.to_f64();
        self.apply(RmsNormalize { epsilon }, &[a])
    }

    /// Tourne chaque paire de composantes de `a`, `[…, T, D]`, de l'angle
    /// dont `cos` et `sin`, `[T, D/2]`, donnent le cosinus et le sinus.
    /// Les angles sont des données : aucun gradient ne leur est propagé.
    pub fn rotate_pairs(&self, a: Var, cos: Var, sin: Var) -> Var {
        self.apply(RotatePairs, &[a, cos, sin])
    }

    /// Met `b`, `[…, B]`, à la suite de `a`, `[…, A]`, le long du dernier
    /// axe.
    pub fn concat(&self, a: Var, b: Var) -> Var {
        self.apply(Concat, &[a, b])
    }

    /// Softmax causal de scores d'attention `[…, N, T]`, multipliés par
    /// `scale` : chaque requête ne distribue son attention que sur sa propre
    /// position et celles qui la précèdent.
    pub fn causal_softmax(&self, a: Var, scale: B::Scalar) -> Var {
        let scale = scale.to_f64();
        self.apply(CausalSoftmax { scale }, &[a])
    }

    /// L'attention causale entière, `causal_softmax(q · kᵀ, scale) · v`,
    /// en une seule opération : voir [`FusedAttention`].
    pub fn attention(&self, q: Var, k: Var, v: Var, scale: B::Scalar) -> Var {
        let scale = scale.to_f64();
        self.apply(FusedAttention { scale }, &[q, k, v])
    }

    /// `x · W + b` en une opération (voir [`Affine`]).
    pub fn affine(&self, x: Var, weight: Var, bias: Var) -> Var {
        self.apply(Affine, &[x, weight, bias])
    }

    /// `x · W (+ b)` pour des poids quantifiés (voir [`QuantizedMatmul`]).
    /// `dequantized` est la valeur du paramètre, les poids déquantifiés.
    pub fn quantized_matmul(
        &self,
        x: Var,
        quantized: &QuantizedWeight<B>,
        dequantized: Var,
        bias: Option<Var>,
    ) -> Var {
        let words = self.constant(quantized.words.clone());
        let scales = self.constant(quantized.scales.clone());
        let mut inputs = vec![x, words, scales, dequantized];
        inputs.extend(bias);
        let op = QuantizedMatmul {
            format: quantized.format,
        };
        self.apply(op, &inputs)
    }

    /// `a · bᵀ` pour une matrice `b` `[N, K]`, sans recopier `b` transposée
    /// (voir [`TransposedMatmul`]).
    pub fn transposed_matmul(&self, a: Var, b: Var) -> Var {
        self.apply(TransposedMatmul, &[a, b])
    }

    /// L'attention de la génération, sur les tables du cache : voir
    /// [`CachedAttention`]. Elle n'a pas de gradient : sa sortie est
    /// enregistrée comme une constante, et ce qui la précède ne reçoit rien
    /// à travers elle.
    pub fn cached_attention(
        &self,
        q: Var,
        keys: Var,
        values: Var,
        step: &CacheStep,
        scale: B::Scalar,
        value_dim: Option<usize>,
    ) -> Var {
        let op = CachedAttention {
            scale: scale.to_f64(),
            value_dim,
        };
        let value = {
            let nodes = self.nodes.borrow();
            let inputs = [q, keys, values, step.slots, step.rows];
            let values: Vec<&B::Tensor> = inputs.iter().map(|var| &nodes[var.0].value).collect();
            self.backend.run(&op, &values)
        };
        self.constant(value)
    }

    // --- Sélection ---

    /// Remplace par `value` les éléments où `mask` est non nul. Le masque
    /// ne reçoit pas de gradient.
    pub fn masked_fill(&self, a: Var, mask: Var, value: B::Scalar) -> Var {
        let op = Op::MaskedFill { input: a, mask };
        let (result, requires_grad) = {
            let nodes = self.nodes.borrow();
            (
                self.backend
                    .masked_fill(&nodes[a.0].value, &nodes[mask.0].value, value),
                nodes[a.0].requires_grad,
            )
        };
        self.push(result, op, requires_grad)
    }

    pub fn gather_rows(&self, a: Var, indices: &[u32]) -> Var {
        let op = Op::GatherRows {
            input: a,
            indices: indices.to_vec(),
        };
        self.unary(a, op, |backend, x| backend.gather_rows(x, indices))
    }

    /// L'inverse de [`Tape::gather_rows`] : part d'un tenseur nul de
    /// `num_rows` lignes et ajoute la ligne `r` de `a` à sa ligne
    /// `indices[r]`.
    pub fn scatter_add_rows(&self, a: Var, indices: &[u32], num_rows: usize) -> Var {
        let op = Op::ScatterAddRows {
            input: a,
            indices: indices.to_vec(),
        };
        self.unary(a, op, |backend, x| {
            backend.scatter_add_rows(x, indices, num_rows)
        })
    }

    // --- Perte ---

    /// Entropie croisée moyenne entre des logits `[N, V]` et `N` cibles.
    /// Softmax et entropie croisée sont fusionnés en une seule opération.
    pub fn cross_entropy(&self, logits: Var, targets: &[u32]) -> Var {
        let targets = targets.to_vec();
        self.apply(CrossEntropy { targets }, &[logits])
    }

    // --- Rétropropagation ---

    /// Calcule le gradient de `loss`, qui doit n'avoir qu'un élément, par
    /// rapport à toutes les feuilles dont elle dépend.
    ///
    /// Chaque nœud reçoit `grad`, le gradient de la perte par rapport à sa
    /// propre valeur, et en déduit la contribution de chacun de ses parents
    /// par la règle de dérivation de son opération.
    pub fn backward(&self, loss: Var) -> Gradients<B> {
        let nodes = self.nodes.borrow();
        let backend = self.backend;
        let value = |var: Var| &nodes[var.0].value;
        let shape = |var: Var| backend.shape(&nodes[var.0].value);

        let loss_shape = shape(loss);
        assert_eq!(
            loss_shape.iter().product::<usize>(),
            1,
            "backward() part d'un scalaire, reçu la forme {loss_shape:?}",
        );

        let mut grads: Vec<Option<B::Tensor>> = (0..=loss.0).map(|_| None).collect();
        // La dérivée de la perte par rapport à elle-même vaut 1.
        grads[loss.0] = Some(backend.ones(loss_shape));

        // Un parent qui ne dépend d'aucune feuille ne reçoit rien : son
        // gradient n'est même pas calculé.
        let needs = |var: Var| nodes[var.0].requires_grad;
        // Ajoute une contribution au gradient d'un parent. Toujours `+=` :
        // un nœud utilisé plusieurs fois reçoit plusieurs contributions.
        let accumulate = |grads: &mut Vec<Option<B::Tensor>>, var: Var, grad: B::Tensor| {
            if !needs(var) {
                return;
            }
            grads[var.0] = Some(match grads[var.0].take() {
                Some(existing) => backend.add(&existing, &grad),
                None => grad,
            });
        };

        for index in (0..=loss.0).rev() {
            // Un nœud sans gradient n'a pas servi au calcul de la perte.
            let Some(grad) = grads[index].take() else {
                continue;
            };
            let node = &nodes[index];
            let out = &node.value;

            match &node.op {
                Op::Leaf => {
                    // Rien à propager : on garde le gradient, c'est le résultat.
                    grads[index] = Some(grad);
                }
                Op::Constant => {}

                // c = a + b  →  da = dc, db = dc
                Op::Add(a, b) => {
                    if needs(*a) {
                        let da = unbroadcast(backend, grad.clone(), shape(*a));
                        accumulate(&mut grads, *a, da);
                    }
                    if needs(*b) {
                        accumulate(&mut grads, *b, unbroadcast(backend, grad, shape(*b)));
                    }
                }
                // c = a - b  →  da = dc, db = -dc
                Op::Sub(a, b) => {
                    if needs(*a) {
                        let da = unbroadcast(backend, grad.clone(), shape(*a));
                        accumulate(&mut grads, *a, da);
                    }
                    if needs(*b) {
                        let db = unbroadcast(backend, backend.neg(&grad), shape(*b));
                        accumulate(&mut grads, *b, db);
                    }
                }
                // c = a · b  →  da = dc · b, db = dc · a
                Op::Mul(a, b) => {
                    if needs(*a) {
                        let da = backend.mul(&grad, value(*b));
                        accumulate(&mut grads, *a, unbroadcast(backend, da, shape(*a)));
                    }
                    if needs(*b) {
                        let db = backend.mul(&grad, value(*a));
                        accumulate(&mut grads, *b, unbroadcast(backend, db, shape(*b)));
                    }
                }
                // c = a / b  →  da = dc / b, db = -dc · c / b
                Op::Div(a, b) => {
                    if needs(*a) {
                        let da = backend.div(&grad, value(*b));
                        accumulate(&mut grads, *a, unbroadcast(backend, da, shape(*a)));
                    }
                    if needs(*b) {
                        let db = backend.neg(&backend.div(&backend.mul(&grad, out), value(*b)));
                        accumulate(&mut grads, *b, unbroadcast(backend, db, shape(*b)));
                    }
                }
                Op::Neg(a) => accumulate(&mut grads, *a, backend.neg(&grad)),
                // c = exp(a)  →  da = dc · c
                Op::Exp(a) => accumulate(&mut grads, *a, backend.mul(&grad, out)),
                // c = ln(a)  →  da = dc / a
                Op::Ln(a) => accumulate(&mut grads, *a, backend.div(&grad, value(*a))),
                // c = √a  →  da = dc / (2c)
                Op::Sqrt(a) => {
                    let two = backend.scalar(B::Scalar::from_f64(2.0));
                    let da = backend.div(&grad, &backend.mul(out, &two));
                    accumulate(&mut grads, *a, da);
                }
                // c = tanh(a)  →  da = dc · (1 - c²)
                Op::Tanh(a) => {
                    let one = backend.scalar(B::Scalar::ONE);
                    let slope = backend.sub(&one, &backend.mul(out, out));
                    accumulate(&mut grads, *a, backend.mul(&grad, &slope));
                }

                // Chaque élément de l'axe réduit a contribué une fois à la
                // somme : il reçoit le gradient de celle-ci, recopié.
                Op::Sum {
                    input,
                    axis,
                    keepdim,
                } => {
                    let input_shape = shape(*input);
                    let grad = if *keepdim {
                        grad
                    } else {
                        let mut kept = input_shape.to_vec();
                        kept[*axis] = 1;
                        backend.reshape(&grad, &kept)
                    };
                    accumulate(&mut grads, *input, backend.expand(&grad, input_shape));
                }
                Op::SumAll(a) => {
                    accumulate(&mut grads, *a, backend.expand(&grad, shape(*a)));
                }

                Op::Reshape(a) => {
                    accumulate(&mut grads, *a, backend.reshape(&grad, shape(*a)));
                }
                // Échanger deux axes est son propre inverse.
                Op::Transpose(a, axis_a, axis_b) => {
                    accumulate(&mut grads, *a, backend.transpose(&grad, *axis_a, *axis_b));
                }
                // Chaque élément a été recopié : son gradient est la somme
                // des gradients de ses copies.
                Op::Expand(a) => {
                    accumulate(&mut grads, *a, unbroadcast(backend, grad, shape(*a)));
                }

                // C = A · B  →  dA = dC · Bᵀ, dB = Aᵀ · dC
                // Une matrice partagée par tout un lot, comme les poids d'une
                // couche linéaire, reçoit la somme sur le lot.
                Op::MatMul(a, b) => {
                    if needs(*a) {
                        let rank_b = shape(*b).len();
                        let b_transposed = backend.transpose(value(*b), rank_b - 2, rank_b - 1);
                        let da = backend.matmul(&grad, &b_transposed);
                        accumulate(&mut grads, *a, unbroadcast(backend, da, shape(*a)));
                    }
                    if needs(*b) {
                        let rank_a = shape(*a).len();
                        let a_transposed = backend.transpose(value(*a), rank_a - 2, rank_a - 1);
                        let db = backend.matmul(&a_transposed, &grad);
                        accumulate(&mut grads, *b, unbroadcast(backend, db, shape(*b)));
                    }
                }

                // y = softmax(x)  →  dx = y · (dy - Σ dy · y)
                Op::Softmax(a) => {
                    let last = shape(*a).len() - 1;
                    let projection = backend.sum(&backend.mul(&grad, out), last, true);
                    let da = backend.mul(out, &backend.sub(&grad, &projection));
                    accumulate(&mut grads, *a, da);
                }
                // y = ln softmax(x)  →  dx = dy - softmax(x) · Σ dy
                Op::LogSoftmax(a) => {
                    let last = shape(*a).len() - 1;
                    let total = backend.sum(&grad, last, true);
                    let da = backend.sub(&grad, &backend.mul(&backend.exp(out), &total));
                    accumulate(&mut grads, *a, da);
                }

                // Les éléments remplacés ne dépendent plus de l'entrée.
                Op::MaskedFill { input, mask } => {
                    let da = backend.masked_fill(&grad, value(*mask), B::Scalar::ZERO);
                    accumulate(&mut grads, *input, da);
                }
                // Une ligne lue plusieurs fois reçoit la somme des gradients
                // de toutes ses lectures.
                Op::GatherRows { input, indices } => {
                    let rows = shape(*input)[0];
                    accumulate(
                        &mut grads,
                        *input,
                        backend.scatter_add_rows(&grad, indices, rows),
                    );
                }
                // Chaque ligne de l'entrée a été ajoutée à une ligne de la
                // sortie : elle en reçoit le gradient.
                Op::ScatterAddRows { input, indices } => {
                    accumulate(&mut grads, *input, backend.gather_rows(&grad, indices));
                }

                // Une opération connaît sa propre règle : elle rend le
                // gradient de chacune de ses entrées, ou rien pour celles
                // qui n'en reçoivent pas.
                Op::Operation {
                    rule,
                    inputs,
                    saved,
                } => {
                    let values: Vec<&B::Tensor> = inputs.iter().map(|var| value(*var)).collect();
                    let needed: Vec<bool> = inputs.iter().map(|var| needs(*var)).collect();
                    let input_grads =
                        rule.backward(backend, &values, out, saved.as_ref(), &grad, &needed);
                    assert_eq!(
                        input_grads.len(),
                        inputs.len(),
                        "une opération doit rendre un gradient par entrée",
                    );
                    for (var, input_grad) in inputs.iter().zip(input_grads) {
                        if let Some(input_grad) = input_grad {
                            accumulate(&mut grads, *var, input_grad);
                        }
                    }
                }
            }
        }

        Gradients { grads }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::{Cpu, CpuTensor};
    use crate::backend::ops::causal_mask;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Cpu<f64>;
    type T = CpuTensor<f64>;

    const EPSILON: f64 = 1e-5;
    const TOLERANCE: f64 = 1e-6;

    /// Réduit une sortie quelconque à un scalaire par une somme pondérée.
    /// Des poids tous différents font arriver un gradient différent sur
    /// chaque élément de la sortie : une règle qui confondrait deux éléments
    /// serait détectée.
    fn weighted_sum(tape: &Tape<B>, out: Var) -> Var {
        let weights = T::randn(&tape.shape(out), &mut StdRng::seed_from_u64(99));
        tape.sum_all(tape.mul(out, tape.constant(weights)))
    }

    /// Compare le gradient calculé par la tape à celui obtenu par
    /// différences finies centrées, pour chaque élément de chaque entrée :
    /// `(f(x + ε) - f(x - ε)) / 2ε`.
    fn gradcheck(shapes: &[&[usize]], build: impl Fn(&Tape<B>, &[Var]) -> Var) {
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(7);
        let inputs: Vec<T> = shapes.iter().map(|s| T::randn(s, &mut rng)).collect();

        let loss_of = |inputs: &[T]| -> f64 {
            let tape = Tape::new(&backend);
            let vars: Vec<Var> = inputs.iter().map(|t| tape.leaf(t.clone())).collect();
            let loss = weighted_sum(&tape, build(&tape, &vars));
            tape.value(loss).item()
        };

        let tape = Tape::new(&backend);
        let vars: Vec<Var> = inputs.iter().map(|t| tape.leaf(t.clone())).collect();
        let loss = weighted_sum(&tape, build(&tape, &vars));
        let grads = tape.backward(loss);

        for (input_index, var) in vars.iter().enumerate() {
            let analytic = grads
                .get(*var)
                .unwrap_or_else(|| panic!("pas de gradient pour l'entrée {input_index}"));
            assert_eq!(analytic.shape(), inputs[input_index].shape());

            for element in 0..inputs[input_index].data().len() {
                let perturbed = |delta: f64| {
                    let mut inputs = inputs.clone();
                    let mut data = inputs[input_index].data().to_vec();
                    data[element] += delta;
                    inputs[input_index] = T::from_vec(data, shapes[input_index]);
                    loss_of(&inputs)
                };
                let numeric = (perturbed(EPSILON) - perturbed(-EPSILON)) / (2.0 * EPSILON);
                let analytic = analytic.data()[element];
                let scale = analytic.abs().max(numeric.abs()).max(1.0);
                assert!(
                    (analytic - numeric).abs() <= TOLERANCE * scale,
                    "entrée {input_index}, élément {element} : \
                     analytique {analytic}, numérique {numeric}",
                );
            }
        }
    }

    /// `x² + 0,5`, strictement positif : pour `ln`, `sqrt` et les divisions.
    fn positive(tape: &Tape<B>, x: Var) -> Var {
        tape.add(tape.mul(x, x), tape.scalar(0.5))
    }

    // --- Élément par élément ---

    #[test]
    fn add() {
        gradcheck(&[&[2, 3], &[2, 3]], |t, v| t.add(v[0], v[1]));
    }

    #[test]
    fn sub() {
        gradcheck(&[&[2, 3], &[2, 3]], |t, v| t.sub(v[0], v[1]));
    }

    #[test]
    fn mul() {
        gradcheck(&[&[2, 3], &[2, 3]], |t, v| t.mul(v[0], v[1]));
    }

    #[test]
    fn div() {
        gradcheck(&[&[2, 3], &[2, 3]], |t, v| t.div(v[0], positive(t, v[1])));
    }

    #[test]
    fn neg() {
        gradcheck(&[&[2, 3]], |t, v| t.neg(v[0]));
    }

    #[test]
    fn exp() {
        gradcheck(&[&[2, 3]], |t, v| t.exp(v[0]));
    }

    #[test]
    fn ln() {
        gradcheck(&[&[2, 3]], |t, v| t.ln(positive(t, v[0])));
    }

    #[test]
    fn sqrt() {
        gradcheck(&[&[2, 3]], |t, v| t.sqrt(positive(t, v[0])));
    }

    #[test]
    fn tanh() {
        gradcheck(&[&[2, 3]], |t, v| t.tanh(v[0]));
    }

    // --- Broadcasting ---

    #[test]
    fn broadcast_row() {
        gradcheck(&[&[2, 3], &[3]], |t, v| t.add(v[0], v[1]));
    }

    #[test]
    fn broadcast_column() {
        gradcheck(&[&[2, 3], &[2, 1]], |t, v| t.mul(v[0], v[1]));
    }

    #[test]
    fn broadcast_both_sides() {
        gradcheck(&[&[2, 1], &[1, 3]], |t, v| t.sub(v[0], v[1]));
    }

    #[test]
    fn broadcast_extra_dimensions() {
        gradcheck(&[&[2, 3, 4], &[4]], |t, v| t.div(v[0], positive(t, v[1])));
    }

    #[test]
    fn broadcast_scalar_leaf() {
        gradcheck(&[&[2, 3], &[]], |t, v| t.mul(v[0], v[1]));
    }

    // --- Accumulation ---

    #[test]
    fn variable_used_twice() {
        // d(x · x) = 2x : les deux contributions doivent s'additionner.
        gradcheck(&[&[2, 3]], |t, v| t.mul(v[0], v[0]));
    }

    #[test]
    fn variable_used_on_several_paths() {
        gradcheck(&[&[2, 3]], |t, v| {
            let a = t.tanh(v[0]);
            let b = t.exp(v[0]);
            t.add(t.mul(a, b), v[0])
        });
    }

    // --- Réductions ---

    #[test]
    fn sum_first_axis() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.sum(v[0], 0, false));
    }

    #[test]
    fn sum_middle_axis() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.sum(v[0], 1, false));
    }

    #[test]
    fn sum_last_axis_keepdim() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.sum(v[0], 2, true));
    }

    #[test]
    fn sum_all() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.sum_all(t.mul(v[0], v[0])));
    }

    #[test]
    fn mean() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.mean(v[0], 1, false));
    }

    // --- Changements de forme ---

    #[test]
    fn reshape() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.reshape(v[0], &[6, 4]));
    }

    #[test]
    fn transpose() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.transpose(v[0], 0, 2));
    }

    #[test]
    fn transpose_then_matmul() {
        gradcheck(&[&[2, 3, 4]], |t, v| {
            t.matmul(v[0], t.transpose(v[0], 1, 2))
        });
    }

    // --- Matmul ---

    #[test]
    fn matmul_two_matrices() {
        gradcheck(&[&[3, 4], &[4, 2]], |t, v| t.matmul(v[0], v[1]));
    }

    #[test]
    fn matmul_batch_times_matrix() {
        // La matrice de poids est partagée par tout le lot.
        gradcheck(&[&[2, 3, 4], &[4, 2]], |t, v| t.matmul(v[0], v[1]));
    }

    #[test]
    fn matmul_batch_times_batch() {
        gradcheck(&[&[2, 2, 3, 4], &[2, 2, 4, 3]], |t, v| t.matmul(v[0], v[1]));
    }

    #[test]
    fn matmul_matrix_times_batch() {
        gradcheck(&[&[3, 4], &[2, 4, 2]], |t, v| t.matmul(v[0], v[1]));
    }

    // --- Softmax ---

    #[test]
    fn softmax() {
        gradcheck(&[&[3, 5]], |t, v| t.softmax(v[0]));
    }

    #[test]
    fn softmax_three_dimensions() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.softmax(v[0]));
    }

    #[test]
    fn log_softmax() {
        gradcheck(&[&[3, 5]], |t, v| t.log_softmax(v[0]));
    }

    // --- Opérations fusionnées ---

    #[test]
    fn gelu() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.gelu(v[0]));
    }

    #[test]
    fn gelu_matches_its_composition() {
        // La version fusionnée doit donner la même valeur que la formule
        // écrite avec les primitives.
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let x = tape.leaf(T::randn(&[3, 5], &mut StdRng::seed_from_u64(2)));
        let scalar = |value: f64| tape.scalar(value);
        let cubed = tape.mul(tape.mul(x, x), x);
        let inner = tape.mul(
            tape.add(x, tape.mul(cubed, scalar(0.044715))),
            scalar((2.0 / std::f64::consts::PI).sqrt()),
        );
        let composed = tape.mul(
            tape.mul(x, scalar(0.5)),
            tape.add(tape.tanh(inner), scalar(1.0)),
        );
        let fused = tape.gelu(x);
        for (a, b) in tape
            .value(fused)
            .data()
            .iter()
            .zip(tape.value(composed).data())
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn normalize() {
        gradcheck(&[&[2, 3, 5]], |t, v| t.normalize(v[0], 1e-5));
    }

    #[test]
    fn silu() {
        gradcheck(&[&[2, 3, 4]], |t, v| t.silu(v[0]));
    }

    #[test]
    fn rms_normalize() {
        gradcheck(&[&[2, 3, 5]], |t, v| t.rms_normalize(v[0], 1e-5));
    }

    #[test]
    fn rms_normalize_of_tiny_rows() {
        // Carrés minuscules : c'est epsilon qui borne la division.
        gradcheck(&[&[2, 4]], |t, v| {
            t.rms_normalize(t.mul(v[0], t.scalar(1e-3)), 1e-5)
        });
    }

    /// Des angles quelconques, tous différents, pour `[length, half]`.
    fn angle_tables(tape: &Tape<B>, length: usize, half: usize) -> (Var, Var) {
        let angles = T::randn(&[length, half], &mut StdRng::seed_from_u64(5));
        let table = |f: fn(f64) -> f64| {
            let data = angles.data().iter().map(|&a| f(a)).collect();
            tape.constant(T::from_vec(data, &[length, half]))
        };
        (table(f64::cos), table(f64::sin))
    }

    #[test]
    fn rotate_pairs() {
        gradcheck(&[&[2, 3, 4, 6]], |t, v| {
            let (cos, sin) = angle_tables(t, 4, 3);
            t.rotate_pairs(v[0], cos, sin)
        });
    }

    #[test]
    fn concat() {
        gradcheck(&[&[2, 3, 4], &[2, 3, 2]], |t, v| t.concat(v[0], v[1]));
    }

    #[test]
    fn expand() {
        // Une dimension de taille 1 étirée, et une dimension ajoutée devant.
        gradcheck(&[&[2, 1, 3]], |t, v| t.expand(v[0], &[2, 4, 3]));
        gradcheck(&[&[3]], |t, v| t.expand(v[0], &[2, 3]));
    }

    #[test]
    fn concatenated_scores() {
        // L'usage réel : une clé faite d'une partie par tête et d'une partie
        // commune à toutes les têtes, étirée puis mise à la suite.
        gradcheck(
            &[&[2, 3, 4, 4], &[2, 3, 4, 2], &[2, 3, 5, 4], &[2, 1, 5, 2]],
            |t, v| {
                let q = t.concat(v[0], v[1]);
                let k = t.concat(v[2], t.expand(v[3], &[2, 3, 5, 2]));
                t.matmul(q, t.transpose(k, 2, 3))
            },
        );
    }

    #[test]
    fn rotated_scores() {
        // L'usage réel : requêtes et clés tournées, puis leur produit.
        gradcheck(&[&[2, 4, 6], &[2, 4, 6]], |t, v| {
            let (cos, sin) = angle_tables(t, 4, 3);
            let q = t.rotate_pairs(v[0], cos, sin);
            let k = t.rotate_pairs(v[1], cos, sin);
            t.matmul(q, t.transpose(k, 1, 2))
        });
    }

    #[test]
    fn normalize_of_nearly_constant_rows() {
        // Variance minuscule : c'est epsilon qui borne la division.
        gradcheck(&[&[2, 4]], |t, v| {
            t.normalize(t.mul(v[0], t.scalar(1e-3)), 1e-5)
        });
    }

    #[test]
    fn causal_softmax() {
        gradcheck(&[&[2, 4, 4]], |t, v| t.causal_softmax(v[0], 0.5));
    }

    #[test]
    fn causal_softmax_of_the_last_queries() {
        // Trois requêtes face à cinq positions, puis une seule.
        gradcheck(&[&[2, 3, 5]], |t, v| t.causal_softmax(v[0], 0.5));
        gradcheck(&[&[2, 1, 4]], |t, v| t.causal_softmax(v[0], 0.5));
    }

    #[test]
    fn attention() {
        // Requêtes, clés et valeurs d'une matrice carrée : l'entraînement.
        gradcheck(&[&[2, 4, 3], &[2, 4, 3], &[2, 4, 2]], |t, v| {
            t.attention(v[0], v[1], v[2], 0.5)
        });
    }

    #[test]
    fn attention_of_the_last_queries() {
        // Deux requêtes face à cinq positions, puis une seule : la génération
        // avec un cache.
        gradcheck(&[&[2, 2, 3], &[2, 5, 3], &[2, 5, 4]], |t, v| {
            t.attention(v[0], v[1], v[2], 0.7)
        });
        gradcheck(&[&[1, 2, 1, 3], &[1, 2, 4, 3], &[1, 2, 4, 3]], |t, v| {
            t.attention(v[0], v[1], v[2], 1.0)
        });
    }

    #[test]
    fn affine() {
        gradcheck(&[&[2, 3, 4], &[4, 5], &[5]], |t, v| {
            t.affine(v[0], v[1], v[2])
        });
    }

    #[test]
    fn affine_has_the_gradients_of_a_product_then_a_sum() {
        // Au bit près : un entraînement ne change pas.
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(5);
        let inputs = [
            T::randn(&[2, 3, 4], &mut rng),
            T::randn(&[4, 5], &mut rng),
            T::randn(&[5], &mut rng),
        ];
        let gradients = |fused: bool| {
            let tape = Tape::new(&backend);
            let v: Vec<Var> = inputs.iter().map(|t| tape.leaf(t.clone())).collect();
            let out = if fused {
                tape.affine(v[0], v[1], v[2])
            } else {
                tape.add(tape.matmul(v[0], v[1]), v[2])
            };
            let grads = tape.backward(weighted_sum(&tape, out));
            v.iter()
                .map(|&var| grads.get(var).unwrap().data().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(gradients(true), gradients(false));
    }

    #[test]
    fn a_frozen_weight_changes_no_other_gradient() {
        // Des poids constants, comme ceux d'un modèle sous un adaptateur :
        // leur gradient n'est pas calculé, les autres sont les mêmes.
        let backend = Cpu::new();
        let mut rng = StdRng::seed_from_u64(6);
        let inputs = [
            T::randn(&[2, 3, 4], &mut rng),
            T::randn(&[4, 5], &mut rng),
            T::randn(&[5], &mut rng),
        ];
        let gradients = |frozen: bool| {
            let tape = Tape::new(&backend);
            let x = tape.leaf(inputs[0].clone());
            let weight = if frozen {
                tape.constant(inputs[1].clone())
            } else {
                tape.leaf(inputs[1].clone())
            };
            let bias = tape.leaf(inputs[2].clone());
            let out = tape.mul(tape.affine(x, weight, bias), tape.matmul(x, weight));
            let grads = tape.backward(weighted_sum(&tape, out));
            assert_eq!(grads.get(weight).is_none(), frozen);
            [x, bias].map(|var| grads.get(var).unwrap().data().to_vec())
        };
        assert_eq!(gradients(true), gradients(false));
    }

    #[test]
    fn quantized_matmul_passes_the_gradient_to_its_input_and_bias() {
        use crate::quantize::Format;
        let mut rng = StdRng::seed_from_u64(8);
        let dequantized = T::randn(&[4, 3], &mut rng);
        // Le backend de référence ne lit que les poids déquantifiés.
        let quantized = QuantizedWeight {
            format: Format::Int8,
            words: T::zeros(&[1, 3]),
            scales: T::zeros(&[1, 3]),
        };
        gradcheck(&[&[2, 5, 4], &[3]], |t, v| {
            let weight = t.constant(dequantized.clone());
            t.quantized_matmul(v[0], &quantized, weight, Some(v[1]))
        });
        // Des poids déquantifiés rangés comme un paramètre ne reçoivent
        // rien, et la rétropropagation ne panique pas.
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let x = tape.leaf(T::randn(&[2, 4], &mut rng));
        let weight = tape.leaf(dequantized.clone());
        let out = tape.quantized_matmul(x, &quantized, weight, None);
        let grads = tape.backward(tape.sum_all(out));
        assert!(grads.get(x).is_some());
        assert!(grads.get(weight).is_none());
    }

    #[test]
    fn cached_attention_lets_no_gradient_through() {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let mut rng = StdRng::seed_from_u64(9);
        let q = tape.leaf(T::randn(&[1, 2, 4], &mut rng));
        let table = tape.leaf(T::randn(&[3, 2, 4], &mut rng));
        let step = CacheStep {
            capacity: 3,
            lines: tape.constant(T::from_vec(vec![2.0], &[1])),
            slots: tape.constant(T::from_vec(vec![0.0, 1.0, 2.0], &[1, 3])),
            rows: tape.constant(T::from_vec(vec![0.0, 3.0], &[1, 2])),
            rope: None,
            rope_single: None,
        };
        let out = tape.cached_attention(q, table, table, &step, 0.5, None);
        let loss = tape.add(tape.sum_all(out), tape.sum_all(q));
        let grads = tape.backward(loss);
        // `q` ne reçoit que le gradient de la somme directe.
        assert_eq!(grads.get(q).unwrap().data(), &[1.0; 8]);
        assert!(grads.get(table).is_none());
    }

    #[test]
    fn transposed_matmul() {
        gradcheck(&[&[2, 3, 4], &[5, 4]], |t, v| {
            t.transposed_matmul(v[0], v[1])
        });
    }

    #[test]
    fn transposed_matmul_is_a_product_by_the_transpose() {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let mut rng = StdRng::seed_from_u64(3);
        let a = tape.constant(T::randn(&[2, 3, 4], &mut rng));
        let b = tape.constant(T::randn(&[5, 4], &mut rng));
        let fused = tape.transposed_matmul(a, b);
        let primitives = tape.matmul(a, tape.transpose(b, 0, 1));
        assert_eq!(tape.value(fused).data(), tape.value(primitives).data());
    }

    #[test]
    fn causal_softmax_four_dimensions() {
        gradcheck(&[&[2, 2, 3, 3]], |t, v| t.causal_softmax(v[0], 1.0));
    }

    #[test]
    fn causal_softmax_matches_mask_then_softmax() {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let scores = tape.leaf(T::randn(&[2, 3, 3], &mut StdRng::seed_from_u64(4)));
        let mask = tape.constant(causal_mask(&Cpu::new(), 3, 3));
        let scaled = tape.mul(scores, tape.scalar(0.5));
        let composed = tape.softmax(tape.masked_fill(scaled, mask, f64::NEG_INFINITY));
        let fused = tape.causal_softmax(scores, 0.5);
        for (a, b) in tape
            .value(fused)
            .data()
            .iter()
            .zip(tape.value(composed).data())
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    // --- Sélection ---

    #[test]
    fn masked_fill() {
        gradcheck(&[&[2, 3, 3]], |t, v| {
            let mask = T::from_vec(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[3, 3]);
            t.masked_fill(v[0], t.constant(mask), -3.0)
        });
    }

    #[test]
    fn causal_attention_weights() {
        // Masque à -∞ puis softmax : les positions masquées ont un poids et
        // un gradient nuls.
        gradcheck(&[&[2, 3, 3]], |t, v| {
            let mask = t.constant(causal_mask(&Cpu::new(), 3, 3));
            t.softmax(t.masked_fill(v[0], mask, f64::NEG_INFINITY))
        });
    }

    #[test]
    fn gather_rows() {
        gradcheck(&[&[5, 3]], |t, v| t.gather_rows(v[0], &[4, 0, 2]));
    }

    #[test]
    fn gather_rows_with_repeated_indices() {
        // La ligne 1 est lue trois fois : son gradient est une somme.
        gradcheck(&[&[5, 3]], |t, v| t.gather_rows(v[0], &[1, 4, 1, 1]));
    }

    #[test]
    fn scatter_add_rows() {
        // Les lignes 0 et 2 de l'entrée tombent sur la même ligne.
        gradcheck(&[&[4, 3]], |t, v| {
            t.scatter_add_rows(v[0], &[5, 1, 5, 0], 6)
        });
    }

    #[test]
    fn gather_then_scatter() {
        // L'aller-retour du mélange d'experts : un groupe de lignes est
        // extrait, transformé, puis remis à sa place.
        gradcheck(&[&[5, 3], &[3, 3]], |t, v| {
            let group = t.gather_rows(v[0], &[4, 0, 2]);
            t.scatter_add_rows(t.matmul(group, v[1]), &[4, 0, 2], 5)
        });
    }

    // --- Perte ---

    #[test]
    fn cross_entropy() {
        gradcheck(&[&[4, 6]], |t, v| t.cross_entropy(v[0], &[0, 5, 2, 2]));
    }

    #[test]
    fn cross_entropy_matches_composition() {
        // La version fusionnée doit donner la même perte et le même gradient
        // que log_softmax suivi de la sélection des cibles.
        let backend = Cpu::new();
        let logits = T::randn(&[4, 6], &mut StdRng::seed_from_u64(3));
        let targets = [0u32, 5, 2, 2];

        let tape = Tape::new(&backend);
        let x = tape.leaf(logits.clone());
        let fused = tape.cross_entropy(x, &targets);
        let fused_loss = tape.value(fused).item();
        let fused_grad = tape.backward(fused).get(x).unwrap().clone();

        let tape = Tape::new(&backend);
        let x = tape.leaf(logits);
        let mut onehot = vec![0.0; 4 * 6];
        for (row, &target) in targets.iter().enumerate() {
            onehot[row * 6 + target as usize] = -1.0 / 4.0;
        }
        let picked = tape.mul(
            tape.log_softmax(x),
            tape.constant(T::from_vec(onehot, &[4, 6])),
        );
        let composed = tape.sum_all(picked);
        let composed_grad = tape.backward(composed).get(x).unwrap().clone();

        assert!((tape.value(composed).item() - fused_loss).abs() < 1e-12);
        for (a, b) in fused_grad.data().iter().zip(composed_grad.data()) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    // --- Compositions ---

    #[test]
    fn mean_of_tanh_of_affine() {
        // mean(tanh(A·x + b)), l'exemple du plan.
        gradcheck(&[&[3, 4], &[4, 2], &[3, 1]], |t, v| {
            let affine = t.add(t.matmul(v[0], v[1]), v[2]);
            t.mean(t.tanh(affine), 1, false)
        });
    }

    #[test]
    fn layer_norm() {
        // (x - moyenne) / √(variance + ε), puis gain et biais.
        gradcheck(&[&[2, 3, 4], &[4], &[4]], |t, v| {
            let centered = t.sub(v[0], t.mean(v[0], 2, true));
            let variance = t.mean(t.mul(centered, centered), 2, true);
            let std = t.sqrt(t.add(variance, t.scalar(1e-5)));
            t.add(t.mul(t.div(centered, std), v[1]), v[2])
        });
    }

    #[test]
    fn tiny_language_model() {
        // Embedding → couche cachée → logits → entropie croisée : toute la
        // chaîne de l'étape 5, avec des tokens répétés.
        gradcheck(&[&[7, 3], &[3, 5], &[5], &[5, 7], &[7]], |t, v| {
            let tokens = [1u32, 6, 1, 0];
            let embedded = t.gather_rows(v[0], &tokens);
            let hidden = t.tanh(t.add(t.matmul(embedded, v[1]), v[2]));
            let logits = t.add(t.matmul(hidden, v[3]), v[4]);
            t.cross_entropy(logits, &[6, 1, 0, 3])
        });
    }

    // --- Comportement de la tape ---

    #[test]
    fn constants_receive_no_gradient() {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let x = tape.leaf(T::ones(&[2]));
        let c = tape.constant(T::ones(&[2]));
        let loss = tape.sum_all(tape.mul(x, c));
        let grads = tape.backward(loss);
        assert!(grads.get(x).is_some());
        assert!(grads.get(c).is_none());
        assert!(
            grads.get(loss).is_none(),
            "seules les feuilles sont gardées"
        );
    }

    #[test]
    fn unused_leaf_has_no_gradient() {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let x = tape.leaf(T::ones(&[2]));
        let unused = tape.leaf(T::ones(&[2]));
        let loss = tape.sum_all(x);
        assert!(tape.backward(loss).get(unused).is_none());
    }

    #[test]
    fn known_gradient() {
        // perte = Σ x²  →  gradient = 2x
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let x = tape.leaf(T::from_vec(vec![1.0, -2.0, 3.0], &[3]));
        let loss = tape.sum_all(tape.mul(x, x));
        assert_eq!(tape.value(loss).item(), 14.0);
        assert_eq!(
            tape.backward(loss).get(x).unwrap().data(),
            &[2.0, -4.0, 6.0]
        );
    }

    #[test]
    #[should_panic(expected = "backward() part d'un scalaire")]
    fn backward_requires_a_scalar() {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        let x = tape.leaf(T::ones(&[2]));
        tape.backward(x);
    }
}
