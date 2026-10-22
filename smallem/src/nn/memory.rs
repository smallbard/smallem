//! La mémoire de n-grammes d'Engram (Cheng et al., 2026), reprise par Qwen :
//! une très grande table de vecteurs, indexée par les deux ou trois derniers
//! tokens, que le modèle consulte sans presque rien calculer.
//!
//! À chaque position, pour chaque ordre `n` (bigrammes, trigrammes) et
//! chacune de ses têtes, le n-gramme qui finit là est haché vers une ligne
//! de la table de la tête (voir [`crate::ngram`]). Les lignes lues sont mises
//! bout à bout en un vecteur `e`, de `d_mém` composantes, puis :
//!
//! ```text
//! v = W_V e,  k = W_K e
//! α = σ(√±(RMSNorm(h) · RMSNorm(k) / √C))   la porte, par position
//! ṽ = α v
//! Y = SiLU(Conv1D(RMSNorm(ṽ))) + ṽ          convolution causale, facultative
//! h ← h + Y
//! ```
//!
//! où `h` est le flux du modèle à l'endroit du module et `√±` la racine
//! signée, `signe(g) √|g|`. La porte compare ce que dit la table à ce que
//! le modèle sait déjà du contexte : une ligne partagée par plusieurs
//! n-grammes, ou hors de propos ici, est fermée.
//!
//! La table vit de trois façons ([`MemoryStorage`]) :
//!
//! - **dense** : un paramètre ordinaire, dont le gradient couvre toute la
//!   table, et que met à jour l'optimiseur du modèle ;
//! - **creuse** : à chaque pas, seules les lignes lues sont copiées dans un
//!   petit tenseur, dont le gradient ne couvre qu'elles ; la mémoire les met
//!   à jour elle-même, par un AdamW ligne par ligne ;
//! - **en mémoire centrale** : la même chose, la table et ses moments restant
//!   hors de la carte.

use std::cell::{Cell, RefCell};
use std::thread;

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use super::architecture::named_choices;
use super::{Module, Parameter, Projection, RmsNorm, child_name, sigmoid};
use crate::autograd::{Gradients, Tape, Var};
use crate::backend::{AdamWStep, Backend, Float, standard_normal};
use crate::cache::{KvCache, SequenceId};
use crate::ngram::NgramHash;
use crate::optim::{AdamWSettings, OptimizerState};

/// La graine des multiplicateurs du hachage. Elle est fixe : un modèle
/// enregistré relit sa table avec le même hachage.
pub const HASH_SEED: u64 = 1234;

named_choices! {
    /// Où vit la table, et comment elle apprend.
    MemoryStorage {
        Dense => "dense",
        Sparse => "sparse",
        Host => "host",
    }
}

/// La forme d'une mémoire de n-grammes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryDims {
    /// Nombre de lignes visé pour chaque table : chacune a un nombre premier
    /// de lignes, le premier au-dessus, puis les suivants.
    pub rows: usize,
    /// Têtes de hachage par ordre.
    pub heads: usize,
    /// Ordre le plus grand : 3 pour les bigrammes et les trigrammes.
    pub max_order: usize,
    /// Composantes du vecteur lu, `d_mém`, réparties entre les tables.
    pub dim: usize,
    /// Le module s'ajoute au flux juste avant le bloc `layer` : 1 le place
    /// après le premier bloc.
    pub layer: usize,
    /// Avec la porte de contexte ; sans, `ṽ = v`.
    pub gate: bool,
    /// Avec la convolution causale.
    pub conv: bool,
    pub storage: MemoryStorage,
    /// Écart-type des lignes de la table à l'initialisation.
    pub table_std: f64,
    /// Facteur du taux d'apprentissage de la table : 5 chez Engram.
    pub lr_scale: f64,
}

impl MemoryDims {
    /// Les choix d'Engram, ramenés à un modèle de largeur `width` : huit
    /// têtes par ordre, bigrammes et trigrammes, `d_mém = C`, après le
    /// premier bloc, avec porte et convolution. Les lignes partent à
    /// l'écart-type des embeddings, 0,02 : aucun des deux articles ne le
    /// donne, et à 1 la table noie le flux au départ (étape G1).
    pub fn engram(width: usize, rows: usize) -> Self {
        Self {
            rows,
            heads: 8,
            max_order: 3,
            dim: width,
            layer: 1,
            gate: true,
            conv: true,
            storage: MemoryStorage::Sparse,
            table_std: 0.02,
            lr_scale: 5.0,
        }
    }

    /// Nombre de tables : une par tête et par ordre.
    pub fn tables(&self) -> usize {
        (self.max_order - 1) * self.heads
    }

    /// Composantes lues dans chaque table.
    pub fn head_dim(&self) -> usize {
        self.dim / self.tables()
    }

    pub fn validate(&self, layers: usize) -> Result<(), String> {
        if self.max_order < 2 || self.heads == 0 || self.rows < 2 {
            return Err(format!("mémoire : forme refusée, {self:?}"));
        }
        if self.dim == 0 || !self.dim.is_multiple_of(self.tables()) {
            return Err(format!(
                "mémoire : {} composantes ne se partagent pas entre {} tables",
                self.dim,
                self.tables()
            ));
        }
        if self.layer > layers {
            return Err(format!(
                "mémoire avant le bloc {}, le modèle n'en a que {layers}",
                self.layer
            ));
        }
        if !(self.table_std > 0.0 && self.lr_scale > 0.0) {
            return Err("mémoire : écart-type et facteur du taux positifs".into());
        }
        Ok(())
    }
}

/// La table, sous l'une des trois formes de [`MemoryStorage`].
pub enum MemoryTable<B: Backend> {
    /// Toutes les tables l'une sous l'autre, `[lignes, d]`.
    Dense(Parameter<B>),
    Sparse(SparseTables<B>),
    Host(HostTables),
}

/// Une table par tête, sur la carte, avec ses moments d'AdamW.
pub struct SparseTables<B: Backend> {
    /// `[lignes, d]` chacune. Ce sont des paramètres pour être enregistrés
    /// avec le modèle, mais aucune tape ne les voit : l'optimiseur du modèle
    /// ne reçoit jamais leur gradient.
    pub values: Vec<Parameter<B>>,
    moments: Vec<Option<(B::Tensor, B::Tensor)>>,
}

/// Une table par tête, en mémoire centrale, avec ses moments d'AdamW.
pub struct HostTables {
    pub values: Vec<Vec<f32>>,
    first: Vec<Vec<f32>>,
    second: Vec<Vec<f32>>,
    dim: usize,
}

/// Ce qu'une passe a lu dans une table creuse : de quoi retrouver, après la
/// rétropropagation, le gradient des lignes lues.
struct Lookup {
    tape: u64,
    /// La feuille de la tape qui porte les lignes lues, toutes tables
    /// réunies, `[lignes lues, d]`.
    rows: Var,
    /// Pour chaque table, ses lignes lues, triées et sans doublon.
    unique: Vec<Vec<u32>>,
}

pub struct Memory<B: Backend> {
    dims: MemoryDims,
    width: usize,
    hash: NgramHash,
    pub table: MemoryTable<B>,
    /// `d_mém → C`
    pub value: Projection<B>,
    /// La porte : `W_K`, `d_mém → C`, et les deux normalisations.
    pub key: Option<Projection<B>>,
    pub hidden_norm: Option<RmsNorm<B>>,
    pub key_norm: Option<RmsNorm<B>>,
    /// La convolution : sa normalisation, et ses poids `[4, C]`, nuls au
    /// départ.
    pub conv_norm: Option<RmsNorm<B>>,
    pub conv: Option<Parameter<B>>,
    lookup: RefCell<Option<Lookup>>,
    /// Le gradient des lignes lues depuis le dernier pas, avec ces lignes :
    /// un par lot quand un pas en cumule plusieurs.
    gradients: Vec<(Vec<Vec<u32>>, B::Tensor)>,
    /// Pas d'AdamW déjà faits sur la table.
    steps: u32,
    /// Faux pour « couper » la mémoire : le module ne s'ajoute plus au flux.
    enabled: Cell<bool>,
}

/// La convolution : noyau de 4, dilatation de 3 (Engram).
const KERNEL: usize = 4;
const DILATION: usize = 3;

impl<B: Backend> Memory<B> {
    pub fn new(backend: &B, dims: MemoryDims, width: usize, rng: &mut impl Rng) -> Self {
        let hash = NgramHash::new(dims.max_order, dims.heads, dims.rows, HASH_SEED);
        let d = dims.head_dim();
        let table = match dims.storage {
            MemoryStorage::Dense => {
                let mut table =
                    Parameter::normal(backend, &[hash.total_rows(), d], dims.table_std, rng)
                        .without_decay();
                table.set_lr_scale(dims.lr_scale);
                MemoryTable::Dense(table)
            }
            MemoryStorage::Sparse => MemoryTable::Sparse(SparseTables {
                values: (0..hash.tables())
                    .map(|t| {
                        // Une ligne de plus, la réserve de `compact`.
                        Parameter::normal(backend, &[hash.rows(t) + 1, d], dims.table_std, rng)
                            .without_decay()
                    })
                    .collect(),
                moments: (0..hash.tables()).map(|_| None).collect(),
            }),
            MemoryStorage::Host => {
                let seeds: Vec<u64> = (0..hash.tables()).map(|_| rng.random()).collect();
                MemoryTable::Host(HostTables::new(&hash, d, dims.table_std, &seeds))
            }
        };
        Self {
            dims,
            width,
            hash,
            table,
            value: Projection::new(backend, dims.dim, width, rng),
            key: dims
                .gate
                .then(|| Projection::new(backend, dims.dim, width, rng)),
            hidden_norm: dims.gate.then(|| RmsNorm::new(backend, width)),
            key_norm: dims.gate.then(|| RmsNorm::new(backend, width)),
            conv_norm: dims.conv.then(|| RmsNorm::new(backend, width)),
            conv: dims
                .conv
                .then(|| Parameter::zeros(backend, &[KERNEL, width])),
            lookup: RefCell::new(None),
            gradients: Vec::new(),
            steps: 0,
            enabled: Cell::new(true),
        }
    }

    /// Coupe la mémoire, ou la rétablit. Coupée, elle ne lit plus rien et
    /// le modèle fonctionne comme s'il ne l'avait pas : on mesure ainsi ce
    /// qu'il lui doit.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
    }

    pub fn dims(&self) -> MemoryDims {
        self.dims
    }

    pub fn hash(&self) -> &NgramHash {
        &self.hash
    }

    /// Déplace une table creuse en mémoire centrale : pour générer sur la
    /// carte sans y garder la table. Les moments d'AdamW ne sont pas
    /// recopiés ; ils repartent de zéro si la table apprend encore.
    pub fn move_to_host(&mut self, backend: &B) -> Result<(), String> {
        let MemoryTable::Sparse(sparse) = &self.table else {
            return Err("seule une table creuse se déplace en mémoire centrale".into());
        };
        let values = sparse
            .values
            .iter()
            .map(|table| {
                backend
                    .to_vec(table.value())
                    .into_iter()
                    .map(|x| x.to_f64() as f32)
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        let empty = || values.iter().map(|_| Vec::new()).collect();
        self.table = MemoryTable::Host(HostTables {
            first: empty(),
            second: empty(),
            values,
            dim: self.dims.head_dim(),
        });
        self.dims.storage = MemoryStorage::Host;
        Ok(())
    }

    /// Nombre de nombres de la table : les paramètres qui ne calculent pas.
    pub fn table_parameters(&self) -> usize {
        self.hash.total_rows() * self.dims.head_dim()
    }

    /// Ajoute la mémoire au flux `h`, `[B, T, C]`, de `B` séquences de
    /// `length` positions. `indices` donne, pour chaque position, une ligne
    /// par table ([`NgramHash::indices`]).
    pub fn forward(&self, tape: &Tape<B>, h: Var, indices: &[u32], length: usize) -> Var {
        self.add_to(tape, h, indices, |normalized, positions| {
            (0..KERNEL)
                .map(|tap| shift(tape, normalized, positions, length, tap * DILATION))
                .collect()
        })
    }

    /// La même chose en génération : `h` est `[1, R, C]`, les nouvelles
    /// positions des séquences `counts`, dans cet ordre. La convolution lit
    /// les positions passées de chaque séquence dans `cache`, qui garde les
    /// dernières.
    pub fn forward_cached(
        &self,
        tape: &Tape<B>,
        h: Var,
        indices: &[u32],
        cache: &mut KvCache<B>,
        counts: &[(SequenceId, usize)],
    ) -> Var {
        let backend = tape.backend();
        let keep = (KERNEL - 1) * DILATION;
        self.add_to(tape, h, indices, |normalized, positions| {
            // Pour chaque séquence, ses lignes passées puis ses nouvelles
            // lignes, l'une sous l'autre, toutes séquences réunies.
            let rows = tape.value(normalized).clone();
            let mut all: Option<B::Tensor> = None;
            let mut sources: Vec<Vec<Option<u32>>> =
                (0..KERNEL).map(|_| Vec::with_capacity(positions)).collect();
            let mut start = 0;
            let mut base = 0;
            for &(id, count) in counts {
                let new: Vec<u32> = (start..start + count).map(|r| r as u32).collect();
                let new = backend.gather_rows(&rows, &new);
                let combined = match cache.memory_rows(id) {
                    Some(past) => backend.concat_rows(past, &new),
                    None => new,
                };
                let total = backend.shape(&combined)[0];
                let past = total - count;
                for (tap, source) in sources.iter_mut().enumerate() {
                    let delay = tap * DILATION;
                    for i in 0..count {
                        // `None` : avant le début de la séquence.
                        let at = (past + i).checked_sub(delay);
                        source.push(at.map(|at| (base + at) as u32));
                    }
                }
                cache.set_memory_rows(backend, id, combined.clone(), keep);
                all = Some(match all {
                    Some(all) => backend.concat_rows(&all, &combined),
                    None => combined,
                });
                base += total;
                start += count;
            }
            let all = all.expect("au moins une séquence");
            // Rien ne remonte vers le passé en génération : les lignes
            // décalées sont des constantes.
            sources
                .into_iter()
                .map(|source| {
                    let lines: Vec<u32> = source.iter().map(|at| at.unwrap_or(0)).collect();
                    let mask: Vec<B::Scalar> = source
                        .iter()
                        .map(|at| B::Scalar::from_f64(if at.is_some() { 1.0 } else { 0.0 }))
                        .collect();
                    let mask = backend.from_vec(mask, &[positions, 1]);
                    tape.constant(backend.mul(&backend.gather_rows(&all, &lines), &mask))
                })
                .collect()
        })
    }

    /// Le calcul commun : lecture, porte, puis la convolution, dont
    /// `shifted` donne les entrées décalées de 0, 3, 6 et 9 positions.
    fn add_to(
        &self,
        tape: &Tape<B>,
        h: Var,
        indices: &[u32],
        shifted: impl FnOnce(Var, usize) -> Vec<Var>,
    ) -> Var {
        if !self.enabled.get() {
            return h;
        }
        let shape = tape.shape(h);
        let positions = indices.len() / self.hash.tables();
        let width = self.width;
        let e = tape.reshape(self.read(tape, indices), &[positions, self.dims.dim]);

        let v = self.value.forward(tape, e);
        let v = match (&self.key, &self.hidden_norm, &self.key_norm) {
            (Some(key), Some(hidden_norm), Some(key_norm)) => {
                let k = key_norm.forward(tape, key.forward(tape, e));
                let flat = hidden_norm.forward(tape, tape.reshape(h, &[positions, width]));
                let scale = tape.scalar(B::Scalar::from_f64(1.0 / (width as f64).sqrt()));
                let score = tape.mul(tape.sum(tape.mul(flat, k), 1, true), scale);
                tape.mul(v, sigmoid(tape, signed_sqrt(tape, score)))
            }
            _ => v,
        };
        let y = match (&self.conv_norm, &self.conv) {
            (Some(norm), Some(weights)) => {
                let normalized = norm.forward(tape, v);
                let mut sum = None;
                for (tap, input) in shifted(normalized, positions).into_iter().enumerate() {
                    let weight = tape.gather_rows(weights.var(tape), &[tap as u32]);
                    let term = tape.mul(input, weight);
                    sum = Some(match sum {
                        Some(sum) => tape.add(sum, term),
                        None => term,
                    });
                }
                tape.add(tape.silu(sum.expect("noyau non vide")), v)
            }
            _ => v,
        };
        tape.add(h, tape.reshape(y, &shape))
    }

    /// Les lignes lues, `[positions × tables, d]`, position par position.
    fn read(&self, tape: &Tape<B>, indices: &[u32]) -> Var {
        let tables = self.hash.tables();
        let backend = tape.backend();
        match &self.table {
            MemoryTable::Dense(table) => {
                let mut offsets = Vec::with_capacity(tables);
                let mut offset = 0u32;
                for t in 0..tables {
                    offsets.push(offset);
                    offset += self.hash.rows(t) as u32;
                }
                let global: Vec<u32> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &index)| offsets[i % tables] + index)
                    .collect();
                tape.gather_rows(table.var(tape), &global)
            }
            MemoryTable::Sparse(_) | MemoryTable::Host(_) => {
                let spares: Vec<u32> = (0..tables).map(|t| self.hash.rows(t) as u32).collect();
                let (unique, local) = compact(indices, tables, &spares);
                let rows = match &self.table {
                    MemoryTable::Sparse(sparse) => {
                        let mut rows: Option<B::Tensor> = None;
                        for (t, lines) in unique.iter().enumerate() {
                            let part = backend.gather_rows(sparse.values[t].value(), lines);
                            rows = Some(match rows {
                                Some(rows) => backend.concat_rows(&rows, &part),
                                None => part,
                            });
                        }
                        rows.expect("au moins une table")
                    }
                    MemoryTable::Host(host) => host.gather(backend, &unique),
                    MemoryTable::Dense(_) => unreachable!(),
                };
                let rows = tape.leaf(rows);
                *self.lookup.borrow_mut() = Some(Lookup {
                    tape: tape.id(),
                    rows,
                    unique,
                });
                tape.gather_rows(rows, &local)
            }
        }
    }

    /// Garde le gradient des lignes lues par la passe de la tape `tape`,
    /// pour [`Memory::apply`]. Sans effet pour une table dense : son
    /// gradient va à l'optimiseur du modèle. Les gradients de plusieurs lots
    /// s'additionnent jusqu'au pas suivant.
    pub fn collect(&mut self, tape: u64, grads: &Gradients<B>) {
        let lookup = self.lookup.get_mut().take();
        let Some(lookup) = lookup.filter(|lookup| lookup.tape == tape) else {
            return;
        };
        if let Some(grad) = grads.get(lookup.rows) {
            self.gradients.push((lookup.unique, grad.clone()));
        }
    }

    /// Le pas d'AdamW des lignes lues depuis le dernier pas, au taux
    /// `learning_rate × lr_scale`, sans décroissance des poids. Le gradient
    /// est la somme de ceux des lots lus, multipliée par `grad_scale`. Les
    /// moments des lignes non lues ne bougent pas, et la correction de biais
    /// suit le nombre de pas de la table, comme le `SparseAdam` de PyTorch.
    pub fn apply(&mut self, backend: &B, learning_rate: f64, grad_scale: f64) {
        let gradients = std::mem::take(&mut self.gradients);
        let (unique, grad) = match gradients.len() {
            0 => return,
            1 => gradients.into_iter().next().expect("un gradient"),
            _ => {
                let spares: Vec<u32> = (0..self.hash.tables())
                    .map(|t| self.hash.rows(t) as u32)
                    .collect();
                merge_gradients(backend, gradients, &spares)
            }
        };
        self.steps += 1;
        let step = AdamWSettings::NANOGPT.step(
            self.steps,
            learning_rate * self.dims.lr_scale,
            0.0,
            grad_scale,
        );
        match &mut self.table {
            MemoryTable::Dense(_) => unreachable!("une table dense n'a pas de gradient à part"),
            MemoryTable::Sparse(sparse) => {
                let mut offset = 0u32;
                for (t, lines) in unique.iter().enumerate() {
                    let count = lines.len() as u32;
                    let range: Vec<u32> = (offset..offset + count).collect();
                    offset += count;
                    let grad = backend.gather_rows(&grad, &range);
                    let table = sparse.values[t].value_mut();
                    let moments = sparse.moments[t].get_or_insert_with(|| {
                        let shape = backend.shape(table).to_vec();
                        (backend.zeros(&shape), backend.zeros(&shape))
                    });
                    let mut rows = backend.gather_rows(table, lines);
                    let mut first = backend.gather_rows(&moments.0, lines);
                    let mut second = backend.gather_rows(&moments.1, lines);
                    backend.adamw_step(&mut rows, &grad, &mut first, &mut second, &step);
                    let at = backend.from_vec(
                        lines
                            .iter()
                            .map(|&l| B::Scalar::from_f64(f64::from(l)))
                            .collect(),
                        &[lines.len()],
                    );
                    backend.write_rows(table, &at, &rows);
                    backend.write_rows(&mut moments.0, &at, &first);
                    backend.write_rows(&mut moments.1, &at, &second);
                }
            }
            MemoryTable::Host(host) => host.adamw(backend, &unique, &grad, &step),
        }
    }

    /// L'état de l'AdamW d'une table creuse, rangé comme celui d'un
    /// optimiseur ([`OptimizerState`]) : un tenseur par paramètre de la
    /// mémoire, dans l'ordre de [`Module::walk`], `None` hors des tables.
    /// Vide pour une table dense, dont l'optimiseur du modèle tient l'état,
    /// et pour une table en mémoire centrale, qui ne s'enregistre pas.
    pub fn optimizer_state(&self) -> OptimizerState<&B::Tensor> {
        let MemoryTable::Sparse(sparse) = &self.table else {
            return OptimizerState::default();
        };
        let others = self.parameters().len() - sparse.values.len();
        let slot = |pick: fn(&Moments<B>) -> &B::Tensor| -> Vec<Option<&B::Tensor>> {
            sparse
                .moments
                .iter()
                .map(|moments| moments.as_ref().map(pick))
                .chain(std::iter::repeat_n(None, others))
                .collect()
        };
        OptimizerState {
            counters: [(MEMORY_STEPS.to_string(), self.steps.to_string())].into(),
            slots: vec![
                (MEMORY_FIRST_MOMENT.to_string(), slot(|m| &m.0)),
                (MEMORY_SECOND_MOMENT.to_string(), slot(|m| &m.1)),
            ],
        }
    }

    /// Les préfixes des séries de [`Memory::optimizer_state`].
    pub fn optimizer_slots(&self) -> Vec<&'static str> {
        match self.table {
            MemoryTable::Sparse(_) => vec![MEMORY_FIRST_MOMENT, MEMORY_SECOND_MOMENT],
            _ => Vec::new(),
        }
    }

    /// Remet l'AdamW d'une table creuse dans un état enregistré. Un
    /// compteur absent, celui d'un point de reprise écrit avant que la
    /// mémoire ne s'y range, fait repartir la table de zéro, comme les
    /// moments absents.
    pub fn restore_optimizer_state(
        &mut self,
        mut state: OptimizerState<B::Tensor>,
    ) -> Result<(), String> {
        let count = self.parameters().len();
        let MemoryTable::Sparse(sparse) = &mut self.table else {
            return Ok(());
        };
        self.steps = match state.counters.get(MEMORY_STEPS) {
            Some(steps) => steps
                .parse()
                .map_err(|_| format!("état de la mémoire : `{MEMORY_STEPS}` illisible"))?,
            None => 0,
        };
        let first = state.take_slot(MEMORY_FIRST_MOMENT, "de la mémoire")?;
        let second = state.take_slot(MEMORY_SECOND_MOMENT, "de la mémoire")?;
        if first.len() != count || second.len() != count {
            return Err("état de la mémoire : un moment par paramètre est attendu".into());
        }
        sparse.moments = first
            .into_iter()
            .zip(second)
            .take(sparse.values.len())
            .map(|pair| match pair {
                (Some(first), Some(second)) => Ok(Some((first, second))),
                (None, None) => Ok(None),
                _ => Err("état de la mémoire : un moment sans l'autre".to_string()),
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }
}

/// Les moments d'AdamW d'une table creuse.
type Moments<B> = (
    <B as crate::backend::Primitives>::Tensor,
    <B as crate::backend::Primitives>::Tensor,
);

/// Nom du compteur de pas de la table creuse dans un point de reprise, et
/// préfixes de ses deux séries de moments.
pub const MEMORY_STEPS: &str = "memory.adamw.steps";
const MEMORY_FIRST_MOMENT: &str = "memory.adamw.m.";
const MEMORY_SECOND_MOMENT: &str = "memory.adamw.v.";

/// Réunit les gradients de plusieurs lots : pour chaque table, l'union des
/// lignes lues, triée et complétée par la ligne de réserve jusqu'au total
/// des lignes des lots (une taille qui ne dépend pas des tokens, comme dans
/// [`compact`]), et la somme des gradients sur ces lignes.
fn merge_gradients<B: Backend>(
    backend: &B,
    gradients: Vec<(Vec<Vec<u32>>, B::Tensor)>,
    spares: &[u32],
) -> (Vec<Vec<u32>>, B::Tensor) {
    let union: Vec<Vec<u32>> = (0..spares.len())
        .map(|t| {
            let total: usize = gradients.iter().map(|(unique, _)| unique[t].len()).sum();
            let mut lines: Vec<u32> = gradients
                .iter()
                .flat_map(|(unique, _)| unique[t].iter().copied())
                .collect();
            lines.sort_unstable();
            lines.dedup();
            lines.resize(total, spares[t]);
            lines
        })
        .collect();
    let mut starts = Vec::with_capacity(union.len());
    let mut rows = 0;
    for lines in &union {
        starts.push(rows);
        rows += lines.len();
    }
    let mut sum: Option<B::Tensor> = None;
    for (unique, grad) in &gradients {
        let at: Vec<u32> = unique
            .iter()
            .enumerate()
            .flat_map(|(t, lines)| {
                let (union, start) = (&union[t], starts[t]);
                lines.iter().map(move |line| {
                    let place = union.binary_search(line).expect("ligne réunie");
                    (start + place) as u32
                })
            })
            .collect();
        let spread = backend.scatter_add_rows(grad, &at, rows);
        sum = Some(match sum {
            Some(sum) => backend.add(&sum, &spread),
            None => spread,
        });
    }
    (union, sum.expect("au moins un gradient"))
}

impl<B: Backend> Module<B> for Memory<B> {
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>)) {
        let name = |field: &str| child_name(prefix, field);
        self.table.walk(&name("table"), visitor);
        self.value.walk(&name("value"), visitor);
        self.key.walk(&name("key"), visitor);
        self.hidden_norm.walk(&name("hidden_norm"), visitor);
        self.key_norm.walk(&name("key_norm"), visitor);
        self.conv_norm.walk(&name("conv_norm"), visitor);
        self.conv.walk(&name("conv"), visitor);
    }

    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    ) {
        let name = |field: &str| child_name(prefix, field);
        self.table.walk_mut(&name("table"), visitor);
        self.value.walk_mut(&name("value"), visitor);
        self.key.walk_mut(&name("key"), visitor);
        self.hidden_norm.walk_mut(&name("hidden_norm"), visitor);
        self.key_norm.walk_mut(&name("key_norm"), visitor);
        self.conv_norm.walk_mut(&name("conv_norm"), visitor);
        self.conv.walk_mut(&name("conv"), visitor);
    }

    fn collect_gradients(&mut self, tape: u64, grads: &Gradients<B>) {
        self.collect(tape, grads);
    }
}

impl<B: Backend> Module<B> for MemoryTable<B> {
    fn walk<'a>(&'a self, prefix: &str, visitor: &mut dyn FnMut(String, &'a Parameter<B>)) {
        match self {
            Self::Dense(table) => table.walk(prefix, visitor),
            Self::Sparse(sparse) => sparse.values.walk(prefix, visitor),
            // Hors de la carte, la table s'enregistre à part.
            Self::Host(_) => {}
        }
    }

    fn walk_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &'a mut Parameter<B>),
    ) {
        match self {
            Self::Dense(table) => table.walk_mut(prefix, visitor),
            Self::Sparse(sparse) => sparse.values.walk_mut(prefix, visitor),
            Self::Host(_) => {}
        }
    }
}

impl HostTables {
    fn new(hash: &NgramHash, dim: usize, std: f64, seeds: &[u64]) -> Self {
        let values = thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .iter()
                .enumerate()
                .map(|(t, &seed)| {
                    let len = (hash.rows(t) + 1) * dim;
                    scope.spawn(move || {
                        let mut rng = StdRng::seed_from_u64(seed);
                        (0..len)
                            .map(|_| (std * standard_normal(&mut rng)) as f32)
                            .collect::<Vec<f32>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let zeros = |t: usize| vec![0.0f32; (hash.rows(t) + 1) * dim];
        Self {
            first: (0..hash.tables()).map(zeros).collect(),
            second: (0..hash.tables()).map(zeros).collect(),
            values,
            dim,
        }
    }

    /// Les lignes `unique[t]` de chaque table `t`, l'une sous l'autre, en un
    /// tenseur du backend.
    fn gather<B: Backend>(&self, backend: &B, unique: &[Vec<u32>]) -> B::Tensor {
        let d = self.dim;
        let count: usize = unique.iter().map(Vec::len).sum();
        let mut data = Vec::with_capacity(count * d);
        for (table, lines) in self.values.iter().zip(unique) {
            for &line in lines {
                let start = line as usize * d;
                data.extend(
                    table[start..start + d]
                        .iter()
                        .map(|&x| B::Scalar::from_f64(f64::from(x))),
                );
            }
        }
        backend.from_vec(data, &[count, d])
    }

    /// Le pas d'AdamW des lignes lues, une table par thread.
    fn adamw<B: Backend>(
        &mut self,
        backend: &B,
        unique: &[Vec<u32>],
        grad: &B::Tensor,
        step: &AdamWStep,
    ) {
        let d = self.dim;
        let grad: Vec<f32> = backend
            .to_vec(grad)
            .into_iter()
            .map(|g| g.to_f64() as f32)
            .collect();
        // Des moments absents, après `Memory::move_to_host`, partent de zéro.
        for ((values, first), second) in self
            .values
            .iter()
            .zip(&mut self.first)
            .zip(&mut self.second)
        {
            first.resize(values.len(), 0.0);
            second.resize(values.len(), 0.0);
        }
        let mut starts = Vec::with_capacity(unique.len());
        let mut offset = 0;
        for lines in unique {
            starts.push(offset);
            offset += lines.len() * d;
        }
        thread::scope(|scope| {
            for ((((values, first), second), lines), start) in self
                .values
                .iter_mut()
                .zip(&mut self.first)
                .zip(&mut self.second)
                .zip(unique)
                .zip(&starts)
            {
                let grad = &grad[*start..*start + lines.len() * d];
                scope.spawn(move || {
                    adamw_rows(values, first, second, lines, grad, d, step);
                });
            }
        });
    }
}

/// AdamW sur les lignes `lines` d'une table, sans décroissance des poids.
fn adamw_rows(
    values: &mut [f32],
    first: &mut [f32],
    second: &mut [f32],
    lines: &[u32],
    grad: &[f32],
    d: usize,
    step: &AdamWStep,
) {
    let (b1, b2) = (step.beta1 as f32, step.beta2 as f32);
    let rate = (step.learning_rate / step.bias_correction1) as f32;
    let c2 = step.bias_correction2 as f32;
    let (epsilon, scale) = (step.epsilon as f32, step.grad_scale as f32);
    for (row, &line) in lines.iter().enumerate() {
        let start = line as usize * d;
        for i in 0..d {
            let g = grad[row * d + i] * scale;
            let m = &mut first[start + i];
            let v = &mut second[start + i];
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            values[start + i] -= rate * *m / ((*v / c2).sqrt() + epsilon);
        }
    }
}

/// Pour des indices `[positions, tables]`, les lignes distinctes de chaque
/// table, triées, et la place de chaque indice dans ces lignes mises bout à
/// bout, table après table.
///
/// Chaque liste est complétée jusqu'à `positions` lignes par la ligne de
/// réserve de sa table, `spares[t]`, que rien ne lit : les tenseurs d'un pas
/// ont ainsi toujours la même taille. Sur la carte, la réserve de tampons
/// garde un tampon par taille rencontrée ; des tailles qui changeraient à
/// chaque pas la feraient grossir sans fin.
fn compact(indices: &[u32], tables: usize, spares: &[u32]) -> (Vec<Vec<u32>>, Vec<u32>) {
    let positions = indices.len() / tables;
    let unique: Vec<Vec<u32>> = (0..tables)
        .map(|t| {
            let mut lines: Vec<u32> = indices.iter().skip(t).step_by(tables).copied().collect();
            lines.sort_unstable();
            lines.dedup();
            lines.resize(positions, spares[t]);
            lines
        })
        .collect();
    let local = indices
        .iter()
        .enumerate()
        .map(|(i, index)| {
            let t = i % tables;
            let at = unique[t].binary_search(index).expect("ligne lue");
            (t * positions + at) as u32
        })
        .collect();
    (unique, local)
}

/// `x` décalé de `delay` positions vers le futur dans chaque séquence de
/// `length` positions : la position `t` reçoit la ligne `t − delay`, ou
/// zéro si elle tombe avant le début de la séquence.
fn shift<B: Backend>(tape: &Tape<B>, x: Var, positions: usize, length: usize, delay: usize) -> Var {
    if delay == 0 {
        return x;
    }
    let source: Vec<u32> = (0..positions)
        .map(|p| if p % length >= delay { p - delay } else { p } as u32)
        .collect();
    let mask: Vec<B::Scalar> = (0..positions)
        .map(|p| B::Scalar::from_f64(if p % length >= delay { 1.0 } else { 0.0 }))
        .collect();
    let mask = tape.constant(tape.backend().from_vec(mask, &[positions, 1]));
    tape.mul(tape.gather_rows(x, &source), mask)
}

/// `signe(x) √|x|`, écrit `x / (x² + ε)^¼` pour rester dérivable en zéro :
/// Engram borne de même `|x|` à 10⁻⁶ sous la racine.
fn signed_sqrt<B: Backend>(tape: &Tape<B>, x: Var) -> Var {
    let epsilon = tape.scalar(B::Scalar::from_f64(1e-12));
    let fourth = tape.sqrt(tape.sqrt(tape.add(tape.mul(x, x), epsilon)));
    tape.div(x, fourth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Primitives;
    use crate::backend::cpu::Cpu;

    fn dims(storage: MemoryStorage) -> MemoryDims {
        MemoryDims {
            rows: 50,
            heads: 2,
            max_order: 3,
            dim: 8,
            layer: 1,
            gate: true,
            conv: true,
            storage,
            table_std: 1.0,
            lr_scale: 5.0,
        }
    }

    #[test]
    fn compact_finds_each_line() {
        // Deux tables, trois positions.
        let indices = [4, 9, 2, 9, 4, 1];
        let (unique, local) = compact(&indices, 2, &[50, 60]);
        assert_eq!(unique, [vec![2, 4, 50], vec![1, 9, 60]]);
        assert_eq!(local, [1, 4, 0, 4, 1, 3]);
    }

    #[test]
    fn the_three_storages_read_and_learn_the_same() {
        let backend = Cpu::<f64>::new();
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
        let (batch, length, width) = (2, 6, 4);
        let mut losses = Vec::new();
        for storage in [
            MemoryStorage::Dense,
            MemoryStorage::Sparse,
            MemoryStorage::Host,
        ] {
            let mut rng = StdRng::seed_from_u64(5);
            let mut memory = Memory::new(&backend, dims(storage), width, &mut rng);
            // Les mêmes nombres dans les trois tables : celles de la table
            // dense, recopiées.
            let reference = {
                let mut rng = StdRng::seed_from_u64(5);
                Memory::new(&backend, dims(MemoryStorage::Dense), width, &mut rng)
            };
            let MemoryTable::Dense(dense) = &reference.table else {
                unreachable!()
            };
            let all = backend.to_vec(dense.value());
            let d = memory.dims.head_dim();
            let mut offset = 0;
            for t in 0..memory.hash.tables() {
                let len = memory.hash.rows(t) * d;
                // La table creuse a une ligne de réserve en plus.
                let mut part = all[offset..offset + len].to_vec();
                part.extend(std::iter::repeat_n(0.0, d));
                offset += len;
                let len = len + d;
                match &mut memory.table {
                    MemoryTable::Dense(_) => {}
                    MemoryTable::Sparse(sparse) => {
                        sparse.values[t].set_value(backend.from_vec(part.to_vec(), &[len / d, d]))
                    }
                    MemoryTable::Host(host) => {
                        host.values[t] = part.iter().map(|&x| x as f32).collect()
                    }
                }
            }
            // Les mêmes projections : une table en mémoire centrale ne tire
            // pas ses nombres du même générateur.
            memory
                .value
                .weight
                .set_value(reference.value.weight.value().clone());
            if let (Some(key), Some(same)) = (&mut memory.key, &reference.key) {
                key.weight.set_value(same.weight.value().clone());
            }
            // Une convolution non nulle, pour qu'elle compte.
            if let Some(conv) = &mut memory.conv {
                conv.set_value(backend.full(&[KERNEL, width], 0.3));
            }
            let indices = memory.hash.indices(&tokens, batch);
            let h0 = backend.from_vec(
                (0..batch * length * width)
                    .map(|i| (i as f64 * 0.37).sin())
                    .collect(),
                &[batch, length, width],
            );
            let tape = Tape::new(&backend);
            let h = tape.leaf(h0.clone());
            let out = memory.forward(&tape, h, &indices, length);
            let loss = tape.sum_all(tape.mul(out, out));
            let grads = tape.backward(loss);
            let value = backend.item(&tape.value(loss)).to_f64();
            // La table dense reçoit un gradient de toute la table, nul hors
            // des lignes lues : sa norme est celle des lignes lues. Les
            // tables creuses font en plus leur pas, et la perte d'après se
            // compare entre elles.
            let (norm, after) = match storage {
                MemoryStorage::Dense => {
                    let MemoryTable::Dense(table) = &memory.table else {
                        unreachable!()
                    };
                    let grad = table.grad(tape.id(), &grads).expect("gradient");
                    (
                        backend
                            .item(&backend.sum_all(&backend.mul(grad, grad)))
                            .to_f64(),
                        0.0,
                    )
                }
                _ => {
                    let id = tape.id();
                    drop(tape);
                    memory.collect(id, &grads);
                    let (_, grad) = memory.gradients.last().expect("gradient");
                    let norm = backend
                        .item(&backend.sum_all(&backend.mul(grad, grad)))
                        .to_f64();
                    memory.apply(&backend, 0.01, 1.0);
                    let tape = Tape::new(&backend);
                    let h = tape.leaf(h0);
                    let out = memory.forward(&tape, h, &indices, length);
                    let loss = tape.sum_all(tape.mul(out, out));
                    (norm, backend.item(&tape.value(loss)).to_f64())
                }
            };
            losses.push((value, norm, after));
        }
        // La table en mémoire centrale est en `f32`.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-6 * b.abs();
        for (value, norm, _) in &losses[1..] {
            assert!(close(*value, losses[0].0), "{losses:?}");
            assert!(close(*norm, losses[0].1), "{losses:?}");
        }
        assert!(close(losses[1].2, losses[2].2), "{losses:?}");
        assert!(
            losses[1].2 < losses[1].0,
            "le pas fait baisser la perte : {losses:?}"
        );
    }

    /// Les gradients de plusieurs lots se réunissent ligne par ligne, en
    /// un tenseur dont la taille ne dépend pas des tokens.
    #[test]
    fn gradients_of_several_batches_add_up_line_by_line() {
        type T = <Cpu<f64> as Primitives>::Tensor;
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(4);
        let mut memory = Memory::new(&backend, dims(MemoryStorage::Sparse), 4, &mut rng);
        for tokens in [[7u32, 8, 9], [8, 9, 3]] {
            let indices = memory.hash.indices(&tokens, 1);
            let tape = Tape::new(&backend);
            let h = tape.leaf(backend.ones(&[1, 3, 4]));
            let out = memory.forward(&tape, h, &indices, 3);
            let grads = tape.backward(tape.sum_all(tape.mul(out, out)));
            let id = tape.id();
            drop(tape);
            memory.collect(id, &grads);
        }
        let d = memory.dims.head_dim();
        let add = |sums: &mut std::collections::BTreeMap<(usize, u32), Vec<f64>>,
                   unique: &[Vec<u32>],
                   grad: &T| {
            let data = backend.to_vec(grad);
            let mut row = 0;
            for (t, lines) in unique.iter().enumerate() {
                for &line in lines {
                    let sum = sums.entry((t, line)).or_insert_with(|| vec![0.0; d]);
                    for (s, g) in sum.iter_mut().zip(&data[row * d..(row + 1) * d]) {
                        *s += g;
                    }
                    row += 1;
                }
            }
        };
        let mut expected = Default::default();
        for (unique, grad) in &memory.gradients {
            add(&mut expected, unique, grad);
        }
        let tables = memory.hash.tables();
        let spares: Vec<u32> = (0..tables).map(|t| memory.hash.rows(t) as u32).collect();
        let (union, sum) =
            merge_gradients(&backend, std::mem::take(&mut memory.gradients), &spares);
        assert_eq!(backend.shape(&sum), [2 * 3 * tables, d]);
        for (lines, spare) in union.iter().zip(&spares) {
            let read: Vec<&u32> = lines.iter().filter(|line| *line != spare).collect();
            assert!(read.windows(2).all(|w| w[0] < w[1]), "{lines:?}");
        }
        let mut merged = Default::default();
        add(&mut merged, &union, &sum);
        // Les lignes de réserve, ajoutées pour compléter, n'ont pas de
        // gradient.
        for ((t, line), value) in &merged {
            if !expected.contains_key(&(*t, *line)) {
                assert_eq!(*line, spares[*t]);
                assert!(value.iter().all(|&g| g == 0.0), "{value:?}");
            }
        }
        for (key, value) in &expected {
            for (a, b) in value.iter().zip(&merged[key]) {
                assert!((a - b).abs() < 1e-12, "{key:?} : {a} et {b}");
            }
        }
    }

    #[test]
    fn the_sparse_step_moves_only_the_lines_read() {
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(1);
        let mut memory = Memory::new(&backend, dims(MemoryStorage::Sparse), 4, &mut rng);
        let tokens = [7u32, 8, 9];
        let indices = memory.hash.indices(&tokens, 1);
        let before: Vec<Vec<f64>> = match &memory.table {
            MemoryTable::Sparse(s) => s.values.iter().map(|p| backend.to_vec(p.value())).collect(),
            _ => unreachable!(),
        };
        let tape = Tape::new(&backend);
        let h = tape.leaf(backend.ones(&[1, 3, 4]));
        let out = memory.forward(&tape, h, &indices, 3);
        let grads = tape.backward(tape.sum_all(tape.mul(out, out)));
        let id = tape.id();
        drop(tape);
        memory.collect(id, &grads);
        memory.apply(&backend, 0.1, 1.0);
        let MemoryTable::Sparse(sparse) = &memory.table else {
            unreachable!()
        };
        let d = memory.dims.head_dim();
        let tables = memory.hash.tables();
        for (t, (before, value)) in before.iter().zip(&sparse.values).enumerate() {
            let after = backend.to_vec(value.value());
            let read: Vec<usize> = indices
                .iter()
                .skip(t)
                .step_by(tables)
                .map(|&i| i as usize)
                .collect();
            for line in 0..memory.hash.rows(t) {
                let moved = before[line * d..(line + 1) * d] != after[line * d..(line + 1) * d];
                assert_eq!(moved, read.contains(&line), "table {t}, ligne {line}");
            }
        }
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(3);
        let mut memory = Memory::new(&backend, dims(MemoryStorage::Dense), 4, &mut rng);
        if let Some(conv) = &mut memory.conv {
            conv.set_value(backend.randn(&[KERNEL, 4], &mut rng));
        }
        let tokens = [2u32, 7, 1, 8, 2, 8, 1, 8];
        let indices = memory.hash.indices(&tokens, 2);
        let start: Vec<f64> = (0..32).map(|i| (i as f64 * 0.71).cos()).collect();
        let loss = |values: &[f64]| -> (f64, Vec<f64>) {
            let tape = Tape::new(&backend);
            let h = tape.leaf(backend.from_vec(values.to_vec(), &[2, 4, 4]));
            let out = memory.forward(&tape, h, &indices, 4);
            let weights = tape.constant(
                backend.from_vec((0..32).map(|i| 1.0 + i as f64 / 10.0).collect(), &[2, 4, 4]),
            );
            let loss = tape.sum_all(tape.mul(tape.mul(out, out), weights));
            let grads = tape.backward(loss);
            (
                backend.item(&tape.value(loss)),
                backend.to_vec(grads.get(h).expect("gradient")),
            )
        };
        let (_, analytic) = loss(&start);
        for i in [0, 5, 13, 22, 31] {
            let step = 1e-6;
            let mut plus = start.clone();
            plus[i] += step;
            let mut minus = start.clone();
            minus[i] -= step;
            let numeric = (loss(&plus).0 - loss(&minus).0) / (2.0 * step);
            assert!(
                (numeric - analytic[i]).abs() < 1e-5 * (1.0 + numeric.abs()),
                "composante {i} : {numeric} contre {}",
                analytic[i]
            );
        }
    }

    #[test]
    fn the_convolution_is_causal() {
        let backend = Cpu::<f64>::new();
        let mut rng = StdRng::seed_from_u64(2);
        let mut memory = Memory::new(&backend, dims(MemoryStorage::Dense), 4, &mut rng);
        if let Some(conv) = &mut memory.conv {
            conv.set_value(backend.full(&[KERNEL, 4], 0.5));
        }
        let run = |tokens: &[u32]| -> Vec<f64> {
            let indices = memory.hash.indices(tokens, 1);
            let tape = Tape::new(&backend);
            let h = tape.leaf(backend.ones(&[1, tokens.len(), 4]));
            let out = memory.forward(&tape, h, &indices, tokens.len());
            backend.to_vec(&tape.value(out))
        };
        let a = run(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let b = run(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 99]);
        // Changer le dernier token ne change que la dernière position.
        assert_eq!(a[..36], b[..36]);
        assert_ne!(a[36..], b[36..]);
    }
}
