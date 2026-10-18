//! Le mélange d'experts (MoE, *Mixture of Experts*) de DeepSeek : un
//! feed-forward fait de plusieurs petits réseaux, dont chaque token n'en
//! traverse que quelques-uns.

use std::cell::{Cell, RefCell};

use rand::Rng;

use super::{Balance, ExpertDims, Parameter, Projection, SwiGluFeedForward, module};
use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};

/// Un feed-forward fait de plusieurs **experts**, chacun un petit SwiGLU.
///
/// Un feed-forward ordinaire applique le même réseau à tous les tokens. Ici,
/// un **aiguilleur** choisit pour chaque token les quelques experts qui vont
/// le traiter. Le modèle peut avoir beaucoup de paramètres, mais chaque
/// token n'en traverse qu'une fraction : le coût du calcul suit le nombre
/// d'experts actifs, pas leur nombre total.
///
/// ```text
/// s_i    = sigmoïde(x · e_i)                 affinité du token pour l'expert i
/// on garde les K plus grandes affinités
/// g_i    = s_i / (somme des s gardées)       0 pour les experts écartés
/// sortie = Σ partagés(x) + Σ g_i × expert_i(x)
/// ```
///
/// Les experts **partagés** sont traversés par tous les tokens, sans
/// aiguillage : ils portent ce qui sert partout, et laissent les autres se
/// spécialiser.
///
/// Le choix des `K` experts n'est pas dérivable : un expert est gardé ou
/// non. Le gradient ne passe donc pas par le choix, mais par les portes
/// `g_i` des experts gardés, qui dépendent des affinités. L'aiguilleur
/// apprend ainsi à noter les experts, et le classement suit.
pub struct MixtureOfExperts<B: Backend> {
    dims: ExpertDims,
    /// Vrai pour calculer chaque expert sur tous les tokens et annuler par
    /// une porte nulle ce qui ne le concerne pas. C'est la définition, lente
    /// et simple ; faux, chaque expert ne reçoit que ses tokens.
    pub dense: bool,
    pub shared: Vec<SwiGluFeedForward<B>>,
    pub experts: Vec<SwiGluFeedForward<B>>,
    /// Une colonne `e_i` par expert aiguillé : `[C, experts]`.
    pub router: Projection<B>,
    /// Un biais par expert aiguillé, ajouté à son affinité pour le seul
    /// choix des experts. Il n'est pas appris par gradient : c'est
    /// [`MixtureOfExperts::rebalance`] qui le règle. Il est rangé parmi les
    /// paramètres pour être enregistré avec eux.
    pub routing_bias: Parameter<B>,
    load: RefCell<Load>,
    /// Le terme d'équilibrage de la dernière passe, avec le numéro de sa
    /// tape.
    penalty: Cell<Option<(u64, Var)>>,
}

/// Nombre de tokens reçus par chaque expert aiguillé, compté deux fois :
/// pour le journal, et pour le prochain réglage des biais.
struct Load {
    since_report: Vec<u64>,
    since_balance: Vec<u64>,
}

impl<B: Backend> MixtureOfExperts<B> {
    /// `proj_std` est l'écart-type d'initialisation de la projection de
    /// sortie de chaque expert (voir [`Block::new`](super::Block::new)).
    pub fn new(
        backend: &B,
        dim: usize,
        dims: ExpertDims,
        proj_std: f64,
        rng: &mut impl Rng,
    ) -> Self {
        if let Err(error) = dims.validate() {
            panic!("{error}");
        }
        let mut expert =
            || SwiGluFeedForward::with_hidden(backend, dim, dims.hidden, proj_std, rng);
        let shared = (0..dims.shared).map(|_| expert()).collect();
        let experts = (0..dims.routed).map(|_| expert()).collect();
        Self {
            dims,
            dense: false,
            shared,
            experts,
            router: Projection::new(backend, dim, dims.routed, rng),
            routing_bias: Parameter::zeros(backend, &[dims.routed]),
            load: RefCell::new(Load {
                since_report: vec![0; dims.routed],
                since_balance: vec![0; dims.routed],
            }),
            penalty: Cell::new(None),
        }
    }

    pub fn dims(&self) -> ExpertDims {
        self.dims
    }

    /// `[…, C]` donne la même forme. Chaque vecteur est un token, traité
    /// indépendamment des autres.
    pub fn forward(&self, tape: &Tape<B>, x: Var) -> Var {
        let backend = tape.backend();
        let shape = tape.shape(x);
        let dim = shape[shape.len() - 1];
        let count: usize = shape[..shape.len() - 1].iter().product();
        let routed = self.dims.routed;
        let flat = tape.reshape(x, &[count, dim]);

        // L'affinité de chaque token pour chaque expert : `[N, E]`.
        let affinity = sigmoid(tape, self.router.forward(tape, flat));

        // Le choix se fait sur l'hôte : `N × E` nombres à rapatrier, puis un
        // tri de `E` valeurs par token.
        let to_f64 = |tensor: &B::Tensor| -> Vec<f64> {
            backend.to_vec(tensor).iter().map(|v| v.to_f64()).collect()
        };
        let scores = to_f64(&tape.value(affinity));
        let bias = to_f64(self.routing_bias.value());
        let routing = route(&scores, &bias, self.dims.active);
        self.record(&routing);

        // Les portes : l'affinité des experts gardés, ramenée à une somme
        // de 1 par token ; zéro pour les autres.
        let mask = routing
            .mask
            .iter()
            .map(|&kept| {
                if kept {
                    B::Scalar::ONE
                } else {
                    B::Scalar::ZERO
                }
            })
            .collect();
        let mask = tape.constant(backend.from_vec(mask, &[count, routed]));
        let kept = tape.mul(affinity, mask);
        let gates = tape.div(kept, tape.sum(kept, 1, true));
        // Une porte par ligne, `[N × E, 1]` : celle du token `n` pour
        // l'expert `i` est à la ligne `n × E + i`.
        let gates = tape.reshape(gates, &[count * routed, 1]);

        if let Balance::Loss { weight } = self.dims.balance {
            let penalty = self.balance_penalty(tape, affinity, &routing, weight);
            self.penalty.set(Some((tape.id(), penalty)));
        }

        let mut output: Option<Var> = None;
        let mut add = |term: Var| {
            output = Some(match output {
                Some(sum) => tape.add(sum, term),
                None => term,
            });
        };
        // Les experts reçoivent leurs tokens par paquets de `chunk`, comme
        // un lot de séquences, plutôt qu'en une seule longue matrice. Le
        // résultat est le même. Le gradient des poids, lui, devient une
        // somme de petits produits au lieu d'un seul produit sur des
        // milliers de lignes, que le kernel du GPU traite six fois moins
        // vite.
        let chunk = Routing::chunk(count);
        let in_chunks = |tokens: Var| {
            let rows = tape.shape(tokens)[0];
            if rows.is_multiple_of(chunk) {
                tape.reshape(tokens, &[rows / chunk, chunk, dim])
            } else {
                tokens
            }
        };
        let apply = |expert: &SwiGluFeedForward<B>, tokens: Var| {
            let rows = tape.shape(tokens)[0];
            tape.reshape(expert.forward(tape, in_chunks(tokens)), &[rows, dim])
        };
        for expert in &self.shared {
            add(apply(expert, flat));
        }
        let everyone: Vec<u32> = (0..count as u32).collect();
        for (index, expert) in self.experts.iter().enumerate() {
            // Un expert que personne n'a choisi n'a rien à calculer.
            if !self.dense && routing.groups[index].is_empty() {
                continue;
            }
            let padded;
            let tokens = if self.dense {
                &everyone
            } else {
                padded = routing.padded_group(index);
                &padded
            };
            let gate_rows: Vec<u32> = tokens
                .iter()
                .map(|&token| token * routed as u32 + index as u32)
                .collect();
            let gate = tape.gather_rows(gates, &gate_rows);
            if self.dense {
                add(tape.mul(apply(expert, flat), gate));
            } else {
                // Le groupe de l'expert est extrait, transformé, puis
                // chaque résultat est remis à la place de son token.
                let group = tape.gather_rows(flat, tokens);
                let transformed = tape.mul(apply(expert, group), gate);
                add(tape.scatter_add_rows(transformed, tokens, count));
            }
        }
        let output = output.expect("chaque token traverse au moins un expert");
        tape.reshape(output, &shape)
    }

    /// Le terme classique d'équilibrage :
    ///
    /// ```text
    /// f_i = part des choix reçue par l'expert i, × E     (1 si tout est équilibré)
    /// P_i = moyenne sur les tokens de s_i / Σ_j s_j
    /// pénalité = poids × Σ f_i × P_i
    /// ```
    ///
    /// `f_i` est un compte : il ne porte pas de gradient. Celui-ci passe par
    /// `P_i`, et pousse à baisser les affinités des experts les plus
    /// chargés.
    fn balance_penalty(
        &self,
        tape: &Tape<B>,
        affinity: Var,
        routing: &Routing,
        weight: f64,
    ) -> Var {
        let backend = tape.backend();
        let routed = self.dims.routed;
        let count = tape.shape(affinity)[0];
        let choices = (count * self.dims.active) as f64;
        let fractions = routing
            .groups
            .iter()
            .map(|group| B::Scalar::from_f64(group.len() as f64 * routed as f64 / choices))
            .collect();
        let fractions = tape.constant(backend.from_vec(fractions, &[routed]));
        let normalized = tape.div(affinity, tape.sum(affinity, 1, true));
        let mean = tape.mean(normalized, 0, false);
        let total = tape.sum_all(tape.mul(mean, fractions));
        tape.mul(total, tape.scalar(B::Scalar::from_f64(weight)))
    }

    /// Le terme d'équilibrage calculé par la dernière passe sur `tape`, à
    /// ajouter à la perte. `None` si la couche n'équilibre pas par la perte.
    pub fn penalty(&self, tape: &Tape<B>) -> Option<Var> {
        self.penalty
            .get()
            .filter(|(id, _)| *id == tape.id())
            .map(|(_, penalty)| penalty)
    }

    fn record(&self, routing: &Routing) {
        let mut load = self.load.borrow_mut();
        for (index, group) in routing.groups.iter().enumerate() {
            load.since_report[index] += group.len() as u64;
            load.since_balance[index] += group.len() as u64;
        }
    }

    /// Nombre de tokens reçus par chaque expert aiguillé depuis le dernier
    /// appel.
    pub fn take_load(&self) -> Vec<u64> {
        let mut load = self.load.borrow_mut();
        let routed = load.since_report.len();
        std::mem::replace(&mut load.since_report, vec![0; routed])
    }

    /// Oublie les comptes en cours : ce qui vient de passer n'était pas de
    /// l'entraînement.
    pub fn reset_load(&self) {
        let mut load = self.load.borrow_mut();
        load.since_report.fill(0);
        load.since_balance.fill(0);
    }

    /// Règle les biais d'aiguillage d'après les tokens reçus depuis le
    /// dernier appel : celui d'un expert plus chargé que la moyenne baisse
    /// d'un cran, celui d'un expert moins chargé monte d'un cran. Sans
    /// effet si la couche n'équilibre pas par les biais.
    pub fn rebalance(&mut self, backend: &B) {
        let counts = {
            let mut load = self.load.borrow_mut();
            let routed = load.since_balance.len();
            std::mem::replace(&mut load.since_balance, vec![0; routed])
        };
        let Balance::Bias { rate } = self.dims.balance else {
            return;
        };
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return;
        }
        let mean = total as f64 / counts.len() as f64;
        let bias = backend
            .to_vec(self.routing_bias.value())
            .iter()
            .zip(&counts)
            .map(|(bias, &count)| {
                let count = count as f64;
                let step = if count > mean {
                    -rate
                } else if count < mean {
                    rate
                } else {
                    0.0
                };
                B::Scalar::from_f64(bias.to_f64() + step)
            })
            .collect();
        *self.routing_bias.value_mut() = backend.from_vec(bias, &[counts.len()]);
    }
}

module!(MixtureOfExperts {
    shared,
    experts,
    router,
    routing_bias
});

/// `1 / (1 + e^(-x))`, élément par élément.
fn sigmoid<B: Backend>(tape: &Tape<B>, x: Var) -> Var {
    let one = tape.scalar(B::Scalar::ONE);
    tape.div(one, tape.add(one, tape.exp(tape.neg(x))))
}

/// Le résultat d'un aiguillage : qui va où.
#[derive(Debug, PartialEq)]
pub struct Routing {
    /// Pour chaque expert, les tokens qui l'ont choisi, dans l'ordre.
    pub groups: Vec<Vec<u32>>,
    /// Pour chaque token puis chaque expert, vrai si l'expert est gardé.
    pub mask: Vec<bool>,
}

impl Routing {
    /// La taille à laquelle les groupes sont arrondis, pour `count` tokens.
    pub fn chunk(count: usize) -> usize {
        (count / 32).max(1)
    }

    /// Le groupe de l'expert `expert`, allongé jusqu'à une taille ronde.
    ///
    /// La taille d'un groupe change à chaque passe. Or le backend GPU
    /// réutilise ses tampons d'une passe à l'autre, et ne le peut que si
    /// les tailles se répètent : des groupes de 2 041, puis 2 077, puis
    /// 1 998 tokens lui feraient réserver de la mémoire à chaque opération.
    /// Les tailles sont donc arrondies au trente-deuxième du nombre de
    /// tokens.
    ///
    /// Le groupe est complété par des tokens qui n'ont *pas* choisi cet
    /// expert. Leur porte pour lui vaut exactement zéro : ils traversent
    /// l'expert, mais leur résultat est annulé et ne reçoit aucun gradient.
    pub fn padded_group(&self, expert: usize) -> Vec<u32> {
        let experts = self.groups.len();
        let count = self.mask.len() / experts;
        let mut group = self.groups[expert].clone();
        let step = Self::chunk(count);
        let target = group.len().next_multiple_of(step).min(count);
        let outsiders =
            (0..count as u32).filter(|&token| !self.mask[token as usize * experts + expert]);
        group.extend(outsiders.take(target - group.len()));
        group
    }
}

/// Choisit pour chaque token ses `active` experts.
///
/// `scores` contient une ligne par token et une colonne par expert ; `bias`
/// a un nombre par expert, ajouté à sa colonne avant le classement. À
/// égalité, l'expert de plus petit rang l'emporte : le résultat ne dépend
/// que des nombres.
pub fn route(scores: &[f64], bias: &[f64], active: usize) -> Routing {
    let experts = bias.len();
    assert!(
        experts > 0 && scores.len().is_multiple_of(experts) && active <= experts,
        "aiguillage : {} scores pour {experts} experts dont {active} actifs",
        scores.len(),
    );
    let mut groups = vec![Vec::new(); experts];
    let mut mask = vec![false; scores.len()];
    let mut order: Vec<usize> = Vec::with_capacity(experts);
    for (token, row) in scores.chunks(experts).enumerate() {
        order.clear();
        order.extend(0..experts);
        // Tri stable et décroissant : à score égal, le rang décide.
        order.sort_by(|&a, &b| (row[b] + bias[b]).total_cmp(&(row[a] + bias[a])));
        for &expert in &order[..active] {
            mask[token * experts + expert] = true;
        }
        // Dans l'ordre des experts, pour que chaque groupe reste trié.
        for (expert, group) in groups.iter_mut().enumerate() {
            if mask[token * experts + expert] {
                group.push(token as u32);
            }
        }
    }
    Routing { groups, mask }
}
