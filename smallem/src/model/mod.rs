//! GPT : un Transformer « decoder-only ».
//!
//! ```text
//! tokens [B, T] → embeddings de tokens + embeddings de positions [B, T, C]
//!               → N blocs (attention causale, puis feed-forward)
//!               → LayerNorm finale
//!               → logits [B, T, V]
//! ```
//!
//! À chaque position, le modèle prédit le token suivant en ne voyant que
//! les positions précédentes. Une séquence de `T` tokens fournit donc `T`
//! exemples d'entraînement d'un coup.

use std::cell::Cell;

use rand::Rng;

use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};
use crate::cache::{CacheStep, KvCache, PAGE_SIZE, SequenceId};
use crate::nn::{
    Adapter, AttentionKind, Block, Calibration, Embedding, Lent, Memory, MixtureOfExperts, Module,
    MultiTokenHead, Norm, Parameter, PositionKind, Rope, module,
};
use crate::optim::OptimizerState;
use crate::quantize::{Format, Quantized};

mod config;

pub use config::GptConfig;

pub struct Gpt<B: Backend> {
    config: GptConfig,
    /// Un vecteur par token du vocabulaire. Cette même table sert, à la
    /// sortie, à transformer les vecteurs en logits.
    token_embedding: Embedding<B>,
    /// Un vecteur par position : l'attention ne connaît pas l'ordre des
    /// tokens, c'est cette table qui le lui apprend. Absente avec RoPE, où
    /// la position entre dans l'attention elle-même.
    position_embedding: Option<Embedding<B>>,
    blocks: Vec<Block<B>>,
    final_norm: Norm<B>,
    /// Le module de prédiction multi-token, s'il y en a un. Il ne sert qu'à
    /// [`Gpt::training_loss`].
    multi_token: Option<MultiTokenHead<B>>,
    /// La mémoire de n-grammes, s'il y en a une.
    memory: Option<Memory<B>>,
    /// Part de l'entraînement déjà faite, entre 0 et 1 : le poids de la
    /// prédiction multi-token en dépend.
    progress: Cell<f64>,
    /// Facteur des logits : 1, sauf sous μP (voir [`Gpt::apply_mup`]). Il
    /// n'est pas enregistré avec les poids.
    output_scale: f64,
}

impl<B: Backend> Gpt<B> {
    /// Un modèle aux poids tirés au hasard. Une configuration refusée par
    /// [`GptConfig::validate`] arrête le programme : voir [`Gpt::try_new`].
    pub fn new(backend: &B, config: GptConfig, rng: &mut impl Rng) -> Self {
        Self::try_new(backend, config, rng).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Comme [`Gpt::new`], une configuration refusée étant une erreur.
    pub fn try_new(backend: &B, config: GptConfig, rng: &mut impl Rng) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            token_embedding: Embedding::new(backend, config.vocab, config.embed, rng),
            position_embedding: match config.architecture.positions {
                PositionKind::Learned => {
                    Some(Embedding::new(backend, config.context, config.embed, rng))
                }
                PositionKind::Rotary => None,
            },
            blocks: (0..config.layers)
                .map(|layer| {
                    let borrows = config.architecture.borrows_keys(layer);
                    Block::new(backend, config.block_shape(), borrows, rng)
                })
                .collect(),
            final_norm: Norm::new(backend, config.architecture.norm, config.embed),
            multi_token: config
                .architecture
                .multi_token
                .map(|_| MultiTokenHead::new(backend, config.block_shape(), rng)),
            memory: config
                .architecture
                .memory
                .map(|dims| Memory::new(backend, dims, config.embed, rng)),
            progress: Cell::new(0.0),
            output_scale: 1.0,
        })
    }

    pub fn config(&self) -> GptConfig {
        self.config
    }

    /// `tokens` contient `batch` séquences de même longueur `T`, l'une après
    /// l'autre. Renvoie les logits `[batch, T, vocab]` : à chaque position,
    /// un score par token possible pour la position suivante.
    pub fn logits(&self, tape: &Tape<B>, tokens: &[u32], batch: usize) -> Var {
        self.head(tape, self.hidden(tape, tokens, batch))
    }

    /// Les états cachés à la sortie du dernier bloc, `[batch, T, C]` : tout
    /// le modèle sauf sa sortie. Les arguments sont ceux de [`Gpt::logits`].
    fn hidden(&self, tape: &Tape<B>, tokens: &[u32], batch: usize) -> Var {
        assert!(
            batch > 0 && tokens.len().is_multiple_of(batch),
            "{} tokens ne forment pas {batch} séquences de même longueur",
            tokens.len(),
        );
        let length = tokens.len() / batch;
        let context = self.config.context;
        assert!(
            length <= context,
            "séquence de {length} tokens, le contexte maximal est {context}"
        );
        let dim = self.config.embed;

        // `[B × T, C] → [B, T, C]`, plus les positions `[T, C]` qui
        // s'étirent sur le lot : toutes les séquences partagent les mêmes.
        let embedded = self.token_embedding.forward(tape, tokens);
        let mut x = tape.reshape(embedded, &[batch, length, dim]);
        if let Some(position_embedding) = &self.position_embedding {
            let positions: Vec<u32> = (0..length as u32).collect();
            x = tape.add(x, position_embedding.forward(tape, &positions));
        }
        // Avec RoPE, rien n'est ajouté à l'entrée : les angles des positions
        // sont calculés une fois ici, et chaque bloc s'en sert dans son
        // attention.
        let rope = self.rope(tape, 0, length);
        // Les lignes de la mémoire ne dépendent que des tokens : elles se
        // calculent avant tout le reste.
        let indices = self
            .memory
            .as_ref()
            .map(|memory| memory.hash().indices(tokens, batch));
        self.through_blocks(
            &mut (),
            x,
            |_, memory, x| {
                let indices = indices.as_ref().expect("indices de la mémoire");
                memory.forward(tape, x, indices, length)
            },
            |_, _, block, x, lent| block.forward(tape, x, rope, None, lent),
        )
    }

    /// La pile de blocs, traversée une fois, ou plusieurs : les mêmes blocs
    /// resservent alors, sur d'autres états cachés. La mémoire, s'il y en a
    /// une, s'ajoute au flux par `memory` avant le bloc de son rang lors de
    /// la première traversée, ou après le dernier bloc. `block` fait passer
    /// le flux par un bloc ; il reçoit le rang de l'étage, `passe × couches
    /// plus bloc`, et les clés et valeurs du dernier bloc qui en a calculé,
    /// pour ceux qui empruntent les leurs. `state` est ce que les deux
    /// partagent : le cache, en génération.
    fn through_blocks<S>(
        &self,
        state: &mut S,
        mut x: Var,
        memory: impl Fn(&mut S, &Memory<B>, Var) -> Var,
        block: impl Fn(&mut S, usize, &Block<B>, Var, &mut Option<Lent>) -> Var,
    ) -> Var {
        let layers = self.blocks.len();
        for pass in 0..self.config.architecture.passes {
            let mut lent = None;
            for (index, layer) in self.blocks.iter().enumerate() {
                if let Some(module) = &self.memory
                    && pass == 0
                    && index == module.dims().layer
                {
                    x = memory(state, module, x);
                }
                x = block(state, pass * layers + index, layer, x, &mut lent);
            }
        }
        if let Some(module) = &self.memory
            && module.dims().layer == layers
        {
            x = memory(state, module, x);
        }
        x
    }

    /// La passe de la génération. `entries` donne, pour une ou plusieurs
    /// séquences suivies par `cache`, les tokens qui s'y ajoutent : toute
    /// l'amorce quand on la lit, un seul token par séquence quand on
    /// génère. Renvoie les logits `[R, vocab]` de ces `R` tokens, dans
    /// l'ordre des entrées.
    ///
    /// Seuls les nouveaux tokens traversent le modèle ; ce qu'ils ont
    /// besoin de savoir du passé, les clés et les valeurs de chaque couche,
    /// sort du cache, qui est complété au passage. C'est le calcul de
    /// [`Gpt::logits`] : à cause du masque causal, une position ne dépend
    /// jamais de celles qui la suivent, et traiter une séquence d'un coup ou
    /// morceau par morceau donne les mêmes nombres.
    ///
    /// Les tokens de toutes les séquences forment une seule suite `[1, R,
    /// C]`. Tout ce qui ne mélange pas les positions, c'est-à-dire tout sauf
    /// l'attention, les traite comme une séquence ordinaire ; l'attention
    /// sait, par le pas du cache, à quelle séquence appartient chacun.
    ///
    /// Le modèle ne voit jamais plus de `context` positions d'une séquence.
    /// Quand la suite ferait déborder une séquence, un modèle à positions
    /// relatives oublie ses plus anciennes positions ; avec des positions
    /// apprises, c'est une erreur.
    pub fn forward_cached(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        entries: &[(SequenceId, &[u32])],
    ) -> Var {
        let hidden = self.hidden_cached(tape, cache, entries);
        self.head_cached(tape, hidden)
    }

    /// Comme [`Gpt::forward_cached`], et rend aussi les états cachés à la
    /// sortie du dernier bloc, `[1, R, C]` : ce dont le module de prédiction
    /// multi-token a besoin.
    pub fn forward_cached_with_hidden(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        entries: &[(SequenceId, &[u32])],
    ) -> (Var, Var) {
        let hidden = self.hidden_cached(tape, cache, entries);
        (hidden, self.head_cached(tape, hidden))
    }

    /// Les états cachés de la passe de [`Gpt::forward_cached`], `[1, R, C]`.
    fn hidden_cached(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        entries: &[(SequenceId, &[u32])],
    ) -> Var {
        let context = self.config.context;
        let dim = self.config.embed;
        if !self.has_relative_positions() {
            for (id, tokens) in entries {
                let total = cache.len(*id) + tokens.len();
                assert!(
                    total <= context,
                    "séquence de {total} tokens, le contexte maximal est {context}"
                );
            }
        }
        let counts: Vec<(SequenceId, usize)> = entries
            .iter()
            .map(|(id, tokens)| (*id, tokens.len()))
            .collect();
        let (mut step, positions) = cache.begin(tape, &counts, context);
        let tokens: Vec<u32> = entries
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().copied())
            .collect();
        let rows = tokens.len();

        let embedded = self.token_embedding.forward(tape, &tokens);
        let mut x = tape.reshape(embedded, &[1, rows, dim]);
        if let Some(position_embedding) = &self.position_embedding {
            let positions: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let embedded = position_embedding.forward(tape, &positions);
            x = tape.add(x, tape.reshape(embedded, &[1, rows, dim]));
        }
        self.set_rope(tape, &mut step, &positions);
        // Les n-grammes des nouveaux tokens ont besoin des derniers tokens
        // de chaque séquence : le cache les garde.
        let indices = self.memory.as_ref().map(|memory| {
            let hash = memory.hash();
            let keep = hash.max_order() - 1;
            let mut indices = Vec::with_capacity(rows * hash.tables());
            let mut gram = vec![0u32; hash.max_order()];
            for (id, tokens) in entries {
                let mut combined = cache.recent(*id).to_vec();
                let past = combined.len();
                combined.extend_from_slice(tokens);
                for t in past..combined.len() {
                    crate::ngram::suffix(&combined, t, &mut gram);
                    indices.extend(hash.position_indices(&gram));
                }
                cache.remember(*id, tokens, keep);
            }
            indices
        });
        self.through_blocks(
            cache,
            x,
            |cache, memory, x| {
                let indices = indices.as_ref().expect("indices de la mémoire");
                memory.forward_cached(tape, x, indices, cache, &counts)
            },
            |cache, stage, block, x, lent| {
                let stage = cache.stage_mut(stage);
                block.forward(tape, x, None, Some((stage, &step)), lent)
            },
        )
    }

    /// Les angles de RoPE des tokens d'une passe de génération, s'il y en
    /// a, dans le pas du cache.
    fn set_rope(&self, tape: &Tape<B>, step: &mut CacheStep, positions: &[usize]) {
        if self.config.architecture.positions == PositionKind::Rotary {
            let rotary = self.config.rotary_dim();
            step.rope = Some(Rope::at(tape, positions, rotary, self.config.heads));
            if let AttentionKind::Latent(_) = self.config.architecture.attention {
                step.rope_single = Some(Rope::at(tape, positions, rotary, 1));
            }
        }
    }

    /// La sortie du modèle en génération, de `[1, R, C]` à `[R, V]` : le
    /// calcul de [`Gpt::head`], par `a · bᵀ`, pour ne pas recopier la table
    /// d'embedding transposée à chaque token.
    fn head_cached(&self, tape: &Tape<B>, hidden: Var) -> Var {
        let rows = tape.shape(hidden)[1];
        let x = self.final_norm.forward(tape, hidden);
        let logits = self.token_embedding.unembed(tape, x);
        let logits = self.scale_output(tape, logits);
        tape.reshape(logits, &[rows, self.config.vocab])
    }

    /// Le module de prédiction multi-token en génération. `hidden` contient
    /// les états cachés `h_i` du modèle, `[n, C]`, de positions consécutives
    /// d'une séquence que suit `cache`, un cache d'un seul étage, et `next`
    /// les tokens `t_{i+1}` qui les suivent. Rend `[n, V]` : à chaque
    /// position, les logits du token `t_{i+2}`.
    ///
    /// Le bloc du module a sa propre attention, sur les positions déjà
    /// passées par lui : d'où son cache.
    pub fn multi_token_cached(
        &self,
        tape: &Tape<B>,
        cache: &mut KvCache<B>,
        id: SequenceId,
        hidden: Var,
        next: &[u32],
    ) -> Var {
        let module = self
            .multi_token
            .as_ref()
            .expect("ce modèle n'a pas de module de prédiction multi-token");
        let (rows, dim) = (next.len(), self.config.embed);
        let (mut step, positions) = cache.begin(tape, &[(id, rows)], self.config.context);
        self.set_rope(tape, &mut step, &positions);
        let hidden = tape.reshape(hidden, &[1, rows, dim]);
        let next = tape.reshape(self.token_embedding.forward(tape, next), &[1, rows, dim]);
        let out = module.forward_cached(tape, hidden, next, cache.stage_mut(0), &step);
        self.head_cached(tape, out)
    }

    /// Un cache pour le module de prédiction multi-token : un seul étage,
    /// une séquence.
    pub fn multi_token_cache(&self) -> KvCache<B> {
        KvCache::new(1, self.config.context.div_ceil(PAGE_SIZE) + 1, PAGE_SIZE)
    }

    /// Passe le modèle, juste construit, en paramétrisation μP (Yang et al.,
    /// 2022), relativement à la largeur `base` : à cette largeur, rien ne
    /// change. Pour une largeur `C = m × base` :
    ///
    /// - les matrices cachées, celles des blocs, sont initialisées
    ///   `√m` fois plus petites, et apprennent avec un taux `m` fois plus
    ///   petit ;
    /// - les logits sont divisés par `m` ;
    /// - la table d'embedding, les gains et les biais ne changent pas.
    ///
    /// L'échelle de l'attention, `1/√D`, ne change pas non plus : dans
    /// notre famille de modèles, les têtes gardent la même taille et c'est
    /// leur nombre qui grandit.
    ///
    /// L'idée : sous la paramétrisation standard, une mise à jour d'Adam
    /// change chaque poids d'environ le taux, et une couche de largeur `C`
    /// additionne `C` de ces changements. Ses sorties bougent donc `m` fois
    /// plus quand le modèle est `m` fois plus large, et le meilleur taux
    /// baisse avec la largeur. Diviser le taux des matrices cachées par `m`
    /// rend ce mouvement indépendant de la largeur.
    pub fn apply_mup(&mut self, backend: &B, base: usize) {
        let m = self.config.embed as f64 / base as f64;
        let shrink = backend.scalar(B::Scalar::from_f64(1.0 / m.sqrt()));
        for block in &mut self.blocks {
            block.walk_mut("", &mut |_, parameter| {
                if backend.shape(parameter.value()).len() == 2 {
                    let value = backend.mul(parameter.value(), &shrink);
                    *parameter.value_mut() = value;
                    parameter.set_lr_scale(1.0 / m);
                }
            });
        }
        self.output_scale = 1.0 / m;
    }

    /// Ajoute un adaptateur LoRA de rang `rank` à chaque couche linéaire des
    /// blocs, et à la table d'embedding, puis gèle tous les autres
    /// paramètres : seuls les adaptateurs apprendront. Rend le nombre de
    /// paramètres entraînés.
    ///
    /// La table d'embedding a son adaptateur, `[V, r] · [r, C]`, parce
    /// qu'elle porte les tokens de rôle, nés identiques : gelée, elle ne
    /// pourrait pas apprendre à les distinguer.
    pub fn add_adapters(&mut self, backend: &B, rank: usize, rng: &mut impl Rng) -> usize {
        for parameter in self.parameters_mut() {
            parameter.set_frozen(true);
        }
        let mut count = 0;
        let (vocab, dim) = (self.config.vocab, self.config.embed);
        let adapter = Adapter::new(backend, vocab, dim, rank, rng);
        count += adapter.parameter_count(backend);
        self.token_embedding.adapter = Some(Box::new(adapter));
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                let shape = backend.shape(linear.weight.value()).to_vec();
                let adapter = Adapter::new(backend, shape[0], shape[1], rank, rng);
                count += adapter.parameter_count(backend);
                linear.adapter = Some(Box::new(adapter));
            }
        }
        count
    }

    /// Ajoute chaque correction à sa matrice, `W ← W + A · B`, retire les
    /// adaptateurs et dégèle tout : le modèle redevient un modèle ordinaire,
    /// qui donne les mêmes logits.
    pub fn merge_adapters(&mut self, backend: &B) {
        if let Some(adapter) = self.token_embedding.adapter.take() {
            adapter.merge_into(backend, &mut self.token_embedding.weight);
        }
        for block in &mut self.blocks {
            for linear in block.linears_mut() {
                if let Some(adapter) = linear.adapter.take() {
                    adapter.merge_into(backend, &mut linear.weight);
                }
            }
        }
        for parameter in self.parameters_mut() {
            parameter.set_frozen(false);
        }
    }

    /// Agrandit le vocabulaire à `vocab` tokens : la table d'embedding,
    /// qui sert aussi à la sortie, reçoit de nouvelles lignes. Chacune est
    /// la moyenne des lignes existantes : un token nouveau commence au
    /// centre de ceux que le modèle connaît, et son logit, à la moyenne des
    /// autres. Les logits des anciens tokens ne changent pas.
    ///
    /// Le vocabulaire s'agrandit avant d'ajouter les adaptateurs LoRA
    /// ([`Gpt::add_adapters`]) et de quantifier la sortie
    /// ([`Gpt::quantize_output`]) : leurs tables ont une ligne par token.
    pub fn grow_vocab(&mut self, backend: &B, vocab: usize) {
        let (old, dim) = (self.config.vocab, self.config.embed);
        assert!(vocab >= old, "le vocabulaire ne peut que grandir");
        assert!(
            self.token_embedding.adapter.is_none() && self.token_embedding.output.is_none(),
            "le vocabulaire s'agrandit avant les adaptateurs et la quantification de la sortie"
        );
        let mut table = backend.to_vec(self.token_embedding.weight.value());
        let mut mean = vec![0.0; dim];
        for row in table.chunks(dim) {
            for (m, value) in mean.iter_mut().zip(row) {
                *m += value.to_f64() / old as f64;
            }
        }
        for _ in old..vocab {
            table.extend(mean.iter().map(|&m| B::Scalar::from_f64(m)));
        }
        *self.token_embedding.weight.value_mut() = backend.from_vec(table, &[vocab, dim]);
        self.config.vocab = vocab;
    }

    /// L'entropie croisée moyenne sur les seules positions de `scored` :
    /// celles dont la cible est un token écrit par l'assistant. Les autres
    /// positions sont lues, et leur contenu change donc la perte, mais leur
    /// cible n'y entre pas.
    pub fn masked_loss(
        &self,
        tape: &Tape<B>,
        tokens: &[u32],
        targets: &[u32],
        scored: &[bool],
        batch: usize,
    ) -> Var {
        assert!(
            tokens.len() == targets.len() && tokens.len() == scored.len(),
            "une cible et un masque par token sont attendus"
        );
        let rows: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&i| scored[i as usize])
            .collect();
        assert!(!rows.is_empty(), "aucune position notée");
        let logits = self.logits(tape, tokens, batch);
        let flat = tape.reshape(logits, &[tokens.len(), self.config.vocab]);
        let kept: Vec<u32> = rows.iter().map(|&row| targets[row as usize]).collect();
        tape.cross_entropy(tape.gather_rows(flat, &rows), &kept)
    }

    /// Quantifie au format `format` les matrices de toutes les couches
    /// linéaires : celles des attentions et des feed-forward. La table
    /// d'embedding, qui sert aussi à la sortie, reste en `f32`. Rend, pour
    /// chaque matrice, son nom et sa forme quantifiée.
    pub fn quantize(&mut self, backend: &B, format: Format) -> Vec<(String, Quantized)> {
        self.quantize_with(backend, |_, weight, rows, cols| {
            Quantized::round_to_nearest(weight, rows, cols, format)
        })
    }

    /// Comme [`Gpt::quantize`], chaque matrice étant quantifiée par
    /// `quantizer`, qui reçoit son nom, ses poids `[rows, cols]` et leurs
    /// dimensions.
    pub fn quantize_with(
        &mut self,
        backend: &B,
        mut quantizer: impl FnMut(&str, &[f64], usize, usize) -> Quantized,
    ) -> Vec<(String, Quantized)> {
        let mut done = Vec::new();
        self.walk_mut("", &mut |name, parameter| {
            if Self::is_quantizable(backend, &name, parameter) {
                let shape = backend.shape(parameter.value()).to_vec();
                let weight: Vec<f64> = backend
                    .to_vec(parameter.value())
                    .iter()
                    .map(|v| v.to_f64())
                    .collect();
                let quantized = quantizer(&name, &weight, shape[0], shape[1]);
                parameter.set_quantized(backend, &quantized);
                done.push((name, quantized));
            }
        });
        done
    }

    /// Quantifie aussi la table d'embedding, pour la sortie du modèle (voir
    /// [`Embedding::quantize_output`](crate::nn::Embedding::quantize_output)).
    pub fn quantize_output(&mut self, backend: &B, format: Format) -> Quantized {
        self.token_embedding.quantize_output(backend, format)
    }

    /// Les matrices que [`Gpt::quantize`] quantifie : celles des couches
    /// linéaires, pas les tables d'embedding.
    fn is_quantizable(backend: &B, name: &str, parameter: &Parameter<B>) -> bool {
        let matrix = backend.shape(parameter.value()).len() == 2;
        matrix && name.ends_with(".weight") && !name.ends_with("embedding.weight")
    }

    /// Commence à observer les entrées de toutes les matrices que
    /// [`Gpt::quantize`] quantifierait (voir [`Parameter::observe`]).
    pub fn start_calibration(&self, backend: &B, host: bool) {
        self.walk("", &mut |name, parameter| {
            if Self::is_quantizable(backend, &name, parameter) {
                parameter.start_calibration(host);
            }
        });
    }

    /// Arrête l'observation, et rend ce qu'elle a appris de chaque matrice.
    pub fn take_calibration(&self) -> Vec<(String, Calibration<B>)> {
        let mut taken = Vec::new();
        self.walk("", &mut |name, parameter| {
            if let Some(calibration) = parameter.take_calibration() {
                taken.push((name, calibration));
            }
        });
        taken
    }

    /// Un cache vide pour ce modèle, de quoi suivre `sequences` séquences
    /// de `context` positions.
    pub fn cache(&self, sequences: usize) -> KvCache<B> {
        KvCache::for_model(&self.config, sequences)
    }

    /// La sortie du modèle : des états cachés `[…, C]` aux logits `[…, V]`.
    fn head(&self, tape: &Tape<B>, x: Var) -> Var {
        let x = self.final_norm.forward(tape, x);

        // Partage des poids : la table `[V, C]` qui a transformé les tokens
        // en vecteurs sert, transposée en `[C, V]`, à transformer les
        // vecteurs en scores de tokens. Le logit d'un token est le produit
        // scalaire entre le vecteur de sortie et l'embedding de ce token.
        if self.token_embedding.output.is_some() {
            return self.scale_output(tape, self.token_embedding.unembed(tape, x));
        }
        let unembedding = tape.transpose(self.token_embedding.table(tape), 0, 1);
        self.scale_output(tape, tape.matmul(x, unembedding))
    }

    /// Les logits multipliés par le facteur de sortie, s'il n'est pas 1.
    fn scale_output(&self, tape: &Tape<B>, logits: Var) -> Var {
        if self.output_scale == 1.0 {
            logits
        } else {
            tape.mul(logits, tape.scalar(B::Scalar::from_f64(self.output_scale)))
        }
    }

    /// Entropie croisée moyenne sur toutes les positions de toutes les
    /// séquences. `targets[i]` est le token qui suit `tokens[i]`.
    pub fn loss(&self, tape: &Tape<B>, tokens: &[u32], targets: &[u32], batch: usize) -> Var {
        assert_eq!(
            tokens.len(),
            targets.len(),
            "une cible par token est attendue"
        );
        let logits = self.logits(tape, tokens, batch);
        // L'entropie croisée attend `[N, V]` : toutes les positions de
        // toutes les séquences deviennent des exemples indépendants.
        let flat = tape.reshape(logits, &[tokens.len(), self.config.vocab]);
        tape.cross_entropy(flat, targets)
    }

    /// La perte à minimiser pendant l'entraînement : celle de
    /// [`Gpt::loss`], plus les termes qui ne servent qu'à guider
    /// l'apprentissage : la pénalité d'équilibrage des mélanges d'experts
    /// qui en ont une, et la perte de la prédiction multi-token.
    ///
    /// La perte de validation reste [`Gpt::loss`] : c'est elle qui se
    /// compare d'un modèle à l'autre.
    pub fn training_loss(
        &self,
        tape: &Tape<B>,
        tokens: &[u32],
        targets: &[u32],
        batch: usize,
    ) -> Var {
        assert_eq!(
            tokens.len(),
            targets.len(),
            "une cible par token est attendue"
        );
        let hidden = self.hidden(tape, tokens, batch);
        let logits = self.head(tape, hidden);
        let flat = tape.reshape(logits, &[tokens.len(), self.config.vocab]);
        let mut loss = tape.cross_entropy(flat, targets);

        if let Some(extra) = self.multi_token_loss(tape, hidden, targets, batch) {
            loss = tape.add(loss, extra);
        }
        for experts in self.experts() {
            if let Some(penalty) = experts.penalty(tape) {
                loss = tape.add(loss, penalty);
            }
        }
        loss
    }

    /// La perte de la prédiction multi-token, déjà multipliée par son
    /// poids. `None` si le modèle n'a pas ce module, ou si son poids est
    /// nul.
    ///
    /// À la position `i`, la cible `targets[i]` est le token `t_{i+1}` :
    /// le module le reçoit en entrée, et doit prédire `targets[i + 1]`,
    /// soit `t_{i+2}`. La dernière position de chaque séquence n'a pas de
    /// cible aussi lointaine : elle est laissée de côté.
    fn multi_token_loss(
        &self,
        tape: &Tape<B>,
        hidden: Var,
        targets: &[u32],
        batch: usize,
    ) -> Option<Var> {
        let module = self.multi_token.as_ref()?;
        let weight = self
            .config
            .architecture
            .multi_token?
            .weight_at(self.progress.get());
        let length = targets.len() / batch;
        if weight == 0.0 || length < 2 {
            return None;
        }
        let dim = self.config.embed;
        // Toutes les positions sauf la dernière de chaque séquence.
        let rows: Vec<u32> = (0..batch)
            .flat_map(|sequence| (0..length - 1).map(move |t| (sequence * length + t) as u32))
            .collect();
        let next: Vec<u32> = rows.iter().map(|&row| targets[row as usize]).collect();
        let after: Vec<u32> = rows.iter().map(|&row| targets[row as usize + 1]).collect();

        let shape = [batch, length - 1, dim];
        let hidden = tape.reshape(hidden, &[batch * length, dim]);
        let hidden = tape.reshape(tape.gather_rows(hidden, &rows), &shape);
        let next = tape.reshape(self.token_embedding.forward(tape, &next), &shape);
        let rope = self.rope(tape, 0, length - 1);
        let logits = self.head(tape, module.forward(tape, hidden, next, rope));
        let flat = tape.reshape(logits, &[rows.len(), self.config.vocab]);
        let loss = tape.cross_entropy(flat, &after);
        Some(tape.mul(loss, tape.scalar(B::Scalar::from_f64(weight))))
    }

    /// La mémoire de n-grammes, s'il y en a une.
    pub fn memory(&self) -> Option<&Memory<B>> {
        self.memory.as_ref()
    }

    pub fn memory_mut(&mut self) -> Option<&mut Memory<B>> {
        self.memory.as_mut()
    }

    /// Le pas d'AdamW des lignes lues de la mémoire depuis le dernier pas :
    /// voir [`Memory::apply`]. Leur gradient a été gardé par les pas
    /// d'entraînement ([`Module::collect_gradients`]).
    pub fn apply_memory_gradient(&mut self, backend: &B, learning_rate: f64, grad_scale: f64) {
        if let Some(memory) = &mut self.memory {
            memory.apply(backend, learning_rate, grad_scale);
        }
    }

    /// L'état de l'AdamW que la mémoire creuse tient à part
    /// ([`Memory::optimizer_state`]), rangé sur tous les paramètres du
    /// modèle : ce qu'un point de reprise ajoute à l'état de l'optimiseur.
    pub fn memory_optimizer_state(&self) -> OptimizerState<&B::Tensor> {
        let Some(memory) = &self.memory else {
            return OptimizerState::default();
        };
        let before = self.memory_offset(memory);
        let mut state = memory.optimizer_state();
        for (_, tensors) in &mut state.slots {
            tensors.splice(0..0, std::iter::repeat_n(None, before));
        }
        state
    }

    /// Les préfixes des séries de [`Gpt::memory_optimizer_state`].
    pub fn memory_optimizer_slots(&self) -> Vec<&'static str> {
        self.memory
            .as_ref()
            .map_or_else(Vec::new, Memory::optimizer_slots)
    }

    /// Remet la mémoire creuse dans un état relu d'un point de reprise,
    /// rangé comme celui de [`Gpt::memory_optimizer_state`].
    pub fn restore_memory_optimizer_state(
        &mut self,
        mut state: OptimizerState<B::Tensor>,
    ) -> Result<(), String> {
        let Some(memory) = &self.memory else {
            return Ok(());
        };
        let before = self.memory_offset(memory);
        for (_, tensors) in &mut state.slots {
            if tensors.len() < before {
                return Err("état de la mémoire : séries trop courtes".into());
            }
            tensors.drain(..before);
        }
        self.memory
            .as_mut()
            .expect("mémoire")
            .restore_optimizer_state(state)
    }

    /// Le rang du premier paramètre de la mémoire parmi ceux du modèle : elle
    /// vient en dernier (voir l'appel à `module!`).
    fn memory_offset(&self, memory: &Memory<B>) -> usize {
        let before = self.parameters().len() - memory.parameters().len();
        debug_assert!(
            self.parameter_names()
                .get(before)
                .is_none_or(|name| name.starts_with("memory"))
        );
        before
    }

    /// Dit au modèle quelle part de l'entraînement est faite, entre 0 et
    /// 1 : le poids de la prédiction multi-token en dépend.
    pub fn set_training_progress(&self, progress: f64) {
        self.progress.set(progress);
    }

    /// Les mélanges d'experts du modèle, un par bloc qui en a un. Celui du
    /// module de prédiction multi-token vient en dernier.
    pub fn experts(&self) -> impl Iterator<Item = &MixtureOfExperts<B>> {
        self.blocks
            .iter()
            .chain(self.multi_token.iter().map(|module| &module.block))
            .filter_map(|block| block.feed_forward.experts())
    }

    fn experts_mut(&mut self) -> impl Iterator<Item = &mut MixtureOfExperts<B>> {
        self.blocks
            .iter_mut()
            .chain(self.multi_token.iter_mut().map(|module| &mut module.block))
            .filter_map(|block| block.feed_forward.experts_mut())
    }

    /// Pour chaque mélange d'experts, le nombre de tokens reçus par chacun
    /// de ses experts aiguillés depuis le dernier appel.
    pub fn take_expert_load(&self) -> Vec<Vec<u64>> {
        self.experts().map(MixtureOfExperts::take_load).collect()
    }

    /// Oublie les comptes de tokens en cours : à appeler après une passe
    /// qui n'était pas de l'entraînement.
    pub fn reset_expert_load(&self) {
        self.experts().for_each(MixtureOfExperts::reset_load);
    }

    /// Règle les biais d'aiguillage après un pas d'entraînement (voir
    /// [`MixtureOfExperts::rebalance`]).
    pub fn rebalance_experts(&mut self, backend: &B) {
        for experts in self.experts_mut() {
            experts.rebalance(backend);
        }
    }

    /// Fait calculer chaque expert sur tous les tokens (`true`) ou sur les
    /// seuls tokens qui l'ont choisi (`false`, le réglage normal).
    pub fn set_dense_experts(&mut self, dense: bool) {
        for experts in self.experts_mut() {
            experts.dense = dense;
        }
    }

    /// Les angles de RoPE des positions `start..start + length`, si le
    /// modèle utilise RoPE.
    fn rope(&self, tape: &Tape<B>, start: usize, length: usize) -> Option<Rope> {
        match self.config.architecture.positions {
            PositionKind::Learned => None,
            PositionKind::Rotary => Some(Rope::new(tape, start, length, self.config.rotary_dim())),
        }
    }

    /// Vrai si le modèle ne connaît que les distances entre positions
    /// (RoPE), et non leur rang dans la séquence. Une fenêtre de tokens
    /// peut alors glisser sans que rien ne soit à recalculer.
    pub fn has_relative_positions(&self) -> bool {
        self.position_embedding.is_none()
    }
}

module!(Gpt {
    token_embedding,
    position_embedding,
    blocks,
    final_norm,
    multi_token,
    memory
});

#[cfg(test)]
pub(crate) mod tests;
