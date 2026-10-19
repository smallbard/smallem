//! Le cache de génération : ce que les couches d'attention gardent des
//! positions déjà traitées, pour toutes les séquences en cours.
//!
//! En génération, le passé ne change pas : ce qu'une position offre aux
//! suivantes ne dépend que d'elle et de celles qui la précèdent. Le garder
//! évite de tout recalculer à chaque nouveau token.
//!
//! Chaque couche range ce qu'elle garde dans des **tables**, une ligne par
//! position, allouées une fois pour toutes : `[lignes, têtes, largeur]`.
//! Une séquence n'y occupe pas des lignes voisines. Les lignes sont
//! distribuées par **pages** de taille fixe, prises dans une réserve
//! commune quand une séquence s'allonge et rendues quand elle se termine,
//! comme la mémoire virtuelle d'un système d'exploitation. Chaque séquence
//! tient la liste de ses pages, et l'attention retrouve ses clés à travers
//! elle.
//!
//! Écrire une position est alors une écriture sur place, dans une ligne
//! libre : rien n'est recopié quand la séquence grandit. Oublier les plus
//! anciennes positions, ou revenir en arrière, ne touche qu'à la liste.

use crate::autograd::{Tape, Var};
use crate::backend::{Backend, Float};
use crate::model::GptConfig;
use crate::nn::Rope;

/// Nombre de lignes d'une page.
pub const PAGE_SIZE: usize = 16;

/// Une séquence suivie par un [`KvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceId(usize);

/// Ce que le cache sait d'une séquence.
#[derive(Clone, Debug, Default)]
struct Sequence {
    /// Ses pages, dans l'ordre des positions.
    pages: Vec<u32>,
    /// Rang, dans la première page, de la plus ancienne position gardée.
    /// Il n'est pas nul quand des positions ont été oubliées.
    start: usize,
    /// Nombre de positions gardées.
    len: usize,
    /// Position du prochain token dans la séquence. Elle dépasse `len` dès
    /// que des positions ont été oubliées.
    next_position: usize,
    /// Ses derniers tokens, du plus ancien au plus récent : les n-grammes
    /// de la mémoire du modèle, s'il en a une, en ont besoin.
    recent: Vec<u32>,
}

/// Le cache de génération de toutes les séquences en cours.
pub struct KvCache<B: Backend> {
    /// Une entrée par couche d'attention et par traversée de la pile.
    stages: Vec<AttentionCache<B>>,
    page_size: usize,
    /// Nombre total de pages : les tables ont `pages × page_size` lignes.
    pages: usize,
    /// Les pages libres ; la prochaine donnée est la dernière.
    free: Vec<u32>,
    /// Indexé par [`SequenceId`] ; `None` pour une séquence terminée.
    sequences: Vec<Option<Sequence>>,
    /// Indexé par [`SequenceId`] : les dernières entrées de la convolution
    /// de la mémoire de n-grammes, s'il y en a une.
    memory_rows: Vec<Option<B::Tensor>>,
}

impl<B: Backend> KvCache<B> {
    /// Un cache vide de `pages` pages de `page_size` lignes, pour un modèle
    /// à `stages` couches d'attention (en comptant chaque traversée de la
    /// pile). Les tables ne sont allouées qu'à la première écriture, quand
    /// leur largeur est connue.
    pub fn new(stages: usize, pages: usize, page_size: usize) -> Self {
        assert!(
            pages > 0 && page_size > 0,
            "un cache a au moins une page d'une ligne"
        );
        assert!(
            pages * page_size <= 1 << 24,
            "{} lignes : un f32 ne numérote exactement que 2²⁴ lignes",
            pages * page_size,
        );
        Self {
            stages: (0..stages).map(|_| AttentionCache::new()).collect(),
            page_size,
            pages,
            free: (0..pages as u32).rev().collect(),
            sequences: Vec::new(),
            memory_rows: Vec::new(),
        }
    }

    /// Un cache vide pour un modèle de configuration `config`, de quoi
    /// suivre `sequences` séquences de `context` positions.
    ///
    /// Une séquence qui a oublié ses plus anciennes positions peut
    /// commencer au milieu d'une page : il lui faut alors une page de plus.
    pub fn for_model(config: &GptConfig, sequences: usize) -> Self {
        let stages = config.layers * config.architecture.passes;
        let pages = sequences * (config.context.div_ceil(PAGE_SIZE) + 1);
        Self::new(stages, pages, PAGE_SIZE)
    }

    /// Nombre de lignes de chaque table.
    pub fn capacity(&self) -> usize {
        self.pages * self.page_size
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Nombre de pages encore libres.
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// Commence une nouvelle séquence, vide.
    pub fn add_sequence(&mut self) -> SequenceId {
        let free = self.sequences.iter().position(Option::is_none);
        let index = free.unwrap_or(self.sequences.len());
        if index == self.sequences.len() {
            self.sequences.push(None);
        }
        self.sequences[index] = Some(Sequence::default());
        SequenceId(index)
    }

    /// Les dernières entrées de la convolution de la mémoire pour la
    /// séquence `id`, `[≤ 9, C]`, ou `None` avant sa première passe.
    pub fn memory_rows(&self, id: SequenceId) -> Option<&B::Tensor> {
        self.memory_rows.get(id.0)?.as_ref()
    }

    pub fn set_memory_rows(&mut self, id: SequenceId, rows: B::Tensor) {
        if self.memory_rows.len() <= id.0 {
            self.memory_rows.resize_with(id.0 + 1, || None);
        }
        self.memory_rows[id.0] = Some(rows);
    }

    /// Termine une séquence : ses pages retournent à la réserve.
    pub fn remove_sequence(&mut self, id: SequenceId) {
        if let Some(rows) = self.memory_rows.get_mut(id.0) {
            *rows = None;
        }
        let sequence = self.sequences[id.0].take().expect("séquence déjà terminée");
        self.free.extend(sequence.pages.iter().rev());
    }

    fn sequence(&self, id: SequenceId) -> &Sequence {
        self.sequences[id.0].as_ref().expect("séquence terminée")
    }

    fn sequence_mut(&mut self, id: SequenceId) -> &mut Sequence {
        self.sequences[id.0].as_mut().expect("séquence terminée")
    }

    /// Les derniers tokens de la séquence `id`, du plus ancien au plus
    /// récent : ceux que [`KvCache::remember`] a gardés.
    pub fn recent(&self, id: SequenceId) -> &[u32] {
        &self.sequence(id).recent
    }

    /// Ajoute `tokens` aux derniers tokens de la séquence `id`, et n'en
    /// garde que les `keep` plus récents.
    pub fn remember(&mut self, id: SequenceId, tokens: &[u32], keep: usize) {
        let recent = &mut self.sequence_mut(id).recent;
        recent.extend_from_slice(tokens);
        let excess = recent.len().saturating_sub(keep);
        recent.drain(..excess);
    }

    /// Nombre de positions gardées pour la séquence `id`.
    pub fn len(&self, id: SequenceId) -> usize {
        self.sequence(id).len
    }

    pub fn is_empty(&self, id: SequenceId) -> bool {
        self.len(id) == 0
    }

    /// Position qu'aura le prochain token de la séquence `id`.
    pub fn next_position(&self, id: SequenceId) -> usize {
        self.sequence(id).next_position
    }

    /// Nombre de pages qu'il faudrait prendre à la réserve pour ajouter
    /// `count` positions à la séquence `id`.
    pub fn pages_needed(&self, id: SequenceId, count: usize) -> usize {
        let sequence = self.sequence(id);
        let lines = sequence.start + sequence.len + count;
        lines
            .div_ceil(self.page_size)
            .saturating_sub(sequence.pages.len())
    }

    /// Nombre de nombres gardés pour chaque position, toutes couches
    /// confondues : ce qu'un token de plus coûte en mémoire. Zéro tant que
    /// rien n'a été écrit.
    pub fn numbers_per_position(&self, backend: &B) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.numbers_per_line(backend))
            .sum()
    }

    /// La ligne des tables où est rangée la position gardée de rang `rank`
    /// (0 pour la plus ancienne) de `sequence`.
    fn line(&self, sequence: &Sequence, rank: usize) -> u32 {
        let index = sequence.start + rank;
        sequence.pages[index / self.page_size] * self.page_size as u32
            + (index % self.page_size) as u32
    }

    /// Oublie les `count` plus anciennes positions de la séquence `id`. Les
    /// pages qu'elles libèrent retournent à la réserve.
    pub fn forget(&mut self, id: SequenceId, count: usize) {
        let page_size = self.page_size;
        let sequence = self.sequence_mut(id);
        assert!(
            count <= sequence.len,
            "oubli de plus de positions qu'il n'y en a"
        );
        sequence.start += count;
        sequence.len -= count;
        let done = sequence.start / page_size;
        let freed: Vec<u32> = sequence.pages.drain(..done).collect();
        sequence.start -= done * page_size;
        self.free.extend(freed.iter().rev());
    }

    /// Ramène la séquence `id` à ses `len` premières positions gardées,
    /// comme si les suivantes n'avaient jamais été écrites : le décodage
    /// spéculatif revient ainsi sur les propositions refusées.
    pub fn truncate(&mut self, id: SequenceId, len: usize) {
        let page_size = self.page_size;
        let sequence = self.sequence_mut(id);
        assert!(
            len <= sequence.len,
            "on ne peut que raccourcir une séquence"
        );
        sequence.next_position -= sequence.len - len;
        sequence.len = len;
        let kept = (sequence.start + len).div_ceil(page_size).max(1);
        let freed: Vec<u32> = sequence
            .pages
            .drain(kept.min(sequence.pages.len())..)
            .collect();
        self.free.extend(freed.iter().rev());
    }

    /// Prépare une passe du modèle : `entries` donne, pour chaque séquence
    /// concernée, combien de nouveaux tokens la traversent. Leurs lignes
    /// sont réservées, leurs positions comptées.
    ///
    /// Les nouveaux tokens de toutes les séquences forment une seule suite
    /// de `R` lignes, dans l'ordre des entrées. La passe renvoie, pour
    /// chaque ligne, la position de son token dans sa séquence ; le pas
    /// rendu porte tout ce que l'attention doit savoir pour lire et écrire
    /// les tables.
    ///
    /// Le modèle ne voit jamais plus de `context` positions d'une séquence :
    /// quand la suite la ferait déborder, ses plus anciennes positions sont
    /// oubliées. Seul un modèle à positions relatives peut le supporter.
    ///
    /// # Panics
    ///
    /// Si la réserve n'a plus assez de pages, ou si une séquence apparaît
    /// deux fois.
    pub fn begin(
        &mut self,
        tape: &Tape<B>,
        entries: &[(SequenceId, usize)],
        context: usize,
    ) -> (CacheStep, Vec<usize>) {
        for (index, &(id, count)) in entries.iter().enumerate() {
            assert!(
                entries[..index].iter().all(|&(other, _)| other != id),
                "une séquence ne peut apparaître qu'une fois dans une passe",
            );
            assert!(
                count <= context,
                "{count} nouveaux tokens, le contexte maximal est {context}"
            );
            let overflow = (self.len(id) + count).saturating_sub(context);
            if overflow > 0 {
                self.forget(id, overflow);
            }
            let needed = self.pages_needed(id, count);
            assert!(
                needed <= self.free.len(),
                "cache plein : {needed} pages demandées, {} libres",
                self.free.len(),
            );
            let pages: Vec<u32> = (0..needed)
                .map(|_| self.free.pop().expect("page"))
                .collect();
            self.sequence_mut(id).pages.extend(pages);
        }

        let backend = tape.backend();
        let max_len = entries
            .iter()
            .map(|&(id, count)| self.len(id) + count)
            .max()
            .unwrap_or(0);
        let integer = |value: usize| B::Scalar::from_f64(value as f64);
        let mut slots = vec![B::Scalar::ZERO; entries.len() * max_len];
        let (mut lines, mut rows, mut positions) = (Vec::new(), Vec::new(), Vec::new());
        for (index, &(id, count)) in entries.iter().enumerate() {
            let known = self.len(id);
            let first = self.next_position(id);
            {
                let sequence = self.sequence_mut(id);
                sequence.len += count;
                sequence.next_position += count;
            }
            let sequence = self.sequence(id);
            for rank in 0..known + count {
                slots[index * max_len + rank] = integer(self.line(sequence, rank) as usize);
            }
            for i in 0..count {
                lines.push(integer(self.line(sequence, known + i) as usize));
                rows.extend([integer(index), integer(known + i + 1)]);
                positions.push(first + i);
            }
        }
        let queries = lines.len();
        let step = CacheStep {
            capacity: self.capacity(),
            lines: tape.constant(backend.from_vec(lines, &[queries])),
            slots: tape.constant(backend.from_vec(slots, &[entries.len(), max_len])),
            rows: tape.constant(backend.from_vec(rows, &[queries, 2])),
            rope: None,
            rope_single: None,
        };
        (step, positions)
    }

    /// Ce que garde l'étage `stage` : pour la passe du modèle.
    pub fn stage_mut(&mut self, stage: usize) -> &mut AttentionCache<B> {
        &mut self.stages[stage]
    }
}

/// Ce que l'attention doit savoir, pendant une passe, des tokens qui la
/// traversent : où écrire ce qu'ils offrent, et où lire ce qu'ils voient.
/// Il est le même pour toutes les couches.
#[derive(Clone, Copy, Debug)]
pub struct CacheStep {
    /// Nombre de lignes des tables.
    pub capacity: usize,
    /// `[R]` : la ligne des tables où écrire chaque nouveau token.
    pub lines: Var,
    /// `[S, L]` : les lignes de chaque séquence, dans l'ordre des positions.
    pub slots: Var,
    /// `[R, 2]` : la séquence de chaque token, et combien de positions il
    /// voit.
    pub rows: Var,
    /// Les angles de RoPE des `R` tokens, chaque ligne répétée pour toutes
    /// les têtes : `[R × H, d/2]`.
    pub rope: Option<Rope>,
    /// Les mêmes angles, une ligne par token : `[R, d/2]`. Pour la partie
    /// tournée des clés de l'attention latente, commune à toutes les têtes.
    pub rope_single: Option<Rope>,
}

/// Ce qu'une couche d'attention garde des positions passées.
///
/// L'attention classique garde la clé et la valeur de chaque tête.
/// L'attention latente garde bien moins : le vecteur latent dont clés et
/// valeurs se déduisent, suivi de la partie tournée de la clé. Une couche ne
/// remplit que les tables qui la concernent.
pub struct AttentionCache<B: Backend> {
    /// Attention classique : `[lignes, H, D]`.
    pub keys: Table<B>,
    /// Attention classique : `[lignes, H, D]`.
    pub values: Table<B>,
    /// Attention latente : `[lignes, 1, d_c + d_r]`.
    pub latents: Table<B>,
}

impl<B: Backend> AttentionCache<B> {
    pub fn new() -> Self {
        Self {
            keys: Table(None),
            values: Table(None),
            latents: Table(None),
        }
    }

    /// Nombre de nombres gardés pour chaque position : c'est ce que coûte
    /// en mémoire un token de plus, pour cette couche.
    pub fn numbers_per_line(&self, backend: &B) -> usize {
        [&self.keys, &self.values, &self.latents]
            .iter()
            .filter_map(|table| table.0.as_ref())
            .map(|tensor| backend.shape(tensor)[1..].iter().product::<usize>())
            .sum()
    }
}

impl<B: Backend> Default for AttentionCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Une table du cache : `[lignes, têtes, largeur]`, ou rien tant qu'aucune
/// position n'y a été écrite.
pub struct Table<B: Backend>(pub Option<B::Tensor>);

impl<B: Backend> Table<B> {
    /// Écrit `rows` `[R, têtes, largeur]` aux lignes de `step`, et rend la
    /// table entière, pour que l'attention la lise. Au premier appel, la
    /// table est créée avec `step.capacity` lignes.
    pub fn write(&mut self, tape: &Tape<B>, step: &CacheStep, rows: Var) -> Var {
        let backend = tape.backend();
        let shape = tape.shape(rows);
        let table = self
            .0
            .get_or_insert_with(|| backend.zeros(&[step.capacity, shape[1], shape[2]]));
        backend.write_rows(table, &tape.value(step.lines), &tape.value(rows));
        tape.constant(table.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::Cpu;

    type B = Cpu<f64>;

    /// Les lignes de chaque séquence, telles que les verrait l'attention
    /// d'une passe qui n'ajoute rien : `slots` d'un pas sans nouveau token.
    fn lines(cache: &KvCache<B>, id: SequenceId) -> Vec<u32> {
        let sequence = cache.sequence(id);
        (0..sequence.len)
            .map(|rank| cache.line(sequence, rank))
            .collect()
    }

    /// Ajoute `count` positions à `id`, sans modèle.
    fn grow(cache: &mut KvCache<B>, id: SequenceId, count: usize) {
        let backend = Cpu::new();
        let tape = Tape::new(&backend);
        cache.begin(&tape, &[(id, count)], 100);
    }

    #[test]
    fn sequences_take_whole_pages_from_the_reserve() {
        let mut cache = KvCache::<B>::new(1, 4, 3);
        let (a, b) = (cache.add_sequence(), cache.add_sequence());
        grow(&mut cache, a, 2);
        grow(&mut cache, b, 1);
        grow(&mut cache, a, 2);
        // Pages données dans l'ordre : 0 à `a`, 1 à `b`, puis 2 à `a`.
        assert_eq!(lines(&cache, a), [0, 1, 2, 6]);
        assert_eq!(lines(&cache, b), [3]);
        assert_eq!(cache.free_pages(), 1);
        assert_eq!(cache.pages_needed(a, 2), 0);
        assert_eq!(cache.pages_needed(a, 3), 1);
        // Une séquence terminée rend ses pages ; la suivante les reprend.
        cache.remove_sequence(a);
        assert_eq!(cache.free_pages(), 3);
        let c = cache.add_sequence();
        assert_eq!(c, a, "le numéro d'une séquence terminée resert");
        grow(&mut cache, c, 1);
        assert_eq!(lines(&cache, c), [0]);
    }

    #[test]
    fn forgetting_and_truncating_free_whole_pages() {
        let mut cache = KvCache::<B>::new(1, 4, 3);
        let id = cache.add_sequence();
        grow(&mut cache, id, 7);
        assert_eq!(lines(&cache, id), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(cache.free_pages(), 1);
        // Oublier quatre positions libère la première page seulement.
        cache.forget(id, 4);
        assert_eq!(lines(&cache, id), [4, 5, 6]);
        assert_eq!((cache.len(id), cache.next_position(id)), (3, 7));
        assert_eq!(cache.free_pages(), 2);
        // Revenir en arrière recule aussi la prochaine position, et rend
        // la page devenue vide.
        cache.truncate(id, 2);
        assert_eq!(lines(&cache, id), [4, 5]);
        assert_eq!((cache.len(id), cache.next_position(id)), (2, 6));
        assert_eq!(cache.free_pages(), 3);
    }

    #[test]
    fn a_pass_describes_its_tokens_for_the_attention() {
        let backend = Cpu::new();
        let mut cache = KvCache::<B>::new(1, 4, 2);
        let (a, b) = (cache.add_sequence(), cache.add_sequence());
        grow(&mut cache, a, 3);
        let tape = Tape::new(&backend);
        let (step, positions) = cache.begin(&tape, &[(b, 2), (a, 1)], 100);
        assert_eq!(positions, [0, 1, 3]);
        // `a` a les lignes 0 à 3 (pages 0 et 1), `b` la page 2.
        assert_eq!(tape.value(step.lines).data(), [4.0, 5.0, 3.0]);
        assert_eq!(
            tape.value(step.slots).data(),
            [4.0, 5.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(tape.value(step.rows).data(), [0.0, 1.0, 0.0, 2.0, 1.0, 4.0]);
    }

    #[test]
    fn a_long_sequence_forgets_what_the_context_no_longer_sees() {
        let backend = Cpu::new();
        let mut cache = KvCache::<B>::new(1, 4, 2);
        let id = cache.add_sequence();
        let tape = Tape::new(&backend);
        cache.begin(&tape, &[(id, 4)], 5);
        let (_, positions) = cache.begin(&tape, &[(id, 3)], 5);
        assert_eq!(positions, [4, 5, 6]);
        assert_eq!((cache.len(id), cache.next_position(id)), (5, 7));
    }

    #[test]
    #[should_panic(expected = "cache plein")]
    fn a_full_cache_refuses_new_positions() {
        let mut cache = KvCache::<B>::new(1, 2, 2);
        let id = cache.add_sequence();
        grow(&mut cache, id, 5);
    }

    #[test]
    #[should_panic(expected = "une séquence ne peut apparaître qu'une fois")]
    fn a_sequence_appears_once_per_pass() {
        let backend = Cpu::new();
        let mut cache = KvCache::<B>::new(1, 2, 2);
        let id = cache.add_sequence();
        cache.begin(&Tape::new(&backend), &[(id, 1), (id, 1)], 10);
    }
}
