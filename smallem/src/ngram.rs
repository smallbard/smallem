//! Les n-grammes d'une séquence de tokens, et leur hachage vers les lignes
//! d'une table : ce que la mémoire de n-grammes ([`crate::nn::Memory`]) lit
//! à chaque position.
//!
//! À la position `t`, le n-gramme d'ordre `n` est fait des `n` derniers
//! tokens, `(x_{t−n+1}, …, x_t)`. Il ne franchit pas une fin de document :
//! les tokens d'un document précédent sont remplacés par le token de fin de
//! document, comme le sont ceux d'avant le début de la séquence. Le token de
//! fin de document lui-même est gardé : il dit qu'un document commence.
//!
//! Chaque ordre a plusieurs **têtes**, et chaque tête sa propre table, d'un
//! nombre premier de lignes, et ses propres multiplicateurs (Engram, Cheng
//! et al., 2026) :
//!
//! ```text
//! ligne = ((x_t · m₀) xor (x_{t−1} · m₁) xor …) mod P
//! ```
//!
//! Deux n-grammes peuvent tomber sur la même ligne d'une table. Avec
//! plusieurs têtes, ils ne partagent toutes leurs lignes que si toutes les
//! têtes se trompent ensemble.

use crate::tokenizer::END_OF_DOCUMENT;

/// Le hachage des n-grammes d'ordre 2 à `max_order`, avec `heads` têtes par
/// ordre. Les tables sont numérotées ordre par ordre : celles des bigrammes
/// d'abord, puis celles des trigrammes.
#[derive(Clone, Debug, PartialEq)]
pub struct NgramHash {
    max_order: usize,
    heads: usize,
    /// Nombre de lignes de chaque table : des nombres premiers, tous
    /// différents.
    rows: Vec<usize>,
    /// Les multiplicateurs de chaque table, un par token du n-gramme, le
    /// plus récent d'abord. Impairs, donc inversibles modulo 2⁶⁴ : deux
    /// tokens différents ne donnent jamais le même produit.
    multipliers: Vec<Vec<u64>>,
}

impl NgramHash {
    /// `rows` est la taille visée de chaque table : la première a le plus
    /// petit nombre premier au-dessus, la suivante le nombre premier
    /// d'après, et ainsi de suite. Les multiplicateurs se tirent de `seed`.
    pub fn new(max_order: usize, heads: usize, rows: usize, seed: u64) -> Self {
        assert!(max_order >= 2, "il faut au moins des bigrammes");
        assert!(heads >= 1, "il faut au moins une tête");
        assert!(rows >= 2, "une table d'au moins deux lignes");
        let tables = (max_order - 1) * heads;
        let mut primes = Vec::with_capacity(tables);
        let mut candidate = rows;
        for _ in 0..tables {
            candidate = next_prime(candidate);
            primes.push(candidate);
            candidate += 1;
        }
        let mut state = seed;
        let multipliers = (0..tables)
            .map(|table| {
                let order = table / heads + 2;
                (0..order).map(|_| splitmix64(&mut state) | 1).collect()
            })
            .collect();
        Self {
            max_order,
            heads,
            rows: primes,
            multipliers,
        }
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Nombre de tables : `(max_order − 1) × heads`.
    pub fn tables(&self) -> usize {
        self.rows.len()
    }

    /// L'ordre des n-grammes que lit la table `table`.
    pub fn order(&self, table: usize) -> usize {
        table / self.heads + 2
    }

    /// Nombre de lignes de la table `table`.
    pub fn rows(&self, table: usize) -> usize {
        self.rows[table]
    }

    /// Nombre total de lignes, toutes tables comprises.
    pub fn total_rows(&self) -> usize {
        self.rows.iter().sum()
    }

    /// La ligne de la table `table` pour un n-gramme donné du plus récent
    /// au plus ancien token. Seuls les `order(table)` premiers comptent.
    pub fn hash(&self, table: usize, recent_first: &[u32]) -> u32 {
        let multipliers = &self.multipliers[table];
        let mut mix = 0u64;
        for (&token, &multiplier) in recent_first.iter().zip(multipliers) {
            mix ^= u64::from(token).wrapping_mul(multiplier);
        }
        (mix % self.rows[table] as u64) as u32
    }

    /// Les lignes lues à chaque position de `batch` séquences de même
    /// longueur, rangées l'une après l'autre dans `tokens`. Renvoie
    /// `[positions, tables]` : pour chaque position, une ligne par table,
    /// chacune comptée dans sa propre table.
    pub fn indices(&self, tokens: &[u32], batch: usize) -> Vec<u32> {
        assert!(
            batch > 0 && tokens.len().is_multiple_of(batch),
            "{} tokens ne forment pas {batch} séquences de même longueur",
            tokens.len(),
        );
        let length = tokens.len() / batch;
        let mut out = Vec::with_capacity(tokens.len() * self.tables());
        let mut gram = vec![0u32; self.max_order];
        for sequence in tokens.chunks(length) {
            for t in 0..sequence.len() {
                suffix(sequence, t, &mut gram);
                for table in 0..self.tables() {
                    out.push(self.hash(table, &gram));
                }
            }
        }
        out
    }

    /// Les lignes lues par une seule position, dont `recent_first` donne le
    /// token puis ceux qui le précèdent, déjà coupés à la fin de document :
    /// le calcul de [`NgramHash::indices`] pour la génération, où le passé
    /// est dans le cache.
    pub fn position_indices(&self, recent_first: &[u32]) -> Vec<u32> {
        (0..self.tables())
            .map(|table| self.hash(table, recent_first))
            .collect()
    }
}

/// Le n-gramme suffixe le plus long qui finit à la position `t` de
/// `sequence`, du plus récent au plus ancien token, dans `gram` : les tokens
/// d'avant la séquence, ou d'avant une fin de document, y deviennent des fins
/// de document.
pub fn suffix(sequence: &[u32], t: usize, gram: &mut [u32]) {
    let end = u32::from(END_OF_DOCUMENT);
    let mut cut = false;
    for (back, slot) in gram.iter_mut().enumerate() {
        *slot = if cut || back > t {
            end
        } else {
            sequence[t - back]
        };
        // Le token de fin de document est gardé ; ce qui le précède ne
        // l'est pas.
        if back > 0 && *slot == end {
            cut = true;
        }
    }
}

/// La clé d'un trigramme donné du plus récent au plus ancien token :
/// `(x_{t−2} · V + x_{t−1}) · V + x_t`, pour un vocabulaire de `V` tokens.
pub fn trigram_key(recent_first: &[u32], vocab: usize) -> u64 {
    let v = vocab as u64;
    (u64::from(recent_first[2]) * v + u64::from(recent_first[1])) * v + u64::from(recent_first[0])
}

/// Les trigrammes distincts de `tokens`, tels que les lit la mémoire (voir
/// [`suffix`]), et leurs comptes, triés par [`trigram_key`]. Les clés sont
/// d'abord réparties selon leur token le plus ancien, puis chaque part est
/// triée sur son propre fil.
pub fn trigram_counts(tokens: &[u32], vocab: usize) -> Vec<(u64, u32)> {
    let v = vocab as u64;
    let key = |t: usize| -> u64 {
        let mut gram = [0u32; 3];
        suffix(tokens, t, &mut gram);
        trigram_key(&gram, vocab)
    };
    let mut sizes = vec![0usize; vocab];
    for t in 0..tokens.len() {
        sizes[(key(t) / (v * v)) as usize] += 1;
    }
    let mut next = Vec::with_capacity(vocab);
    let mut start = 0;
    for size in &sizes {
        next.push(start);
        start += size;
    }
    let mut keys = vec![0u64; tokens.len()];
    for t in 0..tokens.len() {
        let k = key(t);
        let bucket = (k / (v * v)) as usize;
        keys[next[bucket]] = k;
        next[bucket] += 1;
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let mut parts: Vec<&mut [u64]> = Vec::with_capacity(vocab);
    let mut rest = keys.as_mut_slice();
    for size in &sizes {
        let (part, tail) = rest.split_at_mut(*size);
        parts.push(part);
        rest = tail;
    }
    let per_thread = parts.len().div_ceil(threads);
    let counted: Vec<Vec<(u64, u32)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .chunks_mut(per_thread)
            .map(|group| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for part in group.iter_mut() {
                        part.sort_unstable();
                        for run in part.chunk_by(|a, b| a == b) {
                            out.push((run[0], run.len() as u32));
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    counted.into_iter().flatten().collect()
}

/// Le compte de la clé `key` dans une liste triée de [`trigram_counts`], ou
/// 0 si elle n'y est pas.
pub fn count_of(sorted: &[(u64, u32)], key: u64) -> u32 {
    sorted
        .binary_search_by_key(&key, |(k, _)| *k)
        .map_or(0, |i| sorted[i].1)
}

/// Le plus petit nombre premier supérieur ou égal à `n`.
pub fn next_prime(n: usize) -> usize {
    let mut candidate = n.max(2);
    while !is_prime(candidate) {
        candidate += 1;
    }
    candidate
}

fn is_prime(n: usize) -> bool {
    if n < 4 {
        return n >= 2;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let mut divisor = 3;
    while divisor * divisor <= n {
        if n.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// Le générateur SplitMix64 : rapide, et assez mélangé pour tirer des
/// multiplicateurs.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const END: u32 = END_OF_DOCUMENT as u32;

    #[test]
    fn primes_follow_each_other() {
        assert_eq!(next_prime(0), 2);
        assert_eq!(next_prime(14), 17);
        assert_eq!(next_prime(17), 17);
        let hash = NgramHash::new(3, 2, 100, 1);
        assert_eq!(hash.tables(), 4);
        assert_eq!(
            (0..4).map(|t| hash.rows(t)).collect::<Vec<_>>(),
            [101, 103, 107, 109]
        );
        assert_eq!(hash.total_rows(), 420);
        assert_eq!(
            (0..4).map(|t| hash.order(t)).collect::<Vec<_>>(),
            [2, 2, 3, 3]
        );
    }

    #[test]
    fn the_suffix_stops_at_the_start_and_at_a_document_end() {
        let mut gram = [0; 3];
        let sequence = [5, 6, END, 7, 8];
        suffix(&sequence, 0, &mut gram);
        assert_eq!(gram, [5, END, END]);
        suffix(&sequence, 1, &mut gram);
        assert_eq!(gram, [6, 5, END]);
        // La fin de document est gardée, ce qui la précède ne l'est pas.
        suffix(&sequence, 3, &mut gram);
        assert_eq!(gram, [7, END, END]);
        suffix(&sequence, 4, &mut gram);
        assert_eq!(gram, [8, 7, END]);
        // Le token de fin lui-même voit le document qu'il termine.
        suffix(&sequence, 2, &mut gram);
        assert_eq!(gram, [END, 6, 5]);
    }

    #[test]
    fn indices_depend_only_on_the_suffix() {
        let hash = NgramHash::new(3, 4, 1000, 7);
        let tables = hash.tables();
        // Deux séquences : la même fin `9, 3, 4` dans des contextes
        // différents donne les mêmes lignes de trigrammes.
        let tokens = [1, 9, 3, 4, 2, 9, 3, 4];
        let indices = hash.indices(&tokens, 2);
        assert_eq!(indices.len(), tokens.len() * tables);
        let at = |position: usize| &indices[position * tables..(position + 1) * tables];
        assert_eq!(at(3), at(7));
        // `1, 9` et `2, 9` : mêmes unigrammes, bigrammes différents.
        assert_ne!(at(1)[..4], at(5)[..4]);
        for (position, row) in indices.chunks(tables).enumerate() {
            for (table, &index) in row.iter().enumerate() {
                assert!((index as usize) < hash.rows(table), "position {position}");
            }
        }
        // La génération, position par position, lit les mêmes lignes.
        assert_eq!(hash.position_indices(&[4, 3, 9]), at(3));
    }

    #[test]
    fn trigrams_are_counted_as_the_memory_reads_them() {
        let tokens = [1, 2, 3, 1, 2, 3, END, 1, 2];
        let counts = trigram_counts(&tokens, 300);
        let key = |a: u32, b: u32, c: u32| trigram_key(&[c, b, a], 300);
        assert_eq!(count_of(&counts, key(1, 2, 3)), 2);
        // Après la fin de document, l'ancien document ne compte plus : le
        // début du texte et le début du second document se ressemblent.
        assert_eq!(count_of(&counts, key(END, END, 1)), 2);
        assert_eq!(count_of(&counts, key(END, 1, 2)), 2);
        assert_eq!(count_of(&counts, key(3, END, 1)), 0);
        let total: u32 = counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total as usize, tokens.len());
        assert!(counts.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn heads_disagree() {
        let hash = NgramHash::new(2, 8, 1 << 16, 3);
        // Sur 20 000 bigrammes, les 8 têtes d'un même bigramme ne tombent
        // presque jamais toutes sur les lignes d'un même autre bigramme.
        let grams: Vec<[u32; 2]> = (0..20_000u32).map(|i| [i % 8192, i / 8192 + 300]).collect();
        let rows = |g: &[u32; 2]| -> Vec<u32> { (0..8).map(|t| hash.hash(t, g)).collect() };
        let mut all: Vec<Vec<u32>> = grams.iter().map(rows).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), grams.len());
    }
}
