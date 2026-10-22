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
/// [`suffix`]), et leurs comptes, triés par [`trigram_key`] : comme
/// [`trigram_counts_of`] pour une seule séquence.
pub fn trigram_counts(tokens: &[u32], vocab: usize) -> Vec<(u64, u32)> {
    trigram_counts_of(&[tokens], vocab)
}

/// Les trigrammes distincts de plusieurs séquences, chacune lue à part : un
/// trigramme ne va jamais d'une séquence à la suivante. Les clés sont
/// d'abord réparties selon leur token le plus ancien, puis chaque part est
/// triée sur son propre fil.
pub fn trigram_counts_of(sequences: &[&[u32]], vocab: usize) -> Vec<(u64, u32)> {
    let v = vocab as u64;
    fn keys_of(sequence: &[u32], vocab: usize) -> impl Iterator<Item = u64> + '_ {
        let mut gram = [0u32; 3];
        (0..sequence.len()).map(move |t| {
            suffix(sequence, t, &mut gram);
            trigram_key(&gram, vocab)
        })
    }
    let mut sizes = vec![0usize; vocab];
    for sequence in sequences {
        for k in keys_of(sequence, vocab) {
            sizes[(k / (v * v)) as usize] += 1;
        }
    }
    let mut next = Vec::with_capacity(vocab);
    let mut start = 0;
    for size in &sizes {
        next.push(start);
        start += size;
    }
    let mut keys = vec![0u64; start];
    for sequence in sequences {
        for k in keys_of(sequence, vocab) {
            let bucket = (k / (v * v)) as usize;
            keys[next[bucket]] = k;
            next[bucket] += 1;
        }
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
    let counted = on_threads(parts.chunks_mut(per_thread), |group| {
        let mut out = Vec::new();
        for part in group.iter_mut() {
            part.sort_unstable();
            for run in part.chunk_by(|a, b| a == b) {
                out.push((run[0], run.len() as u32));
            }
        }
        out
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
fn next_prime(n: usize) -> usize {
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

/// Applique `work` à chaque élément de `items`, chacun sur son propre fil,
/// et rend les résultats dans l'ordre. La panique d'un fil se propage.
fn on_threads<T: Send, R: Send>(
    items: impl IntoIterator<Item = T>,
    work: impl Fn(T) -> R + Sync,
) -> Vec<R> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .into_iter()
            .map(|item| {
                let work = &work;
                scope.spawn(move || work(item))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|p| std::panic::resume_unwind(p))
            })
            .collect()
    })
}

/// Les comptes d'unigrammes, de bigrammes et de trigrammes d'un corpus fait
/// de plusieurs séquences, tels que les lit la mémoire (voir [`suffix`]) :
/// de quoi faire les modèles par comptage qui disent ce qu'une table sait à
/// elle seule.
///
/// Chaque probabilité conditionnelle divise un compte par celui de son
/// contexte **compté de la même façon** : la somme, sur le token prédit, des
/// n-grammes qui commencent par ce contexte. Les conditionnelles d'un
/// contexte vu somment ainsi à 1, même quand le contexte contient des fins
/// de document ajoutées par [`suffix`].
pub struct NgramCounts {
    vocab: usize,
    tokens: u64,
    unigrams: Vec<u64>,
    /// `V × V`, rangé par contexte : `bigrams[b · V + c]`.
    bigrams: Vec<u32>,
    /// La somme de chaque ligne de `bigrams`.
    bigram_contexts: Vec<u64>,
    /// Triés par [`trigram_key`].
    trigrams: Vec<(u64, u32)>,
    /// Pour chaque contexte `a · V + b` vu, la somme de ses trigrammes ;
    /// triés.
    trigram_contexts: Vec<(u64, u64)>,
}

impl NgramCounts {
    pub fn new(sequences: &[&[u32]], vocab: usize) -> Self {
        let v = vocab as u64;
        let mut unigrams = vec![0u64; vocab];
        let mut bigrams = vec![0u32; vocab * vocab];
        let mut gram = [0u32; 2];
        let mut tokens = 0u64;
        for sequence in sequences {
            tokens += sequence.len() as u64;
            for t in 0..sequence.len() {
                unigrams[sequence[t] as usize] += 1;
                suffix(sequence, t, &mut gram);
                bigrams[gram[1] as usize * vocab + gram[0] as usize] += 1;
            }
        }
        let bigram_contexts = bigrams
            .chunks(vocab)
            .map(|row| row.iter().map(|&c| u64::from(c)).sum())
            .collect();
        let trigrams = trigram_counts_of(sequences, vocab);
        let mut trigram_contexts: Vec<(u64, u64)> = Vec::new();
        for &(key, count) in &trigrams {
            let context = key / v;
            match trigram_contexts.last_mut() {
                Some((last, total)) if *last == context => *total += u64::from(count),
                _ => trigram_contexts.push((context, u64::from(count))),
            }
        }
        Self {
            vocab,
            tokens,
            unigrams,
            bigrams,
            bigram_contexts,
            trigrams,
            trigram_contexts,
        }
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Les bigrammes vus et leurs comptes, triés par clé `b · V + c`.
    pub fn bigrams(&self) -> Vec<(u64, u32)> {
        self.bigrams
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(k, c)| (k as u64, *c))
            .collect()
    }

    /// Les trigrammes vus et leurs comptes, triés par [`trigram_key`].
    pub fn trigrams(&self) -> &[(u64, u32)] {
        &self.trigrams
    }

    /// Les probabilités que donnent au token `x_t` de `sequence` les modèles
    /// uniforme, unigramme, bigramme et trigramme. Un contexte jamais vu
    /// donne une probabilité nulle : l'interpolation s'en remet alors aux
    /// autres modèles.
    pub fn components(&self, sequence: &[u32], t: usize) -> [f64; 4] {
        let v = self.vocab as u64;
        let mut gram = [0u32; 3];
        suffix(sequence, t, &mut gram);
        let (c, b, a) = (u64::from(gram[0]), u64::from(gram[1]), u64::from(gram[2]));
        let ratio = |count: f64, context: u64| {
            if context > 0 {
                count / context as f64
            } else {
                0.0
            }
        };
        let bigram = f64::from(self.bigrams[(b * v + c) as usize]);
        let trigram = f64::from(count_of(&self.trigrams, (a * v + b) * v + c));
        let pair = self
            .trigram_contexts
            .binary_search_by_key(&(a * v + b), |(k, _)| *k)
            .map_or(0, |i| self.trigram_contexts[i].1);
        [
            1.0 / v as f64,
            self.unigrams[c as usize] as f64 / self.tokens as f64,
            ratio(bigram, self.bigram_contexts[b as usize]),
            ratio(trigram, pair),
        ]
    }
}

/// La part des occurrences couverte par les `top` n-grammes les plus
/// fréquents de `grams`.
pub fn coverage(grams: &[(u64, u32)], top: usize) -> f64 {
    let mut counts: Vec<u32> = grams.iter().map(|(_, c)| *c).collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    let kept: u64 = counts.iter().take(top).map(|&c| u64::from(c)).sum();
    kept as f64 / total as f64
}

/// La part des occurrences dont le n-gramme est le plus fréquent de sa
/// ligne : dans la première tête de son ordre, puis dans au moins une de
/// ses têtes. Une ligne partagée sert surtout son n-gramme le plus
/// fréquent. `grams` sont des n-grammes d'ordre `order` (2 ou 3), donnés
/// par leur clé pour un vocabulaire de `vocab` tokens, comme
/// [`trigram_key`] et [`NgramCounts::bigrams`]. Une tête par fil.
pub fn collisions(
    hash: &NgramHash,
    grams: &[(u64, u32)],
    order: usize,
    vocab: usize,
) -> (f64, f64) {
    assert!(
        (2..=hash.max_order()).contains(&order),
        "ordre {order} hors du hachage (2 à {})",
        hash.max_order()
    );
    let v = vocab as u64;
    let heads = hash.heads();
    // Le n-gramme d'une clé, du plus récent au plus ancien token.
    let tokens = |mut key: u64| -> [u32; 3] {
        let mut out = [0u32; 3];
        for slot in out.iter_mut().take(order) {
            *slot = (key % v) as u32;
            key /= v;
        }
        out
    };
    let owned: Vec<Vec<bool>> = on_threads(0..heads, |head| {
        let table = (order - 2) * heads + head;
        let row = |k: u64| hash.hash(table, &tokens(k)) as usize;
        let mut most = vec![0u32; hash.rows(table)];
        for (k, count) in grams {
            let slot = &mut most[row(*k)];
            *slot = (*slot).max(*count);
        }
        grams
            .iter()
            .map(|(k, count)| most[row(*k)] == *count)
            .collect()
    });
    let total: u64 = grams.iter().map(|(_, c)| u64::from(*c)).sum();
    let (mut one, mut all) = (0u64, 0u64);
    for (i, (_, count)) in grams.iter().enumerate() {
        if owned[0][i] {
            one += u64::from(*count);
        }
        if owned.iter().any(|head| head[i]) {
            all += u64::from(*count);
        }
    }
    (one as f64 / total as f64, all as f64 / total as f64)
}

/// Les poids d'interpolation des `used` premières composantes, ajustés par
/// l'algorithme EM : chaque poids devient la part moyenne de la probabilité
/// qu'apporte sa composante.
pub fn interpolation(components: &[[f64; 4]], used: usize) -> Vec<f64> {
    let mut weights = vec![1.0 / used as f64; used];
    for _ in 0..50 {
        let mut shares = vec![0.0; used];
        for row in components {
            let mixed: f64 = (0..used).map(|i| weights[i] * row[i]).sum();
            for i in 0..used {
                shares[i] += weights[i] * row[i] / mixed;
            }
        }
        for (weight, share) in weights.iter_mut().zip(shares) {
            *weight = share / components.len() as f64;
        }
    }
    weights
}

/// La perte moyenne, en nats par token, du mélange des composantes par
/// `weights` (les premières seulement si `weights` est plus court).
pub fn cross_entropy(components: &[[f64; 4]], weights: &[f64]) -> f64 {
    let total: f64 = components
        .iter()
        .map(|row| {
            -weights
                .iter()
                .zip(row)
                .map(|(w, p)| w * p)
                .sum::<f64>()
                .ln()
        })
        .sum();
    total / components.len() as f64
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

    /// Un petit corpus, coupé en deux séquences, avec des fins de document.
    fn corpus() -> Vec<Vec<u32>> {
        vec![
            vec![3, 4, 5, 3, 4, 6, END, 3, 4, 5, END, END, 7],
            vec![4, 5, 3, END, 6, 6, 6, 3],
        ]
    }

    #[test]
    fn conditionals_sum_to_one() {
        let corpus = corpus();
        let sequences: Vec<&[u32]> = corpus.iter().map(Vec::as_slice).collect();
        let vocab = 300;
        let counts = NgramCounts::new(&sequences, vocab);
        // Pour chaque position, la somme sur le token prédit de chaque
        // modèle vaut 1 : on remplace le dernier token par chacun des
        // tokens du vocabulaire.
        for sequence in &corpus {
            for t in 0..sequence.len() {
                let mut sums = [0.0; 4];
                let mut changed = sequence.clone();
                for c in 0..vocab as u32 {
                    changed[t] = c;
                    for (sum, p) in sums.iter_mut().zip(counts.components(&changed, t)) {
                        *sum += p;
                    }
                }
                for (model, sum) in sums.iter().enumerate() {
                    assert!((sum - 1.0).abs() < 1e-12, "modèle {model}, t = {t} : {sum}");
                }
            }
        }
    }

    #[test]
    fn ngrams_do_not_cross_from_one_sequence_to_the_next() {
        let joined: Vec<u32> = corpus().concat();
        let corpus = corpus();
        let apart: Vec<&[u32]> = corpus.iter().map(Vec::as_slice).collect();
        let key = |a: u32, b: u32, c: u32| trigram_key(&[c, b, a], 300);
        // `6, 7 | 4` n'existe que si l'on colle les deux séquences.
        assert_eq!(count_of(&trigram_counts(&joined, 300), key(END, 7, 4)), 1);
        let counts = trigram_counts_of(&apart, 300);
        assert_eq!(count_of(&counts, key(END, 7, 4)), 0);
        // Le début de la seconde séquence compte comme un début de document.
        assert_eq!(count_of(&counts, key(END, END, 4)), 1);
        let total: u32 = counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total as usize, joined.len());
    }

    #[test]
    fn interpolation_prefers_the_component_that_explains_the_data() {
        // La deuxième composante donne toujours plus de probabilité : son
        // poids grandit, et la perte du mélange baisse.
        let rows = vec![[0.1, 0.5, 0.0, 0.0]; 20];
        let weights = interpolation(&rows, 2);
        assert!(weights[1] > 0.99, "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(cross_entropy(&rows, &weights) < cross_entropy(&rows, &[0.5, 0.5]));
        assert!((cross_entropy(&rows, &[0.0, 1.0]) - 0.5f64.ln().abs()).abs() < 1e-12);
    }

    #[test]
    fn coverage_counts_the_most_frequent_first() {
        let grams = [(0, 1), (1, 6), (2, 3)];
        assert_eq!(coverage(&grams, 1), 0.6);
        assert_eq!(coverage(&grams, 3), 1.0);
    }

    #[test]
    fn collisions_vanish_with_large_tables() {
        let corpus = corpus();
        let sequences: Vec<&[u32]> = corpus.iter().map(Vec::as_slice).collect();
        let counts = NgramCounts::new(&sequences, 300);
        let hash = NgramHash::new(3, 4, 1 << 16, 1);
        assert_eq!(collisions(&hash, &counts.bigrams(), 2, 300), (1.0, 1.0));
        assert_eq!(collisions(&hash, counts.trigrams(), 3, 300), (1.0, 1.0));
        // Une table de deux lignes, elle, mélange tout.
        let tiny = NgramHash::new(3, 1, 2, 1);
        let (one, all) = collisions(&tiny, counts.trigrams(), 3, 300);
        assert!(one < 1.0 && one == all);
    }
}
