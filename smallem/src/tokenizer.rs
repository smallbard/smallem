//! Tokeniseurs : d'abord au niveau octet (un token par octet UTF-8, plus un
//! token spécial qui sépare les documents), puis BPE, qui part des octets et
//! apprend à regrouper les suites fréquentes en sous-mots.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;

/// Token de fin de document. Il n'est jamais produit par [`encode`] : c'est
/// l'appelant qui l'insère entre deux documents.
pub const END_OF_DOCUMENT: u16 = 256;

/// 256 octets + le token de fin de document.
pub const VOCAB_SIZE: usize = 257;

/// Le nom du fichier de fusions BPE, à côté des tokens qu'il a produits.
pub const TOKENIZER_FILE: &str = "tokenizer.bpe";

pub fn encode(text: &str) -> Vec<u16> {
    text.bytes().map(u16::from).collect()
}

/// Les tokens spéciaux sont ignorés. Une séquence d'octets qui n'est pas de
/// l'UTF-8 valide (un modèle en produira) est remplacée par `U+FFFD` au lieu
/// de provoquer une erreur.
pub fn decode(tokens: &[u16]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .filter_map(|&t| u8::try_from(t).ok())
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------------------
// BPE (Byte Pair Encoding)
// ---------------------------------------------------------------------------

/// Premier identifiant attribué à une fusion : les octets occupent 0 à 255,
/// la fin de document 256.
pub const FIRST_MERGE: u16 = 257;

/// Un mot fait au plus ce nombre de caractères ; au-delà, il est coupé.
/// Cela borne le coût d'encodage d'une suite sans espace (une URL, une
/// clé encodée, une ligne de tirets).
const MAX_WORD_CHARS: usize = 64;

/// Nombre de mots distincts mémorisés par un [`Encoder`]. Les mots courants
/// arrivent tôt : passé ce seuil, les mots rares sont simplement recalculés.
const CACHE_CAPACITY: usize = 1 << 20;

/// Première ligne d'un fichier de fusions.
const FILE_HEADER: &str = "smallem-bpe 1";

/// Le numéro de la ligne de la première fusion, après l'en-tête.
const FIRST_LINE: usize = 2;

/// Une paire de tokens voisins.
type Pair = (u16, u16);

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Letter,
    Digit,
    Space,
    Other,
}

fn class(c: char) -> CharClass {
    if c.is_alphabetic() {
        CharClass::Letter
    } else if c.is_numeric() {
        CharClass::Digit
    } else if c.is_whitespace() {
        CharClass::Space
    } else {
        CharClass::Other
    }
}

/// Longueur en octets du plus long préfixe de `text` fait de caractères de
/// la classe `wanted`, limité à `max_chars` caractères.
fn run_len(text: &str, wanted: CharClass, max_chars: usize) -> usize {
    text.char_indices()
        .take(max_chars)
        .find(|&(_, c)| class(c) != wanted)
        .map_or_else(
            || {
                text.char_indices()
                    .nth(max_chars)
                    .map_or(text.len(), |(index, _)| index)
            },
            |(index, _)| index,
        )
}

/// Longueur en octets du premier mot de `text`, qui ne doit pas être vide.
fn first_word_len(text: &str) -> usize {
    let first = text.chars().next().expect("texte vide");
    if class(first) != CharClass::Space {
        return run_len(text, class(first), MAX_WORD_CHARS);
    }
    let spaces = run_len(text, CharClass::Space, MAX_WORD_CHARS);
    let Some(next) = text[spaces..].chars().next() else {
        return spaces;
    };
    if class(next) == CharClass::Space || !text[..spaces].ends_with(' ') {
        // Le blanc est suivi d'autres blancs (suite trop longue, coupée), ou
        // il ne finit pas par une espace ordinaire : il forme un mot seul.
        return spaces;
    }
    if spaces > 1 {
        // La dernière espace appartient au mot suivant.
        return spaces - 1;
    }
    1 + run_len(&text[1..], class(next), MAX_WORD_CHARS - 1)
}

/// Découpe un texte en mots, sans rien perdre : les mots mis bout à bout
/// redonnent le texte.
///
/// Un mot est une suite de lettres, une suite de chiffres ou une suite
/// d'autres signes, précédée de l'espace qui l'introduit s'il y en a une :
/// `"Hello, world 42"` donne `"Hello"`, `","`, `" world"`, `" 42"`. Les
/// autres blancs (retours à la ligne, espaces multiples) forment des mots à
/// part.
fn split_words(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let (word, tail) = rest.split_at(first_word_len(rest));
        rest = tail;
        Some(word)
    })
}

/// Ajoute les mots de `text` à un compteur de mots : c'est la seule chose
/// que l'entraînement du BPE a besoin de savoir d'un corpus.
pub fn count_words(text: &str, counts: &mut HashMap<Vec<u8>, u64>) {
    for word in split_words(text) {
        // `get_mut` évite d'allouer une clé pour un mot déjà connu, ce qui
        // est le cas de presque tous.
        match counts.get_mut(word.as_bytes()) {
            Some(count) => *count += 1,
            None => {
                counts.insert(word.as_bytes().to_vec(), 1);
            }
        }
    }
}

/// Un tokeniseur BPE : les 256 octets, le token de fin de document, puis
/// une liste ordonnée de fusions. La fusion numéro `i` remplace une paire de
/// tokens voisins par le token `FIRST_MERGE + i`.
///
/// Sans aucune fusion, c'est exactement le tokeniseur octets.
#[derive(Clone, Debug, PartialEq)]
pub struct Bpe {
    merges: Vec<Pair>,
    /// Pour chaque paire fusionnable, le token qu'elle produit. Comme les
    /// tokens sont numérotés dans l'ordre d'apprentissage, c'est aussi le
    /// rang de la fusion.
    merged: HashMap<Pair, u16>,
    /// Les octets que représente chaque token (aucun pour la fin de
    /// document).
    token_bytes: Vec<Vec<u8>>,
}

impl Bpe {
    /// Le tokeniseur sans fusion : un token par octet.
    pub fn bytes() -> Self {
        Self::from_merges(Vec::new()).expect("une liste vide est valide")
    }

    /// Construit un tokeniseur à partir de ses fusions. Chaque fusion ne
    /// peut assembler que des tokens qui existent déjà au moment où elle
    /// est apprise, et jamais le token de fin de document.
    pub fn from_merges(merges: Vec<Pair>) -> Result<Self, String> {
        Self::checked(merges).map_err(|(index, reason)| match index {
            Some(index) => format!("fusion {index} : {reason}"),
            None => reason,
        })
    }

    /// [`Bpe::from_merges`], dont l'erreur donne à part l'indice de la
    /// fusion fautive : le fichier, lui, parle en lignes.
    fn checked(merges: Vec<Pair>) -> Result<Self, (Option<usize>, String)> {
        if FIRST_MERGE as usize + merges.len() > u16::MAX as usize + 1 {
            return Err((
                None,
                format!("{} fusions ne tiennent pas sur 16 bits", merges.len()),
            ));
        }
        let mut token_bytes: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
        token_bytes.push(Vec::new());
        let mut merged = HashMap::with_capacity(merges.len());
        for (index, &(left, right)) in merges.iter().enumerate() {
            let token = FIRST_MERGE + index as u16;
            for part in [left, right] {
                if part >= token || part == END_OF_DOCUMENT {
                    return Err((Some(index), format!("token {part} invalide")));
                }
            }
            if merged.insert((left, right), token).is_some() {
                return Err((Some(index), "paire déjà fusionnée".into()));
            }
            let mut bytes = token_bytes[left as usize].clone();
            bytes.extend_from_slice(&token_bytes[right as usize]);
            token_bytes.push(bytes);
        }
        Ok(Self {
            merges,
            merged,
            token_bytes,
        })
    }

    /// Apprend des fusions jusqu'à un vocabulaire de `vocab_size` tokens, à
    /// partir du nombre d'occurrences de chaque mot du corpus.
    ///
    /// À chaque tour, la paire de tokens voisins la plus fréquente devient
    /// un nouveau token. Les fusions ne traversent jamais une frontière de
    /// mot, puisque les paires sont comptées à l'intérieur des mots.
    ///
    /// Recompter toutes les paires à chaque tour serait trop lent. Une
    /// fusion ne change que les mots qui contiennent la paire fusionnée :
    /// on tient donc à jour le compte de chaque paire et la liste des mots
    /// où elle apparaît, et on ne retouche que ceux-là.
    pub fn train(word_counts: &HashMap<Vec<u8>, u64>, vocab_size: usize) -> Self {
        assert!(
            (FIRST_MERGE as usize..=u16::MAX as usize + 1).contains(&vocab_size),
            "vocabulaire de {vocab_size} tokens impossible",
        );
        let mut words: Vec<(Vec<u16>, u64)> = word_counts
            .iter()
            .map(|(word, &count)| (word.iter().map(|&byte| u16::from(byte)).collect(), count))
            .collect();

        let mut pair_counts: HashMap<Pair, u64> = HashMap::new();
        // Les mots où une paire est apparue. Un mot peut y rester après
        // avoir perdu la paire : on le vérifie au moment de s'en servir.
        let mut pair_words: HashMap<Pair, Vec<u32>> = HashMap::new();
        for (index, (symbols, count)) in words.iter().enumerate() {
            for pair in symbols.windows(2) {
                let pair = (pair[0], pair[1]);
                *pair_counts.entry(pair).or_default() += count;
                pair_words.entry(pair).or_default().push(index as u32);
            }
        }

        // Tas des paires, la plus fréquente en tête ; à égalité, la plus
        // petite paire l'emporte, pour un résultat déterministe. Le tas
        // n'est jamais corrigé : quand un compte change, on empile une
        // nouvelle entrée, et les entrées périmées sont écartées à la sortie.
        let mut heap: BinaryHeap<(u64, Reverse<Pair>)> = pair_counts
            .iter()
            .map(|(&pair, &count)| (count, Reverse(pair)))
            .collect();

        let mut merges = Vec::new();
        while FIRST_MERGE as usize + merges.len() < vocab_size {
            let Some((count, Reverse(pair))) = heap.pop() else {
                break;
            };
            if pair_counts.get(&pair) != Some(&count) {
                continue;
            }
            let token = FIRST_MERGE + merges.len() as u16;
            merges.push(pair);

            let mut touched = pair_words.remove(&pair).unwrap_or_default();
            touched.sort_unstable();
            touched.dedup();
            let mut changed: HashSet<Pair> = HashSet::new();
            for index in touched {
                let (symbols, word_count) = &mut words[index as usize];
                let replaced = merge_pair(symbols, pair, token);
                if replaced == *symbols {
                    continue;
                }
                // Le mot change : on retire toutes ses anciennes paires des
                // comptes, puis on ajoute les nouvelles.
                for old in symbols.windows(2) {
                    let old = (old[0], old[1]);
                    *pair_counts.get_mut(&old).expect("paire comptée") -= *word_count;
                    changed.insert(old);
                }
                for new in replaced.windows(2) {
                    let new = (new[0], new[1]);
                    *pair_counts.entry(new).or_default() += *word_count;
                    pair_words.entry(new).or_default().push(index);
                    changed.insert(new);
                }
                *symbols = replaced;
            }
            for pair in changed {
                match pair_counts[&pair] {
                    0 => {
                        pair_counts.remove(&pair);
                    }
                    count => heap.push((count, Reverse(pair))),
                }
            }
        }
        Self::from_merges(merges).expect("les fusions apprises sont valides")
    }

    /// Nombre de tokens, fin de document comprise.
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    pub fn merges(&self) -> &[Pair] {
        &self.merges
    }

    /// Les octets que représente un token.
    pub fn token_bytes(&self, token: u16) -> &[u8] {
        &self.token_bytes[token as usize]
    }

    /// Encode un mot : on part de ses octets et on applique les fusions
    /// dans l'ordre où elles ont été apprises. À chaque tour, parmi les
    /// paires présentes, c'est donc la fusion la plus ancienne qui passe.
    fn encode_word(&self, word: &[u8]) -> Vec<u16> {
        let mut symbols: Vec<u16> = word.iter().map(|&byte| u16::from(byte)).collect();
        loop {
            let earliest = symbols
                .windows(2)
                .filter_map(|pair| {
                    let pair = (pair[0], pair[1]);
                    self.merged.get(&pair).map(|&token| (token, pair))
                })
                .min();
            let Some((token, pair)) = earliest else {
                return symbols;
            };
            symbols = merge_pair(&symbols, pair, token);
        }
    }

    /// Un encodeur qui mémorise les mots déjà vus : à utiliser pour encoder
    /// beaucoup de texte.
    pub fn encoder(&self) -> Encoder<'_> {
        Encoder {
            bpe: self,
            cache: HashMap::new(),
        }
    }

    /// Le token de fin de document n'est jamais produit : c'est l'appelant
    /// qui l'insère entre deux documents.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut tokens = Vec::new();
        self.encoder().encode_into(text, &mut tokens);
        tokens
    }

    /// Met bout à bout les octets de chaque token, sans les interpréter.
    pub fn decode_bytes(&self, tokens: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &token in tokens {
            bytes.extend_from_slice(self.token_bytes(token));
        }
        bytes
    }

    /// Comme pour les octets, la fin de document est ignorée et une
    /// séquence qui n'est pas de l'UTF-8 valide donne des `U+FFFD`. Le
    /// décodage UTF-8 se fait sur l'ensemble : un caractère peut être
    /// réparti sur plusieurs tokens.
    pub fn decode(&self, tokens: &[u16]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
    }

    /// Un décodeur qui rend le texte au fur et à mesure que les tokens
    /// arrivent.
    pub fn decoder(&self) -> Decoder<'_> {
        Decoder {
            bpe: self,
            pending: Vec::new(),
        }
    }

    /// Enregistre la table de fusions : une ligne d'en-tête, puis une ligne
    /// `gauche droite` par fusion, dans l'ordre d'apprentissage. Comme les
    /// poids, à côté puis renommée.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut text = format!("{FILE_HEADER}\n");
        for (left, right) in &self.merges {
            text.push_str(&format!("{left} {right}\n"));
        }
        crate::io::write_atomically(path, text.as_bytes())
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let invalid = |message: String| io::Error::new(io::ErrorKind::InvalidData, message);
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        if lines.next() != Some(FILE_HEADER) {
            return Err(invalid("ce n'est pas un fichier de fusions".into()));
        }
        let mut merges = Vec::new();
        for (line_number, line) in (FIRST_LINE..).zip(lines) {
            let parts: Vec<&str> = line.split(' ').collect();
            let (Ok(left), Ok(right), 2) = (
                parts[0].parse(),
                parts.get(1).unwrap_or(&"").parse(),
                parts.len(),
            ) else {
                return Err(invalid(format!("ligne {line_number} illisible : `{line}`")));
            };
            merges.push((left, right));
        }
        Self::checked(merges).map_err(|(index, reason)| match index {
            Some(index) => invalid(format!("ligne {} : {reason}", index + FIRST_LINE)),
            None => invalid(reason),
        })
    }
}

/// Remplace chaque occurrence de `pair` par `token`, de gauche à droite.
fn merge_pair(symbols: &[u16], pair: Pair, token: u16) -> Vec<u16> {
    let mut merged = Vec::with_capacity(symbols.len());
    let mut index = 0;
    while index < symbols.len() {
        if index + 1 < symbols.len() && (symbols[index], symbols[index + 1]) == pair {
            merged.push(token);
            index += 2;
        } else {
            merged.push(symbols[index]);
            index += 1;
        }
    }
    merged
}

/// Décode un texte token par token, pour l'afficher pendant qu'il s'écrit.
///
/// Un caractère peut être réparti sur plusieurs tokens : les octets d'un
/// caractère incomplet sont gardés jusqu'à l'arrivée de la suite. Mis bout
/// à bout, les morceaux rendus sont exactement le texte de [`Bpe::decode`].
pub struct Decoder<'a> {
    bpe: &'a Bpe,
    /// Les octets d'un caractère dont la fin n'est pas encore arrivée.
    pending: Vec<u8>,
}

impl Decoder<'_> {
    /// Le texte que `token` permet d'écrire : parfois rien, quand il ne fait
    /// que commencer un caractère.
    pub fn push(&mut self, token: u16) -> String {
        self.pending.extend_from_slice(self.bpe.token_bytes(token));
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    text.push_str(valid);
                    self.pending.clear();
                    return text;
                }
                Err(error) => {
                    let (valid, rest) = self.pending.split_at(error.valid_up_to());
                    text.push_str(std::str::from_utf8(valid).expect("préfixe valide"));
                    match error.error_len() {
                        // Des octets qui ne commencent aucun caractère : ils
                        // donnent un `U+FFFD`, comme dans `Bpe::decode`.
                        Some(invalid) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            self.pending = rest[invalid..].to_vec();
                        }
                        // Le début d'un caractère : attendre la suite.
                        None => {
                            self.pending = rest.to_vec();
                            return text;
                        }
                    }
                }
            }
        }
    }

    /// Ce qui reste quand le texte s'arrête : un caractère resté incomplet
    /// donne un `U+FFFD`.
    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

/// Encode du texte en se souvenant des mots déjà rencontrés. Dans un texte
/// courant, presque tous les mots ont déjà été vus : l'encodage se réduit
/// alors à une recherche par mot.
pub struct Encoder<'a> {
    bpe: &'a Bpe,
    cache: HashMap<Box<[u8]>, Box<[u16]>>,
}

impl Encoder<'_> {
    /// Ajoute les tokens de `text` à la fin de `tokens`.
    pub fn encode_into(&mut self, text: &str, tokens: &mut Vec<u16>) {
        for word in split_words(text) {
            let word = word.as_bytes();
            if let Some(known) = self.cache.get(word) {
                tokens.extend_from_slice(known);
                continue;
            }
            let encoded = self.bpe.encode_word(word);
            tokens.extend_from_slice(&encoded);
            if self.cache.len() < CACHE_CAPACITY {
                self.cache.insert(word.into(), encoded.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_non_ascii() {
        let text = "Élève, naïve — 日本語 🦀\n";
        assert_eq!(decode(&encode(text)), text);
    }

    #[test]
    fn one_token_per_byte() {
        assert_eq!(encode("é"), vec![0xC3, 0xA9]);
    }

    #[test]
    fn decode_skips_end_of_document() {
        let mut tokens = encode("a");
        tokens.push(END_OF_DOCUMENT);
        tokens.extend(encode("b"));
        assert_eq!(decode(&tokens), "ab");
    }

    #[test]
    fn decode_tolerates_invalid_utf8() {
        // 0xC3 seul est le début d'un caractère sur deux octets, tronqué.
        assert_eq!(decode(&[b'a' as u16, 0xC3]), "a\u{FFFD}");
    }

    // --- BPE ---

    fn counts(text: &str) -> HashMap<Vec<u8>, u64> {
        let mut counts = HashMap::new();
        count_words(text, &mut counts);
        counts
    }

    /// Un texte avec assez de répétitions pour apprendre des fusions.
    fn sample_text() -> String {
        let sentences = [
            "The cat sat on the mat, and the other cat sat on the other mat.\n",
            "L'élève naïve étudie l'été — 日本語 et 🦀 !\n\n",
            "In 1999 and 2024, there were 1999 cats;   in  2025 there are none...\n",
            "\tTabs\tand   spaces ; https://example.com/a_b?c=d&e=f\r\n",
        ];
        (0..40).map(|i| sentences[i % sentences.len()]).collect()
    }

    /// La version la plus simple possible de l'entraînement : tout recompter
    /// à chaque fusion. Elle sert de référence à la version incrémentale.
    fn train_naive(word_counts: &HashMap<Vec<u8>, u64>, vocab_size: usize) -> Vec<Pair> {
        let mut words: Vec<(Vec<u16>, u64)> = word_counts
            .iter()
            .map(|(word, &count)| (word.iter().map(|&b| u16::from(b)).collect(), count))
            .collect();
        let mut merges = Vec::new();
        while FIRST_MERGE as usize + merges.len() < vocab_size {
            let mut pair_counts: HashMap<Pair, u64> = HashMap::new();
            for (symbols, count) in &words {
                for pair in symbols.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_default() += count;
                }
            }
            let Some((&pair, _)) = pair_counts
                .iter()
                .max_by_key(|&(&pair, &count)| (count, Reverse(pair)))
            else {
                break;
            };
            let token = FIRST_MERGE + merges.len() as u16;
            for (symbols, _) in &mut words {
                *symbols = merge_pair(symbols, pair, token);
            }
            merges.push(pair);
        }
        merges
    }

    #[test]
    fn words_keep_their_leading_space() {
        let words: Vec<&str> = split_words("Hello, world 42 times").collect();
        assert_eq!(words, ["Hello", ",", " world", " 42", " times"]);
    }

    #[test]
    fn other_whitespace_forms_its_own_words() {
        let words: Vec<&str> = split_words("a  b\n\nc \n d\t").collect();
        assert_eq!(words, ["a", " ", " b", "\n\n", "c", " \n", " d", "\t"]);
    }

    #[test]
    fn letters_digits_and_signs_are_separated() {
        let words: Vec<&str> = split_words("abc123... été").collect();
        assert_eq!(words, ["abc", "123", "...", " été"]);
    }

    #[test]
    fn words_always_rebuild_the_text() {
        for text in [
            sample_text().as_str(),
            "",
            " ",
            "  ",
            " a",
            "a ",
            "\n ",
            " \n",
        ] {
            assert_eq!(split_words(text).collect::<String>(), text);
        }
    }

    #[test]
    fn long_words_are_cut() {
        let text = "x".repeat(150) + &" ".repeat(150) + "é";
        let lengths: Vec<usize> = split_words(&text).map(|w| w.chars().count()).collect();
        assert_eq!(lengths, [64, 64, 22, 64, 64, 21, 2]);
        assert_eq!(split_words(&text).collect::<String>(), text);
    }

    #[test]
    fn without_merges_bpe_is_the_byte_tokenizer() {
        let bpe = Bpe::bytes();
        let text = sample_text();
        assert_eq!(bpe.vocab_size(), VOCAB_SIZE);
        assert_eq!(bpe.encode(&text), encode(&text));
        assert_eq!(bpe.decode(&bpe.encode(&text)), text);
    }

    #[test]
    fn first_merge_is_the_most_frequent_pair() {
        // « ab » apparaît 3 fois, « bc » 2 fois.
        let bpe = Bpe::train(&counts("ab ab abc bc"), 258);
        assert_eq!(bpe.merges(), &[(u16::from(b'a'), u16::from(b'b'))]);
        assert_eq!(bpe.token_bytes(257), b"ab");
        assert_eq!(bpe.encode("abc"), [257, u16::from(b'c')]);
    }

    #[test]
    fn merges_never_cross_a_word_boundary() {
        // « a b » revient souvent, mais « ␣a » et « ␣b » sont deux mots.
        let bpe = Bpe::train(&counts(&"a b ".repeat(50)), 300);
        for token in FIRST_MERGE..bpe.vocab_size() as u16 {
            let bytes = bpe.token_bytes(token);
            assert!(
                !bytes[1..].contains(&b' '),
                "le token {token} ({bytes:?}) contient une espace intérieure",
            );
        }
        // Seules fusions possibles : « ␣b » et « ␣a », à l'intérieur des mots.
        let (space, a, b) = (u16::from(b' '), u16::from(b'a'), u16::from(b'b'));
        assert_eq!(bpe.merges(), &[(space, b), (space, a)]);
    }

    #[test]
    fn incremental_training_matches_naive_training() {
        let word_counts = counts(&sample_text());
        for vocab_size in [257, 258, 300, 400, 2000] {
            let bpe = Bpe::train(&word_counts, vocab_size);
            assert_eq!(bpe.merges(), train_naive(&word_counts, vocab_size));
        }
    }

    #[test]
    fn training_stops_when_nothing_is_left_to_merge() {
        // Deux mots : « ab » puis rien d'autre à fusionner.
        let bpe = Bpe::train(&counts("ab"), 1000);
        assert_eq!(bpe.vocab_size(), 258);
    }

    #[test]
    fn trained_bpe_round_trips_and_compresses() {
        let text = sample_text();
        let bpe = Bpe::train(&counts(&text), 400);
        let tokens = bpe.encode(&text);
        assert_eq!(bpe.decode(&tokens), text);
        assert!(tokens.len() * 2 < text.len(), "{} tokens", tokens.len());
        assert!(tokens.iter().all(|&t| t != END_OF_DOCUMENT));
    }

    #[test]
    fn unseen_text_still_round_trips() {
        let bpe = Bpe::train(&counts(&sample_text()), 400);
        let text = "Ünbekannte Wörter, ¿qué? 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 \u{0}\u{7f}\n";
        assert_eq!(bpe.decode(&bpe.encode(text)), text);
    }

    #[test]
    fn merges_are_applied_in_learned_order() {
        // Fusions apprises : (a,b) puis (ab,c). Sur « bc » seul, rien ne
        // s'applique ; sur « abc », les deux, dans l'ordre.
        let (a, b, c) = (u16::from(b'a'), u16::from(b'b'), u16::from(b'c'));
        let bpe = Bpe::from_merges(vec![(a, b), (257, c), (b, c)]).unwrap();
        assert_eq!(bpe.encode("abc"), [258]);
        assert_eq!(bpe.encode("bc"), [259]);
        // (a,b) passe avant (b,c), bien que les deux soient possibles.
        assert_eq!(bpe.encode("abcbc"), [258, 259]);
    }

    #[test]
    fn cached_encoding_gives_the_same_tokens() {
        let text = sample_text();
        let bpe = Bpe::train(&counts(&text), 400);
        let mut encoder = bpe.encoder();
        let (mut first, mut second) = (Vec::new(), Vec::new());
        encoder.encode_into(&text, &mut first);
        encoder.encode_into(&text, &mut second);
        assert_eq!(first, second);
        let word_by_word: Vec<u16> = split_words(&text)
            .flat_map(|word| bpe.encode_word(word.as_bytes()))
            .collect();
        assert_eq!(first, word_by_word);
    }

    #[test]
    fn decoder_gives_the_same_text_piece_by_piece() {
        let bpe = Bpe::bytes();
        // Des caractères de 1 à 4 octets, découpés octet par octet, puis des
        // octets qui ne forment pas de l'UTF-8 : une suite seule, un début
        // sans suite, et un caractère coupé par la fin du texte.
        let mut bytes = "aé€😀 fin".as_bytes().to_vec();
        bytes.extend([0x80, b'x', 0xE2, b'y', 0xF0, 0x9F]);
        let tokens: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();

        let mut decoder = bpe.decoder();
        let pieces: Vec<String> = tokens.iter().map(|&t| decoder.push(t)).collect();
        // Le premier octet de « é » n'écrit rien, le second écrit « é ».
        assert_eq!(&pieces[..3], &["a", "", "é"]);
        let text = pieces.concat() + &decoder.finish();
        assert_eq!(text, bpe.decode(&tokens));
        assert_eq!(text.matches(char::REPLACEMENT_CHARACTER).count(), 3);
    }

    #[test]
    fn decoder_handles_characters_split_across_merged_tokens() {
        let words = HashMap::from([("été".as_bytes().to_vec(), 5), (b" a".to_vec(), 1)]);
        let bpe = Bpe::train(&words, 262);
        let tokens = bpe.encode("été a été");
        let mut decoder = bpe.decoder();
        let mut text: String = tokens.iter().map(|&t| decoder.push(t)).collect();
        text += &decoder.finish();
        assert_eq!(text, "été a été");
    }

    #[test]
    fn bpe_decode_skips_end_of_document() {
        let bpe = Bpe::train(&counts(&sample_text()), 400);
        let mut tokens = bpe.encode("The cat");
        tokens.push(END_OF_DOCUMENT);
        tokens.extend(bpe.encode(" sat"));
        assert_eq!(bpe.decode(&tokens), "The cat sat");
    }

    #[test]
    fn merges_survive_a_round_trip_on_disk() {
        let bpe = Bpe::train(&counts(&sample_text()), 400);
        let path = std::env::temp_dir().join(format!("smallem-{}.bpe", std::process::id()));
        bpe.save(&path).unwrap();
        let loaded = Bpe::load(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(loaded, bpe);
    }

    #[test]
    fn invalid_merge_tables_are_rejected() {
        // Un token qui n'existe pas encore, puis la fin de document.
        assert!(Bpe::from_merges(vec![(97, 300)]).is_err());
        assert!(Bpe::from_merges(vec![(97, END_OF_DOCUMENT)]).is_err());
        assert!(Bpe::from_merges(vec![(97, 98), (97, 98)]).is_err());

        let path = std::env::temp_dir().join(format!("smallem-bad-{}.bpe", std::process::id()));
        for content in ["97 98\n", "smallem-bpe 1\n97\n", "smallem-bpe 1\n97 x\n"] {
            fs::write(&path, content).unwrap();
            assert!(Bpe::load(&path).is_err(), "{content:?}");
        }
        // L'erreur donne la ligne du fichier, en-tête compris.
        for content in [
            "smallem-bpe 1\n97 98\n97 x\n",
            "smallem-bpe 1\n97 98\n97 98\n",
        ] {
            fs::write(&path, content).unwrap();
            let error = Bpe::load(&path).unwrap_err().to_string();
            assert!(error.starts_with("ligne 3 "), "{content:?} : {error}");
        }
        fs::remove_file(&path).unwrap();
    }
}
