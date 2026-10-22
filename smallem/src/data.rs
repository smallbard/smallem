//! Lecture des fichiers de tokens produits par `prepare` et tirage des lots :
//! soit un fichier chargé en mémoire ([`Dataset`]), soit plusieurs fichiers
//! laissés sur le disque ([`Shards`]).

use std::fs::{self, File};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use rand::{Rng, RngExt};

/// Un fichier de tokens entièrement chargé en mémoire.
pub struct Dataset {
    tokens: Vec<u16>,
}

/// Un lot de `batch_size` séquences de `context` tokens.
///
/// `inputs` et `targets` sont deux tableaux `[batch_size, context]` rangés
/// ligne par ligne. `targets` est `inputs` décalé d'un token : la cible à la
/// position `t` est le token qui suit `inputs[t]` dans le corpus.
pub struct Batch {
    pub inputs: Vec<u16>,
    pub targets: Vec<u16>,
    pub batch_size: usize,
    pub context: usize,
}

impl Batch {
    /// Les tokens et leurs cibles tels qu'un modèle les attend : à chaque
    /// position, la cible est le token suivant.
    pub fn examples(&self) -> (Vec<u32>, Vec<u32>) {
        let tokens = self.inputs.iter().map(|&t| u32::from(t)).collect();
        let targets = self.targets.iter().map(|&t| u32::from(t)).collect();
        (tokens, targets)
    }
}

/// Un corpus dont on tire des lots, qu'il soit en mémoire ou sur disque.
///
/// C'est tout ce qu'une boucle d'entraînement demande à ses données.
pub trait Batches {
    /// Nombre de tokens du corpus.
    fn token_count(&self) -> usize;

    /// Tire `batch_size` fenêtres de `context` tokens et leurs cibles. Deux
    /// tirages avec le même état de `rng` donnent le même lot.
    fn batch(&self, rng: &mut impl Rng, batch_size: usize, context: usize) -> io::Result<Batch>;
}

impl Batches for Dataset {
    fn token_count(&self) -> usize {
        self.len()
    }

    fn batch(&self, rng: &mut impl Rng, batch_size: usize, context: usize) -> io::Result<Batch> {
        if self.tokens.len() <= context {
            return Err(too_short(self.tokens.len(), context + 1));
        }
        Ok(self.sample_batch(rng, batch_size, context))
    }
}

impl Batches for Shards {
    fn token_count(&self) -> usize {
        self.len()
    }

    fn batch(&self, rng: &mut impl Rng, batch_size: usize, context: usize) -> io::Result<Batch> {
        self.sample_batch(rng, batch_size, context)
    }
}

impl Dataset {
    /// Charge un fichier de `u16` petit-boutistes sans en-tête.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let (pairs, rest) = bytes.as_chunks::<2>();
        if !rest.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "taille impaire : ce n'est pas un fichier de u16",
            ));
        }
        let tokens = pairs.iter().map(|&pair| u16::from_le_bytes(pair)).collect();
        Ok(Self { tokens })
    }

    pub fn from_tokens(tokens: Vec<u16>) -> Self {
        Self { tokens }
    }

    pub fn tokens(&self) -> &[u16] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Vérifie que chaque token est dans un vocabulaire de `vocab` tokens :
    /// sinon, le corpus et le modèle ne vont pas ensemble.
    pub fn check_vocab(&self, vocab: usize) -> io::Result<()> {
        check_tokens(&self.tokens, vocab)
    }

    /// Tire `batch_size` fenêtres de `context + 1` tokens à des positions
    /// aléatoires, indépendantes les unes des autres. Une fenêtre peut
    /// chevaucher une frontière de document : le token de fin de document
    /// fait alors partie de la séquence, comme n'importe quel autre.
    ///
    /// # Panics
    ///
    /// Si le corpus est plus court qu'une fenêtre. [`Batches::batch`] rend
    /// alors une erreur.
    pub fn sample_batch(&self, rng: &mut impl Rng, batch_size: usize, context: usize) -> Batch {
        let window = context + 1;
        assert!(
            self.tokens.len() >= window,
            "{}",
            too_short(self.tokens.len(), window),
        );
        // Dernier départ possible : celui dont la fenêtre finit au dernier token.
        let last_start = self.tokens.len() - window;

        let mut inputs = Vec::with_capacity(batch_size * context);
        let mut targets = Vec::with_capacity(batch_size * context);
        for _ in 0..batch_size {
            let start = rng.random_range(0..=last_start);
            inputs.extend_from_slice(&self.tokens[start..start + context]);
            targets.extend_from_slice(&self.tokens[start + 1..start + window]);
        }
        Batch {
            inputs,
            targets,
            batch_size,
            context,
        }
    }
}

/// Un corpus réparti sur plusieurs fichiers de tokens, qui restent sur le
/// disque : seules les fenêtres tirées sont lues. Le système garde en
/// mémoire les pages déjà lues tant qu'il a de la place, sans qu'on ait à
/// s'en occuper.
pub struct Shards {
    shards: Vec<Shard>,
}

struct Shard {
    file: File,
    /// Nombre de tokens du fichier.
    tokens: usize,
}

impl Shards {
    /// Ouvre des fichiers de `u16` petit-boutistes sans en-tête.
    pub fn open(paths: &[PathBuf]) -> io::Result<Self> {
        if paths.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "aucun fichier de tokens",
            ));
        }
        let mut shards = Vec::new();
        for path in paths {
            let file = File::open(path)?;
            let bytes = file.metadata()?.len() as usize;
            if !bytes.is_multiple_of(2) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} : taille impaire", path.display()),
                ));
            }
            shards.push(Shard {
                file,
                tokens: bytes / 2,
            });
        }
        Ok(Self { shards })
    }

    /// Ouvre tous les fichiers de `dir` dont le nom finit par `suffix`,
    /// dans l'ordre alphabétique : `train.bin` retient aussi bien
    /// `train.bin` que `000_00000.train.bin`.
    pub fn open_dir(dir: impl AsRef<Path>, suffix: &str) -> io::Result<Self> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            let matches = path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(suffix));
            if matches && path.is_file() {
                paths.push(path);
            }
        }
        paths.sort();
        Self::open(&paths).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("{} (*{suffix}) : {error}", dir.as_ref().display()),
            )
        })
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Nombre de tokens du fichier `shard`.
    pub fn shard_len(&self, shard: usize) -> usize {
        self.shards[shard].tokens
    }

    /// Nombre total de tokens.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.tokens).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lit `len` tokens du fichier `shard` à partir du token `start`.
    pub fn read(&self, shard: usize, start: usize, len: usize) -> io::Result<Vec<u16>> {
        let mut bytes = vec![0u8; len * 2];
        self.shards[shard]
            .file
            .read_exact_at(&mut bytes, start as u64 * 2)?;
        let (pairs, _) = bytes.as_chunks::<2>();
        Ok(pairs.iter().map(|&pair| u16::from_le_bytes(pair)).collect())
    }

    /// Comme [`Dataset::sample_batch`] : chaque départ possible, tous
    /// fichiers confondus, a la même chance d'être tiré. Une fenêtre ne
    /// chevauche jamais deux fichiers.
    pub fn sample_batch(
        &self,
        rng: &mut impl Rng,
        batch_size: usize,
        context: usize,
    ) -> io::Result<Batch> {
        let window = context + 1;
        // Nombre de départs valides dans chaque fichier.
        let starts = |shard: &Shard| (shard.tokens + 1).saturating_sub(window);
        let total: usize = self.shards.iter().map(starts).sum();
        if total == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("aucun fichier n'est assez long pour une fenêtre de {window} tokens"),
            ));
        }

        let mut inputs = Vec::with_capacity(batch_size * context);
        let mut targets = Vec::with_capacity(batch_size * context);
        for _ in 0..batch_size {
            // Un départ parmi tous, puis le fichier auquel il appartient.
            let mut start = rng.random_range(0..total);
            let mut shard = 0;
            while start >= starts(&self.shards[shard]) {
                start -= starts(&self.shards[shard]);
                shard += 1;
            }
            let tokens = self.read(shard, start, window)?;
            inputs.extend_from_slice(&tokens[..context]);
            targets.extend_from_slice(&tokens[1..]);
        }
        Ok(Batch {
            inputs,
            targets,
            batch_size,
            context,
        })
    }

    /// Vérifie que chaque token de chaque fichier est dans un vocabulaire
    /// de `vocab` tokens. Les fichiers sont lus en entier, par morceaux :
    /// à faire une fois, à l'ouverture, quand le doute est permis.
    pub fn check_vocab(&self, vocab: usize) -> io::Result<()> {
        self.for_each_chunk(|tokens| check_tokens(tokens, vocab))
    }

    /// Passe tous les tokens à `visit`, fichier par fichier, par morceaux
    /// d'un million : sans charger le corpus en entier.
    fn for_each_chunk(&self, mut visit: impl FnMut(&[u16]) -> io::Result<()>) -> io::Result<()> {
        const CHUNK: usize = 1 << 20;
        for shard in 0..self.shard_count() {
            let len = self.shard_len(shard);
            for start in (0..len).step_by(CHUNK) {
                visit(&self.read(shard, start, CHUNK.min(len - start))?)?;
            }
        }
        Ok(())
    }
}

/// L'erreur d'un corpus de `len` tokens, plus court qu'une fenêtre.
fn too_short(len: usize, window: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("le corpus ({len} tokens) est plus court qu'une fenêtre ({window})"),
    )
}

/// Une erreur si un token de `tokens` sort d'un vocabulaire de `vocab`.
fn check_tokens(tokens: &[u16], vocab: usize) -> io::Result<()> {
    match tokens.iter().find(|&&token| token as usize >= vocab) {
        Some(token) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("token {token} hors du vocabulaire de {vocab} tokens"),
        )),
        None => Ok(()),
    }
}

/// Octets de texte par token d'un corpus, mesuré en relisant chaque token
/// avec `tokenizer`. Il convertit une perte par token en perte par octet,
/// la seule qui se compare d'un tokeniseur à l'autre.
///
/// Le corpus est lu par morceaux, sans être chargé en entier. Un token hors
/// du vocabulaire est une erreur : le corpus et le tokeniseur ne vont pas
/// ensemble.
pub fn bytes_per_token(shards: &Shards, tokenizer: &crate::tokenizer::Bpe) -> io::Result<f64> {
    if shards.is_empty() {
        return Err(io::Error::other("corpus vide"));
    }
    let mut bytes = 0usize;
    shards.for_each_chunk(|tokens| {
        check_tokens(tokens, tokenizer.vocab_size())?;
        bytes += tokens
            .iter()
            .map(|&token| tokenizer.token_bytes(token).len())
            .sum::<usize>();
        Ok(())
    })?;
    Ok(bytes as f64 / shards.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Corpus `0, 1, 2, …` : le token à la position `i` vaut `i`.
    fn counting(len: u16) -> Dataset {
        Dataset::from_tokens((0..len).collect())
    }

    #[test]
    fn examples_use_every_position() {
        let batch = counting(100).sample_batch(&mut StdRng::seed_from_u64(0), 3, 5);
        let (tokens, targets) = batch.examples();
        assert_eq!(tokens.len(), 15);
        assert!(tokens.iter().zip(&targets).all(|(t, n)| *n == t + 1));
    }

    #[test]
    fn memory_and_disk_are_drawn_from_the_same_way() {
        fn draw(source: &impl Batches) -> Batch {
            source.batch(&mut StdRng::seed_from_u64(3), 2, 4).unwrap()
        }
        let dataset = counting(50);
        assert_eq!(Batches::token_count(&dataset), 50);
        let batch = draw(&dataset);
        let direct = dataset.sample_batch(&mut StdRng::seed_from_u64(3), 2, 4);
        assert_eq!(batch.inputs, direct.inputs);
        assert_eq!(batch.targets, direct.targets);
    }

    #[test]
    fn load_reads_little_endian_u16() {
        let path = std::env::temp_dir().join(format!("smallem-data-{}.bin", std::process::id()));
        fs::write(&path, [0x01, 0x00, 0x00, 0x01, 0xFF, 0xFF]).unwrap();
        let dataset = Dataset::load(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(dataset.tokens(), &[1, 256, 65535]);
    }

    #[test]
    fn load_rejects_odd_length() {
        let path = std::env::temp_dir().join(format!("smallem-odd-{}.bin", std::process::id()));
        fs::write(&path, [0x01, 0x00, 0x02]).unwrap();
        let result = Dataset::load(&path);
        fs::remove_file(&path).unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn batch_has_expected_shape() {
        let batch = counting(1000).sample_batch(&mut StdRng::seed_from_u64(0), 8, 16);
        assert_eq!(batch.inputs.len(), 8 * 16);
        assert_eq!(batch.targets.len(), 8 * 16);
    }

    #[test]
    fn targets_are_inputs_shifted_by_one() {
        let batch = counting(1000).sample_batch(&mut StdRng::seed_from_u64(0), 8, 16);
        for (row_inputs, row_targets) in batch.inputs.chunks(16).zip(batch.targets.chunks(16)) {
            // Chaque ligne est une suite de tokens consécutifs du corpus…
            assert!(row_inputs.windows(2).all(|w| w[1] == w[0] + 1));
            // … et la cible est le token suivant.
            assert!(row_inputs.iter().zip(row_targets).all(|(i, t)| *t == i + 1));
        }
    }

    #[test]
    fn every_start_position_is_reachable() {
        // 5 tokens, fenêtre de 4 : seuls les départs 0 et 1 sont valides.
        let dataset = counting(5);
        let mut rng = StdRng::seed_from_u64(0);
        let mut seen = [false; 2];
        for _ in 0..100 {
            let batch = dataset.sample_batch(&mut rng, 1, 3);
            seen[batch.inputs[0] as usize] = true;
            assert_eq!(*batch.targets.last().unwrap(), batch.inputs[0] + 3);
        }
        assert_eq!(seen, [true, true]);
    }

    #[test]
    fn same_seed_gives_same_batch() {
        let dataset = counting(1000);
        let a = dataset.sample_batch(&mut StdRng::seed_from_u64(42), 4, 8);
        let b = dataset.sample_batch(&mut StdRng::seed_from_u64(42), 4, 8);
        assert_eq!(a.inputs, b.inputs);
    }

    #[test]
    #[should_panic(expected = "plus court qu'une fenêtre")]
    fn corpus_shorter_than_window_panics() {
        counting(4).sample_batch(&mut StdRng::seed_from_u64(0), 1, 4);
    }

    /// Écrit des fichiers de tokens dans un dossier temporaire.
    fn shard_dir(name: &str, files: &[(&str, Vec<u16>)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("smallem-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for (file, tokens) in files {
            let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            fs::write(dir.join(file), bytes).unwrap();
        }
        dir
    }

    #[test]
    fn shards_are_found_by_suffix_in_sorted_order() {
        let dir = shard_dir(
            "suffix",
            &[
                ("b.train.bin", vec![10, 11, 12]),
                ("a.train.bin", vec![1, 2]),
                ("a.val.bin", vec![99]),
                ("tokenizer.bpe", vec![7]),
            ],
        );
        let train = Shards::open_dir(&dir, "train.bin").unwrap();
        let val = Shards::open_dir(&dir, "val.bin").unwrap();
        let missing = Shards::open_dir(&dir, "test.bin");
        let first = train.read(0, 0, 2).unwrap();
        let second = train.read(1, 1, 2).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!((train.shard_count(), train.len()), (2, 5));
        assert_eq!((train.shard_len(0), train.shard_len(1)), (2, 3));
        assert_eq!(first, [1, 2]);
        assert_eq!(second, [11, 12]);
        assert_eq!(val.len(), 1);
        assert!(missing.is_err());
    }

    #[test]
    fn single_shard_draws_the_same_batches_as_a_dataset() {
        let tokens: Vec<u16> = (0..1000).collect();
        let dir = shard_dir("same", &[("train.bin", tokens.clone())]);
        let shards = Shards::open_dir(&dir, "train.bin").unwrap();
        let from_disk = shards
            .sample_batch(&mut StdRng::seed_from_u64(7), 8, 16)
            .unwrap();
        fs::remove_dir_all(&dir).unwrap();
        let in_memory =
            Dataset::from_tokens(tokens).sample_batch(&mut StdRng::seed_from_u64(7), 8, 16);
        assert_eq!(from_disk.inputs, in_memory.inputs);
        assert_eq!(from_disk.targets, in_memory.targets);
    }

    #[test]
    fn windows_never_straddle_two_shards() {
        // Fichiers 0..10, 100..104 (trop court pour une fenêtre de 5) et
        // 200..206 : dans une fenêtre, les tokens se suivent toujours.
        let dir = shard_dir(
            "straddle",
            &[
                ("0.train.bin", (0..10).collect()),
                ("1.train.bin", (100..104).collect()),
                ("2.train.bin", (200..206).collect()),
            ],
        );
        let shards = Shards::open_dir(&dir, "train.bin").unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let mut first_tokens = std::collections::BTreeSet::new();
        for _ in 0..200 {
            let batch = shards.sample_batch(&mut rng, 1, 4).unwrap();
            assert!(batch.inputs.windows(2).all(|w| w[1] == w[0] + 1));
            assert_eq!(batch.targets[3], batch.inputs[0] + 4);
            first_tokens.insert(batch.inputs[0]);
        }
        fs::remove_dir_all(&dir).unwrap();
        // Six départs dans le premier fichier, aucun dans le deuxième,
        // deux dans le troisième : tous sont atteints, et eux seuls.
        let expected: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 200, 201];
        assert_eq!(first_tokens.into_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn a_corpus_too_short_is_an_error_when_drawing_through_batches() {
        let error = counting(4)
            .batch(&mut StdRng::seed_from_u64(0), 1, 4)
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let dir = shard_dir("short", &[("train.bin", vec![1, 2, 3])]);
        let shards = Shards::open_dir(&dir, "train.bin").unwrap();
        let error = shards
            .sample_batch(&mut StdRng::seed_from_u64(0), 1, 4)
            .err()
            .unwrap();
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn tokens_outside_the_vocabulary_are_found() {
        assert!(counting(10).check_vocab(10).is_ok());
        assert!(counting(10).check_vocab(9).is_err());
        let dir = shard_dir(
            "vocab",
            &[("a.train.bin", vec![1, 2]), ("b.train.bin", vec![3, 300])],
        );
        let shards = Shards::open_dir(&dir, "train.bin").unwrap();
        let (fits, overflows) = (shards.check_vocab(301), shards.check_vocab(300));
        fs::remove_dir_all(&dir).unwrap();
        assert!(fits.is_ok());
        assert_eq!(overflows.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn odd_sized_shard_is_rejected() {
        let dir = shard_dir("odd", &[]);
        fs::write(dir.join("train.bin"), [1, 2, 3]).unwrap();
        let result = Shards::open_dir(&dir, "train.bin");
        fs::remove_dir_all(&dir).unwrap();
        assert!(result.is_err());
    }
}
