//! Des tâches dont un programme connaît la bonne réponse : de quoi mesurer
//! exactement ce qu'un modèle sait faire, avant et après son
//! post-entraînement.
//!
//! Chaque famille a un générateur, qui tire une question et sa réponse
//! d'après une graine, et une vérification : la réponse du modèle est juste
//! si elle est exactement la réponse attendue, aux espaces près. Les
//! questions sont en anglais : c'est la langue du texte sur lequel le
//! modèle a appris.
//!
//! Les jeux d'évaluation sont tirés avec d'autres graines que
//! l'entraînement, et pour la recopie à l'envers, avec d'autres mots : le
//! modèle ne peut pas les avoir vus.

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use crate::chat::Turn;

/// Une famille de tâches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Additionner deux nombres de 2 ou 3 chiffres.
    Addition,
    /// Recopier un mot à l'envers.
    Reverse,
    /// Trier de 3 à 5 nombres.
    Sort,
    /// Retrouver un fait dans une phrase donnée.
    Lookup,
    /// Additionner deux nombres de 4 chiffres : plus difficile que ce que
    /// montre l'entraînement supervisé.
    LongAddition,
    /// Trier de 6 à 8 nombres : plus difficile que ce que montre
    /// l'entraînement supervisé.
    LongSort,
}

impl Family {
    /// Les quatre familles du fine-tuning supervisé.
    pub const ALL: [Family; 4] = [Self::Addition, Self::Reverse, Self::Sort, Self::Lookup];

    pub fn name(self) -> &'static str {
        match self {
            Self::Addition => "addition",
            Self::Reverse => "envers",
            Self::Sort => "tri",
            Self::Lookup => "lecture",
            Self::LongAddition => "addition longue",
            Self::LongSort => "tri long",
        }
    }

    /// Tire une question et sa réponse. Avec `evaluation`, la recopie à
    /// l'envers prend ses mots dans une liste que l'entraînement ne voit
    /// jamais.
    pub fn sample(self, rng: &mut impl Rng, evaluation: bool) -> Turn {
        let (question, answer) = match self {
            Self::Addition | Self::LongAddition => {
                let range = if self == Self::Addition {
                    10..1000
                } else {
                    1000..10_000
                };
                let (a, b) = (rng.random_range(range.clone()), rng.random_range(range));
                (format!("What is {a} + {b}?"), (a + b).to_string())
            }
            Self::Reverse => {
                let words = if evaluation {
                    EVALUATION_WORDS
                } else {
                    TRAINING_WORDS
                };
                let word = words[rng.random_range(0..words.len())];
                (
                    format!("Write the word backwards: {word}"),
                    word.chars().rev().collect(),
                )
            }
            Self::Sort | Self::LongSort => {
                let count = if self == Self::Sort {
                    rng.random_range(3..=5)
                } else {
                    rng.random_range(6..=8)
                };
                let mut numbers: Vec<u32> = (0..count).map(|_| rng.random_range(0..100)).collect();
                let question = format!("Sort from smallest to largest: {}", join(&numbers));
                numbers.sort_unstable();
                (question, join(&numbers))
            }
            Self::Lookup => {
                let name = NAMES[rng.random_range(0..NAMES.len())];
                let (fact, question, answer) = match rng.random_range(0..3) {
                    0 => {
                        let city = CITIES[rng.random_range(0..CITIES.len())];
                        (
                            format!("{name} lives in {city}."),
                            format!("Where does {name} live?"),
                            city,
                        )
                    }
                    1 => {
                        let color = COLORS[rng.random_range(0..COLORS.len())];
                        (
                            format!("{name} has a {color} car."),
                            format!("What color is {name}'s car?"),
                            color,
                        )
                    }
                    _ => {
                        let animal = ANIMALS[rng.random_range(0..ANIMALS.len())];
                        (
                            format!("{name} owns a {animal}."),
                            format!("What animal does {name} own?"),
                            animal,
                        )
                    }
                };
                (format!("{fact} {question}"), answer.to_string())
            }
        };
        Turn { question, answer }
    }

    /// Vrai si `answer` est la bonne réponse à `turn`.
    pub fn check(turn: &Turn, answer: &str) -> bool {
        answer.trim() == turn.answer.trim()
    }

    /// Le jeu d'évaluation de la famille : `count` questions, toujours les
    /// mêmes, tirées avec une graine que l'entraînement n'utilise pas.
    pub fn evaluation_set(self, count: usize) -> Vec<Turn> {
        let mut rng = StdRng::seed_from_u64(EVALUATION_SEED + self as u64);
        (0..count).map(|_| self.sample(&mut rng, true)).collect()
    }
}

/// Les graines d'entraînement sont petites ; celle-ci en est loin.
const EVALUATION_SEED: u64 = 1 << 40;

fn join(numbers: &[u32]) -> String {
    numbers
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) const TRAINING_WORDS: &[&str] = &[
    "garden", "window", "river", "planet", "doctor", "pencil", "summer", "market", "orange",
    "silver", "forest", "bridge", "candle", "camera", "dragon", "engine", "farmer", "island",
    "jacket", "kitten", "ladder", "monkey", "number", "office", "parent", "rabbit", "school",
    "ticket", "valley", "winter", "yellow", "animal", "button", "castle", "dinner", "family",
    "guitar", "hammer", "letter", "mother", "nature", "people", "rocket", "spring", "tomato",
    "violin", "wonder", "basket", "circle", "desert", "flower", "honey", "lemon", "music", "night",
    "ocean", "paper", "queen", "smile", "table", "water", "zebra", "apple", "bread", "chair",
    "dream", "earth", "glass", "house", "light", "money", "north", "plant", "sound", "train",
    "world", "beach", "cloud", "field", "heart", "metal", "storm", "tiger", "voice",
];

pub(crate) const EVALUATION_WORDS: &[&str] = &[
    "butter", "kettle", "puzzle", "anchor", "pillow", "rescue", "shadow", "thread", "velvet",
    "walnut", "copper", "harbor", "marble", "needle", "saddle", "turtle", "carpet", "falcon",
    "goblet", "lantern",
];

pub(crate) const NAMES: &[&str] = &[
    "Anna", "Paul", "Maria", "John", "Lucy", "Omar", "Elena", "David", "Sara", "Hugo", "Nina",
    "Peter", "Clara", "Leo", "Emma", "Victor",
];

pub(crate) const CITIES: &[&str] = &[
    "Paris", "London", "Berlin", "Madrid", "Rome", "Tokyo", "Cairo", "Lima", "Oslo", "Dublin",
    "Vienna", "Boston",
];

const COLORS: &[&str] = &[
    "red", "blue", "green", "black", "white", "yellow", "gray", "brown",
];

const ANIMALS: &[&str] = &[
    "dog", "cat", "horse", "parrot", "rabbit", "goat", "turtle", "snake",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_are_right_by_construction() {
        let mut rng = StdRng::seed_from_u64(3);
        let turn = Family::Addition.sample(&mut rng, false);
        let numbers: Vec<u32> = turn
            .question
            .trim_end_matches('?')
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        assert_eq!(turn.answer, (numbers[0] + numbers[1]).to_string());
        assert!(Family::check(&turn, &format!(" {} ", turn.answer)));
        assert!(!Family::check(&turn, "0"));
        let sorted = Family::Sort.sample(&mut rng, false);
        let values: Vec<u32> = sorted
            .answer
            .split(' ')
            .map(|w| w.parse().unwrap())
            .collect();
        assert!(values.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn evaluation_sets_are_fixed_and_use_unseen_words() {
        assert_eq!(
            Family::Sort.evaluation_set(5),
            Family::Sort.evaluation_set(5)
        );
        for turn in Family::Reverse.evaluation_set(50) {
            let word: String = turn.answer.chars().rev().collect();
            assert!(EVALUATION_WORDS.contains(&word.as_str()));
            assert!(!TRAINING_WORDS.contains(&word.as_str()));
        }
    }
}
