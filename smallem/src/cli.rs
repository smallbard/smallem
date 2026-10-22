//! Ce que partagent les programmes en ligne de commande : la lecture des
//! arguments.

use std::error::Error;
use std::str::FromStr;

/// Les arguments d'un programme, lus un par un.
pub struct Args {
    args: std::iter::Peekable<Box<dyn Iterator<Item = String>>>,
}

impl Args {
    /// Les arguments de ce processus, sans le nom du programme.
    pub fn from_env() -> Self {
        Self::new(std::env::args().skip(1))
    }

    pub fn new(args: impl IntoIterator<Item = String> + 'static) -> Self {
        let args: Box<dyn Iterator<Item = String>> = Box::new(args.into_iter());
        Self {
            args: args.peekable(),
        }
    }

    /// Consomme l'argument suivant s'il vaut `word` : pour une option dont
    /// la valeur est facultative, comme `--cpu reference`.
    pub fn accept(&mut self, word: &str) -> bool {
        self.args.next_if(|next| next == word).is_some()
    }

    /// La valeur qui suit l'option `flag`, convertie dans le type attendu.
    /// L'erreur nomme l'option : c'est tout ce que voit l'utilisateur.
    pub fn value<T: FromStr>(&mut self, flag: &str) -> Result<T, Box<dyn Error>>
    where
        T::Err: std::fmt::Display,
    {
        let text = self
            .args
            .next()
            .ok_or_else(|| format!("{flag} attend une valeur"))?;
        text.parse()
            .map_err(|error| format!("{flag} : valeur `{text}` refusée ({error})").into())
    }
}

/// Un argument qui n'est pas une option : refuse ce qui commence par `--`,
/// pour qu'une option mal écrite ne soit pas prise pour un fichier.
pub fn positional(arg: &str) -> Result<&str, Box<dyn Error>> {
    if arg.starts_with("--") {
        Err(format!("option inconnue : {arg}").into())
    } else {
        Ok(arg)
    }
}

/// Refuse une option entière qui vaut 0, comme `--steps 0` ou
/// `--eval-every 0` : la boucle ne ferait rien, ou un reste diviserait par
/// zéro. L'erreur nomme la première option fautive.
pub fn at_least_one(options: &[(&str, usize)]) -> Result<(), Box<dyn Error>> {
    match options.iter().find(|(_, value)| *value == 0) {
        Some((flag, _)) => Err(format!("{flag} doit valoir au moins 1").into()),
        None => Ok(()),
    }
}

/// Refuse un réel qui n'est pas fini ou qui sort de `range`, comme un taux
/// d'apprentissage négatif ou une proportion au-delà de 1.
pub fn check_range(
    flag: &str,
    value: f64,
    range: impl std::ops::RangeBounds<f64>,
) -> Result<(), Box<dyn Error>> {
    if value.is_finite() && range.contains(&value) {
        Ok(())
    } else {
        Err(format!("{flag} : valeur {value} refusée").into())
    }
}

/// Refuse un réel qui n'est pas fini et strictement positif.
pub fn check_positive(flag: &str, value: f64) -> Result<(), Box<dyn Error>> {
    use std::ops::Bound;
    check_range(flag, value, (Bound::Excluded(0.0), Bound::Unbounded))
}

impl Iterator for Args {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        self.args.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Args {
        Args::new(list.iter().map(|arg| arg.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn values_are_converted_to_the_expected_type() {
        let mut args = args(&["12", "0.5", "chemin/fichier", "texte"]);
        assert_eq!(args.value::<usize>("--a").unwrap(), 12);
        assert_eq!(args.value::<f64>("--b").unwrap(), 0.5);
        assert_eq!(
            args.value::<std::path::PathBuf>("--c").unwrap(),
            std::path::PathBuf::from("chemin/fichier")
        );
        assert_eq!(args.value::<String>("--d").unwrap(), "texte");
        assert!(args.next().is_none());
    }

    #[test]
    fn an_optional_value_is_taken_only_if_present() {
        let mut args = args(&["--cpu", "reference", "--cpu", "--tokens"]);
        assert_eq!(args.next().unwrap(), "--cpu");
        assert!(args.accept("reference"));
        assert_eq!(args.next().unwrap(), "--cpu");
        assert!(!args.accept("reference"));
        assert_eq!(args.next().unwrap(), "--tokens");
    }

    #[test]
    fn errors_name_the_option() {
        let missing = args(&[]).value::<usize>("--steps").unwrap_err().to_string();
        assert_eq!(missing, "--steps attend une valeur");
        let invalid = args(&["beaucoup"])
            .value::<usize>("--steps")
            .unwrap_err()
            .to_string();
        assert!(
            invalid.starts_with("--steps : valeur `beaucoup` refusée"),
            "{invalid}"
        );
    }

    #[test]
    fn zero_counts_and_values_out_of_range_are_refused() {
        assert!(at_least_one(&[("--steps", 3), ("--batch", 1)]).is_ok());
        let error = at_least_one(&[("--steps", 3), ("--eval-every", 0)]).unwrap_err();
        assert_eq!(error.to_string(), "--eval-every doit valoir au moins 1");
        assert!(check_range("--mix", 0.0, 0.0..1.0).is_ok());
        assert!(check_range("--mix", 1.0, 0.0..1.0).is_err());
        assert!(check_range("--mix", f64::NAN, 0.0..1.0).is_err());
        assert!(check_positive("--lr", 1e-4).is_ok());
        let error = check_positive("--lr", 0.0).unwrap_err().to_string();
        assert_eq!(error, "--lr : valeur 0 refusée");
        assert!(check_positive("--lr", f64::INFINITY).is_err());
    }

    #[test]
    fn an_unknown_option_is_not_a_positional_argument() {
        assert_eq!(positional("results.tsv").unwrap(), "results.tsv");
        assert_eq!(positional("-").unwrap(), "-");
        let error = positional("--xx").unwrap_err().to_string();
        assert_eq!(error, "option inconnue : --xx");
    }
}
