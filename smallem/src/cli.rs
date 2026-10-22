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
    fn an_unknown_option_is_not_a_positional_argument() {
        assert_eq!(positional("results.tsv").unwrap(), "results.tsv");
        assert_eq!(positional("-").unwrap(), "-");
        let error = positional("--xx").unwrap_err().to_string();
        assert_eq!(error, "option inconnue : --xx");
    }
}
