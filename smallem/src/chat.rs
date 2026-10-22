//! Le format des conversations : des tokens de rôle, et un masque qui dit
//! quels tokens l'assistant a écrits.
//!
//! Un modèle de langage continue un texte. Pour qu'il réponde, on lui
//! montre des conversations, où chaque tour commence par un token qui dit
//! qui parle :
//!
//! ```text
//! <|utilisateur|> What is 12 + 30? <|assistant|> 42 <|fin|>
//! ```
//!
//! Ces trois tokens n'existent pas dans le vocabulaire appris sur le texte :
//! ils s'ajoutent après lui, et le modèle reçoit trois lignes d'embedding de
//! plus (voir [`Gpt::grow_vocab`](crate::model::Gpt::grow_vocab)). Le texte
//! entre eux est découpé par le tokeniseur ordinaire.
//!
//! Le masque sert à l'entraînement : la perte ne porte que sur ce que
//! l'assistant écrit, sa réponse et le `<|fin|>` qui la termine. Le modèle
//! lit la question, mais n'apprend pas à la poser.

use crate::tokenizer::Bpe;

/// Les noms des tokens de rôle, dans l'ordre de leurs numéros.
pub const ROLE_NAMES: [&str; 3] = ["<|utilisateur|>", "<|assistant|>", "<|fin|>"];

/// Les numéros des tokens de rôle d'un vocabulaire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roles {
    pub user: u32,
    pub assistant: u32,
    pub end: u32,
}

impl Roles {
    /// Les tokens de rôle qui suivent un vocabulaire de `base` tokens.
    pub fn after(base: usize) -> Self {
        let base = base as u32;
        Self {
            user: base,
            assistant: base + 1,
            end: base + 2,
        }
    }

    /// Le vocabulaire complet : le texte, puis les trois tokens de rôle.
    pub fn vocab(&self) -> usize {
        self.end as usize + 1
    }
}

/// Les noms des deux tokens d'un agent, dans l'ordre de leurs numéros.
pub const TOOL_NAMES: [&str; 2] = ["<|appel|>", "<|resultat|>"];

/// Les numéros des deux tokens qui encadrent un appel d'outil :
///
/// ```text
/// <|appel|> add 4821 3907 <|resultat|> 8728 <|assistant|>
/// ```
///
/// Le modèle écrit `<|appel|>`, l'appel, puis `<|resultat|>` ; le programme
/// écrit le résultat et rend la main par `<|assistant|>`. Ils suivent les
/// tokens de rôle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolTokens {
    pub call: u32,
    pub result: u32,
}

impl ToolTokens {
    /// Les deux tokens qui suivent ceux de `roles`.
    pub fn after(roles: Roles) -> Self {
        Self {
            call: roles.end + 1,
            result: roles.end + 2,
        }
    }

    /// Le vocabulaire complet : le texte, les tokens de rôle, ces deux-là.
    pub fn vocab(&self) -> usize {
        self.result as usize + 1
    }
}

/// Un tour de parole.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub question: String,
    pub answer: String,
}

/// Une conversation mise en tokens : `tokens`, et pour chacun, vrai si
/// l'assistant l'a écrit.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    pub tokens: Vec<u32>,
    pub assistant: Vec<bool>,
}

/// La consigne d'un tour : ce que le modèle lit avant de répondre.
pub fn prompt(tokenizer: &Bpe, roles: Roles, question: &str) -> Vec<u32> {
    let mut tokens = vec![roles.user];
    tokens.extend(tokenizer.encode(question).iter().map(|&t| u32::from(t)));
    tokens.push(roles.assistant);
    tokens
}

/// Un tour entier : la consigne, puis la réponse et `<|fin|>`, écrits par
/// l'assistant.
pub fn encode(tokenizer: &Bpe, roles: Roles, turn: &Turn) -> Encoded {
    let mut tokens = prompt(tokenizer, roles, &turn.question);
    let mut assistant = vec![false; tokens.len()];
    let answer: Vec<u32> = tokenizer
        .encode(&turn.answer)
        .iter()
        .map(|&t| u32::from(t))
        .chain([roles.end])
        .collect();
    assistant.extend(std::iter::repeat_n(true, answer.len()));
    tokens.extend(answer);
    Encoded { tokens, assistant }
}

/// Le texte d'une réponse : ses tokens jusqu'au premier `<|fin|>`, les
/// autres tokens de rôle laissés de côté.
pub fn answer_text(tokenizer: &Bpe, roles: Roles, tokens: &[u32]) -> String {
    let text: Vec<u16> = tokens
        .iter()
        .take_while(|&&t| t != roles.end)
        .filter(|&&t| t < roles.user)
        .map(|&t| t as u16)
        .collect();
    tokenizer.decode(&text).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_marks_exactly_what_the_assistant_writes() {
        let tokenizer = Bpe::bytes();
        let roles = Roles::after(tokenizer.vocab_size());
        let turn = Turn {
            question: "1+1?".into(),
            answer: "2".into(),
        };
        let encoded = encode(&tokenizer, roles, &turn);
        // `<|utilisateur|>`, quatre octets, `<|assistant|>`, puis la réponse
        // et `<|fin|>`.
        let mut expected = vec![roles.user];
        expected.extend("1+1?".bytes().map(u32::from));
        expected.extend([roles.assistant, u32::from(b'2'), roles.end]);
        assert_eq!(encoded.tokens, expected);
        assert_eq!(
            encoded.assistant,
            [false, false, false, false, false, false, true, true]
        );
        assert_eq!(answer_text(&tokenizer, roles, &encoded.tokens[6..]), "2");
    }
}
