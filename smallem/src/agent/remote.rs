//! Le même agent avec un grand modèle, servi par un serveur local qui parle
//! l'API de chat d'OpenAI (`llama-server`, LM Studio, vLLM…).
//!
//! La boucle ne change pas : le modèle demande un outil, le programme
//! l'exécute avec la même [`Toolbox`](super::tools::Toolbox), et rend le
//! résultat. Seul le format change : au lieu de tokens `<|appel|>` appris,
//! les outils sont décrits en JSON dans chaque requête, et le modèle répond
//! par des appels en JSON, que son modèle de conversation sait écrire.
//!
//! Le client HTTP tient en quelques lignes sur `TcpStream` : une requête
//! `POST`, `Connection: close`, et la réponse lue jusqu'au bout.

use std::error::Error;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::{Value, json};

use super::Ending;
use super::missions::Mission;
use super::tools::Tool;

/// Un serveur de modèle, à `address` (`127.0.0.1:8089`).
#[derive(Clone, Debug)]
pub struct Server {
    pub address: String,
    pub model: String,
}

impl Server {
    /// Envoie `body` en `POST` à `path`, et rend le JSON de la réponse.
    pub fn post(&self, path: &str, body: &Value) -> Result<Value, Box<dyn Error>> {
        let body = body.to_string();
        let mut stream = TcpStream::connect(&self.address)?;
        stream.set_read_timeout(Some(Duration::from_secs(600)))?;
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            self.address,
            body.len()
        )?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response)?;
        let (status, payload) = split_response(&response)?;
        if status != 200 {
            return Err(format!("HTTP {status} : {}", String::from_utf8_lossy(&payload)).into());
        }
        Ok(serde_json::from_slice(&payload)?)
    }
}

/// Le code d'une réponse HTTP et son corps, découpé s'il arrive par
/// morceaux (`Transfer-Encoding: chunked`).
fn split_response(response: &[u8]) -> Result<(u16, Vec<u8>), Box<dyn Error>> {
    let end = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("réponse HTTP sans en-têtes")?;
    let head = String::from_utf8_lossy(&response[..end]).to_string();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("code HTTP illisible")?;
    let body = &response[end + 4..];
    let chunked = head
        .lines()
        .any(|l| l.to_ascii_lowercase().starts_with("transfer-encoding:") && l.contains("chunked"));
    if !chunked {
        return Ok((status, body.to_vec()));
    }
    let mut out = Vec::new();
    let mut rest = body;
    loop {
        let line = rest
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("morceau sans taille")?;
        let size = usize::from_str_radix(String::from_utf8_lossy(&rest[..line]).trim(), 16)?;
        if size == 0 {
            break;
        }
        let start = line + 2;
        out.extend(&rest[start..start + size]);
        rest = &rest[start + size + 2..];
    }
    Ok((status, out))
}

/// La description JSON d'un outil, sous le nom que lui donne la mission.
pub fn tool_schema(tool: Tool, name: &str) -> Value {
    let integer = json!({"type": "integer"});
    let (description, properties, required) = match tool {
        Tool::Add => (
            "Adds two numbers.",
            json!({"a": integer, "b": integer}),
            json!(["a", "b"]),
        ),
        Tool::Mul => (
            "Multiplies two numbers.",
            json!({"a": integer, "b": integer}),
            json!(["a", "b"]),
        ),
        Tool::Reverse => (
            "Writes a word backwards.",
            json!({"word": {"type": "string"}}),
            json!(["word"]),
        ),
        Tool::Sort => (
            "Sorts numbers from smallest to largest.",
            json!({"numbers": {"type": "array", "items": integer}}),
            json!(["numbers"]),
        ),
        Tool::Lookup => (
            "Reads a fact about a person in the database: their age or their city.",
            json!({"name": {"type": "string"}, "fact": {"type": "string", "enum": ["age", "city"]}}),
            json!(["name", "fact"]),
        ),
    };
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }
    })
}

/// L'appel JSON du modèle, mis dans la forme texte que lit
/// [`Toolbox::run`](super::tools::Toolbox::run) : `add 4821 3907`.
pub fn call_text(name: &str, tool: Option<Tool>, arguments: &Value) -> String {
    let word = |key: &str| match &arguments[key] {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let args = match tool {
        Some(Tool::Add | Tool::Mul) => format!("{} {}", word("a"), word("b")),
        Some(Tool::Reverse) => word("word"),
        Some(Tool::Sort) => match &arguments["numbers"] {
            Value::Array(items) => items
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" "),
            other => other.to_string(),
        },
        Some(Tool::Lookup) => format!("{} {}", word("name"), word("fact")),
        None => String::new(),
    };
    format!("{name} {args}").trim().to_string()
}

/// Un épisode joué par le grand modèle.
#[derive(Clone, Debug)]
pub struct RemoteEpisode {
    pub calls: Vec<super::Call>,
    pub answer: Option<String>,
    pub ending: Ending,
    /// Tokens lus et écrits, d'après le serveur, sur tout l'épisode.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

/// La consigne système : la seule chose qui dise au grand modèle comment se
/// comporter. Le nôtre l'a apprise ; lui la lit.
pub const SYSTEM: &str = "You answer questions using the tools provided. Call tools when you need them. \
When you have the answer, reply with the answer alone: no sentence, no punctuation, no explanation.";

/// Joue une mission avec le grand modèle.
pub fn run_episode(
    server: &Server,
    mission: &Mission,
    max_calls: usize,
) -> Result<RemoteEpisode, Box<dyn Error>> {
    let mut toolbox = mission.toolbox.clone();
    let mut tools: Vec<Tool> = vec![Tool::Add, Tool::Reverse, Tool::Sort, Tool::Lookup];
    for tool in mission.task.tools() {
        if !tools.contains(&tool) {
            tools.push(tool);
        }
    }
    let schemas: Vec<Value> = tools
        .iter()
        .map(|&t| tool_schema(t, toolbox.name_of(t)))
        .collect();
    let mut messages = vec![
        json!({"role": "system", "content": SYSTEM}),
        json!({"role": "user", "content": mission.question}),
    ];
    let mut episode = RemoteEpisode {
        calls: Vec::new(),
        answer: None,
        ending: Ending::Length,
        prompt_tokens: 0,
        completion_tokens: 0,
    };
    for _ in 0..=max_calls + 1 {
        let response = server.post(
            "/v1/chat/completions",
            &json!({
                "model": server.model,
                "messages": messages,
                "tools": schemas,
                "temperature": 0,
                "max_tokens": 512,
            }),
        )?;
        episode.prompt_tokens += response["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        episode.completion_tokens += response["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        let message = response["choices"][0]["message"].clone();
        let calls = message["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() {
            let text = message["content"].as_str().unwrap_or("").trim().to_string();
            episode.answer = Some(text);
            episode.ending = Ending::Answer;
            return Ok(episode);
        }
        messages.push(message);
        for call in calls {
            if episode.calls.len() == max_calls {
                episode.ending = Ending::Budget;
                return Ok(episode);
            }
            let name = call["function"]["name"].as_str().unwrap_or("").to_string();
            let arguments: Value = match &call["function"]["arguments"] {
                Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
                other => other.clone(),
            };
            let text = call_text(&name, toolbox.tool(&name), &arguments);
            let result = toolbox.run(&text);
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            }));
            episode.calls.push(super::Call { text, result });
        }
    }
    Ok(episode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_are_read_whole_or_by_chunks() {
        let plain = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(split_response(plain).unwrap(), (200, b"{}".to_vec()));
        let chunked =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{\"a\r\n4\r\n\":1}\r\n0\r\n\r\n";
        assert_eq!(
            split_response(chunked).unwrap(),
            (200, b"{\"a\":1}".to_vec())
        );
        let error = b"HTTP/1.1 500 Internal\r\n\r\noops";
        assert_eq!(split_response(error).unwrap().0, 500);
    }

    #[test]
    fn json_calls_become_text_calls() {
        assert_eq!(
            call_text("add", Some(Tool::Add), &json!({"a": 4821, "b": 3907})),
            "add 4821 3907"
        );
        assert_eq!(
            call_text("sort", Some(Tool::Sort), &json!({"numbers": [81, 5]})),
            "sort 81 5"
        );
        assert_eq!(
            call_text(
                "lookup",
                Some(Tool::Lookup),
                &json!({"name": "Paul", "fact": "city"})
            ),
            "lookup Paul city"
        );
        assert_eq!(call_text("fly", None, &json!({})), "fly");
        let schema = tool_schema(Tool::Lookup, "brim");
        assert_eq!(schema["function"]["name"], "brim");
    }
}
