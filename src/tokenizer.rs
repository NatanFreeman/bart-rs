#![allow(dead_code)]

use std::{collections::HashMap, fs, path::Path};
use tracing::{debug, warn};

#[derive(Clone, Copy)]
pub struct Token {
    id: u32,
}

const BART_MAX_SEQ_LEN: usize=1024;

impl Token {
    pub fn from_substr(tokenizer: &WordPieceTokenizer, substr: &str)->Option<Self>{
        let id: u32=*tokenizer.vocab.iter().find(|x|x.1==substr)?.0;
        Some(Self{id})
    }

    pub fn new(tokenizer: &WordPieceTokenizer, id: u32) -> Option<Self> {
        tokenizer.vocab.get(&id)?;
        Some(Self { id })
    }
    pub fn to_substr(&self, tokenizer: &WordPieceTokenizer) -> Option<Box<str>> {
        tokenizer
            .vocab
            .get(&self.id)
            .map(|x| x.to_owned().into_boxed_str().replace("Ġ", "_").into_boxed_str())
    }
    pub fn get_id(&self) -> u32 {
        self.id
    }
}

#[derive(serde::Deserialize)]
pub struct WordPieceTokenizer {
    vocab: HashMap<u32, String>,
}

impl WordPieceTokenizer {
    pub fn new<T: AsRef<Path>>(vocab_path: T) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&contents)?;
        let vocab: HashMap<u32, String> = vocab
            .iter()
            .map(|(token, id)| (*id, token.to_owned()))
            .collect();
        debug!("Loaded vocabulary of {} tokens", vocab.len());
        Ok(Self { vocab })
    }

    pub fn tokenize(&self, text: &str) -> Box<[Token]> {
        let mut tokens = Vec::new();
        let text=text.replace(" ", "Ġ");
            let mut start = 0;
            while start < text.len() {
                let longest = self
                    .vocab
                    .iter()
                    .filter(|(_key, value)| text[start..].starts_with(*value))
                    .max_by_key(|(_key, value)| value.len());

                if let Some(longest) = longest {
                    tokens.push(Token::new(self, *longest.0).unwrap());
                    start += longest.1.len();
                } else {
                    warn!("Unrecognized text sequence. Inserting <unk> token");
                    let unk = self
                        .vocab
                        .clone()
                        .into_iter()
                        .find(|(_, value)| value == "<unk>")
                        .unwrap();
                    tokens.push(Token::new(self, unk.0).unwrap());
                    start += 1;
                }
            }
        debug!("Tokenized text into {} tokens", tokens.len());
        tokens.into()
    }

    /// Formats the given tokens in the way BART was trained to process them
    pub fn format_for_bart(&self, tokens: Box<[Token]>)->Option<Box<[Token]>>{
        let mut tokens=tokens.to_vec();
        tokens.insert(0, Token::from_substr(self, "<s>")?);
        tokens.push(Token::from_substr(self, "</s>")?);
        let padding_length=BART_MAX_SEQ_LEN-tokens.len();
        tokens.reserve(padding_length);
        for _ in 0..padding_length{
            tokens.push(Token::from_substr(self, "<pad>")?);
        }
        Some(tokens.into_boxed_slice())
    }
}
