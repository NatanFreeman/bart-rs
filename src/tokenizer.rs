#![allow(dead_code)]

use std::{collections::HashMap, fs, path::Path};

#[derive(Clone, Copy)]
pub struct Token {
    id: u32,
}

impl Token {
    pub fn new(tokenizer: &WordPieceTokenizer, id: u32) -> Option<Self> {
        tokenizer.vocab.get(&id)?;
        Some(Self { id })
    }
    pub fn to_substr(&self, tokenizer: &WordPieceTokenizer) -> Option<Box<str>> {
        tokenizer
            .vocab
            .get(&self.id)
            .map(|x| x.to_owned().into_boxed_str())
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
        let vocab = vocab
            .iter()
            .map(|(token, id)| (*id, token.to_owned()))
            .collect();
        Ok(Self { vocab })
    }

    pub fn tokenize(&self, text: &str) -> Box<[Token]> {
        let mut tokens = Vec::new();

        for word in text.split(" ") {
            let mut start = 0;
            while start < word.len() {
                let longest = self
                    .vocab
                    .iter()
                    .filter(|(_key, value)| word[start..].starts_with(*value))
                    .max_by_key(|(_key, value)| value.len());

                if let Some(longest) = longest {
                    tokens.push(Token::new(self, *longest.0).unwrap());
                    start += longest.1.len();
                } else {
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
        }

        tokens.into()
    }
}
