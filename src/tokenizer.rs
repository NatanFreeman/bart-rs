#![allow(dead_code)]

use std::{collections::HashMap, fs, path::Path};

use tracing::debug;

#[derive(Clone, Copy)]
pub struct Token {
    id: u32,
}

impl Token {
    pub fn from_substr(tokenizer: &WordPieceTokenizer, substr: &str) -> Option<Self> {
        let id: u32 = *tokenizer.vocab.iter().find(|x| x.1 == substr)?.0;
        Some(Self { id })
    }

    pub fn new(tokenizer: &WordPieceTokenizer, id: u32) -> Option<Self> {
        tokenizer.vocab.get(&id)?;
        Some(Self { id })
    }
    pub fn to_substr(self, tokenizer: &WordPieceTokenizer) -> Option<Box<str>> {
        tokenizer.vocab.get(&self.id).map(|x| {
            x.to_owned()
                .into_boxed_str()
                .replace('Ġ', "_")
                .into_boxed_str()
        })
    }
    pub fn get_id(&self) -> u32 {
        self.id
    }
}

#[derive(serde::Deserialize, Clone, Default)]
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

    pub fn get_vocab(&self) -> &HashMap<u32, String> {
        &self.vocab
    }
}
