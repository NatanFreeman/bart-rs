use std::{collections::HashMap, path::Path};

#[derive(Clone, Copy)]
pub struct Token {
    id: u32,
}

impl Token {
    pub fn new(tokenizer: &WordPieceTokenizer, id: u32) -> Option<Self> {
        if tokenizer.vocab.get(&id).is_none() {
            return None;
        }
        Some(Self { id })
    }
    pub fn to_substr(&self, tokenizer: &WordPieceTokenizer) -> Option<Box<str>> {
        tokenizer
            .vocab
            .get(&self.id)
            .map(|x| x.to_owned().into_boxed_str())
    }
    pub fn get_id(&self)->u32{
        self.id
    }
}

pub struct WordPieceTokenizer {
    vocab: HashMap<u32, String>,
}

impl WordPieceTokenizer {
    pub fn new<T: AsRef<Path>>(vocab_file: T) -> Self {
        let mut vocab = HashMap::new();
        if let Ok(lines) = crate::utils::read_lines(vocab_file) {
            for (i, token) in lines.flatten().enumerate() {
                assert!(vocab.insert(i as u32, token).is_none());
            }
        }
        Self { vocab }
    }

    fn reformat_for_bert(text: &str)-> Box<str>{
        text.to_lowercase().replace(" ", "").into_boxed_str()
    }

    pub fn tokenize(&self, text: &str) -> Box<[Token]> {
        let text = Self::reformat_for_bert(text);
        let mut tokens = Vec::new();

        let mut start = 0;
        while start < text.len() {
            let longest = self
                .vocab
                .iter()
                .filter(|(_key, value)| text[start..].starts_with(*value))
                .max_by_key(|(_key, value)| value.len());

            if let Some(longest) = longest {
                if start > 0 {
                    tokens.push(Token::new(self, *longest.0).unwrap());
                } else {
                    tokens.push(Token::new(self, *longest.0).unwrap());
                }
                start += longest.1.len();
            } else {
                let unk = self
                    .vocab
                    .clone()
                    .into_iter()
                    .find(|(_, value)| value == "[UNK]")
                    .unwrap();
                tokens.push(Token::new(self, unk.0).unwrap());
                start += 1;
            }
        }

        tokens.into()
    }
}
