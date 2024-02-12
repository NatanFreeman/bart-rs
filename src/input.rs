use crate::tokenizer::Token;
use crate::weights::BartTensor;
use crate::WordPieceTokenizer;

use tracing::debug;
use tracing::warn;

pub const BART_MAX_SEQ_LEN: usize = 1024;

/// The state of an input sequence
#[derive(Clone)]
pub struct InputSeq<'a> {
    pub tokens: Box<[Token]>,
    pub embeds: Box<[candle_core::Tensor]>,
    pub tokenizer: &'a WordPieceTokenizer,
}

impl<'a> InputSeq<'a> {
    pub fn new(
        text: Box<str>,
        tokenizer: &'a WordPieceTokenizer,
        token_embeds: BartTensor,
        pos_embeds: BartTensor,
    ) -> Result<Self, candle_core::Error> {
        let tokens = Self::tokenize(&text, tokenizer);
        let tokens = Self::format_for_bart(tokens, tokenizer);
        let embeds = Self::embed(&tokens, token_embeds.get_tensor())?;
        let embeds = Self::add_pos_embeds(embeds, pos_embeds.get_tensor())?;
        Ok(Self {
            tokens,
            embeds,
            tokenizer,
        })
    }

    fn tokenize(text: &str, tokenizer: &WordPieceTokenizer) -> Box<[Token]> {
        let mut tokens = Vec::new();
        let text = text.replace(' ', "Ġ");
        let mut start = 0;
        while start < text.len() {
            let longest = tokenizer
                .get_vocab()
                .iter()
                .filter(|(_key, value)| text[start..].starts_with(*value))
                .max_by_key(|(_key, value)| value.len());

            if let Some(longest) = longest {
                tokens.push(Token::new(tokenizer, *longest.0).unwrap());
                start += longest.1.len();
            } else {
                warn!("Unrecognized text sequence. Inserting <unk> token");
                let unk = tokenizer
                    .get_vocab()
                    .clone()
                    .into_iter()
                    .find(|(_, value)| value == "<unk>")
                    .unwrap();
                tokens.push(Token::new(tokenizer, unk.0).unwrap());
                start += 1;
            }
        }
        debug!("Tokenized text into {} tokens", tokens.len());
        tokens.into()
    }

    /// Formats the given tokens in the way BART was trained to process them
    fn format_for_bart(tokens: Box<[Token]>, tokenizer: &WordPieceTokenizer) -> Box<[Token]> {
        debug!("Formatting input token sequence");
        let mut tokens = tokens.to_vec();
        tokens.insert(
            0,
            Token::from_substr(tokenizer, "<s>").expect("<s> not found in vocab"),
        );
        tokens.push(Token::from_substr(tokenizer, "</s>").expect("</s> not found in vocab"));
        let padding_length = BART_MAX_SEQ_LEN - tokens.len();
        tokens.reserve(padding_length);
        for _ in 0..padding_length {
            tokens.push(Token::from_substr(tokenizer, "<pad>").expect("<pad> not found in vocab"));
        }
        tokens.into_boxed_slice()
    }

    fn embed(
        tokens: &[Token],
        embed_tensor: &candle_core::Tensor,
    ) -> Result<Box<[candle_core::Tensor]>, candle_core::Error> {
        debug!("Assigning token embeddings");
        let mut embeds = Vec::with_capacity(tokens.len());
        for i in tokens.iter() {
            embeds.push(embed_tensor.get(i.get_id() as usize)?)
        }
        Ok(embeds.into_boxed_slice())
    }

    fn add_pos_embeds(
        embeds: Box<[candle_core::Tensor]>,
        pos_embeds: &candle_core::Tensor,
    ) -> Result<Box<[candle_core::Tensor]>, candle_core::Error> {
        let mut comb_embeds = Vec::with_capacity(embeds.len());
        for (i, t) in embeds.iter().enumerate() {
            let with_pos = t + pos_embeds.get(i)?;
            comb_embeds.push(with_pos?);
        }
        Ok(comb_embeds.into_boxed_slice())
    }
}
