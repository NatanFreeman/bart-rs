use crate::tokenizer::Token;
use crate::WordPieceTokenizer;

use tracing::debug;
use tracing::warn;

use candle_core::Tensor;
pub const BART_MAX_SEQ_LEN: usize = 1024;

/// The state of an input sequence
#[derive(Clone, Default)]
pub struct InputSeq<T: InputData> {
    state: T,
}

trait InputData: Default + Clone {}

#[derive(Default, Clone)]
pub struct Empty {}
impl InputData for Empty {}

#[derive(Default, Clone)]
pub struct RawText(Box<str>);
impl InputData for RawText {}

#[derive(Default, Clone)]
pub struct Tokenized {
    tokens: Box<[Token]>,
    tokenizer: WordPieceTokenizer,
}
impl InputData for Tokenized {}

#[derive(Default, Clone)]
pub struct BartTokens(Box<[Token]>);
impl InputData for BartTokens {}

#[derive(Default, Clone)]
pub struct TokenEmbeddings(Box<[Tensor]>);
impl InputData for TokenEmbeddings {}

#[derive(Default, Clone)]
pub struct PositionedEmbeddings(Box<[Tensor]>);
impl InputData for PositionedEmbeddings {}

impl InputSeq<Empty> {
    pub fn new(text: Box<str>) -> InputSeq<RawText> {
        InputSeq {
            state: RawText(text),
            ..InputSeq::default()
        }
    }
}

impl InputSeq<RawText> {
    pub fn tokenize(self, tokenizer: &WordPieceTokenizer) -> InputSeq<Tokenized> {
        let mut tokens = Vec::new();
        let text = self.state.0.replace(' ', "Ġ");
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
        InputSeq {
            state: Tokenized {
                tokens: tokens.into(),
                tokenizer: tokenizer.clone(),
            },
            ..InputSeq::default()
        }
    }
}

impl InputSeq<Tokenized> {
    /// Formats the given tokens in the way BART was trained to process them
    pub fn format_for_bart(self) -> InputSeq<BartTokens> {
        debug!("Formatting input token sequence");
        let mut tokens = self.state.tokens.to_vec();
        tokens.insert(
            0,
            Token::from_substr(&self.state.tokenizer, "<s>").expect("<s> not found in vocab"),
        );
        tokens.push(
            Token::from_substr(&self.state.tokenizer, "</s>").expect("</s> not found in vocab"),
        );
        let padding_length = BART_MAX_SEQ_LEN - tokens.len();
        tokens.reserve(padding_length);
        for _ in 0..padding_length {
            tokens.push(
                Token::from_substr(&self.state.tokenizer, "<pad>")
                    .expect("<pad> not found in vocab"),
            );
        }

        InputSeq {
            state: BartTokens(tokens.into_boxed_slice()),
            ..Default::default()
        }
    }
}

impl InputSeq<BartTokens> {
    pub fn embed(
        self,
        embed_tensor: &candle_core::Tensor,
    ) -> Result<InputSeq<TokenEmbeddings>, candle_core::Error> {
        debug!("Assigning token embeddings");
        let mut embeds = Vec::with_capacity(self.state.0.len());
        for i in self.state.0.iter() {
            embeds.push(embed_tensor.get(i.get_id() as usize)?)
        }
        Ok(InputSeq {
            state: TokenEmbeddings(embeds.into_boxed_slice()),
            ..Default::default()
        })
    }
}
impl InputSeq<TokenEmbeddings> {
    pub fn add_pos_embeds(
        self,
        pos_embeds: &candle_core::Tensor,
    ) -> Result<InputSeq<PositionedEmbeddings>, candle_core::Error> {
        let mut comb_embeds = Vec::with_capacity(self.state.0.len());
        for (i, t) in self.state.0.iter().enumerate() {
            let with_pos = t + pos_embeds.get(i)?;
            comb_embeds.push(with_pos?);
        }
        Ok(InputSeq {
            state: PositionedEmbeddings(comb_embeds.into_boxed_slice()),
            ..Default::default()
        })
    }
}
impl InputSeq<PositionedEmbeddings> {
    pub fn get_embeds(&self) -> &Box<[candle_core::Tensor]> {
        &self.state.0
    }
}
