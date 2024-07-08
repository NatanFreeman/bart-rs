use crate::tokenizer::Token;
use crate::WordPieceTokenizer;

use tracing::debug;
use tracing::warn;

use candle_core::Tensor;
pub const BART_MAX_SEQ_LEN: usize = 1026;

/// The state of an input sequence
#[derive(Clone, Default)]
pub struct InputSeq<T: InputData> {
    pub state: T,
}

pub trait InputData{}

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

#[derive(Clone)]
pub struct TokenEmbeddings(Tensor);
impl Default for TokenEmbeddings {
    fn default() -> Self {
        let device = candle_core::Device::Cpu;
        Self(Tensor::new::<&[f32; 0]>(&[], &device).unwrap())
    }
}
impl InputData for TokenEmbeddings {}

#[derive(Clone)]
pub struct PositionedEmbeddings(Tensor);
impl Default for PositionedEmbeddings {
    fn default() -> Self {
        let device = candle_core::Device::Cpu;
        Self(Tensor::new::<&[f32; 0]>(&[], &device).unwrap())
    }
}
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

        let indices = candle_core::Tensor::from_vec(
            self.state.0.iter().map(|token| token.get_id() as u32).collect(),
            (self.state.0.len(),),
            embed_tensor.device()
        )?;

        let embeds = embed_tensor.index_select(&indices, 0)?;

        Ok(InputSeq {
            state: TokenEmbeddings(embeds),
            ..Default::default()
        })
    }
}

impl InputSeq<TokenEmbeddings> {
    pub fn add_pos_embeds(
        self,
        pos_embeds: &candle_core::Tensor,
    ) -> Result<InputSeq<PositionedEmbeddings>, candle_core::Error> {
        let comb_embeds = (&self.state.0 + pos_embeds)?;

        Ok(InputSeq {
            state: PositionedEmbeddings(comb_embeds),
            ..Default::default()
        })
    }
}
impl InputSeq<PositionedEmbeddings> {
    pub fn get_embeds(&self) -> &candle_core::Tensor {
        &self.state.0
    }
}
