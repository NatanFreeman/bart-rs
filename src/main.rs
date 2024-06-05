#![allow(clippy::boxed_local)]
mod bart_tensor_type;
mod input;
mod tensors;
mod tests;
mod tokenizer;
mod weights;

use gguf_rs::get_gguf_container;
use half::f16;
use tokenizer::WordPieceTokenizer;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use weights::BartTensor;

use crate::bart_tensor_type::BartTensorType;
use crate::input::InputSeq;

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    if let Err(error) = run() {
        tracing::error!("{error}");
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = WordPieceTokenizer::new("bart-large-cnn/vocab.json")?;
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    info!("Reading model from {model_path}");
    let mut container = get_gguf_container(model_path)?;
    let model = container.decode()?;
    let token_embeds = BartTensor::new(&model, &model_path, BartTensorType::EmbedTokensWeights)?;
    let pos_embeds = BartTensor::new(&model, &model_path, BartTensorType::EmbedPositionWeights)?;
    let input_seq = InputSeq::new("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration".into());
    let input_seq = input_seq
        .tokenize(&tokenizer)
        .format_for_bart()
        .embed(&token_embeds.tensor)?
        .add_pos_embeds(&pos_embeds.tensor)?;

    for i in input_seq.get_embeds().iter() {
        println!("{:?}", i.to_vec1::<f16>()?[..5].to_vec());
    }
    Ok(())
}
