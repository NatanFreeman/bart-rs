#![allow(clippy::boxed_local)]
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

use crate::{
    input::InputSeq,
    weights::{BartTensor, BartTensorVar},
};

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
    let token_embeds = BartTensor::new(&model, &model_path, BartTensorVar::EmbedTokensWeights)?;
    let pos_embeds = BartTensor::new(&model, &model_path, BartTensorVar::EmbedPositionWeights)?;
    let input_seq = InputSeq::new("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration".into(),&tokenizer, token_embeds, pos_embeds)?;

    for i in input_seq.embeds.iter() {
        println!("{:?}", i.to_vec1::<f16>()?[..5].to_vec());
    }
    Ok(())
}
