#![allow(clippy::boxed_local)]
mod bart_tensor_type;
mod input;
mod tensors;
mod tests;
mod tokenizer;

use candle_core::Device;
use half::f16;
use tensors::BartTensors;
use tokenizer::WordPieceTokenizer;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::bart_tensor_type::TensorName;
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
    let mut tensors = BartTensors::new(&model_path)?;
    let device = Device::new_metal(0)?;

    let token_embeds = tensors.get_tensor(TensorName::EmbedTokensWeights, &device);
    let pos_embeds = tensors.get_tensor(TensorName::EmbedPositionWeights, &device);
    let input_seq = InputSeq::new("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration".into());
    let input_seq = input_seq
        .tokenize(&tokenizer)
        .format_for_bart()
        .embed(&token_embeds.dequantize(&device)?)? //TODO: remove dequantization
        .add_pos_embeds(&pos_embeds.dequantize(&device)?)?;

    for i in input_seq.get_embeds().iter() {
        println!("{:?}", i.to_vec1::<f32>()?[..5].to_vec());
    }
    Ok(())
}
