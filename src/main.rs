mod tensors;
mod tests;
mod tokenizer;
mod weights;

use gguf_rs::get_gguf_container;
use half::f16;
use tokenizer::WordPieceTokenizer;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::{tokenizer::InputSeq, weights::get_token_embeds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let tokenizer = WordPieceTokenizer::new("bart-large-cnn/vocab.json")?;
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    info!("Reading model from {model_path}");
    let mut container = get_gguf_container(&model_path)?;
    let model = container.decode()?;
    let token_embeds = get_token_embeds(&model, &model_path)?.unwrap();
    let input_seq = InputSeq::new("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration".into(),&tokenizer, token_embeds)?;

    for i in input_seq.embeds.into_iter() {
        println!("{:?}", i.to_vec1::<f16>()?[..5].to_vec());
    }
    Ok(())
}
