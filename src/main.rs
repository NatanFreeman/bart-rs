mod tensors;
mod tests;
mod tokenizer;
mod weights;

use gguf_rs::get_gguf_container;
use half::f16;
use tokenizer::WordPieceTokenizer;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::tokenizer::InputSeqState;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let input_seq = InputSeqState::new("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration".into());
    let tokenizer = WordPieceTokenizer::new("bart-large-cnn/vocab.json")?;
    info!("Tokenizing");
    let input_seq = input_seq.tokenize(&tokenizer).unwrap();
    info!("Formatting tokens in the format BART expects");
    let input_seq = input_seq.format_for_bart(&tokenizer).unwrap();

    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    info!("Reading model from {model_path}");
    let mut container = get_gguf_container(&model_path)?;
    let model = container.decode()?;

    info!("Loading token embeddings tensor");
    let input_seq = input_seq.embed(&model, &model_path)?.unwrap();

    if let InputSeqState::Embeds(embeds) = input_seq {
        for i in embeds.into_iter() {
            println!(
                "{:?}",
                i.to_vec1::<f16>()?[..5].to_vec()
            );
        }
    }
    Ok(())
}
