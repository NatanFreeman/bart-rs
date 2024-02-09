mod tensors;
mod tests;
mod tokenizer;
mod weights;

use gguf_rs::get_gguf_container;
use half::f16;
use tensors::rot90;
use tokenizer::WordPieceTokenizer;
use weights::get_token_embeddings;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    let text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration";
    let tokenizer = WordPieceTokenizer::new("bart-large-cnn/vocab.json")?;
    info!("Tokenizing \"{text}\"");
    let tokens = tokenizer.tokenize(text);
    info!("Formatting tokens in the format BART expects");
    let input=tokenizer.format_for_bart(tokens).unwrap();

    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    info!("Reading model from {model_path}");
    let mut container = get_gguf_container(&model_path)?;
    let model = container.decode()?;
    info!("Loading token embeddings tensor");
    let token_embeddings = get_token_embeddings(&model, &model_path)?.unwrap();
    let token_embeddings=rot90(token_embeddings)?;
    for i in input.iter() {
        let embeddings = token_embeddings.get(i.get_id() as usize)?;

        println!(
            "{} {}: {:?}",
            i.to_substr(&tokenizer).unwrap(),
            i.get_id(),
            embeddings.to_vec1::<f16>()?[..5].to_vec()
        );
    }
    Ok(())
}
