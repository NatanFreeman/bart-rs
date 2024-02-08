mod tensors;
mod tests;
mod tokenizer;
mod weights;

use candle_core::Shape;
use gguf_rs::get_gguf_container;
use half::f16;
use tensors::rot90;
use tokenizer::WordPieceTokenizer;
use weights::get_token_embeddings;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration";
    let tokenizer = WordPieceTokenizer::new("bart-large-cnn/vocab.json")?;
    let tokens = tokenizer.tokenize(text);

    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(&model_path)?;
    let model = container.decode()?;
    let token_embeddings = get_token_embeddings(&model, &model_path)?.unwrap();
    let token_embeddings=rot90(token_embeddings)?;
    for i in tokens.iter() {
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
