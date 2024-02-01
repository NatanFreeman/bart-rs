mod tokenizer;
mod utils;
use tokenizer::WordPieceTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration";
    let tokenizer=WordPieceTokenizer::new("vocab.txt");
    let tokens=tokenizer.tokenize(text);
    for i in tokens.iter(){
        print!(" {}", i.get_id());
    }
    Ok(())
}