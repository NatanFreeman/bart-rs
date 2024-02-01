# transformer-rust
In this repo, I attempt to program the transformer from the [2017 paper "Attention Is All You Need"](https://arxiv.org/abs/1706.03762). I use Rust and huggingface's [candle](https://github.com/huggingface/candle) framework to achieve this. The goal is to fully understand how transformers work by coding one myself.

I specifically implement [BERT](https://huggingface.co/bert-large-uncased) in this project. I did this because I couldn't find the pre-trained model used in the original paper, and I'm pretty sure BERT uses the same architecture.

## Running
Clone the BERT repo:
```bash
git clone https://huggingface.co/bert-large-uncased
```
Run with `cargo`
```bash
cargo run --release
```