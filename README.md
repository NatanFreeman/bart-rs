# bart-rs

In this repo, I attempt to program a transformer architecture introduced in the [2017 paper "Attention Is All You Need"](https://arxiv.org/abs/1706.03762). I use Rust and huggingface's [candle](https://github.com/huggingface/candle) framework to achieve this. The goal is to fully understand how transformers work by coding one myself.

I specifically implement [BART Large](https://huggingface.co/facebook/bart-large-cnn) in this project. I did this because I couldn't find the pre-trained model used in the original paper. The BART architecture is pretty similar as the focus of the paper was on pre-training. 

These are the differences:
- GeLU activation functions are used instead of ReLU
- BART Large's encoder and decoder have 12 layers each, double that of the original transformer
- BART inherits BERT's positional embeddings technic, instead of using trigonometric functions like the original transformer
- As consequence of using positional embeddings, input is padded using a `<pad>` token to fill the max size of 1024
There are more differences but I don't understand them yet

## Running

Clone the BART repo:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/facebook/bart-large-cnn
```

Run with `cargo`

```bash
cargo run --release
```
