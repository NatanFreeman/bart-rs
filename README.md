# bart-rs

This repository presents an implementation of a transformer architecture, as introduced in the seminal [2017 paper "Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The focus of this project is on implementing BART, which was proposed by Facebook AI researchers in the following paper:

- **BART Paper**: Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. (2019). [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)

In this project, I have implemented the BART model using Rust programming language and Hugging Face's [candle](https://github.com/huggingface/candle) framework. The goal is to gain a deep understanding of how transformers work by coding one from scratch.

## Project Overview 🔎

I chose to implement BART Large because I couldn't find the pre-trained model used in the original paper. The architecture I'm focusing on is quite similar to that of the original transformer, with these key differences:

- Using GeLU activation functions instead of ReLU for better gradient flow and improved performance.
- Both BART Large's encoder and decoder have 12 layers each, doubling the number of layers in the original transformer to enhance model complexity.
- Adopting BERT's positional embeddings technique instead of using trigonometric functions like the original transformer, allowing for longer input sequences.
- Due to the use of positional embeddings, input is padded with a `<pad>` token to reach the maximum size of 1026 tokens.

## Running the Project 🚀

To run this project:

1. Clone the BART repository:
   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   git clone https://huggingface.co/facebook/bart-large-cnn
   ```

2. Run with `cargo`:
   ```bash
   ▶️ cargo run --release
   ```
## Features 🌟

- **Rust Implementation 🚀**: This project provides an implementation of BART, written entirely in Rust for efficient and safe execution. Metal GPU acceleration support ensures even faster computations on Apple devices.
- **Candle Framework Integration 🔌**: The use of Hugging Face's Candle framework ensures compatibility with the larger ecosystem of ML tools available in Rust. It also supports GPU computations, enabling you to harness your hardware's full potential.

## Code Structure 📂

The project is divided into several files, each serving a specific purpose in the implementation of BART:

- `main.rs`: Initializes components, loads the pre-trained model, and performs input processing steps 🏠.
- `attn_head.rs`: Defines an attention head structure and methods for creating and accessing neural networks 🧠.
- `input.rs`: Manages input sequences at various stages of tokenization and embedding 🗃️.
- `tensors.rs`: Handles tensors loaded from the GGUF model file 📉.
- `bart_tensor_type.rs`: Defines enumerations for tensor types used in BART 🔠.
- `tokenizer.rs`: Implements a WordPieceTokenizer for converting text to tokens and vice versa 🔡.
- `attn.rs`: Contains the attention mechanism implementation 👀.

## Project Design Explanation 

The project is structured this way to ensure modularity, ease of use, and maintainability while accurately implementing the BART transformer architecture. Each file focuses on a specific aspect of the model, making it easier to understand and work with individual components. The design allows for easy extension and modification in the future if needed 🔄.

 Absolutely, I'd be happy to help with that. Here's how you can add a BibTeX entry and citation information for your project:

## Citing this Project in Your Work 📝

If you use `bart-rs` in your research paper or project, please consider citing the original BART paper and this repository as follows:

### BibTeX Entry

```bibtex
@misc{bart_rs,
  author = {Natan Freeman},
  title = {{bart-rs}: A Rust Implementation of the BART Transformer Architecture},
  year = {2023},
  howpublished = {\url{https://github.com/NatanFreeman/bart-rs}},
}
```

### Citation Info

Natan Freeman (Year). bart-rs: A Rust Implementation of the BART Transformer Architecture. GitHub repository, https://github.com/NatanFreeman/bart-rs.

You should also cite the original BART paper if you're using its architecture or data:

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2019. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv preprint arXiv:1910.13461 (https://arxiv.org/abs/1910.13461).

### Example BibTeX Entry for the Original BART Paper

```bibtex
@article{DBLP:journals/corr/abs-1910-13461,
  author    = {Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and Abdelrahman Mohamed and Omer Levy and Veselin Stoyanov and Luke Zettlemoyer},
  title     = {{BART:} Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension},
  journal   = {CoRR},
  volume    = {abs/1910.13461},
  year      = {2019},
  url       = {http://arxiv.org/abs/1910.13461},
  eprinttype = {arXiv},
  eprint    = {1910.13461},
  timestamp = {Thu, 31 Oct 2019 14:02:26 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1910-13461.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
