#![cfg(test)]

use gguf_rs::get_gguf_container;

use crate::{bart_tensor_type::BartTensorType, tensors::only_zeros, weights::{gguf_tensor_metadata, BartTensor}};

#[test]
fn non_empty_token_embeddings() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(model_path).unwrap();
    let model = container.decode().unwrap();
    let token_embeds =
        BartTensor::new(&model, &model_path, BartTensorType::EmbedTokensWeights).unwrap();

    for i in 0..token_embeds.tensor.shape().clone().into_dims()[0] {
        let embeddings = token_embeds.tensor.get(i).unwrap();
        if only_zeros(&embeddings).unwrap() {
            panic!("Embedding index {i} is empty!");
        }
    }
}

#[test]
fn list_tensors() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(model_path).unwrap();
    let model = container.decode().unwrap();
    let mut tensors: Vec<_> = model.tensors().iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    for i in tensors {
        println!("{i:?}");
    }
}

#[test]
fn parses_gguf() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(model_path).unwrap();
    let model = container.decode().unwrap();
    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());
    println!("Version: {}", model.get_version());
    println!("Metadata: {:?}", model.metadata());
    println!("Number of key value pairs: {}", model.num_kv());
    println!("Tensor count: {}", model.num_tensor());
}

#[test]
fn token_embedding_metadata_found() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(model_path).unwrap();
    let model = container.decode().unwrap();
    gguf_tensor_metadata(&model, &BartTensorType::EmbedTokensWeights).unwrap();
}

#[test]
fn parses_token_embeddings() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(model_path).unwrap();
    let model = container.decode().unwrap();
    println!(
        "Embedding tensor: {:?}",
        BartTensor::new(&model, &model_path, BartTensorType::EmbedTokensWeights).unwrap()
    );
}
