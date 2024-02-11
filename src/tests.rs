#![cfg(test)]
use gguf_rs::get_gguf_container;

use crate::{
    tensors::{only_zeros, rot90},
    weights::{get_token_embeddings, token_embeds_metadata},
};

#[test]
fn non_empty_token_embeddings() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(&model_path).unwrap();
    let model = container.decode().unwrap();
    let token_embeddings = get_token_embeddings(&model, &model_path).unwrap().unwrap();
    let rotated_embeddings = rot90(token_embeddings).unwrap();
    for i in 0..rotated_embeddings.shape().clone().into_dims()[0] {
        let embeddings = rotated_embeddings.get(i as usize).unwrap();
        if only_zeros(&embeddings).unwrap() {
            panic!("Embedding index {i} is empty!");
        }
    }
}

#[test]
fn list_tensors(){
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(&model_path).unwrap();
    let model = container.decode().unwrap();
    let mut tensors: Vec<_>=model.tensors().iter().collect();
    tensors.sort_by(|a,b|a.name.cmp(&b.name));
    for i in tensors{
        println!("{i:?}");
    }
}

#[test]
fn parses_gguf() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(&model_path).unwrap();
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
    let mut container = get_gguf_container(&model_path).unwrap();
    let model = container.decode().unwrap();
    token_embeds_metadata(&model).unwrap();
}

#[test]
fn parses_token_embeddings() {
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(&model_path).unwrap();
    let model = container.decode().unwrap();
    println!(
        "Embedding tensor: {:?}",
        get_token_embeddings(&model, &model_path).unwrap()
    );
}
