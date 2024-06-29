#![cfg(test)]

use gguf_rs::get_gguf_container;

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
