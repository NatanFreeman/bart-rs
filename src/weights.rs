#[cfg(test)]
use gguf_rs::get_gguf_container;

#[test]
fn parses_gguf(){
    let mut container = get_gguf_container("bart-large-cnn/bart-large-cnn_f16.gguf").unwrap();
    let model = container.decode().unwrap();
    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());
    println!("Version: {}", model.get_version());
    println!("Metadata: {:?}", model.metadata());
    println!("Number of key value pairs: {}", model.num_kv());
    println!("Tensor count: {}", model.num_tensor());
    println!("First 10 tensors: {}", model.tensors()[..10].to_vec().iter().map(|x|format!("{x:?}")).collect::<Vec<String>>().join("\n"));
}