use candle_core::Device;
use candle_core::Shape;
use gguf_rs::{get_gguf_container, GGUFModel};
use half::f16;
use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::os::windows::fs::FileExt;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to read tensor file")]
    Disconnect(#[from] io::Error),
    #[error("GGUF file doesn't define embedding weights")]
    NoEmbeddingWeights,
    #[error("Failed to deserialize tensor")]
    Deserialize(#[from] bincode::Error),
    #[error("Failed to build tensor from floats")]
    BuildingTensor(#[from] candle_core::Error),
}

fn get_token_embeddings_metadata(model: &GGUFModel) -> Option<gguf_rs::Tensor> {
    model
        .tensors()
        .iter()
        .find(|x| x.name == "model.decoder.embed_tokens.weight")
        .map(|x| x.to_owned())
}

fn deserialize_floats(buffer: Box<[u8]>) -> Result<Box<[f16]>, bincode::Error> {
    let mut floats = Vec::new();
    for i in 0..buffer.len() / 16 {
        let buf = &buffer[i * 16..i * 16 + 16];
        let mut f = bincode::deserialize::<f16>(buf)?;
        if f > f16::from_f32(1.0) {
            f = f16::from_f32(1.0);
        }
        floats.push(f);
    }
    Ok(floats.into_boxed_slice())
}

fn tensor_from_floats(
    floats: Box<[f16]>,
    tensor_shape: Shape,
) -> Result<candle_core::Tensor, candle_core::Error> {
    let mut total = 1;
    for i in tensor_shape.dims() {
        total = total * i;
    }
    let embedding_tensor =
        candle_core::Tensor::from_vec(floats.to_vec(), tensor_shape, &Device::Cpu)?;
    Ok(embedding_tensor)
}

fn get_token_embeddings<P: AsRef<Path>>(
    model: &GGUFModel,
    model_path: &P,
) -> Result<Option<candle_core::Tensor>, Error> {
    let tensor_metadata = get_token_embeddings_metadata(&model).ok_or(Error::NoEmbeddingWeights)?;
    let model_file = File::open(model_path)?;
    let mut embeddings_buffer = vec![0; tensor_metadata.size as usize * 8];
    model_file.seek_read(&mut embeddings_buffer, tensor_metadata.offset)?;
    let floats = deserialize_floats(embeddings_buffer.into_boxed_slice())?;
    let tensor = tensor_from_floats(
        floats,
        Shape::from_dims(
            tensor_metadata
                .shape
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>()
                .as_slice(),
        ),
    )?;
    Ok(Some(tensor))
}

#[test]
fn parses_input_embeddings(){
    let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
    let mut container = get_gguf_container(&model_path).unwrap();
    let model = container.decode().unwrap();
    println!(
        "Embedding tensor: {:?}",
        get_token_embeddings(&model, &model_path).unwrap()
    );
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
    get_token_embeddings_metadata(&model).unwrap();
}
