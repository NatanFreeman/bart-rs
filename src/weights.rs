use candle_core::Shape;
use gguf_rs::GGUFModel;
use half::f16;
use std::fs::File;
use std::io;
use std::os::windows::fs::FileExt;
use std::path::Path;
use tracing::{debug, error};

use crate::tensors::tensor_from_floats;

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

/// Looks for the metadata for the embedding tensor in the GGUF file
pub fn token_embeds_metadata(model: &GGUFModel) -> Option<gguf_rs::Tensor> {
    model
        .tensors()
        .iter()
        .find(|x| x.name == "model.decoder.embed_tokens.weight")
        .map(|x| x.to_owned())
}

fn deserialize_floats(buffer: Box<[u8]>) -> Result<Box<[f16]>, bincode::Error> {
    let mut floats = Vec::new();
    let mut clamps=0;
    for i in 0..buffer.len() / 16 {
        let buf = &buffer[i * 16..i * 16 + 16];
        let mut f = bincode::deserialize::<f16>(buf)?;
        if f > f16::from_f32(1.0) {
            f = f16::from_f32(1.0);
            clamps+=1;
        }
        else if f < f16::from_f32(-1.0) {
            f = f16::from_f32(1.0);
            clamps+=1;
        }
        floats.push(f);
    }
    debug!("Read {} f16s from buffer", floats.len());
    if clamps>0{
        error!("{clamps} f16s were out of the range -1.0 to 1.0 and were clamped");
    }
    Ok(floats.into_boxed_slice())
}

pub fn get_token_embeddings<P: AsRef<Path>>(
    model: &GGUFModel,
    model_path: &P,
) -> Result<Option<candle_core::Tensor>, Error> {
    let tensor_metadata = token_embeds_metadata(&model).ok_or(Error::NoEmbeddingWeights)?;
    debug!("Embedding tensor metadata {tensor_metadata:?}");
    let mut embeddings_buffer = vec![0; tensor_metadata.size as usize * 8];
    {
        let model_file = File::open(model_path)?;
        model_file.seek_read(&mut embeddings_buffer, tensor_metadata.offset)?;
    }

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
