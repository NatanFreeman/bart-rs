use candle_core::Shape;
use gguf_rs::GGUFModel;
use half::f16;
use std::{fs::File, io, os::windows::fs::FileExt, path::Path};

use crate::{
    bart_tensor_type::{BartTensorType, TensorType},
    tensors::{rot90, tensor_from_floats},
};
use tracing::{debug, error};

#[derive(Clone, Debug)]
pub struct BartTensor {
    pub(crate) t_type: BartTensorType,
    pub tensor: candle_core::Tensor,
}

impl BartTensor {
    pub fn new<P: AsRef<Path>>(
        model: &GGUFModel,
        model_path: &P,
        t_type: BartTensorType,
    ) -> Result<Self, Error> {
        let tensor_metadata =
            gguf_tensor_metadata(model, &t_type).ok_or(Error::MissingTensor(t_type.clone()))?;
        println!("Tensor metadata {tensor_metadata:?}");
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
        let tensor = rot90(tensor)?;

        let ret = Self { t_type, tensor };
        Self::validate_input_embeds(&ret)?;
        Ok(ret)
    }

    pub(crate) fn validate_input_embeds(&self) -> Result<(), Error> {
        let expected_shape = self.t_type.expected_shape();
        let shape = self.tensor.shape();
        if *shape != expected_shape {
            return Err(Error::UnexpectedTensorDims(
                self.t_type.clone(),
                shape.clone(),
                expected_shape,
            ));
        }
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to read tensor file")]
    Disconnect(#[from] io::Error),
    #[error("Tensor {0} not found in GGUF file")]
    MissingTensor(BartTensorType),
    #[error("Failed to deserialize tensor")]
    Deserialize(#[from] bincode::Error),
    #[error("Failed to build tensor from floats")]
    BuildingTensor(#[from] candle_core::Error),
    #[error("tensor {0:?} got an unexpected dimension shape {1:?}. Expected to get {2:?}")]
    UnexpectedTensorDims(BartTensorType, Shape, Shape),
}

pub fn gguf_tensor_metadata(model: &GGUFModel, tensor: &BartTensorType) -> Option<gguf_rs::Tensor> {
    model
        .tensors()
        .iter()
        .find(|x| x.name == tensor.to_string())
        .map(|x| x.to_owned())
}

pub fn deserialize_floats(buffer: Box<[u8]>) -> Result<Box<[f16]>, bincode::Error> {
    let mut floats = Vec::new();
    let mut clamps = 0;
    for i in 0..buffer.len() / 16 {
        let buf = &buffer[i * 16..i * 16 + 16];
        let mut f = bincode::deserialize::<f16>(buf)?;
        if f > f16::from_f32(1.0) {
            f = f16::from_f32(1.0);
            clamps += 1;
        } else if f < f16::from_f32(-1.0) {
            f = f16::from_f32(-1.0);
            clamps += 1;
        }
        floats.push(f);
    }
    debug!("Read {} f16s from buffer", floats.len());
    if clamps > 0 {
        error!("{clamps} f16s were out of the range -1.0 to 1.0 and were clamped");
    }
    Ok(floats.into_boxed_slice())
}
