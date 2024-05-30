use candle_core::Shape;
use gguf_rs::GGUFModel;
use half::f16;
use std::io;
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::{fmt, fs::File};
use tracing::{debug, error};

use crate::tensors::{rot90, tensor_from_floats};

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

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum TensorType {
    Bias,
    Weight,
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Bias => "bias",
                Self::Weight => "weight",
            }
        )
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum AttnType {
    Query,
    Key,
    Value,
}

impl fmt::Display for AttnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Query => "q_proj",
                Self::Key => "k_proj",
                Self::Value => "v_proj",
            }
        )
    }
}

#[derive(Clone, Debug)]
pub struct AttnLayer {
    pub attn_type: AttnType,
    pub tensor_type: TensorType,
    pub layer: usize,
}

impl fmt::Display for AttnLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = format!(
            "model.encoder.layers.{}.self_attn.{}.{}",
            self.layer, self.attn_type, self.tensor_type
        );
        write!(f, "{s}")
    }
}

#[derive(Clone, Debug)]
pub struct OutProjLayer {
    pub tensor_type: TensorType,
    pub layer: usize,
}

impl fmt::Display for OutProjLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = format!(
            "model.encoder.layers.{}.self_attn.out_proj.{}",
            self.layer, self.tensor_type
        );
        write!(f, "{s}")
    }
}
#[derive(Clone, Debug)]
pub enum BartTensorType {
    EmbedPositionWeights,
    EmbedTokensWeights,
    SelfAttn(AttnLayer),
    OutProj(OutProjLayer)
}

#[derive(Clone, Debug)]
pub struct BartTensor {
    t_type: BartTensorType,
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

    fn validate_input_embeds(&self) -> Result<(), Error> {
        let shape = self.tensor.shape();
        let expected_shape = match &self.t_type {
            BartTensorType::EmbedPositionWeights => Shape::from_dims(&[1026, 1024]),
            BartTensorType::EmbedTokensWeights => Shape::from_dims(&[50264, 1024]),
            BartTensorType::SelfAttn(attn) => {
                match attn.tensor_type{
                    TensorType::Bias => Shape::from_dims(&[1024]),
                    TensorType::Weight => Shape::from_dims(&[1024, 1024]),
                }
            },
            BartTensorType::OutProj(proj) => {
                match proj.tensor_type{
                    TensorType::Bias => Shape::from_dims(&[1024]),
                    TensorType::Weight => Shape::from_dims(&[1024, 1024]),
                }
            },
        };
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

impl fmt::Display for BartTensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbedPositionWeights => {
                write!(f, "model.decoder.embed_positions.weight")
            }
            Self::EmbedTokensWeights => write!(f, "model.decoder.embed_tokens.weight"),
            Self::SelfAttn(attn) => {
                write!(f, "{}", attn)
            },
            Self::OutProj(proj)=>{
                write!(f, "{}", proj)
            }
        }
    }
}

pub fn gguf_tensor_metadata(model: &GGUFModel, tensor: &BartTensorType) -> Option<gguf_rs::Tensor> {
    model
        .tensors()
        .iter()
        .find(|x| x.name == tensor.to_string())
        .map(|x| x.to_owned())
}

fn deserialize_floats(buffer: Box<[u8]>) -> Result<Box<[f16]>, bincode::Error> {
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
