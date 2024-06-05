use candle_core::Shape;
use derive_more::Display;
use std::fmt;

#[derive(Clone, Copy, Debug, Display)]
#[repr(u8)]
pub enum TensorType {
    #[display(fmt = "bias")]
    Bias,
    #[display(fmt = "weight")]
    Weight,
}

#[derive(Clone, Copy, Debug, Display)]
#[repr(u8)]
pub enum AttnType {
    #[display(fmt = "q_proj")]
    Query,
    #[display(fmt = "k_proj")]
    Key,
    #[display(fmt = "v_proj")]
    Value,
}

#[derive(Clone, Debug, Display)]
#[display(fmt = "model.encoder.layers.{}.self_attn.{}.{}", "layer", "attn_type", "tensor_type")]
pub struct AttnLayer {
    pub attn_type: AttnType,
    pub tensor_type: TensorType,
    pub layer: usize,
}

#[derive(Clone, Debug, Display)]
#[display(fmt = "model.encoder.layers.{}.self_attn.out_proj.{}", "layer", "tensor_type")]
pub struct OutProjLayer {
    pub tensor_type: TensorType,
    pub layer: usize,
}


#[derive(Clone, Debug, Display)]
pub enum BartTensorType {
    #[display(fmt = "model.decoder.embed_positions.weight")]
    EmbedPositionWeights,
    #[display(fmt = "model.decoder.embed_tokens.weight")]
    EmbedTokensWeights,
    #[display(fmt = "{}", _0)]
    SelfAttn(AttnLayer),
    #[display(fmt = "{}", _0)]
    OutProj(OutProjLayer),
}


impl BartTensorType {
    pub fn expected_shape(&self) -> Shape {
        match self {
            Self::EmbedPositionWeights => Shape::from_dims(&[1026, 1024]),
            Self::EmbedTokensWeights => Shape::from_dims(&[50264, 1024]),
            Self::SelfAttn(AttnLayer { tensor_type, .. })
            | Self::OutProj(OutProjLayer { tensor_type, .. }) => match tensor_type {
                TensorType::Bias => Shape::from_dims(&[1024]),
                TensorType::Weight => Shape::from_dims(&[1024, 1024]),
            },
        }
    }
}
