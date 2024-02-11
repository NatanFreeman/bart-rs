#![allow(dead_code)]
use candle_core::{Device, Shape};
use half::f16;
use tracing::warn;

pub fn tensor_from_floats(
    floats: Box<[f16]>,
    tensor_shape: Shape,
) -> Result<candle_core::Tensor, candle_core::Error> {
    let embedding_tensor =
        candle_core::Tensor::from_vec(floats.to_vec(), tensor_shape, &Device::Cpu)?;
    Ok(embedding_tensor)
}

/// Removes all dimensions from a tensor which a <= 1 in size
pub fn rem_scala_dims(
    tensor: candle_core::Tensor,
) -> Result<candle_core::Tensor, candle_core::Error> {
    let dimensions = tensor.shape().dims();
    let mut rem_dims = Vec::new();
    for i in dimensions {
        if *i > 1 {
            rem_dims.push(*i);
        }
    }
    let new_shape = Shape::from_dims(rem_dims.as_slice());
    if new_shape != *tensor.shape() {
        warn!(
            "Simplified tensor shape {:?} to {new_shape:?}",
            tensor.shape()
        );
    }
    tensor.reshape(new_shape)
}

/// Returns `true` if the given `candle_core::Tensor` is full of zeros
/// Fails if the `candle_core::Tensor` cannot be made one dimensional
pub fn only_zeros(tensor: &candle_core::Tensor) -> Result<bool, candle_core::Error> {
    for i in tensor.to_vec1::<f16>()? {
        if i != f16::from_f32(0.0) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn rot90(tensor: candle_core::Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
    let rotated_tensor = tensor.transpose(0, 1)?;
    let rotated_tensor = rem_scala_dims(rotated_tensor)?;
    Ok(rotated_tensor)
}
