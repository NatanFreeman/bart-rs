use std::ops::Add;

use crate::{
    attn_head::AttnHead,
    input::{InputData, InputSeq, PositionedEmbeddings},
};
use candle_core::{quantized::QTensor, Device, Shape, Tensor};
use itertools::Itertools;
use tracing::debug;

pub struct Encoded {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

impl InputData for Encoded {}
impl Encoded {
    pub fn new(q: Tensor, k: Tensor, v: Tensor) -> Self {
        Self { q, k, v }
    }
}
/// This function is needed to reshape the 1D bias tensor into a 2D tensor
/// where each row corresponds to a row in the input tensor. The purpose of this is to
/// perform an element-wise addition (or broadcasted operation) between the input tensor and the bias tensor.
/// This function takes three arguments: the input tensor, the bias tensor (which is 1D), and the device on which the computation will be performed.
/// It returns a new Tensor that has the same shape as the input tensor but contains the broadcasted bias values.
fn stack_1d_tensor(
    shape: &Shape,
    tensor: &QTensor,
    device: &Device,
) -> candle_core::Result<Tensor> {
    debug!("Reshaping bias {:?} to be 2D", tensor.shape());
    let bias_2d = tensor.dequantize_f16(device)?.unsqueeze(0)?;
    debug!("Filling 2D bias {:?}", bias_2d.shape());
    // Broadcast the 2D bias tensor to have the same shape as the input tensor for element-wise addition
    let full_bias = bias_2d.broadcast_as(shape)?;
    Ok(full_bias)
}

impl AttnHead {
    fn encode(
        &self,
        input: InputSeq<PositionedEmbeddings>,
        device: &Device,
    ) -> candle_core::Result<InputSeq<Encoded>> {
        let input_embeds = input.get_embeds().to_dtype(candle_core::DType::F16)?;

        // Perform matrix multiplication and add bias for each of q, k, v
        let (q, k, v) = [self.get_q(), self.get_k(), self.get_v()]
            .map(|x| {
                debug!(
                    "multiplying input embeds {:?} with weights {:?}",
                    input_embeds.shape(),
                    x.weights.shape()
                );
                let tensor = input_embeds.matmul(&x.weights.dequantize_f16(device)?)?;
                let bias = stack_1d_tensor(tensor.shape(), &x.bias, device)?;

                println!(
                    "adding bias {:?} to matmul {:?}",
                    bias.shape(),
                    tensor.shape()
                );
                tensor.add(&bias)
                //TODO: add bias to every row of 2D matmul result
            })
            .into_iter()
            .collect_tuple()
            .unwrap();

        Ok(InputSeq::<Encoded> {
            state: Encoded::new(q?, k?, v?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::assertions;
    use crate::utils::assertions::Print;
    use crate::TensorName;
    use crate::WordPieceTokenizer;
    use candle_core::Shape;
    use candle_core::{Device, Tensor};
    use half::f16;

    #[test]
    fn test_stack() {
        let device = Device::new_metal(0).unwrap();
        let t = Tensor::from_iter((0..10).map(|i| i as f32), &device).unwrap();
        let t = QTensor::quantize(&t, candle_core::quantized::GgmlDType::F16).unwrap();

        let stacked = stack_1d_tensor(&Shape::from_dims(&[10, 10]), &t, &device).unwrap();
        stacked.print().unwrap();
        let data = [
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
            [
                f16::from_f32(0.0f32),
                f16::from_f32(1.0f32),
                f16::from_f32(2.0f32),
                f16::from_f32(3.0f32),
                f16::from_f32(4.0f32),
                f16::from_f32(5.0f32),
                f16::from_f32(6.0f32),
                f16::from_f32(7.0f32),
                f16::from_f32(8.0f32),
                f16::from_f32(9.0f32),
            ],
        ];

        let expected = Tensor::new(&data, &device).unwrap();

        expected.print().unwrap();
        let cmp = assertions::tensors_equal(&stacked, &expected).unwrap();
        assert!(cmp);
    }

    #[test]
    fn encodes() {
        let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
        let mut tensors = crate::tensors::BartTensors::new(&model_path).unwrap();
        let device = candle_core::Device::new_metal(0).unwrap();
        let tokenizer = WordPieceTokenizer::new("bart-large-cnn/vocab.json").unwrap();

        let token_embeds = tensors.get_tensor(TensorName::EmbedTokensWeights, &device);
        let pos_embeds = tensors.get_tensor(TensorName::EmbedPositionWeights, &device);
        let input_seq = InputSeq::new("The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration".into());
        let input_seq = input_seq
            .tokenize(&tokenizer)
            .format_for_bart()
            .embed(&token_embeds.dequantize(&device).unwrap())
            .unwrap()
            .add_pos_embeds(&pos_embeds.dequantize(&device).unwrap())
            .unwrap();

        for i in 0..12 {
            let attn_head = AttnHead::new(i, &mut tensors, &device).unwrap();
            attn_head.encode(input_seq.clone(), &device).unwrap();
        }
    }
}
