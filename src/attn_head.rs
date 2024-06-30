use std::path::Path;

use candle_core::{quantized::QTensor, Device};

use crate::{
    bart_tensor_type::{AttnLayer, AttnType, TensorName, TensorType},
    tensors::BartTensors,
};

struct NeuralNet {
    bias: QTensor,
    weights: QTensor,
}
struct AttnHead {
    q_layer: NeuralNet,
    k_layer: NeuralNet,
    v_layer: NeuralNet,
}

impl AttnHead {
    fn new(
        layer: usize,
        tensors: &mut BartTensors,
        device: &Device,
    ) -> Result<Self, ()> {
        let mut attn_tensors = Vec::with_capacity(6);

        let attns = [AttnType::Query, AttnType::Key, AttnType::Value];
        let tensors_types = [TensorType::Bias, TensorType::Weight];
        for a in attns {
            for t in tensors_types {
                let tensor = tensors.get_tensor(
                    TensorName::SelfAttn(AttnLayer {
                        attn_type: a,
                        tensor_type: t,
                        layer,
                    }),
                    device,
                );
                attn_tensors.push(tensor);
            }
        }

        return Ok(AttnHead {
            q_layer: NeuralNet {
                bias: attn_tensors.remove(0),
                weights: attn_tensors.remove(0),
            },
            k_layer: NeuralNet {
                bias: attn_tensors.remove(0),
                weights: attn_tensors.remove(0),
            },
            v_layer: NeuralNet {
                bias: attn_tensors.remove(0),
                weights: attn_tensors.remove(0),
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;

    use crate::attn_head::AttnHead;

    #[test]
    fn loads_layers() {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(Level::TRACE)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");

        let model_path = "bart-large-cnn/bart-large-cnn_f16.gguf";
        let mut tensors = crate::tensors::BartTensors::new(&model_path).unwrap();
        let device = candle_core::Device::new_metal(0).unwrap();

        for i in 0..12 {
            println!("loading layer {i}");
            AttnHead::new(i, &mut tensors, &device).unwrap();
        }
    }
}
