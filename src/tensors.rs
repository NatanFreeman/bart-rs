use std::{fs::File, path::Path, str::FromStr};

use candle_core::{quantized::{gguf_file, QTensor}, Device};

use crate::bart_tensor_type::TensorName;

pub struct Tensor {
    name: TensorName,
    tensor: QTensor,
}

impl Tensor {
    pub fn get_tensor(&self) -> &QTensor {
        &self.tensor
    }
}

pub struct BartTensors {
    tensors: gguf_file::Content,
    file: File,
}

impl BartTensors {
    pub fn new<P: AsRef<Path>>(gguf_path: &P) -> candle_core::Result<Self> {
        let mut file = std::fs::File::open(gguf_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        Ok(Self {
            tensors: content,
            file,
        })
    }
    pub fn get_tensor(&mut self, tensor_name: TensorName, device: &Device) -> QTensor {
        self.tensors
            .tensor(&mut self.file, &tensor_name.to_string(), device)
            .expect(&format!("tensor {tensor_name} not found in BartTensors"))
    }
}
