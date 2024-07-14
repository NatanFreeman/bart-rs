#[cfg(test)]
pub mod assertions {
    use candle_core::Tensor;
    use half::f16;
    use num_traits::Float;
    pub trait Print {
        fn print(&self) -> candle_core::Result<()>;
    }
    impl Print for Tensor {
        fn print(&self) -> candle_core::Result<()> {
            for y in 0..self.get(0)?.elem_count() {
                let row = self.get(y)?;
                let dims=row.dims().len() ;
                if dims == 0 {
                    print!("{:?} ", row.to_vec0::<f16>()?);
                } else if dims == 1 {
                    for x in 0..row.elem_count() {
                        let elem = row.get(x)?;
                        print!("{:?} ", elem.to_vec0::<f16>()?);
                    }
                }
                else{
                    panic!("got a tensor with unexpected number of dimensions {dims}")
                }
                println!();
            }
            Ok(())
        }
    }

    pub fn tensors_equal(a: &Tensor, b: &Tensor) -> candle_core::Result<bool> {
        if a.shape() != b.shape() {
            return Ok(false);
        }
        for i in 0..a.get(0)?.elem_count() {
            let a_elem = a.get(i)?;
            let b_elem = b.get(i)?;

            if let (Ok(a_elem), Ok(b_elem)) = (a.to_vec0::<f16>(), b.to_vec0::<f16>()) {
                if (a_elem - b_elem).abs() > f16::EPSILON {
                    return Ok(false);
                }
            } else {
                if tensors_equal(&a_elem, &b_elem)? == false {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}
