use crate::{tensor::ephemeral_view::EphemeralView, traits::ToIterator};

pub struct TensorIter<'a, Dtype> {
    i: i64,
    num_elements: i64,
    tensor_view: EphemeralView<'a, Dtype>,
}

impl<'a, Dtype> TensorIter<'a, Dtype> {
    fn new(tensor: EphemeralView<'a, Dtype>) -> TensorIter<'a, Dtype> {
        TensorIter {
            i: 0,
            num_elements: tensor.shape.num_elements() as i64,
            tensor_view: tensor,
        }
    }
}

impl<'a, Dtype> Iterator for TensorIter<'a, Dtype>
where
    Dtype: Copy,
{
    type Item = Dtype;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.num_elements {
            None
        } else {
            let mut vec_index = 0;
            let mut index = self.i;
            for (len, stride) in
                self.tensor_view.shape.dims_strides.iter().rev()
            {
                vec_index += (index % len) * stride;
                index /= len;
            }
            self.i += 1;
            Some(self.tensor_view.data[vec_index as usize])
        }
    }
}

impl<'a, Dtype> ToIterator<'a, TensorIter<'a, Dtype>, Dtype>
    for EphemeralView<'a, Dtype>
where
    Dtype: Copy,
{
    fn to_iter(&'a self) -> TensorIter<'a, Dtype> {
        TensorIter::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tensor::{
            ephemeral_view::{EphemeralView, ToEphemeralView},
            matrix::Matrix,
            matrix_transpose::MatrixTranspose,
            tensor_storage::IntoTensorStorage,
        },
        traits::ToIterator,
    };

    #[test]
    fn test_tensor_iter() {
        let matrix =
            Matrix::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 4);
        let view = EphemeralView::from(&matrix);

        for (i, val) in view.to_iter().enumerate() {
            assert_eq!(val, i + 1);
        }

        {
            let transposed = view.t();
            for (val, expected) in transposed
                .to_iter()
                .zip(vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12].into_iter())
            {
                assert_eq!(val, expected);
            }
        }

        let t2 = (0..24)
            .into_iter()
            .collect::<Vec<i32>>()
            .into_tensor_storage();
        let t2_view = t2.as_shape([4, 3, 2]);

        // t2_102 has shape [3, 4, 2]
        let t2_102 = t2_view.transpose(vec![1, 0, 2]);
        for (val, expected) in t2_102.to_iter().zip(
            vec![
                0, 1, 6, 7, 12, 13, 18, 19, 2, 3, 8, 9, 14, 15, 20, 21, 4, 5,
                10, 11, 16, 17, 22, 23,
            ]
            .into_iter(),
        ) {
            assert_eq!(val, expected);
        }
    }
}
