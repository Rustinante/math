use crate::tensor::{
    borrow_tensor::BorrowTensor, has_tensor_shape_data::HasTensorShapeData,
    AxisIndex,
};

pub trait MatrixTranspose<'a, Dtype: 'a>:
    HasTensorShapeData<Dtype> + BorrowTensor<'a, Dtype> {
    /// Reverses the axes.
    fn t(&'a self) -> <Self as BorrowTensor<'a, Dtype>>::Output {
        let transposed_axes: Vec<AxisIndex> =
            (0..self.shape().ndim()).into_iter().rev().collect();
        let shape_transpose = self.shape().to_transposed(transposed_axes);
        Self::create_borrowed_tensor(shape_transpose, &self.data())
    }

    /// # Arguments
    /// * `axes` - Must be the same length as `self.shape().ndim()`. For each
    ///   `i`, `axes[i] = j`
    /// means that the original `j`-th axis will be at the `i`-th axis in the
    /// new shape.
    fn transpose(
        &'a self,
        axes: Vec<AxisIndex>,
    ) -> <Self as BorrowTensor<'a, Dtype>>::Output {
        Self::create_borrowed_tensor(
            self.shape().to_transposed(axes),
            &self.data(),
        )
    }
}

impl<'a, Dtype: 'a, T> MatrixTranspose<'a, Dtype> for T where
    T: HasTensorShapeData<Dtype> + BorrowTensor<'a, Dtype>
{
}

#[cfg(test)]
mod tests {
    use crate::tensor::{
        ephemeral_view::EphemeralView, matrix_transpose::MatrixTranspose,
        tensor_shape::HasTensorShape, tensor_storage::IntoTensorStorage,
    };

    #[test]
    fn test_transpose() {
        {
            let storage = vec![1, 2, 3, 4, 5, 6].into_tensor_storage();
            let arr = EphemeralView::new(&storage, [2, 3]);
            let arr_t = arr.t();
            assert_eq!(arr_t.shape().dims(), vec![3, 2]);
            assert_eq!(arr_t.shape().strides(), vec![1, 3]);
            assert_eq!(arr_t.shape().ndim(), 2);
        }
        {
            // the original stride is (12, 3, 1)
            let storage = (0..24)
                .into_iter()
                .collect::<Vec<i32>>()
                .into_tensor_storage();
            let arr = EphemeralView::new(&storage, [2, 4, 3]);
            let arr_t = arr.t();
            assert_eq!(arr_t.shape().dims(), vec![3, 4, 2]);
            assert_eq!(arr_t.shape().strides(), vec![1, 3, 12]);
            assert_eq!(arr_t.shape().ndim(), 3);

            let arr_t01 = arr.transpose(vec![1, 0, 2]);
            assert_eq!(arr_t01.shape().dims(), vec![4, 2, 3]);
            assert_eq!(arr_t01.shape().strides(), vec![3, 12, 1]);
            assert_eq!(arr_t01.shape().ndim(), 3);
        }
        {
            // the original stride is (60, 15, 5, 1)
            let storage = (0..120)
                .into_iter()
                .collect::<Vec<i32>>()
                .into_tensor_storage();

            let arr = EphemeralView::new(&storage, [2, 4, 3, 5]);
            let arr_t = arr.transpose(vec![1, 3, 0, 2]);
            assert_eq!(arr_t.shape().dims(), vec![4, 5, 2, 3]);
            assert_eq!(arr_t.shape().strides(), vec![15, 1, 60, 5]);
            assert_eq!(arr_t.shape().ndim(), 4);
        }
    }
}
