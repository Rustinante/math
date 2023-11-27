use crate::tensor::{
    ephemeral_view::{EphemeralView, ToEphemeralView},
    tensor_shape::TensorShape,
};
use std::ops::{Index, IndexMut};

/// # An N-dimensional Tensor Storage
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorStorage<Dtype> {
    pub vec: Vec<Dtype>,
}

impl<Dtype> Index<usize> for TensorStorage<Dtype>
where
    Dtype: Copy,
{
    type Output = Dtype;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl<Dtype> IndexMut<usize> for TensorStorage<Dtype>
where
    Dtype: Copy,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vec[index]
    }
}

pub trait HasTensorData<Dtype> {
    fn data(&self) -> &TensorStorage<Dtype>;
}

pub trait IntoTensorStorage<Dtype> {
    fn into_tensor_storage(self) -> TensorStorage<Dtype>;
}

impl<Dtype> IntoTensorStorage<Dtype> for Vec<Dtype> {
    fn into_tensor_storage(self) -> TensorStorage<Dtype> {
        TensorStorage {
            vec: self,
        }
    }
}

impl<'a, Dtype> ToEphemeralView<'a, Dtype> for TensorStorage<Dtype> {
    fn as_shape<S: Into<TensorShape>>(
        &'a self,
        shape: S,
    ) -> EphemeralView<'a, Dtype> {
        let target_shape: TensorShape = shape.into();
        assert_eq!(
            target_shape.num_elements(),
            self.vec.len(),
            "number of elements in target shape mismatch"
        );
        EphemeralView {
            shape: target_shape,
            data: &self,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{
        ephemeral_view::ToEphemeralView, indexable_tensor::IndexableTensor,
        tensor_shape::HasTensorShape, tensor_storage::IntoTensorStorage,
    };

    #[test]
    fn test_to_ephemeral_view() {
        {
            let storage = vec![1, 2, 3, 4, 5, 6].into_tensor_storage();
            let view = storage.as_shape([2, 3]);
            let shape = view.shape();
            assert_eq!(shape.dims(), vec![2, 3]);
            assert_eq!(shape.strides(), vec![3, 1]);
            assert_eq!(shape.ndim(), 2);

            assert_eq!(view.at([0, 0]), 1);
            assert_eq!(view.at([0, 2]), 3);
            assert_eq!(view.at([1, 2]), 6);
        }
        {
            let storage = (0..24)
                .into_iter()
                .collect::<Vec<i32>>()
                .into_tensor_storage();
            let view = storage.as_shape([2, 3, 4]);
            let shape = view.shape();
            assert_eq!(shape.dims(), vec![2, 3, 4]);
            assert_eq!(shape.strides(), vec![12, 4, 1]);
            assert_eq!(shape.ndim(), 3);

            assert_eq!(view.at([0, 0, 0]), 0);
            assert_eq!(view.at([0, 0, 1]), 1);
            assert_eq!(view.at([0, 0, 2]), 2);
            assert_eq!(view.at([0, 0, 3]), 3);
            assert_eq!(view.at([0, 1, 0]), 4);
            assert_eq!(view.at([0, 1, 1]), 5);
            assert_eq!(view.at([0, 1, 2]), 6);
            assert_eq!(view.at([0, 1, 3]), 7);
            assert_eq!(view.at([0, 2, 3]), 11);

            assert_eq!(view.at([1, 0, 0]), 12);
            assert_eq!(view.at([1, 0, 1]), 13);
            assert_eq!(view.at([1, 0, 2]), 14);
            assert_eq!(view.at([1, 0, 3]), 15);
            assert_eq!(view.at([1, 1, 0]), 16);
            assert_eq!(view.at([1, 1, 1]), 17);
            assert_eq!(view.at([1, 1, 2]), 18);
            assert_eq!(view.at([1, 1, 3]), 19);
            assert_eq!(view.at([1, 2, 3]), 23);
        }
        {
            let storage = (0..30)
                .into_iter()
                .collect::<Vec<i32>>()
                .into_tensor_storage();
            let view = storage.as_shape(vec![2, 3, 5]);
            let shape = view.shape();
            assert_eq!(shape.dims(), vec![2, 3, 5]);
            assert_eq!(shape.strides(), vec![15, 5, 1]);
            assert_eq!(shape.ndim(), 3);
        }
    }
}
