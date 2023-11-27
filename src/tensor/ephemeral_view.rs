use crate::tensor::{
    borrow_tensor::BorrowTensor,
    tensor_shape::{HasTensorShape, TensorShape},
    tensor_storage::{HasTensorData, TensorStorage},
};

/// # A View of the Underlying Referenced Data as a Particular Shape
/// The underlying `data` has to outlive the `EphemeralView` itself.
///
/// ## Examples
/// ```
/// use math::tensor::{
///     ephemeral_view::ToEphemeralView, tensor_shape::HasTensorShape,
///     tensor_storage::IntoTensorStorage,
/// };
///
/// let storage = vec![1, 2, 3, 4].into_tensor_storage();
/// let view = storage.as_shape(vec![4, 1]);
/// assert_eq!(view.shape().dims(), vec![4, 1]);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EphemeralView<'a, Dtype> {
    pub shape: TensorShape,
    pub data: &'a TensorStorage<Dtype>,
}

/// The implementer can be viewed as a tensor of `shape` through the `as_shape`
/// method. The resulting `EphemeralView` cannot outlive the original data
/// struct.
pub trait ToEphemeralView<'a, Dtype> {
    fn as_shape<S: Into<TensorShape>>(
        &'a self,
        shape: S,
    ) -> EphemeralView<'a, Dtype>;
}

impl<'a, Dtype> EphemeralView<'a, Dtype> {
    pub fn new<S: Into<TensorShape>>(
        data: &'a TensorStorage<Dtype>,
        shape: S,
    ) -> EphemeralView<'a, Dtype> {
        EphemeralView {
            shape: shape.into(),
            data,
        }
    }
}

impl<Dtype> HasTensorShape for EphemeralView<'_, Dtype> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
}

impl<Dtype> HasTensorData<Dtype> for EphemeralView<'_, Dtype> {
    fn data(&self) -> &TensorStorage<Dtype> {
        &self.data
    }
}

impl<'a, Dtype: 'a> BorrowTensor<'a, Dtype> for EphemeralView<'_, Dtype> {
    type Output = EphemeralView<'a, Dtype>;

    fn create_borrowed_tensor(
        shape: TensorShape,
        data_ref: &'a TensorStorage<Dtype>,
    ) -> Self::Output {
        EphemeralView::new(data_ref, shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{
        ephemeral_view::ToEphemeralView, tensor_shape::TensorShape,
        tensor_storage::IntoTensorStorage,
    };

    #[test]
    fn test_as_shape() {
        let data = vec![1, 2, 3, 4].into_tensor_storage();
        let view1 = data.as_shape(TensorShape::from([2, 2]));
        assert_eq!(view1.shape, TensorShape {
            dims_strides: vec![(2, 2), (2, 1)]
        });

        let view2 = data.as_shape(TensorShape {
            dims_strides: vec![(2, 2), (2, 1)],
        });
        assert_eq!(view1, view2);
        assert_eq!(view1.shape.dims(), vec![2, 2]);
        assert_eq!(view1.shape.strides(), vec![2, 1]);
        assert_eq!(view1.shape.ndim(), 2);
        assert_eq!(view2.shape.dims(), vec![2, 2]);
        assert_eq!(view2.shape.strides(), vec![2, 1]);
        assert_eq!(view2.shape.ndim(), 2);
    }
}
