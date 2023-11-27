use crate::tensor::{tensor_shape::TensorShape, tensor_storage::TensorStorage};

pub trait BorrowTensor<'a, Dtype> {
    type Output;

    fn create_borrowed_tensor(
        shape: TensorShape,
        data: &'a TensorStorage<Dtype>,
    ) -> Self::Output;
}
