use crate::tensor::{
    borrow_tensor::BorrowTensor,
    matrix::{Matrix, MatrixTrait},
    tensor_shape::{HasTensorShape, TensorShape},
    tensor_storage::{HasTensorData, TensorStorage},
    Unitless,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixView<'a, Dtype> {
    shape: TensorShape,
    data: &'a TensorStorage<Dtype>,
}

impl<'a, Dtype: 'a> BorrowTensor<'a, Dtype> for Matrix<Dtype> {
    type Output = MatrixView<'a, Dtype>;

    fn create_borrowed_tensor(
        shape: TensorShape,
        data: &'a TensorStorage<Dtype>,
    ) -> Self::Output {
        MatrixView {
            shape,
            data,
        }
    }
}

impl<'a, Dtype> HasTensorData<Dtype> for MatrixView<'a, Dtype> {
    fn data(&self) -> &TensorStorage<Dtype> {
        &self.data
    }
}

impl<'a, Dtype> HasTensorShape for MatrixView<'a, Dtype> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
}

impl<'a, Dtype> MatrixTrait<Dtype> for MatrixView<'a, Dtype> {
    fn num_rows(&self) -> Unitless {
        self.shape.dims_strides[0].0
    }

    fn num_columns(&self) -> Unitless {
        self.shape.dims_strides[1].0
    }
}
