use crate::tensor::{
    tensor_shape::HasTensorShape, tensor_storage::HasTensorData, Unitless,
};

pub trait HasTensorShapeData<Dtype>:
    HasTensorShape + HasTensorData<Dtype> {
    fn coord_to_index(&self, coord: &[Unitless]) -> Unitless {
        assert_eq!(
            coord.len(),
            self.shape().dims_strides.len(),
            "coordinate dimension mismatch"
        );
        let mut index = 0;
        for i in 0..self.shape().ndim() {
            index += coord[i] * self.shape().dims_strides[i].1;
        }
        index
    }
}

impl<Dtype, T> HasTensorShapeData<Dtype> for T where
    T: HasTensorShape + HasTensorData<Dtype>
{
}
