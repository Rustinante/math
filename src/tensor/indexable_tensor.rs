use crate::tensor::{has_tensor_shape_data::HasTensorShapeData, Unitless};

pub trait IndexableTensor<Dtype>: HasTensorShapeData<Dtype>
where
    Dtype: Copy, {
    fn at<T: AsRef<[Unitless]>>(&self, coord: T) -> Dtype {
        self.data()[self.coord_to_index(coord.as_ref()) as usize]
    }
}

impl<Dtype, T> IndexableTensor<Dtype> for T
where
    T: HasTensorShapeData<Dtype>,
    Dtype: Copy,
{
}

#[cfg(test)]
mod tests {
    use crate::tensor::{
        ephemeral_view::ToEphemeralView, indexable_tensor::IndexableTensor,
        tensor_storage::IntoTensorStorage,
    };

    #[test]
    fn test_indexing() {
        let storage = vec![1, 2, 3, 4, 5, 6].into_tensor_storage();
        let view = storage.as_shape([2, 3]);
        assert_eq!(view.at([0, 0]), 1);
        assert_eq!(view.at(&[0, 1]), 2);
        assert_eq!(view.at([0, 2]), 3);

        assert_eq!(view.at(vec![1, 0]), 4);

        let coord = vec![1, 1];
        assert_eq!(view.at(coord), 5);

        let coord = vec![1, 2];
        assert_eq!(view.at(&coord), 6);
    }
}
