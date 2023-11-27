use crate::tensor::{AxisIndex, Unitless};
use num::ToPrimitive;
use std::{collections::HashSet, iter::FromIterator};

/// The shape of an N-dimensional tensor has a size for each dimension, with an
/// associated stride, e.g., a row-major 3 x 5 matrix will have a stride of 5
/// for the dimension of size 3 and a stride of 1 for the dimension of size 5,
/// and the resulting `dims_strides` is `[(3, 5), (5, 1)]`. Index 0 of
/// `dims_strides` always refers to the leftmost dimension.
#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct TensorShape {
    pub dims_strides: Vec<(Unitless, Unitless)>,
}

impl TensorShape {
    pub fn dims(&self) -> Vec<Unitless> {
        self.dims_strides.iter().map(|(dim, _)| *dim).collect()
    }

    pub fn strides(&self) -> Vec<Unitless> {
        self.dims_strides
            .iter()
            .map(|(_, stride)| *stride)
            .collect()
    }

    pub fn ndim(&self) -> usize {
        self.dims_strides.len()
    }

    pub fn num_elements(&self) -> usize {
        if self.dims_strides.len() > 0 {
            self.dims_strides
                .iter()
                .fold(1, |acc, &(d, _)| acc * d as usize)
        } else {
            0
        }
    }

    pub fn to_transposed(&self, axes: Vec<AxisIndex>) -> TensorShape {
        assert_eq!(
            axes.len(),
            self.dims_strides.len(),
            "length of axes ({}) != length of dims_strides ({})",
            axes.len(),
            self.dims_strides.len()
        );
        assert_eq!(
            HashSet::<AxisIndex>::from_iter(axes.clone().into_iter()).len(),
            self.dims_strides.len(),
            "all axes must be distinct"
        );
        let dims_strides =
            axes.into_iter().map(|i| self.dims_strides[i]).collect();
        TensorShape {
            dims_strides,
        }
    }
}

pub trait HasTensorShape {
    fn shape(&self) -> &TensorShape;
}

macro_rules! impl_from_for_tensor_shape {
    ($t:ty) => {
        impl From<$t> for TensorShape {
            fn from(shape: $t) -> Self {
                // default to row-major order
                let strides: Vec<Unitless> = shape
                    .iter()
                    .rev()
                    .scan(1i64, |acc, len| {
                        let s = *acc;
                        *acc *= *len as i64;
                        Some(s)
                    })
                    .collect();

                TensorShape {
                    dims_strides: shape
                        .iter()
                        .map(|s| s.to_i64().unwrap())
                        .zip(strides.into_iter().rev())
                        .collect(),
                }
            }
        }
    };
}

impl_from_for_tensor_shape!(Vec<i32>);
impl_from_for_tensor_shape!(Vec<u32>);
impl_from_for_tensor_shape!(Vec<i64>);
impl_from_for_tensor_shape!(Vec<u64>);
impl_from_for_tensor_shape!(Vec<isize>);
impl_from_for_tensor_shape!(Vec<usize>);

impl_from_for_tensor_shape!(&Vec<i32>);
impl_from_for_tensor_shape!(&Vec<u32>);
impl_from_for_tensor_shape!(&Vec<i64>);
impl_from_for_tensor_shape!(&Vec<u64>);
impl_from_for_tensor_shape!(&Vec<isize>);
impl_from_for_tensor_shape!(&Vec<usize>);

// implementing for small fixed-size arrays for ergonomic reasons
impl_from_for_tensor_shape!([i32; 1]);
impl_from_for_tensor_shape!([i32; 2]);
impl_from_for_tensor_shape!([i32; 3]);
impl_from_for_tensor_shape!([i32; 4]);
impl_from_for_tensor_shape!([i32; 5]);
impl_from_for_tensor_shape!([i32; 6]);
impl_from_for_tensor_shape!([i32; 7]);
impl_from_for_tensor_shape!([i32; 8]);

impl_from_for_tensor_shape!([u32; 1]);
impl_from_for_tensor_shape!([u32; 2]);
impl_from_for_tensor_shape!([u32; 3]);
impl_from_for_tensor_shape!([u32; 4]);
impl_from_for_tensor_shape!([u32; 5]);
impl_from_for_tensor_shape!([u32; 6]);
impl_from_for_tensor_shape!([u32; 7]);
impl_from_for_tensor_shape!([u32; 8]);

impl_from_for_tensor_shape!([i64; 1]);
impl_from_for_tensor_shape!([i64; 2]);
impl_from_for_tensor_shape!([i64; 3]);
impl_from_for_tensor_shape!([i64; 4]);
impl_from_for_tensor_shape!([i64; 5]);
impl_from_for_tensor_shape!([i64; 6]);
impl_from_for_tensor_shape!([i64; 7]);
impl_from_for_tensor_shape!([i64; 8]);

impl_from_for_tensor_shape!([u64; 1]);
impl_from_for_tensor_shape!([u64; 2]);
impl_from_for_tensor_shape!([u64; 3]);
impl_from_for_tensor_shape!([u64; 4]);
impl_from_for_tensor_shape!([u64; 5]);
impl_from_for_tensor_shape!([u64; 6]);
impl_from_for_tensor_shape!([u64; 7]);
impl_from_for_tensor_shape!([u64; 8]);

impl_from_for_tensor_shape!([isize; 1]);
impl_from_for_tensor_shape!([isize; 2]);
impl_from_for_tensor_shape!([isize; 3]);
impl_from_for_tensor_shape!([isize; 4]);
impl_from_for_tensor_shape!([isize; 5]);
impl_from_for_tensor_shape!([isize; 6]);
impl_from_for_tensor_shape!([isize; 7]);
impl_from_for_tensor_shape!([isize; 8]);

impl_from_for_tensor_shape!([usize; 1]);
impl_from_for_tensor_shape!([usize; 2]);
impl_from_for_tensor_shape!([usize; 3]);
impl_from_for_tensor_shape!([usize; 4]);
impl_from_for_tensor_shape!([usize; 5]);
impl_from_for_tensor_shape!([usize; 6]);
impl_from_for_tensor_shape!([usize; 7]);
impl_from_for_tensor_shape!([usize; 8]);

impl_from_for_tensor_shape!(&[isize; 1]);
impl_from_for_tensor_shape!(&[isize; 2]);
impl_from_for_tensor_shape!(&[isize; 3]);
impl_from_for_tensor_shape!(&[isize; 4]);
impl_from_for_tensor_shape!(&[isize; 5]);
impl_from_for_tensor_shape!(&[isize; 6]);
impl_from_for_tensor_shape!(&[isize; 7]);
impl_from_for_tensor_shape!(&[isize; 8]);

impl_from_for_tensor_shape!(&[usize; 1]);
impl_from_for_tensor_shape!(&[usize; 2]);
impl_from_for_tensor_shape!(&[usize; 3]);
impl_from_for_tensor_shape!(&[usize; 4]);
impl_from_for_tensor_shape!(&[usize; 5]);
impl_from_for_tensor_shape!(&[usize; 6]);
impl_from_for_tensor_shape!(&[usize; 7]);
impl_from_for_tensor_shape!(&[usize; 8]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Unitless;

    #[test]
    fn test_tensor_shape() {
        {
            let shape = TensorShape::from([2, 4, 3]);
            assert_eq!(shape.dims(), vec![2, 4, 3]);
            assert_eq!(shape.strides(), vec![12, 3, 1]);
            assert_eq!(shape.ndim(), 3);
        }
        {
            let empty_shape = TensorShape::from(Vec::<Unitless>::new());
            assert_eq!(empty_shape.dims(), vec![]);
            assert_eq!(empty_shape.strides(), vec![]);
            assert_eq!(empty_shape.ndim(), 0);
        }
    }

    #[test]
    fn test_tensor_shape_from_trait() {
        macro_rules! check_from_iter {
            ($iter:expr) => {
                let tensor_shape = TensorShape::from($iter);
                assert_eq!(tensor_shape.dims_strides, vec![
                    (3, 10),
                    (2, 5),
                    (5, 1)
                ]);
            };
        }
        check_from_iter!(vec![3i32, 2, 5]);
        check_from_iter!(vec![3u32, 2, 5]);
        check_from_iter!(vec![3i64, 2, 5]);
        check_from_iter!(vec![3u64, 2, 5]);
        check_from_iter!(vec![3isize, 2, 5]);
        check_from_iter!(vec![3usize, 2, 5]);
        check_from_iter!(&vec![3i32, 2, 5]);
        check_from_iter!(&vec![3u32, 2, 5]);
        check_from_iter!(&vec![3i64, 2, 5]);
        check_from_iter!(&vec![3u64, 2, 5]);
        check_from_iter!(&vec![3isize, 2, 5]);
        check_from_iter!(&vec![3usize, 2, 5]);
        check_from_iter!([3i32, 2, 5]);
        check_from_iter!([3u32, 2, 5]);
        check_from_iter!([3i64, 2, 5]);
        check_from_iter!([3u64, 2, 5]);
        check_from_iter!([3isize, 2, 5]);
        check_from_iter!([3usize, 2, 5]);
        check_from_iter!(&[3isize, 2, 5]);
        check_from_iter!(&[3usize, 2, 5]);
    }
}
