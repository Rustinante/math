//! # A Linear Algebra Library

use num::ToPrimitive;
use std::{collections::HashSet, iter::FromIterator};

pub mod tensor_iter;

pub use tensor_iter::TensorIter;

/// The implementer can be viewed as a tensor of `shape` through the `as_shape`
/// method. The resulting `EphemeralView` cannot outlive the original data
/// struct.
pub trait ToView<'a, Dtype> {
    fn as_shape<S: Into<TensorShape>>(
        &'a self,
        shape: S,
    ) -> EphemeralView<'a, Dtype>;
}

/// The implementer can be converted into a Tensor struct with the specified
/// `shape`.
pub trait IntoTensor<Dtype> {
    fn into_tensor<S: Into<TensorShape>>(self, shape: S) -> Tensor<Dtype>;
}

/// The implementer provides an interface for the underlying data and shape.
pub trait ShapableData<Dtype> {
    /// Returns the underlying data.
    fn data(&self) -> &Vec<Dtype>;

    /// Returns the shape associated with the data.
    fn shape(&self) -> &TensorShape;

    /// Returns a mutable shape associated with the data.
    fn shape_mut(&mut self) -> &mut TensorShape;

    /// Reverses the axes.
    fn t(&self) -> EphemeralView<Dtype> {
        let transposed_axes: Vec<AxisIndex> =
            (0..self.shape().ndim()).into_iter().rev().collect();
        let shape_transpose = self.shape().to_transposed(transposed_axes);
        self.data().as_shape(shape_transpose)
    }

    /// # Arguments
    /// * `axes` - Must be the same length as `self.shape().ndim()`. For each
    ///   `i`, `axes[i] = j`
    /// means that the original `j`-th axis will be at the `i`-th axis in the
    /// new shape.
    fn transpose(&self, axes: Vec<AxisIndex>) -> EphemeralView<Dtype> {
        self.data().as_shape(self.shape().to_transposed(axes))
    }
}

pub type AxisIndex = usize;
pub type Length = i64;
pub type Stride = i64;

/// # An N-dimensional Tensor
///
/// ## Examples
/// ```
/// use math::tensor::{IntoTensor, ShapableData};
///
/// let tensor =
///     vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_tensor([3, 4]);
/// assert_eq!(tensor.shape().ndim(), 2);
/// assert_eq!(tensor.shape().dims(), vec![3, 4]);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tensor<Dtype> {
    data: Vec<Dtype>,
    shape: TensorShape,
}

/// A Tensor provides an interface for the underlying data and shape.
impl<Dtype> ShapableData<Dtype> for Tensor<Dtype> {
    fn data(&self) -> &Vec<Dtype> {
        &self.data
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn shape_mut(&mut self) -> &mut TensorShape {
        &mut self.shape
    }
}

impl<'a, Dtype> ToView<'a, Dtype> for Tensor<Dtype> {
    fn as_shape<S: Into<TensorShape>>(
        &'a self,
        shape: S,
    ) -> EphemeralView<'a, Dtype> {
        let target_shape: TensorShape = shape.into();
        assert_eq!(
            target_shape.num_elements(),
            self.shape.num_elements(),
            "number of elements in target shape mismatch"
        );
        EphemeralView {
            data: &self.data,
            shape: target_shape,
        }
    }
}

/// # A View of the Underlying Referenced Data as a Particular Shape
/// The underlying `data` has to outlive the `EphemeralView` itself.
///
/// ## Examples
/// ```
/// use math::tensor::{IntoTensor, ShapableData, ToView};
///
/// let tensor = vec![1, 2, 3, 4].into_tensor(vec![2, 2]);
/// let view = tensor.as_shape(vec![4, 1]);
/// assert_eq!(view.shape().dims(), vec![4, 1]);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EphemeralView<'a, Dtype> {
    data: &'a Vec<Dtype>,
    shape: TensorShape,
}

/// An EphemeralView provides an interface for the underlying data and shape.
impl<'a, Dtype> ShapableData<Dtype> for EphemeralView<'a, Dtype> {
    fn data(&self) -> &Vec<Dtype> {
        self.data
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn shape_mut(&mut self) -> &mut TensorShape {
        &mut self.shape
    }
}

/// A Tensor can be viewed as an EphemeralView with the same shape.
impl<'a, Dtype> From<&'a Tensor<Dtype>> for EphemeralView<'a, Dtype> {
    fn from(tensor: &'a Tensor<Dtype>) -> Self {
        tensor.data.as_shape(tensor.shape.clone())
    }
}

/// The shape of an N-dimensional tensor has a size for each dimension, with an
/// associated stride, e.g., a row-major 3 x 5 matrix will have a stride of 5
/// for the dimension of size 3 and a stride of 1 for the dimension of size 5,
/// and the resulting `dims_strides` is `[(3, 5), (5, 1)]`. Index 0 of
/// `dims_strides` always refers to the leftmost dimension.
#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct TensorShape {
    dims_strides: Vec<(Length, Stride)>,
}

impl TensorShape {
    pub fn dims(&self) -> Vec<Length> {
        self.dims_strides.iter().map(|(dim, _)| *dim).collect()
    }

    pub fn strides(&self) -> Vec<Stride> {
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

    fn to_transposed(&self, axes: Vec<AxisIndex>) -> TensorShape {
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

impl<'a, Dtype> ToView<'a, Dtype> for Vec<Dtype> {
    fn as_shape<S: Into<TensorShape>>(
        &'a self,
        shape: S,
    ) -> EphemeralView<'a, Dtype> {
        let target_shape: TensorShape = shape.into();
        assert_eq!(
            target_shape.num_elements(),
            self.len(),
            "number of elements in target shape mismatch"
        );
        EphemeralView {
            data: &self,
            shape: target_shape,
        }
    }
}

impl<Dtype> IntoTensor<Dtype> for Vec<Dtype> {
    fn into_tensor<S: Into<TensorShape>>(self, shape: S) -> Tensor<Dtype> {
        let shape: TensorShape = shape.into();
        let num_elements =
            shape.dims().iter().fold(1, |acc, &x| acc * x) as usize;

        if num_elements != self.len() {
            debug!(
                "Total number of elements ({}) in {:?} != vector length ({})",
                num_elements,
                shape,
                self.len()
            );
            assert_eq!(
                num_elements,
                self.len(),
                "Total number of elements ({}) in {:?} != vector length ({})",
                num_elements,
                shape,
                self.len()
            );
        }
        Tensor {
            data: self,
            shape,
        }
    }
}

macro_rules! impl_from_for_tensor_shape {
    ($t:ty) => {
        impl From<$t> for TensorShape {
            fn from(shape: $t) -> Self {
                // default to row-major order
                let strides: Vec<Stride> = shape
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

    #[test]
    fn test_as_shape() {
        let data = vec![1, 2, 3, 4];
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

    #[test]
    fn test_tensor_shape() {
        {
            let shape = TensorShape::from([2, 4, 3]);
            assert_eq!(shape.dims(), vec![2, 4, 3]);
            assert_eq!(shape.strides(), vec![12, 3, 1]);
            assert_eq!(shape.ndim(), 3);
        }
        {
            let empty_shape = TensorShape::from(Vec::<Length>::new());
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

    #[test]
    fn test_tensor_shape_trait() {
        {
            let tensor = vec![1, 2, 3, 4, 5, 6].into_tensor(vec![2, 3]);
            let shape = tensor.shape();
            assert_eq!(shape.dims(), vec![2, 3]);
            assert_eq!(shape.strides(), vec![3, 1]);
            assert_eq!(shape.ndim(), 2);
        }
        {
            let data: Vec<i32> = (0..24).into_iter().collect();
            let mut tensor = data.into_tensor(vec![2, 3, 4]);
            let shape = tensor.shape();
            assert_eq!(shape.dims(), vec![2, 3, 4]);
            assert_eq!(shape.strides(), vec![12, 4, 1]);
            assert_eq!(shape.ndim(), 3);

            let shape = tensor.shape_mut();
            assert_eq!(shape.dims(), vec![2, 3, 4]);
            assert_eq!(shape.strides(), vec![12, 4, 1]);
            assert_eq!(shape.ndim(), 3);
            shape.dims_strides[1].0 = 4;
            shape.dims_strides[2].0 = 3;
            shape.dims_strides[1].1 = 3;
            assert_eq!(tensor.shape().dims(), vec![2, 4, 3]);
            assert_eq!(tensor.shape().dims(), vec![2, 4, 3]);
            assert_eq!(tensor.shape().strides(), vec![12, 3, 1]);
            assert_eq!(tensor.shape().ndim(), 3);
        }
        {
            let data: Vec<i32> = (0..30).into_iter().collect();
            let mut view = data.as_shape(vec![2, 3, 5]);
            let shape = view.shape();
            assert_eq!(shape.dims(), vec![2, 3, 5]);
            assert_eq!(shape.strides(), vec![15, 5, 1]);
            assert_eq!(shape.ndim(), 3);

            let shape = view.shape_mut();
            shape.dims_strides[1].0 = 5;
            shape.dims_strides[2].0 = 3;
            shape.dims_strides[1].1 = 3;
            assert_eq!(shape.dims(), vec![2, 5, 3]);
            assert_eq!(shape.strides(), vec![15, 3, 1]);
            assert_eq!(shape.ndim(), 3);
        }
    }

    #[test]
    fn test_transpose() {
        {
            let arr = vec![1, 2, 3, 4, 5, 6].into_tensor([2, 3]);
            let arr_t = arr.t();
            assert_eq!(arr_t.shape().dims(), vec![3, 2]);
            assert_eq!(arr_t.shape().strides(), vec![1, 3]);
            assert_eq!(arr_t.shape().ndim(), 2);
        }
        {
            // the original stride is (12, 3, 1)
            let arr = (0..24)
                .into_iter()
                .collect::<Vec<i32>>()
                .into_tensor([2, 4, 3]);
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
            let arr = (0..120)
                .into_iter()
                .collect::<Vec<i32>>()
                .into_tensor([2, 4, 3, 5]);

            let arr_t = arr.transpose(vec![1, 3, 0, 2]);
            assert_eq!(arr_t.shape().dims(), vec![4, 5, 2, 3]);
            assert_eq!(arr_t.shape().strides(), vec![15, 1, 60, 5]);
            assert_eq!(arr_t.shape().ndim(), 4);
        }
    }
}
