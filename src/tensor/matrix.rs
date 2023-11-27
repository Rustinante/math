use crate::tensor::{
    ephemeral_view::EphemeralView,
    has_tensor_shape_data::HasTensorShapeData,
    indexable_tensor::IndexableTensor,
    tensor_shape::{HasTensorShape, TensorShape},
    tensor_storage::{HasTensorData, IntoTensorStorage, TensorStorage},
    Unitless,
};
use num::Num;
use std::{
    fmt,
    fmt::Formatter,
    ops::{Index, IndexMut},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix<Dtype> {
    shape: TensorShape,
    storage: TensorStorage<Dtype>,
}

impl<Dtype> Matrix<Dtype>
where
    Dtype: Copy + Num,
{
    pub fn from_vec(
        v: Vec<Dtype>,
        num_rows: Unitless,
        num_columns: Unitless,
    ) -> Matrix<Dtype> {
        assert_eq!(
            v.len(),
            (num_rows * num_columns) as usize,
            "number of elements in the vector does not match the shape"
        );
        Matrix {
            shape: create_row_major_shape(num_rows, num_columns),
            storage: v.into_tensor_storage(),
        }
    }
}

impl<Dtype> HasTensorShape for Matrix<Dtype> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
}

impl<Dtype> HasTensorData<Dtype> for Matrix<Dtype> {
    fn data(&self) -> &TensorStorage<Dtype> {
        &self.storage
    }
}

impl<Dtype> Index<[Unitless; 2]> for Matrix<Dtype>
where
    Dtype: Copy + Num,
{
    type Output = Dtype;

    fn index(&self, index: [Unitless; 2]) -> &Self::Output {
        &self.storage[self.coord_to_index(&[index[0], index[1]]) as usize]
    }
}

impl<Dtype> IndexMut<[Unitless; 2]> for Matrix<Dtype>
where
    Dtype: Copy + Num,
{
    fn index_mut(&mut self, index: [Unitless; 2]) -> &mut Self::Output {
        let index = self.coord_to_index(&[index[0], index[1]]);
        &mut self.storage[index as usize]
    }
}

pub trait MatrixTrait<Dtype> {
    fn num_rows(&self) -> Unitless;

    fn num_columns(&self) -> Unitless;
}

pub trait IndexableMatrix<Dtype>:
    IndexableTensor<Dtype> + MatrixTrait<Dtype>
where
    Dtype: Copy + Num, {
    fn matmul<R>(&self, other: &R) -> Matrix<Dtype>
    where
        R: MatrixTrait<Dtype> + IndexableTensor<Dtype>, {
        let m = self.num_rows();
        let n = self.num_columns();
        let n2 = other.num_rows();
        let l = other.num_columns();
        assert_eq!(n, n2, "self.num_columns {} != other.num_rows {}", n, n2);
        let mut result =
            Matrix::from_vec(vec![Dtype::zero(); (m * l) as usize], m, l);
        for i in 0..m {
            for j in 0..l {
                // multiply the i-th row against the j-th column
                for k in 0..n {
                    let old = result[[i, j]];
                    let x = self.at([i, k]);
                    let y = other.at([k, j]);
                    result[[i, j]] = old + x * y;
                }
            }
        }
        result
    }
}

impl<Dtype, T> IndexableMatrix<Dtype> for T
where
    Dtype: Copy + Num,
    T: MatrixTrait<Dtype> + IndexableTensor<Dtype>,
{
}

impl<Dtype> MatrixTrait<Dtype> for Matrix<Dtype> {
    fn num_rows(&self) -> Unitless {
        self.shape.dims_strides[0].0
    }

    fn num_columns(&self) -> Unitless {
        self.shape.dims_strides[1].0
    }
}

fn create_row_major_shape(
    num_rows: Unitless,
    num_columns: Unitless,
) -> TensorShape {
    TensorShape {
        dims_strides: vec![(num_rows, num_columns), (num_columns, 1)],
    }
}

impl<'a, Dtype> From<&'a Matrix<Dtype>> for EphemeralView<'a, Dtype> {
    fn from(matrix: &'a Matrix<Dtype>) -> Self {
        EphemeralView::new(&matrix.data(), matrix.shape().clone())
    }
}

impl<Dtype> fmt::Display for Matrix<Dtype>
where
    Dtype: Copy + Num + fmt::Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let num_rows = self.num_rows();
        let num_columns = self.num_columns();
        write!(f, "[")?;
        for i in 0..num_rows {
            write!(f, "[")?;
            for j in 0..num_columns {
                if (j + 1) == num_columns {
                    if (i + 1) != num_rows {
                        write!(f, "{}]\n", self[[i, j]])?;
                    } else {
                        write!(f, "{}]", self[[i, j]])?;
                    }
                } else {
                    write!(f, "{}, ", self[[i, j]])?;
                }
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::matrix_transpose::MatrixTranspose;

    #[test]
    fn test_matmul() {
        let a = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2);
        let b = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2);
        let c = a.matmul(&b);
        assert_eq!(c.data().vec, vec![7, 10, 15, 22]);

        let d = Matrix::from_vec(vec![1, 1], 2, 1);
        let e = a.matmul(&d);
        assert_eq!(e.storage.vec, vec![3, 7]);

        let a = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2);
        let res = a.t().matmul(&a);
        assert_eq!(res, Matrix::from_vec(vec![10, 14, 14, 20], 2, 2));
    }

    #[test]
    fn test_print_matrix() {
        fn get_display_string(
            vec: Vec<i32>,
            num_rows: Unitless,
            num_columns: Unitless,
        ) -> String {
            let m = Matrix::from_vec(vec, num_rows, num_columns);
            fmt::format(format_args!("{}", m))
        }

        assert_eq!(
            get_display_string(vec![1, 2, 3, 4], 2, 2),
            "[[1, 2]\n[3, 4]]"
        );
        assert_eq!(get_display_string(vec![1, 2], 2, 1), "[[1]\n[2]]");
        assert_eq!(get_display_string(vec![1, 2], 1, 2), "[[1, 2]]");
        assert_eq!(get_display_string(vec![], 0, 0), "[]");
        assert_eq!(
            get_display_string(
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                3,
                4,
            ),
            "[[1, 2, 3, 4]\n[5, 6, 7, 8]\n[9, 10, 11, 12]]"
        );
    }
}
