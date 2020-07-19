//! A Linear Algebra Library

pub mod tensor;

pub use tensor::{
    EphemeralView, IntoTensor, Length, ShapableData, Stride, Tensor, TensorShape, ToView,
};
