//! # A Linear Algebra Library

pub mod borrow_tensor;
pub mod ephemeral_view;
pub mod has_tensor_shape_data;
pub mod indexable_tensor;
pub mod matrix;
pub mod matrix_transpose;
pub mod matrix_view;
pub mod tensor_iter;
pub mod tensor_shape;
pub mod tensor_storage;

pub type Unitless = i64;
pub type AxisIndex = usize;
