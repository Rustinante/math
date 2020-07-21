//! # Modeling mathematical intervals

use crate::set::contiguous_integer_set::ContiguousIntegerSet;

pub mod trait_impl;
pub mod traits;

pub type IntInterval<T> = ContiguousIntegerSet<T>;

pub type I64Interval = ContiguousIntegerSet<i64>;
