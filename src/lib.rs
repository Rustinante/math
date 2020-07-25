//! # Modeling mathematical constructs and their associated operations.

#![feature(drain_filter)]
#![feature(bool_to_option)]

#[macro_use]
extern crate log;

pub mod histogram;
pub mod interval;
pub mod iter;
pub mod partition;
pub mod sample;
pub mod search;
pub mod set;
pub mod stats;
pub mod tensor;
pub mod traits;
