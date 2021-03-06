//! # Iterator adapters

pub mod binned_interval_iter;
pub mod common_refinement_zip;
pub mod concatenated_iter;
pub mod flat_zip;
pub mod union_zip;
pub mod weighted_sum;

pub use binned_interval_iter::{
    AggregateOp, BinnedIntervalIter, IntoBinnedIntervalIter,
};
pub use common_refinement_zip::{CommonRefinementZip, CommonRefinementZipped};
pub use concatenated_iter::{ConcatenatedIter, IntoConcatIter};
pub use flat_zip::{FlatZipIter, IntoFlatZipIter};
pub use union_zip::{
    AsUnionZipped, IntoUnionZip, UnionZip, UnionZipped, UnionZippedIter,
};
pub use weighted_sum::WeightedSum;
