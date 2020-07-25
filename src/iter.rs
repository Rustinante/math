//! # Iterator adapters

pub mod binned_interval_iter;
pub mod common_refinement_zip;
pub mod flat_zip;
pub mod union_zip;

pub use common_refinement_zip::{CommonRefinementZip, CommonRefinementZipped};
pub use union_zip::{IntoUnionZip, UnionZip, UnionZipped, UnionZippedIter};
