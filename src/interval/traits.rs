use crate::set::traits::Set;
use num::Num;

/// A one-dimensional interval with a start and an end.
/// Whether or not the start and end elements are included in the interval
/// depends on the actual implementor type
pub trait Interval<T>: Set<T>
where
    T: Num + Copy, {
    fn from_boundaries(start: T, end_inclusive: T) -> Self;

    fn get_start(&self) -> T;

    fn get_end(&self) -> T;

    fn length(&self) -> T;
}

pub trait Coalesce<T>: Sized {
    fn coalesce_with(&self, other: &T) -> Option<Self>;
}

/// implementors are container types that should be able to coalesce the
/// contained intervals
pub trait CoalesceIntervals<I: Interval<E>, E: Num + Copy>: Sized {
    fn to_coalesced_intervals(&self) -> Vec<I>;

    fn coalesce_intervals_inplace(&mut self);

    fn into_coalesced(mut self) -> Self {
        self.coalesce_intervals_inplace();
        self
    }
}

pub trait Topology {
    fn is_open(&self) -> bool;
    fn is_closed(&self) -> bool;
}
