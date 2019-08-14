use num::Num;

/// A one-dimensional interval with a start and an end.
/// Whether or not the start and end elements are included in the interval depends
/// on the actual implementor type
pub trait Interval {
    type Element: Num + Copy;

    fn get_start(&self) -> Self::Element;

    fn get_end(&self) -> Self::Element;

    fn length(&self) -> Self::Element;
}

pub trait Coalesce<T>: std::marker::Sized {
    fn coalesce_with(&self, other: &T) -> Option<Self>;
}

/// implementors are container types that should be able to coalesce the contained intervals
pub trait CoalesceIntervals<I: Interval<Element=E>, E: Num + Copy>: std::marker::Sized {
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
