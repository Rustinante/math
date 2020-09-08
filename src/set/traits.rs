pub trait Set<E> {
    fn is_empty(&self) -> bool;

    fn contains(&self, element: &E) -> bool;
}

pub trait Intersect<S, O> {
    fn intersect(&self, other: S) -> O;

    fn has_non_empty_intersection_with(&self, other: S) -> bool;
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CountableType {
    Finite(usize),
    CountablyInfinite,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Cardinality {
    Countable(CountableType),
    Uncountable,
}

pub trait Countable {
    fn count(&self) -> CountableType;

    fn is_finite(&self) -> bool {
        self.count() != CountableType::CountablyInfinite
    }
}

pub trait Finite {
    fn size(&self) -> usize;
}

/// Given two sets of the same type that are `Refineable`, their common
/// refinement can be obtained
pub trait Refineable<O> {
    fn get_common_refinement(&self, other: &Self) -> O;
}
