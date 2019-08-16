pub trait Set<E, R> {
    fn is_empty(&self) -> bool;

    fn contains(&self, element: E) -> bool;

    fn intersect(&self, other: &Self) -> R;
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum CountableType {
    Finite(usize),
    CountablyInfinite,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Cardinality {
    Countable(CountableType),
    Uncountable,
}

pub trait HasCardinality {
    fn get_cardinality(&self) -> Cardinality;
}

pub trait Countable: HasCardinality {
    fn count(&self) -> CountableType;

    fn is_finite(&self) -> bool {
        self.count() != CountableType::CountablyInfinite
    }
}

impl<T: Countable> HasCardinality for T {
    fn get_cardinality(&self) -> Cardinality {
        Cardinality::Countable(T::count(self))
    }
}

pub trait Finite: Countable {
    fn size(&self) -> usize;
}

impl<T: Finite> Countable for T {
    fn count(&self) -> CountableType {
        CountableType::Finite(T::size(self))
    }
}

/// Given two sets of the same type that are `Refineable`, their common refinement can be obtained
pub trait Refineable<O> {
    fn get_common_refinement(&self, other: &Self) -> O;
}
