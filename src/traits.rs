//! # General traits

pub mod trait_impl;

pub trait Collecting<E> {
    fn collect(&mut self, item: E);
}

pub trait HasDuplicate {
    fn has_duplicate(&self) -> bool;
}

pub trait Slicing<I, O> {
    fn slice(self, input: I) -> O;
}

pub trait SubsetIndexable<S, Output> {
    fn get_set_containing(&self, subset: &S) -> Option<Output>;
}

pub trait ToIterator<'s, I: Iterator<Item = R>, R> {
    fn to_iter(&'s self) -> I;
}
