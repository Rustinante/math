pub mod trait_impl;

pub trait Constructable {
    fn new() -> Self;
}

pub trait Collecting<E> {
    fn collect(&mut self, item: E);
}

pub trait SubsetIndexable<S> {
    fn get_set_containing(&self, subset: &S) -> Option<S>;
}

pub trait ToIterator<'s, I: Iterator<Item=R>, R> {
    fn to_iter(&'s self) -> I;
}

pub trait HasDuplicate {
    fn has_duplicate(&self) -> bool;
}
