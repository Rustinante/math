pub mod trait_impl;

pub trait Constructable {
    fn new() -> Self;
}

pub trait Collecting<E> {
    fn collect(&mut self, item: E);
}

pub trait ToIterator<'s, I: Iterator<Item=R>, R> {
    fn to_iter(&'s self) -> I;
}
