use std::{ops::Deref, slice::Iter};

use crate::traits::{Collecting, HasDuplicate, ToIterator};

impl<T> Collecting<T> for Vec<T> {
    #[inline]
    fn collect(&mut self, item: T) {
        self.push(item);
    }
}

impl<'a, T: Clone> Collecting<&'a T> for Vec<T>
where
    &'a T: Deref,
{
    #[inline]
    fn collect(&mut self, item: &'a T) {
        self.push((*item).clone());
    }
}

impl<'a, E> ToIterator<'a, Iter<'a, E>, &'a E> for Vec<E> {
    #[inline]
    fn to_iter(&'a self) -> Iter<'a, E> {
        self.iter()
    }
}

impl<T: std::cmp::Ord> HasDuplicate for Vec<T> {
    fn has_duplicate(&self) -> bool {
        let mut indices: Vec<usize> = (0..self.len()).into_iter().collect();
        indices.sort_by_key(|index| &self[*index]);
        for (a, b) in indices.iter().zip(indices.iter().skip(1)) {
            if self[*a] == self[*b] {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::HasDuplicate;

    #[test]
    fn test_has_duplciate() {
        let mut v = vec![6, 2, 3, 9, 1, 10, 23];
        assert_eq!(v.has_duplicate(), false);
        v.push(3);
        assert_eq!(v.has_duplicate(), true);

        let v = vec!["hi", "ab", "cde", "abc"];
        assert_eq!(v.has_duplicate(), false);
        let v = vec!["hi", "ab", "cde", "ab", "abc"];
        assert_eq!(v.has_duplicate(), true);
    }
}
