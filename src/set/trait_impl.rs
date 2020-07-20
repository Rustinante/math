use std::{collections::HashSet, hash::Hash};

use crate::set::traits::{Finite, Intersect, Set};

impl<T> Finite for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T> Finite for HashSet<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: Clone + Eq + Hash> Set<&T, HashSet<T>> for HashSet<T> {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn contains(&self, item: &T) -> bool {
        self.contains(item)
    }
}

impl<T: Clone + Eq + Hash> Intersect<&HashSet<T>, HashSet<T>> for HashSet<T> {
    fn intersect(&self, other: &HashSet<T>) -> HashSet<T> {
        self.intersection(other).map(|x| x.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::set::traits::Set;
    use std::collections::HashSet;

    #[test]
    fn test_set() {
        let mut s = HashSet::new();
        assert_eq!(s.is_empty(), true);
        s.insert(3);
        assert_eq!(s.is_empty(), false);
        assert_eq!(Set::<&i32, HashSet<i32>>::contains(&s, &3), true);
        assert_eq!(Set::<&i32, HashSet<i32>>::contains(&s, &4), false);
        assert_eq!(Set::<&i32, HashSet<i32>>::contains(&s, &2), false);
    }
}
