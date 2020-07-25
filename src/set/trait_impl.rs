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

impl<T: Clone + Eq + Hash> Set<T> for HashSet<T> {
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

    fn has_non_empty_intersection_with(&self, other: &HashSet<T>) -> bool {
        self.intersection(other).next().is_some()
    }
}

#[cfg(test)]
mod tests {
    use crate::set::traits::{Intersect, Set};
    use std::collections::HashSet;

    #[test]
    fn test_set() {
        let mut s = HashSet::new();
        assert_eq!(s.is_empty(), true);
        s.insert(3);
        assert_eq!(s.is_empty(), false);
        assert_eq!(Set::<i32>::contains(&s, &3), true);
        assert_eq!(Set::<i32>::contains(&s, &4), false);
        assert_eq!(Set::<i32>::contains(&s, &2), false);
    }

    #[test]
    fn test_hashset_intersect() {
        let s1: HashSet<i32> = [1, 2, 3, 4].iter().cloned().collect();
        let s2: HashSet<i32> = [2, 3, 7].iter().cloned().collect();
        let e1: HashSet<i32> = [2, 3].iter().cloned().collect();
        assert_eq!(s1.intersect(&s2), e1);
        let s3 = HashSet::<i32>::new();
        assert_eq!(s1.intersect(&s3), s3);

        assert!(s1.has_non_empty_intersection_with(&s2));
        assert!(!s1.has_non_empty_intersection_with(&s3));
    }
}
