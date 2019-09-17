use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::vec::IntoIter;

/// `K` is the key type
/// `M` is the map type that maps keys of type `K` to values
pub trait UnionZip<K, M> {
    fn union_zip<'a>(&'a self, other: &'a Self) -> UnionZipped<'a, K, M>;
}

/// Takes the sorted union of the two sets of keys for future iteration
/// ```
/// use std::collections::HashMap;
/// use analytic::iter::UnionZip;
/// let m1: HashMap<i32, i32> = vec![
///         (1, 10),
///         (3, 23),
///         (4, 20)
///     ].into_iter().collect();
/// let m2: HashMap<i32, i32> = vec![
///         (0, 4),
///         (1, 20),
///         (4, 20),
///         (9, 29),
///     ].into_iter().collect();
///
/// let mut iter = m1.union_zip(&m2)
///                  .into_iter();
/// assert_eq!(Some((0, (None, Some(&4)))), iter.next());
/// assert_eq!(Some((1, (Some(&10), Some(&20)))), iter.next());
/// assert_eq!(Some((3, (Some(&23), None))), iter.next());
/// assert_eq!(Some((4, (Some(&20), Some(&20)))), iter.next());
/// assert_eq!(Some((9, (None, Some(&29)))), iter.next());
/// assert_eq!(None, iter.next());
/// ```
impl<K, V> UnionZip<K, HashMap<K, V>> for HashMap<K, V>
    where K: Hash + Eq + Clone + Ord {
    fn union_zip<'a>(&'a self, other: &'a Self) -> UnionZipped<'a, K, HashMap<K, V>> {
        let mut keys: Vec<K> = self
            .keys()
            .collect::<HashSet<&K>>()
            .union(
                &other
                    .keys()
                    .collect::<HashSet<&K>>()
            )
            .map(|&k| k.clone())
            .collect();

        keys.sort();

        UnionZipped {
            keys,
            left: &self,
            right: other,
        }
    }
}

pub struct UnionZipped<'a, K, M> {
    keys: Vec<K>,
    left: &'a M,
    right: &'a M,
}

impl<'a, K, V> IntoIterator for UnionZipped<'a, K, HashMap<K, V>>
    where K: Hash + Eq {
    type Item = <UnionZippedIter<'a, K, HashMap<K, V>, IntoIter<K>> as Iterator>::Item;
    type IntoIter = UnionZippedIter<'a, K, HashMap<K, V>, IntoIter<K>>;
    fn into_iter(self) -> Self::IntoIter {
        UnionZippedIter {
            keys: self.keys.into_iter(),
            left: self.left,
            right: self.right,
        }
    }
}

pub struct UnionZippedIter<'a, K, M, I: Iterator<Item=K>> {
    keys: I,
    left: &'a M,
    right: &'a M,
}

impl<'a, K, V> Iterator for UnionZippedIter<'a, K, HashMap<K, V>, IntoIter<K>>
    where K: Hash + Eq {
    type Item = (K, (Option<&'a V>, Option<&'a V>));
    fn next(&mut self) -> Option<Self::Item> {
        match self.keys.next() {
            None => None,
            Some(k) => {
                let left = if self.left.contains_key(&k) { Some(&self.left[&k]) } else { None };
                let right = if self.right.contains_key(&k) { Some(&self.right[&k]) } else { None };
                Some((k, (left, right)))
            }
        }
    }
}
