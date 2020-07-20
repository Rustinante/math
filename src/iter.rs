use crate::{
    partition::ordered_interval_partitions::OrderedIntervalPartitions,
    set::{contiguous_integer_set::ContiguousIntegerSet, traits::Refineable},
    traits::SubsetIndexable,
};
use num::Integer;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::Hash,
    vec::IntoIter,
};

/// `K` is the key type
/// `M` is the map type that maps keys of type `K` to values
pub trait UnionZip<K, M> {
    fn union_zip<'a>(&'a self, other: &'a M) -> UnionZipped<'a, K, M>;
}

pub trait IntoUnionZip<'a, K, M> {
    fn into_union_zip(self, other: &'a M) -> UnionZipped<'a, K, M>;
}

pub struct UnionZipped<'a, K, M> {
    keys: Vec<K>,
    maps: Vec<&'a M>,
}

pub struct UnionZippedIter<'a, K, M, I: Iterator<Item = K>> {
    keys: I,
    maps: Vec<&'a M>,
}

/// Takes the sorted union of the two sets of keys for future iteration
/// ```
/// use analytic::iter::UnionZip;
/// use std::collections::HashMap;
/// let m1: HashMap<i32, i32> = vec![(1, 10), (3, 23), (4, 20)].into_iter().collect();
/// let m2: HashMap<i32, i32> = vec![(0, 4), (1, 20), (4, 20), (9, 29)]
///     .into_iter()
///     .collect();
///
/// let mut iter = m1.union_zip(&m2).into_iter();
/// assert_eq!(Some((0, vec![None, Some(&4)])), iter.next());
/// assert_eq!(Some((1, vec![Some(&10), Some(&20)])), iter.next());
/// assert_eq!(Some((3, vec![Some(&23), None])), iter.next());
/// assert_eq!(Some((4, vec![Some(&20), Some(&20)])), iter.next());
/// assert_eq!(Some((9, vec![None, Some(&29)])), iter.next());
/// assert_eq!(None, iter.next());
/// ```
impl<K, V> UnionZip<K, HashMap<K, V>> for HashMap<K, V>
where
    K: Hash + Eq + Clone + Ord,
{
    fn union_zip<'a>(&'a self, other: &'a Self) -> UnionZipped<'a, K, HashMap<K, V>> {
        let mut keys: Vec<K> = self
            .keys()
            .collect::<HashSet<&K>>()
            .union(&other.keys().collect::<HashSet<&K>>())
            .map(|&k| k.clone())
            .collect();

        keys.sort();

        UnionZipped {
            keys,
            maps: vec![&self, other],
        }
    }
}

impl<'a, K, V> IntoUnionZip<'a, K, HashMap<K, V>> for UnionZipped<'a, K, HashMap<K, V>>
where
    K: Hash + Eq + Clone + Ord,
{
    fn into_union_zip(self, other: &'a HashMap<K, V>) -> UnionZipped<'a, K, HashMap<K, V>> {
        let mut keys: Vec<K> = self
            .keys
            .iter()
            .collect::<HashSet<&K>>()
            .union(&other.keys().collect::<HashSet<&K>>())
            .map(|&k| k.clone())
            .collect();

        keys.sort();

        let mut maps = self.maps;
        maps.push(other);
        UnionZipped {
            keys,
            maps,
        }
    }
}

impl<'a, K, V> IntoIterator for UnionZipped<'a, K, HashMap<K, V>>
where
    K: Hash + Eq,
{
    type IntoIter = UnionZippedIter<'a, K, HashMap<K, V>, IntoIter<K>>;
    type Item = <UnionZippedIter<'a, K, HashMap<K, V>, IntoIter<K>> as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        UnionZippedIter {
            keys: self.keys.into_iter(),
            maps: self.maps,
        }
    }
}

impl<'a, K, V> Iterator for UnionZippedIter<'a, K, HashMap<K, V>, IntoIter<K>>
where
    K: Hash + Eq,
{
    type Item = (K, Vec<Option<&'a V>>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.keys.next() {
            None => None,
            Some(k) => {
                let mapped: Vec<Option<&'a V>> = self
                    .maps
                    .iter()
                    .map(|m| {
                        if m.contains_key(&k) {
                            Some(&m[&k])
                        } else {
                            None
                        }
                    })
                    .collect();

                Some((k, mapped))
            }
        }
    }
}

/// `P` is the partition type
/// `R` is the common refinement type
/// `M` is the map type that maps partitions of type `P` to values
pub trait CommonRefinementZip<P, R, M> {
    fn common_refinement_zip<'a, F>(
        &'a self,
        other: &'a M,
        sort_fn: F,
    ) -> CommonRefinementZipped<'a, P, R, M>
    where
        P: SubsetIndexable<R>,
        F: FnMut(&R, &R) -> Ordering;
}

pub trait IntoCommonRefinementZip<'a, P, R, M> {
    fn into_common_refinement_zip<F>(
        self,
        other: &'a M,
        sort_fn: F,
    ) -> CommonRefinementZipped<'a, P, R, M>
    where
        P: SubsetIndexable<R>,
        F: FnMut(&R, &R) -> Ordering;
}

pub struct CommonRefinementZipped<'a, P, R, M>
where
    P: SubsetIndexable<R>, {
    refined_keys: Vec<R>,
    keys_list: Vec<P>,
    maps: Vec<&'a M>,
}

pub struct CommonRefinementZipIter<'a, P, R, M, I: Iterator<Item = R>>
where
    P: SubsetIndexable<R>, {
    refined_keys_iter: I,
    partitions_list: Vec<P>,
    maps: Vec<&'a M>,
}

pub type RefinedInterval<T> = ContiguousIntegerSet<T>;

impl<V, T: Integer + Copy + Hash>
    CommonRefinementZip<
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
    > for HashMap<RefinedInterval<T>, V>
{
    /// ```
    /// use analytic::{
    ///     interval::traits::Interval,
    ///     iter::{CommonRefinementZip, RefinedInterval},
    /// };
    /// use std::collections::HashMap;
    ///
    /// let m1: HashMap<RefinedInterval<usize>, i32> = vec![
    ///     (RefinedInterval::new(0, 5), 5),
    ///     (RefinedInterval::new(8, 10), 2),
    /// ]
    /// .into_iter()
    /// .collect();
    ///
    /// let m2: HashMap<RefinedInterval<usize>, i32> = vec![
    ///     (RefinedInterval::new(2, 4), 8),
    ///     (RefinedInterval::new(12, 13), 9),
    /// ]
    /// .into_iter()
    /// .collect();
    ///
    /// let mut iter = m1
    ///     .common_refinement_zip(&m2, |a, b| a.get_start().cmp(&b.get_start()))
    ///     .into_iter();
    /// assert_eq!(
    ///     Some((RefinedInterval::new(0, 1), vec![Some(&5), None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(2, 4), vec![Some(&5), Some(&8)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(5, 5), vec![Some(&5), None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(8, 10), vec![Some(&2), None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(12, 13), vec![None, Some(&9)])),
    ///     iter.next()
    /// );
    /// assert_eq!(None, iter.next());
    /// ```
    fn common_refinement_zip<'a, F>(
        &'a self,
        other: &'a Self,
        mut sort_fn: F,
    ) -> CommonRefinementZipped<'a, OrderedIntervalPartitions<T>, RefinedInterval<T>, Self>
    where
        F: FnMut(&RefinedInterval<T>, &RefinedInterval<T>) -> Ordering, {
        let mut keys_1: Vec<RefinedInterval<T>> = self.keys().map(|i| *i).collect();
        keys_1.sort_by(|a, b| sort_fn(a, b));

        let mut keys_2: Vec<RefinedInterval<T>> = other.keys().map(|i| *i).collect();
        keys_2.sort_by(|a, b| sort_fn(a, b));

        let p1 = OrderedIntervalPartitions::from_vec_with_trusted_order(keys_1.clone());
        let p2 = OrderedIntervalPartitions::from_vec_with_trusted_order(keys_2.clone());

        let refined_keys = p1.get_common_refinement(&p2).into_vec();

        CommonRefinementZipped {
            refined_keys,
            keys_list: vec![p1, p2],
            maps: vec![&self, other],
        }
    }
}

impl<'a, V, T: Integer + Copy + Hash>
    IntoCommonRefinementZip<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
    >
    for CommonRefinementZipped<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
    >
{
    /// ```
    /// use analytic::{
    ///     interval::traits::Interval,
    ///     iter::{CommonRefinementZip, IntoCommonRefinementZip, RefinedInterval},
    /// };
    /// use std::collections::HashMap;
    ///
    /// let m1: HashMap<RefinedInterval<usize>, i32> = vec![
    ///     (RefinedInterval::new(0, 10), 5),
    ///     (RefinedInterval::new(16, 17), 21),
    /// ]
    /// .into_iter()
    /// .collect();
    ///
    /// let m2: HashMap<RefinedInterval<usize>, i32> = vec![
    ///     (RefinedInterval::new(2, 3), 8),
    ///     (RefinedInterval::new(12, 20), 9),
    /// ]
    /// .into_iter()
    /// .collect();
    ///
    /// let m3: HashMap<RefinedInterval<usize>, i32> = vec![
    ///     (RefinedInterval::new(2, 4), 7),
    ///     (RefinedInterval::new(9, 10), -1),
    ///     (RefinedInterval::new(15, 20), 0),
    /// ]
    /// .into_iter()
    /// .collect();
    ///
    /// let mut iter = m1
    ///     .common_refinement_zip(&m2, |a, b| a.get_start().cmp(&b.get_start()))
    ///     .into_common_refinement_zip(&m3, |a, b| a.get_start().cmp(&b.get_start()))
    ///     .into_iter();
    ///
    /// assert_eq!(
    ///     Some((RefinedInterval::new(0, 1), vec![Some(&5), None, None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(2, 3), vec![
    ///         Some(&5),
    ///         Some(&8),
    ///         Some(&7)
    ///     ])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(4, 4), vec![Some(&5), None, Some(&7)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(5, 8), vec![Some(&5), None, None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(9, 10), vec![Some(&5), None, Some(&-1)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(12, 14), vec![None, Some(&9), None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(15, 15), vec![None, Some(&9), Some(&0)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(16, 17), vec![
    ///         Some(&21),
    ///         Some(&9),
    ///         Some(&0)
    ///     ])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((RefinedInterval::new(18, 20), vec![None, Some(&9), Some(&0)])),
    ///     iter.next()
    /// );
    /// assert_eq!(None, iter.next());
    /// ```
    fn into_common_refinement_zip<F>(
        self,
        other: &'a HashMap<RefinedInterval<T>, V>,
        mut sort_fn: F,
    ) -> CommonRefinementZipped<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
    >
    where
        F: FnMut(&RefinedInterval<T>, &RefinedInterval<T>) -> Ordering, {
        let mut other_keys: Vec<RefinedInterval<T>> = other.keys().map(|i| *i).collect();
        other_keys.sort_by(|a, b| sort_fn(a, b));

        let other_partitions =
            OrderedIntervalPartitions::from_vec_with_trusted_order(other_keys.clone());
        let refined_keys =
            OrderedIntervalPartitions::from_vec_with_trusted_order(self.refined_keys)
                .get_common_refinement(&other_partitions)
                .into_vec();

        let mut keys_list = self.keys_list;
        keys_list.push(other_partitions);

        let mut maps = self.maps;
        maps.push(other);

        CommonRefinementZipped {
            refined_keys,
            keys_list,
            maps,
        }
    }
}

impl<'a, V, T: Integer + Copy + Hash> IntoIterator
    for CommonRefinementZipped<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
    >
{
    type IntoIter = CommonRefinementZipIter<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
        IntoIter<RefinedInterval<T>>,
    >;
    type Item = <CommonRefinementZipIter<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
        IntoIter<RefinedInterval<T>>,
    > as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        CommonRefinementZipIter {
            refined_keys_iter: self.refined_keys.into_iter(),
            partitions_list: self.keys_list,
            maps: self.maps,
        }
    }
}

impl<'a, V, T: Integer + Copy + Hash> Iterator
    for CommonRefinementZipIter<
        'a,
        OrderedIntervalPartitions<T>,
        RefinedInterval<T>,
        HashMap<RefinedInterval<T>, V>,
        IntoIter<RefinedInterval<T>>,
    >
{
    type Item = (RefinedInterval<T>, Vec<Option<&'a V>>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.refined_keys_iter.next() {
            None => None,
            Some(k) => {
                let mapped: Vec<Option<&'a V>> = self
                    .partitions_list
                    .iter()
                    .zip(self.maps.iter())
                    .map(|(partitions, m)| match partitions.get_set_containing(&k) {
                        None => None,
                        Some(p) => Some(&m[&p]),
                    })
                    .collect();
                Some((k, mapped))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_zip_iter_hashmap() {
        let m1: HashMap<i32, i32> = vec![
            // 0
            (1, 10),
            (3, 23),
            (4, 20),
            // 9
            (12, 6),
            // 14
        ]
        .into_iter()
        .collect();
        let m2: HashMap<i32, i32> = vec![
            (0, 4),
            (1, 20),
            // 3
            (4, 20),
            (9, 29),
            // 14
        ]
        .into_iter()
        .collect();

        let mut iter = m1.union_zip(&m2).into_iter();
        assert_eq!(Some((0, vec![None, Some(&4)])), iter.next());
        assert_eq!(Some((1, vec![Some(&10), Some(&20)])), iter.next());
        assert_eq!(Some((3, vec![Some(&23), None])), iter.next());
        assert_eq!(Some((4, vec![Some(&20), Some(&20)])), iter.next());
        assert_eq!(Some((9, vec![None, Some(&29)])), iter.next());
        assert_eq!(Some((12, vec![Some(&6), None])), iter.next());
        assert_eq!(None, iter.next());

        let m3: HashMap<i32, i32> = vec![
            (0, 9),
            // 1
            (3, 43),
            (4, 8),
            // 9
            // 12
            (14, 68),
        ]
        .into_iter()
        .collect();

        let mut iter2 = m1.union_zip(&m2).into_union_zip(&m3).into_iter();
        assert_eq!(Some((0, vec![None, Some(&4), Some(&9)])), iter2.next());
        assert_eq!(Some((1, vec![Some(&10), Some(&20), None])), iter2.next());
        assert_eq!(Some((3, vec![Some(&23), None, Some(&43)])), iter2.next());
        assert_eq!(
            Some((4, vec![Some(&20), Some(&20), Some(&8)])),
            iter2.next()
        );
        assert_eq!(Some((9, vec![None, Some(&29), None])), iter2.next());
        assert_eq!(Some((12, vec![Some(&6), None, None])), iter2.next());
        assert_eq!(Some((14, vec![None, None, Some(&68)])), iter2.next());
        assert_eq!(None, iter2.next());

        let m4: HashMap<i32, i32> = vec![
            // 0
            // 1
            // 3
            (4, 73),
            // 9
            // 12
            (14, 64),
        ]
        .into_iter()
        .collect();

        let mut iter3 = m1
            .union_zip(&m2)
            .into_union_zip(&m3)
            .into_union_zip(&m4)
            .into_iter();
        assert_eq!(
            Some((0, vec![None, Some(&4), Some(&9), None])),
            iter3.next()
        );
        assert_eq!(
            Some((1, vec![Some(&10), Some(&20), None, None])),
            iter3.next()
        );
        assert_eq!(
            Some((3, vec![Some(&23), None, Some(&43), None])),
            iter3.next()
        );
        assert_eq!(
            Some((4, vec![Some(&20), Some(&20), Some(&8), Some(&73)])),
            iter3.next()
        );
        assert_eq!(Some((9, vec![None, Some(&29), None, None])), iter3.next());
        assert_eq!(Some((12, vec![Some(&6), None, None, None])), iter3.next());
        assert_eq!(
            Some((14, vec![None, None, Some(&68), Some(&64)])),
            iter3.next()
        );
        assert_eq!(None, iter3.next());
    }
}
