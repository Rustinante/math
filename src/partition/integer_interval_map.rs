//! Maps integer intervals to their associated values

use crate::{
    interval::{traits::Interval, I64Interval},
    iter::{CommonRefinementZip, CommonRefinementZipped},
    set::{
        contiguous_integer_set::ContiguousIntegerSet, ordered_integer_set::OrderedIntegerSet,
        traits::Intersect,
    },
    traits::SubsetIndexable,
};
use num::Num;
use std::{collections::BTreeMap, fmt::Debug};

/// Maps `I64Interval`s to values of a numeric type `T`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegerIntervalMap<T> {
    map: BTreeMap<I64Interval, T>,
}

impl<T: Copy + Num> IntegerIntervalMap<T> {
    pub fn new() -> Self {
        IntegerIntervalMap {
            map: BTreeMap::new(),
        }
    }

    /// Adds an integer interval as the `key` with an associated `value`.
    /// Any existing intervals intersecting the `key` will be broken up,
    /// where the region of intersection will have a value being the sum of
    /// the existing value and the new `value`, while the non-intersecting regions
    /// will retain their original values.
    ///
    /// # Example
    /// ```
    /// use analytic::{interval::I64Interval, partition::integer_interval_map::IntegerIntervalMap};
    ///
    /// //                          | value
    /// // -1 0 1 2 3 4             | +2
    /// //                6 7 8     | +4
    /// //            4 5 6 7       | +1
    /// //---------------------------
    /// //  2 2 2 2 2 3 1 5 5 4     | superposed values
    ///
    /// let mut interval_map = IntegerIntervalMap::new();
    /// interval_map.aggregate(I64Interval::new(-1, 4), 2);
    /// interval_map.aggregate(I64Interval::new(6, 8), 4);
    /// interval_map.aggregate(I64Interval::new(4, 7), 1);
    ///
    /// assert_eq!(interval_map.get(&I64Interval::new(-1, 3)), Some(2));
    /// assert_eq!(interval_map.get(&I64Interval::new(4, 4)), Some(3));
    /// assert_eq!(interval_map.get(&I64Interval::new(5, 5)), Some(1));
    /// assert_eq!(interval_map.get(&I64Interval::new(6, 7)), Some(5));
    /// assert_eq!(interval_map.get(&I64Interval::new(8, 8)), Some(4));
    /// assert_eq!(interval_map.get(&I64Interval::new(-1, 4)), None);
    /// assert_eq!(interval_map.get(&I64Interval::new(6, 8)), None);
    /// assert_eq!(interval_map.get(&I64Interval::new(4, 7)), None);
    /// ```
    pub fn aggregate(&mut self, key: I64Interval, value: T) {
        let (start, end) = key.get_start_and_end();
        let mut remaining_interval = OrderedIntegerSet::from_contiguous_integer_sets(vec![key]);
        let mut to_add = Vec::new();
        let mut to_remove = Vec::new();

        // All intervals in the range (start, start)..(end + 1, end + 1) intersect with the key
        // due to the lexicographical ordering of the ContiguousIntegerSet.
        // Furthermore, there can be at most one interval whose start is less than the start of
        // the key, and which intersects the key.
        for (&interval, &val) in self
            .map
            .range(
                ContiguousIntegerSet::new(start, start)
                    ..ContiguousIntegerSet::new(end + 1, end + 1),
            )
            .chain(
                self.map
                    .range(..ContiguousIntegerSet::new(start, start))
                    .rev()
                    .take(1),
            )
        {
            to_remove.push(interval);

            let intersection = interval.intersect(&remaining_interval);
            for &common_interval in intersection.get_intervals_by_ref().iter() {
                remaining_interval -= common_interval;
                to_add.push((common_interval, val + value));
            }
            for outstanding_interval in (interval - intersection).into_intervals() {
                to_add.push((outstanding_interval, val));
            }
        }
        for i in remaining_interval
            .into_non_empty_intervals()
            .into_intervals()
            .into_iter()
        {
            to_add.push((i, value));
        }

        // remove the old and add the new
        for i in to_remove.into_iter() {
            self.map.remove(&i);
        }
        for (k, v) in to_add.into_iter() {
            self.map.insert(k, v);
        }
    }

    /// # Example
    /// ```
    /// use analytic::{interval::I64Interval, partition::integer_interval_map::IntegerIntervalMap};
    ///
    /// let mut interval_map = IntegerIntervalMap::new();
    /// interval_map.aggregate(I64Interval::new(-1, 4), 2);
    /// interval_map.aggregate(I64Interval::new(6, 8), 4);
    /// interval_map.aggregate(I64Interval::new(4, 7), 1);
    ///
    /// let expected = vec![
    ///     (I64Interval::new(-1, 3), 2),
    ///     (I64Interval::new(4, 4), 3),
    ///     (I64Interval::new(5, 5), 1),
    ///     (I64Interval::new(6, 7), 5),
    ///     (I64Interval::new(8, 8), 4),
    /// ];
    /// for ((interval, val), (expected_interval, exptected_val)) in
    ///     interval_map.iter().zip(expected.iter())
    /// {
    ///     assert_eq!(interval, expected_interval);
    ///     assert_eq!(val, exptected_val);
    /// }
    /// ```
    pub fn iter(&self) -> std::collections::btree_map::Iter<I64Interval, T> {
        self.map.iter()
    }

    /// Converts into the underlying `BTreeMap`
    pub fn into_map(self) -> BTreeMap<I64Interval, T> {
        self.map
    }

    /// Returns a `Some` value only if the key corresponds to one of the current exact intervals
    /// and not its subset or superset.
    ///
    /// # Example
    /// ```
    /// use analytic::{interval::I64Interval, partition::integer_interval_map::IntegerIntervalMap};
    ///
    /// let mut interval_map = IntegerIntervalMap::new();
    /// interval_map.aggregate(I64Interval::new(2, 5), 1);
    /// assert_eq!(interval_map.get(&I64Interval::new(2, 5)), Some(1));
    /// assert_eq!(interval_map.get(&I64Interval::new(2, 4)), None);
    /// assert_eq!(interval_map.get(&I64Interval::new(2, 6)), None);
    /// ```
    pub fn get(&self, key: &I64Interval) -> Option<T> {
        self.map.get(key).map(|&k| k)
    }
}

impl<T: Copy + Num + Debug> Default for IntegerIntervalMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

type IntType = i64;
//
// impl<T: Integer + Copy + Hash>
//     CommonRefinementZip<OrderedIntervalPartitions<IntType>, I64Interval, Self>
//     for IntegerIntervalMap<T>
// {
//     fn common_refinement_zip<'a, F>(
//         &'a self,
//         other: &'a Self,
//         _: F,
//     ) -> CommonRefinementZipped<'a, OrderedIntervalPartitions<i64>, I64Interval, Self> {
//         let keys_1: Vec<I64Interval> = self.iter().map(|(&key, _)| key).collect();
//         let keys_2: Vec<I64Interval> = self.iter().map(|(&key, _)| key).collect();
//
//         let p1 = OrderedIntervalPartitions::from_vec_with_trusted_order(keys_1);
//         let p2 = OrderedIntervalPartitions::from_vec_with_trusted_order(keys_2);
//
//         let refined_keys = p1.get_common_refinement(&p2).into_vec();
//
//         CommonRefinementZipped {
//             keys_list: vec![p1, p2],
//             refined_keys,
//             maps: vec![&self, &other],
//         }
//     }
// }

// impl<'a, T: Integer + Copy + Hash> IntoIterator
//     for CommonRefinementZipped<
//         'a,
//         OrderedIntervalPartitions<IntType>,
//         I64Interval,
//         IntegerIntervalMap<T>,
//     >
// {
//     type IntoIter = CommonRefinementZipIter<
//         'a,
//         OrderedIntervalPartitions<T>,
//         I64Interval,
//         IntegerIntervalMap<T>,
//         IntoIter<I64Interval>,
//     >;
//     type Item = <CommonRefinementZipIter<
//         'a,
//         OrderedIntervalPartitions<T>,
//         I64Interval,
//         IntegerIntervalMap<T>,
//         IntoIter<I64Interval>,
//     > as Iterator>::Item;
//
//     fn into_iter(self) -> Self::IntoIter {
//         CommonRefinementZipIter {
//             refined_keys_iter: self.refined_keys.into_iter(),
//             partitions_list: self.keys_list,
//             maps: self.maps,
//         }
//     }
// }
//
// impl<'a, T: Copy + Num> Iterator
//     for CommonRefinementZipIter<
//         'a,
//         OrderedIntervalPartitions<IntType>,
//         I64Interval,
//         IntegerIntervalMap<T>,
//         IntoIter<I64Interval>,
//     >
// {
//     type Item = (I64Interval, Vec<Option<T>>);
//
//     fn next(&mut self) -> Option<Self::Item> {
//         match self.refined_keys_iter.next() {
//             None => None,
//             Some(k) => {
//                 let mapped: Vec<Option<T>> = self
//                     .partitions_list
//                     .iter()
//                     .zip(self.maps.iter())
//                     .map(|(partitions, &m)| match partitions.get_set_containing(&k) {
//                         None => None,
//                         Some(p) => Some(m.get(&p).unwrap()),
//                     })
//                     .collect();
//                 Some((k, mapped))
//             }
//         }
//     }
// }

impl<T> SubsetIndexable<I64Interval, I64Interval> for IntegerIntervalMap<T> {
    fn get_set_containing(&self, subset: &I64Interval) -> Option<I64Interval> {
        let start = subset.get_start();
        // the containing interval must be < (start + 1, start + 1) lexicographically
        for (interval, _) in self
            .map
            .range(..I64Interval::new(start + 1, start + 1))
            .rev()
        {
            if subset.is_subset_of(interval) {
                return Some(*interval);
            }
            if interval.get_end() < start {
                return None;
            }
        }
        return None;
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::{
//         interval::I64Interval, iter::CommonRefinementZip,
//         partition::integer_interval_map::IntegerIntervalMap,
//     };
//     use std::cmp::Ordering;
//
//     #[test]
//     fn test_common_refinement_zip_integer_interval_map() {
//         let mut map1 = IntegerIntervalMap::new();
//         map1.aggregate(I64Interval::new(1, 5), 1);
//         let mut map2 = IntegerIntervalMap::new();
//         map2.aggregate(I64Interval::new(3, 6), 2);
//
//         let refined: Vec<(I64Interval, Vec<Option<i32>>)> = map1
//             .common_refinement_zip(&map2, |_, _| Ordering::Less)
//             .into_iter()
//             .collect();
//
//         let expected = vec![(I64Interval::new(1, 2), vec![Some(&1), None])];
//         assert_eq!(refined, expected);
//     }
// }
