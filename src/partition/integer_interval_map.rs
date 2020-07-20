//! Maps integer intervals to their associated values.

use crate::set::{
    contiguous_integer_set::ContiguousIntegerSet, ordered_integer_set::OrderedIntegerSet,
    traits::Intersect,
};
use num::Num;
use std::{collections::BTreeMap, fmt::Debug};

pub type IntegerInterval = ContiguousIntegerSet<i64>;

/// Maps `IntegerInterval`s to values of a numeric type `T`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegerIntervalMap<T> {
    map: BTreeMap<IntegerInterval, T>,
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
    /// use analytic::partition::integer_interval_map::{IntegerInterval, IntegerIntervalMap};
    ///
    /// // -1 0 1 2 3 4             +2
    /// //                6 7 8     +4
    /// //            4 5 6 7       +1
    /// //----------------------------
    /// //  2 2 2 2 2 3 1 5 5 4
    ///
    /// let mut interval_map = IntegerIntervalMap::new();
    /// interval_map.aggregate(IntegerInterval::new(-1, 4), 2);
    /// interval_map.aggregate(IntegerInterval::new(6, 8), 4);
    /// interval_map.aggregate(IntegerInterval::new(4, 7), 1);
    ///
    /// assert_eq!(interval_map.get(&IntegerInterval::new(-1, 3)), Some(2));
    /// assert_eq!(interval_map.get(&IntegerInterval::new(4, 4)), Some(3));
    /// assert_eq!(interval_map.get(&IntegerInterval::new(5, 5)), Some(1));
    /// assert_eq!(interval_map.get(&IntegerInterval::new(6, 7)), Some(5));
    /// assert_eq!(interval_map.get(&IntegerInterval::new(8, 8)), Some(4));
    /// assert_eq!(interval_map.get(&IntegerInterval::new(-1, 4)), None);
    /// assert_eq!(interval_map.get(&IntegerInterval::new(6, 8)), None);
    /// assert_eq!(interval_map.get(&IntegerInterval::new(4, 7)), None);
    /// ```
    pub fn aggregate(&mut self, key: IntegerInterval, value: T) {
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
    /// use analytic::partition::integer_interval_map::{IntegerInterval, IntegerIntervalMap};
    ///
    /// let mut interval_map = IntegerIntervalMap::new();
    /// interval_map.aggregate(IntegerInterval::new(-1, 4), 2);
    /// interval_map.aggregate(IntegerInterval::new(6, 8), 4);
    /// interval_map.aggregate(IntegerInterval::new(4, 7), 1);
    ///
    /// let expected = vec![
    ///     (IntegerInterval::new(-1, 3), 2),
    ///     (IntegerInterval::new(4, 4), 3),
    ///     (IntegerInterval::new(5, 5), 1),
    ///     (IntegerInterval::new(6, 7), 5),
    ///     (IntegerInterval::new(8, 8), 4),
    /// ];
    /// for ((interval, val), (expected_interval, exptected_val)) in
    ///     interval_map.iter().zip(expected.iter())
    /// {
    ///     assert_eq!(interval, expected_interval);
    ///     assert_eq!(val, exptected_val);
    /// }
    /// ```
    pub fn iter(&self) -> std::collections::btree_map::Iter<IntegerInterval, T> {
        self.map.iter()
    }

    /// Converts into the underlying `BTreeMap`
    pub fn into_map(self) -> BTreeMap<IntegerInterval, T> {
        self.map
    }

    /// Returns a `Some` value only if the key corresponds to one of the current exact intervals
    /// and not its subset or superset.
    ///
    /// # Example
    /// ```
    /// use analytic::partition::integer_interval_map::{IntegerInterval, IntegerIntervalMap};
    ///
    /// let mut interval_map = IntegerIntervalMap::new();
    /// interval_map.aggregate(IntegerInterval::new(2, 5), 1);
    /// assert_eq!(interval_map.get(&IntegerInterval::new(2, 5)), Some(1));
    /// assert_eq!(interval_map.get(&IntegerInterval::new(2, 4)), None);
    /// assert_eq!(interval_map.get(&IntegerInterval::new(2, 6)), None);
    /// ```
    pub fn get(&self, key: &IntegerInterval) -> Option<T> {
        self.map.get(key).map(|&k| k)
    }
}

impl<T: Copy + Num + Debug> Default for IntegerIntervalMap<T> {
    fn default() -> Self {
        Self::new()
    }
}
