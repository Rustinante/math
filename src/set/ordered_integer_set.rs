use crate::{
    interval::traits::{Coalesce, CoalesceIntervals, Interval},
    sample::Sample,
    search::binary_search::BinarySearch,
    set::{
        contiguous_integer_set::{
            ContiguousIntegerSet, ContiguousIntegerSetIter,
        },
        traits::{Finite, Intersect, Set},
    },
    traits::{Collecting, Slicing, ToIterator},
};
use num::{integer::Integer, traits::cast::ToPrimitive, FromPrimitive};
use std::{
    cmp::{min, Ordering},
    iter::Sum,
    ops::Range,
};

pub mod arithmetic;

/// An `OrderedIntegerSet` consists of a sequence of `ContiguousIntegerSet` that
/// are sorted in ascending order where successive intervals are not
/// coalesceable, i.e., if intervals A and B are successive intervals, then
/// A.end + 1 < B.start
///
/// E.g. An `OrderedIntegerSet` containing `ContiguousIntegerSet`s [2,3] and
/// [5,7] will represent the set of integers {2, 3, 5, 6, 7}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedIntegerSet<E: Integer + Copy + ToPrimitive> {
    intervals: Vec<ContiguousIntegerSet<E>>,
}

impl<E: Integer + Copy + ToPrimitive> OrderedIntegerSet<E> {
    pub fn new() -> OrderedIntegerSet<E> {
        OrderedIntegerSet {
            intervals: Vec::new(),
        }
    }

    /// Creates an `OrderedIntegerSet` where the i-th interval is represented by
    /// the i-th two-element array in `slice`.
    ///
    /// E.g. [[2, 3], [5, 7]] will create an `OrderedIntegerSet` representing
    /// {2, 3, 5, 6, 7}, where the contiguous integers are stored as
    /// `ContiguousIntegerSet`s
    ///
    /// Note that the intervals in the `slice` parameters do not have to be
    /// sorted or non-overlapping.
    pub fn from_slice(slice: &[[E; 2]]) -> OrderedIntegerSet<E> {
        let intervals = slice
            .iter()
            .map(|pair| ContiguousIntegerSet::new(pair[0], pair[1]))
            .collect();
        OrderedIntegerSet {
            intervals,
        }
        .into_coalesced()
    }

    pub fn from_contiguous_integer_sets(
        sets: Vec<ContiguousIntegerSet<E>>,
    ) -> OrderedIntegerSet<E> {
        OrderedIntegerSet {
            intervals: sets.into_coalesced(),
        }
    }

    pub fn from_ordered_coalesced_contiguous_integer_sets(
        sets: Vec<ContiguousIntegerSet<E>>,
    ) -> OrderedIntegerSet<E> {
        OrderedIntegerSet {
            intervals: sets,
        }
    }

    /// Returns the smallest element in the set
    /// e.g. {[1,3], [4,8]} -> 1
    pub fn first(&self) -> Option<E> {
        match self.intervals.first() {
            Some(interval) => {
                if interval.is_empty() {
                    None
                } else {
                    Some(interval.get_start())
                }
            }
            None => None,
        }
    }

    /// Returns the largest element in the set
    /// e.g. {[1,3], [4,8]} -> 8
    pub fn last(&self) -> Option<E> {
        match self.intervals.last() {
            Some(interval) => {
                if interval.is_empty() {
                    None
                } else {
                    Some(interval.get_end())
                }
            }
            None => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Returns both the smallest and the largest element in the set
    /// e.g. {[1,3], [4,8]} -> the tuple (1, 8)
    pub fn first_and_last(&self) -> Option<(E, E)> {
        if let Some(first) = self.first() {
            if let Some(last) = self.last() {
                return Some((first, last));
            }
        }
        None
    }

    /// The `slicer` can be any struct that implements the `Slicing` trait.
    /// For example, the `Slicing` trait has been implemented for the
    /// `Range<usize>` struct.
    ///
    /// For an `OrderedIntegerSet` containing n elements, the `Range<usize>`
    /// object created by `a..b` will slice the integer set and return all
    /// the elements from the a-th (inclusive) to the b-th (exclusive) in
    /// the form of an `OrderedIntegerSet`
    pub fn slice<
        'a,
        I: Slicing<&'a OrderedIntegerSet<E>, OrderedIntegerSet<E>>,
    >(
        &'a self,
        slicer: I,
    ) -> OrderedIntegerSet<E> {
        slicer.slice(self)
    }

    #[inline]
    pub fn to_non_empty_intervals(&self) -> Self {
        self.clone().into_non_empty_intervals()
    }

    #[inline]
    pub fn into_non_empty_intervals(mut self) -> Self {
        self.remove_empty_intervals();
        self
    }

    #[inline]
    pub fn remove_empty_intervals(&mut self) {
        self.intervals.drain_filter(|i| i.is_empty());
    }

    #[inline]
    pub fn get_intervals_by_ref(&self) -> &Vec<ContiguousIntegerSet<E>> {
        &self.intervals
    }

    #[inline]
    pub fn into_intervals(self) -> Vec<ContiguousIntegerSet<E>> {
        self.intervals
    }

    #[inline]
    pub fn intervals_iter(&self) -> std::slice::Iter<ContiguousIntegerSet<E>> {
        self.intervals.iter()
    }

    #[inline]
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }
}

impl<E: Integer + Copy + Sum + ToPrimitive> Finite for OrderedIntegerSet<E> {
    #[inline]
    fn size(&self) -> usize {
        self.intervals.iter().map(|&i| i.size()).sum()
    }
}

impl<E: Integer + Copy + ToPrimitive> From<Vec<ContiguousIntegerSet<E>>>
    for OrderedIntegerSet<E>
{
    fn from(intervals: Vec<ContiguousIntegerSet<E>>) -> OrderedIntegerSet<E> {
        OrderedIntegerSet {
            intervals,
        }
        .into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> Set<E> for OrderedIntegerSet<E> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.to_non_empty_intervals().intervals.is_empty()
    }

    fn contains(&self, item: &E) -> bool {
        let item = *item;
        if let Some(first) = self.intervals.first() {
            if first.contains(&item) {
                return true;
            }
        }
        if let Some(last) = self.intervals.last() {
            if last.contains(&item) {
                return true;
            }
        }
        self.intervals
            .iter()
            .filter(|&&interval| interval.contains(&item))
            .count()
            > 0
    }
}

impl<E> Intersect<&OrderedIntegerSet<E>, OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
where
    E: Integer + Copy + ToPrimitive,
{
    fn intersect(&self, other: &OrderedIntegerSet<E>) -> OrderedIntegerSet<E> {
        let mut intersection = Vec::new();
        let rhs_intervals = &other.intervals;
        let rhs_len = rhs_intervals.len();
        let mut j = 0;
        for interval in self.intervals.iter() {
            while j < rhs_len
                && rhs_intervals[j].get_end() < interval.get_start()
            {
                j += 1;
            }
            while j < rhs_len
                && rhs_intervals[j].get_start() <= interval.get_end()
            {
                let rhs_interval = &rhs_intervals[j];
                if let Some(i) = interval.intersect(rhs_interval) {
                    intersection.push(i);
                }
                if rhs_interval.get_end() <= interval.get_end() {
                    j += 1;
                } else {
                    break;
                }
            }
        }
        OrderedIntegerSet::from_contiguous_integer_sets(intersection)
    }

    fn has_non_empty_intersection_with(
        &self,
        other: &OrderedIntegerSet<E>,
    ) -> bool {
        !self.intersect(other).is_empty()
    }
}

impl<E> Intersect<&ContiguousIntegerSet<E>, OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
where
    E: Integer + Copy + ToPrimitive,
{
    #[inline]
    fn intersect(
        &self,
        other: &ContiguousIntegerSet<E>,
    ) -> OrderedIntegerSet<E> {
        other.intersect(self)
    }

    fn has_non_empty_intersection_with(
        &self,
        other: &ContiguousIntegerSet<E>,
    ) -> bool {
        !self.intersect(other).is_empty()
    }
}

impl<E: Integer + Copy + ToPrimitive>
    Intersect<&OrderedIntegerSet<E>, OrderedIntegerSet<E>>
    for ContiguousIntegerSet<E>
{
    fn intersect(&self, other: &OrderedIntegerSet<E>) -> OrderedIntegerSet<E> {
        if self.is_empty() {
            OrderedIntegerSet::new()
        } else {
            let s = OrderedIntegerSet::from_ordered_coalesced_contiguous_integer_sets(vec![
                self.clone()
            ]);
            s.intersect(other)
        }
    }

    fn has_non_empty_intersection_with(
        &self,
        other: &OrderedIntegerSet<E>,
    ) -> bool {
        !self.intersect(other).is_empty()
    }
}

impl<E> CoalesceIntervals<ContiguousIntegerSet<E>, E> for OrderedIntegerSet<E>
where
    E: Integer + Copy + ToPrimitive,
{
    fn to_coalesced_intervals(&self) -> Vec<ContiguousIntegerSet<E>> {
        let mut intervals = self.to_non_empty_intervals().intervals;
        intervals.coalesce_intervals_inplace();
        intervals
    }

    fn coalesce_intervals_inplace(&mut self) {
        self.remove_empty_intervals();
        self.intervals.coalesce_intervals_inplace();
    }
}

impl<E: Integer + Copy + ToPrimitive> Default for OrderedIntegerSet<E> {
    #[inline]
    fn default() -> Self {
        OrderedIntegerSet::new()
    }
}

impl<E: Integer + Copy + ToPrimitive> Collecting<E> for OrderedIntegerSet<E> {
    fn collect(&mut self, item: E) {
        // optimize for the special case where the item is
        // to the right of or coalesceable with the last interval
        if let Some(last_interval) = self.intervals.last_mut() {
            if item > last_interval.get_end() + E::one() {
                self.intervals.push(ContiguousIntegerSet::new(item, item));
                return;
            } else if let Some(interval) = last_interval.coalesce_with(&item) {
                *last_interval = interval;
                return;
            }
        } else {
            // check (1)
            self.intervals.push(ContiguousIntegerSet::new(item, item));
            return;
        }

        // self.intervals.len() is guaranteed to be non-zero because of check
        // (1)
        match self.intervals.binary_search_with_cmp(
            0,
            self.intervals.len(),
            &item,
            |interval, item| {
                if interval.get_start() > *item + E::one() {
                    Ordering::Greater
                } else if interval.get_end() + E::one() < *item {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            },
        ) {
            Ok(i) => {
                self.intervals[i] =
                    self.intervals[i].coalesce_with(&item).unwrap();
                if i > 0 {
                    if let Some(merged) =
                        self.intervals[i - 1].coalesce_with(&self.intervals[i])
                    {
                        self.intervals[i - 1] = merged;
                        self.intervals.remove(i);
                    }
                }
                if let Some(next) = self.intervals.get(i + 1) {
                    if let Some(merged) = next.coalesce_with(&self.intervals[i])
                    {
                        self.intervals[i] = merged;
                        self.intervals.remove(i + 1);
                    }
                }
            }
            Err(interval_index) => {
                // unwrapping is okay due to check (1)
                self.intervals.insert(
                    interval_index.unwrap(),
                    ContiguousIntegerSet::new(item, item),
                );
            }
        };
    }
}

impl<E> Slicing<&OrderedIntegerSet<E>, OrderedIntegerSet<E>> for Range<usize>
where
    E: Integer + Copy + FromPrimitive + ToPrimitive + std::fmt::Debug,
{
    /// the `end` index is exclusive
    fn slice(self, input: &OrderedIntegerSet<E>) -> OrderedIntegerSet<E> {
        if self.start >= self.end {
            return OrderedIntegerSet::new();
        }
        let mut skip = self.start;
        let mut remaining = self.end - self.start;
        let mut contiguous_sets = Vec::new();
        for interval in input.intervals.iter() {
            if remaining == 0 {
                break;
            }
            let size = interval.size();
            if skip > 0 {
                if skip >= size {
                    skip -= size;
                    continue;
                } else {
                    let stop = min(skip + remaining, size);
                    if let Some(s) = interval.slice(skip..stop) {
                        contiguous_sets.push(s);
                    }
                    remaining -= stop - skip;
                    skip = 0;
                }
            } else {
                let increase = min(remaining, size);
                if let Some(s) = interval.slice(0..increase) {
                    contiguous_sets.push(s);
                }
                remaining -= increase;
            }
        }
        OrderedIntegerSet::from_contiguous_integer_sets(contiguous_sets)
    }
}

impl<E> Sample<'_, ContiguousIntegerSetIter<E>, E, OrderedIntegerSet<E>>
    for ContiguousIntegerSet<E>
where
    E: Integer + Copy + ToPrimitive,
{
}

pub struct IntegerSetIter<E: Integer + Copy + ToPrimitive> {
    ordered_integer_set: OrderedIntegerSet<E>,
    current_interval_index: usize,
    current_element_index: E,
}

impl<E: Integer + Copy + ToPrimitive> From<OrderedIntegerSet<E>>
    for IntegerSetIter<E>
{
    fn from(ordered_integer_set: OrderedIntegerSet<E>) -> IntegerSetIter<E> {
        IntegerSetIter {
            ordered_integer_set,
            current_interval_index: 0,
            current_element_index: E::zero(),
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> Iterator for IntegerSetIter<E> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_interval_index
            >= self.ordered_integer_set.intervals.len()
        {
            None
        } else {
            let interval = &self.ordered_integer_set.intervals
                [self.current_interval_index];
            if self.current_element_index.to_usize().unwrap() >= interval.size()
            {
                self.current_interval_index += 1;
                self.current_element_index = E::zero();
                self.next()
            } else {
                let val = interval.get_start() + self.current_element_index;
                self.current_element_index =
                    self.current_element_index + E::one();
                Some(val)
            }
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> ToIterator<'_, IntegerSetIter<E>, E>
    for OrderedIntegerSet<E>
{
    fn to_iter(&self) -> IntegerSetIter<E> {
        IntegerSetIter::from(self.clone())
    }
}

impl<E> Sample<'_, IntegerSetIter<E>, E, OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
where
    E: Integer + Copy + Sum + ToPrimitive,
{
}

#[cfg(test)]
mod tests {
    use num::{integer::Integer, ToPrimitive};

    use crate::{
        interval::traits::*,
        set::traits::{Intersect, Refineable},
        traits::{Collecting, ToIterator},
    };

    use super::{ContiguousIntegerSet, OrderedIntegerSet};

    #[test]
    fn test_ordered_integer_set_iter() {
        let set = OrderedIntegerSet::from_slice(&[[2, 4], [6, 7]]);
        let mut iter = set.to_iter();
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_integer_set_collect() {
        let mut set = OrderedIntegerSet::new();
        set.collect(1);
        set.collect(4);
        set.collect(5);
        set.collect(7);
        set.collect(8);
        set.collect(9);
        assert_eq!(set.into_intervals(), vec![
            ContiguousIntegerSet::new(1, 1),
            ContiguousIntegerSet::new(4, 5),
            ContiguousIntegerSet::new(7, 9)
        ]);

        let mut set =
            OrderedIntegerSet::from_slice(&[[1, 3], [5, 7], [15, 20]]);
        set.collect(-5);
        set.collect(-1);
        set.collect(0);
        set.collect(-10);
        set.collect(4);
        set.collect(10);
        set.collect(12);
        set.collect(13);
        assert_eq!(set.intervals, vec![
            ContiguousIntegerSet::new(-10, -10),
            ContiguousIntegerSet::new(-5, -5),
            ContiguousIntegerSet::new(-1, 7),
            ContiguousIntegerSet::new(10, 10),
            ContiguousIntegerSet::new(12, 13),
            ContiguousIntegerSet::new(15, 20),
        ]);
    }

    #[test]
    fn test_coalesce_with() {
        fn test<E: Copy + Integer + std::fmt::Debug>(
            a: E,
            b: E,
            c: E,
            d: E,
            expected: Option<ContiguousIntegerSet<E>>,
        ) {
            let i1 = ContiguousIntegerSet::new(a, b);
            let i2 = ContiguousIntegerSet::new(c, d);
            let m1 = i1.coalesce_with(&i2);
            let m2 = i2.coalesce_with(&i1);
            assert_eq!(m1, m2);
            assert_eq!(m1, expected);
        }
        test(1, 3, 4, 5, Some(ContiguousIntegerSet::new(1, 5)));
        test(2, 3, 0, 5, Some(ContiguousIntegerSet::new(0, 5)));
        test(2, 5, 1, 3, Some(ContiguousIntegerSet::new(1, 5)));
        test(-3, -1, -1, 2, Some(ContiguousIntegerSet::new(-3, 2)));
        test(3, 5, 7, 9, None);
        test(9, 5, 5, 7, Some(ContiguousIntegerSet::new(5, 7)));
    }

    #[test]
    fn test_sub_contiguous_integer_set() {
        fn test<E: Integer + Copy + ToPrimitive + std::fmt::Debug>(
            a: &[E; 2],
            b: &[E; 2],
            expected: &[[E; 2]],
        ) {
            let s1 = ContiguousIntegerSet::new(a[0], a[1]);
            let s2 = ContiguousIntegerSet::new(b[0], b[1]);
            assert_eq!(s1 - s2, OrderedIntegerSet::from_slice(expected));
        }
        test(&[6, 5], &[-1, 3], &[]);
        test(&[6, 5], &[1, 3], &[]);
        test(&[5, 10], &[3, 1], &[[5, 10]]);
        test(&[5, 8], &[-1, 3], &[[5, 8]]);
        test(&[2, 10], &[4, 9], &[[2, 3], [10, 10]]);
        test(&[2, 10], &[1, 8], &[[9, 10]]);
        test(&[2, 10], &[6, 8], &[[2, 5], [9, 10]]);
        test(&[2, 10], &[2, 10], &[]);
        test(&[2, 10], &[0, 12], &[]);
        test(&[3, 4], &[3, 4], &[]);
        test(&[3, 5], &[3, 3], &[[4, 5]]);
        test(&[3, 4], &[3, 3], &[[4, 4]]);
        test(&[-2, 5], &[-1, 3], &[[-2, -2], [4, 5]]);
        test(&[0usize, 5], &[0, 0], &[[1, 5]]);
        test(&[0usize, 5], &[0, 2], &[[3, 5]]);
        test(&[0usize, 5], &[0, 5], &[]);
    }

    #[test]
    fn test_integer_set_minus_contiguous_integer_set() {
        fn test(a: &[[i32; 2]], b: &[i32; 2], expected: &[[i32; 2]]) {
            let diff = OrderedIntegerSet::from_slice(a)
                - ContiguousIntegerSet::new(b[0], b[1]);
            assert_eq!(diff, OrderedIntegerSet::from_slice(expected));
        }
        test(&[[1, 5], [8, 12], [-4, -2]], &[100, -100], &[
            [-4, -2],
            [1, 5],
            [8, 12],
        ]);
        test(&[[1, 5], [108, 12], [-4, -2]], &[-3, 8], &[[-4, -4]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-3, 8], &[[-4, -4], [9, 12]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-5, 8], &[[9, 12]]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-5, -5], &[
            [-4, -2],
            [1, 5],
            [8, 12],
        ]);
        test(&[[1, 5], [8, 12], [-4, -2]], &[-5, 0], &[[1, 5], [8, 12]]);
        test(&[[1, 5], [8, 12]], &[6, 7], &[[1, 5], [8, 12]]);
        test(&[[1, 5], [8, 12], [25, 100]], &[13, 20], &[
            [1, 5],
            [8, 12],
            [25, 100],
        ]);
    }

    #[test]
    fn test_contiguous_integer_set_minus_integer_set() {
        fn test(a: &[i32; 2], b: &[[i32; 2]], expected: &[[i32; 2]]) {
            let diff = ContiguousIntegerSet::new(a[0], a[1])
                - OrderedIntegerSet::from_slice(b);
            assert_eq!(diff, OrderedIntegerSet::from_slice(expected));
        }
        test(&[1, 12], &[], &[[1, 12]]);
        test(&[1, 12], &[[12, 1]], &[[1, 12]]);
        test(&[1, 12], &[[2, 3], [5, 6]], &[[1, 1], [4, 4], [7, 12]]);
        test(&[1, 12], &[[-1, 3], [10, 13]], &[[4, 9]]);
    }

    #[test]
    fn test_sub_integer_set() {
        fn test(a: &[[i32; 2]], b: &[[i32; 2]], expected: &[[i32; 2]]) {
            let mut diff = OrderedIntegerSet::from_slice(a)
                - OrderedIntegerSet::from_slice(b);
            diff.coalesce_intervals_inplace();
            assert_eq!(diff, OrderedIntegerSet::from_slice(expected));
        }
        test(&[[1, 10]], &[[1, 3], [5, 7]], &[[4, 4], [8, 10]]);
        test(&[[0, 10]], &[[1, 3], [5, 7]], &[[0, 0], [4, 4], [8, 10]]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [5, 7]], &[
            [3, 4],
            [8, 10],
            [15, 20],
        ]);
        test(&[[0, 10], [15, 20]], &[[-1, 2], [18, 22], [5, 7]], &[
            [3, 4],
            [8, 10],
            [15, 17],
        ]);
        test(
            &[[0, 10], [15, 20], [-10, -5]],
            &[[-1, 2], [18, 22], [5, 7], [-12, -3]],
            &[[3, 4], [8, 10], [15, 17]],
        );
    }

    #[test]
    fn test_contiguous_ordered_integer_set_intersect() {
        fn test<E: Integer + Copy + ToPrimitive + std::fmt::Debug>(
            a: &[E; 2],
            b: &[[E; 2]],
            expected: &[[E; 2]],
        ) {
            let s1 = ContiguousIntegerSet::new(a[0], a[1]);
            let s2 = OrderedIntegerSet::from_slice(b);
            assert_eq!(
                s1.intersect(&s2),
                OrderedIntegerSet::from_slice(expected)
            );
            assert_eq!(
                s2.intersect(&s1),
                OrderedIntegerSet::from_slice(expected)
            );
        }
        test(&[0usize, 10], &[[2, 5]], &[[2, 5]]);
        test(&[-3, 10], &[[-5, 12]], &[[-3, 10]]);
        test(&[-5, 12], &[[-3, 10]], &[[-3, 10]]);
        test(&[5, 10], &[[3, 6]], &[[5, 6]]);
        test(&[3, 6], &[[5, 10]], &[[5, 6]]);
        test(&[0usize, 10], &[[0, 2], [8, 9]], &[[0, 2], [8, 9]]);
        test(&[0usize, 10], &[[0, 12]], &[[0, 10]]);
        test(&[0usize, 10], &[[12, 13], [15, 20]], &[]);
        test(&[3usize, 10], &[[0, 3], [9, 12]], &[[3, 3], [9, 10]]);
    }

    #[test]
    fn test_intersect_integer_set() {
        fn test<E: Integer + Copy + ToPrimitive + std::fmt::Debug>(
            a: &[[E; 2]],
            b: &[[E; 2]],
            expected: &[[E; 2]],
        ) {
            let s1 = OrderedIntegerSet::from_slice(a);
            let s2 = OrderedIntegerSet::from_slice(b);
            assert_eq!(s1 - s2, OrderedIntegerSet::from_slice(expected));
        }
        test(&[[0usize, 5], [10, 15]], &[[0, 4]], &[[5, 5], [10, 15]]);
        test(&[[0usize, 5], [10, 15]], &[[0, 12]], &[[13, 15]]);
        test(&[[0usize, 10]], &[[0, 8]], &[[9, 10]]);
    }

    #[test]
    fn test_get_common_refinement_contiguous_integer_set() {
        fn test<E: Integer + Copy + ToPrimitive + std::fmt::Debug>(
            a: &[E; 2],
            b: &[E; 2],
            expected: &[[E; 2]],
        ) {
            let s1 = ContiguousIntegerSet::new(a[0], a[1]);
            let s2 = ContiguousIntegerSet::new(b[0], b[1]);
            let expected = expected
                .iter()
                .map(|[a, b]| ContiguousIntegerSet::new(*a, *b))
                .collect::<Vec<ContiguousIntegerSet<E>>>();
            assert_eq!(s1.get_common_refinement(&s2), expected);
            assert_eq!(s2.get_common_refinement(&s1), expected);
        }
        test(&[0usize, 3], &[4, 5], &[[0, 3], [4, 5]]);
        test(&[0usize, 3], &[3, 5], &[[0, 2], [3, 3], [4, 5]]);
        test(&[0usize, 4], &[2, 5], &[[0, 1], [2, 4], [5, 5]]);
        test(&[0usize, 4], &[2, 10], &[[0, 1], [2, 4], [5, 10]]);
        test(&[0usize, 4], &[0, 3], &[[0, 3], [4, 4]]);
        test(&[0usize, 4], &[0, 8], &[[0, 4], [5, 8]]);
        test(&[0usize, 6], &[2, 3], &[[0, 1], [2, 3], [4, 6]]);
        test(&[0usize, 8], &[0, 8], &[[0, 8]]);
        test(&[0i32, 8], &[0, 8], &[[0, 8]]);
        test(&[-2i32, 4], &[0, 3], &[[-2, -1], [0, 3], [4, 4]]);
        test(&[-2i32, 4], &[0, 3], &[[-2, -1], [0, 3], [4, 4]]);
    }
}
