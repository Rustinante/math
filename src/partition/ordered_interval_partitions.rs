use std::cmp::Ordering;
use std::hash::Hash;

use num::{Integer, ToPrimitive};

use crate::interval::traits::Interval;
use crate::search::binary_search::BinarySearch;
use crate::set::ordered_integer_set::{ContiguousIntegerSet, OrderedIntegerSet};
use crate::set::traits::{Intersect, Refineable};
use crate::traits::SubsetIndexable;

#[derive(Clone, PartialEq, Debug)]
pub struct OrderedIntervalPartitions<E: Integer + Copy> {
    partitions: Vec<ContiguousIntegerSet<E>>,
}

impl<E: Integer + Copy> OrderedIntervalPartitions<E> {
    pub fn from_vec(mut partitions: Vec<ContiguousIntegerSet<E>>)
        -> OrderedIntervalPartitions<E> {
        partitions.sort_by_key(|p| p.get_start());
        OrderedIntervalPartitions { partitions }
    }

    #[inline]
    pub fn from_vec_with_trusted_order(partitions: Vec<ContiguousIntegerSet<E>>)
        -> OrderedIntervalPartitions<E> {
        OrderedIntervalPartitions { partitions }
    }

    pub fn from_slice(slice: &[[E; 2]]) -> OrderedIntervalPartitions<E> {
        OrderedIntervalPartitions::from_vec(slice.iter()
                                                 .map(|s| ContiguousIntegerSet::new(s[0], s[1]))
                                                 .collect())
    }

    /// returns the number of partitions that intersect with the `other`'s partitions
    pub fn num_partitions_overlapped_by(&self, other: &OrderedIntervalPartitions<E>) -> usize {
        let rhs_len = other.num_partitions();
        let mut j = 0;
        let mut num_overlaps = 0;
        for interval in self.partitions.iter() {
            while j < rhs_len && other.partitions[j].get_end() < interval.get_start() {
                j += 1;
            }
            while j < rhs_len && other.partitions[j].get_start() <= interval.get_end() {
                let rhs_interval = &other.partitions[j];
                if interval.intersect(&rhs_interval).is_some() {
                    num_overlaps += 1;
                    break;
                }
                if rhs_interval.get_end() <= interval.get_end() {
                    j += 1;
                } else {
                    break;
                }
            }
        }
        num_overlaps
    }

    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    #[inline]
    pub fn get_partitions_by_ref(&self) -> &Vec<ContiguousIntegerSet<E>> {
        &self.partitions
    }

    #[inline]
    pub fn into_vec(self) -> Vec<ContiguousIntegerSet<E>> {
        self.partitions
    }
}

impl<E: Integer + Copy + ToPrimitive> OrderedIntervalPartitions<E> {
    pub fn to_ordered_integer_set(&self) -> OrderedIntegerSet<E> {
        OrderedIntegerSet::from_contiguous_integer_sets(self.partitions.clone())
    }

    pub fn into_ordered_integer_set(self) -> OrderedIntegerSet<E> {
        OrderedIntegerSet::from_contiguous_integer_sets(self.into_vec())
    }
}

impl<E: Integer + Copy + Hash> SubsetIndexable<ContiguousIntegerSet<E>> for OrderedIntervalPartitions<E> {
    #[inline]
    fn get_set_containing(&self, subset: &ContiguousIntegerSet<E>) -> Option<ContiguousIntegerSet<E>> {
        match self.get_partition_containing(subset) {
            None => None,
            Some(s) => Some(s.1)
        }
    }
}

impl<E: Integer + Copy + Hash> OrderedIntervalPartitions<E> {
    /// if there is a partition containing the subinterval, returns a tuple `(partition_index, partition)`
    pub fn get_partition_containing(
        &self,
        subinterval: &ContiguousIntegerSet<E>,
    ) -> Option<(usize, ContiguousIntegerSet<E>)> {
        match self.partitions.binary_search_with_cmp(
            0, self.partitions.len(), subinterval, |interval, subinterval| {
                if subinterval.is_subset_of(interval) {
                    Ordering::Equal
                } else if interval.get_start() > subinterval.get_start() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }) {
            Ok(i) => Some((i, self.partitions[i])),
            Err(_) => None
        }
    }
}

impl<E: Integer + Copy> Refineable<OrderedIntervalPartitions<E>> for OrderedIntervalPartitions<E> {
    /// # Example
    /// ```
    /// use analytic::partition::ordered_interval_partitions::OrderedIntervalPartitions;
    /// use analytic::set::traits::Refineable;
    /// use analytic::set::ordered_integer_set::ContiguousIntegerSet;
    /// let p1 = OrderedIntervalPartitions::from_slice(&[[-1, 4], [8, 10]]);
    /// let p2 = OrderedIntervalPartitions::from_slice(&[[3, 7]]);
    /// assert_eq!(p1.get_common_refinement(&p2).into_vec(), [[-1, 2], [3, 4], [5, 7], [8, 10]]
    ///                                               .iter()
    ///                                               .map(|[a, b]| ContiguousIntegerSet::new(*a, *b))
    ///                                               .collect::<Vec<ContiguousIntegerSet<i32>>>());
    /// ```
    fn get_common_refinement(&self, other: &OrderedIntervalPartitions<E>) -> OrderedIntervalPartitions<E> {
        let mut i = 0;
        let mut j = 0;
        let end_i = self.num_partitions();
        let end_j = other.num_partitions();
        let mut refinement = Vec::new();
        let mut lhs_remnant = None;
        let mut rhs_remnant = None;
        while i < end_i && j < end_j {
            let lhs = lhs_remnant.unwrap_or(self.partitions[i]);
            let rhs = rhs_remnant.unwrap_or(other.partitions[j]);
            match lhs.intersect(&rhs) {
                None => {
                    if lhs.get_start() <= rhs.get_start() {
                        refinement.push(lhs);
                        i += 1;
                        lhs_remnant = None;
                    } else {
                        refinement.push(rhs);
                        j += 1;
                        rhs_remnant = None;
                    }
                }
                Some(intersection) => {
                    if lhs.get_start() < intersection.get_start() {
                        refinement.push(
                            ContiguousIntegerSet::new(lhs.get_start(), intersection.get_start() - E::one())
                        );
                    }
                    if rhs.get_start() < intersection.get_start() {
                        refinement.push(
                            ContiguousIntegerSet::new(rhs.get_start(), intersection.get_start() - E::one())
                        );
                    }
                    refinement.push(intersection);
                    if lhs.get_end() > intersection.get_end() {
                        lhs_remnant = Some(
                            ContiguousIntegerSet::new(intersection.get_end() + E::one(), lhs.get_end())
                        );
                    } else {
                        lhs_remnant = None;
                        i += 1;
                    }
                    if rhs.get_end() > intersection.get_end() {
                        rhs_remnant = Some(
                            ContiguousIntegerSet::new(intersection.get_end() + E::one(), rhs.get_end())
                        );
                    } else {
                        rhs_remnant = None;
                        j += 1;
                    }
                }
            }
        }
        if let Some(r) = lhs_remnant {
            refinement.push(r);
            i += 1;
        }
        if let Some(r) = rhs_remnant {
            refinement.push(r);
            j += 1;
        }
        while i < end_i {
            refinement.push(self.partitions[i]);
            i += 1;
        }
        while j < end_j {
            refinement.push(other.partitions[j]);
            j += 1;
        }
        OrderedIntervalPartitions::from_vec_with_trusted_order(refinement)
    }
}

#[cfg(test)]
mod tests {
    use num::Integer;

    use crate::partition::ordered_interval_partitions::OrderedIntervalPartitions;
    use crate::set::ordered_integer_set::ContiguousIntegerSet;
    use crate::set::traits::Refineable;

    #[test]
    fn test_ordered_interval_partitions_common_refinement() {
        fn test<E: Integer + Copy + std::fmt::Debug>(a: &[[E; 2]], b: &[[E; 2]], expected: &[[E; 2]]) {
            let s1 = OrderedIntervalPartitions::from_slice(a);
            let s2 = OrderedIntervalPartitions::from_slice(b);
            let expected = expected.iter()
                                   .map(|[a, b]| ContiguousIntegerSet::new(*a, *b))
                                   .collect::<Vec<ContiguousIntegerSet<E>>>();
            assert_eq!(s1.get_common_refinement(&s2).into_vec(), expected);
            assert_eq!(s2.get_common_refinement(&s1).into_vec(), expected);
        }
        test::<usize>(&[], &[], &[]);
        test(&[[0usize, 4], [6, 10]], &[[1, 2], [4, 6]], &[[0, 0], [1, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 10]]);
        test(&[[0usize, 5], [8, 10]], &[[6, 7], [11, 13]], &[[0, 5], [6, 7], [8, 10], [11, 13]]);
        test(&[[-1, 5]], &[[-2, 0]], &[[-2, -2], [-1, 0], [1, 5]]);
        test(&[[0usize, 10], [15, 23]], &[[3, 5], [20, 23]], &[[0, 2], [3, 5], [6, 10], [15, 19], [20, 23]]);
        test(&[[0usize, 10], [15, 23]], &[[3, 5], [15, 23]], &[[0, 2], [3, 5], [6, 10], [15, 23]]);
        test(&[[0usize, 10], [15, 23]], &[[3, 5], [15, 20]], &[[0, 2], [3, 5], [6, 10], [15, 20], [21, 23]]);
        test(&[[1, 100]], &[[3, 5], [23, 30]], &[[1, 2], [3, 5], [6, 22], [23, 30], [31, 100]]);
        test(&[[2usize, 6], [8, 12]], &[[0, 3], [5, 13]], &[[0, 1], [2, 3], [4, 4], [5, 6], [7, 7], [8, 12], [13, 13]]);
        test(&[[0usize, 3], [4, 8]], &[[2, 4]], &[[0, 1], [2, 3], [4, 4], [5, 8]]);
        test(&[[0usize, 4], [5, 7]], &[], &[[0, 4], [5, 7]]);
    }

    #[test]
    fn test_num_overlapped_partitions_by() {
        fn test<E: Integer + Copy + std::fmt::Debug>(a: &[[E; 2]], b: &[[E; 2]], expected: usize) {
            let s1 = OrderedIntervalPartitions::from_slice(a);
            let s2 = OrderedIntervalPartitions::from_slice(b);
            assert_eq!(s1.num_partitions_overlapped_by(&s2), expected);
        }
        test::<usize>(&[], &[], 0);
        test::<usize>(&[[0, 4]], &[], 0);
        test::<usize>(&[[0, 4], [7, 9]], &[], 0);
        test::<usize>(&[[0, 4]], &[[0, 0]], 1);
        test::<usize>(&[[0, 4]], &[[0, 4]], 1);
        test::<usize>(&[[0, 4]], &[[3, 4]], 1);
        test::<usize>(&[[0, 4]], &[[4, 4]], 1);
        test::<usize>(&[[0, 4]], &[[0, 0], [3, 4]], 1);
        test::<usize>(&[[0, 4], [7, 10]], &[[0, 10]], 2);
        test::<usize>(&[[0, 4], [7, 10]], &[[0, 3], [5, 10]], 2);
        test::<usize>(&[[0, 4], [7, 10]], &[[0, 5], [8, 10]], 2);
        test::<usize>(&[[0, 4], [7, 10]], &[[0, 12]], 2);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[6, 15]], 2);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[0, 15]], 3);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[0, 4], [7, 10], [15, 30]], 3);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[0, 8], [15, 30]], 3);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[0, 8]], 2);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[5, 6]], 0);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[[12, 14]], 0);
        test::<usize>(&[[0, 4], [7, 10], [15, 30]], &[], 0);
    }
}
