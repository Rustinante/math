use std::cmp::{Ordering, Reverse};
use std::hash::Hash;

use num::Integer;

use crate::interval::traits::Interval;
use crate::search::binary_search::BinarySearch;
use crate::set::ordered_integer_set::ContiguousIntegerSet;

#[derive(Copy, Clone, Debug)]
pub enum PartitionOrder {
    AscendingStart,
    AscendingEnd,
    DescendingStart,
    DescendingEnd,
}

use PartitionOrder::*;

pub struct OrderedIntervalPartitions<E: Integer + Copy> {
    partitions: Vec<ContiguousIntegerSet<E>>,
    order: PartitionOrder,
}

impl<E: Integer + Copy> OrderedIntervalPartitions<E> {
    pub fn from_vec(mut partitions: Vec<ContiguousIntegerSet<E>>, desired_order: PartitionOrder) -> OrderedIntervalPartitions<E> {
        match desired_order {
            AscendingStart => partitions.sort_by_key(|p| p.get_start()),
            AscendingEnd => partitions.sort_by_key(|p| p.get_end()),
            DescendingStart => partitions.sort_by_key(|p| Reverse(p.get_start())),
            DescendingEnd => partitions.sort_by_key(|p| Reverse(p.get_end())),
        };
        OrderedIntervalPartitions {
            partitions,
            order: desired_order,
        }
    }

    #[inline]
    pub fn from_vec_with_trusted_order(partitions: Vec<ContiguousIntegerSet<E>>, existing_order: PartitionOrder) -> OrderedIntervalPartitions<E> {
        OrderedIntervalPartitions { partitions, order: existing_order }
    }

    #[inline]
    pub fn get_order(&self) -> PartitionOrder {
        self.order
    }
}

impl<E: Integer + Copy + Hash> OrderedIntervalPartitions<E> {
    /// if there is a partition containing the subinterval, returns a tuple `(partition_index, partition)`
    pub fn get_partition_containing(&self, subinterval: &ContiguousIntegerSet<E>) -> Option<(usize, ContiguousIntegerSet<E>)> {
        match
            match self.order {
                AscendingStart => self.partitions.binary_search_with_cmp(
                    0, self.partitions.len(), subinterval, |interval, subinterval| {
                        if subinterval.is_subset_of(interval) {
                            Ordering::Equal
                        } else if interval.get_start() > subinterval.get_start() {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }),
                AscendingEnd => self.partitions.binary_search_with_cmp(
                    0, self.partitions.len(), subinterval, |interval, subinterval| {
                        if subinterval.is_subset_of(interval) {
                            Ordering::Equal
                        } else if interval.get_end() > subinterval.get_end() {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }),
                DescendingStart => self.partitions.binary_search_with_cmp(
                    0, self.partitions.len(), subinterval, |interval, subinterval| {
                        if subinterval.is_subset_of(interval) {
                            Ordering::Equal
                        } else if interval.get_start() > subinterval.get_start() {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    }),
                DescendingEnd => self.partitions.binary_search_with_cmp(
                    0, self.partitions.len(), subinterval, |interval, subinterval| {
                        if subinterval.is_subset_of(interval) {
                            Ordering::Equal
                        } else if interval.get_end() > subinterval.get_end() {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    }),
            } {
            Ok(i) => Some((i, self.partitions[i])),
            Err(_) => None
        }
    }
}
