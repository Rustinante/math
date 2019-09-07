use std::ops::Index;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};

use crate::set::ordered_integer_set::{ContiguousIntegerSet, OrderedIntegerSet};

pub type Partition = OrderedIntegerSet<usize>;

#[derive(Clone, PartialEq, Debug)]
pub struct IntegerPartitions {
    partitions: Vec<Partition>
}

impl IntegerPartitions {
    pub fn new(partitions: Vec<Partition>) -> IntegerPartitions {
        IntegerPartitions {
            partitions
        }
    }

    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    pub fn iter(&self) -> IntegerPartitionIter {
        IntegerPartitionIter {
            partitions: self.partitions.clone(),
            current_cursor: 0,
            end_exclusive: self.partitions.len(),
        }
    }

    pub fn union(&self) -> Partition {
        let intervals: Vec<ContiguousIntegerSet<usize>> =
            self.partitions
                .iter()
                .flat_map(|p| p.get_intervals_by_ref().clone())
                .collect();
        OrderedIntegerSet::from_contiguous_integer_sets(intervals)
    }
}

impl Index<usize> for IntegerPartitions {
    type Output = Partition;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.partitions[index]
    }
}

pub struct IntegerPartitionIter {
    partitions: Vec<Partition>,
    current_cursor: usize,
    end_exclusive: usize,
}

impl IntegerPartitionIter {
    pub fn clone_with_range(&self, start: usize, end_exclusive: usize) -> IntegerPartitionIter {
        IntegerPartitionIter {
            partitions: self.partitions[start..end_exclusive].to_vec(),
            current_cursor: 0,
            end_exclusive: end_exclusive - start,
        }
    }
}

impl Iterator for IntegerPartitionIter {
    type Item = Partition;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_cursor >= self.end_exclusive {
            None
        } else {
            self.current_cursor += 1;
            Some(self.partitions[self.current_cursor - 1].clone())
        }
    }
}

impl ExactSizeIterator for IntegerPartitionIter {
    #[inline]
    fn len(&self) -> usize {
        self.end_exclusive - self.current_cursor
    }
}

impl DoubleEndedIterator for IntegerPartitionIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_cursor >= self.end_exclusive {
            None
        } else {
            self.end_exclusive -= 1;
            Some(self.partitions[self.end_exclusive].clone())
        }
    }
}

impl<'a> IntoParallelIterator for IntegerPartitionIter {
    type Iter = IntegerPartitionParallelIter;
    type Item = <IntegerPartitionParallelIter as ParallelIterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
        IntegerPartitionParallelIter { iter: self }
    }
}

pub struct IntegerPartitionParallelIter {
    iter: IntegerPartitionIter
}

impl ParallelIterator for IntegerPartitionParallelIter {
    type Item = <IntegerPartitionIter as Iterator>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where C: UnindexedConsumer<Self::Item>
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

impl IndexedParallelIterator for IntegerPartitionParallelIter {
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
        where C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where CB: ProducerCallback<Self::Item>,
    {
        callback.callback(IntegerPartitionIterProducer { iter: self.iter })
    }
}

struct IntegerPartitionIterProducer {
    iter: IntegerPartitionIter,
}

impl Producer for IntegerPartitionIterProducer {
    type Item = <IntegerPartitionIter as Iterator>::Item;
    type IntoIter = IntegerPartitionIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            IntegerPartitionIterProducer {
                iter: self.iter.clone_with_range(0, index)
            },
            IntegerPartitionIterProducer {
                iter: self.iter.clone_with_range(index, self.iter.len())
            }
        )
    }
}

impl IntoIterator for IntegerPartitionIterProducer {
    type Item = <IntegerPartitionIter as Iterator>::Item;
    type IntoIter = IntegerPartitionIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    use crate::set::traits::Finite;

    use super::*;

    #[test]
    fn test_integer_partition_par_iter() {
        let partitions = IntegerPartitions::new(vec![
            OrderedIntegerSet::from_slice(&[[1, 3], [6, 9]]),
            OrderedIntegerSet::from_slice(&[[4, 5], [10, 14]]),
            OrderedIntegerSet::from_slice(&[[15, 20], [25, 26]]),
            OrderedIntegerSet::from_slice(&[[21, 24]]),
        ]);
        let mut iter = partitions.iter();
        assert_eq!(iter.next(), Some(OrderedIntegerSet::from_slice(&[[1, 3], [6, 9]])));
        assert_eq!(iter.next(), Some(OrderedIntegerSet::from_slice(&[[4, 5], [10, 14]])));
        assert_eq!(iter.next(), Some(OrderedIntegerSet::from_slice(&[[15, 20], [25, 26]])));
        assert_eq!(iter.next(), Some(OrderedIntegerSet::from_slice(&[[21, 24]])));
        assert_eq!(iter.next(), None);

        let num_elements: usize = partitions.iter().into_par_iter().map(|p| p.size()).sum();
        assert_eq!(num_elements, 26);
    }
}
