//! Modeling a collection of disjoint integer intervals.

use crate::set::{
    contiguous_integer_set::ContiguousIntegerSet,
    ordered_integer_set::OrderedIntegerSet,
};
use num::{Integer, ToPrimitive};
use rayon::iter::{
    plumbing::{
        bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer,
    },
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
use std::ops::Index;

pub type Partition<T> = OrderedIntegerSet<T>;

/// A collection of disjoint integer intervals
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegerPartitions<T: Copy + Integer + ToPrimitive> {
    partitions: Vec<Partition<T>>,
}

impl<T: Copy + Integer + ToPrimitive> IntegerPartitions<T> {
    pub fn new(partitions: Vec<Partition<T>>) -> IntegerPartitions<T> {
        IntegerPartitions {
            partitions,
        }
    }

    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Creates an iterator that iterates through the partitions.
    pub fn iter(&self) -> IntegerPartitionIter<T> {
        IntegerPartitionIter {
            partitions: self.partitions.clone(),
            current_cursor: 0,
            end_exclusive: self.partitions.len(),
        }
    }

    /// Converts the collection of partitions into a single `Partition`
    /// consisting of the same integer elements.
    pub fn union(&self) -> Partition<T> {
        let intervals: Vec<ContiguousIntegerSet<T>> = self
            .partitions
            .iter()
            .flat_map(|p| p.get_intervals_by_ref().clone())
            .collect();
        OrderedIntegerSet::from_contiguous_integer_sets(intervals)
    }
}

impl<T: Copy + Integer + ToPrimitive> Index<usize> for IntegerPartitions<T> {
    type Output = Partition<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.partitions[index]
    }
}

pub struct IntegerPartitionIter<T: Copy + Integer + ToPrimitive> {
    partitions: Vec<Partition<T>>,
    current_cursor: usize,
    end_exclusive: usize,
}

impl<T: Copy + Integer + ToPrimitive> IntegerPartitionIter<T> {
    pub fn clone_with_range(
        &self,
        start: usize,
        end_exclusive: usize,
    ) -> IntegerPartitionIter<T> {
        assert!(
            start <= end_exclusive,
            "start ({}) has to be <= end_exclusive ({})",
            start,
            end_exclusive
        );
        IntegerPartitionIter {
            partitions: self.partitions[start..end_exclusive].to_vec(),
            current_cursor: 0,
            end_exclusive: end_exclusive - start,
        }
    }
}

impl<T: Copy + Integer + ToPrimitive> Iterator for IntegerPartitionIter<T> {
    type Item = Partition<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_cursor >= self.end_exclusive {
            None
        } else {
            self.current_cursor += 1;
            Some(self.partitions[self.current_cursor - 1].clone())
        }
    }
}

impl<T: Copy + Integer + ToPrimitive> ExactSizeIterator
    for IntegerPartitionIter<T>
{
    fn len(&self) -> usize {
        if self.current_cursor >= self.end_exclusive {
            0
        } else {
            self.end_exclusive - self.current_cursor
        }
    }
}

impl<T: Copy + Integer + ToPrimitive> DoubleEndedIterator
    for IntegerPartitionIter<T>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_cursor >= self.end_exclusive {
            None
        } else {
            self.end_exclusive -= 1;
            Some(self.partitions[self.end_exclusive].clone())
        }
    }
}

impl<'a, T: Copy + Integer + Send + ToPrimitive> IntoParallelIterator
    for IntegerPartitionIter<T>
{
    type Item = <IntegerPartitionParallelIter<T> as ParallelIterator>::Item;
    type Iter = IntegerPartitionParallelIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        IntegerPartitionParallelIter {
            iter: self,
        }
    }
}

pub struct IntegerPartitionParallelIter<T: Copy + Integer + ToPrimitive> {
    iter: IntegerPartitionIter<T>,
}

impl<T: Copy + Integer + Send + ToPrimitive> ParallelIterator
    for IntegerPartitionParallelIter<T>
{
    type Item = <IntegerPartitionIter<T> as Iterator>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>, {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

impl<T: Copy + Integer + Send + ToPrimitive> IndexedParallelIterator
    for IntegerPartitionParallelIter<T>
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>, {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>, {
        callback.callback(IntegerPartitionIterProducer {
            iter: self.iter,
        })
    }
}

struct IntegerPartitionIterProducer<T: Copy + Integer + ToPrimitive> {
    iter: IntegerPartitionIter<T>,
}

impl<T: Copy + Integer + Send + ToPrimitive> Producer
    for IntegerPartitionIterProducer<T>
{
    type IntoIter = IntegerPartitionIter<T>;
    type Item = <IntegerPartitionIter<T> as Iterator>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            IntegerPartitionIterProducer {
                iter: self.iter.clone_with_range(0, index),
            },
            IntegerPartitionIterProducer {
                iter: self.iter.clone_with_range(index, self.iter.len()),
            },
        )
    }
}

impl<T: Copy + Integer + ToPrimitive> IntoIterator
    for IntegerPartitionIterProducer<T>
{
    type IntoIter = IntegerPartitionIter<T>;
    type Item = <IntegerPartitionIter<T> as Iterator>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        partition::integer_partitions::{IntegerPartitions, Partition},
        set::{ordered_integer_set::OrderedIntegerSet, traits::Finite},
    };
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    #[test]
    fn test_num_partitions() {
        assert_eq!(IntegerPartitions::<usize>::new(vec![]).num_partitions(), 0);
        assert_eq!(IntegerPartitions::<i32>::new(vec![]).num_partitions(), 0);
        assert_eq!(IntegerPartitions::<i64>::new(vec![]).num_partitions(), 0);
        assert_eq!(
            IntegerPartitions::new(vec![
                Partition::from_slice(&[[0i32, 2]]),
                Partition::from_slice(&[[4, 8], [15, 21]]),
            ])
            .num_partitions(),
            2
        );
        assert_eq!(
            IntegerPartitions::new(vec![Partition::from_slice(&[
                [2usize, 4],
                [5, 6],
                [10, 11]
            ])])
            .num_partitions(),
            1
        );
    }

    #[test]
    fn test_partitions_union() {
        let partitions = IntegerPartitions::<i32>::new(vec![
            Partition::from_slice(&[[1, 3], [8, 9]]),
            Partition::from_slice(&[[4, 5], [10, 14]]),
            Partition::from_slice(&[[21, 24]]),
        ]);
        assert_eq!(
            partitions.union(),
            Partition::<i32>::from_slice(&[
                [1, 3],
                [4, 5],
                [8, 9],
                [10, 14],
                [21, 24]
            ])
        );
    }

    #[test]
    fn test_partitions_iter() {
        macro_rules! test_with_type {
            ($itype:ty) => {
                let partition_list = vec![
                    Partition::<$itype>::from_slice(&[[0, 2], [9, 11]]),
                    Partition::from_slice(&[[4, 8], [15, 21]]),
                ];
                let partitions = IntegerPartitions::new(partition_list.clone());
                for (actual, expected) in
                    partitions.iter().zip(partition_list.iter())
                {
                    assert_eq!(&actual, expected);
                }
            };
        }
        test_with_type!(usize);
        test_with_type!(i32);
        test_with_type!(i64);
    }

    #[test]
    fn test_partitions_next_back() {
        let partitions = IntegerPartitions::new(vec![
            OrderedIntegerSet::from_slice(&[[1, 3], [6, 9]]),
            OrderedIntegerSet::from_slice(&[[4, 5], [10, 14]]),
            OrderedIntegerSet::from_slice(&[[15, 20], [25, 26]]),
            OrderedIntegerSet::from_slice(&[[21, 24]]),
        ]);
        assert_eq!(
            partitions.iter().nth_back(2).unwrap(),
            Partition::<i32>::from_slice(&[[4, 5], [10, 14]])
        );
    }

    #[test]
    fn test_integer_partition_exact_size_iter() {
        assert_eq!(IntegerPartitions::<usize>::new(vec![]).iter().len(), 0);
        assert_eq!(
            IntegerPartitions::new(vec![OrderedIntegerSet::from_slice(&[
                [-10, 20],
                [30, 40]
            ]),])
            .iter()
            .len(),
            1
        );

        assert_eq!(
            IntegerPartitions::new(vec![
                OrderedIntegerSet::from_slice(&[[-1, 2], [6, 9]]),
                OrderedIntegerSet::from_slice(&[[10, 14]]),
                OrderedIntegerSet::from_slice(&[[15, 20], [25, 26]]),
                OrderedIntegerSet::from_slice(&[[21, 24]]),
            ])
            .iter()
            .len(),
            4
        );
    }
    #[test]
    fn test_integer_partition_par_iter() {
        let partitions = IntegerPartitions::new(vec![
            OrderedIntegerSet::from_slice(&[[1, 3], [6, 9]]),
            OrderedIntegerSet::from_slice(&[[4, 5], [10, 14]]),
            OrderedIntegerSet::from_slice(&[[15, 20], [25, 26]]),
            OrderedIntegerSet::from_slice(&[[21, 24]]),
        ]);
        let mut iter = partitions.iter();
        assert_eq!(
            iter.next(),
            Some(OrderedIntegerSet::from_slice(&[[1, 3], [6, 9]]))
        );
        assert_eq!(
            iter.next(),
            Some(OrderedIntegerSet::from_slice(&[[4, 5], [10, 14]]))
        );
        assert_eq!(
            iter.next(),
            Some(OrderedIntegerSet::from_slice(&[[15, 20], [25, 26]]))
        );
        assert_eq!(
            iter.next(),
            Some(OrderedIntegerSet::from_slice(&[[21, 24]]))
        );
        assert_eq!(iter.next(), None);

        let num_elements: usize =
            partitions.iter().into_par_iter().map(|p| p.size()).sum();
        assert_eq!(num_elements, 26);
    }
}
