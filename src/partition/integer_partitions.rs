use crate::set::ordered_integer_set::{OrderedIntegerSet, ContiguousIntegerSet};
use std::ops::Index;

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

    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    pub fn iter(&self) -> IntegerPartitionIter {
        IntegerPartitionIter {
            iter: self.partitions.iter()
        }
    }

    pub fn union(&self) -> Partition {
        let intervals: Vec<ContiguousIntegerSet<usize>> = self.partitions.iter().flat_map(|p| p.get_intervals_by_ref().clone()).collect();
        OrderedIntegerSet::from_contiguous_integer_sets(intervals)
    }
}

impl Index<usize> for IntegerPartitions {
    type Output = Partition;
    fn index(&self, index: usize) -> &Self::Output {
        &self.partitions[index]
    }
}

pub struct IntegerPartitionIter<'a> {
    iter: std::slice::Iter<'a, Partition>,
}

impl<'a> Iterator for IntegerPartitionIter<'a> {
    type Item = &'a Partition;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
