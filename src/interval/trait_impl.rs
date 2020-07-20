use num::Integer;

use crate::interval::traits::{Coalesce, CoalesceIntervals, Interval};

impl<I: Coalesce<I> + Interval<Element=E> + Clone, E: Integer + Copy> CoalesceIntervals<I, E> for Vec<I> {
    fn to_coalesced_intervals(&self) -> Vec<I> {
        let mut intervals: Vec<I> = self.to_vec();
        intervals.coalesce_intervals_inplace();
        intervals
    }

    fn coalesce_intervals_inplace(&mut self) {
        self.sort_by_key(|i| i.get_start());
        let mut coalesced_intervals = Vec::new();
        for interval in self.drain(..) {
            match coalesced_intervals.last_mut() {
                None => coalesced_intervals.push(interval),
                Some(last_interval) => {
                    match last_interval.coalesce_with(&interval) {
                        None => coalesced_intervals.push(interval),
                        Some(new_interval) => *last_interval = new_interval,
                    }
                }
            }
        }
        *self = coalesced_intervals;
    }
}

#[cfg(test)]
mod tests {
    use crate::set::contiguous_integer_set::ContiguousIntegerSet;

    use super::CoalesceIntervals;

    #[test]
    fn test_to_coalesced_intervals() {
        let intervals = vec![
            ContiguousIntegerSet::new(2, 4),
            ContiguousIntegerSet::new(1, 1),
            ContiguousIntegerSet::new(-10, -5),
            ContiguousIntegerSet::new(4, 5),
            ContiguousIntegerSet::new(9, 10),
            ContiguousIntegerSet::new(-1, 3)
        ];
        let sorted_intervals = intervals.to_coalesced_intervals();
        assert_eq!(sorted_intervals, vec![
            ContiguousIntegerSet::new(-10, -5),
            ContiguousIntegerSet::new(-1, 5),
            ContiguousIntegerSet::new(9, 10)
        ])
    }
}
