use crate::{
    interval::traits::{CoalesceIntervals, Interval},
    search::binary_search::BinarySearch,
    set::{
        contiguous_integer_set::ContiguousIntegerSet,
        ordered_integer_set::OrderedIntegerSet, traits::Set,
    },
};
use num::{integer::Integer, traits::cast::ToPrimitive};
use std::{
    cmp::{max, min, Ordering},
    ops::{Sub, SubAssign},
};

impl<E: Integer + Copy + ToPrimitive> Sub<&ContiguousIntegerSet<E>>
    for ContiguousIntegerSet<E>
{
    type Output = OrderedIntegerSet<E>;

    fn sub(self, rhs: &ContiguousIntegerSet<E>) -> Self::Output {
        let a = self.get_start();
        let b = self.get_end();
        let c = rhs.get_start();
        let d = rhs.get_end();
        if self.is_empty() {
            return OrderedIntegerSet::from_ordered_coalesced_contiguous_integer_sets(vec![]);
        }
        if rhs.is_empty() {
            return OrderedIntegerSet::from_ordered_coalesced_contiguous_integer_sets(vec![self]);
        }
        // [a, b] - [c, d]
        let mut diff: Vec<ContiguousIntegerSet<E>> = Vec::with_capacity(2);
        if c > a {
            diff.push(ContiguousIntegerSet::new(a, min(b, c - E::one())));
        }
        let i = ContiguousIntegerSet::new(max(d + E::one(), a), b);
        if !i.is_empty() {
            diff.push(i);
        }
        OrderedIntegerSet::from_ordered_coalesced_contiguous_integer_sets(diff)
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub for ContiguousIntegerSet<E> {
    type Output = OrderedIntegerSet<E>;

    #[inline]
    fn sub(self, rhs: ContiguousIntegerSet<E>) -> Self::Output {
        self - &rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<&ContiguousIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &ContiguousIntegerSet<E>) -> Self::Output {
        if self.is_empty()
            || rhs.is_empty()
            || rhs.get_end() < self.intervals[0].get_start()
            || rhs.get_start() > self.intervals.last().unwrap().get_end()
        {
            return self.clone();
        }
        let num_intervals = self.intervals.len();

        let copy_first_n = self
            .intervals
            .binary_search_with_cmp(
                0,
                num_intervals,
                &rhs.get_start(),
                |interval, &rhs_start| {
                    if interval.get_end() < rhs_start {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                },
            )
            .unwrap_err()
            .unwrap_or(0);

        let mut diff = Vec::new();
        diff.extend_from_slice(&self.intervals[..copy_first_n]);
        let mut copy_from_i_to_end = None;
        for i in copy_first_n..num_intervals {
            let interval = self.intervals[i];
            if interval.get_start() > rhs.get_end() {
                copy_from_i_to_end = Some(i);
                break;
            }
            let mut diff_set = interval - rhs;
            if !diff_set.is_empty() {
                diff.append(&mut diff_set.intervals);
            }
        }
        if let Some(i) = copy_from_i_to_end {
            diff.extend_from_slice(&self.intervals[i..]);
        }
        OrderedIntegerSet::from_ordered_coalesced_contiguous_integer_sets(diff)
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<ContiguousIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: ContiguousIntegerSet<E>) -> Self::Output {
        self - &rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign<&ContiguousIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    fn sub_assign(&mut self, rhs: &ContiguousIntegerSet<E>) {
        *self = self.to_owned() - rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign<ContiguousIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    #[inline]
    fn sub_assign(&mut self, rhs: ContiguousIntegerSet<E>) {
        *self = self.to_owned() - &rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<&OrderedIntegerSet<E>>
    for ContiguousIntegerSet<E>
{
    type Output = OrderedIntegerSet<E>;

    fn sub(self, rhs: &OrderedIntegerSet<E>) -> Self::Output {
        let mut diff = OrderedIntegerSet::from(vec![self]);
        for interval in rhs.intervals_iter() {
            diff -= interval;
        }
        diff.into_coalesced()
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<OrderedIntegerSet<E>>
    for ContiguousIntegerSet<E>
{
    type Output = OrderedIntegerSet<E>;

    #[inline]
    fn sub(self, rhs: OrderedIntegerSet<E>) -> Self::Output {
        self - &rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<&OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    type Output = Self;

    fn sub(self, rhs: &OrderedIntegerSet<E>) -> Self::Output {
        let mut diff = Vec::new();
        let mut rhs_i = 0;
        let num_rhs_intervals = rhs.intervals.len();
        for interval in self.intervals.iter() {
            let mut fragments = vec![*interval];
            while rhs_i < num_rhs_intervals
                && rhs.intervals[rhs_i].get_end() < interval.get_start()
            {
                rhs_i += 1;
            }
            while rhs_i < num_rhs_intervals
                && rhs.intervals[rhs_i].get_start() <= interval.get_end()
            {
                match fragments.last() {
                    None => {}
                    Some(&l) => {
                        fragments.pop();
                        for frag in (l - rhs.intervals[rhs_i]).intervals {
                            fragments.push(frag);
                        }
                    }
                };
                if rhs.intervals[rhs_i].get_end() <= interval.get_end() {
                    rhs_i += 1;
                } else {
                    break;
                }
            }
            diff.append(&mut fragments);
        }
        OrderedIntegerSet::from_ordered_coalesced_contiguous_integer_sets(diff)
    }
}

impl<E: Integer + Copy + ToPrimitive> Sub<OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: OrderedIntegerSet<E>) -> Self::Output {
        self - &rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign<&OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    #[inline]
    fn sub_assign(&mut self, rhs: &OrderedIntegerSet<E>) {
        *self = self.to_owned() - rhs
    }
}

impl<E: Integer + Copy + ToPrimitive> SubAssign<OrderedIntegerSet<E>>
    for OrderedIntegerSet<E>
{
    #[inline]
    fn sub_assign(&mut self, rhs: OrderedIntegerSet<E>) {
        *self = self.to_owned() - &rhs
    }
}

#[cfg(test)]
mod tests {
    use crate::set::{
        contiguous_integer_set::ContiguousIntegerSet,
        ordered_integer_set::OrderedIntegerSet,
    };

    #[test]
    fn test_contiguous_sub_contiguous() {
        macro_rules! test {
            ($a:expr, $b:expr, $c:expr, $d:expr, $expected:expr) => {
                assert_eq!(
                    ContiguousIntegerSet::new($a, $b)
                        - ContiguousIntegerSet::new($c, $d),
                    OrderedIntegerSet::from_slice($expected)
                );
            };
        }
        test!(2, 4, 5, 6, &[[2, 4]]);
        test!(2, 4, 4, 6, &[[2, 3]]);
        test!(2, 4, 4, 4, &[[2, 3]]);
        test!(2, 4, 2, 2, &[[3, 4]]);
        test!(2, 4, 3, 6, &[[2, 2]]);
        test!(2, 5, 3, 4, &[[2, 2], [5, 5]]);
        test!(2, 10, 4, 7, &[[2, 3], [8, 10]]);
        test!(2, 10, 2, 2, &[[3, 10]]);
        test!(2, 10, 2, 3, &[[4, 10]]);
        test!(2, 10, -1, 2, &[[3, 10]]);
        test!(2, 10, -1, 9, &[[10, 10]]);
        test!(5, 10, 1, 8, &[[9, 10]]);
        test!(5, 10, 1, 4, &[[5, 10]]);
        test!(2, 5, 2, 5, &[]);
        test!(2, 5, 0, 8, &[]);
    }

    #[test]
    fn test_ordered_sub_contiguous() {
        macro_rules! test {
            ($ordered:expr, $a:expr, $b:expr, $expected:expr) => {
                assert_eq!(
                    OrderedIntegerSet::from_slice($ordered)
                        - ContiguousIntegerSet::new($a, $b),
                    OrderedIntegerSet::from_slice($expected)
                );
            };
        }
        test!(&[], 2, 3, &[]);
        test!(&[[4, 10]], 2, 3, &[[4, 10]]);
        test!(&[[4, 10]], -2, 3, &[[4, 10]]);
        test!(&[[4, 10]], 12, 13, &[[4, 10]]);
        test!(&[[4, 10]], 3, 4, &[[5, 10]]);
        test!(&[[4, 10]], 4, 4, &[[5, 10]]);
        test!(&[[4, 10]], 10, 10, &[[4, 9]]);
        test!(&[[4, 10]], 10, 12, &[[4, 9]]);
        test!(&[[0, 10]], 3, 5, &[[0, 2], [6, 10]]);
        test!(&[[0, 10]], 0, 10, &[]);
        test!(&[[0, 10]], -1, 11, &[]);
        test!(&[[0, 3], [6, 10]], 0, 11, &[]);
        test!(&[[0, 3], [6, 10]], 0, 2, &[[3, 3], [6, 10]]);
        test!(&[[0, 3], [6, 10]], 0, 3, &[[6, 10]]);
        test!(&[[0, 3], [6, 10]], 0, 5, &[[6, 10]]);
        test!(&[[0, 3], [6, 10]], 0, 6, &[[7, 10]]);
        test!(&[[0, 3], [6, 10]], 0, 9, &[[10, 10]]);
        test!(&[[0, 3], [6, 10]], 1, 1, &[[0, 0], [2, 3], [6, 10]]);
        test!(&[[0, 3], [6, 10]], 1, 2, &[[0, 0], [3, 3], [6, 10]]);
        test!(&[[0, 3], [6, 10]], 2, 3, &[[0, 1], [6, 10]]);
        test!(&[[0, 3], [6, 10]], 2, 6, &[[0, 1], [7, 10]]);
        test!(&[[0, 3], [6, 10]], 2, 9, &[[0, 1], [10, 10]]);
        test!(&[[0, 3], [6, 10]], 3, 9, &[[0, 2], [10, 10]]);
        test!(&[[0, 3], [6, 10]], 5, 9, &[[0, 3], [10, 10]]);
        test!(&[[0, 3], [6, 10]], 8, 8, &[[0, 3], [6, 7], [9, 10]]);
        test!(&[[0, 3], [6, 9], [12, 15]], 0, 14, &[[15, 15]]);
        test!(&[[0, 3], [6, 9], [12, 15]], 0, 15, &[]);
        test!(&[[0, 3], [6, 9], [12, 15]], 2, 7, &[[0, 1], [8, 9], [
            12, 15
        ]]);
        test!(&[[0, 3], [6, 9], [12, 15]], 3, 12, &[[0, 2], [13, 15]]);
        test!(&[[0, 3], [6, 9], [12, 15]], 3, 15, &[[0, 2]]);
        test!(&[[0, 3], [6, 9], [12, 15]], 9, 12, &[[0, 3], [6, 8], [
            13, 15
        ]]);
    }
}
