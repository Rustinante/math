use crate::{
    interval::traits::{Coalesce, Interval},
    set::traits::{Finite, Intersect, Refineable, Set},
    traits::{Slicing, ToIterator},
};
use num::{integer::Integer, traits::cast::ToPrimitive, FromPrimitive};
use std::{
    cmp::{max, min},
    ops::Range,
};

pub type IntegerIntervalRefinement<E> = Vec<ContiguousIntegerSet<E>>;

/// Represents the set of integers in `[start, end]`.
/// `Ord` is automatically derived so that comparison is done lexicographically
/// with `start` first and `end` second.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct ContiguousIntegerSet<E: Integer + Copy> {
    start: E,
    end: E,
}

impl<E: Integer + Copy> ContiguousIntegerSet<E> {
    /// Creates an integer set `[start, end]`, where the `end` is inclusive.
    pub fn new(start: E, end: E) -> Self {
        ContiguousIntegerSet {
            start,
            end,
        }
    }

    #[inline]
    pub fn get_start_and_end(&self) -> (E, E) {
        (self.start, self.end)
    }

    pub fn is_subset_of(&self, other: &ContiguousIntegerSet<E>) -> bool {
        // note that the empty set is a subset of any set
        self.is_empty() || (other.start <= self.start && self.end <= other.end)
    }

    pub fn is_strict_subset_of(&self, other: &ContiguousIntegerSet<E>) -> bool {
        self.is_subset_of(&other) && (self != other) && !other.is_empty()
    }

    #[inline]
    pub fn slice<
        'a,
        I: Slicing<&'a ContiguousIntegerSet<E>, Option<ContiguousIntegerSet<E>>>,
    >(
        &'a self,
        slicer: I,
    ) -> Option<ContiguousIntegerSet<E>> {
        slicer.slice(self)
    }
}

impl<E: Integer + Copy> Set<E> for ContiguousIntegerSet<E> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.start > self.end
    }

    #[inline]
    fn contains(&self, item: &E) -> bool {
        let item = *item;
        item >= self.start && item <= self.end
    }
}

impl<E: Integer + Copy> Interval<E> for ContiguousIntegerSet<E> {
    fn from_boundaries(start: E, end_inclusive: E) -> Self {
        ContiguousIntegerSet::new(start, end_inclusive)
    }

    #[inline]
    fn get_start(&self) -> E {
        self.start
    }

    #[inline]
    fn get_end(&self) -> E {
        self.end
    }

    fn length(&self) -> E {
        if self.start > self.end {
            E::zero()
        } else {
            self.end - self.start + E::one()
        }
    }
}

impl<E: Integer + Copy>
    Intersect<&ContiguousIntegerSet<E>, Option<ContiguousIntegerSet<E>>>
    for ContiguousIntegerSet<E>
{
    fn intersect(
        &self,
        other: &ContiguousIntegerSet<E>,
    ) -> Option<ContiguousIntegerSet<E>> {
        if self.is_empty()
            || other.is_empty()
            || other.end < self.start
            || other.start > self.end
        {
            None
        } else {
            Some(ContiguousIntegerSet::new(
                max(self.start, other.start),
                min(self.end, other.end),
            ))
        }
    }

    fn has_non_empty_intersection_with(
        &self,
        other: &ContiguousIntegerSet<E>,
    ) -> bool {
        self.intersect(other).is_some()
    }
}

/// returns an interval if only if the two intervals can be merged into
/// a single non-empty interval.
/// An empty interval can be merged with any other non-empty interval
impl<E: Integer + Copy> Coalesce<Self> for ContiguousIntegerSet<E> {
    fn coalesce_with(&self, other: &Self) -> Option<Self> {
        if self.is_empty() && other.is_empty() {
            None
        } else if self.is_empty() {
            Some(*other)
        } else if other.is_empty() {
            Some(*self)
        } else {
            if self.start > other.end + E::one()
                || self.end + E::one() < other.start
            {
                None
            } else {
                Some(ContiguousIntegerSet::new(
                    min(self.start, other.start),
                    max(self.end, other.end),
                ))
            }
        }
    }
}

impl<E: Integer + Copy + ToPrimitive> Finite for ContiguousIntegerSet<E> {
    fn size(&self) -> usize {
        if self.start > self.end {
            0
        } else {
            (self.end - self.start + E::one()).to_usize().unwrap()
        }
    }
}

impl<E> Slicing<&ContiguousIntegerSet<E>, Option<ContiguousIntegerSet<E>>>
    for Range<usize>
where
    E: Integer + Copy + FromPrimitive + ToPrimitive,
{
    fn slice(
        self,
        input: &ContiguousIntegerSet<E>,
    ) -> Option<ContiguousIntegerSet<E>> {
        if self.start >= self.end || self.start >= input.size() {
            None
        } else {
            Some(ContiguousIntegerSet::new(
                input.start + E::from_usize(self.start).unwrap(),
                input.start + E::from_usize(self.end).unwrap() - E::one(),
            ))
        }
    }
}

impl<E> Refineable<IntegerIntervalRefinement<E>> for ContiguousIntegerSet<E>
where
    E: Integer + Copy + ToPrimitive,
{
    fn get_common_refinement(
        &self,
        other: &ContiguousIntegerSet<E>,
    ) -> IntegerIntervalRefinement<E> {
        let (a, b) = self.get_start_and_end();
        let (c, d) = other.get_start_and_end();
        if self.is_empty() {
            return if other.is_empty() {
                Vec::new()
            } else {
                vec![other.clone()]
            };
        }
        if other.is_empty() {
            return vec![self.clone()];
        }
        match self.intersect(other) {
            None => {
                if self.start <= other.start {
                    vec![self.clone(), other.clone()]
                } else {
                    vec![other.clone(), self.clone()]
                }
            }
            Some(intersection) => {
                let mut refinement = Vec::new();
                if a < intersection.start {
                    refinement.push(ContiguousIntegerSet::new(
                        a,
                        intersection.start - E::one(),
                    ));
                }
                if c < intersection.start {
                    refinement.push(ContiguousIntegerSet::new(
                        c,
                        intersection.start - E::one(),
                    ));
                }
                refinement.push(intersection);
                if b > intersection.end {
                    refinement.push(ContiguousIntegerSet::new(
                        intersection.end + E::one(),
                        b,
                    ));
                }
                if d > intersection.end {
                    refinement.push(ContiguousIntegerSet::new(
                        intersection.end + E::one(),
                        d,
                    ));
                }
                refinement
            }
        }
    }
}

impl<E: Integer + Copy> Coalesce<E> for ContiguousIntegerSet<E> {
    fn coalesce_with(&self, other: &E) -> Option<Self> {
        if self.is_empty() {
            Some(ContiguousIntegerSet::new(*other, *other))
        } else {
            if self.start > *other + E::one() || self.end + E::one() < *other {
                None
            } else {
                Some(ContiguousIntegerSet::new(
                    min(self.start, *other),
                    max(self.end, *other),
                ))
            }
        }
    }
}

/// An iterator that iterates through the integers in the contiguous integer
/// set.
pub struct ContiguousIntegerSetIter<E: Integer + Copy> {
    contiguous_integer_set: ContiguousIntegerSet<E>,
    current: E,
}

impl<E: Integer + Copy> ToIterator<'_, ContiguousIntegerSetIter<E>, E>
    for ContiguousIntegerSet<E>
{
    #[inline]
    fn to_iter(&self) -> ContiguousIntegerSetIter<E> {
        ContiguousIntegerSetIter::from(*self)
    }
}

impl<E: Integer + Copy> From<ContiguousIntegerSet<E>>
    for ContiguousIntegerSetIter<E>
{
    fn from(
        contiguous_integer_set: ContiguousIntegerSet<E>,
    ) -> ContiguousIntegerSetIter<E> {
        ContiguousIntegerSetIter {
            contiguous_integer_set,
            current: E::zero(),
        }
    }
}

impl<E: Integer + Copy> Iterator for ContiguousIntegerSetIter<E> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.contiguous_integer_set.end {
            None
        } else {
            let val = self.current;
            self.current = self.current + E::one();
            Some(val)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::set::{
        contiguous_integer_set::ContiguousIntegerSet, traits::Intersect,
    };

    #[test]
    fn test_ord() {
        macro_rules! expect_le {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                let s1 = ContiguousIntegerSet::new($a, $b);
                let s2 = ContiguousIntegerSet::new($c, $d);
                assert!(s1 < s2);
            };
        }
        macro_rules! expect_ge {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                let s1 = ContiguousIntegerSet::new($a, $b);
                let s2 = ContiguousIntegerSet::new($c, $d);
                assert!(s1 > s2);
            };
        }
        expect_le!(2, 5, 3, 4);
        expect_le!(2, 5, 3, 5);
        expect_le!(2, 5, 3, 8);
        expect_le!(2, 3, 4, 5);
        expect_le!(2, 5, 2, 8);
        expect_ge!(2, 5, 2, 4);
        expect_ge!(2, 5, 1, 8);
        expect_ge!(2, 5, 1, 5);
        expect_ge!(2, 5, 1, 3);
        assert_eq!(
            ContiguousIntegerSet::new(2, 2),
            ContiguousIntegerSet::new(2, 2)
        );
    }

    #[test]
    fn test_intersect() {
        let s1 = ContiguousIntegerSet::new(2, 5);
        let s2 = ContiguousIntegerSet::new(2, 2);
        let s3 = ContiguousIntegerSet::new(4, 8);
        let s4 = ContiguousIntegerSet::new(-3, -1);
        assert_eq!(s1.intersect(&s2), Some(ContiguousIntegerSet::new(2, 2)));
        assert_eq!(s1.intersect(&s3), Some(ContiguousIntegerSet::new(4, 5)));
        assert_eq!(s1.intersect(&s4), None);
        assert!(s1.has_non_empty_intersection_with(&s2));
        assert!(s1.has_non_empty_intersection_with(&s3));
        assert!(!s1.has_non_empty_intersection_with(&s4));
    }

    #[test]
    fn test_is_subset_of() {
        macro_rules! ab_is_subset_of_cd {
            (
                $a:expr,
                $b:expr,
                $c:expr,
                $d:expr,
                $is_subset:expr,
                $is_strict_subset:expr
            ) => {
                let s1 = ContiguousIntegerSet::new($a as i32, $b);
                let s2 = ContiguousIntegerSet::new($c as i32, $d);
                assert_eq!(s1.is_subset_of(&s2), $is_subset);
                assert_eq!(s1.is_strict_subset_of(&s2), $is_strict_subset);

                let s1 = ContiguousIntegerSet::new($a as i64, $b);
                let s2 = ContiguousIntegerSet::new($c as i64, $d);
                assert_eq!(s1.is_subset_of(&s2), $is_subset);
                assert_eq!(s1.is_strict_subset_of(&s2), $is_strict_subset);
            };
        }

        macro_rules! test_nonnegative_abcd {
            (
                $a:expr,
                $b:expr,
                $c:expr,
                $d:expr,
                $is_subset:expr,
                $is_strict_subset:expr
            ) => {
                // test the signed type
                ab_is_subset_of_cd!(
                    $a,
                    $b,
                    $c,
                    $d,
                    $is_subset,
                    $is_strict_subset
                );

                let s1 = ContiguousIntegerSet::new($a as u32, $b);
                let s2 = ContiguousIntegerSet::new($c as u32, $d);
                assert_eq!(s1.is_subset_of(&s2), $is_subset);
                assert_eq!(s1.is_strict_subset_of(&s2), $is_strict_subset);

                let s1 = ContiguousIntegerSet::new($a as u64, $b);
                let s2 = ContiguousIntegerSet::new($c as u64, $d);
                assert_eq!(s1.is_subset_of(&s2), $is_subset);
                assert_eq!(s1.is_strict_subset_of(&s2), $is_strict_subset);
            };
        }

        // signs
        // - - - -
        ab_is_subset_of_cd!(-10, -5, -20, -25, false, false);
        ab_is_subset_of_cd!(-10, -5, -25, -20, false, false);
        ab_is_subset_of_cd!(-10, -5, -25, -10, false, false);
        ab_is_subset_of_cd!(-10, -5, -25, -7, false, false);
        ab_is_subset_of_cd!(-10, -5, -10, -10, false, false);
        ab_is_subset_of_cd!(-10, -5, -10, -7, false, false);
        ab_is_subset_of_cd!(-10, -5, -8, -7, false, false);
        ab_is_subset_of_cd!(-10, -5, -8, -5, false, false);
        ab_is_subset_of_cd!(-10, -5, -8, -1, false, false);
        ab_is_subset_of_cd!(-10, -5, -5, -1, false, false);
        ab_is_subset_of_cd!(-10, -5, -2, -1, false, false);

        ab_is_subset_of_cd!(-10, -5, -25, -5, true, true);
        ab_is_subset_of_cd!(-10, -5, -25, -1, true, true);
        ab_is_subset_of_cd!(-10, -5, -10, -5, true, false);
        ab_is_subset_of_cd!(-10, -5, -10, -2, true, true);

        // the empty set is a subset of any set
        ab_is_subset_of_cd!(-5, -10, -20, -25, true, false);
        ab_is_subset_of_cd!(-5, -10, -25, -20, true, true);
        ab_is_subset_of_cd!(-5, -10, -25, -2, true, true);
        ab_is_subset_of_cd!(-5, -10, -10, -2, true, true);
        ab_is_subset_of_cd!(-5, -10, -2, -1, true, true);

        // signs
        // - - - +
        ab_is_subset_of_cd!(-15, -5, -10, 0, false, false);
        ab_is_subset_of_cd!(-15, -5, -5, 0, false, false);
        ab_is_subset_of_cd!(-15, -5, -2, 0, false, false);
        ab_is_subset_of_cd!(-15, -5, 0, 0, false, false);

        ab_is_subset_of_cd!(-15, -5, -10, 1, false, false);
        ab_is_subset_of_cd!(-15, -5, -5, 1, false, false);
        ab_is_subset_of_cd!(-15, -5, -2, 1, false, false);
        ab_is_subset_of_cd!(-15, -5, 0, 1, false, false);

        ab_is_subset_of_cd!(-15, -5, -20, 0, true, true);
        ab_is_subset_of_cd!(-15, -5, -15, 0, true, true);
        ab_is_subset_of_cd!(-15, -5, -20, 4, true, true);
        ab_is_subset_of_cd!(-15, -5, -15, 4, true, true);

        // the empty set is a subset of any set
        ab_is_subset_of_cd!(-10, -15, -20, 0, true, true);
        ab_is_subset_of_cd!(-10, -20, -20, 0, true, true);
        ab_is_subset_of_cd!(-10, -25, -20, 0, true, true);
        ab_is_subset_of_cd!(-20, -30, -20, 0, true, true);
        ab_is_subset_of_cd!(-25, -30, -20, 0, true, true);

        // signs
        // - - + -
        ab_is_subset_of_cd!(-20, -10, 1, -1, false, false);
        ab_is_subset_of_cd!(-20, -10, 1, -15, false, false);
        ab_is_subset_of_cd!(-20, -10, 1, -10, false, false);
        ab_is_subset_of_cd!(-20, -10, 1, -20, false, false);
        ab_is_subset_of_cd!(-20, -10, 1, -25, false, false);

        ab_is_subset_of_cd!(-20, -10, 0, -1, false, false);
        ab_is_subset_of_cd!(-20, -10, 0, -15, false, false);
        ab_is_subset_of_cd!(-20, -10, 0, -10, false, false);
        ab_is_subset_of_cd!(-20, -10, 0, -20, false, false);
        ab_is_subset_of_cd!(-20, -10, 0, -25, false, false);

        // the empty set is a subset of any set
        ab_is_subset_of_cd!(-10, -20, 1, -1, true, false);
        ab_is_subset_of_cd!(-10, -20, 1, -15, true, false);
        ab_is_subset_of_cd!(-10, -20, 1, -10, true, false);
        ab_is_subset_of_cd!(-10, -20, 1, -20, true, false);
        ab_is_subset_of_cd!(-10, -20, 1, -25, true, false);
        ab_is_subset_of_cd!(-10, -20, 0, -1, true, false);
        ab_is_subset_of_cd!(-10, -20, 0, -15, true, false);
        ab_is_subset_of_cd!(-10, -20, 0, -10, true, false);
        ab_is_subset_of_cd!(-10, -20, 0, -20, true, false);
        ab_is_subset_of_cd!(-10, -20, 0, -25, true, false);

        // signs
        // - - + +
        ab_is_subset_of_cd!(-20, -10, 0, 0, false, false);
        ab_is_subset_of_cd!(-20, -10, 0, 1, false, false);
        ab_is_subset_of_cd!(-20, -10, 1, 1, false, false);
        ab_is_subset_of_cd!(-20, -10, 3, 5, false, false);

        // the empty set is a subset of any set
        ab_is_subset_of_cd!(-10, -20, 1, 1, true, true);
        ab_is_subset_of_cd!(-10, -20, 1, 2, true, true);
        ab_is_subset_of_cd!(-10, -20, 2, 1, true, false);

        // signs
        // - + - -
        ab_is_subset_of_cd!(-20, 0, -40, -30, false, false);
        ab_is_subset_of_cd!(-20, 0, -40, -20, false, false);
        ab_is_subset_of_cd!(-20, 0, -40, -10, false, false);
        ab_is_subset_of_cd!(-20, 0, -20, -20, false, false);
        ab_is_subset_of_cd!(-20, 0, -20, -10, false, false);
        ab_is_subset_of_cd!(-20, 0, -11, -10, false, false);
        ab_is_subset_of_cd!(-20, 0, -10, -10, false, false);

        ab_is_subset_of_cd!(-20, 3, -40, -30, false, false);
        ab_is_subset_of_cd!(-20, 2, -40, -20, false, false);
        ab_is_subset_of_cd!(-20, 1, -40, -10, false, false);
        ab_is_subset_of_cd!(-20, 3, -20, -20, false, false);
        ab_is_subset_of_cd!(-20, 2, -20, -10, false, false);
        ab_is_subset_of_cd!(-20, 1, -11, -10, false, false);
        ab_is_subset_of_cd!(-20, 3, -10, -10, false, false);

        ab_is_subset_of_cd!(-20, 3, -10, -12, false, false);
        ab_is_subset_of_cd!(-20, 0, -10, -20, false, false);
        ab_is_subset_of_cd!(-20, 2, -10, -25, false, false);
        ab_is_subset_of_cd!(-20, 0, -20, -25, false, false);
        ab_is_subset_of_cd!(-20, 1, -21, -25, false, false);

        // signs
        // - + - +
        ab_is_subset_of_cd!(-20, 0, -19, 0, false, false);
        ab_is_subset_of_cd!(-20, 0, -19, 1, false, false);
        ab_is_subset_of_cd!(-20, 1, -19, 0, false, false);
        ab_is_subset_of_cd!(-20, 1, -19, 1, false, false);

        ab_is_subset_of_cd!(-20, 1, -20, 0, false, false);
        ab_is_subset_of_cd!(-20, 2, -20, 1, false, false);
        ab_is_subset_of_cd!(-20, 1, -21, 0, false, false);
        ab_is_subset_of_cd!(-20, 2, -22, 1, false, false);

        ab_is_subset_of_cd!(-20, 0, -20, 0, true, false);
        ab_is_subset_of_cd!(-20, 0, -20, 1, true, true);
        ab_is_subset_of_cd!(-20, 1, -20, 1, true, false);
        ab_is_subset_of_cd!(-20, 0, -100, 0, true, true);
        ab_is_subset_of_cd!(-20, 0, -100, 1, true, true);
        ab_is_subset_of_cd!(-20, 1, -100, 1, true, true);

        // signs
        // - + + -
        ab_is_subset_of_cd!(-20, 10, 10, -20, false, false);
        ab_is_subset_of_cd!(-20, 10, 10, -22, false, false);
        ab_is_subset_of_cd!(-20, 10, 10, -18, false, false);

        ab_is_subset_of_cd!(-20, 10, 8, -20, false, false);
        ab_is_subset_of_cd!(-20, 10, 8, -22, false, false);
        ab_is_subset_of_cd!(-20, 10, 8, -18, false, false);

        ab_is_subset_of_cd!(-20, 10, 12, -20, false, false);
        ab_is_subset_of_cd!(-20, 10, 12, -22, false, false);
        ab_is_subset_of_cd!(-20, 10, 12, -18, false, false);

        // signs
        // - + + +
        ab_is_subset_of_cd!(-20, 0, 10, 20, false, false);
        ab_is_subset_of_cd!(-20, 0, 15, 20, false, false);
        ab_is_subset_of_cd!(-20, 0, 0, 20, false, false);
        ab_is_subset_of_cd!(-20, 10, 10, 20, false, false);
        ab_is_subset_of_cd!(-20, 10, 15, 20, false, false);
        ab_is_subset_of_cd!(-20, 10, 0, 20, false, false);

        // signs
        // + - - -
        // the empty set is a subset of any set
        ab_is_subset_of_cd!(10, -20, -10, -10, true, true);
        ab_is_subset_of_cd!(0, -20, -20, -10, true, true);
        ab_is_subset_of_cd!(10, -20, -20, -20, true, true);
        ab_is_subset_of_cd!(0, -20, -30, -20, true, true);
        ab_is_subset_of_cd!(10, -20, -30, -30, true, true);

        // signs
        // + - - +
        // the empty set is a subset of any set
        ab_is_subset_of_cd!(10, -20, -18, 10, true, true);
        ab_is_subset_of_cd!(10, -20, -20, 10, true, true);
        ab_is_subset_of_cd!(10, -20, -22, 10, true, true);

        ab_is_subset_of_cd!(10, -20, -18, 12, true, true);
        ab_is_subset_of_cd!(10, -20, -20, 12, true, true);
        ab_is_subset_of_cd!(10, -20, -22, 12, true, true);

        ab_is_subset_of_cd!(10, -20, -18, 8, true, true);
        ab_is_subset_of_cd!(10, -20, -20, 8, true, true);
        ab_is_subset_of_cd!(10, -20, -22, 8, true, true);

        // signs
        // + - + -
        // the empty set is a subset of any set
        ab_is_subset_of_cd!(10, -20, 10, -18, true, false);
        ab_is_subset_of_cd!(10, -20, 10, -20, true, false);
        ab_is_subset_of_cd!(10, -20, 10, -22, true, false);

        ab_is_subset_of_cd!(10, -20, 12, -18, true, false);
        ab_is_subset_of_cd!(10, -20, 12, -20, true, false);
        ab_is_subset_of_cd!(10, -20, 12, -22, true, false);

        ab_is_subset_of_cd!(10, -20, 8, -18, true, false);
        ab_is_subset_of_cd!(10, -20, 8, -20, true, false);
        ab_is_subset_of_cd!(10, -20, 8, -22, true, false);

        // signs
        // + - + +
        // the empty set is a subset of any set
        ab_is_subset_of_cd!(10, -20, 0, 8, true, true);
        ab_is_subset_of_cd!(10, -20, 0, 10, true, true);
        ab_is_subset_of_cd!(10, -20, 0, 12, true, true);
        ab_is_subset_of_cd!(10, -20, 4, 8, true, true);
        ab_is_subset_of_cd!(10, -20, 4, 10, true, true);
        ab_is_subset_of_cd!(10, -20, 4, 12, true, true);

        // signs
        // + + - -
        ab_is_subset_of_cd!(0, 20, -10, -20, false, false);
        ab_is_subset_of_cd!(10, 20, -10, -20, false, false);

        // the empty set is a subset of any set
        ab_is_subset_of_cd!(20, 10, -20, -10, true, true);
        ab_is_subset_of_cd!(20, 10, -10, -20, true, false);

        // signs
        // + + - +
        ab_is_subset_of_cd!(10, 20, -10, 0, false, false);
        ab_is_subset_of_cd!(10, 20, -10, 5, false, false);
        ab_is_subset_of_cd!(10, 20, -10, 10, false, false);
        ab_is_subset_of_cd!(0, 20, -10, 0, false, false);
        ab_is_subset_of_cd!(0, 20, -10, 5, false, false);
        ab_is_subset_of_cd!(0, 20, -10, 10, false, false);

        ab_is_subset_of_cd!(10, 20, -10, 20, true, true);
        ab_is_subset_of_cd!(10, 20, -10, 25, true, true);
        ab_is_subset_of_cd!(0, 20, -10, 20, true, true);
        ab_is_subset_of_cd!(0, 20, -10, 25, true, true);

        // the empty set is a subset of any set
        ab_is_subset_of_cd!(20, 10, -10, 0, true, true);
        ab_is_subset_of_cd!(20, 10, -10, 25, true, true);

        // signs
        // + + + -
        ab_is_subset_of_cd!(10, 20, 0, -10, false, false);
        ab_is_subset_of_cd!(10, 20, 5, -10, false, false);
        ab_is_subset_of_cd!(10, 20, 10, -10, false, false);
        ab_is_subset_of_cd!(10, 20, 20, -10, false, false);
        ab_is_subset_of_cd!(10, 20, 25, -10, false, false);

        ab_is_subset_of_cd!(0, 20, 0, -10, false, false);
        ab_is_subset_of_cd!(0, 20, 5, -10, false, false);
        ab_is_subset_of_cd!(0, 20, 10, -10, false, false);
        ab_is_subset_of_cd!(0, 20, 20, -10, false, false);
        ab_is_subset_of_cd!(0, 20, 25, -10, false, false);

        // signs
        // + + + +
        test_nonnegative_abcd!(0, 10, 0, 0, false, false);
        test_nonnegative_abcd!(0, 10, 0, 5, false, false);
        test_nonnegative_abcd!(0, 10, 0, 10, true, false);
        test_nonnegative_abcd!(0, 10, 0, 15, true, true);

        test_nonnegative_abcd!(0, 10, 5, 5, false, false);
        test_nonnegative_abcd!(0, 10, 5, 10, false, false);
        test_nonnegative_abcd!(0, 10, 5, 15, false, false);

        test_nonnegative_abcd!(0, 10, 10, 10, false, false);
        test_nonnegative_abcd!(0, 10, 10, 11, false, false);

        test_nonnegative_abcd!(0, 10, 15, 15, false, false);
        test_nonnegative_abcd!(0, 10, 15, 20, false, false);

        test_nonnegative_abcd!(5, 10, 0, 0, false, false);
        test_nonnegative_abcd!(5, 10, 0, 2, false, false);
        test_nonnegative_abcd!(5, 10, 0, 5, false, false);
        test_nonnegative_abcd!(5, 10, 0, 7, false, false);
        test_nonnegative_abcd!(5, 10, 0, 10, true, true);
        test_nonnegative_abcd!(5, 10, 0, 15, true, true);

        test_nonnegative_abcd!(5, 10, 2, 2, false, false);
        test_nonnegative_abcd!(5, 10, 2, 5, false, false);
        test_nonnegative_abcd!(5, 10, 2, 7, false, false);
        test_nonnegative_abcd!(5, 10, 2, 10, true, true);
        test_nonnegative_abcd!(5, 10, 2, 15, true, true);

        test_nonnegative_abcd!(5, 10, 5, 5, false, false);
        test_nonnegative_abcd!(5, 10, 5, 7, false, false);
        test_nonnegative_abcd!(5, 10, 5, 10, true, false);
        test_nonnegative_abcd!(5, 10, 5, 15, true, true);

        test_nonnegative_abcd!(5, 10, 8, 8, false, false);
        test_nonnegative_abcd!(5, 10, 8, 9, false, false);
        test_nonnegative_abcd!(5, 10, 8, 10, false, false);
        test_nonnegative_abcd!(5, 10, 8, 12, false, false);

        test_nonnegative_abcd!(5, 10, 10, 10, false, false);
        test_nonnegative_abcd!(5, 10, 10, 15, false, false);

        test_nonnegative_abcd!(5, 10, 15, 15, false, false);
        test_nonnegative_abcd!(5, 10, 15, 16, false, false);

        // the empty set is a subset of any set
        test_nonnegative_abcd!(10, 0, 0, 8, true, true);
        test_nonnegative_abcd!(10, 0, 0, 10, true, true);
        test_nonnegative_abcd!(10, 0, 0, 12, true, true);
        test_nonnegative_abcd!(10, 0, 1, 8, true, true);
        test_nonnegative_abcd!(10, 0, 1, 10, true, true);
        test_nonnegative_abcd!(10, 0, 1, 12, true, true);
        test_nonnegative_abcd!(10, 2, 0, 8, true, true);
        test_nonnegative_abcd!(10, 2, 0, 10, true, true);
        test_nonnegative_abcd!(10, 2, 0, 12, true, true);
        test_nonnegative_abcd!(10, 2, 1, 8, true, true);
        test_nonnegative_abcd!(10, 2, 1, 10, true, true);
        test_nonnegative_abcd!(10, 2, 1, 12, true, true);

        test_nonnegative_abcd!(10, 0, 10, 0, true, false);
        test_nonnegative_abcd!(10, 0, 10, 2, true, false);
        test_nonnegative_abcd!(10, 2, 10, 0, true, false);
        test_nonnegative_abcd!(10, 2, 10, 2, true, false);
        test_nonnegative_abcd!(10, 2, 10, 4, true, false);
    }
}
