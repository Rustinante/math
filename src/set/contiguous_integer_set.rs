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

/// Represents the set of integers in [start, end].
/// `Ord` is automatically derived so that comparison is done lexicographically
/// with `start` first and `end` second.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct ContiguousIntegerSet<E: Integer + Copy> {
    start: E,
    end: E,
}

impl<E: Integer + Copy> ContiguousIntegerSet<E> {
    pub fn new(start: E, end: E) -> ContiguousIntegerSet<E> {
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
        self.start >= other.start && self.end <= other.end
    }

    #[inline]
    pub fn slice<'a, I: Slicing<&'a ContiguousIntegerSet<E>, Option<ContiguousIntegerSet<E>>>>(
        &'a self,
        slicer: I,
    ) -> Option<ContiguousIntegerSet<E>> {
        slicer.slice(self)
    }
}

impl<E: Integer + Copy> Set<E, Option<ContiguousIntegerSet<E>>> for ContiguousIntegerSet<E> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.start > self.end
    }

    #[inline]
    fn contains(&self, item: E) -> bool {
        item >= self.start && item <= self.end
    }
}

impl<E: Integer + Copy> Interval for ContiguousIntegerSet<E> {
    type Element = E;

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

impl<E: Integer + Copy> Intersect<&ContiguousIntegerSet<E>, Option<ContiguousIntegerSet<E>>>
    for ContiguousIntegerSet<E>
{
    fn intersect(&self, other: &ContiguousIntegerSet<E>) -> Option<ContiguousIntegerSet<E>> {
        if self.is_empty() || other.is_empty() || other.end < self.start || other.start > self.end {
            None
        } else {
            Some(ContiguousIntegerSet::new(
                max(self.start, other.start),
                min(self.end, other.end),
            ))
        }
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
            if self.start > other.end + E::one() || self.end + E::one() < other.start {
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

impl<E> Slicing<&ContiguousIntegerSet<E>, Option<ContiguousIntegerSet<E>>> for Range<usize>
where
    E: Integer + Copy + FromPrimitive + ToPrimitive,
{
    fn slice(self, input: &ContiguousIntegerSet<E>) -> Option<ContiguousIntegerSet<E>> {
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
                    refinement.push(ContiguousIntegerSet::new(a, intersection.start - E::one()));
                }
                if c < intersection.start {
                    refinement.push(ContiguousIntegerSet::new(c, intersection.start - E::one()));
                }
                refinement.push(intersection);
                if b > intersection.end {
                    refinement.push(ContiguousIntegerSet::new(intersection.end + E::one(), b));
                }
                if d > intersection.end {
                    refinement.push(ContiguousIntegerSet::new(intersection.end + E::one(), d));
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

/// An iterator that iterates through the integers in the contiguous integer set.
pub struct ContiguousIntegerSetIter<E: Integer + Copy> {
    contiguous_integer_set: ContiguousIntegerSet<E>,
    current: E,
}

impl<E: Integer + Copy> ToIterator<'_, ContiguousIntegerSetIter<E>, E> for ContiguousIntegerSet<E> {
    #[inline]
    fn to_iter(&self) -> ContiguousIntegerSetIter<E> {
        ContiguousIntegerSetIter::from(*self)
    }
}

impl<E: Integer + Copy> From<ContiguousIntegerSet<E>> for ContiguousIntegerSetIter<E> {
    fn from(contiguous_integer_set: ContiguousIntegerSet<E>) -> ContiguousIntegerSetIter<E> {
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
    use crate::set::contiguous_integer_set::ContiguousIntegerSet;

    #[test]
    fn test_ord() {
        assert!(ContiguousIntegerSet::new(2, 5) < ContiguousIntegerSet::new(3, 4));
        assert!(ContiguousIntegerSet::new(2, 5) < ContiguousIntegerSet::new(3, 5));
        assert!(ContiguousIntegerSet::new(2, 5) < ContiguousIntegerSet::new(3, 8));
        assert!(ContiguousIntegerSet::new(2, 3) < ContiguousIntegerSet::new(4, 5));
        assert!(ContiguousIntegerSet::new(2, 5) > ContiguousIntegerSet::new(2, 4));
        assert!(ContiguousIntegerSet::new(2, 5) < ContiguousIntegerSet::new(2, 8));
        assert_eq!(
            ContiguousIntegerSet::new(2, 2),
            ContiguousIntegerSet::new(2, 2)
        );
        assert!(ContiguousIntegerSet::new(2, 5) > ContiguousIntegerSet::new(1, 8));
        assert!(ContiguousIntegerSet::new(2, 5) > ContiguousIntegerSet::new(1, 5));
        assert!(ContiguousIntegerSet::new(2, 5) > ContiguousIntegerSet::new(1, 3));
    }
}
