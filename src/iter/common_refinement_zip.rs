//! If part of the iterators outputs are disjoint increasing integer intervals,
//! then the iterators can be zipped together and the iteration can proceeds at
//! the granularity of the common refinements of all the integer intervals.

use crate::{
    interval::{traits::Interval, IntInterval},
    set::traits::Intersect,
};
use num::{Integer, Num, ToPrimitive};
use std::{
    collections::BTreeSet,
    marker::{PhantomData, Sized},
};

pub trait CommonRefinementZip<B, X, P, V>
where
    B: Copy + Num + Ord,
    Self: Iterator<Item = X> + Sized,
    P: Clone + Interval<B> + for<'b> Intersect<&'b P, Option<P>>, {
    fn get_interval_value_extractor(&self) -> Box<dyn Fn(<Self as Iterator>::Item) -> (P, V)>;

    fn common_refinement_zip(
        mut self,
        mut other: Self,
    ) -> CommonRefinementZipped<B, Self, X, P, V> {
        let extractor = self.get_interval_value_extractor();
        let mut intervals = Vec::new();
        let mut values = Vec::new();
        match self.next() {
            None => {
                intervals.push(None);
                values.push(None);
            }
            Some(x) => {
                let (interval, value) = extractor(x);
                intervals.push(Some(interval));
                values.push(Some(value));
            }
        }
        match other.next() {
            None => {
                intervals.push(None);
                values.push(None);
            }
            Some(x) => {
                let (interval, value) = extractor(x);
                intervals.push(Some(interval));
                values.push(Some(value));
            }
        }
        CommonRefinementZipped {
            iters: vec![self, other],
            intervals,
            values,
            extractor,
            phantom: PhantomData,
        }
    }
}

/// # Example
/// ```
/// use analytic::{
///     interval::{traits::Interval, IntInterval},
///     iter::CommonRefinementZip,
/// };
/// use std::collections::BTreeMap;
///
/// let m1: BTreeMap<IntInterval<usize>, i32> =
///     vec![(IntInterval::new(0, 5), 5), (IntInterval::new(8, 10), 2)]
///         .into_iter()
///         .collect();
///
/// let m2: BTreeMap<IntInterval<usize>, i32> =
///     vec![(IntInterval::new(2, 4), 8), (IntInterval::new(12, 13), 9)]
///         .into_iter()
///         .collect();
///
/// let mut iter = m1.iter().common_refinement_zip(m2.iter());
/// assert_eq!(
///     Some((IntInterval::new(0, 1), vec![Some(5), None])),
///     iter.next()
/// );
/// assert_eq!(
///     Some((IntInterval::new(2, 4), vec![Some(5), Some(8)])),
///     iter.next()
/// );
/// assert_eq!(
///     Some((IntInterval::new(5, 5), vec![Some(5), None])),
///     iter.next()
/// );
/// assert_eq!(
///     Some((IntInterval::new(8, 10), vec![Some(2), None])),
///     iter.next()
/// );
/// assert_eq!(
///     Some((IntInterval::new(12, 13), vec![None, Some(9)])),
///     iter.next()
/// );
/// assert_eq!(None, iter.next());
/// ```
impl<'a, V, B: Integer + Copy + ToPrimitive>
    CommonRefinementZip<B, (&'a IntInterval<B>, &'a V), IntInterval<B>, V>
    for std::collections::btree_map::Iter<'a, IntInterval<B>, V>
where
    B: 'a,
    V: 'a + Clone,
{
    fn get_interval_value_extractor(
        &self,
    ) -> Box<dyn Fn(<Self as Iterator>::Item) -> (IntInterval<B>, V)> {
        Box::new(|item| ((*item.0).clone(), (*item.1).clone()))
    }
}

impl<'a, V, B: Integer + Copy + ToPrimitive>
    CommonRefinementZip<B, (IntInterval<B>, V), IntInterval<B>, V>
    for std::collections::btree_map::IntoIter<IntInterval<B>, V>
where
    B: 'a,
{
    fn get_interval_value_extractor(
        &self,
    ) -> Box<dyn Fn(<Self as Iterator>::Item) -> (IntInterval<B>, V)> {
        Box::new(|item| (item.0, item.1))
    }
}

/// # Iterator Algorithm Description
/// Given a list of iterators, a list of the current minimum interval for each iterator will be
/// maintained together with their associated values. Then, at each pass the smallest minimum common
/// refinement of the current intervals is subtracted from each interval. A list of values will be
/// returned along with the common refinement. Each value will be the value associated with the
/// iterated interval if the common refinement has a non-empty intersection with the corresponding
/// interval, and `None` otherwise.
///
/// If an interval becomes empty after the subtraction, the corresponding iterator will be called
/// to replace the interval with the next interval, together with the associated values.
///
/// # Fields
/// * `iters`: the list of zipped iterators.
/// * `intervals`: the intervals assocaited with each iterator for the current pass.
/// * `values`: the values associated with each iterator for the current pass.
/// * `extractor`: a function that extracts a tuple of (interval, value) from each of the items
///   yielded from the iterators.
pub struct CommonRefinementZipped<B, I, X, P, V>
where
    B: Copy + Num + Ord,
    I: Iterator<Item = X> + Sized,
    P: Clone + Interval<B> + for<'b> Intersect<&'b P, Option<P>>, {
    iters: Vec<I>,
    intervals: Vec<Option<P>>,
    values: Vec<Option<V>>,
    extractor: Box<dyn Fn(X) -> (P, V)>,
    phantom: PhantomData<B>,
}

impl<B, I, X, P, V> Iterator for CommonRefinementZipped<B, I, X, P, V>
where
    B: Copy + Num + Ord,
    I: Iterator<Item = X> + Sized,
    P: Clone + Interval<B> + for<'b> Intersect<&'b P, Option<P>>,
    V: Clone,
{
    type Item = (P, Vec<Option<V>>);

    fn next(&mut self) -> Option<Self::Item> {
        let starts: BTreeSet<B> = self
            .intervals
            .iter()
            .filter_map(|i| i.clone().and_then(|i| Some(i.get_start())))
            .collect();

        let ends: BTreeSet<B> = self
            .intervals
            .iter()
            .filter_map(|i| i.clone().and_then(|i| Some(i.get_end())))
            .collect();

        let mut starts_iter = starts.iter();
        let min_start = match starts_iter.next() {
            // if all intervals are empty, it means that all the iterators have been exhausted
            None => return None,
            Some(&a) => a,
        };
        let second_min_start = starts_iter.next();
        let min_end = *ends.iter().next().unwrap();

        let min_refinement = match second_min_start {
            Some(&second_min_start) => {
                if second_min_start <= min_end {
                    P::from_boundaries(min_start, second_min_start - B::one())
                } else {
                    P::from_boundaries(min_start, min_end)
                }
            }
            None => P::from_boundaries(min_start, min_end),
        };

        let mut refinement_values = Vec::new();
        for ((interval, iter), v) in self
            .intervals
            .iter_mut()
            .zip(self.iters.iter_mut())
            .zip(self.values.iter_mut())
        {
            match interval {
                Some(i) => {
                    if i.has_non_empty_intersection_with(&min_refinement) {
                        refinement_values.push((*v).clone());

                        // subtract the min_refinement from the interval
                        // min_start <= i.get_start() <= min_end <= i.get_end()
                        let remainder =
                            P::from_boundaries(min_refinement.get_end() + B::one(), i.get_end());
                        if remainder.is_empty() {
                            match iter.next() {
                                None => {
                                    *interval = None;
                                    *v = None;
                                }
                                Some(x) => {
                                    let (new_interval, new_val) = (self.extractor)(x);
                                    *interval = Some(new_interval);
                                    *v = Some(new_val);
                                }
                            }
                        } else {
                            *interval = Some(remainder);
                        }
                    } else {
                        refinement_values.push(None);
                    }
                }
                None => {
                    refinement_values.push(None);
                }
            }
        }
        Some((min_refinement, refinement_values))
    }
}

impl<B, I, X, P, V> CommonRefinementZipped<B, I, X, P, V>
where
    B: Copy + Num + Ord,
    I: Iterator<Item = X> + Sized,
    P: Clone + Interval<B> + for<'b> Intersect<&'b P, Option<P>>,
{
    /// ```
    /// use analytic::{
    ///     interval::{traits::Interval, IntInterval},
    ///     iter::CommonRefinementZip,
    /// };
    /// use std::collections::BTreeMap;
    ///
    /// let m1: BTreeMap<IntInterval<usize>, i32> =
    ///     vec![(IntInterval::new(0, 10), 5), (IntInterval::new(16, 17), 21)]
    ///         .into_iter()
    ///         .collect();
    ///
    /// let m2: BTreeMap<IntInterval<usize>, i32> =
    ///     vec![(IntInterval::new(2, 3), 8), (IntInterval::new(12, 20), 9)]
    ///         .into_iter()
    ///         .collect();
    ///
    /// let m3: BTreeMap<IntInterval<usize>, i32> = vec![
    ///     (IntInterval::new(2, 4), 7),
    ///     (IntInterval::new(9, 10), -1),
    ///     (IntInterval::new(15, 20), 0),
    /// ]
    /// .into_iter()
    /// .collect();
    ///
    /// let mut iter = m1
    ///     .iter()
    ///     .common_refinement_zip(m2.iter())
    ///     .common_refinement_flat_zip(m3.iter());
    ///
    /// assert_eq!(
    ///     Some((IntInterval::new(0, 1), vec![Some(5), None, None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(2, 3), vec![Some(5), Some(8), Some(7)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(4, 4), vec![Some(5), None, Some(7)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(5, 8), vec![Some(5), None, None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(9, 10), vec![Some(5), None, Some(-1)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(12, 14), vec![None, Some(9), None])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(15, 15), vec![None, Some(9), Some(0)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(16, 17), vec![Some(21), Some(9), Some(0)])),
    ///     iter.next()
    /// );
    /// assert_eq!(
    ///     Some((IntInterval::new(18, 20), vec![None, Some(9), Some(0)])),
    ///     iter.next()
    /// );
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn common_refinement_flat_zip(
        mut self,
        mut other: I,
    ) -> CommonRefinementZipped<B, I, X, P, V>
    where
        I: Iterator<Item = X> + Sized, {
        match other.next() {
            None => {
                self.intervals.push(None);
                self.values.push(None);
            }
            Some(x) => {
                let (i, v) = (self.extractor)(x);
                self.intervals.push(Some(i.clone()));
                self.values.push(Some(v));
            }
        }
        self.iters.push(other);
        CommonRefinementZipped {
            iters: self.iters,
            intervals: self.intervals,
            values: self.values,
            extractor: self.extractor,
            phantom: PhantomData,
        }
    }
}
