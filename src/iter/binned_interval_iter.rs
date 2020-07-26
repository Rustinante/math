//! Binning the iterator output into fixed size intervals if the output
//! is of the form `(I64Interval, T)`. Only bins with non-empty intersections
//! with those intervals will be returned.

use crate::{
    interval::{traits::Interval, I64Interval},
    iter::CommonRefinementZip,
    set::traits::{Finite, Intersect},
};
use num::{FromPrimitive, Num};
use std::cmp::Ordering;

/// The value of the associated with `Min` and `Max` are the initial min and max values.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum AggregateOp {
    Max,
    Min,
    Sum,
}

pub trait IntoBinnedIntervalIter<V>
where
    Self: Iterator + Sized,
    V: Copy + Num + FromPrimitive + PartialOrd, {
    fn into_binned_interval_iter(
        self,
        bin_size: i64,
        aggregate_op: AggregateOp,
        interval_value_extractor: Box<dyn Fn(<Self as Iterator>::Item) -> (I64Interval, V)>,
    ) -> BinnedIntervalIter<Self, V>;
}

impl<I, V> IntoBinnedIntervalIter<V> for I
where
    I: Iterator,
    V: Copy + Num + FromPrimitive + PartialOrd,
{
    fn into_binned_interval_iter(
        self,
        bin_size: i64,
        aggregate_op: AggregateOp,
        interval_value_extractor: Box<dyn Fn(<I as Iterator>::Item) -> (I64Interval, V)>,
    ) -> BinnedIntervalIter<Self, V> {
        BinnedIntervalIter::new(self, bin_size, aggregate_op, interval_value_extractor)
    }
}

/// With imaginary bins of size `bin_size` and aligned at `0`,
/// returns a value for each bin that intersects one or more intervals from
/// the original iterator `iter`, where the value at each intersection is
/// obtained by applying the operation specified by the `aggregate_op` for
/// all the overlapping intervals and their associated values, where the value of each
/// overlapping interval is multiplied by the length of the interval if the `aggregate_op`
/// is `Sum`.
///
/// # Panics
/// The iterator will panic if the intervals returned by the original `iter` are not
/// disjoint or increasing.
///
/// # Example
/// ```
/// use math::{
///     interval::I64Interval,
///     iter::binned_interval_iter::{AggregateOp, IntoBinnedIntervalIter},
///     partition::integer_interval_map::IntegerIntervalMap,
/// };
///
/// let bin_size = 5;
/// let mut interval_map = IntegerIntervalMap::new();
/// interval_map.aggregate(I64Interval::new(-1, 1), 2);
/// interval_map.aggregate(I64Interval::new(14, 17), -1);
///
/// // interval coordinates                       | value
/// // -1 | 0 1  |   ...   |        |              | +2
/// //    |      |   ...   |     14 | 15 16 17     | -1
/// //---------------------------------------------
/// //  2 || 4   ||  ...   || -1   || -3          | bin sum
/// //  2 || 2   ||  ...   || -1   || -1          | bin max
/// //  2 || 2   ||  ...   || -1   || -1          | bin min
/// assert_eq!(
///     interval_map
///         .iter()
///         .into_binned_interval_iter(
///             bin_size,
///             AggregateOp::Sum,
///             Box::new(|(&interval, &val)| (interval, val))
///         )
///         .collect::<Vec<(I64Interval, i32)>>(),
///     vec![
///         (I64Interval::new(-5, -1), 2),
///         (I64Interval::new(0, 4), 4),
///         (I64Interval::new(10, 14), -1),
///         (I64Interval::new(15, 19), -3),
///     ]
/// );
/// assert_eq!(
///     interval_map
///         .iter()
///         .into_binned_interval_iter(
///             bin_size,
///             AggregateOp::Max,
///             Box::new(|(&interval, &val)| (interval, val))
///         )
///         .collect::<Vec<(I64Interval, i32)>>(),
///     vec![
///         (I64Interval::new(-5, -1), 2),
///         (I64Interval::new(0, 4), 2),
///         (I64Interval::new(10, 14), -1),
///         (I64Interval::new(15, 19), -1),
///     ]
/// );
/// assert_eq!(
///     interval_map
///         .iter()
///         .into_binned_interval_iter(
///             bin_size,
///             AggregateOp::Min,
///             Box::new(|(&interval, &val)| (interval, val))
///         )
///         .collect::<Vec<(I64Interval, i32)>>(),
///     vec![
///         (I64Interval::new(-5, -1), 2),
///         (I64Interval::new(0, 4), 2),
///         (I64Interval::new(10, 14), -1),
///         (I64Interval::new(15, 19), -1),
///     ]
/// );
/// ```
pub struct BinnedIntervalIter<I, V>
where
    I: Iterator,
    V: Copy + Num + FromPrimitive + PartialOrd, {
    iter: I,
    bin_size: i64,
    aggregate_op: AggregateOp,
    iter_item_interval_value_extractor: Box<dyn Fn(<I as Iterator>::Item) -> (I64Interval, V)>,
    current_interval_val: Option<(I64Interval, V)>,
    current_bin: Option<I64Interval>,
}

impl<I, V> BinnedIntervalIter<I, V>
where
    I: Iterator,
    V: Copy + Num + FromPrimitive + PartialOrd,
{
    pub fn new(
        mut iter: I,
        bin_size: i64,
        aggregate_op: AggregateOp,
        iter_item_interval_value_extractor: Box<dyn Fn(<I as Iterator>::Item) -> (I64Interval, V)>,
    ) -> Self {
        assert!(bin_size >= 1, "bin_size must be at least 1");
        let current_interval_val = iter
            .next()
            .map(|item| iter_item_interval_value_extractor(item));
        BinnedIntervalIter {
            iter,
            bin_size,
            aggregate_op,
            iter_item_interval_value_extractor,
            current_interval_val,
            current_bin: None,
        }
    }
}

impl<I, V> Iterator for BinnedIntervalIter<I, V>
where
    I: Iterator,
    V: Copy + Num + FromPrimitive + PartialOrd,
{
    type Item = (I64Interval, V);

    /// After every iteration, `self.current_bin` can be
    /// * `None`: indicating that the current interval has not been processed at all
    /// * `Some`: indicating the last used bin
    ///
    /// and `self.current_interval_val` can be
    /// * `None`: indicating that all the intervals have been processed
    /// * `Some`: indicating that the current interval still has unprocessed elements
    ///
    /// # panics: if the intervals returned by the original `iter` are not disjoint or increasing.
    fn next(&mut self) -> Option<Self::Item> {
        let current_interval = &self.current_interval_val;
        match current_interval {
            None => None,
            Some((mut interval, mut val)) => {
                let mut aggregate: Option<V> = None;

                let interval_start = interval.get_start();

                // the start of the first bin that overlaps the interval
                let first_overlap_bin_start = if interval_start >= 0 {
                    (interval_start / self.bin_size) * self.bin_size
                } else {
                    // take the ceiling towards the negative direction
                    ((interval_start - (self.bin_size - 1)) / self.bin_size) * self.bin_size
                };

                let bin_start = match self.current_bin {
                    None => {
                        // have not processed the current interval at all yet
                        first_overlap_bin_start
                    }
                    Some(old_bin) => {
                        if old_bin.get_end() < interval_start {
                            first_overlap_bin_start
                        } else {
                            old_bin.get_end() + 1
                        }
                    }
                };
                let bin_end_inclusive = bin_start + self.bin_size - 1;
                self.current_bin = Some(I64Interval::new(bin_start, bin_end_inclusive));

                loop {
                    aggregate = match self.aggregate_op {
                        AggregateOp::Max => Some(aggregate.map_or_else(
                            || val,
                            |agg| match agg.partial_cmp(&val).unwrap() {
                                Ordering::Less => val,
                                _ => agg,
                            },
                        )),
                        AggregateOp::Min => Some(aggregate.map_or_else(
                            || val,
                            |agg| match agg.partial_cmp(&val).unwrap() {
                                Ordering::Greater => val,
                                _ => agg,
                            },
                        )),
                        AggregateOp::Sum => Some(
                            aggregate.unwrap_or(V::zero())
                                + val
                                    * V::from_usize(
                                        self.current_bin
                                            .unwrap()
                                            .intersect(&interval)
                                            .map_or_else(|| 0, |i| i.size()),
                                    )
                                    .unwrap(),
                        ),
                    };

                    let interval_end_inclusive = interval.get_end();

                    // Either the interval is contained in the bin
                    // or it extends rightwards beyond the bin.
                    if interval_end_inclusive <= bin_end_inclusive {
                        // If it is contained in the bin, we will get the next interval.
                        self.current_interval_val = self
                            .iter
                            .next()
                            .map(|item| (self.iter_item_interval_value_extractor)(item));
                        match self.current_interval_val {
                            None => {
                                break;
                            }
                            Some((i, v)) => {
                                assert!(
                                    interval_end_inclusive < i.get_start(),
                                    "previous interval end ({}) >= next interval start ({})",
                                    interval_end_inclusive,
                                    i.get_start()
                                );
                                interval = i;
                                val = v;
                                if interval.get_start() > bin_end_inclusive {
                                    break;
                                }
                            }
                        };
                    } else {
                        // Otherwise, the current bin has received all the information
                        // from the intersecting intervals and is ready to be returned.
                        break;
                    }
                }
                Some((self.current_bin.unwrap(), aggregate.unwrap()))
            }
        }
    }
}

type IntType = i64;

impl<I, V> CommonRefinementZip<IntType, (I64Interval, V), I64Interval, V>
    for BinnedIntervalIter<I, V>
where
    I: Iterator,
    V: Copy + Num + FromPrimitive + PartialOrd,
{
    fn get_interval_value_extractor(
        &self,
    ) -> Box<dyn Fn(<Self as Iterator>::Item) -> (I64Interval, V)> {
        Box::new(|item| (item.0, item.1))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        interval::I64Interval,
        iter::{
            binned_interval_iter::{AggregateOp, IntoBinnedIntervalIter},
            CommonRefinementZip,
        },
        partition::integer_interval_map::IntegerIntervalMap,
    };

    #[test]
    fn test_binned_interval_iter() {
        let bin_size = 3;
        let mut interval_map = IntegerIntervalMap::new();
        interval_map.aggregate(I64Interval::new(-1, 4), 2);
        interval_map.aggregate(I64Interval::new(6, 8), 4);
        interval_map.aggregate(I64Interval::new(4, 7), 1);

        // interval coordinates           | value
        // -1 | 0 1 2 | 3 4   |           | +2
        //    |       |       | 6 7 8     | +4
        //    |       |   4 5 | 6 7       | +1
        //---------------------------------
        //  2 | 2 2 2 | 2 3 1 | 5 5 4 |   | superposed values
        //  2 || 6    || 6    || 14   ||  | bin sum
        //  2 || 2    || 3    || 5    ||  | bin max
        //  2 || 2    || 1    || 4    ||  | bin min

        macro_rules! get_actual {
            ($op:expr) => {
                interval_map
                    .iter()
                    .into_binned_interval_iter(
                        bin_size,
                        $op,
                        Box::new(|(&interval, &val)| (interval, val)),
                    )
                    .collect::<Vec<(I64Interval, i32)>>()
            };
        }

        assert_eq!(get_actual!(AggregateOp::Sum), vec![
            (I64Interval::new(-3, -1), 2),
            (I64Interval::new(0, 2), 6),
            (I64Interval::new(3, 5), 6),
            (I64Interval::new(6, 8), 14),
        ]);
        assert_eq!(get_actual!(AggregateOp::Max), vec![
            (I64Interval::new(-3, -1), 2),
            (I64Interval::new(0, 2), 2),
            (I64Interval::new(3, 5), 3),
            (I64Interval::new(6, 8), 5),
        ]);
        assert_eq!(get_actual!(AggregateOp::Min), vec![
            (I64Interval::new(-3, -1), 2),
            (I64Interval::new(0, 2), 2),
            (I64Interval::new(3, 5), 1),
            (I64Interval::new(6, 8), 4),
        ]);

        interval_map.aggregate(I64Interval::new(2, 4), -3);
        interval_map.aggregate(I64Interval::new(14, 16), -2);

        // interval coordinates           | value
        // -1 | 0 1 2 | 3 4   |           | +2
        //    |       |       | 6 7 8     | +4
        //    |       |   4 5 | 6 7       | +1
        //    |     2 | 3 4   |           | -3
        //---------------------------------
        //  2 | 2 2 -1|-1 0 1 | 5 5 4 |   | superposed values
        //  2 || 3    || 0    || 14   ||  | bin sum
        //  2 || 2    || 1    || 5    ||  | bin max
        //  2 || -1   || -1   || 4    ||  | bin min
        assert_eq!(get_actual!(AggregateOp::Sum), vec![
            (I64Interval::new(-3, -1), 2),
            (I64Interval::new(0, 2), 3),
            (I64Interval::new(3, 5), 0),
            (I64Interval::new(6, 8), 14),
            (I64Interval::new(12, 14), -2),
            (I64Interval::new(15, 17), -4),
        ]);
        assert_eq!(get_actual!(AggregateOp::Max), vec![
            (I64Interval::new(-3, -1), 2),
            (I64Interval::new(0, 2), 2),
            (I64Interval::new(3, 5), 1),
            (I64Interval::new(6, 8), 5),
            (I64Interval::new(12, 14), -2),
            (I64Interval::new(15, 17), -2),
        ]);
        assert_eq!(get_actual!(AggregateOp::Min), vec![
            (I64Interval::new(-3, -1), 2),
            (I64Interval::new(0, 2), -1),
            (I64Interval::new(3, 5), -1),
            (I64Interval::new(6, 8), 4),
            (I64Interval::new(12, 14), -2),
            (I64Interval::new(15, 17), -2),
        ]);
    }

    #[test]
    fn test_common_refinement_zip() {
        let bin_size = 3;
        let mut map1 = IntegerIntervalMap::new();
        map1.aggregate(I64Interval::new(-1, 4), 2);
        map1.aggregate(I64Interval::new(6, 8), 4);
        map1.aggregate(I64Interval::new(4, 7), 1);

        // interval coordinates           | value
        // -1 | 0 1 2 | 3 4   |           | +2
        //    |       |       | 6 7 8     | +4
        //    |       |   4 5 | 6 7       | +1
        //---------------------------------
        //  2 | 2 2 2 | 2 3 1 | 5 5 4 |   | superposed values
        //  2 || 6    || 6    || 14   ||  | bin sum

        let mut map2 = IntegerIntervalMap::new();
        map2.aggregate(I64Interval::new(1, 2), -2);
        map2.aggregate(I64Interval::new(6, 8), 9);
        // interval coordinates            | value
        // |  1   2 |       |              | -2
        // |        |       |  6   7   8   | +9
        //---------------------------------
        // | -2  -2 |       |  9   9   9   | superposed values
        // || -4    ||      || 27          | bin sum

        assert_eq!(
            map2.iter()
                .into_binned_interval_iter(
                    bin_size,
                    AggregateOp::Sum,
                    Box::new(|(&interval, &val)| (interval, val))
                )
                .collect::<Vec<(I64Interval, i32)>>(),
            vec![(I64Interval::new(0, 2), -4), (I64Interval::new(6, 8), 27)]
        );

        let actual: Vec<(I64Interval, Vec<Option<i32>>)> = map1
            .iter()
            .into_binned_interval_iter(
                bin_size,
                AggregateOp::Sum,
                Box::new(|(&interval, &val)| (interval, val)),
            )
            .common_refinement_zip(map2.iter().into_binned_interval_iter(
                bin_size,
                AggregateOp::Sum,
                Box::new(|(&interval, &val)| (interval, val)),
            ))
            .collect();

        let expected = vec![
            (I64Interval::new(-3, -1), vec![Some(2), None]),
            (I64Interval::new(0, 2), vec![Some(6), Some(-4)]),
            (I64Interval::new(3, 5), vec![Some(6), None]),
            (I64Interval::new(6, 8), vec![Some(14), Some(27)]),
        ];
        assert_eq!(actual, expected);
    }
}
