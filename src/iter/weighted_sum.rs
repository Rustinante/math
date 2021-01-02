use crate::set::traits::Finite;
use num::{FromPrimitive, Num};

/// # Example
/// ```
/// use math::{
///     interval::I64Interval,
///     iter::{
///         binned_interval_iter::{AggregateOp, IntoBinnedIntervalIter},
///         weighted_sum::WeightedSum,
///     },
///     partition::integer_interval_map::IntegerIntervalMap,
/// };
///
/// let bin_size = 5;
/// let mut interval_map = IntegerIntervalMap::new();
/// interval_map.aggregate(I64Interval::new(-1, 1), 2);
/// interval_map.aggregate(I64Interval::new(14, 17), -1);
///
/// // interval coordinates                        | value
/// // -1 | 0 1  |   ...   |        |              | +2
/// //    |      |   ...   |     14 | 15 16 17     | -1
/// //---------------------------------------------
/// // 0.4|| 0.8 ||  ...   || -0.2 || -0.6         | bin average
/// assert_eq!(
///     interval_map
///         .iter()
///         .into_binned_interval_iter(
///             bin_size,
///             AggregateOp::Average,
///             Box::new(|(&interval, &val)| (interval, val as f64))
///         )
///         .weighted_sum(),
///     2.0
/// );
/// ```
pub trait WeightedSum<Item, V>
where
    V: Copy + Num + FromPrimitive + PartialOrd, {
    fn weighted_sum(self) -> V;
}

macro_rules! impl_weighed_sum_float {
    ($dtype:ty) => {
        impl<I, P> WeightedSum<(P, $dtype), $dtype> for I
        where
            I: Iterator<Item = (P, $dtype)>,
            P: Finite,
        {
            fn weighted_sum(self) -> $dtype {
                crate::stats::kahan_sigma(self, |(interval, value)| {
                    (interval.size() as $dtype) * value
                })
            }
        }
    };
}

impl_weighed_sum_float!(f32);
impl_weighed_sum_float!(f64);

#[cfg(test)]
mod tests {
    use crate::{interval::I64Interval, iter::weighted_sum::WeightedSum};

    #[test]
    fn test_weighted_sum() {
        let arr = vec![
            (I64Interval::new(2, 5), 2f32),
            (I64Interval::new(10, 25), 1.5),
            (I64Interval::new(1, 5), 0.5),
        ];
        // 4 * 2 + 16 * 1.5 + 5 * 0.5
        assert_eq!(arr.into_iter().weighted_sum(), 34.5);

        let arr = vec![
            (I64Interval::new(0, 9), -2f64),
            (I64Interval::new(9, 15), 1.5),
            (I64Interval::new(0, 20), 0.5),
        ];
        // 10 * (-2) + 7 * 1.5 + 21 * 0.5
        assert_eq!(arr.into_iter().weighted_sum(), 1.0);
    }
}
