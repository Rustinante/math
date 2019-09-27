use std::cmp;
use std::fmt;
use std::ops::Deref;

use num::traits::{NumAssign, NumOps};
use num::traits::cast::{FromPrimitive, ToPrimitive};

use crate::traits::{Collecting, ToIterator};

pub struct Histogram<T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    boundaries: Vec<T>,
    counters: Vec<usize>,
    num_less_than_min: usize,
    num_larger_than_max: usize,
}

impl<T> Histogram<T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    pub fn new<'a>(
        elements: &'a Vec<T>,
        num_intervals: usize,
        min: T,
        max: T,
    ) -> Result<Histogram<T>, String>
        where &'a T: Deref {
        if num_intervals == 0 {
            return Err(format!("num_intervals should be positive, received {}", num_intervals));
        }
        if max < min {
            return Err(format!("max ({}) has to be >= min ({})", max, min));
        }
        let n = match T::from_usize(num_intervals) {
            Some(n) => n,
            None => return Err(
                format!("failed to convert num_intervals: usize ({}) to type T", num_intervals)
            )
        };
        let delta = (max - min) / n;
        if delta <= T::zero() {
            return Err(format!(
                "cannot create positive interval legnths for the given \
                min({}) max({}) and num_intervals({})",
                min, max, num_intervals
            ));
        }

        let mut boundaries = vec![min];
        let mut acc = min;
        for _ in 1..num_intervals {
            acc += delta;
            boundaries.push(acc);
        }
        boundaries.push(max);

        let mut num_larger_than_max = 0;
        let mut num_less_than_min = 0;
        let mut counters = vec![0usize; num_intervals];
        for a in elements.iter() {
            if *a < min {
                num_less_than_min += 1;
            } else if *a > max {
                num_larger_than_max += 1;
            } else {
                let i = match ((*a - min) / delta).to_usize() {
                    Some(i) => i,
                    None => return Err(
                        format!("failed to convert {} to an usize index", (*a - min) / delta)
                    )
                };
                counters[cmp::min(i, num_intervals - 1)] += 1;
            }
        }
        Ok(Histogram { boundaries, counters, num_less_than_min, num_larger_than_max })
    }

    #[inline]
    pub fn num_intervals(&self) -> usize {
        self.boundaries.len() - 1
    }

    pub fn get_ratios(&self) -> Vec<f64> {
        let mut ratio_distribution = vec![0f64; self.num_intervals()];
        let total = self.counters.iter().sum::<usize>() as f64;
        for (i, ratio) in ratio_distribution.iter_mut().enumerate() {
            *ratio = self.counters[i] as f64 / total;
        }
        ratio_distribution
    }

    pub fn new_with_auto_range<'a>(
        elements: &'a Vec<T>,
        num_intervals: usize,
    ) -> Result<Histogram<T>, String>
        where &'a T: Deref, T: Ord {
        let min = match elements.iter().min() {
            None => return Err(
                format!("failed to extract the min elements when range is set to auto")
            ),
            Some(min) => min
        };
        let max = match elements.iter().max() {
            None => return Err(
                format!("failed to extract the max elements when range is set to auto")
            ),
            Some(max) => max
        };
        Histogram::new(elements, num_intervals, *min, *max)
    }

    #[inline]
    pub fn get_boundaries(&self) -> &Vec<T> {
        &self.boundaries
    }

    #[inline]
    pub fn get_counters(&self) -> &Vec<usize> {
        &self.counters
    }

    #[inline]
    pub fn min_boundary(&self) -> T {
        *self.boundaries.first().unwrap()
    }

    #[inline]
    pub fn max_boundary(&self) -> T {
        *self.boundaries.last().unwrap()
    }
}

impl<T> Collecting<T> for Histogram<T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    fn collect(&mut self, item: T) {
        let delta = self.boundaries[1] - self.boundaries[0];
        let num_intervals = self.num_intervals();
        let min_boundary = self.min_boundary();
        if item < min_boundary {
            self.num_less_than_min += 1;
        } else if item > self.max_boundary() {
            self.num_larger_than_max += 1;
        } else {
            let i = ((item - min_boundary) / delta).to_usize().unwrap();
            self.counters[cmp::min(i, num_intervals - 1)] += 1;
        }
    }
}

impl<T> fmt::Display for Histogram<T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ratios = self.get_ratios();
        let mut cum = 0usize;
        let mut ratio_cum = 0f64;
        let mut reverse_ratio_cum = 1f64;
        let last_i = self.num_intervals() - 1;
        writeln!(f, "{:>11.2}  {:>12.2} {:>16} {:>16} {:>16} {:>16} {:>16}",
                 "", "", "count", "cum_count", "ratio", "cum_ratio", "rev_cum_ratio")?;
        for i in 0..last_i {
            cum += self.counters[i];
            ratio_cum += ratios[i];
            writeln!(f,
                     "[{:>10.2}, {:>10.2}): {:>16.2} {:>16.2} {:>16.4} {:>16.4} {:>16.4}",
                     self.boundaries[i],
                     self.boundaries[i + 1],
                     self.counters[i],
                     cum,
                     ratios[i],
                     ratio_cum,
                     reverse_ratio_cum
            )?;
            reverse_ratio_cum -= ratios[i];
        }
        cum += self.counters[last_i];
        ratio_cum += ratios[last_i];
        writeln!(
            f,
            "[{:>10.2}, {:>10.2}]: {:>16.2} {:>16.2} {:>16.4} {:>16.4} {:>16.4}",
            self.boundaries[last_i],
            self.boundaries[self.num_intervals()],
            self.counters[last_i],
            cum,
            ratios[last_i],
            ratio_cum,
            reverse_ratio_cum
        )?;
        writeln!(
            f,
            "\n\
            ({:>10}, {:>10.2}) count: {}\n\
            ({:>10.2}, {:>10}) count: {}",
            "-inf",
            self.min_boundary(),
            self.num_less_than_min,
            self.max_boundary(),
            "inf",
            self.num_larger_than_max,
        )?;
        Ok(())
    }
}

pub type HistogramEntry<T> = (T, T, usize);

impl<'a, T> ToIterator<'a, HistogramIter<'a, T>, HistogramEntry<T>> for Histogram<T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    fn to_iter(&'a self) -> HistogramIter<'a, T> {
        HistogramIter {
            histogram: &self,
            cursor: 0,
        }
    }
}

pub struct HistogramIter<'a, T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    histogram: &'a Histogram<T>,
    cursor: usize,
}

/// An iterator that iterates through the entries of the histogram
/// ```
/// use analytic::histogram::Histogram;
/// use analytic::traits::ToIterator;
/// let histogram = Histogram::new(&vec![4., 0., 3.5], 2, 0., 7.).unwrap();
/// let mut iter = histogram.to_iter();
/// assert_eq!(Some((0., 3.5, 1)), iter.next());
/// assert_eq!(Some((3.5, 7., 2)), iter.next());
/// assert_eq!(None, iter.next());
/// ```
impl<'a, T> Iterator for HistogramIter<'a, T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    type Item = HistogramEntry<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.cursor;
        if i >= self.histogram.num_intervals() {
            None
        } else {
            self.cursor += 1;
            Some((
                self.histogram.boundaries[i],
                self.histogram.boundaries[i + 1],
                self.histogram.counters[i],
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::{Collecting, ToIterator};

    use super::Histogram;

    #[test]
    fn test_histogram() {
        let elements = vec![4., 0., 3.5];
        let num_intervals = 2;
        let mut histogram = match Histogram::new(&elements, num_intervals, 0., 7.) {
            Ok(h) => h,
            Err(why) => {
                eprintln!("{}", why);
                assert!(false, why);
                return;
            }
        };
        histogram.collect(4.);
        let mut iter = histogram.to_iter();
        assert_eq!(Some((0., 3.5, 1)), iter.next());
        assert_eq!(Some((3.5, 7., 3)), iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(num_intervals + 1, histogram.boundaries.len());
        assert_eq!(num_intervals, histogram.counters.len());
        assert_eq!(histogram.counters[0], 1);
        assert_eq!(histogram.counters[1], 3);
    }
}
