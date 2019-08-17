use std::cmp;
use std::fmt;
use std::ops::Deref;

use num::traits::{NumAssign, NumOps};
use num::traits::cast::{FromPrimitive, ToPrimitive};


pub struct Histogram<T> where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    num_intervals: usize,
    boundaries: Vec<T>,
    counters: Vec<usize>,
}

impl<T> Histogram<T> where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    pub fn new<'a>(elements: &'a Vec<T>, num_intervals: usize, min: T, max: T) -> Result<Histogram<T>, String>
        where &'a T: Deref {
        if num_intervals == 0 {
            return Err(format!("num_intervals should be positive, received {}", num_intervals));
        }
        let n = match T::from_usize(num_intervals) {
            Some(n) => n,
            None => return Err(format!("failed to convert num_intervals: usize ({}) to type T", num_intervals))
        };
        let delta = (max - min) / n;
        if delta <= T::zero() {
            return Err(format!("cannot create positive interval legnths for the given min({}) max({}) and num_intervals({})",
                               min, max, num_intervals));
        }

        let mut boundaries = vec![min];
        let mut acc = min;
        for _ in 1..num_intervals {
            acc += delta;
            boundaries.push(acc);
        }
        boundaries.push(max);

        let mut counters = vec![0usize; num_intervals];
        for a in elements.iter() {
            if *a < min || *a > max {
                return Err(format!("{} out of range [{}, {}]", a, min, max));
            }
            let i = match ((*a - min) / delta).to_usize() {
                Some(i) => i,
                None => return Err(format!("failed to convert {} to an usize index", (*a - min) / delta))
            };
            counters[cmp::min(i, num_intervals - 1)] += 1;
        }
        Ok(Histogram { num_intervals, boundaries, counters })
    }

    pub fn get_ratios(&self) -> Vec<f64> {
        let mut ratio_distribution = vec![0f64; self.num_intervals];
        let total = self.counters.iter().sum::<usize>() as f64;
        for (i, ratio) in ratio_distribution.iter_mut().enumerate() {
            *ratio = self.counters[i] as f64 / total;
        }
        ratio_distribution
    }

    pub fn new_with_auto_range<'a>(elements: &'a Vec<T>, num_intervals: usize) -> Result<Histogram<T>, String>
        where &'a T: Deref, T: Ord {
        let min = match elements.iter().min() {
            None => return Err(format!("failed to extract the min elements when range is set to auto")),
            Some(min) => min
        };
        let max = match elements.iter().max() {
            None => return Err(format!("failed to extract the max elements when range is set to auto")),
            Some(max) => max
        };
        Histogram::new(elements, num_intervals, *min, *max)
    }

    pub fn get_boundaries(&self) -> &Vec<T> {
        &self.boundaries
    }

    pub fn get_counters(&self) -> &Vec<usize> {
        &self.counters
    }
}

impl<T> fmt::Display for Histogram<T>
    where T: PartialOrd + NumAssign + NumOps + FromPrimitive + ToPrimitive + Copy + fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ratios = self.get_ratios();
        let mut cum = 0usize;
        let mut ratio_cum = 0f64;
        let mut reverse_ratio_cum = 1f64;
        let last_i = self.num_intervals - 1;
        writeln!(f, "{:>11.2}  {:>12.2} {:>16} {:>16} {:>16} {:>16} {:>16}",
                 "", "", "count", "cum_count", "ratio", "cum_ratio", "rev_cum_ratio")?;
        for i in 0..last_i {
            cum += self.counters[i];
            ratio_cum += ratios[i];
            writeln!(f, "[{:>10.2}, {:>10.2}): {:>16.2} {:>16.2} {:>16.4} {:>16.4} {:>16.4}",
                     self.boundaries[i], self.boundaries[i + 1], self.counters[i], cum, ratios[i], ratio_cum, reverse_ratio_cum)?;
            reverse_ratio_cum -= ratios[i];
        }
        cum += self.counters[last_i];
        ratio_cum += ratios[last_i];
        writeln!(f, "[{:>10.2}, {:>10.2}]: {:>16.2} {:>16.2} {:>16.4} {:>16.4} {:>16.4}",
                 self.boundaries[last_i], self.boundaries[self.num_intervals],
                 self.counters[last_i], cum, ratios[last_i], ratio_cum, reverse_ratio_cum)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Histogram;

    #[test]
    fn test_histogram() {
        let elements = vec![0., 3.5];
        let num_intervals = 2;
        let histogram = match Histogram::new(&elements, num_intervals, 0., 7.) {
            Ok(h) => h,
            Err(why) => {
                eprintln!("{}", why);
                assert!(false, why);
                return;
            }
        };
        println!("boundaries: {:.2?}", histogram.boundaries);
        println!("elements: {:.2?}", elements);
        println!("counters: {:?}", histogram.counters);
        println!("{}", histogram);
        assert_eq!(num_intervals + 1, histogram.boundaries.len());
        assert_eq!(num_intervals, histogram.counters.len());
    }
}
