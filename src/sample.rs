use rand::distributions::{Distribution, Uniform};

use crate::set::traits::Finite;
use crate::traits::{Collecting, Constructable, ToIterator};

pub mod trait_impl;

pub trait Sample<'a, I: Iterator<Item=E>, E, O: Collecting<E> + Constructable>: Finite + ToIterator<'a, I, E> {
    /// samples `size` elements without replacement
    /// `size`: the number of samples to be drawn
    /// returns Err if `size` is larger than the population size
    fn sample_subset_without_replacement<'s: 'a>(&'s self, size: usize) -> Result<O, String> {
        let mut remaining = self.size();
        if size > remaining {
            return Err(format!("desired sample size {} > population size {}", size, remaining));
        }
        let mut samples = O::new();
        let mut needed = size;
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0., 1.);

        for element in self.to_iter() {
            if uniform.sample(&mut rng) <= (needed as f64 / remaining as f64) {
                samples.collect(element);
                needed -= 1;
            }
            remaining -= 1;
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use crate::set::ordered_integer_set::{ContiguousIntegerSet, OrderedIntegerSet};
    use crate::set::traits::Finite;

    use super::Sample;

    #[test]
    fn test_sample() {
        let interval = ContiguousIntegerSet::new(0, 100);
        let num_samples = 25;
        let samples = interval.sample_subset_without_replacement(num_samples).unwrap();
        assert_eq!(samples.size(), num_samples);

        let set = OrderedIntegerSet::from_slice(&[[-89, -23], [-2, 100], [300, 345]]);
        let num_samples = 18;
        let samples = set.sample_subset_without_replacement(num_samples).unwrap();
        assert_eq!(samples.size(), num_samples);
    }
}
