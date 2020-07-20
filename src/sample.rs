use rand::distributions::{Distribution, Uniform};

use crate::{
    set::traits::Finite,
    traits::{Collecting, ToIterator},
};

pub mod trait_impl;

pub trait Sample<'a, I: Iterator<Item = E>, E, O: Collecting<E> + Default>:
    Finite + ToIterator<'a, I, E> {
    /// samples `size` elements without replacement
    /// `size`: the number of samples to be drawn
    /// returns Err if `size` is larger than the population size
    fn sample_subset_without_replacement<'s: 'a>(&'s self, size: usize) -> Result<O, String> {
        let mut remaining = self.size();
        if size > remaining {
            return Err(format!(
                "desired sample size {} > population size {}",
                size, remaining
            ));
        }
        let mut samples = O::default();
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

    fn sample_with_replacement<'s: 'a>(&'s self, size: usize) -> Result<O, String> {
        let population_size = self.size();
        if population_size == 0 {
            return Err("cannot sample from a population of 0 elements".to_string());
        }
        let mut samples = O::default();
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0., population_size as f64);
        for _ in 0..size {
            samples.collect(
                self.to_iter()
                    .nth(uniform.sample(&mut rng) as usize)
                    .unwrap(),
            );
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use crate::set::{
        contiguous_integer_set::ContiguousIntegerSet, ordered_integer_set::OrderedIntegerSet,
        traits::Finite,
    };

    use super::Sample;

    #[test]
    fn test_sampling_without_replacement() {
        let interval = ContiguousIntegerSet::new(0, 100);
        let num_samples = 25;
        let samples = interval
            .sample_subset_without_replacement(num_samples)
            .unwrap();
        assert_eq!(samples.size(), num_samples);

        let set = OrderedIntegerSet::from_slice(&[[-89, -23], [-2, 100], [300, 345]]);
        let num_samples = 18;
        let samples = set.sample_subset_without_replacement(num_samples).unwrap();
        assert_eq!(samples.size(), num_samples);
    }

    #[test]
    fn test_sampling_with_replacement() {
        let num_samples = 25;
        let v = vec![1];
        let samples = v.sample_with_replacement(num_samples);
        assert_eq!(samples, Ok(vec![1; num_samples]));
        assert!(Vec::<f32>::new()
            .sample_with_replacement(num_samples)
            .is_err());
    }
}
