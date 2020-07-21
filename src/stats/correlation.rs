use crate::stats::{kahan_sigma, kahan_sigma_return_counter};
use num::ToPrimitive;

/// The weights and values are of type `T`.
pub fn weighted_correlation<'a, T, I: Iterator<Item = &'a T>, F1, F2, F3>(
    a_iter: F1,
    b_iter: F2,
    weight_iter: F3,
) -> f64
where
    T: ToPrimitive + Copy + 'a,
    F1: Fn() -> I,
    F2: Fn() -> I,
    F3: Fn() -> I, {
    let (weight_sum, num_weight_steps) =
        kahan_sigma_return_counter(weight_iter(), |weight| weight.to_f64().unwrap());

    let (weighted_sum_a, num_a_steps) =
        kahan_sigma_return_counter(a_iter().zip(weight_iter()), |(val, weight)| {
            val.to_f64().unwrap() * weight.to_f64().unwrap()
        });
    let mean_a = weighted_sum_a / weight_sum;

    let (weighted_sum_b, num_b_steps) =
        kahan_sigma_return_counter(b_iter().zip(weight_iter()), |(val, weight)| {
            val.to_f64().unwrap() * weight.to_f64().unwrap()
        });
    let mean_b = weighted_sum_b / weight_sum;

    assert_eq!(
        num_a_steps, num_b_steps,
        "num_a_steps ({}) != num_b_steps ({})",
        num_a_steps, num_b_steps
    );
    assert_eq!(
        num_a_steps, num_weight_steps,
        "num_a_steps ({}) != num_weight_steps ({})",
        num_a_steps, num_weight_steps
    );

    let numerator = kahan_sigma(
        a_iter().zip(b_iter()).zip(weight_iter()),
        |((a, b), weight)| {
            let a_f64 = a.to_f64().unwrap();
            let b_f64 = b.to_f64().unwrap();
            (a_f64 - mean_a) * (b_f64 - mean_b) * weight.to_f64().unwrap()
        },
    );

    let sqrt_a = kahan_sigma(a_iter().zip(weight_iter()), |(val_a, weight)| {
        let diff = val_a.to_f64().unwrap() - mean_a;
        diff * diff * weight.to_f64().unwrap()
    })
    .sqrt();

    let sqrt_b = kahan_sigma(b_iter().zip(weight_iter()), |(val_b, weight)| {
        let diff = val_b.to_f64().unwrap() - mean_b;
        diff * diff * weight.to_f64().unwrap()
    })
    .sqrt();

    numerator / sqrt_a / sqrt_b
}

#[cfg(test)]
mod tests {
    use crate::stats::correlation::weighted_correlation;
    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_weighted_correlation() {
        let u1 = vec![1, 1, 0];
        let v1 = vec![0, 1, 0];
        let w1 = vec![1, 1, 1];
        let w2 = vec![1, 3, 1];

        let c1 = weighted_correlation(|| u1.iter(), || v1.iter(), || w1.iter());
        let c2 = weighted_correlation(|| u1.iter(), || v1.iter(), || w2.iter());
        assert!((c1 - 0.5).abs() < TOLERANCE);
        assert!((c2 - 0.61237243).abs() < TOLERANCE);

        let u2 = vec![2, -3, 5, 10];
        let v2 = vec![1, -2, 0, 5];
        let w3 = vec![1, 3, 5, 1];
        let c3 = weighted_correlation(|| u2.iter(), || v2.iter(), || w3.iter());
        assert!((c3 - 0.85208861).abs() < TOLERANCE);
    }
}
