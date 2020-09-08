use crate::stats::{kahan_sigma, kahan_sigma_return_counter};
use num::ToPrimitive;

/// The weights and values are of type `T`.
pub fn weighted_correlation<T, V, I: Iterator<Item = T>, F1, F2>(
    get_iter: F1,
    get_a_b_weight: F2,
) -> f64
where
    V: Copy + ToPrimitive,
    F1: Fn() -> I,
    F2: Fn(T) -> (V, V, V), {
    let (weight_sum, num_weight_steps) =
        kahan_sigma_return_counter(get_iter(), |x| {
            get_a_b_weight(x).2.to_f64().unwrap()
        });

    let (weighted_sum_a, num_a_steps) =
        kahan_sigma_return_counter(get_iter(), |x| {
            let (a, _, w) = get_a_b_weight(x);
            a.to_f64().unwrap() * w.to_f64().unwrap()
        });
    let mean_a = weighted_sum_a / weight_sum;

    let (weighted_sum_b, num_b_steps) =
        kahan_sigma_return_counter(get_iter(), |x| {
            let (_, b, w) = get_a_b_weight(x);
            b.to_f64().unwrap() * w.to_f64().unwrap()
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

    let numerator = kahan_sigma(get_iter(), |x| {
        let (a, b, w) = get_a_b_weight(x);
        (a.to_f64().unwrap() - mean_a)
            * (b.to_f64().unwrap() - mean_b)
            * w.to_f64().unwrap()
    });

    let sqrt_a = kahan_sigma(get_iter(), |x| {
        let (a, _, w) = get_a_b_weight(x);
        let diff = a.to_f64().unwrap() - mean_a;
        diff * diff * w.to_f64().unwrap()
    })
    .sqrt();

    let sqrt_b = kahan_sigma(get_iter(), |x| {
        let (_, b, w) = get_a_b_weight(x);
        let diff = b.to_f64().unwrap() - mean_b;
        diff * diff * w.to_f64().unwrap()
    })
    .sqrt();

    numerator / sqrt_a / sqrt_b
}

#[cfg(test)]
mod tests {
    use crate::{
        iter::flat_zip::IntoFlatZipIter,
        stats::correlation::weighted_correlation,
    };

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_weighted_correlation() {
        let u1 = vec![1, 1, 0];
        let v1 = vec![0, 1, 0];
        let w1 = vec![1, 1, 1];
        let w2 = vec![1, 3, 1];

        let c1 = weighted_correlation(
            || u1.iter().flat_zip(v1.iter()).flat_zip(w1.iter()),
            |x| (*x[0], *x[1], *x[2]),
        );
        let c2 = weighted_correlation(
            || u1.iter().flat_zip(v1.iter()).flat_zip(w2.iter()),
            |x| (*x[0], *x[1], *x[2]),
        );
        assert!((c1 - 0.5).abs() < TOLERANCE);
        assert!((c2 - 0.61237243).abs() < TOLERANCE);

        let u2 = vec![2, -3, 5, 10];
        let v2 = vec![1, -2, 0, 5];
        let w3 = vec![1, 3, 5, 1];
        let c3 = weighted_correlation(
            || u2.iter().flat_zip(v2.iter()).flat_zip(w3.iter()),
            |x| (*x[0], *x[1], *x[2]),
        );
        assert!((c3 - 0.85208861).abs() < TOLERANCE);
    }
}
