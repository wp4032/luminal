// Utilities for creating and initializing random tensors and buffers.

use rand::{rng, Rng};
use num_traits::{Float, FromPrimitive};

/// Generate a vector of uniformly distributed random values.
pub fn uniform_vec<T: Float + FromPrimitive>(size: usize, low: T, high: T) -> Vec<T> {
    let mut _rng = rng();
    (0..size).map(|_| {
        let f: f32 = _rng.random_range(low.to_f32().unwrap()..high.to_f32().unwrap());
        T::from_f32(f).unwrap()
    }).collect()
}

/// Generate a vector of normally distributed random values.
pub fn randn_vec<T: Float + FromPrimitive>(size: usize, mean: T, std: T) -> Vec<T> {
    let mut _rng = rng();
    (0..size).map(|_| {
        let u1: f32 = _rng.random_range(0.0..1.0);
        let u2: f32 = _rng.random_range(0.0..1.0);
        let sample: f32 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        T::from_f32(sample * std.to_f32().unwrap() + mean.to_f32().unwrap()).unwrap()
    }).collect()
}

/// Generate a vector of uniformly distributed random values.
pub fn uniform<T: Float + FromPrimitive>(shape: &[usize], low: T, high: T) -> Vec<T> {
    let size = shape.iter().product();
    uniform_vec(size, low, high)
}

/// Generate a vector of normally distributed random values.
pub fn randn<T: Float + FromPrimitive>(shape: &[usize], mean: T, std: T) -> Vec<T> {
    let size = shape.iter().product();
    randn_vec(size, mean, std)
}

/// Generate a vector of random values sampled from [-scale, scale) range.
pub fn uniform_vec_scale<T: Float + FromPrimitive>(size: usize, scale: T) -> Vec<T> {
    uniform_vec(size, -scale, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_vec() {
        let data = uniform_vec(10, -2.0, 2.0);
        assert_eq!(data.len(), 10);
        for &val in &data {
            assert!(val >= -2.0 && val < 2.0);
        }
    }

    #[test]
    fn test_randn_vec() {
        let data = randn_vec(1000, 0.0, 1.0);
        assert_eq!(data.len(), 1000);

        // Basic statistical checks (rough approximation)
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();

        // Should be roughly normally distributed with mean ~0 and std ~1
        assert!(mean.abs() < 0.1);
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_tensor_data_shapes() {
        let shape = [2, 3, 4];
        let data = uniform(&shape, -1.0, 1.0);
        let expected_size: usize = shape.iter().product();
        assert_eq!(data.len(), expected_size);

        let data_scaled = uniform_vec_scale(expected_size, 0.5);
        assert_eq!(data_scaled.len(), expected_size);
        for &val in &data_scaled {
            assert!(val >= -0.5 && val < 0.5);
        }
    }
}
