use crate::RadixKey;
use block_pseudorand::block_rand;
use rayon::prelude::*;
use std::fmt::Debug;
use std::ops::{Shl, ShlAssign, Shr, ShrAssign};

pub trait NumericTest<T>:
    RadixKey
    + Sized
    + Copy
    + Debug
    + PartialEq
    + Ord
    + Send
    + Sync
    + Shl<Output = T>
    + Shr<Output = T>
    + ShrAssign
    + ShlAssign
{
}

impl<T> NumericTest<T> for T where
    T: RadixKey
        + Sized
        + Copy
        + Debug
        + PartialEq
        + Ord
        + Send
        + Sync
        + Shl<Output = T>
        + Shr<Output = T>
        + ShrAssign
        + ShlAssign
{
}

pub fn gen_inputs<T>(n: usize, shift: T) -> Vec<T>
where
    T: NumericTest<T>,
{
    let mut inputs: Vec<T> = block_rand(n);

    inputs[0..(n / 2)].par_iter_mut().for_each(|v| *v >>= shift);
    inputs[(n / 2)..n].par_iter_mut().for_each(|v| *v <<= shift);

    inputs
}

pub fn gen_input_set<T>(shift: T) -> Vec<Vec<T>>
where
    T: NumericTest<T>,
{
    let inputs = gen_inputs(50_000_000, shift);

    // Middle values are used for the case where shift is provided
    vec![inputs.clone()]
}

pub fn gen_bench_input_set<T>(shift: T) -> Vec<Vec<T>>
where
    T: NumericTest<T>,
{
    let inputs = gen_inputs(200_000_000, shift);

    // Middle values are used for the case where shift is provided
    vec![
        inputs.clone(),
        inputs[50_000_000..150_000_000].to_vec(),
        inputs[75_000_000..125_000_000].to_vec(),
        inputs[95_000_000..105_000_000].to_vec(),
        inputs[97_500_000..102_500_000].to_vec(),
        inputs[99_000_000..101_000_000].to_vec(),
        inputs[99_500_000..100_500_000].to_vec(),
        inputs[99_750_000..100_250_000].to_vec(),
        inputs[99_850_000..100_150_000].to_vec(),
        inputs[99_900_000..100_100_000].to_vec(),
        inputs[99_950_000..100_050_000].to_vec(),
        inputs[99_975_000..100_025_000].to_vec(),
        inputs[99_995_000..100_005_000].to_vec(),
        inputs[99_997_500..100_002_500].to_vec(),
    ]
}

pub fn validate_sort<T, F>(mut inputs: Vec<T>, sort_fn: F)
where
    T: NumericTest<T>,
    F: Fn(&mut [T]),
{
    let mut inputs_clone = inputs.clone();

    sort_fn(&mut inputs);
    inputs_clone.sort_unstable();

    assert_eq!(inputs, inputs_clone);
}

pub fn sort_comparison_suite<T, F>(shift: T, sort_fn: F)
where
    F: Fn(&mut [T]),
    T: NumericTest<T>,
{
    let input_set = gen_input_set(shift);

    for s in input_set {
        validate_sort(s, &sort_fn);
    }
}
