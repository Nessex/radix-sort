use crate::tuner::{Algorithm, Tuner, TuningParams};
use crate::{RadixKey, RadixSort};
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

pub struct SingleAlgoTuner {
    pub(crate) algo: Algorithm,
}

impl Tuner for SingleAlgoTuner {
    #[inline]
    fn pick_algorithm(&self, _p: &TuningParams, _counts: &[usize]) -> Algorithm {
        self.algo
    }
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
    let n = 50_000_000;
    let half = n / 2;
    let inputs = gen_inputs(n, shift);

    // Middle values are used for the case where shift is provided
    let mut out = vec![
        vec![],
        inputs[..1].to_vec(),
        inputs[(half - 5)..(half + 5)].to_vec(),
        inputs[(half - 50)..(half + 50)].to_vec(),
        inputs[(half - 2_500)..(half + 2_500)].to_vec(),
        inputs[(half - 5_000)..(half + 5_000)].to_vec(),
        inputs[(half - 25_000)..(half + 25_000)].to_vec(),
        inputs[(half - 50_000)..(half + 50_000)].to_vec(),
        inputs[(half - 100_000)..(half + 100_000)].to_vec(),
        inputs[(half - 150_000)..(half + 150_000)].to_vec(),
        inputs[(half - 250_000)..(half + 250_000)].to_vec(),
        inputs[(half - 500_000)..(half + 500_000)].to_vec(),
        inputs[(half - 1_000_000)..(half + 1_000_000)].to_vec(),
        inputs[(half - 2_500_000)..(half + 2_500_000)].to_vec(),
        inputs[(half - 5_000_000)..(half + 5_000_000)].to_vec(),
        inputs[(half - 25_000_000)..(half + 25_000_000)].to_vec(),
        inputs,
    ];

    out.reverse();

    out
}

pub fn validate_sort<T, F>(mut inputs: Vec<T>, sort_fn: F)
where
    T: NumericTest<T> + Debug,
    F: Fn(&mut [T]),
{
    let mut inputs_clone = inputs.clone();

    sort_fn(&mut inputs);

    let mut eq = true;

    for i in inputs.windows(2) {
        if i[0] > i[1] {
            eq = false;
            break;
        }
    }

    if eq {
        return;
    }

    inputs_clone.sort_unstable();

    for i in 0..inputs.len() {
        let a = inputs[i];
        let b = inputs_clone[i];
        assert_eq!(
            a,
            b,
            "Mismatch at index {:?} vs. {:?}",
            inputs[i - 5..i + 5].to_vec(),
            inputs_clone[i - 5..i + 5].to_vec()
        );
    }
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

pub fn validate_u32_patterns<F>(sort_fn: F)
where
    F: Fn(&mut [u32]),
{
    let input_sets: Vec<Vec<u32>> = vec![
        vec![u32::MAX; 128],
        block_rand(128),
        block_rand(128_000),
        block_rand(4),
    ];

    for inputs in input_sets.iter() {
        validate_sort(inputs.clone(), &sort_fn);

        // Empty levels
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0xFFFF_FF00)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0xFFFF_00FF)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0xFF00_FFFF)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0x00FF_FFFF)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0x0000_FFFF)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0xFFFF_0000)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );

        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0b10000000000000000000000000000000)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0b00000000000000000000000000000001)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0b11111111111111111111111111111110)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0b01111111111111111111111111111111)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0b10101010101010101010101010101010)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        validate_sort(
            inputs
                .iter()
                .map(|v| *v & 0b01010101010101010101010101010101)
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
        // This is to test a heavily skewed distribution
        // Particularly for the logic in the standard tuner that can trigger Algorithm::LrLsb
        validate_sort(
            inputs
                .iter()
                .enumerate()
                .map(|(i, v)| match i {
                    0..=4 => (*v & 0xFFFF_FF00) | 0x77,
                    _ => *v & 0xFFFF_FF00,
                })
                .collect::<Vec<u32>>(),
            &sort_fn,
        );
    }
}

pub fn sort_single_algorithm<T>(count: usize, algo: Algorithm)
where
    T: RadixKey + Sized + Copy + Debug + PartialEq + Ord + Send + Sync,
{
    let mut input_set = block_rand::<T>(count);
    let mut input_set_expected = input_set.clone();
    input_set
        .radix_sort_builder()
        .with_tuner(&SingleAlgoTuner { algo })
        .sort();

    input_set_expected.sort_unstable();

    assert_eq!(input_set, input_set_expected);
}
