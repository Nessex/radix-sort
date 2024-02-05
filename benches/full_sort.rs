#[cfg(not(bench))]
compile_error!("This binary must be run with `RUSTFLAGS='--cfg bench'`");

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rdst::utils::bench_utils::{bench_common, bench_medley};
use rdst::utils::test_utils::NumericTest;
use rdst::RadixSort;

fn full_sort_common<T>(c: &mut Criterion, shift: T, name_suffix: &str)
where
    T: NumericTest<T>,
{
    let tests: Vec<(&str, Box<dyn Fn(Vec<_>)>)> = vec![
        (
            "rdst",
            Box::new(|mut input| {
                input.radix_sort_unstable();
                black_box(input);
            }),
        ),
        (
            "rdst_low_mem",
            Box::new(|mut input| {
                input.radix_sort_builder().with_low_mem_tuner().sort();
                black_box(input);
            }),
        ),
    ];

    bench_common(c, shift, &("full_sort_".to_owned() + name_suffix), tests);
}

fn full_sort_medley_set<T>(c: &mut Criterion, suffix: &str, shift: T)
where
    T: NumericTest<T>,
{
    let tests: Vec<(&str, Box<dyn Fn(Vec<T>)>)> = vec![
        (
            "rdst",
            Box::new(|mut input| {
                input.radix_sort_unstable();
                black_box(input);
            }),
        ),
        (
            "rdst_low_mem",
            Box::new(|mut input| {
                input.radix_sort_builder().with_low_mem_tuner().sort();
                black_box(input);
            }),
        ),
    ];

    bench_medley(c, &("full_sort_medley_".to_owned() + suffix), tests, shift);
}

fn full_sort_u32(c: &mut Criterion) {
    full_sort_common(c, 0u32, "u32");
}

fn full_sort_u64(c: &mut Criterion) {
    full_sort_common(c, 0u64, "u64");
}

fn full_sort_u128(c: &mut Criterion) {
    full_sort_common(c, 0u128, "u128");
}

fn full_sort_u32_bimodal(c: &mut Criterion) {
    full_sort_common(c, 16u32, "u32_bimodal");
}

fn full_sort_u64_bimodal(c: &mut Criterion) {
    full_sort_common(c, 32u64, "u64_bimodal");
}

fn full_sort_u128_bimodal(c: &mut Criterion) {
    full_sort_common(c, 64u128, "u128_bimodal");
}

fn full_sort_medley(c: &mut Criterion) {
    full_sort_medley_set(c, "u32", 0u32);
    full_sort_medley_set(c, "u32_bimodal", 16u32);
    full_sort_medley_set(c, "u64", 0u64);
    full_sort_medley_set(c, "u64_bimodal", 32u64);
    full_sort_medley_set(c, "u128", 0u128);
    full_sort_medley_set(c, "u128_bimodal", 64u128);
}

criterion_group!(
    benches,
    full_sort_u32,
    full_sort_u64,
    full_sort_u128,
    full_sort_u32_bimodal,
    full_sort_u64_bimodal,
    full_sort_u128_bimodal,
    full_sort_medley,
);
criterion_main!(benches);
