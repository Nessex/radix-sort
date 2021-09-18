/// NOTE: The primary use-case for this example is for running a large sort with cargo-instruments.
/// It must be run with `--features=bench`.
///
/// e.g.
/// ```
/// RUSTFLAGS='-g -C opt-level=3 -C force-frame-pointers=y -C target-cpu=native -C target-feature=+neon' cargo +nightly instruments -t time --example profiling --features=bench
/// ```
use rdst::test_utils::gen_inputs;
use rdst::RadixSort;

fn main() {
    // Randomly generate an array of
    // 200_000_000 u64's with half shifted >> 32 and half shifted << 32
    let mut inputs = gen_inputs(200_000_000, 32u64);

    inputs.radix_sort_unstable();
    println!("{:?}", &inputs[0..5]);
}