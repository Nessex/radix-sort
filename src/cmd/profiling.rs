/// NOTE: The primary use-case for this example is for running a large sort with cargo-instruments.
/// It must be run with `--features=tuning`.
///
/// e.g.
/// ```
/// RUSTFLAGS='--cfg bench --cfg tuning -g -C opt-level=3 -C force-frame-pointers=y -C target-cpu=apple-m1 -C target-feature=+neon' cargo +nightly instruments -t time --bin profiling --features profiling
/// ```

#[cfg(not(all(tuning, bench)))]
compile_error!("This binary must be run with `RUSTFLAGS='--cfg tuning --cfg bench'`");

use rdst::tuner::{Algorithm, Tuner, TuningParams};
use rdst::utils::test_utils::gen_inputs;
use rdst::RadixSort;
use std::thread::sleep;
use std::time::{Duration, Instant};

struct MyTuner {}

impl Tuner for MyTuner {
    fn pick_algorithm(&self, p: &TuningParams, _: &[usize]) -> Algorithm {
        if p.input_len < 128 {
            return Algorithm::Comparative;
        }

        let depth = p.total_levels - p.level - 1;
        match depth {
            0 => Algorithm::MtLsb,
            _ => Algorithm::Lsb,
        }
    }
}

fn main() {
    // Randomly generate an array of
    // 200_000_000 u64's with half shifted >> 32 and half shifted << 32
    let mut inputs = gen_inputs(200_000_000, 0u32);
    let mut inputs_2 = gen_inputs(200_000_000, 0u32);

    // Input generation is multi-threaded and hard to differentiate from the actual
    // sorting algorithm, depending on the profiler. This makes it more obvious.
    sleep(Duration::from_millis(300));

    inputs.radix_sort_builder().with_tuner(&MyTuner {}).sort();

    // A second run, for comparison
    sleep(Duration::from_millis(300));
    let time = Instant::now();
    inputs_2.radix_sort_builder().with_tuner(&MyTuner {}).sort();

    let e = time.elapsed().as_millis();
    println!("Elapsed: {}ms", e);

    // Ensure nothing gets optimized out
    println!("{:?} {:?}", &inputs[0], &inputs_2[0]);
}
