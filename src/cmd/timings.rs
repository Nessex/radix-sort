//! # timings
//!
//! This is used to run the sorting algorithm across a medley of inputs and output the results
//! as CSV. All numbers are in nanoseconds. It is intended to provide a quick way of getting a
//! performance cross-section of any given change to the algorithms in this crate.
//!
//! ## Usage
//!
//! You may need to tweak the command below for your own machine.
//!
//! ```
//! RUSTFLAGS='--cfg bench --cfg tuning -C opt-level=3 -C target-cpu=apple-m1 -C target-feature=+neon' cargo +nightly run --bin timings --features timings -- 1234 "Hello world"
//! ```
//!
//!  - `1234` is where you place the ID for your run. If you are just running a brief test this can be `N/A`, otherwise it should be something like a commit SHA that you can use to find the code for this run again.
//!  - `Hello world` is a description so you can be aware of what you were testing when running this experiment
//!  - `HEADERS=true` is an environment variable you can add to print a header row before the output

#![feature(string_remove_matches)]

#[cfg(not(all(tuning, bench)))]
compile_error!("This binary must be run with `RUSTFLAGS='--cfg tuning --cfg bench'`");

use rdst::utils::bench_utils::gen_bench_exponential_input_set;
use rdst::{RadixKey, RadixSort};
use std::time::Instant;
#[cfg(all(tuning, not(target_env = "msvc")))]
use tikv_jemallocator::Jemalloc;

#[cfg(all(tuning, not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

fn print_row(data: Vec<String>) {
    let mut first = true;

    for mut o in data {
        if !first {
            print!(",");
        } else {
            first = false;
        }

        o.remove_matches("\"");

        if o.contains(" ") {
            print!("\"{}\"", o);
        } else {
            print!("{}", o);
        }
    }

    print!("\n");
}

fn bench<T>(inputs: Vec<Vec<T>>, name: &str, results: &mut Vec<String>, headers: &mut Vec<String>)
where
    T: RadixKey + Clone + Copy + Send + Sync,
{
    for i in inputs {
        if i.len() == 0 {
            continue;
        }
        headers.push(format!("{}_{}", name, i.len()).to_string());
        // Warmup
        let mut warmup = i.clone();
        warmup.radix_sort_unstable();
        drop(warmup);

        let mut scores = Vec::new();
        let n = 5usize;
        for _ in 0..n {
            let mut to_sort = i.clone();
            let time = Instant::now();
            to_sort.radix_sort_unstable();
            let elapsed = time.elapsed().as_nanos();
            let per_sec = ((i.len() as f64 / elapsed as f64) * 1_000_000_000f64) as u64;
            scores.push(per_sec);
            drop(to_sort);
        }

        scores.sort_unstable();
        let items_per_sec = scores[2];

        results.push(items_per_sec.to_string());
    }
}

fn main() {
    let print_headers: bool = std::env::var("HEADERS")
        .unwrap_or_else(|_| "false".to_string())
        .parse()
        .unwrap();
    let mut out: Vec<String> = std::env::args().skip(1).take(2).collect();
    assert_eq!(out.len(), 2);
    let mut headers = vec!["id".to_string(), "description".to_string()];

    let inputs = gen_bench_exponential_input_set(0u32);
    bench(inputs, "u32", &mut out, &mut headers);

    let inputs = gen_bench_exponential_input_set(16u32);
    bench(inputs, "u32_bimodal", &mut out, &mut headers);

    let inputs = gen_bench_exponential_input_set(0u64);
    bench(inputs, "u64", &mut out, &mut headers);

    let inputs = gen_bench_exponential_input_set(32u64);
    bench(inputs, "u64_bimodal", &mut out, &mut headers);

    let inputs = gen_bench_exponential_input_set(0u128);
    bench(inputs, "u128", &mut out, &mut headers);

    let inputs = gen_bench_exponential_input_set(64u128);
    bench(inputs, "u128_bimodal", &mut out, &mut headers);

    if print_headers {
        print_row(headers);
    }

    print_row(out);
}
