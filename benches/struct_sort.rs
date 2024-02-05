use block_pseudorand::block_rand;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rdst::{RadixKey, RadixSort};
use std::cmp::Ordering;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct LargeStruct {
    pub sort_key: f32,
    pub a: (u32, u32),
    pub b: usize,
    pub c: usize,
    pub d: Option<(u32, u32)>,
}

impl RadixKey for LargeStruct {
    const LEVELS: usize = 4;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        self.sort_key.get_level(level)
    }
}

impl PartialOrd for LargeStruct {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.sort_key.partial_cmp(&other.sort_key)
    }
}

impl PartialEq for LargeStruct {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.sort_key == other.sort_key
    }
}

fn gen_input_t2d(n: usize) -> Vec<LargeStruct> {
    let mut data: Vec<f32> = block_rand((n / 10) * 9);
    data.radix_sort_unstable();

    let mut data_2: Vec<f32> = block_rand(n / 10);
    data.append(&mut data_2);

    data.into_iter()
        .map(|v| LargeStruct {
            sort_key: v,
            a: (0, 0),
            b: 0,
            c: 0,
            d: None,
        })
        .collect()
}

fn gen_input_sorted(n: usize) -> Vec<LargeStruct> {
    let mut data: Vec<f32> = block_rand(n);
    data.radix_sort_unstable();

    data.into_iter()
        .map(|v| LargeStruct {
            sort_key: v,
            a: (0, 0),
            b: 0,
            c: 0,
            d: None,
        })
        .collect()
}

fn full_sort_struct(c: &mut Criterion) {
    let mut input_sets: Vec<(&str, Vec<LargeStruct>)> = vec![
        ("160k random", gen_input_t2d(160_000)),
        ("409k random", gen_input_t2d(409_600)),
        ("160k already sorted", gen_input_sorted(160_000)),
        ("409k already sorted", gen_input_sorted(409_600)),
    ];

    let tests: Vec<(&str, Box<dyn Fn(Vec<LargeStruct>)>)> = vec![
        (
            "rdst",
            Box::new(|mut input| {
                input
                    .radix_sort_builder()
                    .with_single_threaded_tuner()
                    .sort();
                black_box(input);
            }),
        ),
        (
            "sort",
            Box::new(|mut input| {
                input.sort_unstable_by_key(|v| {
                    let s = v.sort_key.to_bits();

                    if s >> 31 == 1 {
                        !s
                    } else {
                        s ^ (1 << 31)
                    }
                });
                black_box(input);
            }),
        ),
    ];

    let mut group = c.benchmark_group("struct_sort");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for set in input_sets.iter_mut() {
        let l = set.1.len();
        group.throughput(Throughput::Elements(l as u64));

        for t in tests.iter() {
            let name = format!("{} {}", set.0, t.0);

            group.bench_with_input(BenchmarkId::new(name, l), set, |bench, set| {
                bench.iter_batched(|| set.1.clone(), &*t.1, BatchSize::SmallInput);
            });
        }
    }

    group.finish();
}

criterion_group!(struct_sort, full_sort_struct,);
criterion_main!(struct_sort);
