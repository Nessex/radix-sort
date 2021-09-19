use crate::director::director;
use crate::tuning_parameters::TuningParameters;
use crate::utils::*;
use crate::RadixKey;
use arbitrary_chunks::ArbitraryChunks;
use partition::partition_index;
use rayon::prelude::*;
use std::cmp::min;
use try_mutex::TryMutex;

struct ScannerBucketInner<'a, T> {
    write_head: usize,
    read_head: usize,
    chunk: &'a mut [T],
    locally_partitioned: bool,
}

struct ScannerBucket<'a, T> {
    index: usize,
    len: isize,
    inner: TryMutex<ScannerBucketInner<'a, T>>,
}

#[inline]
fn get_scanner_buckets<'a, T>(
    counts: &[usize; 256],
    bucket: &'a mut [T],
) -> Vec<ScannerBucket<'a, T>> {
    let mut out: Vec<_> = bucket
        .arbitrary_chunks_mut(counts.to_vec())
        .enumerate()
        .map(|(index, chunk)| ScannerBucket {
            index,
            len: chunk.len() as isize,
            inner: TryMutex::new(ScannerBucketInner {
                write_head: 0,
                read_head: 0,
                chunk,
                locally_partitioned: false,
            }),
        })
        .collect();

    out.sort_by_key(|b| b.len);
    out.reverse();

    out
}

fn scanner_thread<T>(
    scanner_buckets: &Vec<ScannerBucket<T>>,
    level: usize,
    scanner_read_size: isize,
    uniform_threshold: usize,
) where
    T: RadixKey + Copy,
{
    let mut stash: Vec<Vec<T>> = Vec::with_capacity(256);
    stash.resize(256, Vec::with_capacity(128));
    let mut finished_count = 0;
    let mut finished_map = [false; 256];

    // Locally partition chunk into [correct bucket | incorrect bucket] in-place.
    // This provides a speed improvement when there are just a few larger outlier buckets as there
    // is less data to move around to temporary storage etc.
    // In the case of buckets not above the uniform_threshold, we can ignore them as the
    // partitioning adds unnecessary overhead in that case.
    for m in scanner_buckets {
        if (m.len as usize) < uniform_threshold {
            continue;
        }

        let mut guard = match m.inner.try_lock() {
            Some(g) => g,
            None => continue,
        };

        if !guard.locally_partitioned {
            guard.locally_partitioned = true;

            let index = m.index as u8;
            let start = partition_index(&mut guard.chunk, |v| v.get_level(level) == index);

            guard.read_head = start;
            guard.write_head = start;
        }
    }

    'outer: loop {
        for m in scanner_buckets {
            if finished_map[m.index] {
                continue;
            }

            let mut guard = match m.inner.try_lock() {
                Some(g) => g,
                None => continue,
            };

            if guard.write_head >= m.len as usize {
                finished_count += 1;
                finished_map[m.index] = true;

                if finished_count == scanner_buckets.len() {
                    break 'outer;
                }

                continue;
            }

            let read_start = guard.read_head as isize;
            let to_read = min(m.len - read_start, scanner_read_size);

            if to_read > 0 {
                let to_read = to_read as usize;
                let end = guard.read_head + to_read;
                let read_data = &guard.chunk[guard.read_head..end];
                let chunks = read_data.chunks_exact(8);
                let rem = chunks.remainder();

                chunks.into_iter().for_each(|chunk| {
                    let a = chunk[0].get_level(level) as usize;
                    let b = chunk[1].get_level(level) as usize;
                    let c = chunk[2].get_level(level) as usize;
                    let d = chunk[3].get_level(level) as usize;

                    stash[a].push(chunk[0]);
                    stash[b].push(chunk[1]);
                    stash[c].push(chunk[2]);
                    stash[d].push(chunk[3]);

                    let e = chunk[4].get_level(level) as usize;
                    let f = chunk[5].get_level(level) as usize;
                    let g = chunk[6].get_level(level) as usize;
                    let h = chunk[7].get_level(level) as usize;

                    stash[e].push(chunk[4]);
                    stash[f].push(chunk[5]);
                    stash[g].push(chunk[6]);
                    stash[h].push(chunk[7]);
                });

                rem.into_iter().for_each(|v| {
                    let a = v.get_level(level) as usize;
                    stash[a].push(*v);
                });

                guard.read_head += to_read;
            }

            let to_write = min(
                stash[m.index].len() as isize,
                guard.read_head as isize - guard.write_head as isize,
            );

            if to_write < 1 {
                continue;
            }

            let to_write = to_write as usize;
            let split = stash[m.index].len() - to_write;
            let some = stash[m.index].split_off(split);
            let end = guard.write_head + to_write;
            let start = guard.write_head;

            guard.chunk[start..end].copy_from_slice(&some);

            guard.write_head += to_write;

            if guard.write_head >= m.len as usize {
                finished_count += 1;
                finished_map[m.index] = true;

                if finished_count == scanner_buckets.len() {
                    break 'outer;
                }
            }
        }
    }
}

// scanning_radix_sort does a parallel MSB-first sort. Following this, depending on the number of
// elements remaining in each bucket, it will either do an MSB-sort or an LSB-sort, making this
// a dynamic hybrid sort.
pub fn scanning_radix_sort<T>(
    tuning: &TuningParameters,
    bucket: &mut [T],
    start_level: usize,
    parallel_count: bool,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    let (msb_counts, level) =
        if let Some(s) = get_counts_and_level_descending(bucket, start_level, 0, parallel_count) {
            s
        } else {
            return;
        };

    let len = bucket.len();
    let uniform_threshold = ((len / tuning.cpus) as f64 * 1.4) as usize;
    let scanner_buckets = get_scanner_buckets(&msb_counts, bucket);
    let threads = min(tuning.cpus, scanner_buckets.len());

    (0..threads).into_par_iter().for_each(|_| {
        scanner_thread(
            &scanner_buckets,
            level,
            tuning.scanner_read_size as isize,
            uniform_threshold,
        );
    });

    // Drop some data before recursing to reduce memory usage
    drop(scanner_buckets);

    if level == 0 {
        return;
    }

    let len_limit = ((len / tuning.cpus) as f64 * 1.4) as usize;
    let mut long_chunks = Vec::new();
    let mut average_chunks = Vec::with_capacity(256);

    for chunk in bucket.arbitrary_chunks_mut(msb_counts.to_vec()) {
        if chunk.len() > len_limit && chunk.len() > tuning.scanning_sort_threshold {
            long_chunks.push(chunk);
        } else {
            average_chunks.push(chunk);
        }
    }

    long_chunks
        .into_iter()
        .for_each(|chunk| scanning_radix_sort(tuning, chunk, level - 1, true));

    average_chunks
        .into_par_iter()
        .for_each(|chunk| director(tuning, chunk, len, level - 1));
}

#[cfg(test)]
mod tests {
    use crate::sorts::scanning_radix_sort::scanning_radix_sort;
    use crate::test_utils::{sort_comparison_suite, NumericTest};
    use crate::tuning_parameters::TuningParameters;

    fn test_scanning_sort<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        let tuning = TuningParameters::new(T::LEVELS);
        sort_comparison_suite(shift, |inputs| {
            scanning_radix_sort(&tuning, inputs, T::LEVELS - 1, false)
        });
    }

    #[test]
    pub fn test_u8() {
        test_scanning_sort(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_scanning_sort(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_scanning_sort(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_scanning_sort(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_scanning_sort(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_scanning_sort(32usize);
    }
}
