//! `scanning_sort` is a custom algorithm for rdst. It is a multi-threaded, MSB first radix sort.
//!
//! Scanning sort works by:
//!
//!  1. Chunk the input array into buckets based on the counts for this level
//!  2. Create a worker for each rayon global thread pool thread (roughly, one per core)
//!  2. Create a temporary thread-local buffer for each worker (one vec for each radix)
//!  3. Each thread:
//!  3.1. Iterates over the buckets, trying to gain a mutex lock on one
//!  3.2. On first lock of the bucket, it partitions the bucket into [correct data | incorrect data] in-place
//!  3.3. Scan over the contents of the bucket, picking up data that shouldn't be there and putting it in the thread-local buffer
//!  3.4. Writes any buffered contents that _should_ be in this bucket, into the bucket
//!  3.5. Repeats 3 until all buckets are completely filled with the correct data
//!
//! Along the way, each output bucket has a read head and a write head, which is a pointer to the latest content read and written respectively.
//! When the read head reaches the end of the bucket, there is no more content to be buffered by any worker.
//! When the write head reaches the end of the bucket, that bucket contains all data that should be there, and is marked completed.
//! Once there are no more buckets that can be locked by the worker (all remaining buckets are locked), each worker exits.
//! Once all buckets are completed, and all workers have exited, the sort is finished.
//!
//! Thread-local buffers can hold up to 128 values for each radix, or 32,768 values in total. There's one per thread, so the total amount of memory can add up to quite a lot.
//! 128 values was chosen based upon performance numbers from benchmarking, and is not currently configurable.
//!
//! ## Characteristics
//!
//!  * out-of-place
//!  * multi-threaded
//!  * unstable
//!
//! ## Performance
//!
//! For large inputs, this is the fastest multi-threaded sorting algorithm in rdst. The additional
//! overhead of the thread-local stores and mutexes prevents it from being fast for smaller inputs
//! however, so it should not be used in all situations.

use crate::counts::{Counts, PrefixSums};
use crate::sorter::Sorter;
use crate::RadixKey;
use arbitrary_chunks::ArbitraryChunks;
use partition::partition_index;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::{max, min};
use std::rc::Rc;
use std::sync::Mutex;

struct ScannerBucketInner<'a, T> {
    write_head: usize,
    read_head: usize,
    chunk: &'a mut [T],
    locally_partitioned: bool,
}

struct ScannerBucket<'a, T> {
    index: usize,
    len: isize,
    inner: Mutex<ScannerBucketInner<'a, T>>,
}

#[inline]
fn get_scanner_buckets<'a, T>(
    counts: &Counts,
    prefix_sums: &PrefixSums,
    bucket: &'a mut [T],
) -> Vec<ScannerBucket<'a, T>> {
    let mut running_count = 0;
    let mut out: Vec<_> = bucket
        .arbitrary_chunks_mut(counts.inner())
        .enumerate()
        .map(|(index, chunk)| {
            let head = prefix_sums[index] - running_count;
            running_count += chunk.len();

            ScannerBucket {
                index,
                len: chunk.len() as isize,
                inner: Mutex::new(ScannerBucketInner {
                    write_head: head,
                    read_head: head,
                    chunk,
                    locally_partitioned: false,
                }),
            }
        })
        .collect();

    out.sort_by_key(|b| b.len);
    out.reverse();

    out
}

fn scanner_thread<T>(
    scanner_buckets: &[ScannerBucket<T>],
    level: usize,
    scanner_read_size: isize,
    uniform_threshold: usize,
) where
    T: RadixKey + Copy,
{
    let mut stash: Vec<Vec<T>> = vec![Vec::with_capacity(128); 256];
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
            Ok(g) => g,
            Err(_) => continue,
        };

        if !guard.locally_partitioned {
            guard.locally_partitioned = true;

            let index = m.index as u8;
            let start = partition_index(guard.chunk, |v| v.get_level(level) == index);

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
                Ok(g) => g,
                Err(_) => continue,
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

                rem.iter().for_each(|v| {
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
            let end = guard.write_head + to_write;
            let start = guard.write_head;
            guard.chunk[start..end].copy_from_slice(&stash[m.index][split..]);
            stash[m.index].truncate(split);

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

pub fn scanning_sort<T>(bucket: &mut [T], counts: &Counts, prefix_sums: &PrefixSums, level: usize)
where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    let len = bucket.len();
    let threads = current_num_threads();
    let uniform_threshold = ((len / threads) as f64 * 1.4) as usize;
    let scanner_buckets = get_scanner_buckets(counts, prefix_sums, bucket);
    let threads = min(threads, scanner_buckets.len());
    let scaling_factor = max(1, (threads as f32).log2().ceil() as isize) as usize;
    let scanner_read_size = (32768 / scaling_factor) as isize;

    (0..threads).into_par_iter().for_each(|_| {
        scanner_thread(
            &scanner_buckets,
            level,
            scanner_read_size,
            uniform_threshold,
        );
    });
}

impl<'a> Sorter<'a> {
    // scanning_radix_sort does a parallel MSB-first sort. Following this, depending on the number of
    // elements remaining in each bucket, it will either do an MSB-sort or an LSB-sort, making this
    // a dynamic hybrid sort.
    pub(crate) fn scanning_sort_adapter<T>(
        &self,
        bucket: &mut [T],
        counts: Rc<RefCell<Counts>>,
        level: usize,
    ) where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if bucket.len() < 2 {
            return;
        }

        let prefix_sums = self.cm.prefix_sums(&counts.borrow());
        scanning_sort(bucket, &counts.borrow(), &prefix_sums.borrow(), level);
        self.cm.return_counts(prefix_sums);

        if level == 0 {
            return;
        }

        self.director(bucket, counts, level - 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::counts::CountManager;
    use crate::sorter::Sorter;
    use crate::tuner::Algorithm;
    use crate::tuners::StandardTuner;
    use crate::utils::test_utils::{
        sort_comparison_suite, sort_single_algorithm, validate_u32_patterns, NumericTest,
    };
    use crate::RadixKey;

    fn test_scanning_sort<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| {
            let cm = CountManager::default();
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = cm.counts(inputs, T::LEVELS - 1);

            sorter.scanning_sort_adapter(inputs, counts, T::LEVELS - 1)
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

    #[test]
    pub fn test_basic_integration() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::Scanning);
    }

    #[test]
    pub fn test_u32_patterns() {
        validate_u32_patterns(|inputs| {
            let cm = CountManager::default();
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = cm.counts(inputs, u32::LEVELS - 1);

            sorter.scanning_sort_adapter(inputs, counts, u32::LEVELS - 1)
        });
    }
}
