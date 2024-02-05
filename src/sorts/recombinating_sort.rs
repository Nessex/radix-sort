//! `recombinating_sort` is a multi-threaded, out-of-place, unstable radix sort unique to rdst. It
//! operates on a set of tiles, which are sub-sections of the original data of roughly the same size.
//!
//! It works by:
//!  1. Sorting each tile out-of-place into a temp array
//!  2. Calculating prefix sums of each tile
//!  3. Splitting the output array based upon the aggregated counts of all tiles
//!  4. Writing out the final data for each global count ("country" in regions sort terminology) in parallel
//!
//! Because each thread operates on separate tiles, and then separate output buckets, this is parallel from start to finish.
//! The intermediate tiles mean this requires 2n memory relative to the input, plus some memory for each set of counts, and incurs two copies for each item.
//!
//! ## Characteristics
//!
//!  * out-of-place
//!  * multi-threaded
//!  * unstable
//!
//! ## Performance
//!
//! This is typically the best performing multi-threaded sorting algorithm until you hit memory
//! constraints. As this is an out-of-place algorithm, you need 2n memory relative to the input for
//! this sort, and eventually the extra allocation and freeing required eats away at the performance.

use crate::counts::{CountManager, Counts};
use crate::sorter::Sorter;
use crate::sorts::out_of_place_sort::out_of_place_sort;
use crate::RadixKey;
use arbitrary_chunks::ArbitraryChunks;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub fn recombinating_sort<T>(
    cm: &CountManager,
    bucket: &mut [T],
    counts: &Counts,
    tile_counts: Vec<Counts>,
    tile_size: usize,
    level: usize,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    cm.with_tmp_buffer(bucket, |cm, bucket, tmp_bucket| {
        bucket
            .par_chunks(tile_size)
            .zip(tmp_bucket.par_chunks_mut(tile_size))
            .zip(tile_counts.par_iter())
            .for_each(|((chunk, tmp_chunk), counts)| {
                let sums = cm.prefix_sums(counts);
                out_of_place_sort(chunk, tmp_chunk, level, &mut sums.borrow_mut());
                cm.return_counts(sums);
            });

        bucket
            .arbitrary_chunks_mut(counts.inner())
            .enumerate()
            .par_bridge()
            .for_each(|(index, global_chunk)| {
                let mut read_offset = 0;
                let mut write_offset = 0;

                for tile_c in tile_counts.iter() {
                    let sum = if index == 0 {
                        0
                    } else {
                        tile_c.into_iter().take(index).sum::<usize>()
                    };
                    let read_start = read_offset + sum;
                    let read_end = read_start + tile_c[index];
                    let read_slice = &tmp_bucket[read_start..read_end];
                    let write_end = write_offset + read_slice.len();

                    global_chunk[write_offset..write_end].copy_from_slice(read_slice);

                    read_offset += tile_size;
                    write_offset = write_end;
                }
            });
    });
}

impl<'a> Sorter<'a> {
    pub(crate) fn recombinating_sort_adapter<T>(
        &self,
        bucket: &mut [T],
        counts: Rc<RefCell<Counts>>,
        tile_counts: Vec<Counts>,
        tile_size: usize,
        level: usize,
    ) where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if bucket.len() < 2 {
            return;
        }

        recombinating_sort(
            &self.cm,
            bucket,
            &counts.borrow(),
            tile_counts,
            tile_size,
            level,
        );

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
    use crate::utils::{aggregate_tile_counts, cdiv, get_tile_counts};
    use crate::RadixKey;
    use rayon::current_num_threads;

    fn test_recombinating_sort<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| {
            let level = T::LEVELS - 1;
            let tile_size = cdiv(inputs.len(), current_num_threads());

            if inputs.len() == 0 {
                return;
            }

            let cm = CountManager::default();
            let sorter = Sorter::new(true, &StandardTuner);

            let (tile_counts, _) = get_tile_counts(&cm, inputs, tile_size, level);
            let counts = aggregate_tile_counts(&cm, &tile_counts);

            sorter.recombinating_sort_adapter(inputs, counts, tile_counts, tile_size, T::LEVELS - 1)
        });
    }

    #[test]
    pub fn test_u8() {
        test_recombinating_sort(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_recombinating_sort(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_recombinating_sort(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_recombinating_sort(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_recombinating_sort(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_recombinating_sort(32usize);
    }

    #[test]
    pub fn test_basic_integration() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::Recombinating);
    }

    #[test]
    pub fn test_u32_patterns() {
        validate_u32_patterns(|inputs| {
            let level = u32::LEVELS - 1;
            let tile_size = cdiv(inputs.len(), current_num_threads());

            if inputs.len() == 0 {
                return;
            }

            let cm = CountManager::default();
            let sorter = Sorter::new(true, &StandardTuner);

            let (tile_counts, _) = get_tile_counts(&cm, inputs, tile_size, level);
            let counts = aggregate_tile_counts(&cm, &tile_counts);

            sorter.recombinating_sort_adapter(inputs, counts, tile_counts, tile_size, level)
        });
    }
}
