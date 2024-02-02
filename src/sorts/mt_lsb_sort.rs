//! `mt_lsb_sort` is a multi-threaded Least-Significant Bit first radix sort. Multi-threading
//! is achieved by splitting the data into tiles, counting those tiles independently and
//! using the aggregated prefix sums to generate offsets in the output array for each thread
//! to write to for each radix.
//!
//! The output array is split by tile-counts sorted by radix and those output array chunks are
//! distributed to each thread. As they are distributed in the order in which they originally
//! appeared, the output remains stable just like a typical single-threaded LSB sort.
//!
//! ## Characteristics
//!
//!  * out-of-place
//!  * multi-threaded
//!  * stable
//!  * lsb-first
//!
//! ## Performance
//!
//! While this does not provide the best performance overall, the performance of this algorithm
//! is extremely stable and predictable. It gracefully degrades into a single-threaded LSB radix sort
//! when the tiles are reduced to 1 with only a little overhead.
//!
//! ## Optimizations
//!
//! This shares pretty much all optimizations implemented for LsbSort.
//!
//! ## `mt_oop_sort` Variant
//!
//! This variant uses the same algorithm as `mt_lsb_sort` but uses it in msb-first order.

use crate::counts::Counts;
use crate::sorter::Sorter;
use crate::utils::*;
use crate::RadixKey;
use arbitrary_chunks::ArbitraryChunks;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub fn mt_lsb_sort<T>(
    src_bucket: &[T],
    dst_bucket: &mut [T],
    tile_counts: &[Counts],
    tile_size: usize,
    level: usize,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    let tiles = tile_counts.len();
    let mut minor_counts = Vec::with_capacity(256 * tiles);

    for b in 0..256 {
        for tile in tile_counts.iter() {
            minor_counts.push(tile[b]);
        }
    }

    let mut chunks: Vec<&mut [T]> = dst_bucket.arbitrary_chunks_mut(&minor_counts).collect();
    chunks.reverse();

    let mut collated_chunks: Vec<Vec<&mut [T]>> = Vec::with_capacity(tiles);
    collated_chunks.resize_with(tiles, Vec::new);

    for _ in 0..256 {
        for coll_chunk in collated_chunks.iter_mut().take(tiles) {
            coll_chunk.push(chunks.pop().unwrap());
        }
    }

    collated_chunks
        .into_par_iter()
        .zip(src_bucket.par_chunks(tile_size))
        .for_each(|(mut buckets, bucket)| {
            if bucket.is_empty() {
                return;
            }

            let mut offsets = [0usize; 256];
            let mut ends = [0usize; 256];

            for (i, b) in buckets.iter().enumerate() {
                if b.is_empty() {
                    continue;
                }

                ends[i] = b.len() - 1;
            }

            let mut left = 0;
            let mut right = bucket.len() - 1;
            let pre = bucket.len() % 8;

            for _ in 0..pre {
                let b = bucket[right].get_level(level) as usize;

                buckets[b][ends[b]] = bucket[right];
                ends[b] = ends[b].wrapping_sub(1);
                right = right.saturating_sub(1);
            }

            if pre == bucket.len() {
                return;
            }

            let end = (bucket.len() - pre) / 2;

            while left < end {
                let bl_0 = bucket[left].get_level(level) as usize;
                let bl_1 = bucket[left + 1].get_level(level) as usize;
                let bl_2 = bucket[left + 2].get_level(level) as usize;
                let bl_3 = bucket[left + 3].get_level(level) as usize;
                let br_0 = bucket[right].get_level(level) as usize;
                let br_1 = bucket[right - 1].get_level(level) as usize;
                let br_2 = bucket[right - 2].get_level(level) as usize;
                let br_3 = bucket[right - 3].get_level(level) as usize;

                buckets[bl_0][offsets[bl_0]] = bucket[left];
                offsets[bl_0] += 1;
                buckets[br_0][ends[br_0]] = bucket[right];
                ends[br_0] = ends[br_0].wrapping_sub(1);
                buckets[bl_1][offsets[bl_1]] = bucket[left + 1];
                offsets[bl_1] += 1;
                buckets[br_1][ends[br_1]] = bucket[right - 1];
                ends[br_1] = ends[br_1].wrapping_sub(1);
                buckets[bl_2][offsets[bl_2]] = bucket[left + 2];
                offsets[bl_2] += 1;
                buckets[br_2][ends[br_2]] = bucket[right - 2];
                ends[br_2] = ends[br_2].wrapping_sub(1);
                buckets[bl_3][offsets[bl_3]] = bucket[left + 3];
                offsets[bl_3] += 1;
                buckets[br_3][ends[br_3]] = bucket[right - 3];
                ends[br_3] = ends[br_3].wrapping_sub(1);

                left += 4;
                right = right.wrapping_sub(4);
            }
        });
}

impl<'a> Sorter<'a> {
    pub(crate) fn mt_lsb_sort_adapter<T>(
        &self,
        bucket: &mut [T],
        start_level: usize,
        end_level: usize,
        tile_size: usize,
    ) where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if bucket.len() < 2 {
            return;
        }

        self.cm.with_tmp_buffer(bucket.len(), |_, tmp_bucket| {
            let levels: Vec<usize> = (start_level..=end_level).collect();
            let mut invert = false;

            for level in levels {
                let (tile_counts, already_sorted) = if invert {
                    get_tile_counts(&self.cm, tmp_bucket, tile_size, level)
                } else {
                    get_tile_counts(&self.cm, bucket, tile_size, level)
                };

                if already_sorted {
                    continue;
                }

                if invert {
                    mt_lsb_sort(tmp_bucket, bucket, &tile_counts, tile_size, level)
                } else {
                    mt_lsb_sort(bucket, tmp_bucket, &tile_counts, tile_size, level)
                };

                invert = !invert;
            }

            if invert {
                bucket
                    .par_chunks_mut(tile_size)
                    .zip(tmp_bucket.par_chunks(tile_size))
                    .for_each(|(chunk, tmp_chunk)| {
                        chunk.copy_from_slice(tmp_chunk);
                    });
            }
        });
    }

    pub(crate) fn mt_oop_sort_adapter<T>(
        &self,
        bucket: &mut [T],
        level: usize,
        counts: Rc<RefCell<Counts>>,
        tile_counts: Vec<Counts>,
        tile_size: usize,
    ) where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if bucket.len() <= 1 {
            return;
        }

        self.cm.with_tmp_buffer(bucket.len(), |_, tmp_bucket| {
            mt_lsb_sort(bucket, tmp_bucket, &tile_counts, tile_size, level);

            bucket
                .par_chunks_mut(tile_size)
                .zip(tmp_bucket.par_chunks(tile_size))
                .for_each(|(chunk, tmp_chunk)| {
                    chunk.copy_from_slice(tmp_chunk);
                });
        });

        self.director(bucket, counts, level - 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::sorter::Sorter;
    use crate::tuner::Algorithm;
    use crate::tuners::StandardTuner;
    use crate::utils::cdiv;
    use crate::utils::test_utils::{
        sort_comparison_suite, sort_single_algorithm, validate_u32_patterns, NumericTest,
    };
    use crate::RadixKey;
    use rayon::current_num_threads;

    fn test_mt_lsb_sort_adapter<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| {
            if inputs.len() == 0 {
                return;
            }

            let sorter = Sorter::new(true, &StandardTuner);
            let tile_size = cdiv(inputs.len(), current_num_threads());

            sorter.mt_lsb_sort_adapter(inputs, 0, T::LEVELS - 1, tile_size);
        });
    }

    #[test]
    pub fn test_u8() {
        test_mt_lsb_sort_adapter(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_mt_lsb_sort_adapter(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_mt_lsb_sort_adapter(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_mt_lsb_sort_adapter(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_mt_lsb_sort_adapter(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_mt_lsb_sort_adapter(32usize);
    }

    #[test]
    pub fn test_basic_integration() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::MtLsb);
    }

    #[test]
    pub fn test_regression_issue_5() {
        // Replicates https://github.com/Nessex/rdst/issues/5
        // MtLsb returns unsorted data when there is only 1 tile
        sort_single_algorithm::<u32>(400, Algorithm::MtLsb);
    }

    #[test]
    pub fn test_u32_patterns() {
        validate_u32_patterns(|inputs| {
            if inputs.len() == 0 {
                return;
            }

            let sorter = Sorter::new(true, &StandardTuner);
            let tile_size = cdiv(inputs.len(), current_num_threads());

            sorter.mt_lsb_sort_adapter(inputs, 0, u32::LEVELS - 1, tile_size);
        });
    }
}
