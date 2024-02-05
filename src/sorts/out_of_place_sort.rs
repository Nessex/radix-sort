//! `out_of_place_sort` is an out-of-place single-threaded radix sort. This is the classic academic
//! implementation of counting sort. There are 4 different variants implemented here with varying
//! optimizations.
//!
//! This is used as a building block for other complete sorting algorithms.
//!
//! ### Standard out_of_place_sort
//!
//! This implementation is a very simple out-of-place counting sort. The only notable optimization
//! is to process data in chunks to take some advantage of multiple execution ports in each CPU core.
//!
//! ### out_of_place_sort_with_counts
//!
//! As the name suggests, this variant is the same as the standard out_of_place_sort except that
//! as it sorts into the output array, it also checks the next level and adds it to a counts array.
//!
//! This shortcut shaves off a tiny bit of time that would be spent counting the next level before
//! sorting. It doesn't make a huge difference as you impair caching and similar that would
//! otherwise perform better in both the sort and the next counting pass. That said, it is
//! significant enough to still include as an option.
//!
//! ### lr_out_of_place_sort
//!
//! This variant of the standard out_of_place_sort uses two sets of cursors, one left and one right
//! cursor for writing data. This is able to remain stable as it inspects the input array starting
//! from the right for all values placed to the right side of each output bucket. Thus maintaining
//! the stable ordering of values.
//!
//! This provides a significant performance benefit when there are many identical values as
//! typically a pair of identical values would prevent the CPU from using multiple execution ports.
//! With this variant however, the CPU can safely and independently work on two identical values at the
//! same time as there is no overlapping variable access in either the output array or the prefix
//! sums array.
//!
//! ### lr_out_of_place_sort_with_counts
//!
//! As with the other with_counts variant, this combines the left-right optimization with counting
//! the next level.
//!
//! ## Characteristics
//!
//!  * out-of-place
//!  * single-threaded
//!  * lsb-first

use crate::counts::{Counts, EndOffsets, PrefixSums};
use crate::RadixKey;

#[inline]
pub fn out_of_place_sort<T>(
    src_bucket: &[T],
    dst_bucket: &mut [T],
    level: usize,
    prefix_sums: &mut PrefixSums,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    if src_bucket.len() < 2 {
        dst_bucket.copy_from_slice(src_bucket);
        return;
    }

    let chunks = src_bucket.chunks_exact(8);
    let rem = chunks.remainder();

    chunks.into_iter().for_each(|chunk| {
        let a = chunk[0].get_level(level) as usize;
        let b = chunk[1].get_level(level) as usize;
        let c = chunk[2].get_level(level) as usize;
        let d = chunk[3].get_level(level) as usize;
        let e = chunk[4].get_level(level) as usize;
        let f = chunk[5].get_level(level) as usize;
        let g = chunk[6].get_level(level) as usize;
        let h = chunk[7].get_level(level) as usize;

        dst_bucket[prefix_sums[a]] = chunk[0];
        prefix_sums[a] += 1;
        dst_bucket[prefix_sums[b]] = chunk[1];
        prefix_sums[b] += 1;
        dst_bucket[prefix_sums[c]] = chunk[2];
        prefix_sums[c] += 1;
        dst_bucket[prefix_sums[d]] = chunk[3];
        prefix_sums[d] += 1;
        dst_bucket[prefix_sums[e]] = chunk[4];
        prefix_sums[e] += 1;
        dst_bucket[prefix_sums[f]] = chunk[5];
        prefix_sums[f] += 1;
        dst_bucket[prefix_sums[g]] = chunk[6];
        prefix_sums[g] += 1;
        dst_bucket[prefix_sums[h]] = chunk[7];
        prefix_sums[h] += 1;
    });

    rem.iter().for_each(|val| {
        let b = val.get_level(level) as usize;
        dst_bucket[prefix_sums[b]] = *val;
        prefix_sums[b] += 1;
    });
}

#[inline]
pub fn out_of_place_sort_with_counts<T>(
    src_bucket: &[T],
    dst_bucket: &mut [T],
    level: usize,
    prefix_sums: &mut PrefixSums,
    next_counts: &mut Counts,
    scratch_counts: &mut Counts,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    if src_bucket.is_empty() {
        return;
    } else if src_bucket.len() == 1 {
        dst_bucket.copy_from_slice(src_bucket);
        next_counts[src_bucket[0].get_level(level) as usize] = 1;
        return;
    }

    let next_level = level + 1;

    let chunks = src_bucket.chunks_exact(8);
    let rem = chunks.remainder();

    chunks.into_iter().for_each(|chunk| {
        let b0 = chunk[0].get_level(level) as usize;
        let bn0 = chunk[0].get_level(next_level) as usize;
        let b1 = chunk[1].get_level(level) as usize;
        let bn1 = chunk[1].get_level(next_level) as usize;
        let b2 = chunk[2].get_level(level) as usize;
        let bn2 = chunk[2].get_level(next_level) as usize;
        let b3 = chunk[3].get_level(level) as usize;
        let bn3 = chunk[3].get_level(next_level) as usize;
        let b4 = chunk[4].get_level(level) as usize;
        let bn4 = chunk[4].get_level(next_level) as usize;
        let b5 = chunk[5].get_level(level) as usize;
        let bn5 = chunk[5].get_level(next_level) as usize;
        let b6 = chunk[6].get_level(level) as usize;
        let bn6 = chunk[6].get_level(next_level) as usize;
        let b7 = chunk[7].get_level(level) as usize;
        let bn7 = chunk[7].get_level(next_level) as usize;

        dst_bucket[prefix_sums[b0]] = chunk[0];
        prefix_sums[b0] += 1;
        next_counts[bn0] += 1;
        dst_bucket[prefix_sums[b1]] = chunk[1];
        prefix_sums[b1] += 1;
        scratch_counts[bn1] += 1;
        dst_bucket[prefix_sums[b2]] = chunk[2];
        prefix_sums[b2] += 1;
        next_counts[bn2] += 1;
        dst_bucket[prefix_sums[b3]] = chunk[3];
        prefix_sums[b3] += 1;
        scratch_counts[bn3] += 1;
        dst_bucket[prefix_sums[b4]] = chunk[4];
        prefix_sums[b4] += 1;
        next_counts[bn4] += 1;
        dst_bucket[prefix_sums[b5]] = chunk[5];
        prefix_sums[b5] += 1;
        scratch_counts[bn5] += 1;
        dst_bucket[prefix_sums[b6]] = chunk[6];
        prefix_sums[b6] += 1;
        next_counts[bn6] += 1;
        dst_bucket[prefix_sums[b7]] = chunk[7];
        prefix_sums[b7] += 1;
        scratch_counts[bn7] += 1;
    });

    rem.iter().for_each(|val| {
        let b = val.get_level(level) as usize;
        let bn = val.get_level(next_level) as usize;
        dst_bucket[prefix_sums[b]] = *val;
        prefix_sums[b] += 1;
        next_counts[bn] += 1;
    });

    for i in 0..256 {
        next_counts[i] += scratch_counts[i];
    }
}

#[inline]
pub fn lr_out_of_place_sort<T>(
    src_bucket: &[T],
    dst_bucket: &mut [T],
    level: usize,
    prefix_sums: &mut PrefixSums,
    ends: &mut EndOffsets,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    if src_bucket.len() < 2 {
        dst_bucket.copy_from_slice(src_bucket);
        return;
    }

    let mut left = 0;
    let mut right = src_bucket.len() - 1;
    let pre = src_bucket.len() % 8;

    for _ in 0..pre {
        let b = src_bucket[right].get_level(level) as usize;

        ends[b] = ends[b].saturating_sub(1);
        dst_bucket[ends[b]] = src_bucket[right];
        right = right.saturating_sub(1);
    }

    if pre == src_bucket.len() {
        return;
    }

    let end = (src_bucket.len() - pre) / 2;

    while left < end {
        let bl_0 = src_bucket[left].get_level(level) as usize;
        let bl_1 = src_bucket[left + 1].get_level(level) as usize;
        let bl_2 = src_bucket[left + 2].get_level(level) as usize;
        let bl_3 = src_bucket[left + 3].get_level(level) as usize;
        let br_0 = src_bucket[right].get_level(level) as usize;
        let br_1 = src_bucket[right - 1].get_level(level) as usize;
        let br_2 = src_bucket[right - 2].get_level(level) as usize;
        let br_3 = src_bucket[right - 3].get_level(level) as usize;

        dst_bucket[prefix_sums[bl_0]] = src_bucket[left];
        prefix_sums[bl_0] = prefix_sums[bl_0].wrapping_add(1);
        ends[br_0] = ends[br_0].saturating_sub(1);
        dst_bucket[ends[br_0]] = src_bucket[right];
        dst_bucket[prefix_sums[bl_1]] = src_bucket[left + 1];
        prefix_sums[bl_1] = prefix_sums[bl_1].wrapping_add(1);
        ends[br_1] = ends[br_1].saturating_sub(1);
        dst_bucket[ends[br_1]] = src_bucket[right - 1];
        dst_bucket[prefix_sums[bl_2]] = src_bucket[left + 2];
        prefix_sums[bl_2] = prefix_sums[bl_2].wrapping_add(1);
        ends[br_2] = ends[br_2].saturating_sub(1);
        dst_bucket[ends[br_2]] = src_bucket[right - 2];
        dst_bucket[prefix_sums[bl_3]] = src_bucket[left + 3];
        prefix_sums[bl_3] = prefix_sums[bl_3].wrapping_add(1);
        ends[br_3] = ends[br_3].saturating_sub(1);
        dst_bucket[ends[br_3]] = src_bucket[right - 3];

        left += 4;
        right -= 4;
    }
}

#[inline]
pub fn lr_out_of_place_sort_with_counts<T>(
    src_bucket: &[T],
    dst_bucket: &mut [T],
    level: usize,
    prefix_sums: &mut PrefixSums,
    ends: &mut EndOffsets,
    next_counts: &mut Counts,
    counts_scratch: &mut Counts,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    if src_bucket.is_empty() {
        return;
    } else if src_bucket.len() == 1 {
        dst_bucket.copy_from_slice(src_bucket);
        next_counts[src_bucket[0].get_level(level) as usize] = 1;
        return;
    }

    let next_level = level + 1;
    let mut left = 0;
    let mut right = src_bucket.len() - 1;
    let pre = src_bucket.len() % 8;

    for _ in 0..pre {
        let b = src_bucket[right].get_level(level) as usize;
        let bn = src_bucket[right].get_level(next_level) as usize;

        ends[b] = ends[b].saturating_sub(1);
        dst_bucket[ends[b]] = src_bucket[right];
        right = right.saturating_sub(1);
        next_counts[bn] += 1;
    }

    if pre == src_bucket.len() {
        return;
    }

    let end = (src_bucket.len() - pre) / 2;

    while left < end {
        let bl_0 = src_bucket[left].get_level(level) as usize;
        let bl_1 = src_bucket[left + 1].get_level(level) as usize;
        let bl_2 = src_bucket[left + 2].get_level(level) as usize;
        let bl_3 = src_bucket[left + 3].get_level(level) as usize;
        let br_0 = src_bucket[right].get_level(level) as usize;
        let br_1 = src_bucket[right - 1].get_level(level) as usize;
        let br_2 = src_bucket[right - 2].get_level(level) as usize;
        let br_3 = src_bucket[right - 3].get_level(level) as usize;

        dst_bucket[prefix_sums[bl_0]] = src_bucket[left];
        ends[br_0] = ends[br_0].saturating_sub(1);
        dst_bucket[ends[br_0]] = src_bucket[right];
        prefix_sums[bl_0] = prefix_sums[bl_0].wrapping_add(1);

        dst_bucket[prefix_sums[bl_1]] = src_bucket[left + 1];
        ends[br_1] = ends[br_1].saturating_sub(1);
        dst_bucket[ends[br_1]] = src_bucket[right - 1];
        prefix_sums[bl_1] = prefix_sums[bl_1].wrapping_add(1);

        dst_bucket[prefix_sums[bl_2]] = src_bucket[left + 2];
        ends[br_2] = ends[br_2].saturating_sub(1);
        dst_bucket[ends[br_2]] = src_bucket[right - 2];
        prefix_sums[bl_2] = prefix_sums[bl_2].wrapping_add(1);

        dst_bucket[prefix_sums[bl_3]] = src_bucket[left + 3];
        ends[br_3] = ends[br_3].saturating_sub(1);
        dst_bucket[ends[br_3]] = src_bucket[right - 3];
        prefix_sums[bl_3] = prefix_sums[bl_3].wrapping_add(1);

        let bnl_0 = src_bucket[left].get_level(next_level) as usize;
        let bnl_1 = src_bucket[left + 1].get_level(next_level) as usize;
        let bnl_2 = src_bucket[left + 2].get_level(next_level) as usize;
        let bnl_3 = src_bucket[left + 3].get_level(next_level) as usize;
        let bnr_0 = src_bucket[right].get_level(next_level) as usize;
        let bnr_1 = src_bucket[right - 1].get_level(next_level) as usize;
        let bnr_2 = src_bucket[right - 2].get_level(next_level) as usize;
        let bnr_3 = src_bucket[right - 3].get_level(next_level) as usize;

        next_counts[bnl_0] += 1;
        counts_scratch[bnr_0] += 1;
        next_counts[bnl_1] += 1;
        counts_scratch[bnr_1] += 1;
        next_counts[bnl_2] += 1;
        counts_scratch[bnr_2] += 1;
        next_counts[bnl_3] += 1;
        counts_scratch[bnr_3] += 1;

        left += 4;
        right = right.wrapping_sub(4);
    }

    for i in 0..256 {
        next_counts[i] += counts_scratch[i];
    }
}
