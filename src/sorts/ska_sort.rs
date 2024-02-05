//! `ska_sort` is a single-threaded, in-place algorithm described by Malte Skarupke.
//!
//! <https://probablydance.com/2016/12/27/i-wrote-a-faster-sorting-algorithm/>
//!
//! This implementation isn't entirely faithful to the original, however it follows the general
//! principle of skipping over the largest output bucket and simply swapping the remaining buckets
//! until the entire thing is sorted.
//!
//! The in-place nature of this algorithm makes it very efficient memory-wise.
//!
//! ## Characteristics
//!
//!  * in-place
//!  * memory efficient
//!  * unstable
//!  * single-threaded
//!
//! ## Performance
//!
//! This is generally slower than `lsb_sort` for smaller types T or smaller input arrays. For larger
//! types or inputs, the memory efficiency of this algorithm can make it faster than `lsb_sort`.

use crate::counts::{Counts, EndOffsets, PrefixSums};
use crate::sorter::Sorter;
use crate::RadixKey;
use partition::partition_index;
use std::cell::RefCell;
use std::rc::Rc;

pub fn ska_sort<T>(
    bucket: &mut [T],
    prefix_sums: &mut PrefixSums,
    end_offsets: &EndOffsets,
    level: usize,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    let mut finished = 0;
    let mut finished_map = [false; 256];
    let mut largest = 0;
    let mut largest_index = 0;

    for i in 0..256 {
        let rem = end_offsets[i] - prefix_sums[i];
        if rem == 0 {
            finished_map[i] = true;
            finished += 1;
        } else if rem > largest {
            largest = rem;
            largest_index = i;
        }
    }

    if largest == bucket.len() {
        // Already sorted
        return;
    } else if largest > (bucket.len() / 2) {
        // Partition in-place the largest chunk so we don't spend all our time
        // swapping things in and out that are already in the correct place.

        let li = largest_index as u8;
        let offs = partition_index(
            &mut bucket[prefix_sums[largest_index]..end_offsets[largest_index]],
            |v| v.get_level(level) == li,
        );

        prefix_sums[largest_index] += offs;
    }

    if !finished_map[largest_index] {
        finished_map[largest_index] = true;
        finished += 1;
    }

    while finished != 256 {
        for b in 0..256 {
            if finished_map[b] {
                continue;
            } else if prefix_sums[b] >= end_offsets[b] {
                finished_map[b] = true;
                finished += 1;
            }

            for i in prefix_sums[b]..end_offsets[b] {
                let new_b = bucket[i].get_level(level) as usize;
                bucket.swap(prefix_sums[new_b], i);
                prefix_sums[new_b] += 1;
            }
        }
    }
}

impl<'a> Sorter<'a> {
    pub(crate) fn ska_sort_adapter<T>(
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
        let end_offsets = self.cm.end_offsets(&counts.borrow(), &prefix_sums.borrow());

        ska_sort(
            bucket,
            &mut prefix_sums.borrow_mut(),
            &end_offsets.borrow(),
            level,
        );

        self.cm.return_counts(prefix_sums);
        self.cm.return_counts(end_offsets);

        if level == 0 {
            return;
        }

        self.director(bucket, counts, level - 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::sorter::Sorter;
    use crate::tuner::Algorithm;
    use crate::tuners::StandardTuner;
    use crate::utils::test_utils::{
        sort_comparison_suite, sort_single_algorithm, validate_u32_patterns, NumericTest,
    };
    use crate::RadixKey;

    fn test_ska_sort_adapter<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = sorter.cm.counts(inputs, T::LEVELS - 1);

            sorter.ska_sort_adapter(inputs, counts, T::LEVELS - 1);
        });
    }

    #[test]
    pub fn test_u8() {
        test_ska_sort_adapter(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_ska_sort_adapter(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_ska_sort_adapter(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_ska_sort_adapter(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_ska_sort_adapter(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_ska_sort_adapter(32usize);
    }

    #[test]
    pub fn test_basic_integration() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::Ska);
    }

    #[test]
    pub fn test_u32_patterns() {
        validate_u32_patterns(|inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = sorter.cm.counts(inputs, u32::LEVELS - 1);

            sorter.ska_sort_adapter(inputs, counts, u32::LEVELS - 1);
        });
    }
}
