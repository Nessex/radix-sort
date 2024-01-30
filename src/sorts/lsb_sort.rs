//! `lsb_sort` is a Least-Significant Bit first radix sort.
//!
//! ## Characteristics
//!
//!  * out-of-place
//!  * stable
//!  * single-threaded
//!
//! ## Performance
//!
//! This sort is generally the fastest single-threaded algorithm implemented in this crate.
//!
//! ## Optimizations
//!
//! ### Ping-pong arrays
//!
//! As this is an out-of-place algorithm, we can save some copying by using the temporary array
//! as our input array, and our original array as our output array on odd runs of the algorithm.
//!
//! ### Level skipping
//!
//! When a level has all counts in one bucket (i.e. all values are equal), we can skip the level
//! entirely. This is done by checking counts rather than the actual data.
//!
//! ### Counting while sorting
//!
//! This is implemented in the underlying `out_of_place_sort`. While sorting, we also count the next
//! level to provide a small but significant performance boost. This is not a huge win as it removes
//! some caching benefits etc., but has been benchmarked at roughly 5-15% speedup.

use crate::sorter::Sorter;
use crate::sorts::out_of_place_sort::{
    lr_out_of_place_sort, lr_out_of_place_sort_with_counts, out_of_place_sort,
    out_of_place_sort_with_counts,
};
use std::cell::RefCell;
use std::rc::Rc;

use crate::counts::{CountMeta, Counts};
use crate::RadixKey;

impl<'a> Sorter<'a> {
    pub(crate) fn lsb_sort_adapter<T>(
        &self,
        lr: bool,
        bucket: &mut [T],
        last_counts: Rc<RefCell<Counts>>,
        start_level: usize,
        end_level: usize,
    ) where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if bucket.len() < 2 {
            return;
        }

        self.cm.with_tmp_buffer(bucket.len(), |cm, tmp_bucket| {
            let mut invert = false;
            let mut use_next_counts = false;
            let mut counts = cm.get_empty_counts();
            let mut meta = CountMeta::default();
            let mut next_counts: Rc<RefCell<Counts>> = cm.get_empty_counts();

            for level in start_level..=end_level {
                if level == end_level {
                    cm.return_counts(counts);
                    counts = last_counts.clone();
                } else if use_next_counts {
                    counts.borrow_mut().clear();
                    (counts, next_counts) = (next_counts, counts);
                } else {
                    let mut c_mut = counts.borrow_mut();
                    c_mut.clear();
                    cm.count_into(&mut c_mut, &mut meta, bucket, level);
                    drop(c_mut);
                    next_counts.borrow_mut().clear();

                    if meta.already_sorted {
                        use_next_counts = false;
                        continue;
                    }
                };

                let counts = counts.borrow();
                let sums_rc = cm.prefix_sums(&counts);
                let mut sums = sums_rc.borrow_mut();
                let should_count = end_level != 0 && level < (end_level - 1);
                use_next_counts = should_count;

                match (lr, invert, should_count) {
                    (true, true, true) => {
                        let ends = cm.end_offsets(&counts, &sums);
                        let scratch_counts = cm.get_empty_counts();
                        lr_out_of_place_sort_with_counts(
                            tmp_bucket,
                            bucket,
                            level,
                            &mut sums,
                            &mut ends.borrow_mut(),
                            &mut next_counts.borrow_mut(),
                            &mut scratch_counts.borrow_mut(),
                        );
                        cm.return_counts(ends);
                        cm.return_counts(scratch_counts);
                    }
                    (true, true, false) => {
                        let ends = cm.end_offsets(&counts, &sums);
                        lr_out_of_place_sort(
                            tmp_bucket,
                            bucket,
                            level,
                            &mut sums,
                            &mut ends.borrow_mut(),
                        );
                        cm.return_counts(ends);
                    }
                    (true, false, true) => {
                        let ends = cm.end_offsets(&counts, &sums);
                        let scratch_counts = cm.get_empty_counts();
                        lr_out_of_place_sort_with_counts(
                            bucket,
                            tmp_bucket,
                            level,
                            &mut sums,
                            &mut ends.borrow_mut(),
                            &mut next_counts.borrow_mut(),
                            &mut scratch_counts.borrow_mut(),
                        );
                        cm.return_counts(ends);
                        cm.return_counts(scratch_counts);
                    }
                    (true, false, false) => {
                        let ends = cm.end_offsets(&counts, &sums);
                        lr_out_of_place_sort(
                            bucket,
                            tmp_bucket,
                            level,
                            &mut sums,
                            &mut ends.borrow_mut(),
                        );
                        cm.return_counts(ends);
                    }
                    (false, true, true) => {
                        let scratch_counts = cm.get_empty_counts();
                        out_of_place_sort_with_counts(
                            tmp_bucket,
                            bucket,
                            level,
                            &mut sums,
                            &mut next_counts.borrow_mut(),
                            &mut scratch_counts.borrow_mut(),
                        );
                        cm.return_counts(scratch_counts);
                    }
                    (false, true, false) => out_of_place_sort(tmp_bucket, bucket, level, &mut sums),
                    (false, false, true) => {
                        let scratch_counts = cm.get_empty_counts();
                        out_of_place_sort_with_counts(
                            bucket,
                            tmp_bucket,
                            level,
                            &mut sums,
                            &mut next_counts.borrow_mut(),
                            &mut scratch_counts.borrow_mut(),
                        );
                        cm.return_counts(scratch_counts);
                    }
                    (false, false, false) => {
                        out_of_place_sort(bucket, tmp_bucket, level, &mut sums)
                    }
                };

                drop(sums);
                cm.return_counts(sums_rc);

                invert = !invert;
            }

            cm.return_counts(counts);
            cm.return_counts(next_counts);

            if invert {
                bucket.copy_from_slice(tmp_bucket);
            }
        });
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

    fn test_lsb_sort_adapter<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = sorter.cm.counts(inputs, T::LEVELS - 1);

            sorter.lsb_sort_adapter(false, inputs, counts, 0, T::LEVELS - 1)
        });

        sort_comparison_suite(shift, |inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = sorter.cm.counts(inputs, T::LEVELS - 1);

            sorter.lsb_sort_adapter(true, inputs, counts, 0, T::LEVELS - 1);
        });
    }

    #[test]
    pub fn test_u8() {
        test_lsb_sort_adapter(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_lsb_sort_adapter(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_lsb_sort_adapter(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_lsb_sort_adapter(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_lsb_sort_adapter(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_lsb_sort_adapter(32usize);
    }

    #[test]
    pub fn test_basic_integration() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::Lsb);
    }

    #[test]
    pub fn test_basic_integration_lr() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::LrLsb);
    }

    #[test]
    pub fn test_u32_patterns() {
        validate_u32_patterns(|inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            let (counts, _) = sorter.cm.counts(inputs, u32::LEVELS - 1);

            sorter.lsb_sort_adapter(true, inputs, counts, 0, u32::LEVELS - 1);
        });
    }
}
