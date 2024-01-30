//! `comparative_sort` is a radix-aware comparison sort. It operates on radixes rather than
//! whole numbers to support all the same use-cases as the original radix sort including
//! sorting across multiple keys or partial keys etc.
//!
//! The purpose of this sort is to ensure that the library can provide a simpler interface. Without
//! this sort, users would have to implement both `RadixKey` for the radix sort, _and_ `Ord` for
//! the comparison sort. With this, only `RadixKey` is required.
//!
//! While the performance generally sucks, it is still faster than setting up for a full radix sort
//! in situations where there are very few items.
//!
//! ## Characteristics
//!
//!  * in-place
//!  * unstable
//!  * single-threaded
//!
//! ## Performance
//!
//! This is even slower than a typical comparison sort and so is only used as a fallback for very
//! small inputs. However for those very small inputs it provides a significant speed-up due to
//! having essentially no overhead (from count arrays, buffers etc.) compared to a radix sort.

use crate::sorter::Sorter;
use crate::RadixKey;
use std::cmp::Ordering;

impl<'a> Sorter<'a> {
    pub(crate) fn comparative_sort<T>(&self, bucket: &mut [T], start_level: usize)
    where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if bucket.len() < 2 {
            return;
        }

        bucket.sort_unstable_by(|a, b| -> Ordering {
            let mut level = start_level;
            loop {
                let cmp = a.get_level(level).cmp(&b.get_level(level));

                if level != 0 && cmp == Ordering::Equal {
                    level -= 1;
                    continue;
                }

                return cmp;
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

    fn test_comparative_sort_adapter<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            sorter.comparative_sort(inputs, T::LEVELS - 1);
        });
    }

    #[test]
    pub fn test_u8() {
        test_comparative_sort_adapter(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_comparative_sort_adapter(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_comparative_sort_adapter(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_comparative_sort_adapter(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_comparative_sort_adapter(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_comparative_sort_adapter(32usize);
    }

    #[test]
    pub fn test_basic_integration() {
        sort_single_algorithm::<u32>(1_000_000, Algorithm::Comparative);
    }

    #[test]
    pub fn test_u32_patterns() {
        validate_u32_patterns(|inputs| {
            let sorter = Sorter::new(true, &StandardTuner);
            sorter.comparative_sort(inputs, u32::LEVELS - 1);
        });
    }
}
