use crate::director::director;
use crate::tuner::Tuner;
use crate::utils::*;
use crate::RadixKey;

// Based upon (with modifications):
// https://probablydance.com/2016/12/27/i-wrote-a-faster-sorting-algorithm/
pub fn ska_sort<T>(bucket: &mut [T], prefix_sums: &mut [usize], end_offsets: &mut [usize], level: usize)
where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    let mut finished = 1;
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

        end_offsets[i] -= 1;
    }

    if largest == bucket.len() {
        // Already sorted
        return;
    }

    finished_map[largest_index] = true;

    while finished != 256 {
        for b in 0..256 {
            if finished_map[b] {
                continue;
            } else if prefix_sums[b] > end_offsets[b] {
                finished_map[b] = true;
                finished += 1;
                continue;
            }

            let mut left = prefix_sums[b];
            let mut right = end_offsets[b];

            loop {
                if left == right {
                    let b = bucket[left].get_level(level) as usize;
                    bucket.swap(prefix_sums[b], left);
                    prefix_sums[b] += 1;
                    break;
                } else if left > right {
                    break;
                }

                let bl = bucket[left].get_level(level) as usize;
                let br = bucket[right].get_level(level) as usize;
                bucket.swap(prefix_sums[bl], left);
                bucket.swap(end_offsets[br], right);
                prefix_sums[bl] += 1;
                end_offsets[br] = end_offsets[br].saturating_sub(1);
                left += 1;
                right = right.saturating_sub(1);
            }
        }
    }
}

#[allow(dead_code)]
pub fn ska_sort_adapter<T>(
    tuner: &(dyn Tuner + Send + Sync),
    in_place: bool,
    bucket: &mut [T],
    level: usize,
) where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    let (counts, level) = if let Some(s) = get_counts_and_level_descending(bucket, level, 0, false)
    {
        s
    } else {
        return;
    };

    let plateaus = detect_plateaus(bucket, level);
    let (mut prefix_sums, mut end_offsets) = apply_plateaus(bucket, &counts, &plateaus);

    ska_sort(bucket, &mut prefix_sums, &mut end_offsets, level);

    if level == 0 {
        return;
    }

    director(tuner, in_place, bucket, counts.to_vec(), level - 1);
}

#[cfg(test)]
mod tests {
    use crate::sorts::ska_sort::ska_sort_adapter;
    use crate::test_utils::{sort_comparison_suite, NumericTest};
    use crate::tuner::DefaultTuner;

    fn test_ska_sort_adapter<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        let tuner = DefaultTuner {};
        sort_comparison_suite(shift, |inputs| {
            ska_sort_adapter(&tuner, true, inputs, T::LEVELS - 1)
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
}
