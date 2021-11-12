use crate::sort_manager::SortManager;
#[cfg(feature = "tuning")]
use crate::tuner::Tuner;
use crate::RadixKey;

pub trait RadixSort {
    /// radix_sort_unstable runs a radix sort based upon the `rdst::RadixKey` implementation
    /// of `T` in your `Vec<T>` or `[T]`.
    ///
    /// ```
    /// use rdst::RadixSort;
    ///
    /// let mut values = [3, 1, 2];
    /// values.radix_sort_unstable();
    ///
    /// assert_eq!(values, [1, 2, 3]);
    /// ```
    fn radix_sort_unstable(&mut self);

    /// radix_sort_unstable_with_tuning runs a radix sort with a provided set of tuning parameters.
    ///
    /// ```
    /// use rdst::{RadixSort, TuningParameters};
    /// let tuning = TuningParameters {
    ///     cpus: 1,
    ///     regions_sort_threshold: 100_000,
    ///     scanning_sort_threshold: 100_000,
    ///     recombinating_sort_threshold: 50_000,
    ///     ska_sort_threshold: 10_000,
    ///     par_count_threshold: 10_000,
    ///     scanner_read_size: 10_000,
    ///     in_place_sort_lsb_threshold: 10_000,
    /// };
    ///
    /// let mut values = [3, 1, 2];
    /// values.radix_sort_unstable_with_tuning(tuning);
    ///
    /// assert_eq!(values, [1, 2, 3]);
    /// ```
    #[cfg(feature = "tuning")]
    fn radix_sort_unstable_with_tuning(&mut self, tuner: Box<dyn Tuner + Send + Sync>);

    /// radix_sort_unstable runs the actual radix sort based upon the `rdst::RadixKey` implementation
    /// of `T` in your `Vec<T>` or `[T]`.
    ///
    /// It uses *mostly* in-place algorithms, providing significantly reduced memory usage. In
    /// general use, this is typically slightly slower than the regular sort provided by this
    /// library, however for some use-cases and platforms it may actually be faster. This has
    /// been seen in workloads with extremely unbalanced distributions.
    ///
    /// This utilizes a variant of regions sort (Obeya, Kahssay, Fan and Shun. 2019), so it has
    /// significantly better performance than traditional (typically single-threaded) in-place
    /// radix sorting algorithms such as American Flag sort.
    ///
    /// ```
    /// use rdst::RadixSort;
    ///
    /// let mut values = [3, 1, 2];
    /// values.radix_sort_unstable();
    ///
    /// assert_eq!(values, [1, 2, 3]);
    /// ```
    fn radix_sort_in_place_unstable(&mut self);

    #[cfg(feature = "tuning")]
    fn radix_sort_in_place_unstable_with_tuning(&mut self, tuner: Box<dyn Tuner + Send + Sync>);
}

impl<T> RadixSort for Vec<T>
where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    fn radix_sort_unstable(&mut self) {
        let sm = SortManager::new::<T>();
        sm.sort(self);
    }

    #[cfg(feature = "tuning")]
    fn radix_sort_unstable_with_tuning(&mut self, tuner: Box<dyn Tuner + Send + Sync>) {
        let sm = SortManager::new_with_tuning::<T>(tuner);
        sm.sort(self);
    }

    fn radix_sort_in_place_unstable(&mut self) {
        let sm = SortManager::new::<T>();
        sm.sort_in_place(self);
    }

    #[cfg(feature = "tuning")]
    fn radix_sort_in_place_unstable_with_tuning(&mut self, tuner: Box<dyn Tuner + Send + Sync>) {
        let sm = SortManager::new_with_tuning::<T>(tuner);
        sm.sort_in_place(self);
    }
}

impl<T> RadixSort for [T]
where
    T: RadixKey + Sized + Send + Copy + Sync,
{
    fn radix_sort_unstable(&mut self) {
        let sm = SortManager::new::<T>();
        sm.sort(self);
    }

    #[cfg(feature = "tuning")]
    fn radix_sort_unstable_with_tuning(&mut self, tuner: Box<dyn Tuner + Send + Sync>) {
        let sm = SortManager::new_with_tuning::<T>(tuner);
        sm.sort(self);
    }

    fn radix_sort_in_place_unstable(&mut self) {
        let sm = SortManager::new::<T>();
        sm.sort_in_place(self);
    }

    #[cfg(feature = "tuning")]
    fn radix_sort_in_place_unstable_with_tuning(&mut self, tuner: Box<dyn Tuner + Send + Sync>) {
        let sm = SortManager::new_with_tuning::<T>(tuner);
        sm.sort_in_place(self);
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{sort_comparison_suite, NumericTest};
    use crate::RadixSort;

    fn test_full_sort<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| inputs.radix_sort_unstable());
    }

    fn test_in_place_full_sort<T>(shift: T)
    where
        T: NumericTest<T>,
    {
        sort_comparison_suite(shift, |inputs| inputs.radix_sort_in_place_unstable());
    }

    #[test]
    pub fn test_u8() {
        test_full_sort(0u8);
    }

    #[test]
    pub fn test_u16() {
        test_full_sort(8u16);
    }

    #[test]
    pub fn test_u32() {
        test_full_sort(16u32);
    }

    #[test]
    pub fn test_u64() {
        test_full_sort(32u64);
    }

    #[test]
    pub fn test_u128() {
        test_full_sort(64u128);
    }

    #[test]
    pub fn test_usize() {
        test_full_sort(32usize);
    }

    #[test]
    pub fn test_i8() {
        test_full_sort(0i8);
    }

    #[test]
    pub fn test_i16() {
        test_full_sort(8i16);
    }

    #[test]
    pub fn test_i32() {
        test_full_sort(16i32);
    }

    #[test]
    pub fn test_i64() {
        test_full_sort(32i64);
    }

    #[test]
    pub fn test_i128() {
        test_full_sort(64i128);
    }

    #[test]
    pub fn test_isize() {
        test_full_sort(32isize);
    }

    #[test]
    pub fn test_f32() {
        test_full_sort(16u32);
    }

    #[test]
    pub fn test_f64() {
        test_full_sort(32u64);
    }

    #[test]
    pub fn test_in_place_u8() {
        test_in_place_full_sort(0u8);
    }

    #[test]
    pub fn test_in_place_u16() {
        test_in_place_full_sort(8u16);
    }

    #[test]
    pub fn test_in_place_u32() {
        test_in_place_full_sort(16u32);
    }

    #[test]
    pub fn test_in_place_u64() {
        test_in_place_full_sort(32u64);
    }

    #[test]
    pub fn test_in_place_u128() {
        test_in_place_full_sort(64u128);
    }

    #[test]
    pub fn test_in_place_usize() {
        test_in_place_full_sort(32usize);
    }

    #[test]
    pub fn test_in_place_i8() {
        test_in_place_full_sort(0i8);
    }

    #[test]
    pub fn test_in_place_i16() {
        test_in_place_full_sort(8i16);
    }

    #[test]
    pub fn test_in_place_i32() {
        test_in_place_full_sort(16i32);
    }

    #[test]
    pub fn test_in_place_i64() {
        test_in_place_full_sort(32i64);
    }

    #[test]
    pub fn test_in_place_i128() {
        test_in_place_full_sort(64i128);
    }

    #[test]
    pub fn test_in_place_isize() {
        test_in_place_full_sort(32isize);
    }

    #[test]
    pub fn test_in_place_f32() {
        test_in_place_full_sort(16u32);
    }

    #[test]
    pub fn test_in_place_f64() {
        test_in_place_full_sort(32u64);
    }
}
