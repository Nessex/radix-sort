use crate::sorter::Sorter;
use crate::tuner::Tuner;
use crate::tuners::SingleThreadedTuner;
#[cfg(feature = "multi-threaded")]
use crate::tuners::{LowMemoryTuner, StandardTuner};
use crate::RadixKey;

pub struct RadixSortBuilder<'a, T> {
    data: &'a mut [T],
    multi_threaded: bool,
    tuner: &'a (dyn Tuner + Send + Sync),
}

impl<'a, T> RadixSortBuilder<'a, T>
where
    T: RadixKey + Copy + Send + Sync,
{
    pub(crate) fn new(data: &'a mut [T]) -> Self {
        // TODO(nathan): Try to make this a compile-time assert
        // This is an invariant of RadixKey that must be upheld.
        assert_ne!(T::LEVELS, 0, "RadixKey must have at least 1 level");

        #[cfg(feature = "multi-threaded")]
        let (tuner, multi_threaded) = (&StandardTuner, true);
        #[cfg(not(feature = "multi-threaded"))]
        let (tuner, multi_threaded) = (&SingleThreadedTuner, false);

        Self {
            data,
            multi_threaded,
            tuner,
        }
    }

    /// `with_parallel(bool)` controls whether or not multiple algorithms will be allowed
    /// to run in parallel on different threads. This will NOT control whether
    /// multi-threaded algorithms will get used.
    ///
    /// If you also want the algorithms chosen to be only single-threaded algorithms,
    /// combine this with `with_single_threaded_tuner()`.
    ///
    /// ```
    /// use rdst::RadixSort;
    /// let mut data: Vec<usize> = vec![5, 22, 3, 7, 9];
    ///
    /// data
    ///     .radix_sort_builder()
    ///     .with_parallel(false)
    ///     .with_single_threaded_tuner()
    ///     .sort();
    /// ```
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.multi_threaded = parallel;

        self
    }

    /// `with_low_mem_tuner()` configures the sort to use a bunch of algorithms that use less
    /// memory for large inputs than the standard tuning. These algorithms include multi-threaded
    /// algorithms for better performance. In some situations, this tuning will be faster than the
    /// standard tuning, but in general use it will be slightly slower.
    ///
    /// ```
    /// use rdst::RadixSort;
    /// let mut data: Vec<usize> = vec![5, 22, 3, 7, 9];
    ///
    /// data
    ///     .radix_sort_builder()
    ///     .with_low_mem_tuner()
    ///     .sort();
    /// ```
    #[cfg(feature = "multi-threaded")]
    pub fn with_low_mem_tuner(mut self) -> Self {
        self.tuner = &LowMemoryTuner;

        self
    }

    /// `with_single_threaded_tuner()` configures the sort to use a tuner which only uses
    /// single-threaded sorting algorithms. This will NOT control whether or not algorithms are
    /// allowed to be run in parallel.
    ///
    /// For fully single-threaded operation, combine this with `with_parallel(false)`.
    ///
    /// ```
    /// use rdst::RadixSort;
    /// let mut data: Vec<usize> = vec![5, 22, 3, 7, 9];
    ///
    /// data
    ///     .radix_sort_builder()
    ///     .with_single_threaded_tuner()
    ///     .with_parallel(false)
    ///     .sort();
    /// ```
    pub fn with_single_threaded_tuner(mut self) -> Self {
        self.tuner = &SingleThreadedTuner;

        self
    }

    /// `with_tuner()` allows you to provide your own tuning for which sorting algorithm to use
    /// in a given situation.
    ///
    /// ```
    /// use rdst::RadixSort;
    /// use rdst::tuner::{Algorithm, Tuner, TuningParams};
    /// use rdst::counts::Counts;
    ///
    /// struct MyTuner;
    ///
    /// impl Tuner for MyTuner {
    ///     fn pick_algorithm(&self, p: &TuningParams, _counts: &[usize]) -> Algorithm {
    ///         if p.input_len >= 500_000 {
    ///             Algorithm::Ska
    ///         } else {
    ///             Algorithm::Lsb
    ///         }
    ///     }
    /// }
    ///
    /// let mut data: Vec<usize> = vec![5, 22, 3, 7, 9];
    ///
    /// data
    ///     .radix_sort_builder()
    ///     .with_tuner(&MyTuner {})
    ///     .sort();
    /// ```
    pub fn with_tuner(mut self, tuner: &'a (dyn Tuner + Send + Sync)) -> Self {
        self.tuner = tuner;

        self
    }

    /// `sort()` runs the configured sorting algorithm and consumes the RadixSortBuilder to return
    /// your mutable vec / slice back to you.
    ///
    /// ```
    /// use rdst::RadixSort;
    /// let mut data: Vec<usize> = vec![5, 22, 3, 7, 9];
    ///
    /// data
    ///     .radix_sort_builder()
    ///     .with_parallel(false)
    ///     .with_single_threaded_tuner()
    ///     .sort();
    ///
    /// data[0] = 123;
    /// ```
    pub fn sort(self) {
        // By definition, this is already sorted
        if self.data.len() <= 1 {
            return;
        }

        let sorter = Sorter::new(self.multi_threaded, self.tuner);
        sorter.top_level_director(self.data);
    }
}
