//! # rdst
//!
//! rdst is a flexible native Rust implementation of multi-threaded unstable radix sort.
//!
//! ## Usage
//!
//! ```rust
//! use rdst::RadixSort;
//! let mut my_vec = vec![4, 7, 1, 6, 5, 3, 2, 8, 9];
//!
//! my_vec.radix_sort_unstable();
//! ```
//!
//! In the simplest case, you can use this sort by simply calling `my_vec.radix_sort_unstable()`. If you have a custom type to sort, you may need to implement `RadixKey` for that type.
//!
//! ## Default Implementations
//!
//! `RadixKey` is implemented for `Vec` and `[T]` of the following types out-of-the-box:
//!
//!  * `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
//!  * `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
//!  * `f32`, `f64`
//!  * `[u8; N]`
//!
//! ### Implementing `RadixKey`
//!
//! To be able to sort custom types, implement `RadixKey` as below.
//!
//!  * `LEVELS` should be set to the total number of bytes you will consider for each item being sorted
//!  * `get_level` should return the corresponding bytes from the least significant byte to the most significant byte
//!
//! Notes:
//! * This allows you to implement radix keys that span multiple values, or to implement radix keys that only look at part of a value.
//! * You should try to make this as fast as possible, so consider using branchless implementations wherever possible
//!
//! ```
//! use rdst::RadixKey;
//!
//! struct MyType(u32);
//!
//! impl RadixKey for MyType {
//!     const LEVELS: usize = 4;
//!
//!     #[inline]
//!     fn get_level(&self, level: usize) -> u8 {
//!         (self.0 >> (level * 8)) as u8
//!     }
//! }
//! ```
//!
//! #### Partial `RadixKey`
//!
//! If you know your type has bytes that will always be zero, you can skip those bytes to speed up the sorting process. For instance, if you have a `u32` where values never exceed `10000`, you only need to consider two of the bytes. You could implement this as such:
//!
//! ```
//! use rdst::RadixKey;
//! struct U32Wrapper(u32);
//!
//! impl RadixKey for U32Wrapper {
//!     const LEVELS: usize = 2;
//!
//!     #[inline]
//!     fn get_level(&self, level: usize) -> u8 {
//!         (self.0 >> (level * 8)) as u8
//!     }
//! }
//! ```
//!
//! #### Multi-value `RadixKey`
//!
//! If your type has multiple values you need to search by, simply create a `RadixKey` that spans both values.
//!
//! ```
//! use rdst::RadixKey;
//! struct MyStruct {
//!     key_1: u8,
//!     key_2: u8,
//!     key_3: u8,
//! }
//! impl RadixKey for MyStruct {
//!     const LEVELS: usize = 3;
//!
//!     #[inline]
//!     fn get_level(&self, level: usize) -> u8 {
//!         match level {
//!           0 => self.key_1,
//!           1 => self.key_2,
//!           _ => self.key_3,
//!         }
//!     }
//! }
//! ```
//!
//! ## Low-memory Variant
//!
//! ```
//! use rdst::RadixSort;
//! let mut my_vec: Vec<usize> = vec![10, 15, 0, 22, 9];
//! my_vec
//!     .radix_sort_builder()
//!     .with_low_mem_tuner()
//!     .sort();
//! ```
//!
//! This library also includes a _mostly_ in-place variant of radix sort. This is useful in cases where memory or memory bandwidth are more limited. Generally, this algorithm is slightly slower than the standard algorithm, however in specific circumstances this algorithm may even provide a speed boost. It is worth benchmarking against your use-case if you need the ultimate level of performance.
//!
//! ## Single-threaded Variant
//!
//! To make this library use an entirely single-threaded set of algorithms and processes, you can use the following snippet.
//!
//! ```
//! use rdst::RadixSort;
//! let mut my_vec: Vec<usize> = vec![10, 15, 0, 22, 9];
//!
//! my_vec
//!     .radix_sort_builder()
//!     // Use a tuner that only includes single-threaded algorithms
//!     .with_single_threaded_tuner()
//!     // Don't run multiple algorithms (even single-threaded ones) in parallel
//!     .with_parallel(false)
//!     .sort();
//! ```
//!
//! ## Custom Tuners
//!
//! Tuners are things which you can implement to control which sorting algorithms are used. There are many radix sorting algorithms implemented as part of this crate, and they all have their pros and cons. If you have a very specific use-case it may be worth your time to tune the sort yourself.
//!
//! ```
//! use rdst::RadixSort;
//! use rdst::tuner::{Algorithm, Tuner, TuningParams};
//!
//! struct MyTuner;
//!
//! impl Tuner for MyTuner {
//!     fn pick_algorithm(&self, p: &TuningParams, _counts: &[usize]) -> Algorithm {
//!         if p.input_len >= 500_000 {
//!             Algorithm::Ska
//!         } else {
//!             Algorithm::Lsb
//!         }
//!     }
//! }
//!
//! let mut my_vec: Vec<usize> = vec![10, 25, 9, 22, 6];
//! my_vec
//!     .radix_sort_builder()
//!     .with_tuner(&MyTuner {})
//!     .sort();
//! ```
//!
//! ## License
//!
//! Licensed under either of
//!
//! * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
//! * MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
//!
//! at your option.
//!
//! ### Contribution
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod radix_key;
#[cfg(feature = "default-implementations")]
mod radix_key_impl;
mod radix_sort_builder;

#[cfg(not(any(test, feature = "bench")))]
mod sorts;
#[cfg(any(test, feature = "bench"))]
pub mod sorts;

#[cfg(not(any(test, feature = "bench", feature = "tuning")))]
mod utils;
#[cfg(any(test, feature = "bench", feature = "tuning"))]
pub mod utils;

mod radix_sort;
mod sorter;
mod tuners;

// Public modules
pub mod tuner;

// Public exports
pub use radix_key::RadixKey;
pub use radix_sort::RadixSort;
#[doc(inline)]
pub use radix_sort_builder::RadixSortBuilder;
