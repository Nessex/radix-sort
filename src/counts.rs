use std::cell::RefCell;

use std::ops::{Index, IndexMut};
use std::ptr::copy_nonoverlapping;

use crate::RadixKey;
use std::rc::Rc;
use std::slice::{Iter, SliceIndex};

#[derive(Default)]
pub struct CountManager {}

#[repr(C, align(4096))]
#[derive(Clone)]
pub struct Counter([usize; 256 * 4]);

impl Default for Counter {
    fn default() -> Self {
        Counter([0usize; 256 * 4])
    }
}

#[repr(C, align(2048))]
#[derive(Clone)]
pub struct Counts([usize; 256]);
pub type PrefixSums = Counts;
pub type EndOffsets = Counts;

impl<I> Index<I> for Counts
where
    I: SliceIndex<[usize]>,
{
    type Output = I::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &I::Output {
        &self.0[index]
    }
}

impl<I> IndexMut<I> for Counts
where
    I: SliceIndex<[usize]>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        &mut self.0[index]
    }
}

impl Default for Counts {
    fn default() -> Self {
        Counts([0usize; 256])
    }
}

#[derive(Default, Clone, Copy)]
pub struct CountMeta {
    pub first: u8,
    pub last: u8,
    pub already_sorted: bool,
}

#[derive(Default)]
struct ThreadContext {
    pub counter: RefCell<Counter>,
    pub counts: RefCell<Vec<Rc<RefCell<Counts>>>>,
    pub tmp: RefCell<Vec<u8>>,
}

impl CountManager {
    thread_local! {
        static THREAD_CTX: ThreadContext = Default::default();
    }

    #[inline(always)]
    pub fn get_empty_counts(&self) -> Rc<RefCell<Counts>> {
        if let Some(counts) = Self::THREAD_CTX.with(|ct| ct.counts.borrow_mut().pop()) {
            counts
        } else {
            Default::default()
        }
    }

    #[inline(always)]
    pub fn return_counts(&self, counts: Rc<RefCell<Counts>>) {
        counts.borrow_mut().clear();
        Self::THREAD_CTX.with(|ct| ct.counts.borrow_mut().push(counts));
    }

    #[inline(always)]
    pub fn count_into<T: RadixKey>(
        &self,
        counts: &mut Counts,
        meta: &mut CountMeta,
        bucket: &[T],
        level: usize,
    ) {
        Self::THREAD_CTX.with(|ct| {
            ct.counter
                .borrow_mut()
                .count_into(counts, meta, bucket, level)
        })
    }

    #[inline(always)]
    pub fn counts<T: RadixKey>(&self, bucket: &[T], level: usize) -> (Rc<RefCell<Counts>>, bool) {
        let counts = self.get_empty_counts();
        let mut meta = CountMeta::default();
        Self::THREAD_CTX.with(|ct| {
            ct.counter
                .borrow_mut()
                .count_into(&mut counts.borrow_mut(), &mut meta, bucket, level)
        });

        (counts, meta.already_sorted)
    }

    #[inline(always)]
    pub fn prefix_sums(&self, counts: &Counts) -> Rc<RefCell<PrefixSums>> {
        let sums = self.get_empty_counts();
        let mut s = sums.borrow_mut();

        let mut running_total = 0;
        for (i, c) in counts.into_iter().enumerate() {
            s[i] = running_total;
            running_total += c;
        }
        drop(s);

        sums
    }

    #[inline(always)]
    pub fn end_offsets(
        &self,
        counts: &Counts,
        prefix_sums: &PrefixSums,
    ) -> Rc<RefCell<EndOffsets>> {
        let end_offsets = self.get_empty_counts();
        let mut eo = end_offsets.borrow_mut();

        eo[0..255].copy_from_slice(&prefix_sums[1..256]);
        eo[255] = counts[255] + prefix_sums[255];
        drop(eo);

        end_offsets
    }

    #[inline(always)]
    pub fn with_tmp_buffer<T, F>(&self, src_bucket: &mut [T], mut f: F)
    where
        T: Copy,
        F: FnMut(&CountManager, &mut [T], &mut [T]),
    {
        Self::THREAD_CTX.with(|ct| {
            let byte_len = std::mem::size_of_val(src_bucket);
            let mut t = ct.tmp.borrow_mut();

            if t.len() < byte_len {
                *t = Vec::with_capacity(byte_len);
            }

            // Safety: The buffer is guaranteed to have enough capacity by the logic above.
            // As the data is copied from the source buffer to the temporary buffer, and
            // T is Copy, the data is therefore correctly initialized (assuming the source itself is).
            // Len is set to 0 until the end to ensure that the compiler doesn't assume the buffer
            // is fully initialized before that point.
            let tmp = unsafe {
                t.set_len(0);
                let ptr = t.as_mut_ptr() as *mut T;
                copy_nonoverlapping(src_bucket.as_ptr(), ptr, src_bucket.len());
                t.set_len(byte_len);
                std::slice::from_raw_parts_mut(ptr, src_bucket.len())
            };

            f(self, src_bucket, tmp);
        });
    }
}

impl Counter {
    #[inline(always)]
    fn clear(&mut self) {
        self.0.iter_mut().for_each(|x| *x = 0);
    }

    #[inline(always)]
    pub fn count_into<T: RadixKey>(
        &mut self,
        counts: &mut Counts,
        meta: &mut CountMeta,
        bucket: &[T],
        level: usize,
    ) {
        #[cfg(feature = "work_profiles")]
        println!("({}) COUNT", level);

        self.clear();
        counts.clear();

        if bucket.is_empty() {
            meta.first = 0;
            meta.last = 0;
            meta.already_sorted = true;
            return;
        } else if bucket.len() == 1 {
            let b = bucket[0].get_level(level) as usize;
            counts[b] += 1;

            meta.first = b as u8;
            meta.last = b as u8;
            meta.already_sorted = true;
            return;
        }

        let mut already_sorted = true;
        let first = bucket.first().unwrap().get_level(level);
        let last = bucket.last().unwrap().get_level(level);

        let mut continue_from = bucket.len();
        let mut prev = 0usize;

        // First, count directly into the output buffer until we find a value that is out of order.
        for (i, item) in bucket.iter().enumerate() {
            let b = item.get_level(level) as usize;
            counts[b] += 1;

            if b < prev {
                continue_from = i + 1;
                already_sorted = false;
                break;
            }

            prev = b;
        }

        if continue_from == bucket.len() {
            meta.first = first;
            meta.last = last;
            meta.already_sorted = already_sorted;
            return;
        }

        let chunks = bucket[continue_from..].chunks_exact(4);
        let rem = chunks.remainder();

        chunks.into_iter().for_each(|chunk| {
            let a = chunk[0].get_level(level) as usize;
            let b = chunk[1].get_level(level) as usize;
            let c = chunk[2].get_level(level) as usize;
            let d = chunk[3].get_level(level) as usize;

            self.0[a * 4] += 1;
            self.0[1 + b * 4] += 1;
            self.0[2 + c * 4] += 1;
            self.0[3 + d * 4] += 1;
        });

        rem.iter().for_each(|v| {
            let b = v.get_level(level) as usize;
            counts[b] += 1;
        });

        for i in 0..256 {
            let agg = self.0[i * 4] + self.0[1 + i * 4] + self.0[2 + i * 4] + self.0[3 + i * 4];
            counts[i] += agg;
        }

        meta.first = first;
        meta.last = last;
        meta.already_sorted = already_sorted;
    }
}

impl Counts {
    #[inline(always)]
    pub fn clear(&mut self) {
        self.0.iter_mut().for_each(|x| *x = 0);
    }

    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn inner(&self) -> &[usize; 256] {
        &self.0
    }
}

impl IntoIterator for Counts {
    type Item = usize;
    type IntoIter = core::array::IntoIter<usize, 256>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Counts {
    type Item = &'a usize;
    type IntoIter = Iter<'a, usize>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
