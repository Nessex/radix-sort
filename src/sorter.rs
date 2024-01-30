use crate::counts::{CountManager, Counts};
use crate::tuner::{Algorithm, Tuner, TuningParams};
use crate::utils::*;
use crate::RadixKey;
use arbitrary_chunks::ArbitraryChunks;
#[cfg(feature = "multi-threaded")]
use rayon::current_num_threads;
#[cfg(feature = "multi-threaded")]
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::max;
use std::rc::Rc;

pub struct Sorter<'a> {
    multi_threaded: bool,
    pub(crate) tuner: &'a (dyn Tuner + Send + Sync),
    pub(crate) cm: CountManager,
}

impl<'a> Sorter<'a> {
    pub fn new(multi_threaded: bool, tuner: &'a (dyn Tuner + Send + Sync)) -> Self {
        Self {
            multi_threaded,
            tuner,
            cm: CountManager::default(),
        }
    }

    #[inline]
    fn run_sort<T>(
        &self,
        level: usize,
        bucket: &mut [T],
        counts: Rc<RefCell<Counts>>,
        tile_counts: Option<Vec<Counts>>,
        #[allow(unused)] tile_size: usize,
        algorithm: Algorithm,
    ) where
        T: RadixKey + Copy + Sized + Send + Sync + 'a,
    {
        if cfg!(feature = "multi-threaded") {
            if let Some(tc) = tile_counts {
                match algorithm {
                    Algorithm::MtOop => {
                        self.mt_oop_sort_adapter(bucket, level, counts, tc, tile_size)
                    }
                    Algorithm::Recombinating => {
                        self.recombinating_sort_adapter(bucket, counts, tc, tile_size, level)
                    }
                    Algorithm::Regions => {
                        self.regions_sort_adapter(bucket, counts, tc, tile_size, level)
                    }
                    _ => match algorithm {
                        Algorithm::MtLsb => self.mt_lsb_sort_adapter(bucket, 0, level, tile_size),
                        Algorithm::Scanning => self.scanning_sort_adapter(bucket, counts, level),
                        Algorithm::Comparative => self.comparative_sort(bucket, level),
                        Algorithm::LrLsb => self.lsb_sort_adapter(true, bucket, counts, 0, level),
                        Algorithm::Lsb => self.lsb_sort_adapter(false, bucket, counts, 0, level),
                        Algorithm::Ska => self.ska_sort_adapter(bucket, counts, level),
                        _ => panic!(
                            "Bad algorithm: {:?} with unused tc for len: {}",
                            algorithm,
                            bucket.len()
                        ),
                    },
                }
            } else {
                match algorithm {
                    Algorithm::MtLsb => self.mt_lsb_sort_adapter(bucket, 0, level, tile_size),
                    Algorithm::Comparative => self.comparative_sort(bucket, level),
                    Algorithm::LrLsb => self.lsb_sort_adapter(true, bucket, counts, 0, level),
                    Algorithm::Lsb => self.lsb_sort_adapter(false, bucket, counts, 0, level),
                    Algorithm::Ska => self.ska_sort_adapter(bucket, counts, level),
                    Algorithm::Scanning => self.scanning_sort_adapter(bucket, counts, level),
                    _ => panic!("Bad algorithm: {:?} for len: {}", algorithm, bucket.len()),
                }
            }
        } else {
            match algorithm {
                Algorithm::LrLsb => self.lsb_sort_adapter(true, bucket, counts, 0, level),
                Algorithm::Lsb => self.lsb_sort_adapter(false, bucket, counts, 0, level),
                Algorithm::Ska => self.ska_sort_adapter(bucket, counts, level),
                Algorithm::Comparative => self.comparative_sort(bucket, level),
                // XXX: The compiler currently doesn't recognize that the other options are not available due to the
                // missing feature flag, so we need to add a catch-all here.
                _ => panic!("Bad algorithm: {:?} for len: {}", algorithm, bucket.len()),
            }
        }
    }

    fn handle_chunk<T>(
        &self,
        chunk: &mut [T],
        level: usize,
        parent_len: Option<usize>,
        threads: usize,
    ) where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        if chunk.len() <= 1 {
            return;
        } else if chunk.len() <= 128 {
            self.comparative_sort(chunk, level);
            return;
        }

        let use_tiles =
            cfg!(feature = "multi-threaded") && self.multi_threaded && chunk.len() >= 260_000;
        let tile_size = if use_tiles {
            max(30_000, cdiv(chunk.len(), threads))
        } else {
            chunk.len()
        };
        let tp = TuningParams {
            threads,
            level,
            total_levels: T::LEVELS,
            input_len: chunk.len(),
            parent_len,
        };

        let mut tile_counts: Option<Vec<Counts>> = None;
        let mut already_sorted = false;

        if use_tiles {
            let (tc, s) = get_tile_counts(&self.cm, chunk, tile_size, level);
            tile_counts = Some(tc);
            already_sorted = s;
        }

        let counts = if let Some(tile_counts) = &tile_counts {
            aggregate_tile_counts(&self.cm, tile_counts)
        } else {
            let (rc, ra) = self.cm.counts(chunk, level);
            already_sorted = ra;

            rc
        };

        if already_sorted || (chunk.len() >= 30_000 && is_homogenous(&counts.borrow())) {
            if level != 0 {
                self.director(chunk, counts, level - 1);
            }

            return;
        }

        let algorithm = self.tuner.pick_algorithm(&tp, counts.borrow().inner());

        // Ensure tile_counts is always set when it is required
        if tile_counts.is_none() {
            tile_counts = match algorithm {
                #[cfg(feature = "multi-threaded")]
                Algorithm::MtOop
                | Algorithm::MtLsb
                | Algorithm::Recombinating
                | Algorithm::Regions => Some(vec![counts.borrow().clone()]),
                _ => None,
            };
        }

        #[cfg(feature = "work_profiles")]
        println!("({}) PAR: {:?}", level, algorithm);

        self.run_sort(level, chunk, counts, tile_counts, tile_size, algorithm);
    }

    #[inline]
    pub fn top_level_director<T>(&self, bucket: &mut [T])
    where
        T: RadixKey + Sized + Send + Copy + Sync + 'a,
    {
        #[cfg(feature = "multi-threaded")]
        let threads = current_num_threads();

        #[cfg(not(feature = "multi-threaded"))]
        let threads = 1;

        let level = T::LEVELS - 1;

        self.handle_chunk(bucket, level, None, threads);
    }

    #[inline]
    #[cfg(feature = "multi-threaded")]
    pub fn multi_threaded_director<T>(
        &self,
        bucket: &'a mut [T],
        counts: Rc<RefCell<Counts>>,
        level: usize,
    ) where
        T: 'a + RadixKey + Send + Copy + Sync,
    {
        let parent_len = Some(bucket.len());
        let threads = current_num_threads();

        let segment_size = cdiv(bucket.len(), threads);

        let mut running_total = 0;
        let mut radix_start = 255;
        let mut radix_end = 255;
        let mut finished = false;

        let cbb = counts.borrow();
        let cb = cbb.inner();

        let mut bucket: &'a mut [T] = bucket;
        let mut jobs: Vec<(&'a mut [T], &[usize])> = Vec::with_capacity(threads);

        'outer: for _ in 0..threads {
            loop {
                running_total += cb[radix_start];

                if finished {
                    break 'outer;
                } else if radix_start == 0 {
                    let b: &'a mut [T] = std::mem::take(&mut bucket);
                    finished = true;
                    jobs.push((b, &cb[radix_start..=radix_end]));
                    continue 'outer;
                } else if running_total >= segment_size {
                    let b: &'a mut [T] = std::mem::take(&mut bucket);
                    let (rest, seg) = b.split_at_mut(b.len() - running_total);
                    bucket = rest;
                    let ret = (seg, &cb[radix_start..=radix_end]);

                    radix_start -= 1;
                    radix_end = radix_start;
                    running_total = 0;

                    jobs.push(ret);
                    continue 'outer;
                } else {
                    radix_start -= 1;
                }
            }
        }

        jobs.into_par_iter().for_each(|(seg, c)| {
            seg.arbitrary_chunks_mut(c)
                .for_each(|chunk| self.handle_chunk(chunk, level, parent_len, threads));
        });

        drop(cbb);
        self.cm.return_counts(counts);
    }

    #[inline]
    pub fn single_threaded_director<T>(
        &self,
        bucket: &mut [T],
        counts: Rc<RefCell<Counts>>,
        level: usize,
    ) where
        T: RadixKey + Send + Sync + Copy + 'a,
    {
        let parent_len = Some(bucket.len());
        let threads = 1;

        bucket
            .arbitrary_chunks_mut(counts.borrow().inner())
            .for_each(|chunk| self.handle_chunk(chunk, level, parent_len, threads));

        self.cm.return_counts(counts);
    }

    #[inline]
    pub fn director<T>(&self, bucket: &mut [T], counts: Rc<RefCell<Counts>>, level: usize)
    where
        T: RadixKey + Send + Sync + Copy + 'a,
    {
        if cfg!(feature = "multi-threaded") && self.multi_threaded {
            #[cfg(feature = "multi-threaded")]
            self.multi_threaded_director(bucket, counts, level);
        } else {
            self.single_threaded_director(bucket, counts, level);
        }
    }
}
