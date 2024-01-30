use crate::counts::{CountManager, CountMeta, Counts};
use crate::RadixKey;
#[cfg(feature = "multi-threaded")]
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[inline]
pub const fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[inline]
pub fn get_tile_counts<T>(
    cm: &CountManager,
    bucket: &[T],
    tile_size: usize,
    level: usize,
) -> (Vec<Counts>, bool)
where
    T: RadixKey + Copy + Sized + Send + Sync,
{
    #[cfg(feature = "work_profiles")]
    println!("({}) TILE_COUNT", level);

    let num_tiles = cdiv(bucket.len(), tile_size);
    let mut tiles: Vec<Counts> = vec![Counts::new(); num_tiles];
    let mut meta: Vec<CountMeta> = vec![CountMeta::default(); num_tiles];

    #[cfg(feature = "multi-threaded")]
    bucket
        .par_chunks(tile_size)
        .zip(tiles.par_iter_mut())
        .zip(meta.par_iter_mut())
        .for_each(|((chunk, counts), meta)| {
            cm.count_into(counts, meta, chunk, level);
        });

    #[cfg(not(feature = "multi-threaded"))]
    bucket
        .chunks(tile_size)
        .zip(tiles.par_iter_mut())
        .zip(meta.par_iter_mut())
        .for_each(|((chunk, counts), meta)| {
            cm.count_into(counts, meta, chunk, level);
        });

    let mut all_sorted = true;

    if tiles.len() == 1 {
        // If there is only one tile, we already have a flag for if it is sorted
        all_sorted = meta[0].already_sorted;
    } else {
        // Check if any of the tiles, or any of the tile boundaries are unsorted
        for w in meta.windows(2) {
            let left = &w[0];
            let right = &w[1];
            if !left.already_sorted || !right.already_sorted || right.first < left.last {
                all_sorted = false;
                break;
            }
        }
    }

    (tiles, all_sorted)
}

#[inline]
pub fn aggregate_tile_counts(cm: &CountManager, tile_counts: &[Counts]) -> Rc<RefCell<Counts>> {
    let out = cm.get_empty_counts();
    let mut counts = out.borrow_mut();

    for tile in tile_counts.iter() {
        for i in 0..256usize {
            counts.add(i, tile[i]);
        }
    }

    drop(counts);

    out
}

#[inline]
pub fn is_homogenous(counts: &Counts) -> bool {
    let mut seen = false;
    for c in counts.into_iter() {
        if *c > 0 {
            if seen {
                return false;
            } else {
                seen = true;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use crate::counts::CountManager;
    use crate::utils::get_tile_counts;

    #[test]
    pub fn test_get_tile_counts_correctly_marks_already_sorted_single_tile() {
        let cm = CountManager::default();
        let mut data: Vec<u8> = vec![0, 5, 2, 3, 1];

        let (_counts, already_sorted) = get_tile_counts(&cm, &mut data, 5, 0);
        assert_eq!(already_sorted, false);

        let mut data: Vec<u8> = vec![0, 0, 1, 1, 2];

        let (_counts, already_sorted) = get_tile_counts(&cm, &mut data, 5, 0);
        assert_eq!(already_sorted, true);
    }

    #[test]
    pub fn test_get_tile_counts_correctly_marks_already_sorted_multiple_tiles() {
        let cm = CountManager::default();
        let mut data: Vec<u8> = vec![0, 5, 2, 3, 1];

        let (_counts, already_sorted) = get_tile_counts(&cm, &mut data, 2, 0);
        assert_eq!(already_sorted, false);

        let mut data: Vec<u8> = vec![0, 0, 1, 1, 2];

        let (_counts, already_sorted) = get_tile_counts(&cm, &mut data, 2, 0);
        assert_eq!(already_sorted, true);
    }
}
