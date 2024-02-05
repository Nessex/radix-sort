use crate::RadixKey;

impl RadixKey for u8 {
    const LEVELS: usize = 1;

    #[inline(always)]
    fn get_level(&self, _: usize) -> u8 {
        *self
    }
}

impl RadixKey for u16 {
    const LEVELS: usize = 2;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

impl RadixKey for u32 {
    const LEVELS: usize = 4;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

impl RadixKey for u64 {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

impl RadixKey for u128 {
    const LEVELS: usize = 16;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

#[cfg(target_pointer_width = "16")]
impl RadixKey for usize {
    const LEVELS: usize = 2;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

#[cfg(target_pointer_width = "32")]
impl RadixKey for usize {
    const LEVELS: usize = 4;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl RadixKey for usize {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS);

        if cfg!(target_endian = "little") {
            unsafe {
                (self as *const Self as *const u8)
                    .wrapping_add(level)
                    .read()
            }
        } else {
            (self >> (level * 8)) as u8
        }
    }
}

impl<const N: usize> RadixKey for [u8; N] {
    const LEVELS: usize = N;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        self[level]
    }
}

impl RadixKey for i8 {
    const LEVELS: usize = 1;

    #[inline(always)]
    fn get_level(&self, _: usize) -> u8 {
        (*self ^ i8::MIN) as u8
    }
}

impl RadixKey for i16 {
    const LEVELS: usize = 2;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ i16::MIN) >> (level * 8)) as u8
    }
}

impl RadixKey for i32 {
    const LEVELS: usize = 4;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ i32::MIN) >> (level * 8)) as u8
    }
}

impl RadixKey for i64 {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ i64::MIN) >> (level * 8)) as u8
    }
}

impl RadixKey for i128 {
    const LEVELS: usize = 16;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ i128::MIN) >> (level * 8)) as u8
    }
}

#[cfg(target_pointer_width = "16")]
impl RadixKey for isize {
    const LEVELS: usize = 2;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ isize::MIN) >> (level * 8)) as u8
    }
}

#[cfg(target_pointer_width = "32")]
impl RadixKey for isize {
    const LEVELS: usize = 4;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ isize::MIN) >> (level * 8)) as u8
    }
}

#[cfg(target_pointer_width = "64")]
impl RadixKey for isize {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        ((self ^ isize::MIN) >> (level * 8)) as u8
    }
}

impl RadixKey for f32 {
    const LEVELS: usize = 4;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        let mut s = self.to_bits() as i32;

        s ^= (((s >> 31) as u32) >> 1) as i32;

        ((s ^ i32::MIN) >> (level * 8)) as u8
    }
}

impl RadixKey for f64 {
    const LEVELS: usize = 8;

    #[inline(always)]
    fn get_level(&self, level: usize) -> u8 {
        let mut s = self.to_bits() as i64;
        s ^= (((s >> 63) as u64) >> 1) as i64;

        ((s ^ i64::MIN) >> (level * 8)) as u8
    }
}
