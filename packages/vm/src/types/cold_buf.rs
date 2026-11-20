//! Dual-mode `#![no_std]` cold buffer wrappers (`ColdBuf<'a, T>` and `ColdSlice<'a, T>`)
//! supporting `--no-default-features` (`!alloc`) borrowed storage and optional `alloc` `Vec`.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

/// Mutable storage backing usable with zero heap allocation (`Borrowed`) or optional `alloc` (`Owned`).
#[derive(Debug)]
pub enum ColdBuf<'a, T> {
    /// Caller-supplied mutable slice (valid under `--no-default-features`).
    Borrowed(&'a mut [T]),
    /// Heap-owned vector (enabled only under `feature = "alloc"`).
    #[cfg(feature = "alloc")]
    Owned(Vec<T>),
}

impl<'a, T> ColdBuf<'a, T> {
    /// Wraps an externally allocated mutable slice with zero heap allocation.
    #[inline]
    pub fn borrowed(slice: &'a mut [T]) -> Self {
        Self::Borrowed(slice)
    }

    /// Allocates an owned buffer filled with `len` copies of `value`.
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn owned_filled(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        Self::Owned(alloc::vec![value; len])
    }

    /// Returns the number of elements in the backing buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns `true` when the backing buffer contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an immutable view over the backing slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Borrowed(slice) => slice,
            #[cfg(feature = "alloc")]
            Self::Owned(vec) => vec.as_slice(),
        }
    }

    /// Returns a mutable view over the backing slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(slice) => slice,
            #[cfg(feature = "alloc")]
            Self::Owned(vec) => vec.as_mut_slice(),
        }
    }
}

impl<T> Deref for ColdBuf<'_, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for ColdBuf<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Immutable slice storage usable without `alloc` (`Borrowed`) or with optional `alloc` (`Owned`).
#[derive(Debug, Clone)]
pub enum ColdSlice<'a, T> {
    /// Zero-copy borrowed slice from a loaded `.dnam-bin` or static buffer.
    Borrowed(&'a [T]),
    /// Heap-owned vector (enabled only under `feature = "alloc"`).
    #[cfg(feature = "alloc")]
    Owned(Vec<T>),
}

impl<'a, T> ColdSlice<'a, T> {
    /// Returns an empty borrowed slice.
    #[inline]
    pub const fn empty() -> Self {
        Self::Borrowed(&[])
    }

    /// Wraps a borrowed slice with zero heap allocation.
    #[inline]
    pub const fn borrowed(slice: &'a [T]) -> Self {
        Self::Borrowed(slice)
    }

    /// Allocates an owned copy from an immutable slice.
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn from_slice_owned(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self::Owned(slice.to_vec())
    }

    /// Returns an immutable slice view.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Borrowed(slice) => slice,
            #[cfg(feature = "alloc")]
            Self::Owned(vec) => vec.as_slice(),
        }
    }
}

impl<T> Deref for ColdSlice<'_, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_buf_and_cold_slice_borrowed_and_owned() {
        let mut backing = [1.0_f32, 2.0, 3.0];
        let mut borrowed = ColdBuf::borrowed(&mut backing);
        assert_eq!(borrowed.len(), 3);
        assert!(!borrowed.is_empty());
        borrowed[1] = 4.5;
        assert_eq!(&*borrowed, &[1.0_f32, 4.5, 3.0]);

        let mut empty_arr: [f32; 0] = [];
        let empty = ColdBuf::borrowed(&mut empty_arr);
        assert!(empty.is_empty());

        let imm = ColdSlice::borrowed(&[7_u16, 8, 9]);
        assert_eq!(&*imm, &[7_u16, 8, 9]);
        assert!(ColdSlice::<u8>::empty().is_empty());

        #[cfg(feature = "alloc")]
        {
            let mut owned = ColdBuf::owned_filled(0.25_f32, 4);
            owned[3] = 0.75;
            assert_eq!(&*owned, &[0.25, 0.25, 0.25, 0.75]);
            let owned_slice = ColdSlice::from_slice_owned(&[10_u8, 20]);
            assert_eq!(&*owned_slice, &[10_u8, 20]);
        }
    }
}
