use std::hash::Hash;

use indexmap::IndexSet;
use rustc_hash::FxBuildHasher;

/// Quick-and-dirty interner backed by `IndexSet`.
#[derive(Clone, Debug)]
pub struct Interner<T>(IndexSet<T, FxBuildHasher>);

/// Opaque handle identifying an interned value.
///
/// Values of this type are cheap to copy and compare. They are only meaningful
/// in the context of the `Interner` instance that produced them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Interned(u16);

impl<T: Hash + Eq> Interner<T> {
    /// Interns `value`, returning its handle.
    ///
    /// # Panics
    ///
    /// Panics if the interner exceeds its maximum capacity (currently 2^16).
    #[inline]
    #[track_caller]
    pub fn intern(&mut self, value: T) -> Interned {
        let (index, _exists) = self.0.insert_full(value);
        Interned(index.try_into().unwrap())
    }

    /// Returns the value corresponding to `handle`.
    ///
    /// # Panics
    ///
    /// Panics if `handle` is out-of-bounds.
    #[inline]
    #[track_caller]
    pub fn lookup(&self, handle: Interned) -> &T {
        &self.0[handle.0 as usize]
    }
}

impl<T> Default for Interner<T> {
    fn default() -> Self {
        Self(IndexSet::with_hasher(FxBuildHasher))
    }
}
