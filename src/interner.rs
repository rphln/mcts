use std::{hash::Hash, rc::Rc};

use rustc_hash::FxHashMap;

/// A stable identifier for an interned value.
///
/// Handles are assigned in insertion order and can later be used to resolve the
/// corresponding value.
pub type Interned = u16;

/// Stores unique values and assigns each one a stable handle.
///
/// The same value always maps to the same handle. Interned values are retained
/// for the lifetime of the interner, and lookups by handle preserve insertion
/// order.
#[derive(Clone, Debug)]
pub struct Interner<T> {
    values: Vec<Rc<T>>,
    index: FxHashMap<Rc<T>, Interned>,
}

impl<T> Interner<T> {
    /// Creates a new, empty interner.
    pub fn new() -> Self {
        Self { values: Vec::default(), index: FxHashMap::default() }
    }

    /// Returns the handle for `value`, inserting it if it is not already
    /// present.
    ///
    /// If `value` has been interned before, its existing handle is returned.
    /// Otherwise, `value` is stored and assigned the next available handle.
    ///
    /// # Panics
    ///
    /// Panics if the number of interned values exceeds `u16::MAX`.
    pub fn get_or_intern(&mut self, value: T) -> Interned
    where
        T: Hash + Eq,
    {
        if let Some(&handle) = self.index.get(&value) {
            return handle;
        }

        let value = Rc::new(value);
        let handle = self.values.len().try_into().expect("interner capacity exceeded");

        self.values.push(value.clone());
        self.index.insert(value, handle);

        handle
    }

    /// Returns the interned value associated with `handle`, if it exists.
    #[inline]
    pub fn resolve(&self, handle: Interned) -> Option<&T> {
        self.values.get(handle as usize).map(Rc::as_ref)
    }

    /// Returns an iterator over all interned values in insertion order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.values.iter().map(Rc::as_ref)
    }
}

impl<T> Default for Interner<T> {
    fn default() -> Self {
        Self::new()
    }
}
