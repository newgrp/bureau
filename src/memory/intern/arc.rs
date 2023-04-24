use core::{
    alloc::Layout,
    borrow::Borrow,
    fmt::{self, Debug, Formatter},
    hash::{BuildHasher, Hash, Hasher},
    mem,
    ops::Deref,
    ptr,
};
use std::collections::hash_map::RandomState;

use alloc::{
    alloc as _alloc,
    boxed::Box,
    sync::{Arc, Weak},
};
use dashmap::{DashSet, SharedValue};
use hashbrown::hash_map::RawEntryMut;

use crate::memory::ByAddress;

use super::{InternFrom, Interner};

/// Adds the `hash_one()` method that is currently experimental in the standard library.
trait HashOne: BuildHasher {
    fn hash_one<T: Hash + ?Sized>(&self, x: &T) -> u64;
}
impl<S: BuildHasher> HashOne for S {
    fn hash_one<T: Hash + ?Sized>(&self, x: &T) -> u64 {
        let mut h = self.build_hasher();
        x.hash(&mut h);
        h.finish()
    }
}

/// Internal representation behind [`Arctern`].
#[repr(C)]
struct Inner<T: Eq + Hash + ?Sized, S: BuildHasher> {
    interner: Arc<Interior<T, S>>,
    data: T,
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Inner<T, S> {
    /// Creates a partially initialized `Inner<T>` allocated with the global allocator. The
    /// `interner` field is set, but the `data` is uninitialized memory.
    ///
    /// Returns a pointer to the allocated `Inner` and the offset of `data` within the allocation.
    ///
    /// # Panics
    ///
    /// If layout computations overflow, allocation fails, or if the returned allocation extends
    /// beyond `isize::MAX` bytes.
    fn partial_for_layout(data_layout: Layout, interner: Arc<Interior<T, S>>) -> (*mut u8, usize) {
        let (layout, offset) = Layout::new::<Arc<Interior<T, S>>>()
            .extend(data_layout)
            .unwrap();
        // Needed for compatibility with `#[repr(C)]`.
        let layout = layout.pad_to_align();

        // SAFETY: We know that `layout.size() > 0` because `layout` contains an
        // `Arc<Interior<Self, S>>`, which is positively sized.
        let raw = unsafe { _alloc::alloc(layout) };
        if raw.is_null() {
            _alloc::handle_alloc_error(layout);
        }
        // Ensure that all offsets within the buffer won't overflow `isize`.
        let _: isize = (raw as usize)
            .checked_add(layout.size())
            .unwrap()
            .try_into()
            .unwrap();

        // SAFETY: The allocation layout was computed to start with an `Arc<Interior<T, S>>`.
        unsafe { ptr::write(raw as *mut Arc<Interior<T, S>>, interner) };
        (raw, offset)
    }
}

impl<T: Debug + Eq + Hash + ?Sized, S: BuildHasher> Debug for Inner<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.data.fmt(f)
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Drop for Inner<T, S> {
    fn drop(&mut self) {
        #[allow(unstable_name_collisions)]
        let hash = self.interner.hasher.hash_one(&self.data);
        self.interner.remove_by_pointer(hash, self as *const Self);
    }
}

/// A value that can be used as an input for interning. Used to determine if the value already
/// exists in the interner and construct a new value if it doesn't.
trait Query<T: Eq + Hash + ?Sized>
where
    T: Borrow<Self::Borrowable>,
{
    /// A common borrowable type between `Self` and `T`.
    type Borrowable: Eq + Hash + ?Sized;

    /// Borrows for lookup in the interner.
    fn borrowable(&self) -> &Self::Borrowable;

    /// Consumes the query and returns a new entry for the interner.
    fn into_arc_inner<S: BuildHasher>(self, interner: Arc<Interior<T, S>>) -> Arc<Inner<T, S>>;
}

impl<T: Eq + Hash> Query<T> for T {
    type Borrowable = T;

    fn borrowable(&self) -> &Self::Borrowable {
        self
    }

    fn into_arc_inner<S: BuildHasher>(self, interner: Arc<Interior<T, S>>) -> Arc<Inner<T, S>> {
        Arc::new(Inner {
            interner,
            data: self,
        })
    }
}

#[cfg(feature = "unstable-ptr-metadata")]
impl<T: Eq + Hash, I: Borrow<[T]> + IntoIterator<Item = T>> Query<[T]> for I
where
    I::IntoIter: ExactSizeIterator,
{
    type Borrowable = [T];

    fn borrowable(&self) -> &Self::Borrowable {
        self.borrow()
    }

    fn into_arc_inner<S: BuildHasher>(self, interner: Arc<Interior<[T], S>>) -> Arc<Inner<[T], S>> {
        let iter = self.into_iter();
        let len = iter.len();

        let (raw, offset) = Inner::partial_for_layout(Layout::array::<T>(len).unwrap(), interner);
        if mem::size_of::<T>() > 0 {
            // SAFETY: Checked by `partial_for_layout()`.
            let base = unsafe { raw.offset(offset as isize) } as *mut T;
            for (i, value) in iter.enumerate() {
                // SAFETY: Checked by `partial_for_layout()`.
                unsafe { ptr::write(base.offset(i as isize), value) };
            }
        }

        // This cast relies on the current implementation of pointer metadata, which is why it is
        // gated by the `unstable-ptr-metadata` feature.
        let inner = ptr::slice_from_raw_parts_mut(raw, len) as *mut Inner<[T], S>;
        Arc::from(unsafe { Box::from_raw(inner) })
    }
}

#[cfg(feature = "unstable-ptr-metadata")]
impl<A: AsRef<str>> Query<str> for A {
    type Borrowable = str;

    fn borrowable(&self) -> &Self::Borrowable {
        self.as_ref()
    }

    fn into_arc_inner<S: BuildHasher>(self, interner: Arc<Interior<str, S>>) -> Arc<Inner<str, S>> {
        let (raw, offset) = Inner::partial_for_layout(
            Layout::from_size_align(self.as_ref().len(), 1).unwrap(),
            interner,
        );
        // SAFETY: Checked by `partial_for_layout()`.
        let base = unsafe { raw.offset(offset as isize) };
        // SAFETY: Checked by `partial_for_layout()`.
        unsafe { ptr::copy_nonoverlapping(self.as_ref().as_ptr(), base, self.as_ref().len()) }

        // This cast relies on the current implementation of pointer metadata, which is why it is
        // gated by the `unstable-ptr-metadata` feature.
        let inner = ptr::slice_from_raw_parts_mut(raw, self.as_ref().len()) as *mut Inner<str, S>;
        Arc::from(unsafe { Box::from_raw(inner) })
    }
}

/// An interned `T` value that removes itself from the interner when the last reference is dropped.
///
/// Equality comparisons on `Arctern` values use pointer comparison.
#[cfg(any(feature = "std", test))]
pub struct Arctern<T: Eq + Hash + ?Sized, S: BuildHasher = RandomState> {
    inner: ByAddress<Arc<Inner<T, S>>>,
}

/// An interned `T` value that removes itself from the interner when the last reference is dropped.
///
/// Equality comparisons on `Arctern` values use pointer comparison.
#[cfg(not(any(feature = "std", test)))]
pub struct Arctern<T: Eq + Hash + ?Sized, S: BuildHasher> {
    inner: ByAddress<Arc<Inner<T, S>>>,
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Arctern<T, S> {
    /// Constructs a new `Arctern` from an inner value.
    fn new(inner: Arc<Inner<T, S>>) -> Arctern<T, S> {
        Arctern {
            inner: ByAddress(inner),
        }
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> AsRef<T> for Arctern<T, S> {
    fn as_ref(&self) -> &T {
        &self.inner.data
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Clone for Arctern<T, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Debug + Eq + Hash + ?Sized, S: BuildHasher> Debug for Arctern<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.inner.data.fmt(f)
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Deref for Arctern<T, S> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Eq for Arctern<T, S> {}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> PartialEq for Arctern<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// The key type stored in [`Arcterner`].
struct Key<T: Eq + Hash + ?Sized, S: BuildHasher> {
    /// A weak pointer to the contained value so that we don't create a reference count cycle.
    ///
    /// The key will be removed when the `Inner` value is dropped, so we don't leak memory from the
    /// cycle of allocation reference counts.
    weak: Weak<Inner<T, S>>,
    /// Cached since we won't always have access to the raw data.
    hash: u64,
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Key<T, S> {
    /// Constructs a new key using the given hasher to compute the hash of the value.
    fn new(inner: &Arc<Inner<T, S>>, hasher: &S) -> Key<T, S> {
        Key {
            weak: Arc::downgrade(inner),
            hash: {
                #[allow(unstable_name_collisions)]
                hasher.hash_one(&inner.data)
            },
        }
    }

    /// Upgrades to an [`Arctern`], or returns [`None`] if the value is being removed.
    fn upgrade(&self) -> Option<Arctern<T, S>> {
        Some(Arctern::new(self.weak.upgrade()?))
    }
}

impl<T: Debug + Eq + Hash + ?Sized, S: BuildHasher> Debug for Key<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(arc) = self.weak.upgrade() {
            arc.fmt(f)
        } else {
            write!(f, "(evicted)")
        }
    }
}

/// Keys that cannot be upgraded are consider inequal to all keys (including themselves).
///
/// Technically, this means that [`Key`] equality isn't actually reflexive. However, this [`Eq`]
/// implementation will only ever be invoked during hash set growing and shrinking, where the same
/// key won't be in the map twice.
impl<T: Eq + Hash + ?Sized, S: BuildHasher> Eq for Key<T, S> {}

/// Implements [`Hash`] by feeding the chached hash into the new hasher.
///
/// In general, hashing is not idempotent (i.e., `hash(hash(value)) != hash(value)`), so re-hashing
/// can cause logic errors. However, this [`Hash`] implementation will only ever be consumed by
/// [`NoHash`], where hashing a `u64` is a no-op.
impl<T: Eq + Hash + ?Sized, S: BuildHasher> Hash for Key<T, S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

/// Keys that cannot be upgraded are consider inequal to all keys (including themselves).
impl<T: Eq + Hash + ?Sized, S: BuildHasher> PartialEq for Key<T, S> {
    fn eq(&self, other: &Self) -> bool {
        (|| Some(self.weak.upgrade()?.data == other.weak.upgrade()?.data))() == Some(true)
    }
}

/// A [`BuildHasher`] that builds [`NoHasher`]s.
#[derive(Clone, Copy, Debug, Default)]
struct NoHash;

impl BuildHasher for NoHash {
    type Hasher = NoHasher;

    fn build_hasher(&self) -> Self::Hasher {
        NoHasher::default()
    }
}

/// A [`Hasher`] that hashes `u64`s into their values.
///
/// Panics if given a value other than a `u64`.
#[derive(Debug, Default)]
struct NoHasher(u64);

impl Hasher for NoHasher {
    fn write(&mut self, _: &[u8]) {
        panic!("Raw `write()` called on `NoHasher`")
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// The contents of an [`Arcterner`].
///
/// Separated from [`Arcterner`] so that users don't have to write `Arc<Arcterner<T>>` everywhere.
struct Interior<T: Eq + Hash + ?Sized, S: BuildHasher> {
    hasher: S,
    table: DashSet<Key<T, S>, NoHash>,
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Interior<T, S> {
    /// Removes a key from the hash set via its hash and pointer value.
    fn remove_by_pointer(&self, hash: u64, pointer: *const Inner<T, S>) {
        let mut shard = self.table.shards()[self.table.determine_shard(hash as usize)].write();
        if let RawEntryMut::Occupied(o) = shard
            .raw_entry_mut()
            .from_hash(hash, |key| ptr::eq(key.weak.as_ptr(), pointer))
        {
            o.remove();
        }
    }
}

impl<T: Debug + Eq + Hash + ?Sized, S: BuildHasher> Debug for Interior<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.table.fmt(f)
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher + Default> Default for Interior<T, S> {
    fn default() -> Self {
        Self {
            hasher: Default::default(),
            table: Default::default(),
        }
    }
}

/// An interner that removes unreferenced values.
#[cfg(any(feature = "std", test))]
pub struct Arcterner<T: Eq + Hash + ?Sized, S: BuildHasher = RandomState> {
    interior: Arc<Interior<T, S>>,
}

/// An interner that removes unreferenced values.
#[cfg(not(any(feature = "std", test)))]
pub struct Arcterner<T: Eq + Hash + ?Sized, S: BuildHasher> {
    interior: Arc<Interior<T, S>>,
}

#[cfg(any(feature = "std", test))]
impl<T: Eq + Hash + ?Sized> Arcterner<T> {
    /// Constructs an empty interner.
    pub fn new() -> Arcterner<T> {
        Arcterner::default()
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Arcterner<T, S> {
    /// Interns a value using the [`Query`] trait.
    fn intern_by_query<Q: Query<T>>(&self, query: Q) -> Arctern<T, S>
    where
        T: Borrow<Q::Borrowable>,
    {
        #[allow(unstable_name_collisions)]
        let hash = self.interior.hasher.hash_one(query.borrowable());
        let mut shard = self.interior.table.shards()
            [self.interior.table.determine_shard(hash as usize)]
        .write();

        // We need to persist a refcount to the `Inner` as soon as we match on it, so we write an
        // `Arc` to it here if we match.
        let mut arctern = None;
        match shard.raw_entry_mut().from_hash(hash, |key| {
            if let Some(arc) = key.upgrade() {
                if Borrow::<Q::Borrowable>::borrow(&*arc) == query.borrowable() {
                    arctern = Some(arc);
                    return true;
                }
            }
            false
        }) {
            // This should never panic, since the branch should only be hit if we found a match.
            RawEntryMut::Occupied(_) => arctern.unwrap(),
            RawEntryMut::Vacant(v) => {
                let inner = query.into_arc_inner(self.interior.clone());
                let key = Key::new(&inner, &self.interior.hasher);
                v.insert_hashed_nocheck(hash, key, SharedValue::new(()));
                Arctern::new(inner)
            }
        }
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Clone for Arcterner<T, S> {
    fn clone(&self) -> Self {
        Self {
            interior: self.interior.clone(),
        }
    }
}

impl<T: Debug + Eq + Hash + ?Sized, S: BuildHasher> Debug for Arcterner<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.interior.fmt(f)
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher + Default> Default for Arcterner<T, S> {
    fn default() -> Self {
        Self {
            interior: Default::default(),
        }
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher> Interner<T> for Arcterner<T, S> {
    type Id = Arctern<T, S>;

    fn get<'a>(&'a self, id: &'a Self::Id) -> &'a T {
        &*id
    }
}

impl<T: Eq + Hash, S: BuildHasher, U: Eq + Hash> InternFrom<T, U> for Arcterner<T, S>
where
    T: From<U>,
{
    fn intern(&self, seed: U) -> Self::Id {
        self.intern_by_query(T::from(seed))
    }
}

#[cfg(feature = "unstable-ptr-metadata")]
impl<T: Eq + Hash, S: BuildHasher, I: Borrow<[T]> + IntoIterator<Item = T>> InternFrom<[T], I>
    for Arcterner<[T], S>
where
    I::IntoIter: ExactSizeIterator,
{
    fn intern(&self, seed: I) -> Self::Id {
        self.intern_by_query(seed)
    }
}

#[cfg(feature = "unstable-ptr-metadata")]
impl<S: BuildHasher, A: AsRef<str>> InternFrom<str, A> for Arcterner<str, S> {
    fn intern(&self, seed: A) -> Self::Id {
        self.intern_by_query(&seed)
    }
}

#[cfg(test)]
mod tests {
    use core::{
        array,
        borrow::Borrow,
        convert, mem,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use std::thread::{self, JoinHandle};

    use alloc::{boxed::Box, collections::VecDeque, sync::Arc};
    use intern::InternFrom;
    use once_cell::sync::Lazy;
    use parking_lot::Mutex;

    use crate::memory::{intern, Arctern};

    use super::Arcterner;

    #[test]
    fn uniqueness_test() {
        intern::uniqueness_test::<Arcterner<u8>, _>(convert::identity)
    }

    #[test]
    fn multithreaded_test() {
        intern::multithreaded_test::<Arcterner<u8>, _>(convert::identity)
    }

    #[test]
    fn multithreaded_drop_test() {
        static DROP_COUNTERS: Lazy<[AtomicUsize; u8::MAX as usize + 1]> =
            Lazy::new(|| array::from_fn(|_| AtomicUsize::new(0)));

        #[derive(Eq, Hash, PartialEq)]
        struct DropCounted(u8);
        impl Borrow<u8> for DropCounted {
            fn borrow(&self) -> &u8 {
                &self.0
            }
        }
        impl Drop for DropCounted {
            fn drop(&mut self) {
                DROP_COUNTERS[self.0 as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
        impl From<u8> for DropCounted {
            fn from(value: u8) -> Self {
                DropCounted(value)
            }
        }

        const MAX_ALIVE: usize = 8;
        let alive = Arc::new(Mutex::new(VecDeque::<Arctern<DropCounted>>::new()));
        let removal_counters = Arc::new(
            array::from_fn::<AtomicUsize, { u8::MAX as usize + 1 }, _>(|_| AtomicUsize::new(0)),
        );

        let interner = Arc::new(Arcterner::new());
        array::from_fn::<JoinHandle<()>, 10, _>(|_| {
            let alive = alive.clone();
            let removal_counters = removal_counters.clone();
            let interner = interner.clone();
            thread::spawn(move || {
                for _ in 0..1000 {
                    let val = rand::random::<u8>();
                    let mut alive = alive.lock();
                    // We have to check for existence under the lock before we intern, or else we'll
                    // have to drop an extra time.
                    if alive.iter().any(|arc| arc.0 == val) {
                        continue;
                    }
                    // We have to insert under the lock, or else other threads may be keeping
                    // elements we don't know about alive.
                    let arc = interner.intern(val);
                    alive.push_front(arc);
                    if alive.len() > MAX_ALIVE {
                        let removed = alive.pop_back().unwrap();
                        removal_counters[(*removed).0 as usize].fetch_add(1, Ordering::Relaxed);
                        mem::drop(removed);
                    }
                }
            })
        })
        .into_iter()
        .for_each(|j| j.join().expect("Failed to join thread"));

        let actual = DROP_COUNTERS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect::<Box<[_]>>();
        let expected = removal_counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect::<Box<[_]>>();
        assert_eq!(actual, expected);
    }
}
