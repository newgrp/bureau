use core::{
    borrow::Borrow,
    fmt::{self, Debug, Formatter},
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    mem,
};
#[cfg(any(feature = "std", test))]
use std::collections::hash_map::RandomState;

use dashmap::{DashMap, SharedValue};

use crate::memory::{AllocateFrom, Allocator};

use super::{InternFrom, Interner};

/// We need this wrapper to avoid conflicting with the blanket `T: Borrow<T>` impl.
#[derive(Eq, Hash, PartialEq)]
#[repr(transparent)]
struct BorrowWrap<Q>(Q);

impl<Q> BorrowWrap<Q> {
    fn new<'a>(q: &'a Q) -> &'a BorrowWrap<Q> {
        // Safe because of `#[repr(transparent)]`.
        unsafe { mem::transmute(q) }
    }
}

/// A wrapper around `P` that implements [`Eq`] and [`Hash`]. Also borrows as [`BorrowWrap<Q>`]
/// whenever `P: Borrow<T>` and `T: Borrow<Q>`.
struct Reborrow<P, T: ?Sized>(P, PhantomData<T>);

impl<P, T: ?Sized> Reborrow<P, T> {
    /// Constructs a new `Reborrow`.
    fn new(pointer: P) -> Reborrow<P, T> {
        Reborrow(pointer, PhantomData)
    }
}

impl<P, T: ?Sized, Q> Borrow<BorrowWrap<Q>> for Reborrow<P, T>
where
    P: Borrow<T>,
    T: Borrow<Q>,
{
    fn borrow(&self) -> &BorrowWrap<Q> {
        BorrowWrap::new(self.0.borrow().borrow())
    }
}

impl<P: Debug, T: ?Sized> Debug for Reborrow<P, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Format transparently.
        self.0.fmt(f)
    }
}

impl<P: Eq, T: ?Sized> Eq for Reborrow<P, T> {}

impl<P: Hash, T: ?Sized> Hash for Reborrow<P, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<P: PartialEq, T: ?Sized> PartialEq for Reborrow<P, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// A thread-safe interner generic over backing storage.
///
/// Interned values are stored in an allocator of type `A` and returned as IDs of type
/// `<A::Pointer>`. Applications may choose to immediately wrap these return values in
/// [`ByAddress`](crate::memory::ByAddress), but they should be aware of the pitfalls documented on
/// that type. Care must also be taken to choose an allocator where [`Clone`]d pointers point to the
/// same address as the pointer they were cloned from.
///
/// Saved values are not forgotten during the interner's lifetime, so long-lived `SharedInterner`s
/// may effectively leak memory.
///
/// `SharedInterner` uses locks internally, so insertions may deadlock if the insertion process
/// somehow recurses into calling another insertion. This will not happen in most use cases.
#[cfg(any(feature = "std", test))]
pub struct SharedInterner<T: ?Sized, A: Allocator<T>, S = RandomState> {
    storage: A,
    table: DashMap<Reborrow<A::Pointer, T>, (), S>,
}

/// A thread-safe interner generic over backing storage.
///
/// Interned values are stored in an allocator of type `A` and returned as values of type
/// `<A::Pointer>`. Applications may choose to immediately wrap these return values in
/// [`ByAddress`](crate::memory::ByAddress), but they should be aware of the pitfalls documented on
/// that type. Care must also be taken to choose an allocator where [`Clone`]d pointers point to the
/// same address as the pointer they were cloned from.
///
/// Saved values are not forgotten during the interner's lifetime, so long-lived `SharedInterner`s
/// may effectively leak memory.
///
/// `SharedInterner` uses locks internally, so insertions may deadlock if the insertion process
/// somehow recurses into calling another insertion. This will not happen in most use cases.
#[cfg(not(any(feature = "std", test)))]
pub struct SharedInterner<T: ?Sized, A: Allocator<T>, S> {
    storage: A,
    table: DashMap<Reborrow<A::Pointer, T>, (), S>,
}

#[cfg(feature = "std")]
impl<T: ?Sized, A: Allocator<T>> SharedInterner<T, A>
where
    A::Pointer: Eq + Hash,
{
    /// Constructs an empty interner.
    pub fn new() -> SharedInterner<T, A>
    where
        A: Default,
    {
        Default::default()
    }

    /// Constructs an empty interner using the given hasher type.
    pub fn new_in<S: BuildHasher + Clone + Default>() -> SharedInterner<T, A, S>
    where
        A: Default,
    {
        Default::default()
    }

    /// Constructs an empty interner that uses the given backing storage.
    pub fn with_storage(storage: A) -> SharedInterner<T, A> {
        SharedInterner {
            storage,
            table: DashMap::default(),
        }
    }

    /// Constructs an empty interner that uses the given backing storage and hasher type.
    pub fn with_storage_in<S: BuildHasher + Clone + Default>(
        storage: A,
    ) -> SharedInterner<T, A, S> {
        SharedInterner {
            storage,
            table: DashMap::default(),
        }
    }
}

impl<T: ?Sized, A: Allocator<T>, S: BuildHasher + Clone> SharedInterner<T, A, S>
where
    A::Pointer: Eq + Hash,
{
    /// Constructs an empty interner using the given hasher.
    pub fn with_hasher(hasher: S) -> SharedInterner<T, A, S>
    where
        A: Default,
    {
        SharedInterner {
            storage: Default::default(),
            table: DashMap::with_hasher(hasher),
        }
    }

    /// Constructs an empty interner that uses the given backing storage and hasher.
    pub fn with_storage_and_hasher(storage: A, hasher: S) -> SharedInterner<T, A, S> {
        SharedInterner {
            storage,
            table: DashMap::with_hasher(hasher),
        }
    }
}

impl<T: ?Sized, A: Allocator<T> + Debug, S: BuildHasher + Clone> Debug for SharedInterner<T, A, S>
where
    A::Pointer: Debug + Eq + Hash,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SharedInterner")
            .field("storage", &self.storage)
            .field("table", &self.table)
            .finish()
    }
}

impl<T: ?Sized, A: Allocator<T> + Default, S: BuildHasher + Clone + Default> Default
    for SharedInterner<T, A, S>
where
    A::Pointer: Eq + Hash,
{
    fn default() -> Self {
        SharedInterner {
            storage: A::default(),
            table: DashMap::default(),
        }
    }
}

impl<T: Eq + ?Sized, A: Allocator<T>, S> Interner<T> for SharedInterner<T, A, S>
where
    A::Pointer: Clone + Eq,
{
    type Id = A::Pointer;

    fn get<'a>(&'a self, id: &'a Self::Id) -> &'a T {
        &*id
    }
}

/// The [`Borrow`] impls ensure that `Q::hash()` can be used as a proxy for `A::Pointer::hash()`,
/// but only if `T` (the intermediate type of the two [`Borrow`] impls) is [`Hash`].
impl<
        Q: Eq + Hash,
        T: Borrow<Q> + Eq + Hash + ?Sized,
        A: AllocateFrom<T, Q>,
        S: BuildHasher + Clone,
    > InternFrom<T, Q> for SharedInterner<T, A, S>
where
    A::Pointer: Borrow<T> + Clone + Eq + Hash,
{
    fn intern(&self, seed: Q) -> Self::Id {
        // We need to perform some careful hashmap surgery atomically, so look up the shard and lock
        // it manually.
        let key = BorrowWrap::new(&seed);
        let mut shard = self.table.shards()[self.table.determine_map(key)].write();
        if let Some((reborrow, _)) = shard.get_key_value(key) {
            return reborrow.0.clone();
        }
        let pointer = self.storage.alloc(seed);
        shard.insert(Reborrow::new(pointer.clone()), SharedValue::new(()));
        pointer
    }
}

/// A `SharedInterner` can also be used as an allocator in its own right.
impl<T: ?Sized, A: Allocator<T>, S> Allocator<T> for SharedInterner<T, A, S> {
    type Pointer = A::Pointer;
}

impl<
        Q: Eq + Hash,
        T: Borrow<Q> + Eq + Hash + ?Sized,
        A: AllocateFrom<T, Q>,
        S: BuildHasher + Clone,
    > AllocateFrom<T, Q> for SharedInterner<T, A, S>
where
    A::Pointer: Borrow<T> + Clone + Eq + Hash,
{
    fn alloc(&self, seed: Q) -> Self::Pointer {
        self.intern(seed)
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::{
        intern::{self},
        ArcAllocator, ByAddress, RcAllocator,
    };

    use super::SharedInterner;

    #[test]
    fn uniqueness_test() {
        intern::uniqueness_test::<SharedInterner<u8, RcAllocator>, _>(ByAddress)
    }

    #[test]
    fn multithreaded_test() {
        intern::multithreaded_test::<SharedInterner<u8, ArcAllocator>, _>(ByAddress)
    }
}
