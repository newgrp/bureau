//! [Interning](https://en.wikipedia.org/wiki/Interning_(computer_science)) is an optimization
//! technique where logically equal values of a type are consolidated into the same memory location
//! and referred to by an ID.
//!
//! This can result in large memory savings when the active instances of the interned type often
//! contain many duplicates. It can also result in performance improvements since interned values
//! may be compared by ID rather than by their contents.
//!
//! Interning is most often used with strings, but it is also sometimes used with other types. For
//! instance, Bill Gosper's
//! [Hashlife](https://www.sciencedirect.com/science/article/abs/pii/0167278984902513) algorithm
//! interns [quadtree](https://en.wikipedia.org/wiki/Quadtree) nodes to optimize
//! [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).

pub mod shared;

#[cfg(test)]
use std::{
    array,
    fmt::Debug,
    sync::Arc,
    thread::{self, JoinHandle},
};

#[cfg(test)]
use parking_lot::Mutex;

/// An interner for values of type `T`.
///
/// `Interner` only specifies the ID type and a method to recover values from IDs. The
/// [`InternFrom`] and [`InternFromIterator`] subtraits provide methods to intern new values.
pub trait Interner<T: Eq + ?Sized> {
    /// The ID type used by this interner.
    type Id: Clone + Eq;

    /// Gets the value associated with the given ID.
    ///
    /// Implementations must guarantee that if `id1 == id2`, then `self.get(id1) == self.get(id2)`.
    /// However, the addresses of the returned references `self.get(id1)` and `self.get(id2)` may
    /// differ. This allows features like small-string optimization, where an interner may decide to
    /// store "small" strings inline in IDs and only allocate shared storage for "large" strings.
    fn get<'a>(&'a self, id: &'a Self::Id) -> &'a T;
}

/// Interns `T` values constructed from "seed" values of type `S`.
///
/// The seed type allows broad generic implementations of `InternFrom`. This is especially useful
/// for interning unsized types, which can't be passed directly by value.
pub trait InternFrom<T: Eq + ?Sized, S>: Interner<T> {
    /// Interns a `T` value constructed from the seed.
    fn intern(&self, seed: S) -> Self::Id;
}

/// Tests that the values fed into an interner are deduplicated. Compares IDs after passing them
/// through the given function.
#[cfg(test)]
pub(crate) fn uniqueness_test<I: Default + InternFrom<u8, u8>, K>(f: impl Fn(I::Id) -> K)
where
    K: Debug + Eq,
{
    let mut canonicals = array::from_fn::<Option<K>, { u8::MAX as usize + 1 }, _>(|_| None);
    let interner = I::default();
    for _ in 0..10000 {
        let val = rand::random();
        let new = f(interner.intern(val));
        match &canonicals[val as usize] {
            Some(old) => {
                assert_eq!(new, *old);
            }
            None => {
                canonicals[val as usize] = Some(new);
            }
        }
    }
}

/// Tests that the values fed into an interner are deduplicated even when interned from multiple
/// threads. Compares IDs after passing them through the given function.
#[cfg(test)]
pub(crate) fn multithreaded_test<I: 'static + Default + InternFrom<u8, u8> + Send + Sync, K>(
    f: impl 'static + Clone + Fn(I::Id) -> K + Send,
) where
    K: 'static + Debug + Eq + Send,
{
    let canonicals = Arc::new(array::from_fn::<
        Mutex<Option<K>>,
        { u8::MAX as usize + 1 },
        _,
    >(|_| Mutex::new(None)));
    let interner = Arc::new(I::default());
    array::from_fn::<JoinHandle<()>, 1000, _>(|_| {
        let f = f.clone();
        let canonicals = canonicals.clone();
        let interner = interner.clone();
        thread::spawn(move || {
            for _ in 0..10000 {
                let val = rand::random();
                let new = f(interner.intern(val));
                let mut entry = canonicals[val as usize].lock();
                match &*entry {
                    Some(old) => {
                        assert_eq!(new, *old);
                    }
                    None => {
                        *entry = Some(new);
                    }
                }
            }
        })
    })
    .into_iter()
    .for_each(|j| j.join().expect("Failed to join thread"))
}
