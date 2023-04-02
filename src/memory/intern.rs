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
    fn get(&self, id: &Self::Id) -> &T;
}

/// Interns `T` values constructed from "seed" values of type `S`.
///
/// The seed type allows broad generic implementations of `InternFrom`. This is especially useful
/// for interning unsized types, which can't be passed directly by value.
pub trait InternFrom<T: Eq + ?Sized, S>: Interner<T> {
    /// Interns a `T` value constructed from the seed.
    fn intern(&self, seed: S) -> Self::Id;
}

/// Interns `[T]` slices constructed from iterators.
///
/// See [`super::AllocatorFromIterator`] for an explanation of why this trait is separate from
/// [`InternFrom`].
pub trait InternFromIterator<T: Eq>: Interner<[T]> {
    /// Interns a `[T]` slice constructed from the iterator.
    fn intern_from_iter(&self, iter: impl IntoIterator<Item = T>) -> Self::Id;
}
