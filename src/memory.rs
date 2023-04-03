mod intern;
mod pointer;

use core::ops::Deref;

use alloc::{boxed::Box, rc::Rc, sync::Arc};
pub use intern::{shared::SharedInterner, InternFrom, InternFromIterator, Interner};
pub use pointer::ByAddress;

/// An allocator for `T` values that returns pointers of a particular type.
///
/// `Allocator` only specifies the type of the pointer returned from allocations. The
/// [`AllocateFrom`] and [`AllocateFromIterator`] subtraits provide methods that do allocation.
///
/// Not to be confused with [`core::alloc::Allocator`], which abstracts allocation at a lower level.
pub trait Allocator<T: ?Sized> {
    /// The pointer type returned by the allocator.
    type Pointer: Deref<Target = T>;
}

/// Allocates `T` values using "seed" values of type `S`.
///
/// The seed type allows broad generic implementations of `AllocateFrom`. This is especially useful
/// for allocating unsized types, which can't be passed directly by value.
pub trait AllocateFrom<T: ?Sized, S>: Allocator<T> {
    /// Allocates an object from the seed value.
    fn alloc(&self, seed: S) -> Self::Pointer;
}

/// Allocates `[T]` slices from iterators.
///
/// This trait is separated from [`AllocateFrom`] to allow generic implementations that could
/// otherwise conflict. For instance, [`BoxAllocator`] implements [`AllocateFrom<T, S>`] whenever
/// `Box<T>: From<S>`. The compiler won't allow an additional generic implementation of
/// [`AllocateFrom<[T], I>`](AllocateFrom) for [`BoxAllocator`] for all `I: IntoIterator<Item = T>`
/// because for some `T` and `I`, `Box<[T]>` could hypothetically also implement `From<I>`, which
/// would result in a conflicting implementation of [`AllocateFrom`]. (The compiler eagerly finds
/// hypothetical conflicts for trait implementations like this.) As such, we include
/// `AllocateFromIterator` as a separate trait because [`From`]- and [`FromIterator`]-based
/// allocator implementations are so common.
pub trait AllocateFromIterator<T>: Allocator<[T]> {
    /// Allocates a slice from an iterator.
    fn alloc_from_iter(&self, iter: impl IntoIterator<Item = T>) -> Self::Pointer;
}

/// An allocator for boxed values.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoxAllocator;

impl<T: ?Sized> Allocator<T> for BoxAllocator {
    type Pointer = Box<T>;
}

impl<T: ?Sized, S> AllocateFrom<T, S> for BoxAllocator
where
    Box<T>: From<S>,
{
    fn alloc(&self, seed: S) -> Self::Pointer {
        Box::from(seed)
    }
}

impl<T> AllocateFromIterator<T> for BoxAllocator {
    fn alloc_from_iter(&self, iter: impl IntoIterator<Item = T>) -> Self::Pointer {
        Box::from_iter(iter)
    }
}

/// An allocator for reference-counted values.
#[derive(Clone, Copy, Debug, Default)]
pub struct RcAllocator;

impl<T: ?Sized> Allocator<T> for RcAllocator {
    type Pointer = Rc<T>;
}

impl<T: ?Sized, S> AllocateFrom<T, S> for RcAllocator
where
    Rc<T>: From<S>,
{
    fn alloc(&self, seed: S) -> Self::Pointer {
        Rc::from(seed)
    }
}

impl<T> AllocateFromIterator<T> for RcAllocator {
    fn alloc_from_iter(&self, iter: impl IntoIterator<Item = T>) -> Self::Pointer {
        Rc::from_iter(iter)
    }
}

/// An allocator for atomically reference-counted values.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArcAllocator;

impl<T: ?Sized> Allocator<T> for ArcAllocator {
    type Pointer = Arc<T>;
}

impl<T: ?Sized, S> AllocateFrom<T, S> for ArcAllocator
where
    Arc<T>: From<S>,
{
    fn alloc(&self, seed: S) -> Self::Pointer {
        Arc::from(seed)
    }
}

impl<T> AllocateFromIterator<T> for ArcAllocator {
    fn alloc_from_iter(&self, iter: impl IntoIterator<Item = T>) -> Self::Pointer {
        Arc::from_iter(iter)
    }
}
