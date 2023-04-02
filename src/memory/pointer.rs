use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    ptr,
};

/// A thin wrapper around a pointer type with slightly different semantics: the "identity" of a
/// `ByAddress` is determined by its pointer's address, rather than the pointee value.
///
/// In particular, this means that:
///
/// * [`Eq`], [`Ord`], [`PartialEq`], and [`PartialOrd`] compare addresses
/// * [`Hash`] is computed using the pointer value
///
/// Example use cases where these semantics are valuable:
///
/// 1. *Unique IDs.* If each instance of some type is allocated behind a pointer, then the address
///    of the allocation can be used as a unique ID for the instance.
/// 2. *[Interned](https://en.wikipedia.org/wiki/String_interning) values.* When all equal instances
///    of a type are guaranteed to reside at the same address in memory, then comparing by address
///    saves computation time.
///
/// # Zero-sized types (ZSTs)
///
/// ZSTs are generally allocated with [`NonNull::dangling()`](ptr::NonNull), which always returns
/// the same pointer value. As such `ByAddress` pointers to ZSTs will usually all be/ considered
/// equal, which is usually what the user wants anyway. If you want to be able to distinguish
/// between different instances of a ZST, do not use `ByAddress`.
///
/// # Trait objects (i.e., `dyn Trait`)
///
/// Using `ByAddress` with trait object pointers (i.e., `dyn Trait`) is ill-advised. As discussed in
/// the [`std::ptr::eq`] docs:
///
/// > Comparing trait object pointers (`*const dyn Trait`) is unreliable: pointers to values of the
///   same underlying type can compare inequal (because vtables are duplicated in multiple codegen
///   units), and pointers to values of *different* underlying type can compare equal (since
///   identical vtables can be deduplicated within a codegen unit).
///
/// # No `Borrow` implementation
///
/// Note that while `ByAddress<P>: AsRef` if `P: AsRef`, `ByAddress<P>` is deliberately **not**
/// [`Borrow`](core::borrow::Borrow), since the address's hash is not the same as the pointee's
/// hash.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ByAddress<P>(pub P);

impl<T: ?Sized, P: Deref> AsMut<T> for ByAddress<P>
where
    P: AsMut<T>,
{
    fn as_mut(&mut self) -> &mut T {
        self.0.as_mut()
    }
}

impl<T: ?Sized, P: Deref> AsRef<T> for ByAddress<P>
where
    P: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.0.as_ref()
    }
}

impl<P: Deref> Deref for ByAddress<P> {
    type Target = P::Target;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<P: DerefMut> DerefMut for ByAddress<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<P: Deref> Eq for ByAddress<P> {}

impl<P: Deref> Hash for ByAddress<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(&*self.0, state)
    }
}

impl<P: Deref> Ord for ByAddress<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        (&*self.0 as *const P::Target).cmp(&(&*other.0 as *const P::Target))
    }
}

impl<P: Deref> PartialEq for ByAddress<P> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(&*self.0, &*other.0)
    }
}

impl<P: Deref> PartialOrd for ByAddress<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
