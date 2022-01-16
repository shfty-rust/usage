//! # A convenient alternative to the newtype pattern
//!
//! When building an API in Rust, a common dilemma is the choice between
//! type aliases and the newtype pattern.
//!
//! Consider an application-specific collection of IDs with an underlying type of `Vec<usize>`;
//!
//! Using a type alias, one may choose to define it as ```pub type Identifiers = Vec<usize>```
//! in order to provide consumers of the type with unfettered access to the underlying `Vec`
//! methods, and allow `Identifiers` to be used interchangeably with `Vec<usize>`.
//!
//! Conversely, it could also be defined via newtype as ```pub struct Identifiers(Vec<usize>)```.
//! This creates a semantically-distinct type from `Vec<usize>`, but obscures access
//! to its underlying methods.
//!
//! Creating distinct types like this is one of the strengths of Rust's type system,
//! as it allows data dependencies to be encoded at type-time instead of runtime:
//! ```
//! pub struct Identifiers(Vec<usize>);
//!
//! pub fn create_ids() -> Identifiers {
//!     Identifiers(vec![0, 1, 2, 3])
//! }
//!
//! pub fn munge_ids(ids: Identifiers) {
//!     // ...
//! }
//!
//! // Valid
//! let ids: Identifiers = create_ids(); // Known-correct IDs provided by a trusted function
//! munge_ids(ids);
//!
//! // Not valid
//! // let ids = vec![999, 6, 876]; // IDs created arbitrarily, no guarantee of correctness
//! // munge_ids(ids); // Compiler error, incorrect type
//! ```
//!
//! In some cases, obscuring access to the underlying type's methods can be
//! desirable, as it allows the available functionality to be determined
//! by the API, thus implicitly providing information about its usage
//! to the library consumer.
//!
//! However, this is not true in all cases. Collection types like `Vec` are
//! case-in-point; they have so many useful methods and trait implementations
//! that manually re-exposing each one on a newtype becomes impractical,
//! possibly resulting in an overly restrictive design.
//!
//! In these cases, the inner type can be marked as `pub`,
//! and / or certain useful access traits like [`Into`], [`Borrow`] and [`Deref`] can be implemented.
//!
//! From an API standpoint, this combines the qualities of both type aliasing
//! and newtype: The type is distinct, but provides direct access to its underlying data.
//! (Though note that this also means breaking changes to the inner type will propagate outward.)
//!
//! Usage aims to model this sub-pattern as a generalized, reusable struct with
//! intuitive implementations for standard derivable, construction and access traits.
//!
//! ## Implementation
//!
//! It does this by using two generic parameters: Type `U` to act as a tag identifying it as a
//! distinct type, and type `T` for underlying data.
//!
//! `U` is represented by a [`PhantomData`], thus decoupling its trait implementations from those of the `Usage`.
//!
//! Construction and access trait implementations are predicated on `T`, allowing the `Usage` to
//! transparently act like its underlying type in as many contexts as possible.
//!
//! ## Limitations
//!
//! Due to coherence rules, foreign traits may not be implemented for foreign types.
//! Thus, it's infeasible to implement foreign traits on `Usage`; as a library type, it's foreign by design.
//!
//! This can be worked around by implementing the foreign trait over the Usage's `T` parameter
//! , or by using a newtype that implements said trait as the `T` instead.
//!
//! For cases where implementing over `Usage` is unavoidable,
//! such as compatibility with certain `std` traits or those from commonly-used crates,
//! feel free to send a pull request with the new functionality gated behind a feature flag
//! as per the existing `rayon` and `bytemuck` implementations.
//!

mod as_usage;
pub use as_usage::*;

use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

/// Wrapper type for creating a transparent-yet-distinct type over some underlying data.
/// ```
/// use usage::Usage;
///
/// enum Window {};
/// enum Surface {};
/// enum Texture {};
///
/// type Size = (u32, u32);
///
/// type WindowSize = Usage<Window, Size>;
/// type SurfaceSize = Usage<Surface, Size>;
/// type TextureSize = Usage<Texture, Size>;
/// ```
pub struct Usage<U, T> {
    pub data: T,
    _phantom: PhantomData<U>,
}

// Derived traits
impl<U, T> std::fmt::Debug for Usage<U, T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Usage")
            .field("data", &self.data)
            .field(
                "_phantom",
                &format!("PhantomData<{}>", std::any::type_name::<U>()),
            )
            .finish()
    }
}

impl<U, T> Default for Usage<U, T>
where
    T: Default,
{
    fn default() -> Self {
        Usage {
            data: Default::default(),
            _phantom: Default::default(),
        }
    }
}

impl<U, T> Copy for Usage<U, T> where T: Copy {}

impl<U, T> Clone for Usage<U, T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Usage {
            data: self.data.clone(),
            _phantom: Default::default(),
        }
    }
}

impl<U, T> PartialEq for Usage<U, T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data.eq(&other.data)
    }
}

impl<U, T> Eq for Usage<U, T>
where
    T: Eq,
{
    fn assert_receiver_is_total_eq(&self) {
        self.data.assert_receiver_is_total_eq()
    }
}

impl<U, T> PartialOrd for Usage<U, T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl<U, T> Ord for Usage<U, T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data.cmp(&other.data)
    }
}

impl<U, T> std::hash::Hash for Usage<U, T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

// Construction traits
impl<U, T> From<T> for Usage<U, T> {
    fn from(t: T) -> Self {
        U::as_usage(t)
    }
}

impl<U, T, V> FromIterator<V> for Usage<U, T>
where
    T: FromIterator<V>,
{
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        U::as_usage(iter.into_iter().collect())
    }
}

#[cfg(feature = "rayon")]
mod rayon_impl {
    use super::*;
    use rayon::iter::{
        FromParallelIterator, IntoParallelIterator, ParallelExtend, ParallelIterator,
    };

    impl<U, T, V> FromParallelIterator<V> for Usage<U, T>
    where
        T: FromParallelIterator<V>,
        V: Send,
    {
        fn from_par_iter<I: rayon::iter::IntoParallelIterator<Item = V>>(par_iter: I) -> Self {
            U::as_usage(par_iter.into_par_iter().collect())
        }
    }

    impl<U, T, V> ParallelExtend<V> for Usage<U, T>
    where
        T: ParallelExtend<V>,
        V: Send,
    {
        fn par_extend<I: IntoParallelIterator<Item = V>>(&mut self, par_iter: I) {
            self.data.par_extend(par_iter)
        }
    }
}

#[cfg(feature = "bytemuck")]
mod bytemuck_impl {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    unsafe impl<U, T> Zeroable for Usage<U, T> where T: Zeroable {}

    unsafe impl<U, T> Pod for Usage<U, T>
    where
        U: 'static,
        T: Pod,
    {
    }
}

// Data access traits
impl<U, T> Borrow<T> for Usage<U, T> {
    fn borrow(&self) -> &T {
        &self.data
    }
}

impl<U, T> BorrowMut<T> for Usage<U, T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<U, T> Deref for Usage<U, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<U, T> DerefMut for Usage<U, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<U, T> Usage<U, T> {
    /// Convert `Usage<T>` into `T` by value
    pub fn into_inner(self) -> T {
        self.data
    }
}
