use super::Usage;

/// Utility trait for constructing a [`Usage<U, T>`]
/// ```rust
/// use usage::{Usage, AsUsage};
///
/// pub enum Contrived {}
/// let contrived_vec: Usage<Contrived, Vec<usize>> = Contrived::as_usage(vec![1, 2, 3]);
/// ```
pub trait AsUsage: Sized {
    fn as_usage<T>(data: T) -> Usage<Self, T> {
        Usage {
            data,
            _phantom: Default::default(),
        }
    }
}

impl<T> AsUsage for T {}
