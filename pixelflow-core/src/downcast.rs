#![deny(unsafe_code,rustdoc::bare_urls)]
#![cfg_attr(not(feature = "std"), no_std)]

// for compatibility with no std and macros
#[doc(hidden)]
#[cfg(not(feature = "std"))]
pub extern crate core as __std;
#[doc(hidden)]
#[cfg(feature = "std")]
pub extern crate std as __std;
#[doc(hidden)]
pub extern crate alloc as __alloc;

use __std::any::Any;
use __alloc::{boxed::Box, rc::Rc};

#[cfg(feature = "sync")]
use __alloc::sync::Arc;

/// Supports conversion to `Any`. Traits to be extended by `impl_downcast!` must extend `Downcast`.
pub trait Downcast: Any {
    /// Converts `Box<dyn Trait>` (where `Trait: Downcast`) to `Box<dyn Any>`, which can then be
    /// `downcast` into `Box<dyn ConcreteType>` where `ConcreteType` implements `Trait`.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    /// Converts `Rc<Trait>` (where `Trait: Downcast`) to `Rc<Any>`, which can then be further
    /// `downcast` into `Rc<ConcreteType>` where `ConcreteType` implements `Trait`.
    fn into_any_rc(self: Rc<Self>) -> Rc<dyn Any>;
    /// Converts `&Trait` (where `Trait: Downcast`) to `&Any`. This is needed since Rust cannot
    /// generate `&Any`'s vtable from `&Trait`'s.
    fn as_any(&self) -> &dyn Any;
    /// Converts `&mut Trait` (where `Trait: Downcast`) to `&Any`. This is needed since Rust cannot
    /// generate `&mut Any`'s vtable from `&mut Trait`'s.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any> Downcast for T {
    fn into_any(self: Box<Self>) -> Box<dyn Any> { self }
    fn into_any_rc(self: Rc<Self>) -> Rc<dyn Any> { self }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Extends `Downcast` for `Send` traits to support upcasting to `Box<dyn Any + Send>` as well.
pub trait DowncastSend: Downcast + Send {
    /// Converts `Box<Trait>` (where `Trait: DowncastSend`) to `Box<dyn Any + Send>`, which
    /// can then be `downcast` into `Box<ConcreteType>` where `ConcreteType` implements `Trait`.
    fn into_any_send(self: Box<Self>) -> Box<dyn Any + Send>;
}

impl<T: Any + Send> DowncastSend for T {
    fn into_any_send(self: Box<Self>) -> Box<dyn Any + Send> { self }
}

#[cfg(feature = "sync")]
/// Extends `DowncastSend` for `Sync` traits to support upcasting to `Box<dyn Any + Send + Sync>`
/// and `Arc` downcasting.
pub trait DowncastSync: DowncastSend + Sync {
    /// Converts `Box<Trait>` (where `Trait: DowncastSync`) to `Box<dyn Any + Send + Sync>`,
    /// which can then be `downcast` into `Box<ConcreteType>` where `ConcreteType` implements
    /// `Trait`.
    fn into_any_sync(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
    /// Converts `Arc<Trait>` (where `Trait: DowncastSync`) to `Arc<Any>`, which can then be
    /// `downcast` into `Arc<ConcreteType>` where `ConcreteType` implements `Trait`.
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

#[cfg(feature = "sync")]
impl<T: Any + Send + Sync> DowncastSync for T {
    fn into_any_sync(self: Box<Self>) -> Box<dyn Any + Send + Sync> { self }
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> { self }
}

/// Adds downcasting support to traits that extend `Downcast` by defining forwarding
/// methods to the corresponding implementations on `std::any::Any` in the standard library.
///
/// See <https://users.rust-lang.org/t/how-to-create-a-macro-to-impl-a-provided-type-parametrized-trait/5289>
/// for why this is implemented this way to support templatized traits.
#[macro_export]
macro_rules! impl_downcast {
    (@impl_full
        $trait_:ident [$($param_types:tt)*]
        for [$($forall_types:ident),*]
        where [$($preds:tt)*]
    ) => {
        impl_downcast! {
            @inject_where
                [impl<$($forall_types),*> dyn $trait_<$($param_types)*>]
                types [$($forall_types),*]
                where [$($preds)*]
                [{
                    impl_downcast! { @impl_body $trait_ [$($param_types)*] }
                }]
        }
    };

    (@impl_full_sync
        $trait_:ident [$($param_types:tt)*]
        for [$($forall_types:ident),*]
        where [$($preds:tt)*]
    ) => {
        impl_downcast! {
            @inject_where
                [impl<$($forall_types),*> dyn $trait_<$($param_types)*>]
                types [$($forall_types),*]
                where [$($preds)*]
                [{
                    impl_downcast! { @impl_body $trait_ [$($param_types)*] }
                    impl_downcast! { @impl_body_sync $trait_ [$($param_types)*] }
                }]
        }
    };

    (@impl_body $trait_:ident [$($types:tt)*]) => {
        /// Returns true if the trait object wraps an object of type `__T`.
        #[inline]
        pub fn is<__T: $trait_<$($types)*>>(&self) -> bool {
            $crate::downcast::Downcast::as_any(self).is::<__T>()
        }
        /// Returns a boxed object from a boxed trait object if the underlying object is of type
        /// `__T`. Returns the original boxed trait if it isn't.
        #[inline]
        pub fn downcast<__T: $trait_<$($types)*>>(
            self: $crate::__alloc::boxed::Box<Self>
        ) -> $crate::__std::result::Result<$crate::__alloc::boxed::Box<__T>, $crate::__alloc::boxed::Box<Self>> {
            if self.is::<__T>() {
                Ok($crate::downcast::Downcast::into_any(self).downcast::<__T>().unwrap())
            } else {
                Err(self)
            }
        }
        /// Returns an `Rc`-ed object from an `Rc`-ed trait object if the underlying object is of
        /// type `__T`. Returns the original `Rc`-ed trait if it isn't.
        #[inline]
        pub fn downcast_rc<__T: $trait_<$($types)*>>(
            self: $crate::__alloc::rc::Rc<Self>
        ) -> $crate::__std::result::Result<$crate::__alloc::rc::Rc<__T>, $crate::__alloc::rc::Rc<Self>> {
            if self.is::<__T>() {
                Ok($crate::downcast::Downcast::into_any_rc(self).downcast::<__T>().unwrap())
            } else {
                Err(self)
            }
        }
        /// Returns a reference to the object within the trait object if it is of type `__T`, or
        /// `None` if it isn't.
        #[inline]
        pub fn downcast_ref<__T: $trait_<$($types)*>>(&self) -> $crate::__std::option::Option<&__T> {
            $crate::downcast::Downcast::as_any(self).downcast_ref::<__T>()
        }
        /// Returns a mutable reference to the object within the trait object if it is of type
        /// `__T`, or `None` if it isn't.
        #[inline]
        pub fn downcast_mut<__T: $trait_<$($types)*>>(&mut self) -> $crate::__std::option::Option<&mut __T> {
            $crate::downcast::Downcast::as_any_mut(self).downcast_mut::<__T>()
        }
    };

    (@impl_body_sync $trait_:ident [$($types:tt)*]) => {
        /// Returns an `Arc`-ed object from an `Arc`-ed trait object if the underlying object is of
        /// type `__T`. Returns the original `Arc`-ed trait if it isn't.
        #[inline]
        pub fn downcast_arc<__T: $trait_<$($types)*> + $crate::__std::any::Any + $crate::__std::marker::Send + $crate::__std::marker::Sync>(
            self: $crate::__alloc::sync::Arc<Self>,
        ) -> $crate::__std::result::Result<$crate::__alloc::sync::Arc<__T>, $crate::__alloc::sync::Arc<Self>>
        {
            if self.is::<__T>() {
                Ok($crate::downcast::DowncastSync::into_any_arc(self).downcast::<__T>().unwrap())
            } else {
                Err(self)
            }
        }
    };

    (@inject_where [$($before:tt)*] types [] where [] [$($after:tt)*]) => {
        impl_downcast! { @as_item $($before)* $($after)* }
    };

    (@inject_where [$($before:tt)*] types [$($types:ident),*] where [] [$($after:tt)*]) => {
        impl_downcast! {
            @as_item
                $($before)*
                where $( $types: $crate::__std::any::Any + 'static ),*
                $($after)*
        }
    };
    (@inject_where [$($before:tt)*] types [$($types:ident),*] where [$($preds:tt)+] [$($after:tt)*]) => {
        impl_downcast! {
            @as_item
                $($before)*
                where
                    $( $types: $crate::__std::any::Any + 'static, )*
                    $($preds)*
                $($after)*
        }
    };

    (@as_item $i:item) => { $i };

    // No type parameters.
    ($trait_:ident   ) => { impl_downcast! { @impl_full $trait_ [] for [] where [] } };
    ($trait_:ident <>) => { impl_downcast! { @impl_full $trait_ [] for [] where [] } };
    (sync $trait_:ident   ) => { impl_downcast! { @impl_full_sync $trait_ [] for [] where [] } };
    (sync $trait_:ident <>) => { impl_downcast! { @impl_full_sync $trait_ [] for [] where [] } };
    // Type parameters.
    ($trait_:ident < $($types:ident),* >) => {
        impl_downcast! { @impl_full $trait_ [$($types),*] for [$($types),*] where [] }
    };
    (sync $trait_:ident < $($types:ident),* >) => {
        impl_downcast! { @impl_full_sync $trait_ [$($types),*] for [$($types),*] where [] }
    };
    // Type parameters and where clauses.
    ($trait_:ident < $($types:ident),* > where $($preds:tt)+) => {
        impl_downcast! { @impl_full $trait_ [$($types),*] for [$($types),*] where [$($preds)*] }
    };
    (sync $trait_:ident < $($types:ident),* > where $($preds:tt)+) => {
        impl_downcast! { @impl_full_sync $trait_ [$($types),*] for [$($types),*] where [$($preds)*] }
    };
    // Associated types.
    ($trait_:ident assoc $($atypes:ident),*) => {
        impl_downcast! { @impl_full $trait_ [$($atypes = $atypes),*] for [$($atypes),*] where [] }
    };
    (sync $trait_:ident assoc $($atypes:ident),*) => {
        impl_downcast! { @impl_full_sync $trait_ [$($atypes = $atypes),*] for [$($atypes),*] where [] }
    };
    // Associated types and where clauses.
    ($trait_:ident assoc $($atypes:ident),* where $($preds:tt)+) => {
        impl_downcast! { @impl_full $trait_ [$($atypes = $atypes),*] for [$($atypes),*] where [$($preds)*] }
    };
    (sync $trait_:ident assoc $($atypes:ident),* where $($preds:tt)+) => {
        impl_downcast! { @impl_full_sync $trait_ [$($atypes = $atypes),*] for [$($atypes),*] where [$($preds)*] }
    };
    // Type parameters and associated types.
    ($trait_:ident < $($types:ident),* > assoc $($atypes:ident),*) => {
        impl_downcast! {
            @impl_full
                $trait_ [$($types),*, $($atypes = $atypes),*]
                for [$($types),*, $($atypes),*]
                where []
        }
    };
    (sync $trait_:ident < $($types:ident),* > assoc $($atypes:ident),*) => {
        impl_downcast! {
            @impl_full_sync
                $trait_ [$($types),*, $($atypes = $atypes),*]
                for [$($types),*, $($atypes),*]
                where []
        }
    };
    // Type parameters, associated types, and where clauses.
    ($trait_:ident < $($types:ident),* > assoc $($atypes:ident),* where $($preds:tt)+) => {
        impl_downcast! {
            @impl_full
                $trait_ [$($types),*, $($atypes = $atypes),*]
                for [$($types),*, $($atypes),*]
                where [$($preds)*]
        }
    };
    (sync $trait_:ident < $($types:ident),* > assoc $($atypes:ident),* where $($preds:tt)+) => {
        impl_downcast! {
            @impl_full_sync
                $trait_ [$($types),*, $($atypes = $atypes),*]
                for [$($types),*, $($atypes),*]
                where [$($preds)*]
        }
    };
    // Concretely-parametrized types.
    (concrete $trait_:ident < $($types:ident),* >) => {
        impl_downcast! { @impl_full $trait_ [$($types),*] for [] where [] }
    };
    (sync concrete $trait_:ident < $($types:ident),* >) => {
        impl_downcast! { @impl_full_sync $trait_ [$($types),*] for [] where [] }
    };
    // Concretely-associated types types.
    (concrete $trait_:ident assoc $($atypes:ident = $aty:ty),*) => {
        impl_downcast! { @impl_full $trait_ [$($atypes = $aty),*] for [] where [] }
    };
    (sync concrete $trait_:ident assoc $($atypes:ident = $aty:ty),*) => {
        impl_downcast! { @impl_full_sync $trait_ [$($atypes = $aty),*] for [] where [] }
    };
    // Concretely-parametrized types with concrete associated types.
    (concrete $trait_:ident < $($types:ident),* > assoc $($atypes:ident = $aty:ty),*) => {
        impl_downcast! { @impl_full $trait_ [$($types),*, $($atypes = $aty),*] for [] where [] }
    };
    (sync concrete $trait_:ident < $($types:ident),* > assoc $($atypes:ident = $aty:ty),*) => {
        impl_downcast! { @impl_full_sync $trait_ [$($types),*, $($atypes = $aty),*] for [] where [] }
    };
}
