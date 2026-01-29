//! Minimal Objective-C runtime bindings.
//!
//! This module provides just enough FFI to interact with Cocoa without
//! pulling in the heavy `objc2` crate family.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]

use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_long};

// --- Types ---

pub type Class = *mut c_void;
pub type id = *mut c_void;
pub type SEL = *mut c_void;
pub type BOOL = c_char;

// Common ABI types
pub const YES: BOOL = 1;
pub const NO: BOOL = 0;
pub const nil: id = std::ptr::null_mut();

// --- Smart Pointers ---

/// A minimal wrapper for Objective-C objects that handles retain/release (ARC).
#[repr(transparent)]
pub struct Id(id);

impl Id {
    /// Create an Id from a raw pointer, assuming ownership (already retained).
    /// If ptr is nil, returns None.
    pub unsafe fn from_raw(ptr: id) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self(ptr))
        }
    }

    /// Create an Id from a raw pointer that needs to be retained.
    pub unsafe fn retain(ptr: id) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            let _: () = msg_send_void(ptr, selector("retain"));
            Some(Self(ptr))
        }
    }

    pub fn as_ptr(&self) -> id {
        self.0
    }
}

impl Clone for Id {
    fn clone(&self) -> Self {
        unsafe {
            let _: () = msg_send_void(self.0, selector("retain"));
            Self(self.0)
        }
    }
}

impl Drop for Id {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send_void(self.0, selector("release"));
        }
    }
}

// Allow sending Ids between threads if the underlying object is thread-safe.
// Most Cocoa UI objects are MainThreadOnly, but we'll handle that enforcement dynamically or via logic.
unsafe impl Send for Id {}
unsafe impl Sync for Id {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NSRect {
    pub origin: NSPoint,
    pub size: NSSize,
}

impl NSRect {
    pub fn new(origin: NSPoint, size: NSSize) -> Self {
        Self { origin, size }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NSPoint {
    pub x: f64,
    pub y: f64,
}

impl NSPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NSSize {
    pub width: f64,
    pub height: f64,
}

impl NSSize {
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

// --- FFI ---

#[link(name = "objc")]
extern "C" {
    pub fn objc_getClass(name: *const c_char) -> Class;
    pub fn sel_registerName(name: *const c_char) -> SEL;

    // We define a few variants of msgSend to handle different return types/arguments
    // strictly for type safety in our manual bindings, though in C it's variadic.
    // In Rust, we'll just cast `objc_msgSend` to the signature we need.
    pub fn objc_msgSend(self_: id, op: SEL, ...) -> id;

    // Class registration
    pub fn objc_allocateClassPair(
        superclass: Class,
        name: *const c_char,
        extraBytes: usize,
    ) -> Class;
    pub fn objc_registerClassPair(cls: Class);
    pub fn class_addMethod(
        cls: Class,
        name: SEL,
        imp: unsafe extern "C" fn(),
        types: *const c_char,
    ) -> BOOL;
}

// --- Helpers ---

// --- Helpers ---

pub fn class(name: &str) -> Class {
    let c_name = std::ffi::CString::new(name).unwrap();
    unsafe { objc_getClass(c_name.as_ptr()) }
}

pub fn selector(name: &str) -> SEL {
    let c_name = std::ffi::CString::new(name).unwrap();
    unsafe { sel_registerName(c_name.as_ptr()) }
}

/// Helper to simplify message sending.
/// This macro casts the generic msgSend to the appropriate function pointer type.
/// It assumes the return type is 'id' by default unless cast.
///
/// Example:
/// `let x: id = msg_send![obj, sel];`
/// `let x: BOOL = msg_send![obj, sel];` (Implicit coercion works for return type inference in limited cases, but explicitly casting the result is safer)
#[macro_export]
macro_rules! msg_send {
    ($obj:expr, $sel:ident) => {{
        let sel_name = stringify!($sel);
        // Default to returning id
        unsafe {
            let sel = $crate::platform::macos::objc::selector(sel_name);
            $crate::platform::macos::objc::msg_send_any($obj, sel)
        }
    }};
    ($obj:expr, $sel:ident : $arg:expr) => {{
        let sel_name = concat!(stringify!($sel), ":");
        unsafe {
            let sel = $crate::platform::macos::objc::selector(sel_name);
            $crate::platform::macos::objc::msg_send_any_1($obj, sel, $arg)
        }
    }}; // Add more patterns as needed
}

// Typed wrappers for common return types to keep unsafe blocks cleaner.

pub unsafe fn msg_send_void(obj: id, sel: SEL) {
    let fn_ptr: unsafe extern "C" fn(id, SEL) = std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel);
}

pub unsafe fn msg_send_id(obj: id, sel: SEL) -> id {
    let fn_ptr: unsafe extern "C" fn(id, SEL) -> id =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel)
}

pub unsafe fn msg_send_bool(obj: id, sel: SEL) -> BOOL {
    let fn_ptr: unsafe extern "C" fn(id, SEL) -> BOOL =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel)
}

// Generic unchecked send (returns T, which defaults to id in macro usage, but here strictly generic)
// NOT SAFE: Caller must ensure T matches ABI.
pub unsafe fn msg_send_any<T>(obj: id, sel: SEL) -> T {
    let fn_ptr: unsafe extern "C" fn(id, SEL) -> T =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel)
}

pub unsafe fn msg_send_any_1<T, A>(obj: id, sel: SEL, arg: A) -> T {
    let fn_ptr: unsafe extern "C" fn(id, SEL, A) -> T =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, arg)
}

pub unsafe fn msg_send_any_2<T, A, B>(obj: id, sel: SEL, arg1: A, arg2: B) -> T {
    let fn_ptr: unsafe extern "C" fn(id, SEL, A, B) -> T =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, arg1, arg2)
}

pub unsafe fn msg_send_any_4<T, A, B, C, D>(
    obj: id,
    sel: SEL,
    arg1: A,
    arg2: B,
    arg3: C,
    arg4: D,
) -> T {
    let fn_ptr: unsafe extern "C" fn(id, SEL, A, B, C, D) -> T =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, arg1, arg2, arg3, arg4)
}

pub unsafe fn nsstring_to_string(ns_str: id) -> Option<String> {
    if ns_str.is_null() {
        return None;
    }
    let utf8_sel = selector("UTF8String");
    let utf8: *const c_char = msg_send_any(ns_str, utf8_sel);
    if utf8.is_null() {
        return None;
    }
    let c_str = std::ffi::CStr::from_ptr(utf8);
    Some(c_str.to_string_lossy().into_owned())
}

pub unsafe fn string_to_nsstring(s: &str) -> Id {
    let cls = class("NSString");
    let alloc = selector("alloc");
    let alloc_obj: id = msg_send_id(cls, alloc);

    let init = selector("initWithUTF8String:");
    let c_str = std::ffi::CString::new(s).unwrap();
    let obj: id = msg_send_any_1(alloc_obj, init, c_str.as_ptr());

    // Id::from_raw assumes already retained? No, alloc+init returns retained (ownership +1).
    // Id::from_raw assumes we take ownership. Yes.
    Id::from_raw(obj).unwrap()
}
