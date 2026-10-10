//! A frame past one page is entered a page at a time, each page touched before
//! the next is taken, so it cannot step over a thread's guard page.
//!
//! The observation: a page in the middle of the kernel's frame is made
//! inaccessible, and a handler on an alternate stack records what faulted
//! there and where the stack pointer was, then makes the page accessible again
//! and lets the kernel continue. A probe is a store at the stack pointer, so
//! it faults with the stack pointer inside the page. Without it, the first
//! touch of that page is some slot of the finished frame, with the stack
//! pointer half a frame below.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use core::ffi::c_void;
use core::sync::atomic::{AtomicUsize, Ordering};
use pixelflow_codegen::emit::compile;
use pixelflow_ir::LatticeShape;

mod rows {
    include!("support/deep_frame.rs");
}
include!("support/knob.rs");

const PAGE: usize = 4096;
const WIDTH: usize = 11;
const ROWS: usize = 3;
const ORIGIN: [f32; 2] = [2.0, 5.0];
/// What the thread's stack holds beyond the frame: the call's own frames and
/// the handler's room to be unmasked.
const STACK_SLACK: usize = 256 * 1024;
const ALTERNATE_STACK: usize = 64 * 1024;

/// The page made inaccessible, and what the first fault in it saw.
static GUARD: AtomicUsize = AtomicUsize::new(0);
static FAULT_ADDRESS: AtomicUsize = AtomicUsize::new(0);
static FAULT_STACK_POINTER: AtomicUsize = AtomicUsize::new(0);

extern "C" fn on_fault(_: libc::c_int, info: *mut libc::siginfo_t, context: *mut c_void) {
    let guard = GUARD.load(Ordering::SeqCst);
    let (address, stack_pointer) = unsafe {
        let context = &*context.cast::<libc::ucontext_t>();
        (
            (*info).si_addr() as usize,
            context.uc_mcontext.gregs[libc::REG_RSP as usize] as usize,
        )
    };
    if address & !(PAGE - 1) != guard {
        // Not ours: the default action, on the retry.
        unsafe { libc::signal(libc::SIGSEGV, libc::SIG_DFL) };
        return;
    }
    FAULT_ADDRESS.store(address, Ordering::SeqCst);
    FAULT_STACK_POINTER.store(stack_pointer, Ordering::SeqCst);
    unsafe {
        libc::mprotect(
            guard as *mut c_void,
            PAGE,
            libc::PROT_READ | libc::PROT_WRITE,
        )
    };
}

fn run(code: &pixelflow_codegen::emit::CompiledKernel) -> Vec<f32> {
    let mut out = vec![f32::NAN; ROWS * WIDTH];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    unsafe { code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
    out
}

#[test]
fn a_frame_past_a_page_touches_every_page_as_it_takes_it() {
    if !selection() {
        return;
    }
    let (arena, root) = rows::deep_frame(rows::DEEP_FRAME_TERMS);
    let shape = LatticeShape::new([WIDTH as u32, ROWS as u32]);
    let compiled = compile(&arena, root, shape).expect("a deep frame compiles");
    let frame = usize::try_from(compiled.frame_bytes).expect("a frame fits the address space");
    assert!(
        frame > 8 * PAGE,
        "the fixture no longer forces a deep frame"
    );
    let wanted = run(&compiled.code);

    let mut alternate = vec![0u8; ALTERNATE_STACK];
    let got = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(frame + STACK_SLACK)
            .spawn_scoped(scope, || {
                let here = 0u8;
                let guard = (core::ptr::addr_of!(here) as usize - frame / 2) & !(PAGE - 1);
                GUARD.store(guard, Ordering::SeqCst);
                unsafe {
                    let stack = libc::stack_t {
                        ss_sp: alternate.as_mut_ptr().cast(),
                        ss_flags: 0,
                        ss_size: ALTERNATE_STACK,
                    };
                    assert_eq!(libc::sigaltstack(&stack, core::ptr::null_mut()), 0);
                    let mut action: libc::sigaction = core::mem::zeroed();
                    action.sa_sigaction = on_fault as *const () as usize;
                    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
                    assert_eq!(
                        libc::sigaction(libc::SIGSEGV, &action, core::ptr::null_mut()),
                        0
                    );
                    assert_eq!(
                        libc::mprotect(guard as *mut c_void, PAGE, libc::PROT_NONE),
                        0
                    );
                }
                run(&compiled.code)
            })
            .expect("the thread starts")
            .join()
            .expect("the kernel runs to the end")
    });

    let address = FAULT_ADDRESS.load(Ordering::SeqCst);
    let stack_pointer = FAULT_STACK_POINTER.load(Ordering::SeqCst);
    assert_ne!(address, 0, "no instruction touched the page mid-frame");
    assert_eq!(
        address, stack_pointer,
        "the page was first touched by a slot of the finished frame, not by the probe that took it"
    );
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        wanted.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a probe changed the kernel's values"
    );
}
