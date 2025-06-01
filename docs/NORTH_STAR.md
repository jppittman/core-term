# core Terminal Design Document - Target Architecture

**Version:** 1.9.15
**Date:** 2025-05-31
**Status:** Active

## 1. Introduction & Vision

`core` aims to be a correct, reasonably performant, and maintainable terminal emulator written in Rust. This document outlines a target architecture refactored from the initial `st`-inspired design towards a clearer separation of concerns, adopting an **Interpreter/Virtual Machine (VM) model** for its core logic.

The central idea is that the terminal's state machine (`TerminalEmulator`) acts as a self-contained VM. It processes a defined set of inputs (`EmulatorInput`) derived from PTY output and user actions, updates its internal state based on these inputs, and signals required external side-effects (`EmulatorAction`) without performing direct I/O. This approach enhances modularity, testability, and clarifies the role of each component.

A core principle is **simplicity and stability**, favoring well-established, CPU-based rendering techniques over GPU acceleration to avoid complex driver dependencies.

### 1.1. Core Feature Scope & Philosophy

A primary functional goal is to achieve feature parity with the `st` (simple terminal) terminal emulator, focusing on its core terminal emulation capabilities (VT100/VT220/XTerm compatibility, character set handling, SGR attributes, mouse reporting, etc.). Configurable copy/paste functionality is also a key feature.

Following the suckless philosophy, features explicitly out of scope, in line with the project's principles of simplicity and encouraging composition with dedicated tools (like `tmux`), include:
* Built-in scrollback.
* Advanced font rendering features like ligatures or complex text shaping beyond a monospaced grid.
* Image support or other graphical extensions beyond standard terminal capabilities.

## 2. Core Concepts & Design Principles

* **Configuration (`Config`):** Loaded once at startup (typically in `main.rs`) and made globally accessible via a mechanism like `std::sync::OnceLock` or `std::lazy::SyncLazy` (e.g., `static CONFIG: OnceLock<Config>`). This provides easy read-only access to settings like appearance, behavior, and keybindings throughout the application. The loading mechanism aims for robustness, with clear fallbacks to default values.

* **Terminal Emulator (Interpreter/VM):** The `TerminalEmulator` struct is the core state machine. It functions like a specialized interpreter or virtual machine where `EmulatorInput`s are its "opcodes." It manages the terminal's logical state (grid, cursor, modes, selection, character sets, per-line dirty flags) and, upon processing an input, updates this state and may produce `EmulatorAction`s, which are requests for external side-effects. It operates purely on its state and inputs, performing no direct I/O, which greatly enhances its testability and predictability. It consults the global `CONFIG` for behavioral rules (e.g., auto-wrap).

* **Emulator Input (Instruction Set):** The `enum EmulatorInput` is the comprehensive set of instructions for the `TerminalEmulator`. It wraps:
    * `AnsiCommand`: Parsed sequences from the primary I/O (PTY) like `CSI...m` (SGR) or `CSI...J` (Erase).
    * `UserInputAction`: Abstracted user interactions like `KeyInput`, `MouseInput` (with cell-based coordinates), `InitiateCopy`, `PasteText(String)`. These are typically derived from `BackendEvent`s.
    * `ControlEvent`: Internal signals like `Resize { cols, rows }` or `FrameRendered`.

* **Parser Pipeline:** Consists of `AnsiLexer` (byte stream to `AnsiToken`, handling UTF-8) and `AnsiParser` (`AnsiToken` stream to structured `AnsiCommand`). This pipeline is used by the `Platform` facade when it processes data read from the primary I/O channel (PTY), isolating the `AppOrchestrator` and `TerminalEmulator` from raw byte streams.

* **Emulator Action (Side-Effect Request):** `enum EmulatorAction` defines requests from the `TerminalEmulator` for the `AppOrchestrator` (acting via the `Platform` facade) to perform. Examples: `WritePty(Vec<u8>)` (for responses or echoed input), `SetTitle(String)`, `RingBell`, `CopyToClipboard(String)`.

* **`platform::Platform` Trait (Central Facade):**
    * This is the single, unified abstraction layer through which the `AppOrchestrator` interacts with all platform-specific services. Concrete implementations (e.g., `platform::native::NativePlatform` for Unix-like systems using X11 or console, or a future `platform::web::WasmPlatform`) encapsulate the "how" of platform interaction.
    * **Responsibilities of Concrete Implementations:** Internally, a concrete `Platform` implementation (like `NativePlatform`) owns and manages the lower-level platform components: the `PtyChannel` (for shell communication), the `Driver` (for UI rendering and input event handling), and an `EventInformer` (for polling or receiving event notifications from these sources).
    * **Methods Exposed:** The trait defines methods for:
        * Event Polling: `poll_system_events() -> Result<Vec<SystemEvent>>` (abstracts the actual polling mechanism like epoll, kqueue, or JS callbacks).
        * Primary I/O: `primary_io_read()`, `primary_io_write_all()`, `primary_io_resize()`.
        * UI Driver Event Processing: `driver_process_ui_events() -> Result<Vec<BackendEvent>>` (called when `UiInputReady` is signaled).
        * Platform Action Execution: `driver_execute_actions(Vec<PlatformActionCommand>)` (for batched rendering and UI updates).
        * State Querying: `driver_get_platform_state() -> PlatformState` (for metrics like font size, display dimensions, scale factor).
    * **Benefit:** The `AppOrchestrator` remains agnostic to how these services are implemented across different operating systems or environments (e.g., native vs. web).

* **`SystemEvent` Enum (in `platform` module or `platform::events`):**
    * This enum is yielded by `Platform::poll_system_events()` and signals to the `AppOrchestrator` what kind of event source is ready or if a system-level event occurred.
        * `PrimaryIoReady`: The main I/O channel (PTY) has data. Orchestrator will call `Platform.primary_io_read()`.
        * `UiInputReady`: The UI `Driver` has platform events. Orchestrator will call `Platform.driver_process_ui_events()`.
        * `Tick`: For periodic tasks (cursor blinking, application timeouts).
        * `Error(anyhow::Error)`: A critical error from the event polling mechanism.
        * `ShutdownAdvised`: A signal that the application should terminate.
    * This abstraction allows the `AppOrchestrator`'s event loop to be generic.

* **AppOrchestrator (`orchestrator.rs`):**
    * The central coordinator, taking a `Box<dyn Platform>` for all external interactions, plus references to `TerminalEmulator`, `AnsiParser`, and `Renderer`.
    * Its main `process_event_cycle()` method drives the application:
        1.  Calls `platform.poll_system_events()` to get `SystemEvent`s.
        2.  Dispatches based on `SystemEvent`:
            * `PrimaryIoReady` -> calls `platform.primary_io_read()`, passes data through its `AnsiParser`, gets `AnsiCommand`s, wraps as `EmulatorInput::Ansi`, then feeds to `TerminalEmulator`.
            * `UiInputReady` -> calls `platform.driver_process_ui_events()` to get `BackendEvent`s, translates them to `EmulatorInput::User` or `EmulatorInput::Control`, then feeds to `TerminalEmulator`.
        3.  Collects `EmulatorAction`s from `TerminalEmulator`.
        4.  Translates `EmulatorAction`s into calls to `platform.primary_io_write_all()` or into `PlatformActionCommand`s.
        5.  Orchestrates rendering: Gets `RenderSnapshot` from `TerminalEmulator`, uses `Renderer` to get drawing-related `PlatformActionCommand`s, combines with other commands, and sends the batch to `platform.driver_execute_actions()`.

* **`PlatformActionCommand` Enum (in `platform::backends`):**
    * This is the command language for the `AppOrchestrator` and `Renderer` to instruct the `Platform` facade (and its internal `Driver`) on UI operations.
    * It includes rendering primitives (`ClearAll`, `DrawTextRun`, `FillRect`), frame control (`PresentFrame`), and window management tasks (`SetWindowTitle`, `RingBell`, `SetNativeCursorVisibility`, `SetFocus`).
    * Batching these commands via `Platform.driver_execute_actions()` allows for potentially optimized execution by the `Driver`.

* **Rendering Driver (`Driver` Trait - Simplified, in `platform::backends`):**
    * This trait defines the contract for specific UI backends (e.g., X11, Console, macOS Cocoa, Wayland client). It is an *internal component* used by concrete `Platform` implementations (e.g., `NativePlatform` owns a `Box<dyn Driver>`).
    * **Simplified Operational Methods:**
        1.  `process_ui_events(&mut self) -> Result<Vec<BackendEvent>>`: Polls or processes native UI events and translates them into abstract `BackendEvent`s.
        2.  `execute_platform_actions(&mut self, actions: Vec<PlatformActionCommand>) -> Result<()>`: Takes a batch of abstract commands and executes them using platform-specific APIs (e.g., Xlib calls, Cocoa drawing).
    * **Lifecycle/State Methods:** Includes `new()`, `cleanup()`, `get_event_fd()` (for native event informers to monitor if the driver uses an FD, e.g., Wayland or X11 connection FD), and `get_platform_state()` (for metrics like font/display dimensions).
    * This lean interface ensures the `Driver` is focused on translation and execution, not complex logic.

* **BackendEvent (Abstract Platform Input):** (Defined in `platform::backends`). These are platform-agnostic representations of user/system interactions (e.g., `Key { symbol: KeySymbol::Enter, ... }`, `Mouse { ... }`, `Resize { width_px, height_px }`) generated by `Driver::process_ui_events()`. The `AppOrchestrator` receives these (via the `Platform` facade) and translates them into `EmulatorInput`.

* **Renderer (`Renderer` Struct):** A backend-agnostic component. It takes a `RenderSnapshot` from the `TerminalEmulator` and translates the terminal's visual state (dirty lines, selection, cursor) into a `Vec<PlatformActionCommand>` containing drawing operations. It handles visual logic like reversing colors for selections.

* **(Other concepts like `PlatformState` remain structurally similar but are accessed/used via the `Platform` facade, providing crucial metrics for layout and resizing decisions by the `AppOrchestrator`.)**

### 2.1. Event Handling Architecture: Discussion of Approaches

The way the `AppOrchestrator` receives and processes events is critical for simplicity, cross-platform compatibility, and responsiveness. Two main architectural approaches are considered:

**Approach A: Synchronous Polling Facade (Current Model in v1.9.14)**

* **Mechanism:** The `AppOrchestrator` calls `platform.poll_system_events(timeout)`, which is a synchronous (blocking or timed-out) call from the orchestrator's perspective.
    * A concrete `Platform` implementation (e.g., `NativePlatform`) uses an internal `EventInformer` (like `EpollEventInformer`) that blocks on OS-level I/O multiplexing calls (e.g., `epoll_wait`).
    * For platforms with inherently asynchronous event models (like WebAssembly with JavaScript callbacks, or GUI toolkits like macOS Cocoa with its own run loop), their `Platform::poll_system_events()` method would need to *simulate* this synchronous poll. This typically involves the `Platform` implementation checking internal queues populated by the asynchronous/callback mechanisms. The `poll()` call itself would then be non-blocking or have a very short timeout.
* **Pros:**
    * The `AppOrchestrator`'s main loop logic can be straightforward and synchronous.
    * For native platforms where FDs can be multiplexed efficiently with calls like `epoll` (e.g., PTY FDs, X11/Wayland connection FDs), this model maps quite directly and can be very efficient without introducing application-level threading for basic event multiplexing.
* **Cons:**
    * Can create an "impedance mismatch" for platforms with fundamentally asynchronous, callback-driven event systems (e.g., WASM, macOS/Cocoa). The `Platform` implementation for these systems needs to bridge their native async model to the synchronous `poll()` interface, which can add complexity or lead to less idiomatic platform integration (e.g., the `poll()` might effectively become a quick check of queues rather than a true blocking wait).
    * If not carefully managed, a blocking `poll()` could make the application less responsive if long timeouts are used and other non-FD based events (e.g., from internal timers not integrated into the poller) need timely processing.
* **"Suckless" / Simplicity Angle:** Simple for the orchestrator's direct logic. For native *nix, it leverages OS capabilities well. Complexity is pushed into the `Platform` implementations for async-native environments.

**Approach B: Asynchronous Channel-Based (Event Bus) Model**

* **Mechanism:**
    * The `platform::create_platform_services()` function (or the concrete `Platform` constructor) would be responsible for setting up event *producer tasks*. These tasks run somewhat independently (e.g., a dedicated thread for PTY reading on native, or JavaScript event handlers for WASM).
    * These producers, when they detect an event or receive data, process it to a certain stage (e.g., PTY reader + parser yielding `AnsiCommand`s; UI driver translating platform events to `BackendEvent`s) and send the result over channels (e.g., `std::sync::mpsc` channels or async channels if an async runtime is used).
    * `platform::create_platform_services()` would return a structure to `main.rs` containing the *receiving ends* of these input channels, alongside handles or senders for dispatching actions *back* to the platform (e.g., for writing to PTY or executing UI commands).
    * The `AppOrchestrator` would be constructed with these channel receivers and action dispatchers. Its `run()` method would then use a `select!`-like mechanism (from an async runtime, or a loop with non-blocking `try_recv` calls) to react to messages arriving on these channels.
* **Pros:**
    * **Better Decoupling:** Event production is cleanly separated from consumption. Each platform aspect (PTY I/O, UI event handling) can manage its event loop or blocking operations independently without stalling the `AppOrchestrator`.
    * **Natural Fit for Asynchronous Platforms:** This model aligns very well with WASM/JavaScript (JS callbacks send to channels) and GUI toolkits like Cocoa (toolkit's run loop processes UI events and sends results to a channel). There's less need to "force" an async model to look synchronous.
    * **Reactive Orchestrator:** The `AppOrchestrator` becomes purely reactive, processing inputs as they arrive on channels. Its main loop can be very declarative.
    * **Simplified Individual Components:** The event producer tasks can be simpler as they focus on one job (e.g., read PTY, parse, send). The `Platform` trait itself might even dissolve into a collection of channel endpoints and action dispatchers.
* **Cons:**
    * **Concurrency Complexity:** Introduces concurrency (threads for native, or an async runtime). This requires careful management of `Send`/`Sync`, lifetimes, and potential synchronization primitives if state needs to be shared beyond what channels provide (though channels are excellent for avoiding shared mutable state).
    * **Async Runtime Dependency:** For ergonomic `select!` and task management, an async runtime (like Tokio or async-std) is often chosen, adding a significant dependency if the project aims for absolute minimality. `std::sync::mpsc` with manual non-blocking polling is an alternative but less powerful for complex scenarios.
    * **Channel Management:** Setting up and managing the lifecycle of channels and producer tasks adds initial structural complexity.
* **"Suckless" / Simplicity Angle:**
    * An async runtime can be seen as "not suckless." However, if the alternative is significantly more complex adapter code within each `Platform` implementation to bridge async to sync, then channels might lead to *overall simpler and more maintainable platform-specific code*. The core logic in the `AppOrchestrator` reacting to message streams can be very simple.
    * The existing architecture already involves significant message passing (`EmulatorInput`, `EmulatorAction`, `SystemEvent`, `BackendEvent`, `PlatformActionCommand`), so a channel-based flow for some of these is a natural extension. As noted, shared mutable state across threads can be minimized with careful channel design.

**Decision Point:** The choice between these two primary event handling models (Approach A: Synchronous Poll Facade vs. Approach B: Async Channel-Based) is a key architectural decision. Approach A is currently embodied in the document. Approach B represents a shift towards a more explicitly asynchronous/concurrent internal design, which could offer benefits for certain platforms and simplify the "impedance matching" task. The project does not have to commit to one exclusively; a hybrid approach or an evolution towards channels could be considered. For now, v1.9.14 proceeds with the Synchronous Polling Facade (Approach A) due to its more straightforward initial implementation for native targets that map well to `epoll`/`kqueue`.

## 4. Data & Control Flow
(Based on Approach A - Synchronous Polling Facade)
The flow is now:
`main` -> (selects & creates concrete `Platform` impl like `platform::native::NativePlatform::new()`, then boxes it) -> `AppOrchestrator` gets `Box<dyn Platform>`.
`run_application` loop calls `orchestrator.process_event_cycle()`.
`orchestrator` uses `Platform.poll_system_events()` -> `SystemEvent`.
Based on `SystemEvent`:
  -> `Platform.primary_io_read()` -> `AnsiParser` -> `EmulatorInput::Ansi` -> `TerminalEmulator`
  -> `Platform.driver_process_ui_events()` -> `BackendEvent` -> `EmulatorInput::User/Control` -> `TerminalEmulator`
`TerminalEmulator` -> `EmulatorAction`.
`AppOrchestrator` translates `EmulatorAction` to `PlatformActionCommand` (if for UI/Driver) or calls direct `Platform` methods (like `primary_io_write_all`).
Rendering involves `Renderer` producing `PlatformActionCommand`s.
All `PlatformActionCommand`s are sent via `Platform.driver_execute_actions()`.

## 5. Benefits of this Architecture
(Based on Approach A - Synchronous Polling Facade)
* **Maximal `main.rs` Simplicity:** `main.rs` delegates all platform-specific instantiation to the chosen concrete `Platform` type's constructor, selected via `#[cfg]`.
* **Strong Platform Encapsulation:** The `Platform` trait provides a single facade. Concrete platform implementations (`NativePlatform`, etc.) manage their internal components (PTY, Driver, EventInformer) and bridge platform-specific event models to the synchronous `poll_system_events()` interface.
* **Simplified `Driver` Trait:** The underlying `Driver` (used by concrete `Platform` implementers) has a lean interface, primarily focused on processing UI input and executing batched platform actions.
* **Clear `AppOrchestrator` Role:** Works with the abstract `Platform` facade and `SystemEvent`s, maintaining a synchronous event processing loop.

## 6. Future Considerations
* **Platform Support (Wayland, macOS, WebAssembly):**
    * New concrete implementations of the `platform::Platform` trait would be created for each new target (e.g., `platform::macos::MacosPlatform`, `platform::web::WasmPlatform`).
    * These implementations would encapsulate all platform-specifics: PTY-like communication (e.g., WebSockets for WASM), UI driver (`WasmDriver` rendering to Canvas), and event informing mechanisms (e.g., JS callbacks bridging to the `poll_system_events()` interface for WASM, or `NSRunLoop` integration for macOS).
    * The `AppOrchestrator`'s core logic would remain unchanged, as it consumes the `Box<dyn Platform>`.
* **Adopting Asynchronous Model (Approach B):** The channel-based model (Approach B) remains a viable alternative for future evolution, especially if the complexities of adapting highly asynchronous platforms (like Web or GUI toolkits) to the synchronous `poll_system_events()` interface of Approach A become too burdensome, or if more fine-grained concurrency within the application is desired. This would involve a more significant refactoring of the `AppOrchestrator`'s main loop and the `Platform` trait's event delivery mechanism.
* **(Other future considerations from previous versions remain relevant).**

