# core Terminal Design Document - Target Architecture

**Version:** 1.9.17
**Date:** 2025-06-03
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

* **Configuration (`Config`):** Loaded once at startup (typically in `main.rs`) and made globally accessible via a mechanism like `std::sync::OnceLock` or `std::lazy::SyncLazy` (e.g., `static CONFIG: OnceLock<Config>`). This provides easy read-only access to settings like appearance, behavior, and keybindings throughout the application. The loading mechanism aims for robustness, with clear fallbacks to default values. Keybindings are defined as a mapping from key combinations (`KeySymbol`, `Modifiers`) directly to `UserInputAction`s, allowing for both compile-time defaults ("suckless-style") and runtime overrides via a configuration file.

* **Terminal Emulator (Interpreter/VM):** The `TerminalEmulator` struct is the core state machine. It functions like a specialized interpreter or virtual machine where `EmulatorInput`s are its "opcodes." It manages the terminal's logical state (grid, cursor, modes, selection, character sets, per-line dirty flags) and, upon processing an input, updates this state and may produce `EmulatorAction`s, which are requests for external side-effects. It operates purely on its state and inputs, performing no direct I/O, which greatly enhances its testability and predictability. It consults the global `CONFIG` for behavioral rules (e.g., auto-wrap).

* **Emulator Input (Instruction Set):** The `enum EmulatorInput` is the comprehensive set of instructions for the `TerminalEmulator`. It wraps:
    * `AnsiCommand`: Parsed sequences from the primary I/O (PTY) like `CSI...m` (SGR) or `CSI...J` (Erase).
    * `UserInputAction`: Abstracted user interactions like `KeyInput`, `MouseInput` (with cell-based coordinates), `InitiateCopy`, `PasteText(String)`, or application-level requests like `RequestZoom`. These are typically derived from `BackendEvent`s, potentially after processing through the keybinding system.
    * `ControlEvent`: Internal signals like `Resize { cols, rows }` or `FrameRendered`.

* **Parser Pipeline:** Consists of `AnsiLexer` (byte stream to `AnsiToken`, handling UTF-8) and `AnsiParser` (`AnsiToken` stream to structured `AnsiCommand`). This pipeline is used by the `AppOrchestrator` when it processes data read from the primary I/O channel (PTY), isolating the `TerminalEmulator` from raw byte streams.

* **Emulator Action (Side-Effect Request):** `enum EmulatorAction` defines requests from the `TerminalEmulator` for the `AppOrchestrator` to perform. Examples: `WritePty(Vec<u8>)` (for responses or echoed input), `SetTitle(String)`, `RingBell`, `CopyToClipboard(String)`.

* **Concrete Platform Struct (e.g., `LinuxX11Platform`, `WasmPlatform`):**
    * This is an RAII-compliant struct responsible for managing all platform-specific resources for a given target (e.g., Linux with X11, or WebAssembly). It can be referred to simply as "the `Platform`" in general discussion.
    * **Initialization (`new()`):** Its constructor (e.g., `LinuxX11Platform::new() -> Result<(Self, PlatformState)>`) sets up all necessary components:
        * The PTY channel (e.g., `NixPty` for Unix, WebSocket for WASM).
        * The UI `Driver` (e.g., `XDriver`, `ConsoleDriver`, or a Canvas-based driver for WASM).
        * Internal communication channels (e.g., `std::sync::mpsc`) for passing events from platform sources (PTY, UI driver) to the `AppOrchestrator` and for passing actions from the `AppOrchestrator` back to the platform components.
        * Internal event production mechanisms (e.g., an `epoll` loop, potentially in a dedicated thread, or JS event listener bridges for WASM). These mechanisms read from raw platform sources and send structured events (like `Vec<u8>` from PTY or `BackendEvent` from UI) over the internal channels.
        * It returns itself and the initial `PlatformState` (font metrics, display size).
    * **Interaction API for `AppOrchestrator`:** It provides methods for the `AppOrchestrator` to:
        * Poll for events: e.g., `poll_pty_data(&self) -> Result<Option<Vec<u8>>>`, `poll_ui_event(&self) -> Result<Option<BackendEvent>>`. These methods would typically perform a non-blocking check (e.g., `try_recv()`) on the internal receiver channels.
        * Dispatch actions: e.g., `dispatch_pty_action(&self, PtyActionCommand)`, `dispatch_ui_action(&self, UiActionCommand)`. These methods would send commands over internal sender channels to the PTY handler or UI driver.
        * Query current state: `get_current_platform_state(&self) -> PlatformState`.
    * **Cleanup (`Drop`):** Its `Drop` implementation ensures graceful shutdown of all owned resources, including signaling internal event producer tasks to terminate, joining threads, closing the PTY, and cleaning up UI driver resources.
    * **Selection:** Platform selection is done at compile time using `#[cfg]` attributes in `main.rs` to choose which concrete `Platform` struct (e.g., `LinuxX11Platform`) is instantiated.

* **`AppOrchestrator` (`orchestrator.rs`):**
    * The central coordinator. It takes a reference to the concrete `Platform` struct (e.g., `&'a LinuxX11Platform`), references to `TerminalEmulator`, `AnsiParser`, and owns the `Renderer`.
    * Its main `process_event_cycle()` method drives the application:
        1.  Calls methods on the `Platform` reference (e.g., `platform.poll_pty_data()`, `platform.poll_ui_event()`) to get events.
        2.  Processes these events:
            * PTY data is passed through its `AnsiParser` to get `AnsiCommand`s, wrapped as `EmulatorInput::Ansi`, then fed to `TerminalEmulator`.
            * `BackendEvent::Key` events are first mapped against configured keybindings (defined in `Config`). This mapping yields a `UserInputAction`.
            * The `AppOrchestrator` then inspects the resulting `UserInputAction`:
                * If it's an application-level command (e.g., `RequestZoom`), the orchestrator handles it directly, potentially by interacting with the `Driver` (via the `Platform` reference) or modifying application state.
                * Otherwise (e.g., for `KeyInput`, `InitiateCopy`), the `UserInputAction` is wrapped as `EmulatorInput::User` and sent to the `TerminalEmulator`.
            * Other `BackendEvent`s (mouse, resize, focus) are translated into `EmulatorInput::User` or `EmulatorInput::Control` as appropriate and typically sent to the `TerminalEmulator`.
        3.  Collects `EmulatorAction`s from `TerminalEmulator`.
        4.  Translates `EmulatorAction`s into calls to `platform.dispatch_pty_action()` or into `UiActionCommand`s for `platform.dispatch_ui_action()`.
        5.  Orchestrates rendering: Gets `TerminalSnapshot` from `TerminalEmulator`, uses `Renderer` to get drawing-related `RenderCommand`s. These, along with other UI-related `EmulatorAction`s (like setting title), are packaged into `UiActionCommand`s and sent via `platform.dispatch_ui_action()`.

* **`UiActionCommand` / `PtyActionCommand` Enums:**
    * These define the command languages for the `AppOrchestrator` to instruct the concrete `Platform` struct (and its internal PTY handler or UI `Driver`) on operations to perform.
    * `UiActionCommand` would include rendering commands (`Render(Vec<RenderCommand>)`), window management (`SetTitle`), etc.
    * `PtyActionCommand` would include `Write(Vec<u8>)`, `ResizePty { cols, rows }`.
    *(Note: `PlatformActionCommand` from previous versions is conceptually absorbed/represented by `UiActionCommand` or directly by `RenderCommand` within it).*

* **Rendering Driver (`Driver` Trait - in `platform::backends`):**
    * This trait defines the contract for specific UI backends (e.g., X11, Console). It is an *internal component* used and owned by concrete `Platform` struct implementations (e.g., `LinuxX11Platform` owns an `XDriver`).
    * **Responsibilities:**
        1.  Translating native platform UI events (keyboard, mouse, resize) into abstract `BackendEvent`s. The concrete `Platform` struct's internal event loop/poller would call a method on the `Driver` to get these events, which are then sent over an internal channel to the `AppOrchestrator`.
        2.  Executing abstract `RenderCommand`s (and other UI-related commands like `SetTitle`) received from the concrete `Platform` struct (which got them via an internal channel from the `AppOrchestrator`).
        3.  Managing platform-specific graphics resources (fonts, colors, drawing contexts).
        4.  Providing font and display metrics via a `get_platform_state()` method.
    * The `Driver` interacts with the OS/display system directly. Its lifecycle (`new`, `cleanup`/`Drop`) is managed by the owning concrete `Platform` struct.

* **BackendEvent (Abstract Platform Input):** (Defined in `platform::backends`). Platform-agnostic representations of user/system interactions generated by the `Driver` and passed (via the concrete `Platform` struct's internal channels) to the `AppOrchestrator`.

* **Renderer (`Renderer` Struct):** A backend-agnostic component. It takes a `TerminalSnapshot` from the `TerminalEmulator` and translates the terminal's visual state into a `Vec<RenderCommand>`.

### 2.1. Event Handling Architecture

The architecture employs a consistent event-passing model for the `AppOrchestrator`. The `AppOrchestrator` interacts with a concrete `Platform` struct (e.g., `LinuxX11Platform`, `WasmPlatform`) by calling methods on it to poll for events and dispatch actions.

* **Event Production (Internal to the concrete `Platform` struct):**
    * Each concrete `Platform` struct is responsible for setting up and managing its own internal mechanisms for producing events from the PTY and the UI `Driver`.
    * **Native Platforms (e.g., Linux/X11):** This typically involves an `epoll` (or similar I/O multiplexing) loop, potentially running in a dedicated thread. This loop monitors file descriptors for the PTY and the X11 connection (or other UI event sources). When activity is detected, data is read, processed (e.g., PTY bytes to `Vec<u8>`, X11 events to `BackendEvent`), and then sent to internal channels, the receiving ends of which are polled by the `AppOrchestrator` via methods on the `Platform` struct.
    * **WebAssembly (WASM):** JavaScript event listeners (for keyboard, mouse, WebSocket messages for PTY) would trigger Rust/WASM callback functions. These callbacks would translate the JS events/data into the appropriate Rust types (`Vec<u8>` or `BackendEvent`) and send them to internal channels, similarly polled by the `AppOrchestrator`.
    * **Single-Threaded Polling Variation for Native Platforms:** If a dedicated event-producing thread is not desired (to keep the application strictly single-threaded in its main logic), the `Platform` struct could expose a `pump_platform_events()` method. The `AppOrchestrator` would then call this method periodically in its main loop. This `pump` method would perform non-blocking checks on its event sources (e.g., non-blocking `epoll_wait`, non-blocking PTY read) and push any discovered events onto the internal channels that the `AppOrchestrator` polls.

* **Event Consumption (by `AppOrchestrator`):**
    * The `AppOrchestrator`'s main loop calls polling methods on its `Platform` reference (e.g., `platform.poll_pty_data()`, `platform.poll_ui_event()`).
    * These methods perform non-blocking reads (e.g., `try_recv()`) from the internal channels managed by the `Platform` struct.
    * This ensures that the `AppOrchestrator` always interacts with the platform layer via a consistent, message-based API, regardless of how the `Platform` struct internally sources and queues its events.

* **Action Dispatch (from `AppOrchestrator`):**
    * The `AppOrchestrator` sends commands to the PTY or UI driver by calling methods on its `Platform` reference (e.g., `platform.dispatch_pty_action(...)`, `platform.dispatch_ui_action(...)`).
    * These methods, in turn, send messages over internal channels to the tasks/components within the `Platform` struct that are responsible for handling PTY writes or UI rendering/window management.

This unified approach ensures that the `AppOrchestrator`'s logic remains clean and consistent, interacting with the platform layer through a well-defined set of event polling and action dispatching methods, while the complexities of actual event generation and action execution are encapsulated within each concrete `Platform` implementation.

## 4. Data & Control Flow

The primary data and control flow is as follows:

1.  **Initialization (`main.rs`):**
    * Selects (via `#[cfg]`) and instantiates the appropriate concrete `Platform` struct (e.g., `LinuxX11Platform::new()`). This `Platform` struct initializes its internal PTY channel, UI `Driver`, internal communication channels, and event production mechanisms. It returns itself (the `Platform` instance) and the initial `PlatformState`.
    * Creates the `TerminalEmulator`, `AnsiParser`, and `Renderer`.
    * Creates the `AppOrchestrator`, providing it with a reference to the concrete `Platform` struct, the initial `PlatformState`, and other components.

2.  **Main Event Loop (`AppOrchestrator`):**
    * The `AppOrchestrator` calls methods on the `Platform` reference to poll for events:
        * `platform.poll_pty_data()`: Retrieves data from the PTY (sourced from an internal channel fed by the PTY reader).
        * `platform.poll_ui_event()`: Retrieves UI events from the `Driver` (sourced from an internal channel fed by the UI event handler).
    * **PTY Data Processing:** PTY data is fed to the `AnsiParser`, which produces `AnsiCommand`s. These are wrapped into `EmulatorInput::Ansi` and passed to the `TerminalEmulator`.
    * **UI Event Processing:**
        * `BackendEvent::Key` events are first mapped against configured keybindings (from `Config.keybindings`). This mapping function (`map_key_event_to_action`) returns an `Option<UserInputAction>`.
        * If a binding matches (`Some(action)`), the `AppOrchestrator` inspects this `UserInputAction`.
            * If it's an application-level command (e.g., `RequestZoom`), the orchestrator handles it directly (e.g., by adjusting font scale and triggering a resize/redraw sequence).
            * Otherwise, the mapped `UserInputAction` is wrapped into `EmulatorInput::User` and sent to the `TerminalEmulator`.
        * If no binding matches (`None`), the original `BackendEvent::Key` details are used to construct a default `UserInputAction::KeyInput`, which is then wrapped into `EmulatorInput::User` and sent to the `TerminalEmulator`.
        * Other `BackendEvent`s (mouse, resize, focus) are translated into `EmulatorInput::User` or `EmulatorInput::Control` and typically sent to the `TerminalEmulator`.
    * **Emulator State Update:** The `TerminalEmulator` processes `EmulatorInput`s, updates its internal state, and may produce `EmulatorAction`s.
    * **Action Handling:** The `AppOrchestrator` processes `EmulatorAction`s:
        * `WritePty`: Translated to a `PtyActionCommand` and sent via `platform.dispatch_pty_action()`.
        * UI-related actions (e.g., `SetTitle`, `RingBell`, `SetCursorVisibility`, `CopyToClipboard`): Translated into `UiActionCommand`s and sent via `platform.dispatch_ui_action()`.
    * **Rendering:**
        * The `AppOrchestrator` gets a `TerminalSnapshot` from the `TerminalEmulator`.
        * The `Renderer` processes the snapshot and generates a `Vec<RenderCommand>`.
        * These render commands are packaged into a `UiActionCommand::Render(...)` and sent via `platform.dispatch_ui_action()`. The internal `Driver` within the `Platform` struct executes these commands.

## 5. Benefits of this Architecture

* **Strong Platform Encapsulation & RAII:** Each concrete `Platform` struct (e.g., `LinuxX11Platform`) is an RAII object that fully manages its own resources (PTY, UI `Driver`, internal event sources, communication channels). Cleanup is handled by its `Drop` implementation.
* **Clear `AppOrchestrator` Role:** The `AppOrchestrator` interacts with a consistent set of methods on the `Platform` reference for event polling and action dispatching, remaining agnostic to the internal implementation details of each platform.
* **Testability:** The `TerminalEmulator` and `Renderer` remain highly testable due to their pure-logic nature. Concrete `Platform` implementations can be tested by mocking their interactions with the OS or display system. The `AppOrchestrator` can be tested by providing a mock `Platform` struct (if a common `PlatformAccess` trait is used for its methods) or by testing against a controlled platform setup.
* **Flexible Event Production:** The internal event production mechanism within each `Platform` struct can be tailored to the target environment (e.g., threaded `epoll` for native, JS callbacks for WASM, or a pumped single-threaded poller) without changing the API exposed to the `AppOrchestrator`.
* **Configurable Keybindings:** A clear system for mapping key events to actions, supporting both compile-time defaults and runtime user configuration, with intelligent dispatch by the orchestrator.

## 6. Future Considerations

* **Platform Support (Wayland, macOS, WebAssembly):**
    * New platform support involves creating new concrete `Platform` structs (e.g., `WaylandPlatform`, `MacosPlatform`, `WasmPlatform`).
    * Each new `Platform` struct will implement its own `new()` constructor to set up its specific PTY-like communication (e.g., Wayland protocols, macOS PTY utilities, WebSockets for WASM), its UI `Driver` (e.g., Wayland client, Cocoa view, Canvas renderer), internal channels, and event production logic.
    * The `Drop` implementation for each new `Platform` struct will handle its specific resource cleanup.
    * The `AppOrchestrator`'s core logic remains largely unchanged, as it interacts with the `Platform` struct via the defined polling and dispatch methods.
* **`PlatformAccess` Trait (Optional):** If direct compile-time selection via `#[cfg]` and passing a concrete `&PlatformType` to the `AppOrchestrator` becomes cumbersome for testing across different platforms within the same test suite, or if a slim abstraction over the `poll_*` and `dispatch_*` methods is desired for the orchestrator's generics, a minimal `PlatformAccess` trait could be defined. Concrete `Platform` structs would implement this trait. For initial development, direct use of the concrete `Platform` type (selected by `#[cfg]`) in `AppOrchestrator` (or making `AppOrchestrator` generic over it) is likely simpler.
* **(Other future considerations from previous versions remain relevant).**
