use anyhow::Result;
use std::os::unix::io::RawFd;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_output, delegate_pointer,
    delegate_seat, delegate_shm, delegate_xdg_shell, delegate_xdg_window,
    output::{OutputHandler, OutputState},
    registry::RegistryState,
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, keysyms, Modifiers as SctkModifiers},
        pointer::{PointerEvent, PointerHandler},
        SeatHandler, SeatState, Capability as SeatCapability, SeatError,
    },
    // Explicitly import only what's directly used or needed for clarity
    shell::xdg::window::{Window, WindowConfigure, WindowHandler, WindowState},
    // XdgShell, XdgShellHandler, XdgShellState, XdgSurface, XdgSurfaceUserData, UserData will be fully qualified or handled by delegates
    shm::ShmHandler, // ShmState and SimplePool will be fully qualified
};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface::{self, WlSurface}},
    Connection, QueueHandle,
};
use wayland_protocols::xdg::shell::client::xdg_wm_base; // This is fine if used for xdg_wm_base.ping etc.
use xkbcommon::xkb::Keysym as XkbKeysym; // For KeyboardHandler keysyms type


use crate::platform::backends::{
    BackendEvent, Driver, FocusState, KeySymbol, Modifiers, MouseButton,
    PlatformState, RenderCommand, CursorVisibility,
    DEFAULT_WINDOW_WIDTH_CHARS, DEFAULT_WINDOW_HEIGHT_CHARS,
};
use crate::config::CONFIG;
use crate::color::Color;
use crate::glyph::AttrFlags;


// Define a simple struct for our Wayland driver
pub struct WaylandDriver {
    conn: Connection,
    event_queue: wayland_client::EventQueue<WaylandState>,
    queue_handle: QueueHandle<WaylandState>,
    wayland_state: WaylandState,
    window: Option<Window>,
    current_buffer: Option<SimplePool>,
    surface_contents: Option<WlSurface>, // To store the surface for drawing
    width_px: u16,
    height_px: u16,
    font_cell_width_px: usize,
    font_cell_height_px: usize,
    scale_factor: f64,
}

// Define the state for our Wayland event loop
struct WaylandState {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm_state: smithay_client_toolkit::shm::ShmState, // Fully qualified
    compositor_state: CompositorState,
    xdg_shell_state: smithay_client_toolkit::shell::xdg::XdgShellState, // Fully qualified
    xdg_window_state: WindowState,  // Uses imported WindowState from shell::xdg::window
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
    // TODO: Add fields for keyboard state, pointer state, etc.
    key_events: Vec<BackendEvent>,
    pointer_events: Vec<BackendEvent>,
    close_requested: bool,
    resized: bool, // Kept for now, might be superseded by pending_resize logic
    current_modifiers: SctkModifiers,
    pending_resize: Option<(u16, u16)>, // For pending resize from configure
}

impl WaylandDriver {
    fn calculate_initial_dimensions() -> (u16, u16, usize, usize, f64) {
        // For now, use default values.
        // TODO: Implement proper font metric calculation.
        let font_cell_width_px = CONFIG.font.width_px.unwrap_or(8) as usize;
        let font_cell_height_px = CONFIG.font.height_px.unwrap_or(16) as usize;
        let width_px = (DEFAULT_WINDOW_WIDTH_CHARS * font_cell_width_px) as u16;
        let height_px = (DEFAULT_WINDOW_HEIGHT_CHARS * font_cell_height_px) as u16;
        let scale_factor = 1.0; // Default scale factor
        (width_px, height_px, font_cell_width_px, font_cell_height_px, scale_factor)
    }
}

impl Driver for WaylandDriver {
    fn new() -> Result<Self> {
        let conn = Connection::connect_to_env().context("Failed to connect to Wayland display")?;
        let (globals, event_queue) = registry_queue_init::<WaylandState>(&conn).context("Failed to initialize registry queue")?;
        let queue_handle = event_queue.handle();

        let (initial_width_px, initial_height_px, font_cell_width_px, font_cell_height_px, scale_factor) =
            WaylandDriver::calculate_initial_dimensions();

        let mut wayland_state = WaylandState {
            registry_state: RegistryState::new(&globals),
            seat_state: SeatState::new(&globals, &queue_handle),
            output_state: OutputState::new(&globals, &queue_handle),
            shm_state: smithay_client_toolkit::shm::ShmState::new(&globals, &queue_handle), // Fully qualified
            compositor_state: CompositorState::bind(&globals, &queue_handle)
                .context("Failed to bind CompositorState")?,
            xdg_shell_state: smithay_client_toolkit::shell::xdg::XdgShellState::new(&globals, &queue_handle) // Fully qualified
                .context("Failed to create XdgShellState")?,
            xdg_window_state: WindowState::default(),
            keyboard: None,
            pointer: None,
            key_events: Vec::new(),
            pointer_events: Vec::new(),
            close_requested: false,
            resized: false,
            current_modifiers: SctkModifiers::default(),
        };

        // For Wayland, window creation is usually deferred until after initial setup.
        // We'll create it later or as part of a separate method if needed by the orchestrator.
        // For now, we'll initialize the driver and the window will be created when first shown/needed.

        Ok(Self {
            conn,
            event_queue,
            queue_handle,
            wayland_state,
            window: None,
            current_buffer: None,
            surface_contents: None,
            width_px: initial_width_px,
            height_px: initial_height_px,
            font_cell_width_px,
            font_cell_height_px,
            scale_factor,
        })
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        Some(self.conn.prepare_read().unwrap().connection_fd())
    }

    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        // Dispatch pending events
        self.event_queue.dispatch_pending(&mut self.wayland_state)
            .context("Failed to dispatch Wayland event queue")?;

        let mut events = Vec::new();
        events.append(&mut self.wayland_state.key_events);
        events.append(&mut self.wayland_state.pointer_events);

        if self.wayland_state.close_requested {
            events.push(BackendEvent::CloseRequested);
            self.wayland_state.close_requested = false; // Reset flag
        }

        // Handle pending resize
        if let Some((new_width, new_height)) = self.wayland_state.pending_resize.take() {
            if self.width_px != new_width || self.height_px != new_height {
                log::info!("WaylandDriver: Applying pending resize from {}x{} to {}x{}", self.width_px, self.height_px, new_width, new_height);
                self.width_px = new_width;
                self.height_px = new_height;
                self.current_buffer = None; // Force buffer recreation on next render
                events.push(BackendEvent::Resize { width_px: self.width_px, height_px: self.height_px });
            }
        }
        // Remove old self.wayland_state.resized logic if fully replaced by pending_resize
        // if self.wayland_state.resized {
        // events.push(BackendEvent::Resize { width_px: self.width_px, height_px: self.height_px });
        // self.wayland_state.resized = false;
        // }

        Ok(events)
    }

    fn get_platform_state(&self) -> PlatformState {
        PlatformState {
            event_fd: self.get_event_fd(),
            font_cell_width_px: self.font_cell_width_px,
            font_cell_height_px: self.font_cell_height_px,
            scale_factor: self.scale_factor,
            display_width_px: self.width_px,
            display_height_px: self.height_px,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        if self.window.is_none() {
            let surface = self.wayland_state.compositor_state.create_surface(&self.queue_handle);
            let window = self.wayland_state.xdg_shell_state.create_window(
                surface.clone(), // Clone surface for the window state
                self.wayland_state.xdg_window_state.clone(),
                &self.queue_handle,
                WindowConfigure::default()
                    .set_title(CONFIG.window.title.clone())
                    .set_app_id(CONFIG.window.app_id.clone())
                    .set_initial_size((self.width_px as i32, self.height_px as i32)),
            );
            self.window = Some(window.context("Failed to create XDG window")?);
            self.surface_contents = Some(surface); // Store the WlSurface for drawing
            log::info!("Wayland window and surface created. Dimensions: {}x{}", self.width_px, self.height_px);
        }

        let surface = self.surface_contents.as_ref().context("Surface not initialized for rendering")?;

        if self.width_px == 0 || self.height_px == 0 {
            log::warn!("Skipping rendering due to zero width ({}) or height ({}).", self.width_px, self.height_px);
            return Ok(());
        }

        let stride = self.width_px as i32 * 4; // 4 bytes per pixel (ARGB8888)
        // buffer_size is not directly used by create_buffer, but good for mmap calculation if done manually
        // let buffer_size = stride * self.height_px as i32;

        if self.current_buffer.is_none() {
            log::debug!("Creating new SHM pool for rendering. Dimensions: {}x{}", self.width_px, self.height_px);
            // The size passed to create_simple_pool is a hint or minimum,
            // SCTK will manage actual allocations.
            // A reasonable size would be for one buffer.
            // let pool_size = (stride * self.height_px as i32) as usize; // Not directly used by create_simple_pool
            let pool = self.wayland_state.shm_state.create_simple_pool(|fd, size| { // shm_state is now fully qualified type
                let mem = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        size, // size is provided by SCTK based on needs
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        fd,
                        0,
                    )
                };
                if mem == libc::MAP_FAILED {
                    // It's critical to handle mmap failure.
                    let error_msg = format!("mmap failed with error: {}", std::io::Error::last_os_error());
                    log::error!("{}", error_msg);
                    // panic!("{}", error_msg); // Panicking in a callback can be risky.
                                            // Consider a way to signal this error back to the main loop.
                    return (std::ptr::null_mut(), 0); // Return invalid mapping
                }
                (mem as *mut u8, size)
            }).context("Failed to create SHM pool")?;
            self.current_buffer = Some(pool);
        }

        let buffer_pool = self.current_buffer.as_mut().unwrap();

        // Create the buffer for the current frame.
        // The `create_buffer` method from `SimplePool` handles the actual memory allocation from the mmapped area.
        let (buffer, canvas) = match buffer_pool.create_buffer( // SimplePool is smithay_client_toolkit::shm::pool::SimplePool
            self.width_px as i32,
            self.height_px as i32,
            stride,
            wl_shm::Format::Argb8888,
        ) {
            Ok(b) => b,
            Err(e) => {
                // This can happen if the pool is exhausted or dimensions are too large.
                // Attempt to recreate the pool once if this fails.
                log::warn!("Failed to create buffer, attempting to recreate pool: {:?}", e);
                self.current_buffer = None; // Clear current pool
                // Re-create pool
                // let pool_size = (stride * self.height_px as i32) as usize; // Not directly used
                 let new_pool = self.wayland_state.shm_state.create_simple_pool(|fd, size| {
                    let mem = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
                    if mem == libc::MAP_FAILED { (std::ptr::null_mut(), 0) } else { (mem as *mut u8, size) }
                }).context("Failed to recreate SHM pool after buffer creation failure")?;
                self.current_buffer = Some(new_pool);
                // Try creating buffer again
                self.current_buffer.as_mut().unwrap().create_buffer( // SimplePool is smithay_client_toolkit::shm::pool::SimplePool
                    self.width_px as i32,
                    self.height_px as i32,
                    stride,
                    wl_shm::Format::Argb8888,
                ).context("Failed to create buffer even after pool recreation")?
            }
        };

        let current_width_px = self.width_px as usize;
        let current_height_px = self.height_px as usize;
        let font_width = self.font_cell_width_px;
        let font_height = self.font_cell_height_px;

        // Helper function (closure) for setting a pixel
        let set_pixel = |data: &mut [u8], x: usize, y: usize, color: Color| {
            if x < current_width_px && y < current_height_px {
                let offset = (y * current_width_px + x) * 4;
                data[offset] = (color.b * 255.0) as u8;
                data[offset + 1] = (color.g * 255.0) as u8;
                data[offset + 2] = (color.r * 255.0) as u8;
                data[offset + 3] = 0xFF; // Alpha (fully opaque)
            }
        };

        // Process RenderCommands
        // The canvas is mutable here.
        let mut initial_clear_done = false;
        for command in commands {
            match command {
                RenderCommand::ClearAll { bg } => {
                    for y_px in 0..current_height_px {
                        for x_px in 0..current_width_px {
                            set_pixel(canvas, x_px, y_px, bg);
                        }
                    }
                    initial_clear_done = true;
                    log::trace!("RenderCommand: ClearAll with color {:?}", bg);
                }
                RenderCommand::FillRect { x, y, width, height, color, .. } => {
                    let px_x = x * font_width;
                    let px_y = y * font_height;
                    let px_width = width * font_width;
                    let px_height = height * font_height;

                    for cur_y in px_y..(px_y + px_height).min(current_height_px) {
                        for cur_x in px_x..(px_x + px_width).min(current_width_px) {
                            set_pixel(canvas, cur_x, cur_y, color);
                        }
                    }
                    log::trace!("RenderCommand: FillRect at ({},{}) size {}x{} with color {:?}", x,y,width,height,color);
                }
                RenderCommand::DrawTextRun { x, y, text, fg: _fg, bg, .. } => {
                    // Simplified: Treat as FillRect with background color
                    let cell_width = text.chars().count(); // Simple char count for now
                    let px_x = x * font_width;
                    let px_y = y * font_height;
                    let px_width = cell_width * font_width;
                    let px_height = font_height; // Text runs are single-line

                    for cur_y in px_y..(px_y + px_height).min(current_height_px) {
                        for cur_x in px_x..(px_x + px_width).min(current_width_px) {
                            set_pixel(canvas, cur_x, cur_y, bg);
                        }
                    }
                    log::trace!("RenderCommand: DrawTextRun (simplified) at ({},{}) text '{}' with bg {:?}", x,y,text,bg);
                    // TODO: Actual text rendering using fg color and font.
                }
                // Other RenderCommand variants would be handled here
                _ => {
                    log::warn!("Unhandled RenderCommand: {:?}", command);
                }
            }
        }

        // If no ClearAll command was processed, fill with default background.
        // This ensures the buffer is always initialized.
        if !initial_clear_done {
            log::debug!("No ClearAll command found, filling with default background.");
            let default_bg = CONFIG.colors.primary.background;
            for y_px in 0..current_height_px {
                for x_px in 0..current_width_px {
                    set_pixel(canvas, x_px, y_px, default_bg);
                }
            }
        }

        surface.attach(Some(buffer.buffer()), 0, 0);
        surface.damage_buffer(0, 0, self.width_px as i32, self.height_px as i32); // Damage the entire buffer
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        if let Some(surface) = &self.surface_contents {
            surface.commit();
            // Flushing ensures that the commit request is sent to the compositor promptly.
            self.conn.flush().context("Failed to flush Wayland connection post-commit")?;

            // Dispatching events after commit allows processing frame callbacks, etc.
            // This helps synchronize rendering with the compositor.
            // Use a read with timeout to prevent blocking indefinitely if no events arrive.
            if self.conn.prepare_read().is_some() {
                 // Timeout is important here to avoid blocking if the compositor is slow or unresponsive
                self.event_queue.dispatch_pending(&mut self.wayland_state) // Consider using .read_events() and then .dispatch() for more control
                    .context("Failed to dispatch Wayland event queue post-commit")?;
            }
            log::trace!("WaylandDriver: present - surface committed and events dispatched.");
        } else {
            log::warn!("WaylandDriver: present called but no surface to commit.");
        }
        Ok(())
    }

    fn set_title(&mut self, title: &str) {
        if let Some(window) = &self.window {
            window.set_title(title.to_string());
        }
        log::debug!("WaylandDriver: set_title: {}", title);
    }

    fn bell(&mut self) {
        // TODO: Implement bell (e.g., using libcanberra or a visual flash)
        log::debug!("WaylandDriver: bell called. (Not yet implemented)");
    }

    fn set_cursor_visibility(&mut self, _visibility: CursorVisibility) {
        // TODO: Implement cursor visibility changes if possible with Wayland
        // This might involve creating a custom cursor surface.
        log::debug!("WaylandDriver: set_cursor_visibility called. (Not yet implemented)");
    }

    fn set_focus(&mut self, _focus_state: FocusState) {
        // Wayland handles focus automatically. This might be a no-op
        // or used for internal state tracking if needed.
        log::debug!("WaylandDriver: set_focus called. (Not yet implemented)");
    }

    fn cleanup(&mut self) -> Result<()> {
        // TODO: Implement cleanup (destroy window, disconnect from Wayland)
        if let Some(window) = self.window.take() {
            window.destroy();
        }
        log::info!("WaylandDriver cleanup. (Not yet fully implemented)");
        Ok(())
    }

    // Optional selection handling
    fn own_selection(&mut self, _selection_name_atom: u64, _text: String) {
        log::debug!("WaylandDriver: own_selection called. (Not yet implemented)");
    }

    fn request_selection_data(&mut self, _selection_name_atom: u64, _target_atom: u64) {
        log::debug!("WaylandDriver: request_selection_data called. (Not yet implemented)");
    }
}

// Implement the necessary handlers for smithay-client-toolkit
impl CompositorHandler for WaylandState {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
        // TODO: Handle scale factor changes
        log::debug!("Scale factor changed: {}", _new_factor);
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        // This is where you would typically redraw if needed,
        // but our rendering is driven by execute_render_commands.
    }
}

impl OutputHandler for WaylandState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl XdgShellHandler for WaylandState {
    fn xdg_shell_state(&mut self) -> &mut smithay_client_toolkit::shell::xdg::XdgShellState { // Fully qualified
        &mut self.xdg_shell_state
    }

    // new_xdg_surface is not directly part of XdgShellHandler in SCTK 0.17 in this way.
    // Surface creation and user data attachment is handled differently.
    // This method will be removed if delegate_xdg_shell! handles it,
    // or adjusted if it's a callback from a different mechanism.
    // For now, commenting out as it's a source of errors and likely misaligned with SCTK 0.17.
    /*
    fn new_xdg_surface(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: smithay_client_toolkit::shell::xdg::XdgSurface, // Fully qualified
    ) -> smithay_client_toolkit::shell::xdg::XdgSurfaceUserData<Self> { // Fully qualified
        smithay_client_toolkit::shell::xdg::XdgSurfaceUserData::new(WindowHandlerData::default(), Arc::new(Mutex::new(())))
    }
    */
}

// Custom struct to hold data for XdgSurfaceUserData if needed, or use a simple type like Arc<Mutex<()>>
#[derive(Default, Debug, Clone)]
struct WindowHandlerData {} // Placeholder
use std::sync::{Arc, Mutex}; // Add if not already present at top level

impl WindowHandler for WaylandState {
    // fn xdg_window_state(&mut self) -> &mut XdgWindowState { // Removed as per SCTK 0.17 delegate pattern
    //     &mut self.xdg_window_state
    // }

    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        self.close_requested = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _window: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        // Apply new size if provided
        if let Some(size) = configure.new_size {
            if size.0 > 0 && size.1 > 0 { // Ensure valid dimensions
                self.pending_resize = Some((size.0 as u16, size.1 as u16));
                log::debug!("WindowHandler: configure event received, new_size: {:?}, pending_resize set.", size);
            } else {
                log::warn!("WindowHandler: configure event received invalid size {:?}", size);
            }
        }
        // Acknowledge the configure event. This is important for XDG shell.
        // The window reference might not always be the one we are tracking if multiple windows were theoretically supported by this state.
        // However, for a single window app, this should be fine.
        if let Some(window) = _window {
            window.configure_ack(configure.serial);
            log::trace!("WindowHandler: Acknowledged configure serial: {}", configure.serial);
        } else {
            log::warn!("WindowHandler: configure called without a window reference to ack.");
        }
        // TODO: Handle other configure states (maximized, fullscreen, etc.)
    }
}


impl SeatHandler for WaylandState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _seat: wl_seat::WlSeat) {
        // Seat created, capabilities will be handled in new_capability
    }

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: &wl_seat::WlSeat, // seat is borrowed
        capability: SeatCapability,
    ) -> Result<(), SeatError> {
        match capability {
            SeatCapability::Keyboard => {
                let keyboard = self.seat_state
                    .get_keyboard_with_repeat(
                        qh,
                        seat, // Pass seat by reference
                        None, // Use default repeat configuration
                        self.registry_state.clone(),
                    )
                    .map_err(|_| SeatError::Other("Failed to create keyboard".to_string()))?; // Handle error appropriately
                self.keyboard = Some(keyboard);
                log::info!("Keyboard capability added.");
            }
            SeatCapability::Pointer => {
                let pointer = self.seat_state.get_pointer(qh, seat) // Pass seat by reference
                    .map_err(|_| SeatError::Other("Failed to create pointer".to_string()))?; // Handle error appropriately
                self.pointer = Some(pointer);
                log::info!("Pointer capability added.");
            }
            SeatCapability::Touch => {
                // Touch capability not handled in this example
                log::info!("Touch capability added but not handled.");
            }
            _unknown => {
                log::warn!("Unknown seat capability: {:?}", _unknown);
            }
        }
        Ok(())
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _seat: &wl_seat::WlSeat, // seat is borrowed
        capability: SeatCapability,
    ) -> Result<(), SeatError> {
        match capability {
            SeatCapability::Keyboard => {
                if let Some(keyboard) = self.keyboard.take() {
                    keyboard.release();
                }
                log::info!("Keyboard capability removed.");
            }
            SeatCapability::Pointer => {
                if let Some(pointer) = self.pointer.take() {
                    pointer.release();
                }
                log::info!("Pointer capability removed.");
            }
            SeatCapability::Touch => {
                log::info!("Touch capability removed but was not handled.");
            }
            _unknown => {
                 log::warn!("Attempting to remove unknown seat capability: {:?}", _unknown);
            }
        }
        Ok(())
    }

    fn remove_seat(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _seat: wl_seat::WlSeat) {
        // Seat removed
        self.keyboard = None;
        self.pointer = None;
        log::info!("Seat removed.");
    }
}


impl KeyboardHandler for WaylandState {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _surface: &wl_surface::WlSurface,
        _serial: u32,
        _raw: &[u32], // Raw keycodes
        keysyms: &[XkbKeysym], // Corrected type to xkbcommon::xkb::Keysym
    ) {
        // TODO: Handle keyboard focus enter
        log::debug!("Keyboard focus entered. Raw: {:?}, Keysyms: {:?}", _raw, keysyms);
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _surface: &wl_surface::WlSurface,
        _serial: u32,
    ) {
        // TODO: Handle keyboard focus leave
    }

    fn press_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _kbd: &wl_keyboard::WlKeyboard,
        _serial: u32,
        event: KeyEvent,
    ) {
        // event.keysym is RawKeysym (u32). Need to convert to keysyms::KEY_* constants for matching.
        // This requires an XKB context and state, which is complex here.
        // For now, we'll pass the raw keysym to sctk_keysym_to_keysymbol and handle it there.
        // A more robust solution involves xkbcommon state.
        let raw_keysym = event.keysym; // This is likely u32 (RawKeysym)
        let key_symbol = sctk_keysym_to_keysymbol(raw_keysym);
        let text = event.utf8.unwrap_or_default();
        let modifiers = sctk_modifiers_to_modifiers(self.current_modifiers);

        self.key_events.push(BackendEvent::Key {
            symbol: key_symbol,
            modifiers,
            text,
        });
    }

    fn release_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _kbd: &wl_keyboard::WlKeyboard,
        _serial: u32,
        _event: KeyEvent,
    ) {
        // Key release events are often not needed by terminals if using press_key
    }

    fn update_modifiers(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _kbd: &wl_keyboard::WlKeyboard,
        _serial: u32,
        modifiers: SctkModifiers,
    ) {
        self.current_modifiers = modifiers;
    }
}

// Parameter sctk_sym changed to raw_sym (u32), as KeyEvent.keysym is RawKeysym.
// We now use the `keysyms` module for matching.
fn sctk_keysym_to_keysymbol(raw_sym: u32) -> KeySymbol {
    // This function now takes a raw u32 keysym.
    // To use methods like is_printable() or to_unicode(), this raw value
    // would ideally be converted to an xkbcommon::Keysym object using an XKB context.
    // Without an XKB context here, we can only match against raw values if we know them,
    // or directly use the constants from the `keysyms` module.
    // For simplicity, directly use the constants from smithay_client_toolkit::seat::keyboard::keysyms (which re-exports xkbcommon keysyms)
    match raw_sym {
        keysyms::XKB_KEY_BackSpace => KeySymbol::Backspace,
        keysyms::XKB_KEY_Tab => KeySymbol::Tab,
        keysyms::XKB_KEY_Return => KeySymbol::Enter,
        keysyms::XKB_KEY_Escape => KeySymbol::Escape,
        keysyms::XKB_KEY_space => KeySymbol::Space,
        keysyms::XKB_KEY_Delete => KeySymbol::Delete,
        keysyms::XKB_KEY_Home => KeySymbol::Home,
        keysyms::XKB_KEY_End => KeySymbol::End,
        keysyms::XKB_KEY_Page_Up => KeySymbol::PageUp,
        keysyms::XKB_KEY_Page_Down => KeySymbol::PageDown,
        keysyms::XKB_KEY_Left => KeySymbol::Left,
        keysyms::XKB_KEY_Up => KeySymbol::Up,
        keysyms::XKB_KEY_Right => KeySymbol::Right,
        keysyms::XKB_KEY_Down => KeySymbol::Down,
        keysyms::XKB_KEY_F1 => KeySymbol::F(1),
        keysyms::XKB_KEY_F2 => KeySymbol::F(2),
        keysyms::XKB_KEY_F3 => KeySymbol::F(3),
        keysyms::XKB_KEY_F4 => KeySymbol::F(4),
        keysyms::XKB_KEY_F5 => KeySymbol::F(5),
        keysyms::XKB_KEY_F6 => KeySymbol::F(6),
        keysyms::XKB_KEY_F7 => KeySymbol::F(7),
        keysyms::XKB_KEY_F8 => KeySymbol::F(8),
        keysyms::XKB_KEY_F9 => KeySymbol::F(9),
        keysyms::XKB_KEY_F10 => KeySymbol::F(10),
        keysyms::XKB_KEY_F11 => KeySymbol::F(11),
        keysyms::XKB_KEY_F12 => KeySymbol::F(12),
        // For printable characters, proper conversion from raw keysym to char is needed.
        // This often involves an XKB state. A simplified approach might be:
        // ks if (ks >= keysyms::XKB_KEY_space && ks <= keysyms::XKB_KEY_asciitilde) => KeySymbol::Char(char::from_u32(ks).unwrap_or('?')),
        // However, this is a very rough approximation.
        // For now, we'll skip direct char conversion here and rely on event.utf8 from press_key.
        // The KeySymbol::Char variant will be mostly populated by event.utf8.
        // If event.utf8 is None, and it's a printable key, this mapping won't produce a Char.
        _ => KeySymbol::Unknown(raw_sym), // Store the raw keysym if unknown
    }
}

fn sctk_modifiers_to_modifiers(sctk_mods: SctkModifiers) -> Modifiers {
    let mut mods = Modifiers::empty();
    if sctk_mods.ctrl {
        mods |= Modifiers::CTRL;
    }
    if sctk_mods.alt {
        mods |= Modifiers::ALT;
    }
    if sctk_mods.shift {
        mods |= Modifiers::SHIFT;
    }
    // SCTK doesn't directly map to a "Super" or "Logo" modifier in its base struct.
    // This would require more platform-specific handling if needed.
    mods
}


impl PointerHandler for WaylandState {
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            match event.kind {
                smithay_client_toolkit::seat::pointer::PointerEventKind::Enter { serial: _ } => {
                    // TODO: Handle pointer enter (e.g., set cursor icon)
                }
                smithay_client_toolkit::seat::pointer::PointerEventKind::Leave { serial: _ } => {
                    // TODO: Handle pointer leave
                }
                smithay_client_toolkit::seat::pointer::PointerEventKind::Motion { .. } => {
                    // We could generate MouseMove events here if needed,
                    // but terminals often only care about button presses with coordinates.
                }
                smithay_client_toolkit::seat::pointer::PointerEventKind::Press { button, .. } => {
                    let btn = match button {
                        0x110 => MouseButton::Left, // BTN_LEFT
                        0x111 => MouseButton::Right, // BTN_RIGHT
                        0x112 => MouseButton::Middle, // BTN_MIDDLE
                        _ => MouseButton::Other(button as u8),
                    };
                    self.pointer_events.push(BackendEvent::MouseButtonPress {
                        button: btn,
                        x: event.position.0 as u16, // TODO: Convert to cell coords
                        y: event.position.1 as u16, // TODO: Convert to cell coords
                        modifiers: sctk_modifiers_to_modifiers(self.current_modifiers),
                    });
                }
                smithay_client_toolkit::seat::pointer::PointerEventKind::Release { button, .. } => {
                     let btn = match button {
                        0x110 => MouseButton::Left, // BTN_LEFT
                        0x111 => MouseButton::Right, // BTN_RIGHT
                        0x112 => MouseButton::Middle, // BTN_MIDDLE
                        _ => MouseButton::Other(button as u8),
                    };
                    self.pointer_events.push(BackendEvent::MouseButtonRelease {
                        button: btn,
                        x: event.position.0 as u16, // TODO: Convert to cell coords
                        y: event.position.1 as u16, // TODO: Convert to cell coords
                        modifiers: sctk_modifiers_to_modifiers(self.current_modifiers),
                    });
                }
                smithay_client_toolkit::seat::pointer::PointerEventKind::Axis { .. } => {
                    // TODO: Handle scroll events (convert to MouseButton::ScrollUp/Down)
                }
            }
        }
    }
}

impl ShmHandler for WaylandState {
    fn shm_state(&mut self) -> &mut smithay_client_toolkit::shm::ShmState { // Fully qualified
        &mut self.shm_state
    }
}

// Delegate macro calls for smithay-client-toolkit
delegate_compositor!(WaylandState);
delegate_output!(WaylandState);
delegate_shm!(WaylandState);
delegate_seat!(WaylandState);
delegate_keyboard!(WaylandState);
delegate_pointer!(WaylandState);
delegate_xdg_shell!(WaylandState);
delegate_xdg_window!(WaylandState);

// Implement the RegistryHandler trait for WaylandState
// This is crucial for smithay-client-toolkit to discover and bind globals.
// REMOVED: impl wayland_client::globals::GlobalListContents for WaylandState
// As GlobalListContents is a struct, not a trait. ProvidesRegistryState is used instead.

impl smithay_client_toolkit::registry::ProvidesRegistryState for WaylandState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    smithay_client_toolkit::registry_handlers!();
}

// Add a basic context import for anyhow
use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    // This test requires a running Wayland compositor to succeed.
    // It might be best to run it manually or in a CI environment with Wayland.
    // Add #[ignore] if it causes issues in environments without Wayland.
    #[test]
    #[ignore] // Ignoring by default as it requires a Wayland compositor
    fn test_wayland_driver_new() -> Result<()> {
        // Ensure logger is initialized for tests, if driver relies on it.
        // env_logger::builder().is_test(true).try_init().ok();

        match WaylandDriver::new() {
            Ok(mut driver) => {
                info!("WaylandDriver::new() test successful, driver instance created.");
                // Perform a minimal operation if possible, e.g., getting initial state
                let initial_state = driver.get_platform_state();
                info!("Initial platform state: {:?}", initial_state);
                // Call cleanup to release resources
                driver.cleanup()?;
                Ok(())
            }
            Err(e) => {
                // If a Wayland compositor is not running, this test will likely fail here.
                // We can check for specific error kinds if needed, but for a smoke test,
                // just failing the test is okay.
                error!("WaylandDriver::new() test failed: {:?}", e);
                Err(e)
            }
        }
    }
}
