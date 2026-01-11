use crate::ansi::commands::AnsiCommand;
use crate::color::Color;
use crate::config::Config;
use crate::glyph::Glyph;
use crate::io::PtyCommand;
use crate::term::TerminalEmulator;
use actor_scheduler::{Actor, ActorScheduler, Message, ActorStatus};
use pixelflow_core::{
    Add, And, At, Discrete, Ge, Le, Manifold, ManifoldExt, Mul, Select, Sub, W, X, Y, Z,
};
use pixelflow_graphics::fonts::loader::{LoadedFont, MmapSource};
use pixelflow_graphics::ColorCube as PlatformColorCube;
use pixelflow_graphics::{CachedGlyph, GlyphCache, Positioned, SpatialBSP};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::AppData;
use pixelflow_runtime::api::public::EngineHandle;
use pixelflow_runtime::{
    EngineEventControl, EngineEventData, EngineEventManagement,
};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

/// Path to the embedded font file
const FONT_PATH: &str = "pixelflow-graphics/assets/NotoSansMono-Regular.ttf";

/// Bounded glyph manifold (returns coverage in [0,1], 0 if out of bounds).
/// Select<Cond, CachedGlyph, f32>
type BoundedGlyph =
    Select<And<And<And<Ge<X, f32>, Le<X, f32>>, Ge<Y, f32>>, Le<Y, f32>>, CachedGlyph, f32>;

/// Positioned glyph manifold
type PositionedGlyph = At<Sub<X, f32>, Sub<Y, f32>, Z, W, BoundedGlyph>;

/// Lerp manifold: X + Z * (Y - X)
/// Maps X -> background, Y -> foreground, Z -> coverage
type LerpManifold = Add<X, Mul<Z, Sub<Y, X>>>;

/// Blended channel: Lerp(coverage, background, foreground)
type BlendedChannel = At<f32, f32, PositionedGlyph, f32, LerpManifold>;

/// Concrete leaf type: a terminal cell with background and foreground blending.
type TerminalCellLeaf = At<
    BlendedChannel,    // R
    BlendedChannel,    // G
    BlendedChannel,    // B
    f32,               // A
    PlatformColorCube, // M
>;

/// Terminal application implementing Actor trait.
///
/// Receives engine events (frame requests, input) and responds with rendered
/// terminal content via the engine handle.
pub struct TerminalApp {
    pub emulator: TerminalEmulator,
    pty_tx: SyncSender<PtyCommand>,
    pub pty_rx: Receiver<Vec<AnsiCommand>>,
    config: Config,
    engine_tx: EngineHandle,
    /// Memory-mapped font file.
    loaded_font: Arc<LoadedFont<MmapSource>>,
    /// Cached rasterized glyphs.
    glyph_cache: GlyphCache,
}

/// Parameters for constructing a TerminalApp.
pub struct TerminalAppParams {
    /// Terminal emulator instance.
    pub emulator: TerminalEmulator,
    /// Channel to send commands to PTY (writes and resizes).
    pub pty_tx: SyncSender<PtyCommand>,
    /// Channel to receive parsed ANSI commands from PTY.
    pub pty_rx: Receiver<Vec<AnsiCommand>>,
    /// Application configuration.
    pub config: Config,
    /// Unregistered engine handle (app will call register()).
    pub unregistered_engine: pixelflow_runtime::UnregisteredEngineHandle,
    /// Window configuration for registration.
    pub window_config: pixelflow_runtime::WindowConfig,
}

impl TerminalApp {
    /// Helper to create a positioned terminal cell with background blending.
    ///
    /// Composition: bg + cov * (fg - bg)
    #[inline(always)]
    fn make_terminal_cell(
        glyph: CachedGlyph,
        offset_x: f32,
        offset_y: f32,
        cell_width: f32,
        cell_height: f32,
        fg: [f32; 4],
        bg: [f32; 4],
    ) -> impl Manifold<Output = Discrete> + Clone {
        // Create bounded glyph in local coordinates [0, width] x [0, height]
        // IMPORTANT: Bound BEFORE translating to avoid evaluating every glyph for every pixel
        let cond = X.ge(0.0) & X.le(cell_width) & Y.ge(0.0) & Y.le(cell_height);
        let bounded = Select {
            cond,
            if_true: glyph,
            if_false: 0.0f32,
        };

        // Position the glyph in global coordinates
        let positioned = At {
            inner: bounded,
            x: X - offset_x,
            y: Y - offset_y,
            z: Z,
            w: W,
        };

        // Lerp expression: X + Z * (Y - X)
        // We map: X -> bg, Y -> fg, Z -> coverage
        let lerp = X + Z * (Y - X);

        // Blend each channel
        // Note: we use positioned (which is a Manifold) as the Z coordinate
        // The At combinator expects manifolds for coordinates.
        // Also note: f32 constants are auto-lifted to Manifolds.
        let r = At {
            inner: lerp,
            x: bg[0],
            y: fg[0],
            z: positioned.clone(),
            w: 0.0,
        };
        let g = At {
            inner: lerp,
            x: bg[1],
            y: fg[1],
            z: positioned.clone(),
            w: 0.0,
        };
        let b = At {
            inner: lerp,
            x: bg[2],
            y: fg[2],
            z: positioned,
            w: 0.0,
        };

        // Pack into platform color cube
        let blended = At {
            inner: PlatformColorCube::default(),
            x: r,
            y: g,
            z: b,
            w: 1.0,
        };

        // Only show the cell (including background) when pixel is in bounds
        // When out of bounds, return transparent black
        let in_bounds = X.ge(offset_x)
            & X.le(offset_x + cell_width)
            & Y.ge(offset_y)
            & Y.le(offset_y + cell_height);

        // Create a transparent black color (all zeros)
        let transparent = At {
            inner: PlatformColorCube::default(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };

        Select {
            cond: in_bounds,
            if_true: blended,
            if_false: transparent,
        }
    }

    /// Creates a new terminal app (internal - use spawn_terminal_app instead).
    fn new_registered(params: TerminalAppParamsRegistered) -> Self {
        // Memory-map the font file
        // Try workspace-relative path first, then crate-relative (for tests)
        let source = MmapSource::open(FONT_PATH)
            .or_else(|_| MmapSource::open(&format!("../{}", FONT_PATH)))
            .expect("Failed to open font file");

        let loaded_font = Arc::new(LoadedFont::new(source).expect("Failed to parse font"));

        // Create glyph cache and pre-warm with ASCII
        let cell_height = params.config.appearance.cell_height_px as f32;
        let mut glyph_cache = GlyphCache::with_capacity(128);
        glyph_cache.warm_ascii(&loaded_font.font(), cell_height);

        Self {
            emulator: params.emulator,
            pty_tx: params.pty_tx,
            pty_rx: params.pty_rx,
            config: params.config,
            engine_tx: params.engine_tx,
            loaded_font,
            glyph_cache,
        }
    }

    /// Build a render manifold from the current terminal state.
    fn build_manifold(
        &mut self,
    ) -> (
        Arc<dyn Manifold<Output = Discrete> + Send + Sync>,
        (f32, f32),
    ) {
        // Get terminal snapshot
        let snapshot = match self.emulator.get_render_snapshot() {
            Some(s) => s,
            None => {
                let (r, g, b, a) = self.config.colors.background.to_f32_rgba();
                return (
                    Arc::new(At {
                        inner: PlatformColorCube::default(),
                        x: r,
                        y: g,
                        z: b,
                        w: a,
                    }),
                    (0.0, 0.0),
                );
            }
        };

        let (cols, rows) = snapshot.dimensions;
        let cell_width = snapshot.cell_width_px as f32;
        let cell_height = snapshot.cell_height_px as f32;
        let grid_width = cols as f32 * cell_width;
        let grid_height = rows as f32 * cell_height;

        // Default colors
        let default_fg = self.config.colors.foreground;
        let default_bg = self.config.colors.background;

        // Build 2-level BSP: Vertical (Rows) -> Horizontal (Cells)
        let mut row_items = Vec::new();

        for row in 0..rows {
            let line = &snapshot.lines[row];
            let mut cell_items = Vec::new();

            for col in 0..cols {
                let glyph = &line.cells[col];

                let (ch, fg_color, cell_bg) = match glyph {
                    Glyph::Single(cc) | Glyph::WidePrimary(cc) => {
                        let fg = if cc.attr.fg == Color::Default {
                            default_fg
                        } else {
                            cc.attr.fg
                        };
                        let bg = if cc.attr.bg == Color::Default {
                            default_bg
                        } else {
                            cc.attr.bg
                        };
                        (cc.c, fg, bg)
                    }
                    Glyph::WideSpacer => continue, // Skip spacers
                };

                /*
                // Skip empty cells
                if ch == ' ' || ch == '\0' {
                    continue;
                }
                */

                // Get cached glyph - glyph_scaled now accounts for descenders
                if let Some(cached) =
                    self.glyph_cache
                        .get(&self.loaded_font.font(), ch, cell_height)
                {
                    let (fg_r, fg_g, fg_b, fg_a) = fg_color.to_f32_rgba();
                    let (bg_r, bg_g, bg_b, bg_a) = cell_bg.to_f32_rgba();

                    let x = col as f32 * cell_width;
                    let y = row as f32 * cell_height;

                    cell_items.push(Positioned {
                        bounds: (x, y, x + cell_width, y + cell_height),
                        leaf: Self::make_terminal_cell(
                            cached,
                            x,
                            y,
                            cell_width,
                            cell_height,
                            [fg_r, fg_g, fg_b, fg_a],
                            [bg_r, bg_g, bg_b, bg_a],
                        ),
                    });
                }
            }

            // If row has cells, wrap them in a horizontal BSP and add to row list
            if !cell_items.is_empty() {
                let y_min = row as f32 * cell_height;
                let y_max = y_min + cell_height;

                row_items.push(Positioned {
                    bounds: (0.0, y_min, grid_width, y_max),
                    leaf: SpatialBSP::from_positioned(cell_items),
                });
            }
        }

        // If no rows have content, just return background
        if row_items.is_empty() {
            let (r, g, b, a) = default_bg.to_f32_rgba();
            return (
                Arc::new(At {
                    inner: PlatformColorCube::default(),
                    x: r,
                    y: g,
                    z: b,
                    w: a,
                }),
                (grid_width, grid_height),
            );
        }

        // Build top-level vertical BSP from row items
        let top_bsp = SpatialBSP::from_positioned(row_items);
        (Arc::new(top_bsp), (grid_width, grid_height))
    }

    /// Send a rendered frame to the engine.
    fn send_frame(&mut self) {
        let (manifold, grid_bounds) = self.build_manifold();

        // 1. Create default background manifold
        let default_bg = self.config.colors.background;
        let (r, g, b, a) = default_bg.to_f32_rgba();
        let background = At {
            inner: PlatformColorCube::default(),
            x: r,
            y: g,
            z: b,
            w: a,
        };

        // 2. Wrap SpatialBSP in a Select that clips to grid bounds
        // cond = (x >= 0) & (x < grid_width) & (y >= 0) & (y < grid_height)
        let (gw, gh) = grid_bounds;
        let cond = X.ge(0.0) & X.lt(gw) & Y.ge(0.0) & Y.lt(gh);

        eprintln!("[TERM] Grid bounds: {}x{}", gw, gh);
        eprintln!("[TERM] about to create Select scene");
        let scene = Select {
            cond,
            if_true: manifold,
            if_false: background,
        };

        eprintln!("[TERM] Select created, wrapping in Arc");
        let data = AppData::RenderSurface(Arc::new(scene));
        eprintln!("[TERM] Scene Arc created");
        eprintln!("[TERM] sending to engine");
        if let Err(e) = self
            .engine_tx
            .send(Message::Data(EngineData::FromApp(data)))
        {
            log::warn!("Failed to send frame to engine: {}", e);
        }
        eprintln!("[TERM] send_frame() complete");
    }
}

impl Actor<EngineEventData, EngineEventControl, EngineEventManagement> for TerminalApp {
    fn handle_data(&mut self, data: EngineEventData) {
        match data {
            EngineEventData::RequestFrame { .. } => {
                // Engine is requesting a frame - build and send it
                self.send_frame();
            }
        }
    }

    fn handle_control(&mut self, ctrl: EngineEventControl) {
        match ctrl {
            EngineEventControl::WindowCreated {
                id,
                width_px,
                height_px,
                scale,
            } => {
                log::info!(
                    "[TERM] Window created: id={}, {}x{} pixels, scale={}",
                    id.0,
                    width_px,
                    height_px,
                    scale
                );

                // Window is now ready - send initial frame to start VSync loop
                self.send_frame();
            }
            EngineEventControl::Resized {
                id: _,
                width_px,
                height_px,
            } => {
                use crate::term::{ControlEvent, EmulatorAction, EmulatorInput};
                // Convert u32 pixels to u16 for ControlEvent
                // Saturate at u16::MAX to prevent overflow panics
                let width_u16 = width_px.min(u16::MAX as u32) as u16;
                let height_u16 = height_px.min(u16::MAX as u32) as u16;

                let input = EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_u16,
                    height_px: height_u16,
                });

                // Process the resize and handle the resulting action
                if let Some(action) = self.emulator.interpret_input(input) {
                    if let EmulatorAction::ResizePty { cols, rows } = action {
                        // Send resize command to PTY write thread
                        if let Err(e) = self.pty_tx.send(PtyCommand::Resize { cols, rows }) {
                            log::warn!("Failed to send PTY resize command: {}", e);
                        }
                    }
                }

                // Request a redraw after resize
                self.send_frame();
            }
            EngineEventControl::CloseRequested => {
                unimplemented!("CloseRequested handler - need to cleanup and shutdown");
            }
            EngineEventControl::ScaleChanged { id, scale } => {
                log::warn!(
                    "ScaleChanged: id={}, scale={} - NOT IMPLEMENTED",
                    id.0,
                    scale
                );
                unimplemented!("ScaleChanged handler - need to adjust font sizes and redraw");
            }
        }
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) {
        match mgmt {
            EngineEventManagement::KeyDown { key, mods, text } => {
                use crate::term::{EmulatorAction, EmulatorInput, UserInputAction};

                let input = EmulatorInput::User(UserInputAction::KeyInput {
                    symbol: key,
                    modifiers: mods,
                    text,
                });

                if let Some(action) = self.emulator.interpret_input(input) {
                    match action {
                        EmulatorAction::WritePty(bytes) => {
                            if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                                log::warn!("Failed to send input to PTY: {}", e);
                            }
                        }
                        EmulatorAction::Quit => {
                            // Handle quit - send quit to engine
                            use pixelflow_runtime::api::public::AppManagement;
                            self.engine_tx
                                .send(Message::Management(AppManagement::Quit))
                                .expect("Failed to send Quit to engine");
                        }
                        EmulatorAction::SetTitle(_) => {
                            unimplemented!("EmulatorAction::SetTitle not yet implemented");
                        }
                        EmulatorAction::RingBell => {
                            unimplemented!("EmulatorAction::RingBell not yet implemented");
                        }
                        EmulatorAction::RequestRedraw => {
                            self.send_frame();
                        }
                        EmulatorAction::SetCursorVisibility(_) => {
                            unimplemented!(
                                "EmulatorAction::SetCursorVisibility not yet implemented"
                            );
                        }
                        EmulatorAction::CopyToClipboard(_) => {
                            unimplemented!("EmulatorAction::CopyToClipboard not yet implemented");
                        }
                        EmulatorAction::RequestClipboardContent => {
                            unimplemented!(
                                "EmulatorAction::RequestClipboardContent not yet implemented"
                            );
                        }
                        EmulatorAction::ResizePty { cols, rows } => {
                            // Send resize command to PTY write thread
                            if let Err(e) = self.pty_tx.send(PtyCommand::Resize { cols, rows }) {
                                log::warn!("Failed to send PTY resize command: {}", e);
                            }
                        }
                    }
                }
            }
            EngineEventManagement::MouseClick { button, x, y } => {
                let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                log::trace!(
                    "Mouse click: button={:?} at cell ({}, {})",
                    button,
                    col,
                    row
                );
                // TODO: Handle mouse tracking modes, selection
            }
            EngineEventManagement::MouseRelease { button, x, y } => {
                let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                log::trace!(
                    "Mouse release: button={:?} at cell ({}, {})",
                    button,
                    col,
                    row
                );
                // TODO: Handle mouse tracking modes, end selection
            }
            EngineEventManagement::MouseMove { x, y, mods: _ } => {
                let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                log::trace!("Mouse move: cell ({}, {})", col, row);
                // TODO: Handle mouse tracking modes, drag selection
            }
            EngineEventManagement::MouseScroll {
                x: _,
                y: _,
                dx,
                dy,
                mods: _,
            } => {
                log::trace!("Mouse scroll: delta=({}, {})", dx, dy);
                // TODO: Implement scrollback navigation
                // Should modify viewport offset and trigger redraw
            }
            EngineEventManagement::FocusGained => {
                log::trace!("Focus gained");
                // Some applications care about focus for bracketed paste mode
                // Could send \x1b[I if bracketed paste is enabled
            }
            EngineEventManagement::FocusLost => {
                log::trace!("Focus lost");
                // Some applications care about focus for bracketed paste mode
                // Could send \x1b[O if bracketed paste is enabled
            }
            EngineEventManagement::Paste(text) => {
                log::trace!("Paste: {} bytes", text.len());
                // Send pasted text to PTY
                self.pty_tx
                    .send(PtyCommand::Write(text.into_bytes()))
                    .expect("Failed to send paste to PTY");
            }
        }
    }

    fn park(&mut self, _hint: ActorStatus) -> ActorStatus {
        // Drain any queued ANSI commands from PTY
        use crate::term::EmulatorInput;

        let mut found_data = false;
        while let Ok(commands) = self.pty_rx.try_recv() {
            found_data = true;
            for cmd in commands {
                self.emulator.interpret_input(EmulatorInput::Ansi(cmd));
            }
        }

        // Send frames eagerly when PTY data arrives (out of band from VSync).
        // This ensures low latency, especially at lower frame rates.
        if found_data {
            self.send_frame();
        }

        // If we found data, keep polling. Otherwise block on actor messages.
        if found_data {
            ActorStatus::Busy
        } else {
            ActorStatus::Idle
        }
    }
}

/// Creates terminal app and spawns it in a thread.
///
/// This function handles registration atomically:
/// 1. Creates the app actor's channel
/// 2. Registers the app with the engine (sends RegisterApp + CreateWindow)
/// 3. Spawns the app thread with the registered engine handle
pub fn spawn_terminal_app(
    params: TerminalAppParams,
) -> std::io::Result<(
    actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
)> {
    // Create app actor's channel
    let (app_handle, mut app_rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(10, 128);

    // Register with engine (sends RegisterApp + CreateWindow atomically)
    use pixelflow_runtime::WindowDescriptor;
    let window_descriptor = WindowDescriptor {
        width: params.window_config.width,
        height: params.window_config.height,
        title: params.window_config.title.clone(),
        resizable: true,
    };

    let app_arc = std::sync::Arc::new(app_handle.clone());
    let engine_tx = params.unregistered_engine
        .register(app_arc, window_descriptor)
        .expect("Failed to register app with engine");

    log::info!("[TERM] App registered with engine, window creation requested");

    // Create app with registered engine handle
    let app_params_registered = TerminalAppParamsRegistered {
        emulator: params.emulator,
        pty_tx: params.pty_tx,
        pty_rx: params.pty_rx,
        config: params.config,
        engine_tx,
    };

    let mut app = TerminalApp::new_registered(app_params_registered);

    // Spawn app thread
    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_handle, handle))
}

/// Parameters after registration (internal use).
struct TerminalAppParamsRegistered {
    emulator: TerminalEmulator,
    pty_tx: SyncSender<PtyCommand>,
    pty_rx: Receiver<Vec<AnsiCommand>>,
    config: Config,
    engine_tx: EngineHandle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::AnsiCommand;
    use crate::io::PtyCommand;
    use crate::term::{EmulatorInput, TerminalEmulator, UserInputAction};
    use actor_scheduler::{Actor, ActorStatus};
    use pixelflow_runtime::input::{KeySymbol, Modifiers};
    use pixelflow_runtime::{EngineEventControl, EngineEventManagement, WindowId};
    use std::sync::mpsc::{Receiver, SyncSender};

    // Define a DummyPixel struct for testing
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    struct DummyPixel;
    impl pixelflow_graphics::render::Pixel for DummyPixel {
        fn from_u32(_: u32) -> Self {
            Self
        }
        fn to_u32(self) -> u32 {
            0
        }
        fn from_rgba(_r: f32, _g: f32, _b: f32, _a: f32) -> Self {
            Self
        }
    }

    // Helper to create a test instance
    fn create_test_app() -> (
        TerminalApp,
        Receiver<PtyCommand>,
        SyncSender<Vec<AnsiCommand>>,
        pixelflow_runtime::api::private::EngineActorHandle,
    ) {
        let emulator = TerminalEmulator::new(80, 24);
        let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(128);
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel(128);

        // Create a dummy engine handle
        let (engine_tx, _) = actor_scheduler::ActorScheduler::new(10, 10);

        let config = Config::default();
        let params = TerminalAppParamsRegistered {
            emulator,
            pty_tx,
            pty_rx: cmd_rx,
            config,
            engine_tx: EngineHandle::new_for_test(engine_tx.clone()),
        };
        let app = TerminalApp::new_registered(params);

        (app, pty_rx, cmd_tx, engine_tx)
    }

    #[test]
    fn test_handle_control_resize() {
        let (mut app, pty_rx, _cmd_tx, _) = create_test_app();

        // Initial size is 80x24
        use crate::term::TerminalInterface;
        let snapshot_initial = app.emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(snapshot_initial.dimensions, (80, 24));

        // Send resize event
        // Default config: cell width 10, height 16.
        // Resize to 1000x800 -> 100x50 cells.
        let resize_event = EngineEventControl::Resized {
            id: WindowId(0),
            width_px: 1000,
            height_px: 800,
        };
        app.handle_control(resize_event);

        // Verify resize via snapshot
        let snapshot_new = app.emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(
            snapshot_new.dimensions,
            (100, 50),
            "Emulator should have resized to 100x50"
        );

        // Verify PTY resize command was sent
        let cmd = pty_rx.try_recv().expect("Should receive resize command");
        assert_eq!(
            cmd,
            PtyCommand::Resize {
                cols: 100,
                rows: 50
            },
            "PTY resize command should match new dimensions"
        );
    }

    #[test]
    fn test_handle_management_keydown() {
        let (mut app, pty_rx, _cmd_tx, _) = create_test_app();

        // Simulate KeyDown
        let key_event = EngineEventManagement::KeyDown {
            key: KeySymbol::Char('a'),
            mods: Modifiers::empty(),
            text: Some("a".to_string()),
        };

        app.handle_management(key_event);

        // We expect 'a' to be sent to PTY wrapped in PtyCommand::Write
        let received = pty_rx.try_recv();
        assert!(received.is_ok(), "Should receive data on PTY channel");
        let cmd = received.unwrap();
        assert_eq!(cmd, PtyCommand::Write(vec![b'a']));
    }
}
