use crate::ansi::commands::AnsiCommand;
use crate::color::Color;
use crate::config::Config;
use crate::glyph::Glyph;
use crate::io::PtyCommand;
use crate::term::TerminalEmulator;
use actor_scheduler::{Actor, ActorScheduler, Message, ParkHint};
use pixelflow_core::{Add, At, Discrete, Manifold, ManifoldExt, Mul, Select, Sub, W, X, Y, Z, Ge, Le, And};
use pixelflow_graphics::fonts::loader::{LoadedFont, MmapSource};
use pixelflow_graphics::render::Pixel;
use pixelflow_graphics::{CachedGlyph, GlyphCache, Positioned, SpatialBSP};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::AppData;
use pixelflow_runtime::{
    EngineActorHandle, EngineEventControl, EngineEventData, EngineEventManagement,
};
use pixelflow_graphics::ColorCube as PlatformColorCube;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender};

/// Path to the embedded font file
const FONT_PATH: &str = "pixelflow-graphics/assets/NotoSansMono-Regular.ttf";

/// Bounded glyph manifold (returns coverage in [0,1], 0 if out of bounds).
/// Select<Cond, CachedGlyph, f32>
type BoundedGlyph = Select<
    And<And<And<Ge<X, f32>, Le<X, f32>>, Ge<Y, f32>>, Le<Y, f32>>,
    CachedGlyph,
    f32
>;

/// Positioned glyph manifold
type PositionedGlyph = At<Sub<X, f32>, Sub<Y, f32>, Z, W, BoundedGlyph>;

/// Lerp manifold: X + Z * (Y - X)
/// Maps X -> background, Y -> foreground, Z -> coverage
type LerpManifold = Add<X, Mul<Z, Sub<Y, X>>>;

/// Blended channel: Lerp(coverage, background, foreground)
type BlendedChannel = At<f32, f32, PositionedGlyph, f32, LerpManifold>;

/// Concrete leaf type: a terminal cell with background and foreground blending.
type TerminalCellLeaf = At<
    BlendedChannel,     // R
    BlendedChannel,     // G
    BlendedChannel,     // B
    f32,                // A
    PlatformColorCube,  // M
>;

/// Terminal application implementing Actor trait.
///
/// Receives engine events (frame requests, input) and responds with rendered
/// terminal content via the engine handle.
pub struct TerminalApp<P: Pixel> {
    pub emulator: TerminalEmulator,
    pty_tx: SyncSender<PtyCommand>,
    pub pty_rx: Receiver<Vec<AnsiCommand>>,
    config: Config,
    engine_tx: EngineActorHandle<P>,
    /// Memory-mapped font file.
    loaded_font: Arc<LoadedFont<MmapSource>>,
    /// Cached rasterized glyphs.
    glyph_cache: GlyphCache,
}

/// Parameters for constructing a TerminalApp.
pub struct TerminalAppParams<P: Pixel> {
    /// Terminal emulator instance.
    pub emulator: TerminalEmulator,
    /// Channel to send commands to PTY (writes and resizes).
    pub pty_tx: SyncSender<PtyCommand>,
    /// Channel to receive parsed ANSI commands from PTY.
    pub pty_rx: Receiver<Vec<AnsiCommand>>,
    /// Application configuration.
    pub config: Config,
    /// Handle to the engine actor.
    pub engine_tx: EngineActorHandle<P>,
}


impl<P: Pixel> TerminalApp<P> {
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
        let r = At { inner: lerp, x: bg[0], y: fg[0], z: positioned.clone(), w: 0.0 };
        let g = At { inner: lerp, x: bg[1], y: fg[1], z: positioned.clone(), w: 0.0 };
        let b = At { inner: lerp, x: bg[2], y: fg[2], z: positioned, w: 0.0 };

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
        let in_bounds = X.ge(offset_x) & X.le(offset_x + cell_width) &
                       Y.ge(offset_y) & Y.le(offset_y + cell_height);

        // Create a transparent black color (all zeros)
        let transparent = At {
            inner: PlatformColorCube::default(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };

        Select { cond: in_bounds, if_true: blended, if_false: transparent }
    }

    /// Creates a new terminal app.
    pub fn new(params: TerminalAppParams<P>) -> Self {
        // Memory-map the font file
        // Try workspace-relative path first, then crate-relative (for tests)
        let source = MmapSource::open(FONT_PATH).or_else(|_| {
            MmapSource::open(&format!("../{}", FONT_PATH))
        }).expect("Failed to open font file");
        
        let loaded_font = Arc::new(LoadedFont::new(source).expect("Failed to parse font"));

        // Create glyph cache and pre-warm with ASCII
        let cell_height = params.config.appearance.cell_height_px as f32;
        let mut glyph_cache = GlyphCache::with_capacity(128);
        glyph_cache.warm_ascii(&loaded_font.font(), cell_height);

        let mut app = Self {
            emulator: params.emulator,
            pty_tx: params.pty_tx,
            pty_rx: params.pty_rx,
            config: params.config,
            engine_tx: params.engine_tx,
            loaded_font,
            glyph_cache,
        };

        // Send initial frame to kick off window creation and VSync
        app.send_frame();

        app
    }


    /// Build a render manifold from the current terminal state.
    fn build_manifold(&mut self) -> (Arc<dyn Manifold<Output = Discrete> + Send + Sync>, (f32, f32)) {
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
                        let fg = if cc.attr.fg == Color::Default { default_fg } else { cc.attr.fg };
                        let bg = if cc.attr.bg == Color::Default { default_bg } else { cc.attr.bg };
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
                if let Some(cached) = self.glyph_cache.get(&self.loaded_font.font(), ch, cell_height) {
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
                            [bg_r, bg_g, bg_b, bg_a]
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
        (
            Arc::new(top_bsp),
            (grid_width, grid_height),
        )
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
        if let Err(e) = self.engine_tx.send(Message::Data(EngineData::FromApp(data))) {
            log::warn!("Failed to send frame to engine: {}", e);
        }
        eprintln!("[TERM] send_frame() complete");
    }
}

impl<P: Pixel> Actor<EngineEventData, EngineEventControl, EngineEventManagement>
    for TerminalApp<P>
{
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
            EngineEventControl::Resize(width_px, height_px) => {
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
                // Handle close request - could signal quit to engine
            }
            EngineEventControl::ScaleChanged(_scale) => {
                // Handle scale change
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
                            self.engine_tx.send(Message::Management(AppManagement::Quit))
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
                            unimplemented!("EmulatorAction::SetCursorVisibility not yet implemented");
                        }
                        EmulatorAction::CopyToClipboard(_) => {
                            unimplemented!("EmulatorAction::CopyToClipboard not yet implemented");
                        }
                        EmulatorAction::RequestClipboardContent => {
                            unimplemented!("EmulatorAction::RequestClipboardContent not yet implemented");
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
            EngineEventManagement::MouseClick { .. } => {
                unimplemented!("MouseClick not yet implemented");
            }
            EngineEventManagement::MouseRelease { .. } => {
                unimplemented!("MouseRelease not yet implemented");
            }
            EngineEventManagement::MouseMove { .. } => {
                unimplemented!("MouseMove not yet implemented");
            }
            EngineEventManagement::MouseScroll { .. } => {
                unimplemented!("MouseScroll not yet implemented");
            }
            EngineEventManagement::FocusGained => {
                unimplemented!("FocusGained not yet implemented");
            }
            EngineEventManagement::FocusLost => {
                unimplemented!("FocusLost not yet implemented");
            }
            EngineEventManagement::Paste(_) => {
                unimplemented!("Paste not yet implemented");
            }
        }
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        // Drain any queued ANSI commands from PTY
        use crate::term::EmulatorInput;

        let mut found_data = false;
        while let Ok(commands) = self.pty_rx.try_recv() {
            found_data = true;
            for cmd in commands {
                self.emulator.interpret_input(EmulatorInput::Ansi(cmd));
            }
        }

        // CRITICAL FIX: If we processed PTY output, the grid state changed.
        // We MUST call send_frame() to rebuild the manifold and trigger a render.
        // Without this, grid updates but display never refreshes!
        if found_data {
            self.send_frame();
        }

        // If we found data, keep polling. Otherwise block on actor messages.
        if found_data {
            ParkHint::Poll
        } else {
            ParkHint::Wait
        }
    }
}

/// Creates terminal app and spawns it in a thread.
pub fn spawn_terminal_app<P: Pixel + 'static>(
    params: TerminalAppParams<P>,
) -> std::io::Result<(
    actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
)> {
    let (app_tx, mut app_rx) =
        ActorScheduler::<EngineEventData, EngineEventControl, EngineEventManagement>::new(10, 128);

    let mut app = TerminalApp::new(params);

    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_tx, handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::AnsiCommand;
    use crate::io::PtyCommand;
    use crate::term::{EmulatorInput, TerminalEmulator, UserInputAction};
    use actor_scheduler::{Actor, ParkHint};
    use pixelflow_runtime::input::{KeySymbol, Modifiers};
    use pixelflow_runtime::{EngineEventControl, EngineEventManagement};
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
        TerminalApp<DummyPixel>,
        Receiver<PtyCommand>,
        SyncSender<Vec<AnsiCommand>>,
        pixelflow_runtime::EngineActorHandle<DummyPixel>,
    ) {
        let emulator = TerminalEmulator::new(80, 24);
        let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(128);
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel(128);

        // Create a dummy engine handle
        let (engine_tx, _) = actor_scheduler::ActorScheduler::new(10, 10);

        let config = Config::default();
        let params = TerminalAppParams {
            emulator,
            pty_tx,
            pty_rx: cmd_rx,
            config,
            engine_tx: engine_tx.clone(),
        };
        let app = TerminalApp::new(params);

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
        let resize_event = EngineEventControl::Resize(1000, 800);
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
            PtyCommand::Resize { cols: 100, rows: 50 },
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
