use crate::config::Config;
use crate::orchestrator::OrchestratorSender;
use crate::platform::actions::PlatformAction;
use crate::surface::{GridBuffer, TerminalSurface};
use crate::term::snapshot::TerminalSnapshot;
use crate::term::ControlEvent;
use pixelflow_core::pipe::Surface;
use pixelflow_engine::{AppAction, AppState, Application, EngineEvent};
use std::sync::mpsc::Receiver;

pub struct CoreTermApp {
    orchestrator_tx: OrchestratorSender,
    display_action_rx: Receiver<PlatformAction>,
    current_snapshot: Option<Box<TerminalSnapshot>>,
    config: Config,
}

impl CoreTermApp {
    pub fn new(
        orchestrator_tx: OrchestratorSender,
        display_action_rx: Receiver<PlatformAction>,
        config: Config,
    ) -> Self {
        Self {
            orchestrator_tx,
            display_action_rx,
            current_snapshot: None,
            config,
        }
    }
}

impl Application for CoreTermApp {
    fn on_event(&mut self, event: EngineEvent) -> AppAction {
        match event {
            EngineEvent::Wake => {
                while let Ok(action) = self.display_action_rx.try_recv() {
                    match action {
                        PlatformAction::RequestRedraw(snapshot) => {
                            self.current_snapshot = Some(snapshot);
                            return AppAction::Redraw;
                        }
                        PlatformAction::SetTitle(title) => {
                            return AppAction::SetTitle(title);
                        }
                        PlatformAction::CopyToClipboard(text) => {
                            return AppAction::CopyToClipboard(text);
                        }
                        PlatformAction::RequestPaste => {
                            return AppAction::RequestPaste;
                        }
                        PlatformAction::ShutdownComplete => {
                            return AppAction::Quit;
                        }
                        _ => {}
                    }
                }
                AppAction::Continue
            }
            EngineEvent::CloseRequested => AppAction::Quit,
            EngineEvent::Tick => {
                let _ = self.orchestrator_tx.send(ControlEvent::RequestSnapshot);
                AppAction::Continue
            }
            _ => {
                // Forward input events to Orchestrator
                let _ = self.orchestrator_tx.send(event);
                AppAction::Continue
            }
        }
    }

    fn render(&mut self, _state: &AppState) -> Option<Box<dyn Surface<u32> + Send + Sync>> {
        // Take ownership of the snapshot (will return it after building grid)
        let snapshot = self.current_snapshot.take()?;

        // Convert config colors to u32
        let default_fg: u32 = self.config.colors.foreground.into();
        let default_bg: u32 = self.config.colors.background.into();

        let grid = GridBuffer::from_snapshot(&snapshot, default_fg, default_bg);

        // Return snapshot to orchestrator for reuse (completes circulation)
        let _ = self.orchestrator_tx.send(ControlEvent::FrameRendered(snapshot));

        let terminal = TerminalSurface {
            grid,
            cell_width: self.config.appearance.cell_width_px,
            cell_height: self.config.appearance.cell_height_px,
        };

        // Future: terminal.over(cursor).over(selection)
        Some(Box::new(terminal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::orchestrator_channel::{create_orchestrator_channels, OrchestratorEvent};
    use crate::term::snapshot::{SnapshotLine, TerminalSnapshot};
    use crate::glyph::{ContentCell, Glyph, Attributes};
    use pixelflow_engine::AppState;

    fn create_test_snapshot(cols: usize, rows: usize) -> Box<TerminalSnapshot> {
        let empty_glyph = Glyph::Single(ContentCell {
            c: ' ',
            attr: Attributes::default(),
        });

        let lines = (0..rows)
            .map(|_| SnapshotLine {
                is_dirty: true,
                cells: vec![empty_glyph.clone(); cols],
            })
            .collect();

        Box::new(TerminalSnapshot {
            dimensions: (cols, rows),
            lines,
            cursor_state: None,
            selection: crate::term::snapshot::Selection::default(),
            cell_width_px: 10,
            cell_height_px: 16,
        })
    }

    /// Regression test: Verify snapshot is returned after render() completes.
    ///
    /// Bug: The snapshot circulation was broken because render() consumed the snapshot
    /// but never returned it via ControlEvent::FrameRendered. This caused subsequent
    /// frames to fail with "No snapshot available (buffer out)".
    #[test]
    fn render_returns_snapshot_via_frame_rendered() {
        let (orchestrator_tx, ui_rx, _pty_rx) = create_orchestrator_channels(16);
        let (display_tx, display_rx) = std::sync::mpsc::sync_channel(16);

        let mut app = CoreTermApp::new(
            orchestrator_tx,
            display_rx,
            Config::default(),
        );

        // Simulate orchestrator sending a snapshot
        let snapshot = create_test_snapshot(80, 24);
        display_tx.send(PlatformAction::RequestRedraw(snapshot)).unwrap();

        // Process the event to store the snapshot
        app.on_event(EngineEvent::Wake);

        // Call render - this should consume and return the snapshot
        let state = AppState {
            width_px: 800,
            height_px: 384,
            scale_factor: 1.0,
        };
        let surface = app.render(&state);
        assert!(surface.is_some(), "render() should return a surface");

        // Verify snapshot was returned via FrameRendered
        let event = ui_rx.try_recv().expect("FrameRendered should be sent");
        match event {
            OrchestratorEvent::Control(ControlEvent::FrameRendered(returned_snapshot)) => {
                assert_eq!(returned_snapshot.dimensions, (80, 24));
            }
            other => panic!("Expected FrameRendered, got {:?}", other),
        }

        // Verify current_snapshot is now None (was consumed)
        assert!(app.current_snapshot.is_none());
    }

    /// Verify multiple render cycles work (snapshot circulation is not broken)
    #[test]
    fn multiple_render_cycles_work() {
        let (orchestrator_tx, ui_rx, _pty_rx) = create_orchestrator_channels(16);
        let (display_tx, display_rx) = std::sync::mpsc::sync_channel(16);

        let mut app = CoreTermApp::new(
            orchestrator_tx,
            display_rx,
            Config::default(),
        );

        let state = AppState {
            width_px: 800,
            height_px: 384,
            scale_factor: 1.0,
        };

        // Simulate 3 render cycles
        for i in 0..3 {
            // Send snapshot
            let snapshot = create_test_snapshot(80, 24);
            display_tx.send(PlatformAction::RequestRedraw(snapshot)).unwrap();
            app.on_event(EngineEvent::Wake);

            // Render
            let surface = app.render(&state);
            assert!(surface.is_some(), "Cycle {}: render() should return surface", i);

            // Verify FrameRendered was sent
            let event = ui_rx.try_recv();
            assert!(
                matches!(event, Ok(OrchestratorEvent::Control(ControlEvent::FrameRendered(_)))),
                "Cycle {}: FrameRendered should be sent", i
            );
        }
    }
}
