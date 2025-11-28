use crate::config::Config;
use crate::renderer::{Renderer, PlatformState, RenderCommand};
use crate::orchestrator::OrchestratorSender;
use crate::platform::actions::PlatformAction;
use crate::term::snapshot::TerminalSnapshot;
use crate::term::unicode::get_char_display_width;
use pixelflow_engine::{Application, AppState, EngineEvent, AppAction};
use pixelflow_render::commands::Op;
use std::sync::mpsc::Receiver;

pub struct CoreTermApp {
    orchestrator_tx: OrchestratorSender,
    display_action_rx: Receiver<PlatformAction>,
    renderer: Renderer,
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
            renderer: Renderer::new(),
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
                        },
                        PlatformAction::SetTitle(title) => {
                            return AppAction::SetTitle(title);
                        },
                        PlatformAction::CopyToClipboard(text) => {
                             return AppAction::CopyToClipboard(text);
                        },
                        PlatformAction::RequestPaste => {
                             return AppAction::RequestPaste;
                        },
                        PlatformAction::ShutdownComplete => {
                            return AppAction::Quit;
                        }
                        _ => {}
                    }
                }
                AppAction::Continue
            }
            EngineEvent::CloseRequested => AppAction::Quit,
            _ => {
                // Forward input events to Orchestrator
                let _ = self.orchestrator_tx.send(event);
                AppAction::Continue
            }
        }
    }

    fn render(&mut self, state: &AppState) -> Vec<Op<Vec<u8>>> {
        let Some(snapshot) = &self.current_snapshot else {
            return Vec::new();
        };

        let cell_width = self.config.appearance.cell_width_px;
        let cell_height = self.config.appearance.cell_height_px;

        let platform_state = PlatformState {
             event_fd: None,
             font_cell_width_px: cell_width,
             font_cell_height_px: cell_height,
             scale_factor: state.scale_factor,
             display_width_px: state.width_px as u16,
             display_height_px: state.height_px as u16,
        };

        let commands = self.renderer.prepare_render_commands(snapshot, &self.config, &platform_state);

        let mut ops = Vec::new();

        for cmd in commands {
            match cmd {
                RenderCommand::ClearAll { bg } => {
                    ops.push(Op::Clear { color: bg });
                },
                RenderCommand::FillRect { x, y, width, height, color, .. } => {
                     ops.push(Op::Rect {
                         x: x * cell_width,
                         y: y * cell_height,
                         w: width * cell_width,
                         h: height * cell_height,
                         color,
                     });
                },
                RenderCommand::DrawTextRun { x, y, text, fg, bg, .. } => {
                     let mut current_x = x;
                     for ch in text.chars() {
                         ops.push(Op::Text {
                             ch,
                             x: current_x, // Grid coords
                             y,
                             fg,
                             bg,
                         });
                         let width = get_char_display_width(ch);
                         current_x += width;
                     }
                },
                _ => {}
            }
        }
        ops
    }
}
