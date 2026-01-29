use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EngineConfig {
    pub window: WindowConfig,
    pub performance: PerformanceConfig,
}

/// Defines settings related to performance and rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    pub min_draw_latency_ms: Duration,
    pub max_draw_latency_ms: Duration,
    /// Target frames per second for display refresh.
    /// The vsync thread will attempt to present frames at this rate.
    /// Default: 120 FPS (8.33ms per frame) - supports ProMotion displays
    pub target_fps: u32,
    /// Number of threads for parallel rasterization (default: 4)
    /// Set to 1 for single-threaded rendering
    pub render_threads: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        PerformanceConfig {
            min_draw_latency_ms: Duration::from_millis(2),
            max_draw_latency_ms: Duration::from_millis(33),
            target_fps: 144,
            render_threads: 12,
        }
    }
}

/// Defines basic window settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowConfig {
    pub title: String,
    /// Window width in pixels
    pub width: u32,
    /// Window height in pixels
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig {
            title: "Pixelflow Application".to_string(),
            width: 800,
            height: 600,
        }
    }
}
