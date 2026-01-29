use serde::{Deserialize, Serialize};

/// Defines the visibility state of the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorVisibility {
    /// The cursor is visible.
    Visible,
    /// The cursor is hidden.
    Hidden,
}

impl From<bool> for CursorVisibility {
    fn from(visible: bool) -> Self {
        if visible {
            CursorVisibility::Visible
        } else {
            CursorVisibility::Hidden
        }
    }
}
