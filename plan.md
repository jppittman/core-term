1. **Target:** `pixelflow-runtime/src/platform/macos/window.rs`
   - `MacWindow::set_visible(&mut self, visible: bool)` uses a boolean argument `visible`.
2. **Analysis:**
   - This explicitly violates the `STYLE.md` guideline against boolean arguments: "Avoid functions that take boolean arguments. They make the call site unclear... Prefer using enums or splitting the function into two separate functions."
   - The method logic is simple: if `visible` is true, it shows the window; if false, it hides it. This is a perfect candidate for splitting into `show()` and `hide()`.
3. **Execution:**
   - Replace `pub fn set_visible(&mut self, visible: bool)` in `MacWindow` with `pub fn show(&mut self)` and `pub fn hide(&mut self)`.
   - Update `pixelflow-runtime/src/platform/macos/platform.rs` which calls this method to use the new methods based on the `visible` boolean it receives from `DisplayControl::SetVisible`.
4. **Pre-commit step:**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
5. **Submit:**
   - Commit and submit.
