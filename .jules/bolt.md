# Bolt's Journal

## 2024-05-24 - String Allocation in Selection
**Learning:** The `get_selected_text` method allocates a new `String` for every line in the selection to build the line content before appending it to the main buffer. This is inefficient for large selections.
**Action:** Append characters directly to the result buffer and handle whitespace trimming by tracking indices, avoiding intermediate allocations.
