# myterm Style Guide

This document outlines the coding style conventions for the `myterm` project. Consistency helps maintain readability and ease of collaboration.

## Comments

The primary goal is code clarity. Comments should supplement clear code structure, not replace it.

1.  **Clarity Over Comments:** Prioritize clear variable/function names, type definitions, constants, and enums over explanatory comments. If the code is hard to understand without a comment, consider refactoring the code first.
2.  **Rustdoc for Public APIs:** Use Rustdoc (`///`) comments to document all public functions, structs, enums, traits, and modules. Explain *what* the item does, its parameters, return values, and any potential panics or important usage notes.
3.  **Avoid Obvious Comments:** Do not add comments that merely restate what the code clearly shows (e.g., `// increment i` for `i += 1;`). They add noise.
4.  **Non-Obvious Code is a Smell:** If a comment is required to explain a complex or non-obvious piece of logic, treat it as a signal that the underlying code might be too complex and could benefit from simplification or refactoring.
5.  **No Version Control Comments:** Do not use comments for version control (i.e., leaving large blocks of old code commented out). Use Git for history. Commented-out code is only acceptable for *brief*, *clear*, *working* examples of alternative usage immediately adjacent to the code it relates to.

## Code Structure

1.  **Avoid Deep Nesting:** Prefer guard clauses and early returns to deeply nested `if`/`else` or `match` blocks. Aim for a flatter control flow where possible. (Think Go's `if err != nil { return err }` style).

## Magic Numbers

1.  **Define Constants:** Avoid using literal numbers other than 0, 1, or 2 directly in the code if their meaning isn't immediately obvious from context. Define `const`ants with clear names instead (e.g., `const MAX_CSI_PARAMS: usize = 16;`).

## Testing

1.  **Test Public API:** Unit tests (`#[test]`) should primarily focus on testing the public/exported API of a module or crate. Testing internal implementation details can make tests brittle and harder to refactor.

## Flexibility

1.  **Break Rules Sensibly:** These are guidelines, not immutable laws. If strictly adhering to a rule would result in code that is significantly more complex, less readable, or otherwise "ridiculous," use your judgment and break the rule, perhaps leaving a brief comment explaining the rationale if necessary.

