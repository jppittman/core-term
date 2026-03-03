1. **Analyze target document:**
   Read `docs/STYLE.md` to identify style guidelines, particularly the strict rule against boolean arguments.

2. **Develop style enforcer tool:**
   Create a Python script (`~/self_created_tools/style_enforcer.py`) to randomly sample `.rs` files and detect boolean arguments in function signatures.

3. **Iteratively identify and fix violations:**
   - Run the style enforcer to find a violation.
   - For each violation, replace the boolean argument with an enum (or separate functions, per the style guide).
   - Update call sites.
   - Run `cargo check` to ensure fixes don't break the build (ignoring pre-existing errors in `pixelflow-ir`).

4. **Review changes:**
   Check the modifications using `git diff` to ensure they accurately fix the style violation without introducing logical errors.

5. **Pre-commit and submit:**
   - Run `pre_commit_instructions` tool to verify required pre-commit steps.
   - Commit changes and submit the PR.
