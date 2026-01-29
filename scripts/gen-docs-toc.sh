#!/bin/bash
# Generate a Table of Contents for Rust documentation
# Output goes to .claude/DOCS_TOC.md for AI context

set -e
DOC_ROOT="${1:-target/doc}"
OUTPUT="${2:-.claude/DOCS_TOC.md}"

# Build docs if they don't exist
if [ ! -d "$DOC_ROOT" ]; then
    echo "Docs not found, running cargo doc --no-deps..."
    cargo doc --no-deps --quiet
fi

mkdir -p "$(dirname "$OUTPUT")"

cat > "$OUTPUT" << 'HEADER'
# Project Documentation Index

Auto-generated TOC of `cargo doc` output. Use this to find types, traits, and modules.

---

HEADER

echo "## Crates" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# List crate-level modules
for crate_dir in "$DOC_ROOT"/*/; do
    crate=$(basename "$crate_dir")
    # Skip non-crate directories
    [[ "$crate" == "src" || "$crate" == "static.files" || "$crate" == "trait.impl" || "$crate" == "type.impl" || "$crate" == "search.index" ]] && continue

    echo "### $crate" >> "$OUTPUT"
    echo "" >> "$OUTPUT"

    # Modules
    modules=$(find "$crate_dir" -name "index.html" -type f 2>/dev/null | \
        sed "s|$DOC_ROOT/$crate/||" | \
        sed 's|/index.html||' | \
        grep -v '^$' | \
        grep -v '^index.html$' | \
        sort)

    if [ -n "$modules" ]; then
        echo "**Modules:**" >> "$OUTPUT"
        for mod in $modules; do
            echo "- \`$mod\`" >> "$OUTPUT"
        done
        echo "" >> "$OUTPUT"
    fi

    # Structs
    structs=$(find "$crate_dir" -name "struct.*.html" 2>/dev/null | \
        sed 's|.*/struct\.||;s|\.html||' | sort -u | tr '\n' ', ' | sed 's/,$//')
    [ -n "$structs" ] && echo "**Structs:** $structs" >> "$OUTPUT" && echo "" >> "$OUTPUT"

    # Traits
    traits=$(find "$crate_dir" -name "trait.*.html" 2>/dev/null | \
        sed 's|.*/trait\.||;s|\.html||' | sort -u | tr '\n' ', ' | sed 's/,$//')
    [ -n "$traits" ] && echo "**Traits:** $traits" >> "$OUTPUT" && echo "" >> "$OUTPUT"

    # Enums
    enums=$(find "$crate_dir" -name "enum.*.html" 2>/dev/null | \
        sed 's|.*/enum\.||;s|\.html||' | sort -u | tr '\n' ', ' | sed 's/,$//')
    [ -n "$enums" ] && echo "**Enums:** $enums" >> "$OUTPUT" && echo "" >> "$OUTPUT"

    # Functions
    fns=$(find "$crate_dir" -name "fn.*.html" 2>/dev/null | \
        sed 's|.*/fn\.||;s|\.html||' | sort -u | tr '\n' ', ' | sed 's/,$//')
    [ -n "$fns" ] && echo "**Functions:** $fns" >> "$OUTPUT" && echo "" >> "$OUTPUT"

    echo "---" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Generated: $OUTPUT"
