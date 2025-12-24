# Font Rendering Design

**Status**: Implemented / PixelFlow 1.0

## 1. Overview

In PixelFlow 1.0, Fonts are just **Manifolds**.
A Font file is a factory that produces `Manifold<Output = Field>` (coverage values).

## 2. The Model

*   **Glyphs are Manifolds**: A glyph is a function `f(x, y) -> coverage`.
    *   Implemented via winding number algorithm for quadratic Bézier curves.
    *   Produces binary coverage: inside (1.0) or outside (0.0).
*   **Fonts are Factories**: A `Font` object (loaded from TTF) provides methods to get a Manifold for a specific character.
    *   `font.glyph(char) -> Option<Glyph>`
    *   `font.glyph_scaled(char, size) -> Option<Glyph>` (with affine transform applied)

## 3. Glyph Caching

Font caching is implemented via the **categorical morphism** pattern:

```text
           cache_at(size)
    Glyph ──────────────► CachedGlyph
      │                        │
      │ Manifold<I>           │ Manifold<Field>
      ▼                        ▼
    coverage                 coverage
```

### The Morphism

Caching is a morphism between evaluation strategies:
- **`Glyph`**: Evaluates mathematically via winding numbers (infinite resolution)
- **`CachedGlyph`**: Evaluates from texture memory via SIMD gather (fixed resolution)

Both implement `Manifold`, so they compose identically. The cache transforms evaluation strategy while preserving the algebraic interface.

### GlyphCache: The Functor

`GlyphCache` is a functor that memoizes the baking morphism:

```rust
let font = Font::parse(data).unwrap();
let mut cache = GlyphCache::new();

// Cache glyphs at specific sizes (happy path: fast)
let cached = cache.get(&font, 'A', 16.0);

// Pre-warm the cache at startup
cache.warm_ascii(&font, 16.0);
```

### Size Bucketing

To balance cache efficiency with quality, sizes are quantized to multiples of 4 pixels:
- A 17px request uses the 20px bucket
- A 13px request uses the 16px bucket
- Minimum bucket size is 8px

### CachedText

For text rendering with caching:

```rust
let mut cache = GlyphCache::new();
cache.warm_ascii(&font, 16.0);

// Compose text using cached glyphs
let text = CachedText::new(&font, &mut cache, "Hello", 16.0);
execute(&Lift(text), buffer, shape);
```

### Infinite Resolution Fallback

Raw `Glyph` manifolds still work for arbitrary sizes. Use the uncached path when:
- Exact size matching is critical
- Sizes vary frequently
- Memory constraints prevent caching

```rust
// Uncached: mathematical evaluation at any size
let glyph = font.glyph_scaled('A', 17.3).unwrap();
```

## 4. Text Rendering Flow

1.  **Layout**: The application calculates character positions.
2.  **Composition**: Create a manifold representing the text:
    *   `Text`: Uncached, mathematical composition
    *   `CachedText`: Cached, texture-based composition
3.  **Materialization**: The engine samples the composite manifold.
    *   Uncached: Winding number evaluation per sample
    *   Cached: SIMD gather from texture memory

## 5. Performance

Benchmarks comparing binary (Field) vs anti-aliased (Jet2) vs cached:

| Glyph | Field (binary) | Jet2 (AA) | Cached | Notes |
|-------|----------------|-----------|--------|-------|
| 'A' (simple) | 68 ns | 492 ns | 67 ns | AA is 7x slower |
| '@' (complex) | 702 ns | 1.75 µs | 67 ns | **Cache: 10-26x faster** |

Cache operations:

| Operation | Time |
|-----------|------|
| Cache hit | 27 ns |
| Cache miss | 3.4 µs |
| ASCII warm (95 chars) | ~450 µs |

**Key insights:**
- Complex glyphs benefit massively from caching (10x+)
- AA (Jet2) is 2-7x slower than binary - caching is essential
- Cache hits are 100x+ faster than cache misses
- Pre-warming at startup amortizes the baking cost

**When to cache:**
- Terminal/editor with fixed font sizes (always)
- Complex Unicode glyphs (CJK, emoji, symbols)
- Anti-aliased rendering (Jet2 is expensive)

**When not to cache:**
- Memory-constrained environments
- Highly variable font sizes (zooming UI)
- One-off characters that won't repeat

## 6. Integration

*   **Fonts are Manifolds**: No special rendering path needed.
*   **Caching is Composable**: `CachedGlyph` works with all combinators.
*   **Platform-Agnostic**: Font parsing produces standard manifolds.

This design embodies the categorical principle: **caching is a morphism, not a special case**.
