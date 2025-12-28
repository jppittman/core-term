## 2025-12-28 - Inherent Methods Shadowing Optimized Trait Implementations
**Learning:** I discovered that `Jet3` and `Jet2` had optimized `sqrt` implementations in their `Numeric` trait implementation, but their *inherent* `sqrt` methods (which shadow the trait methods when called directly) were still using the unoptimized slow path.
**Action:** Always check both trait implementations AND inherent methods when optimizing types in Rust, as inherent methods take precedence and might be legacy/unoptimized code.
