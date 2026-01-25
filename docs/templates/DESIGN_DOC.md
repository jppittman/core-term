# Design Doc Template

## Metadata
- **Author**:
- **Status**: Draft | Review | Approved | Implemented
- **Created**: YYYY-MM-DD
- **Reviewers**:

---

## 1. Overview

### 1.1 Problem Statement
_What problem are we solving? Why now?_

### 1.2 Goals
_What does success look like? Be specific._

### 1.3 Non-Goals
_What are we explicitly NOT doing?_

---

## 2. Background

### 2.1 Current State
_How does the system work today?_

### 2.2 Prior Art
_What existing solutions/papers/libraries informed this design?_

---

## 3. Design

### 3.1 Architecture
_High-level structure. Diagrams encouraged._

```
┌─────────┐     ┌─────────┐
│ Input   │────▶│ Output  │
└─────────┘     └─────────┘
```

### 3.2 Interfaces
_Define the contracts. This is what gets copied into issues._

```rust
pub trait Foo {
    fn bar(&self) -> Result<Baz>;
}
```

### 3.3 Data Flow
_How does data move through the system?_

### 3.4 Error Handling
_What can go wrong? How do we handle it?_

---

## 4. Implementation Plan

### 4.1 Task Breakdown

| Task | File(s) | Deps | Estimate | Assignee |
|------|---------|------|----------|----------|
| T1: Description | `path/file.rs` | None | S/M/L | Jules/Claude |
| T2: Description | `path/other.rs` | T1 | S/M/L | Jules/Claude |

### 4.2 Parallelization
_Which tasks can run concurrently?_

```
T1 ──────────────┐
                 ├──▶ T4 (integration)
T2 ──┬──▶ T3 ───┘
     │
     └──▶ (blocked)
```

### 4.3 Risk Assessment
_What could go wrong? Mitigation?_

---

## 5. Testing Strategy

### 5.1 Unit Tests
_Per-task test requirements._

### 5.2 Integration Tests
_End-to-end validation._

---

## 6. Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|-------------|------|------|---------|
| Option A | ... | ... | ... |

---

## 7. Open Questions
_Decisions that need input before proceeding._

- [ ] Question 1?
- [ ] Question 2?
