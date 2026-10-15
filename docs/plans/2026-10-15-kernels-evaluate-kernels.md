# Kernels evaluate kernels

**Status:** accepted (JP, 2026-10-14 and 2026-10-15). Every ruling below is
binding and overrides any earlier plan, design, skill or comment where they
disagree. The earlier text is history, not a second opinion; where it is a
plan, it is marked superseded at the top and left otherwise untouched.

## 1. `If` has one shape

JP, 2026-10-15:

> I don't want to handle divergence in our cpu version. Select/If always
> always branches. We might expose the other to the user if you really want,
> but this is just too much code for our purposes.

> If is: if all(m) { run true } if !any(m) { run false } run select(m, true,
> false). That's always been what if is supposed to be. I feel like we don't
> need any other version ... I think it always looks like that.

The guards are read as `any(m)` and `!all(m)`, so that on a batch whose mask
is mixed both arms have run before the select (with `all`/`!any` a mixed
batch would run neither). So an `If(m, a, b)` compiles to exactly:

```
if any(m)  { a's own block }
if !all(m) { b's own block }
select(m, a, b)
```

- An arm's **own block** is the nodes only that arm reads. An arm that owns
  nothing (a constant, a value the condition or the other arm also reads)
  has no block and therefore no guard: `(x² + y²).lt(1).select(green, black)`
  is `mul, fma, compare, select`, no branch. `m.select(reflective_shader,
  background)` guards the whole shader, so a batch outside the sphere never
  evaluates it.
- There is **no second shape**: no uniform-only path that skips the select,
  no blend-only path, no mispredict pricing, no decision about *whether* a
  block with work deserves its guard. Any such machinery is deleted on sight.
- The guards read `m` at runtime, per batch. Nothing in the compiler
  estimates how often `m` is true.

### Cost

JP, 2026-10-15: "You don't know how often m is true. Or the branching factor.
Average them and move on. ... Or do max(false, true)."

`cost(If(m, a, b)) = cost(m) + (cost(a) + cost(b)) / 2 + cost(select)`.
Average, not max: with no information it is the unbiased guess, and max would
push extraction away from `If` toward computing both arms. The cost is for
extraction choosing among equivalent forms; it never changes which batches
run which arm. Demand weighting, arm probability and projected regions are
deleted; `docs/plans/2026-10-12-placement-reads-demand.md` is superseded.

## 2. Kernels evaluate kernels; there are no memory reads

JP, 2026-10-15:

> No reading adjacent pixels. There's no memory reads. There's kernel evals.
> ... A bound buffer is a kernel, but you never get to see the memory mid
> write.

> Actually, we don't have bound buffers. That's a bad name. We have discrete
> kernels. It's a type error. And they can be backed by memory because
> referentially transparent.

The language has kernels, and a kernel evaluates kernels. A **discrete
kernel** is a kernel on a lattice, `Lattice → ℝ`. It is a value: referentially
transparent, so an implementation may tabulate it in memory, and the law
`index(collapse(f)) = f` is what makes the table *be* the function rather
than back it. There is no write in the language: `collapse` yields a whole
value, so no intermediate state is observable. Memoization (baking, the JIT
cache, a glyph cache) is semantically invisible.

### Writing is application

JP, 2026-10-15:

> The scheduler has the ability to turn kernel programs into blocks of
> memory. And those blocks of memory, those lattices, are still kernels ...
> materializing a kernel gives you a discrete manifold. It's closed in that
> way. ... I want all blocks of memory to act as discrete manifolds or
> kernels. ... the ability to write is kind of outside the language. It's in
> the application, the unification of a program and the schedule is what
> gives you the ability to write. And then after you've written, you get
> something back that has the same interface as the program itself ...
> something that can be sampled at discrete points.

```
apply : Program × Schedule → DiscreteManifold
DiscreteManifold ⊂ Kernel            (sampled at lattice points)
index(apply(f, s)) = f  on the lattice
```

Neither the language nor a backend has a write. Applying a program to a
schedule produces a block of memory, and producing it *is* the write;
`collapse` is that application. What comes back has the program's interface,
so materialization is closed: a baked glyph, the cell table and the frame are
the same kind of thing as the programs that made them. Every block of memory
a kernel is handed is a discrete manifold; there is no untyped buffer, and the
ABI's `out` pointer is the tabulation in progress. `Write`/`Seq` nodes are the
CPU scheduler's mechanics for performing an application, inside its own
chain: not a language effect, not a concept.

Consequences:

- **"Bound buffer", `BoundManifold`, `bind(&[(id, buf)])` are not concepts.**
  A discrete kernel is a `Kernel` and composes like one. The ABI still hands
  the compiled code the tables it reads, as it hands it the output; that is
  the backend's business, not the language's.
- **Discrete and continuous are distinct types.** JP: "It's a type error."
  Evaluating a discrete kernel at a continuous coordinate is refused. Lifting
  it to a continuous kernel is explicit and names the sampling
  (`nearest()`, `bilinear()`, ...), so the opinion lives in the type. A
  discrete kernel composed with an integer translation stays discrete.
- **`Gather`, `Broadcast` and a plain load are not language ops.** They are
  three realizations of one thing, "evaluate discrete kernel `D` at
  `(u, v)`", and the CPU scheduler picks one from the variance it already
  computes: uniform `(u, v)` → broadcast; the lattice position plus a
  translation → contiguous vector load; lane-varying → gather. The IR has no
  gather; the backend does.
- **No arm can fault.** Under §1 an arm runs on inactive lanes of a mixed
  batch with garbage inputs. Arithmetic on garbage is NaN and the select
  discards it; a discrete kernel's domain is its lattice, so its evaluation
  outside the lattice is defined by the type, not by whatever memory lies
  past a table.

## 3. Premises for the schedule (rulings so far, design pending)

JP, 2026-10-14, deciding what is part of the schedule by imagining GPU and
TPU backends beside CPU SIMD:

- "we're a pull renderer. Each pixel is going to be computed independently.
  Our gpu backend is one pixel per thread." The lattice is the GPU's grid;
  lane packing, row loops and stripes exist only where we vectorize across
  pixels ourselves (CPU, TPU's vector unit). The lattice's traversal is the
  schedule's, and wrapping it into the kernel as folds in pixelflow-core
  (`collapse` is a fold) bakes the CPU schedule into a target-agnostic crate.
- "Yeah it's a schedule decision, but I'd put a backend dependent register
  allocator in the CPU scheduler." On a GPU the vendor compiler allocates;
  we have no allocator there. So each backend is one module exposing one
  function, `(kernel, lattice) → code`, with its own `Pass` chain inside;
  the CPU module's chain is Nest → Own → Layout → Select → Allocate →
  Assemble and owns register allocation, rematerialization and carries.
- The kernel's own folds (`sum_over`) are per pixel on every target:
  algorithm, not schedule.
- Materialization (which sub-kernels to tabulate) is schedule, per backend.

The full table (algorithm / schedule / backend, per concept, with where each
lives today) and the module layout are the next note.

## 4. What this supersedes

- CLAUDE.md "`If` contains an if" (uniform takes an arm, mixed blends):
  rewritten to §1.
- `docs/plans/2026-10-12-placement-reads-demand.md`: superseded by §1's cost.
- `docs/plans/2026-09-07-demand-is-a-dag-property.md`: its denotation of
  demand stands as history; its cost consequences do not.
- Ontology entries for `If`, arm, demand, placement, root, park, lane,
  gather, bound buffer: rewritten to §1 and §2.
- Every doc, comment or skill that says "bound buffer", "bind", "mixed-lane
  path" or treats `Gather` as a language op.
