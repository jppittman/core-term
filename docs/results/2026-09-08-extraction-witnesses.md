# Extraction witnesses: the extractor walks past terms it provably holds

**Date:** 2026-09-08
**Denotation:** `docs/plans/2026-09-08-extraction-witnesses.md` (written before
the instrument; nothing in it was revised after the run).
**Instrument:** `pixelflow-pipeline`'s `extraction_witnesses` bin, over
`pixelflow-search`'s `egraph::witness` (feature-gated on
`provenance-journal`).
**Data:** `2026-09-08-extraction-witnesses.{csv,json}` (witness rows and the
divergence explanations) and `2026-09-08-extraction-witnesses-budgets.csv`
(the budget ladder, including the tie-break A/B).

> **What is deterministic here.** `dag_cost`, `objective`, `live_classes`,
> `tied_classes`, the choice maps and every classification are functions of
> the term and the graph, so they are exact and reproducible. Only the
> `seconds` column is wall clock, and it was taken on a shared box under
> heavy load (see §6) — it is reported for scale, never as a claim.

*(Sections below are filled from the run. Placeholders that survive to
review are a bug, not a caveat.)*

## 1. What a witness is, in one paragraph

The e-graph is monotone: the run at a larger class cap performs every
application the smaller run performed and then more, so the larger graph
represents a superset of the terms. When the extractor's own output at the
smaller cap is *cheaper* than its output at the larger one, that cheaper
term is provably present in the bigger graph — and the extractor walked past
it. The instrument takes such a term, looks each of its subterms up in the
bigger graph by hash-cons (read-only, and a miss is a loud failure rather
than a skip, because monotonicity is exactly what is being tested), and gets
a second choice map `C_T` over the same classes. Greedy's map is `C_G`. The
divergence set is where they differ; the frontier is the divergent classes
below which they agree, and it is at a frontier class that the extractor's
own comparison can be read off directly.
