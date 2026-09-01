# Control-arm comparison: linear Guide vs per-rule rate lookup

DEV split: `pixelflow-pipeline/data/strict_labels_dev.jsonl` (155231 samples).

| guide | DEV AUC-ROC | DEV PR-AUC |
|---|---:|---:|
| linear model (candidate-local features) | 0.9902 | 0.4595 |
| PerRuleRateGuide (rule_idx only, control) | 0.9368 | 0.1083 |

**Gap**: AUC +0.0535, PR-AUC +0.3511.

REAL GAP (AUC gap +0.0535 >= 0.02): the linear model meaningfully outranks a per-rule lookup table — candidate-local features are carrying signal beyond each rule's base rate. PR-AUC gap is +0.3511.
