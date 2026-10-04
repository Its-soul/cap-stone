# Decisions

| Date | Decision | Reason / alternatives |
|---|---|---|
| 2026-10-04 | Standard-library in-memory DAG and frozen versioned artifacts | No database needed. NetworkX is optional; a small explicit DAG avoids an extra dependency. |
| 2026-10-04 | Conservative declared dependency closure | Semantic edge discovery is advisory future work; never use ML to silently remove required edges. Reachability does not prove causal minimality. |
| 2026-10-04 | Rebuild derived artifacts before re-verification | Copying stale claims into a new certificate would not be recovery. Missing recipes must block. |
| 2026-10-04 | Hash-chained audit journal, no blockchain | Simple local integrity foundation; signatures/Merkle proofs remain designs. An independently retained checkpoint is needed to detect truncation/rewriting. |
| 2026-10-04 | Only deterministic tests executed | User explicitly forbids all model and research runs in this phase. |
| 2026-10-04 | Changes cover bindings to all prior versions | A second change before recovery must still find consumers bound to the first version. Checking only the most recent old node misses them. |
| 2026-10-04 | Preserve unaffected certificates without reissuing | Current support versions are checked atomically at mock commit. Epoch fences recovery plans and delayed issuance requests, not unrelated certificate validity. |
| 2026-10-04 | Advisory small NLI encoder; binary head only after fine-tuning | Semantic reasoning may help propose edges; pretrained entailment does not prove required dependency. No ML in graph/state/hash logic. |
| 2026-10-04 | Validation baseline before test; explicit frozen-selection gate | Reduces premature test inspection. Training selects checkpoints on validation only. Heavy functions guard before imports/downloads. |
| 2026-10-04 | Runnable toy arithmetic adapter, external tasks deferred | Gives a concrete prepared workflow without pretending it validates the research hypothesis. B is an explicitly unsafe cached-control policy. |
| 2026-10-04 | Stop after artifact validation | Foundation scope is complete; training/evaluation/research experimentation stays unrun as instructed. |
