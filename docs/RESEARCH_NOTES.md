# Research boundaries

Hypothesis: dependency-aware certificate invalidation may recover valid state
more efficiently than full recomputation while preserving support validity.
**Not established. No research experiments have run.**

- Declared-edge closure is ordinary graph reachability, not novel causal inference.
  Missing edges cause under-invalidation; extra edges cause needless rebuilding.
- Configured work units are explanatory estimates. Savings = 1 − selective/full;
  zero full cost is undefined (`None`), negative savings are retained.
- Certificate cost includes verification and issuance; receipt artifacts have
  zero additional cost to avoid double counting. Source-fetch costs are excluded
  equally from both policies in the prepared toy adapter. Real adapters must
  record and include fetch/capture/verification/failed-attempt overhead.
- A/B/C are conceptual controls, not enough for a publication claim. Add same
  verifier single-agent, certificate/revocation methods and strong restart controls
  when the full study begins. PAFA is one reference, not an architectural baseline.
- Baseline B intentionally ignores external source changes and accepts cached
  certificates. Its failures are an unsafe-control illustration, not evidence
  against a strong competing method.
- Baseline C invalidates but does not recover. Compare joint useful completion,
  support-valid commits and cost; always abstaining cannot win.
- Reliability is empirical support-valid commit rate plus useful completion,
  not multiplying independent agent probabilities. Independence is not assumed.
- Unit tests establish bounded code behavior on fixtures, not accuracy, latency,
  recovery efficiency or publication-grade evidence.

Future study: review annotation/task validity, freeze paired worlds, vary faults,
graph depth, branching, repeated changes and edge visibility; compare budgets,
report failures and clustered confidence intervals. Keep model edge proposals
advisory until false-negative consequences are studied. No causal minimality,
cryptographic authenticity or general AI correctness is claimed by this phase.

