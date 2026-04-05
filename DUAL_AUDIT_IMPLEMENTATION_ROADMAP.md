## Dual Audit Implementation Roadmap

### Goal

Turn Tatetuck into:

- an institutional-grade biotech PM engine
- a validated event-driven catalyst bot
- a scientifically grounded clinical/regulatory interpreter

### Ownership

- `Meitner`:
  finance/trading realism, backtest integrity, execution-book validation
- `Averroes`:
  scientific interpretation, trial readout parsing, regulatory ontology
- `Codex`:
  shared catalyst architecture, integration, gating, roadmap execution

### Priority 0: False-Alpha Killers

#### Finance / Trading

1. Freeze catalyst event scoring point-in-time at `first_seen_as_of`
   - Remove use of later snapshot context for event selection, corroboration, and scoring.
2. Add clustered-event controls
   - Enforce one active catalyst per ticker or ticker-family window in the catalyst backtest.
3. Stop overstating catalyst exactness metrics
   - Compute exact timing/outcome rates from relevant evaluated sleeves instead of returning misleading zeros.
4. Separate PM alpha from true execution-book alpha
   - Add an execution-book validation layer that runs the real planner and execution simulator.

#### Scientific / Biotech

1. Stop scoring tolerability/activity language as hard positive efficacy success
   - `well tolerated`, `RP2D`, `encouraging activity`, and similar phrases cannot be treated as definitive wins.
2. Replace brittle binary outcome parsing with structured mixed-outcome handling
   - Represent efficacy, safety, regulatory state, and mixed outcomes separately.
3. Build a real regulatory ontology
   - Distinguish submission, acceptance, AdCom, approval, CRL, label expansion, and generic updates.
4. Reduce promotional-language false positives
   - Mixed business updates cannot become catalyst wins without explicit clinical/regulatory outcome language.

### Priority 1: Catalyst Product Validity

#### Finance / Trading

1. Rebuild pre-event validation on event instances
   - Event-level walk-forward, next-bar execution, pessimistic slippage, clustered-event controls.
2. Add family-level catalyst scorecards
   - Validate `phase1`, `phase2`, `phase3`, `clinical_readout`, `pdufa`, `adcom`, `label_expansion`, `regulatory_update` independently.
3. Tighten catalyst deployment gates
   - No user-facing catalyst ranking unless the sleeve and family gates pass.

#### Scientific / Biotech

1. Make outcome matching program-aware
   - Prefer exact program/family matches over loose ticker-level text matches.
2. Expand structured trial evidence
   - Comparator, randomization, blinding, endpoint hierarchy, subgroup flag, cohort design, effect size, and safety burden.
3. Add uncertainty-aware parsing
   - Mixed, ambiguous, and safety-only outcomes should remain unresolved when evidence is weak.

### Priority 2: Institutional Confidence

#### Finance / Trading

1. Strengthen leakage audit coverage
   - Verify point-in-time joins, planner inputs, and catalyst evaluation windows.
2. Unify evaluation and execution costs
   - Cost-adjusted PM validation must use the same cost model as deployment.
3. Add rolling deployment gates
   - PM, catalyst, freshness, regime, and family slices all need trailing pass/fail status.

#### Scientific / Biotech

1. Tighten phase/family normalization
   - Avoid text-fragile phase inference and false family matches.
2. Add parser-confidence discipline
   - Confidence must reflect ambiguity, not just keyword presence.
3. Restrict manual review to exact-source rows
   - Preserve auditability for the last-mile catalyst review queue.

### Cross-Lane Acceptance Criteria

#### PM Engine

- execution-book validation exists and is reported separately
- PM alpha remains positive after realistic planner/execution simulation
- PM alpha and catalyst alpha are never conflated in surfaces or gates

#### Catalyst Bot

- pre-event long sleeve has enough rows, windows, positive spread, positive lower bound, and positive rank IC
- post-event reaction sleeve validates independently
- family depth passes on at least two catalyst families
- unvalidated sleeves stay gated in Discord and PM-facing outputs

#### Scientific Grounding

- safety/tolerability/activity no longer masquerade as efficacy success
- mixed outcomes are represented explicitly
- regulatory states are specific and auditable
- parser confidence tracks true evidence quality

### Execution Order

1. Fix point-in-time catalyst leakage and clustering.
2. Fix clinical/regulatory outcome interpretation.
3. Recompute catalyst validation on the repaired stack.
4. Add execution-book PM validation and align cost models.
5. Tighten user-facing catalyst and PM gates.
6. Expand catalyst family depth and scientific evidence fields.
