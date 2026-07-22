# Findings

Formal, literature-style record of every metric and result produced in this project.
Each entry should note: date, method/configuration, exact numbers, and any caveats
needed to interpret them correctly. Write for a reader who wasn't in the room —
this is the raw material for `paper-draft.md`.

No results yet. First entries will land once Phase 2 (baseline reproduction) and
Phase 3 (SGT-QAT drafter experiment) produce numbers.

## Template for future entries

### [Date] — [Experiment name]

**Method**: model(s), configuration, hardware, vLLM version/commit, spec-decode
parameters (num speculative tokens, etc.).

**Metrics**:
- Acceptance rate:
- Wall-clock speedup (vs. no spec-decode):
- Memory footprint:

**Comparison baseline(s)**:

**Caveats / threats to validity**:

**Raw data**: `results/<filename>`
