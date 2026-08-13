# Paper Draft Prep — SGT-QAT Drafter for Speculative Decoding in vLLM

Self-contained bundle for preparing/sharing the second research paper. Everything
here is drawn from the `sgt-qat-draft` project; numbers trace to raw data in
`raw-results/` and the formal record in `findings.md`.

**Project in one line**: benchmark a Sensitivity-Guided Targeted QAT (SGT-QAT)
quantized Qwen3-1.7B checkpoint as a speculative-decoding drafter for Qwen3-8B,
against vLLM's built-in EAGLE-3 drafter and a no-speculation baseline
(acceptance rate, wall-clock speedup, memory footprint).

**Headline result**: SGT-QAT reaches near-parity draft *quality* with EAGLE-3
(mean acceptance length 2.443 vs. 2.474 on real prompts) but is a wall-clock
*regression* (0.43× — slower than no speculation), because a full dense 1.7B
model is the wrong architectural shape for the drafter role regardless of
quantization. Pushing quantization harder to win on memory collapses quality
(~12× worse perplexity). The honest framing is quality-preservation +
architectural-cost, not an EAGLE-parity claim.

---

## What's in here

| File / folder | What it is |
|---|---|
| `paper-draft.md` | The working paper draft — abstract, method, results, discussion, limitations, future work. Every number cited. |
| `results-tables.md` | **Start here for the numbers.** Clean, consolidated, paper-ready tables (Tables 1–5) with sources and caveats. |
| `findings.md` | Formal chronological results record — the source of truth every table draws from, with full methods per run. |
| `methodology-context.md` | Cross-session methodology/state notes (copy of the project's `context.md`) — useful background for a collaborator. |
| `figures/` | Six poster/paper-ready SVG charts + `generate_charts.py` (the source that produces them) + a README explaining each. |
| `raw-results/` | The raw benchmark JSON files backing every number. Canonical (real-prompt) runs only; superseded placeholder-prompt runs are omitted. |

## Suggested reading order

1. `results-tables.md` — the numbers and what they mean, in five tables.
2. `paper-draft.md` — the narrative built around those numbers.
3. `figures/` — the visuals (open the SVGs in a browser).
4. `findings.md` / `raw-results/` — dig into methods and raw data as needed.

## Status (as of 2026-08-10)

- All Phase 3 experiments are complete with real (mt-bench) data; tables and
  figures are final for the numbers they cover.
- The paper draft is a **first draft** — content is accurate and sourced, but
  it hasn't been through a writing/tone pass or venue-specific shaping.
- One open external thread (does not block the paper): the vLLM tooling
  limitation that forced the memory comparison to be non-apples-to-apples is
  filed upstream (issue #49893) and partially fixed; a W3 packing-format issue
  remains open. See "Known limitations" in `results-tables.md`.

## A note on the numbers

Only real-prompt (mt-bench) runs are included here. An earlier round used
placeholder smoke-test prompts and produced materially different (and
misleading) numbers — those are superseded and deliberately excluded from this
bundle to avoid confusion. If you see a number that doesn't match something
elsewhere, check it's not from the old placeholder set (which lives only in the
main repo's `results/`, clearly dated before 2026-07-25).
