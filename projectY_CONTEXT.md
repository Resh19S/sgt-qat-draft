# Project Y — Context / Bootstrap Handoff

> **This file is a seed document.** It was written inside Project X's repo
> (`sgt-qat-draft`) but is meant to be **moved into Project Y's own repo/folder**
> when you create it, and read first by any fresh Claude Code session working on
> Project Y. Rename it to `CLAUDE.md` or `docs/context.md` there if you prefer.
> Project Y is a **new, separate repo** — deliberately not built inside Project X
> (different thesis, different paper; keeps Project X's self-contained story clean).

---

## 1. What Project Y is (one line)

Design a **speculative-decoding drafter that matches EAGLE-3 on speed while
keeping (or beating) its acceptance quality** — by building an **EAGLE-style
shallow drafter head** and compressing/training it with the **Sensitivity-Guided
Targeted QAT (SGT-QAT)** method from Project X.

## 2. Why this project exists (the finding that motivates it)

Project X (`sgt-qat-draft`) benchmarked a **quantized dense** Qwen3-1.7B
checkpoint (SGT-QAT) as a drafter for Qwen3-8B, against vLLM's built-in EAGLE-3
and a no-speculation baseline. Result (real mt-bench prompts):

| Condition | tok/s | Speedup | Mean acceptance length |
|---|---|---|---|
| No spec-decode | 76.6 | 1.00× | — |
| EAGLE-3 | 165.4 | **2.16×** | 2.474 |
| SGT-QAT drafter (dense 1.7B) | 32.9 | **0.43×** | 2.443 |

**The key finding**: SGT-QAT *matches EAGLE-3 on accuracy* (2.443 vs. 2.474 mean
acceptance length) but is a **wall-clock regression** (0.43× — slower than not
speculating at all). Reason: a full 28-layer dense 1.7B model is too expensive
per drafting step; its (good) acceptance rate can't pay for that cost.
Quantization is a *memory* lever, not a *speed* lever — fewer bits ≠ fewer
layers. **The only thing that moves speed is a shallower architecture.**

That's the gap Project Y attacks.

## 3. The core hypothesis / technical framing

Speculative-decoding speedup ≈ (tokens saved by acceptance) − (drafter per-step
cost). EAGLE-3 wins by being **structurally cheap**: a shallow (1–2 layer) head
that **reuses the target model's embeddings/hidden states**, trained via
distillation against the target.

**Project Y's bet**: build an EAGLE-style shallow head (cheap per step → speed
competitive with EAGLE-3) and apply **sensitivity-guided targeted QAT** to it —
testing whether our QAT method buys an **acceptance-quality edge** over a vanilla
EAGLE-3 head at comparable speed.

**Be honest about difficulty (do not overclaim — this ethos carried from Project
X):**
- *Matching* EAGLE-3 speed with a shallow QAT head is **plausible**.
- *Genuinely beating* EAGLE-3 on **both** speed and accuracy is **ambitious** —
  EAGLE-3 is already heavily optimized.
- Realistic, publishable goal: **match speed, and measure whether targeted-QAT
  on the head gives an accuracy edge.** Frame it as a hypothesis to test, not a
  promised dual-axis win. If QAT gives no edge, that itself is a citable result.

## 4. Open design questions to resolve EARLY (before writing training code)

1. **Build on EAGLE-3's architecture, or from scratch?** vLLM already has a
   working EAGLE-3 implementation for Qwen3 (see §6 for the file). Cheapest path
   may be to adapt/retrain an EAGLE-3-shaped head rather than invent a new one.
2. **Embedding/hidden-state sharing mechanism** — how the head consumes the
   target's hidden states (this is central to EAGLE's speed). Understand this
   from the vLLM source before designing.
3. **How to train the head** — distillation setup against Qwen3-8B: what target
   signal (logits? hidden states?), what data, how many steps. This is the bulk
   of the work and is *new* relative to Project X (X never trained a head, only
   quantized an existing dense model).
4. **Where SGT-QAT enters** — apply sensitivity-guided targeted QAT to the head
   after (or during) distillation. Decide W-bits, which head layers to protect,
   whether the head is even big enough for mixed-precision to matter.
5. **num_speculative_tokens / draft depth** — held at 3 in Project X; revisit.
6. **Eval** — reuse Project X's harness (§5) unchanged so numbers are directly
   comparable to the table in §2.

## 5. Project X assets to REUSE (don't rebuild)

Project X repo path (this machine): **`/mnt/windows/projects/Project X`**
(GitHub: `Resh19S/sgt-qat-draft`). Reuse, don't re-derive:

- **Benchmark harness**: `notebooks/common/bench_utils.py` — `BenchResult`,
  `run_benchmark()`, `load_benchmark_prompts()` (mt-bench via vLLM's own dataset
  utils), `extract_spec_decode_metrics()`, GPU-memory measurement, `summarize()`.
  Copy this over so Project Y's numbers are directly comparable.
- **Baseline numbers to match/beat** (real mt-bench, Qwen3-8B target, 80 prompts,
  256 max tokens, `num_speculative_tokens=3`, `max_model_len=4096`, A100-40GB):
  - No-spec: **76.6 tok/s** (1.00×)
  - EAGLE-3: **165.4 tok/s** (2.16×), mean acceptance length **2.474**,
    per-position acceptance **[0.714, 0.465, 0.296]**, in-vLLM memory delta
    **1.047 GiB**. Drafter model: `Tengyunw/qwen3_8b_eagle3`.
  - SGT-QAT dense drafter (the thing we're improving on): 32.9 tok/s (0.43×),
    mean AL 2.443, per-position [0.668, 0.456, 0.319].
  - Raw JSONs: Project X `results/*2026-07-25*.json`, and the consolidated
    `paper-draft-prep/results-tables.md`.
- **SGT-QAT method + checkpoint**: the quantization know-how (sensitivity
  ranking → mixed-precision GPTQ via `llmcompressor.oneshot()` with
  `config_groups` → targeted QAT with a custom `FakeQuantize` STE module). See
  Project X `notebooks/01_export_sgt_qat_checkpoint.ipynb` and
  `notebooks/05_aggressive_quant_tradeoff.ipynb`. Flagship Qwen3-1.7B checkpoint
  (W4/W3, seed 42, WikiText-2 PPL 15.91) lives on Google Drive:
  `MyDrive/sgt-qat-draft-checkpoints/qwen3-1.7b-sgt-qat` (compressed) and
  `…-plain` (decompressed).
- **Prior QAT paper** (method origins): `/mnt/windows/projects/quant research`
  (`notebooks/15_mixed_precision_guided_targeted_qat.ipynb`, `paper.md`,
  `findings.md`) — read before re-deriving SGT-QAT mechanics.

## 6. vLLM plumbing to understand (from Project X's Phase 1 orientation)

`vendor/vllm/` in Project X holds a read-only clone for source orientation
(gitignored; not built). Key files (paths drift across vLLM versions — verify
against the actual checkout, don't trust from memory):

- **EAGLE-3 for Qwen3**: `vendor/vllm/vllm/model_executor/models/qwen3_eagle3.py`
  — the concrete shallow-head implementation Project Y should study first.
- Speculative-decode config wiring: `SpeculativeConfig`, `method="eagle3"` vs.
  `method="draft_model"`.
- **Known tooling limitation (still open, relevant if Project Y ships a
  quantized head)**: vLLM's `draft_model` path cannot load compressed-tensors
  *packed* checkpoints; a W3 packing-format mismatch remains unresolved upstream.
  Filed as `vllm-project/vllm#49893` (fix PR #49900 addresses the target-matching
  half). Full trail: Project X `docs/vllm-bug-report-draft.md`,
  `notebooks/06_*`, `notebooks/07_*`. Implication: EAGLE-style heads load via the
  `eagle3` path (not `draft_model`), so this limitation may not bite Project Y —
  but confirm early which load path a QAT'd head would use.

## 7. Conventions to inherit from Project X

- **All experiment code = Jupyter notebooks meant for Colab Pro** (clone repo →
  mount Drive → save results). No standalone local `.py` CLI pipelines.
- **Checkpoints live on Google Drive**, not git (too large); handed between
  Colab sessions via Drive copy, with `du -sh` size verification (Drive copies
  have been flaky mid-large-file). When two toolchains have conflicting deps
  (e.g. `llmcompressor` pins `torch<=2.12`, a vLLM source build pins
  `torch==2.13`), split into **two Colab sessions handed off via Drive** rather
  than fighting one environment.
- **`docs/context.md` / `findings.md` / `logs.md`** convention: context =
  cross-session state; findings = formal results (every real number, with
  methods); logs = informal running log. Update after every real result.
- **Ethos: verify, don't guess. Real numbers only** — never fabricate or
  reconstruct benchmark data; always paste actual output before writing a
  findings entry.
- **Git**: user pushes manually (no push creds on the local machine — commit
  locally only unless explicitly asked). **Do NOT add a `Co-Authored-By: Claude`
  tag to commits** (standing user preference, carried from Project X).

## 8. Scope / non-goals

- Don't re-run or re-derive Project X's results — reference them.
- Don't build Project Y inside Project X's repo — new repo, reuse via copy/reference.
- Keep the honest framing from Project X: report whatever the data shows
  (including "QAT gave no acceptance edge" if that's the outcome) rather than
  forcing a win narrative.

## 9. Suggested first steps for the Project Y session

1. Read `qwen3_eagle3.py` in Project X's `vendor/vllm` to understand EAGLE-3's
   architecture and how it consumes the target's hidden states (the source of
   its speed).
2. Read the prior QAT paper + Project X notebooks 01/05 to reload the SGT-QAT
   recipe.
3. Decide the design questions in §4 (especially: adapt EAGLE-3 vs. from
   scratch; the distillation-training setup — the genuinely new part).
4. Copy `bench_utils.py` over and confirm you can reproduce EAGLE-3's baseline
   numbers (§5) before building anything, so the harness is trusted.
5. Only then start on the head architecture + training.
