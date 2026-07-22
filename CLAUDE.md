# Project: QAT-Drafter Speculative Decoding in vLLM

## Goal

Use the user's own quantized Qwen3-1.7B checkpoint — produced by the prior
"Sensitivity-Guided Targeted QAT" research project — as a **draft model** for
speculative decoding in vLLM, targeting Qwen3-8B (full precision). Benchmark it
against:
1. No speculative decoding (plain autoregressive Qwen3-8B).
2. vLLM's built-in EAGLE-3 drafter.

Metrics: acceptance rate, wall-clock speedup, memory footprint.

This is intended to become the user's **second research paper**, following on from
the prior QAT paper.

## User background

- Deep expertise in quantization/QAT mechanics — has already written a full paper on
  Sensitivity-Guided Targeted QAT (see `/mnt/windows/projects/quant research`). Do not
  over-explain quantization concepts (GPTQ, fake-quant/STE, per-group scale/zero-point,
  mixed-precision protection) — assume fluency there.
- Newer to vLLM's internals (spec-decode worker architecture, drafter interfaces,
  engine config wiring). Explain vLLM-specific plumbing in more depth than quantization
  mechanics when it comes up.

## Prior project reference (read before re-deriving QAT method details)

`/mnt/windows/projects/quant research/`:
- `notebooks/15_mixed_precision_guided_targeted_qat.ipynb` — the exact flagship recipe:
  Stage 1 = single `llmcompressor.oneshot()` GPTQ call with explicit `config_groups`
  (W4 on the ~15% most sensitive/"protected" layers, W3 on the rest); Stage 2 = QAT
  fine-tuning (custom `FakeQuantize` STE module, 500 steps, AdamW8bit) on only the
  still-W3 layers.
- `findings.md`, `paper.md`, `starred_findings_explained.md` — prior paper's numbers
  and writeup.
- **Known gap**: that pipeline never calls `save_pretrained()` — no quantized
  checkpoint was ever persisted to disk. Only JSON perplexity/sensitivity metrics went
  to Google Drive. `notebooks/01_export_sgt_qat_checkpoint.ipynb` in *this* repo is
  where that gap gets closed (adapts notebook 15, adds an actual save step).
- **Open question, not yet resolved**: after Stage 1+2, the model's tensors are
  ordinary fp32/bf16 values shaped by fake-quantization — `save_pretrained()` writes a
  normal-size HF checkpoint, not a bit-packed low-memory one. If "memory footprint" in
  this project's benchmark is meant to reflect real deployment savings, real compressed
  export (e.g. via `llmcompressor` compressed-tensors, or a real re-quantization pass)
  is a separate problem from what notebook 15 does. Resolve this explicitly in Phase 2,
  don't assume it's already handled.
- Calibration seed 42 is the "verified clean" flagship seed — default to it for
  reproducibility with the prior paper's headline numbers.

## Workflow conventions

- All actual experiment/benchmark code lives in `notebooks/` as Jupyter notebooks
  meant to run in **Colab Pro**, mirroring the prior project's `notebooks/00`-`17`
  pattern (see that repo for the clone-into-Colab / Drive-mount / results-save idioms).
  Do not write standalone local `.py` CLI scripts for the experiment pipeline itself.
- `vendor/vllm/` holds a git clone of `vllm-project/vllm` for local, read-only source
  orientation. It is `.gitignore`'d — never commit it. Building/installing it happens
  separately (in Colab, when Phase 2 actually needs to run it), not required for
  Phase 1 orientation.
- `docs/context.md` — cross-session context transfer. Update at the end of any session
  that makes progress, so a fresh session can pick up cold.
- `docs/findings.md` — literature-style record: every metric/result, with methods,
  algorithms, and any changes noted formally. Update whenever a real number is
  produced (a benchmark run, a reproduced baseline, etc.).
- `docs/logs.md` — loose, informal running log of what happened, in the moment.
  Lower bar than findings.md; can be messy.
- `docs/paper-draft.md` — empty until the user starts drafting the second paper.
- `results/` — one JSON (or CSV) file per benchmark run, timestamped filename, same
  convention as the prior project's `results/`.
- No GitHub remote is configured yet. The user will provide one later to pair progress
  — don't set one up unprompted.

## Phase 1-4 definition of done

- **Phase 1 (orientation)**: Have identified, in the actual cloned `vendor/vllm`
  source (not from memory — vLLM's internals shift across versions), the drafter
  interface contract and the exact file(s) implementing EAGLE/EAGLE-3. Understood how
  a drafter model gets wired into the engine via spec-decode config.
- **Phase 2 (baseline reproduction)**: vLLM's built-in EAGLE-3 spec-decode path runs
  end-to-end against Qwen3-8B, producing reproducible acceptance-rate/latency/memory
  numbers logged to `results/`. The SGT-QAT checkpoint has been exported
  (`notebooks/01_export_sgt_qat_checkpoint.ipynb`) to `checkpoints/qwen3-1.7b-sgt-qat/`.
- **Phase 3 (own experiment)**: The exported SGT-QAT checkpoint runs as a drafter
  through the same benchmark harness, producing numbers directly comparable to Phase 2.
- **Phase 4 (packaging)**: `docs/findings.md` has a properly written, apples-to-apples
  comparison (methods + numbers) across no-spec / EAGLE-3 / SGT-QAT-drafter.
  `docs/context.md` is current. `results/` holds the raw run artifacts backing every
  claim in findings.md.

## Out of scope (for now)

**Do not** set up GitHub issues, pull requests, upstream contribution branches, or any
outreach-related tooling/files. That is a distinct later phase the user will explicitly
ask for — nothing in Phase 1-4 should assume or prepare for it.
