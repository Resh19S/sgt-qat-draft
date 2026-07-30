# QAT-Drafter Speculative Decoding in vLLM

Using a Sensitivity-Guided Targeted QAT Qwen3-1.7B checkpoint (from the prior
[`quant research`](/mnt/windows/projects/quant%20research) project) as a draft model
for speculative decoding in vLLM, benchmarked against vLLM's built-in EAGLE-3 drafter
and against no speculative decoding at all — targeting Qwen3-8B.

See `CLAUDE.md` for full project context, background, and phase definitions.

## Setup

### 1. Clone vLLM (source orientation)

```bash
git clone https://github.com/vllm-project/vllm vendor/vllm
```

This is for reading vLLM's source (spec-decode plumbing, EAGLE implementation) during
Phase 1 orientation. It is not committed to this repo's git history (`.gitignore`d).
Building/installing vLLM itself (`pip install -e vendor/vllm` or similar, plus CUDA
toolchain) is deferred until Phase 2, when we actually need to run it — most of that
work will happen in Colab Pro, not locally.

### 2. Notebooks

All experiment/benchmark code lives in `notebooks/`, designed to run in Colab Pro,
mirroring the prior project's notebook-driven workflow:

- `00_colab_setup.ipynb` — environment/dependency bootstrap.
- `01_export_sgt_qat_checkpoint.ipynb` — reproduces the Sensitivity-Guided Targeted QAT
  recipe and persists an actual checkpoint (the prior project never saved one).
- `02_baseline_eagle3.ipynb` — vLLM's built-in EAGLE-3 spec-decode baseline.
- `03_sgt_qat_drafter_bench.ipynb` — benchmarks the exported SGT-QAT checkpoint as a
  drafter, same harness as `02`.
- `04_compressed_checkpoint_memory.ipynb` — standalone VRAM footprint of the
  genuinely compressed checkpoint (vLLM can't load it as a drafter directly — see
  `docs/findings.md`).
- `05_aggressive_quant_tradeoff.ipynb` — tests whether pushing quantization down
  a further bit-width tier (W3/W2 mix, vs. the flagship W4/W3) narrows the memory
  gap against EAGLE-3, and what it costs in quality (WikiText-2 PPL proxy).
- `06_vllm_draft_model_compressed_tensors_bug_repro.ipynb` — minimal, public,
  self-contained repro of the vLLM `draft_model`-loading bug filed as
  [vllm-project/vllm#49893](https://github.com/vllm-project/vllm/issues/49893)
  (fix in [PR #49900](https://github.com/vllm-project/vllm/pull/49900)).
- `07_w3_packing_format_isolation_checks.ipynb` — follow-up isolation checks
  for the maintainer's W3-packing-format diagnosis on #49893 (not yet run).

### 3. Results & docs

- `results/` — one timestamped JSON/CSV file per benchmark run.
- `docs/context.md` — cross-session context transfer.
- `docs/findings.md` — formal, literature-style record of methods and results.
- `docs/logs.md` — informal running log.
- `docs/paper-draft.md` — draft of the eventual second paper.

## Status

Phase 3 complete, Phase 4 (packaging) in progress. Notebooks 01-03 have all run
with real numbers: SGT-QAT drafter beats EAGLE-3 on acceptance rate (2.488 vs.
2.023 mean acceptance length) but loses on realized speed (0.44x vs. 1.79x),
because vLLM can't load the drafter in its intended compressed form — see
`docs/findings.md` for the full write-up and `docs/context.md` for current state.
