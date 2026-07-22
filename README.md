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

### 3. Results & docs

- `results/` — one timestamped JSON/CSV file per benchmark run.
- `docs/context.md` — cross-session context transfer.
- `docs/findings.md` — formal, literature-style record of methods and results.
- `docs/logs.md` — informal running log.
- `docs/paper-draft.md` — draft of the eventual second paper.

## Status

Phase 1 (orientation) in progress. See `docs/context.md` for current state.
