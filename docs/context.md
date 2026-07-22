# Context (cross-session transfer)

Read this first in any new session. Update it at the end of every session that makes
real progress, so a cold session can pick up without re-deriving anything.

## Current phase

Phase 2 — baseline reproduction + checkpoint export. Notebook 01 (checkpoint export)
has been run successfully and produced a real, validated checkpoint. Notebook 02
(EAGLE-3 baseline) is written but not yet run.

## State as of 2026-07-22

- Repo scaffolded: structure, CLAUDE.md, docs skeletons, notebook skeletons.
- Confirmed: the prior QAT project (`/mnt/windows/projects/quant research`) never
  persisted an actual quantized checkpoint — only JSON metrics on Drive. Re-exporting
  the checkpoint (via `notebooks/01_export_sgt_qat_checkpoint.ipynb`) is in scope here.
- `vendor/vllm` cloned (shallow, `--depth 1`) into `vendor/vllm`. Not built/installed yet.
- No benchmark, export, or drafter-loading code written yet.

### Phase 1 orientation findings (vLLM checkout as of clone date 2026-07-22)

- **Drafter interface contract**: `vllm/v1/spec_decode/llm_base_proposer.py`,
  class `SpecDecodeBaseProposer`. Key methods any drafter proposer implements/overrides:
  `propose(...)` (the core method — takes target token ids/positions/hidden states,
  returns draft token ids), `_get_model()`, `load_model()`, `_create_draft_vllm_config()`,
  `_maybe_share_embeddings()`, `_maybe_share_lm_head()`.
- **The exact class we want**: `vllm/v1/spec_decode/draft_model.py`, class
  `DraftModelProposer(SpecDecodeBaseProposer)`. This is vLLM's generic "load an arbitrary
  HF-format model as a draft model" path — it's what will load the SGT-QAT checkpoint,
  *not* the EAGLE-specific code path. It handles vocab-size checks (or a
  `VocabMapping` for heterogeneous vocabularies between draft/target tokenizers) and
  builds its own `VllmConfig` for the draft model via `replace(...)`.
- **EAGLE / EAGLE-3 implementations** (for reference, not what we'll use):
  `vllm/model_executor/models/llama_eagle.py`, `llama_eagle3.py`, and — directly
  relevant since our target is Qwen3 — `vllm/model_executor/models/qwen3_eagle3.py`.
  Proposer-side orchestration lives in `vllm/v1/spec_decode/eagle.py` (thin, subclasses
  the same `SpecDecodeBaseProposer`) and `vllm/v1/worker/gpu/spec_decode/eagle/`.
- **Config wiring**: `vllm/config/speculative.py`, class `SpeculativeConfig`.
  `method: SpeculativeMethod` is a `Literal` including `"draft_model"`, `"eagle"`,
  `"eagle3"`, etc. `__post_init__` (~line 666) **auto-detects `method="draft_model"` as
  the default** whenever the model path isn't recognized as an EAGLE/ngram naming
  convention and isn't a custom-class path — meaning passing the SGT-QAT checkpoint's
  local path as `speculative_config={"model": "<path>", ...}` should route through
  `DraftModelProposer` with no extra config needed, as long as `num_speculative_tokens`
  is set and vocab sizes match (or heterogeneous-vocab mapping is acceptable).
- Not yet read in detail: `DraftModelProposer._raise_if_draft_tp_mismatch` implications
  for our single-GPU Colab setup (likely irrelevant, TP=1 both sides), and the full
  `propose()` body (500+ lines) — worth a closer read together before writing notebook 03.

### Phase 1 orientation — completed via `examples/features/speculative_decoding/spec_decode_offline.py`

This example script effectively closes out Phase 1: it's a working, runnable harness
that covers both methods we need, plus native acceptance-rate metrics.

- **`speculative_config` shape confirmed for both paths**:
  - EAGLE-3 baseline: `{"method": "eagle3", "model": <eagle3_head_repo_or_path>,
    "num_speculative_tokens": N, "disable_padded_drafter_batch": ..., "parallel_drafting": ...}`
  - Our SGT-QAT drafter: `{"method": "draft_model", "model": <path to
    checkpoints/qwen3-1.7b-sgt-qat>, "num_speculative_tokens": N, "enforce_eager": ...,
    "max_model_len": ..., "parallel_drafting": ..., "use_heterogeneous_vocab": ...}`
    (heterogeneous vocab shouldn't be needed — our drafter is a Qwen3 model sharing the
    target's tokenizer/vocab.)
  - Both get passed straight into `LLM(model=<target>, speculative_config=..., ...)`.
- **Acceptance-rate metrics are built into vLLM**, no custom instrumentation needed:
  `llm.get_metrics()` exposes `vllm:spec_decode_num_drafts`,
  `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens`, and
  `vllm:spec_decode_num_accepted_tokens_per_pos` (a `Vector`, gives acceptance rate at
  each speculative position). Mean acceptance length = `1 + num_accepted/num_drafts`.
- **Not covered by this example — still need to add ourselves in the harness**:
  wall-clock speedup (needs explicit timing around `llm.generate()`, comparing against
  a `speculative_config=None` run) and memory footprint (needs
  `torch.cuda.max_memory_allocated()`/`nvidia-smi`-style measurement, plus the
  open question in CLAUDE.md about whether the SGT-QAT checkpoint should be a real
  compressed export for a fair memory comparison against EAGLE-3's small head).
- There's also a built-in `--test` mode with known-good expected acceptance lengths for
  Llama-3.1-8B EAGLE/EAGLE-3 (2.296 / 2.811) as a sanity check pattern — useful precedent
  for how to structure our own "does this number look sane" checks once we're on Qwen3.

## Decisions (2026-07-23)

- **Checkpoint export**: real compressed export, not a plain `save_pretrained()`. The
  SGT-QAT checkpoint must be actually re-quantized/packed (llmcompressor
  compressed-tensors save, `save_compressed=True`) so VRAM/disk size reflects genuine
  W3/W4 savings — needed for the memory-footprint comparison to mean anything against
  EAGLE-3's small head. This is more work than a plain save but was chosen deliberately
  over the cheaper option.
- **EAGLE-3 baseline checkpoint**: `Tengyunw/qwen3_8b_eagle3` (HF Hub) — a published
  EAGLE-3 head for Qwen3-8B, compatible with `vllm/model_executor/models/qwen3_eagle3.py`.

## Notebook 01 — RUN, checkpoint exists (2026-07-22)

Ran on Colab Pro, A100-SXM4-40GB, unchanged recipe (`BATCH_TOKENS=1024`, seed 42,
`PROTECT_FRAC=0.15` target → 0.1562 actual, 53/196 layers protected at W4).

- Stage 1 PPL: 22.37. Stage 1+2 (final) PPL: **15.91**.
- Corrected recovery vs. prior project's method: **68.0%** — higher than the prior
  project's own two runs of the identical recipe (63.4%/65.8%). Same recipe/seed,
  different hardware (A100 here) — worth a second run before trusting this as a real
  improvement rather than a hardware/kernel-numerics artifact. See `docs/findings.md`
  for the full write-up and caveats.
- Checkpoint: `checkpoints/qwen3-1.7b-sgt-qat/`, 1184.8 MB, confirmed genuinely
  compressed (vs. ~3.4GB fp16) — the `save_compressed=True` assumption held.
- Result JSON: `results/export_sgt_qat_checkpoint_seed42_2026-07-22T20-49-26.json`
  (being added to the repo now — was initially only on the ephemeral Colab VM disk,
  not Drive; **the checkpoint itself has the same ephemeral-disk risk and needs its
  own backup plan** — likely Drive or a release asset rather than a raw git push, given
  its size vs. GitHub's 100MB-per-file limit; not yet resolved which).
- Fixed along the way: `.gitignore` originally excluded `results/*.json` and all of
  `checkpoints/` — the results exclusion was a mistake (now fixed, results/ is
  tracked); checkpoints/ staying gitignored is intentional (too large / needs a real
  storage decision, see above).
- Also fixed: notebook 01's Colab bootstrap cell needs `!git clone ...` (bare `git
  clone` in a Colab code cell is a Python `SyntaxError` — cells are Python by default,
  shell commands need the `!` prefix). `BATCH_TOKENS` in the committed notebook was
  briefly lowered to 256 as an OOM workaround for smaller GPUs (L4/T4, ~22GB) but
  reverted back to 1024 as the default since that's what actually produced this
  validated checkpoint on the A100 run — the 256 OOM fix is now just a comment for
  smaller-GPU users, not the default.

## Next step (Phase 2, in progress)

1. Get the checkpoint backed up somewhere durable (Drive, or check per-file shard
   sizes and decide git/LFS vs. Drive-only) — do this before anything else touches it.
2. Run `notebooks/02_baseline_eagle3.ipynb` (no-spec + EAGLE-3 via
   `Tengyunw/qwen3_8b_eagle3`, against Qwen3-8B) — not yet run. Needs vLLM
   installed in the Colab session (see the notebook's setup-cell TODO: pin to
   `vendor/vllm`'s commit vs. plain `pip install vllm` — still undecided) and enough
   VRAM for an 8B target (A100 territory, same as notebook 01).
3. Only then write `notebooks/03_sgt_qat_drafter_bench.ipynb` (same harness,
   `method="draft_model"` pointed at `checkpoints/qwen3-1.7b-sgt-qat/`) — now unblocked
   since a real checkpoint exists to point at.

## Open questions / decisions pending

- How to durably store the 1184.8MB checkpoint (git/LFS vs. Drive vs. HF Hub private
  repo) — not yet decided, see above.
- Whether the 68.0% recovery result replicates on a second run/seed, or was specific
  to this A100 run — flagged in findings.md, not yet re-checked.
- Exact vLLM version/commit to pin for reproducibility — not yet decided.
