# Context (cross-session transfer)

Read this first in any new session. Update it at the end of every session that makes
real progress, so a cold session can pick up without re-deriving anything.

## Current phase

Phase 1 — orientation in vLLM's spec-decode / drafter code. Initial orientation pass
done (see below); deeper reading (propose() internals, qwen3_eagle3.py, config wiring
end-to-end) still to do interactively with the user.

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

## Next step (Phase 2, in progress)

1. `notebooks/01_export_sgt_qat_checkpoint.ipynb` — written (adapts notebook 15's
   Stage1 GPTQ + Stage2 targeted-QAT recipe, seed=42, PROTECT_FRAC=0.15, ends in a
   compressed `save_pretrained(..., save_compressed=True)`). Not yet run — needs a GPU
   Colab session to execute and validate.
2. `notebooks/common/bench_utils.py` — written: shared harness used by both baseline
   and SGT-QAT benchmark notebooks (build `LLM`, run generation with timing +
   `torch.cuda.max_memory_allocated()`, pull acceptance metrics via `llm.get_metrics()`,
   save a structured JSON to `results/`).
3. `notebooks/02_baseline_eagle3.ipynb` — written: runs three conditions (no-spec,
   EAGLE-3 via `Tengyunw/qwen3_8b_eagle3`) via `bench_utils`, against Qwen3-8B. Not yet
   run.
4. Still to do: `notebooks/03_sgt_qat_drafter_bench.ipynb` (same harness,
   `method="draft_model"` pointed at the exported checkpoint) — write once notebook 01
   has actually been run and produced a real checkpoint to point at, so paths/configs
   can be confirmed rather than assumed.
5. Everything above needs an actual GPU (Colab Pro) run to validate — nothing has been
   executed yet, only authored against the vLLM API confirmed in Phase 1.

## Open questions / decisions pending

- Whether the exported SGT-QAT checkpoint needs a real compressed/bit-packed save
  (vs. plain fp32/bf16 `save_pretrained()`) for the memory-footprint comparison to be
  meaningful. Not resolved — see CLAUDE.md.
- Exact vLLM version/commit to pin for reproducibility — not yet decided.
