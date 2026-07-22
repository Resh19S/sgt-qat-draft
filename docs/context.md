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

## Next step

Read `propose()` in `llm_base_proposer.py` and `qwen3_eagle3.py` together with the user
to fully understand the EAGLE-3 reference implementation's shape (for the baseline,
notebook 02) before writing any notebook code. Then confirm the exact
`speculative_config` dict shape needed to invoke `DraftModelProposer` end-to-end (check
`vllm/engine`/`LLM(...)` entrypoint docs or examples under
`vendor/vllm/examples/features/speculative_decoding/`).

## Open questions / decisions pending

- Whether the exported SGT-QAT checkpoint needs a real compressed/bit-packed save
  (vs. plain fp32/bf16 `save_pretrained()`) for the memory-footprint comparison to be
  meaningful. Not resolved — see CLAUDE.md.
- Exact vLLM version/commit to pin for reproducibility — not yet decided.
