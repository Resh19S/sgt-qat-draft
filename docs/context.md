# Context (cross-session transfer)

Read this first in any new session. Update it at the end of every session that makes
real progress, so a cold session can pick up without re-deriving anything.

## Current phase

Phase 2/3 boundary. Notebook 01 (checkpoint export) and notebook 02 (EAGLE-3
baseline) have both run successfully with real, trustworthy numbers. Notebook 03
(SGT-QAT drafter) is written but **blocked on an unresolved vLLM loading failure**
(see below) — not just waiting on compute anymore, there's a real technical
question to answer first.

## BLOCKED: notebook 03 draft_model loading (2026-07-24)

First real attempt at loading `checkpoints/qwen3-1.7b-sgt-qat/` via
`method="draft_model"` crashed with `RuntimeError: Engine core initialization
failed. See root cause above. Failed core proc(s): {}` — the actual underlying
exception from vLLM's spawned worker subprocess never appeared in the Colab cell
output (known rough edge with vLLM's multiprocess V1 engine in notebooks).

**Ruled out via source reading in `vendor/vllm` (free, no compute cost)**:
- NOT `DraftModelProposer` stripping quantization awareness. Initial hypothesis
  was that `_create_draft_vllm_config()` setting `quant_config=None` disables
  quantization for the draft model. Traced further: `vllm/config/utils.py`'s
  `replace()` fully reconstructs the dataclass (`cls(**dict)`), which re-triggers
  `VllmConfig.__post_init__`, which re-derives `quant_config` fresh from
  `model_config` (the draft's own config/checkpoint) whenever it's `None`. So the
  draft model's own `quantization_config` (confirmed present in
  `checkpoints/qwen3-1.7b-sgt-qat/config.json` — `quant_method: compressed-tensors`,
  `format: pack-quantized`, mixed W3/W4 groups) *should* still be correctly
  detected. This is not the bug.
- NOT (as far as traced) a hard "3-bit unsupported" restriction —
  `WNA16_SUPPORTED_TYPES_MAP` in
  `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
  includes `3: scalar_types.uint3b4`. Didn't fully trace kernel-dispatch logic
  (`get_scheme`/`_get_scheme_from_parts`) far enough to rule out a narrower
  restriction there, so this isn't fully closed out, just not a clean hit.

**Next diagnostic step, not yet tried**: set `os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']
= '0'` (found in `vllm/envs.py`) *before* the `LLM(...)` call, to force the engine
in-process so a crash surfaces its real traceback directly instead of being
swallowed at the subprocess boundary. User ran out of Colab compute (0.43 units)
before the drafter cell even started executing, so this is untested. **Resume here
first** once compute is available — get the real traceback before trying anything
else, rather than continuing to guess from source alone.

If it turns out to be a genuine architectural incompatibility (vLLM's
`draft_model` path can't load this particular compressed-tensors format/bit-width
combo), the fallback is re-exporting a plain (uncompressed) fp16 version of the
checkpoint from the `model_mixed` object in notebook 01 — sacrifices the
memory-footprint story for the drafter itself (would then be a full ~3.4GB fp16
model, not 1.18GB compressed) but should sidestep whatever this loading issue is.
Not yet needed; get the real error first.

## Notebook 02 — RUN, real numbers (2026-07-23)

EAGLE-3 (`Tengyunw/qwen3_8b_eagle3`) vs. no-spec-decode, Qwen3-8B, A100-40GB, 80
placeholder prompts, low-concurrency (sequential, not batched) methodology.
**Speedup: 1.79x** (76.29 → 136.76 tok/s). Mean acceptance length 2.023, GPU memory
delta +1.30GiB. Full numbers and caveats in `docs/findings.md`.

Two harness bugs found and fixed along the way (both in `notebooks/common/bench_utils.py`):
1. `speculative_config` dict gets mutated in place by `LLM(...)` (gains a
   non-JSON-serializable `torch.dtype` field) — fixed by snapshotting it before
   passing to `LLM()`.
2. GPU memory metric read `torch.cuda.max_memory_allocated()` in the notebook kernel
   process, but vLLM's V1 engine runs model execution in separate worker
   subprocess(es) — that always read ~0. Fixed by querying `nvidia-smi` for
   whole-device memory instead.

Also: an initial batched run (all 80 prompts as one `llm.generate(prompts)` call)
produced a misleading 0.98x "EAGLE-3 is slower" result — not a bug, but the wrong
methodology (spec-decode's benefit is a low-concurrency effect, masked by heavy
batching). Switched `run_benchmark()` to sequential single-request generation;
notebook 03 uses the same approach for a fair three-way comparison later.

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
- Result JSON: `results/export_sgt_qat_checkpoint_seed42_2026-07-22T20-49-26.json`,
  pushed to git via a `GITHUB_TOKEN` Colab secret (see "Checkpoint storage" below).
- **Checkpoint backup confirmed (2026-07-23)**: copied to
  `MyDrive/sgt-qat-draft-checkpoints/qwen3-1.7b-sgt-qat/` and verified byte-identical
  via `du -sh` on both sides (1.2G / 1.2G). First copy attempt silently truncated
  (Drive was 95% full, ~0.64GB free — not enough for the 1.2GB file); freed space and
  retried successfully. The checkpoint is durable now — no longer only on the
  ephemeral Colab VM disk.
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

## Checkpoint storage — decided: Google Drive (2026-07-23)

`checkpoints/qwen3-1.7b-sgt-qat/model.safetensors` is 1.2GB as a single file —
over GitHub's 100MB-per-file hard limit, and over GitHub's free Git LFS quota (1GB
total) too, so neither plain git nor LFS work here. HF Hub (private repo) was
considered — it would've let vLLM load the drafter by repo id, same pattern as the
EAGLE-3 baseline — but the user chose **Google Drive only** instead.

**Convention**: checkpoint lives at
`My Drive/sgt-qat-draft-checkpoints/qwen3-1.7b-sgt-qat/` in the project owner's Drive.
Each Colab session that needs it (export notebook re-runs, or notebook 03's
benchmark) should:
1. Mount Drive (`from google.colab import drive; drive.mount('/content/drive')`).
2. Copy it to local disk before loading into vLLM — don't point `speculative_config`
   directly at a Drive-mounted path; Drive I/O can be flaky mid-large-file-read, and
   the copy only costs a few seconds since model loading happens once at `LLM(...)`
   init, not per-token:
   ```python
   !cp -r /content/drive/MyDrive/sgt-qat-draft-checkpoints/qwen3-1.7b-sgt-qat checkpoints/
   ```
3. Then `speculative_config={"method": "draft_model", "model": "checkpoints/qwen3-1.7b-sgt-qat", ...}`
   as normal.

git push of the checkpoint was attempted and correctly blocked twice: once by
`.gitignore` (intentional), once by missing GitHub auth in the Colab session (fixed
separately via a `GITHUB_TOKEN` Colab secret for pushing the *results* JSON, which
did successfully go through git — only the checkpoint binary itself goes via Drive).

## Next step

1. **User is buying more Colab compute** (ran out mid-session on 2026-07-23/24,
   twice now). Once available: set `VLLM_ENABLE_V1_MULTIPROCESSING=0` before
   re-attempting the notebook 03 drafter cell, to get the real crash traceback —
   see "BLOCKED" section above. This is the actual next action, not just "run
   notebook 03" — there's an unresolved technical question first.
2. Once the real error is known: either fix it directly, or fall back to a plain
   uncompressed fp16 re-export of the checkpoint if it's a genuine format
   incompatibility (see fallback plan above).
3. Before treating any notebook 03 run as comparable to notebook 02: check
   `NUM_PROMPTS` matches what notebook 02 actually used (both default to 80 in the
   committed notebooks — confirm the Colab copies agree, since hand-patched cells
   in a live session can drift from what's in git).
4. Real numbers exist for two of three conditions now (no-spec, EAGLE-3). Once
   notebook 03 produces the third (SGT-QAT drafter), Phase 4 (packaging) can start —
   `docs/findings.md` needs the 3-way comparison write-up, still with the caveat that
   all of this used placeholder prompts, not a real benchmark dataset.

## Workflow note (2026-07-23)

Local machine (this Claude session) cannot push to GitHub — no credentials
configured here. User pushes manually from this local machine's `git push` after
each session (previously routed some pushes through Colab directly, which caused a
duplicate-cell editing bug once — see `docs/logs.md`). Going forward: commit locally
here, user handles all pushes, no need to re-flag the credential issue each time.

## Open questions / decisions pending

- Whether the 68.0% recovery result (notebook 01) replicates on a second run/seed,
  or was specific to this A100 run — flagged in findings.md, not yet re-checked.
- Exact vLLM version/commit to pin for reproducibility — deliberately deferred
  (plain `pip install vllm` used for notebook 02), revisit if it matters later.
- Prompts are placeholder smoke-test text throughout Phase 2 so far — a real
  benchmark dataset (e.g. mt-bench, matching vLLM's own example convention) is
  needed before any of these numbers are final/paper-ready.
