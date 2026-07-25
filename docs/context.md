# Context (cross-session transfer)

Read this first in any new session. Update it at the end of every session that makes
real progress, so a cold session can pick up without re-deriving anything.

## Current phase

Phase 4 (packaging) in progress, with two parallel follow-ups launched
2026-07-25: (a) real prompts (mt-bench) wired into notebooks 02/03 to replace
the placeholder text those numbers were built on, and (b) notebook 05, testing
whether more aggressive quantization narrows the memory gap toward EAGLE-3
without breaking the acceptance-rate advantage. Neither has been run yet.

## Notebook 05 — WRITTEN, not run (2026-07-25)

Answers the "could we theoretically compress further and match/beat EAGLE-3 on
memory" question with an actual experiment instead of speculation. Same recipe
as notebook 01, shifted down one bit-width tier: protected layers W4→W3, rest
W3→W2 (avg ~2.15 bits/weight vs. notebook 01's 3.156). Deliberately cheap — no
8B target, no vLLM, just the 1.7B model (same cost class as notebook 01) — uses
WikiText-2 PPL as a quality proxy rather than a full in-vLLM acceptance-rate
re-run. Measures standalone VRAM the same way notebook 04 did, for direct
comparison. Real risk flagged in the notebook itself: 2-bit QAT may just break
(NaN loss, PPL collapse) — that outcome would itself answer the question
("no, can't push this far"), not indicate a bug.

**Do not write the paper as if this experiment already confirms SGT-QAT would
match/beat EAGLE-3 if only vLLM supported compressed drafters** — that's not
what the current data shows (see notebook 04 below; the flagship compressed
checkpoint's standalone memory already exceeds EAGLE-3's full in-vLLM number).
The honest framing is: untested hypothesis, plausible from bits-per-weight
arithmetic, with a real risk of breaking quality before reaching competitive
memory. Notebook 05 exists to actually test it, not to manufacture support for
a predetermined conclusion.

## Notebooks 02/03 — updated for real prompts, not yet re-run (2026-07-25)

Added `bench_utils.load_benchmark_prompts()` (mt-bench via vLLM's own dataset
utilities, same convention as `spec_decode_offline.py`'s `--test` mode) and
wired it into both notebooks' config cells, replacing the placeholder text (3
sentences cycled to 80) all prior findings.md numbers were built on. Raises
loudly on failure rather than silently falling back to placeholder text again.
**User will run these in sequence (02 → 03)** while notebook 05 runs in
parallel for the memory question. All prior 2026-07-23/24 result numbers in
findings.md are placeholder-prompt results and will be superseded once these
re-runs land — don't delete them, but don't cite them as final either.

## Notebook 04 — RUN, real number (2026-07-25)

Standalone VRAM footprint of the genuinely compressed checkpoint (no vLLM, no
decompression): **1.629 GiB**. Notable: this is *already bigger* than EAGLE-3's
full in-vLLM memory delta (1.047 GiB, weights + its own KV cache + serving
overhead) — even excluding KV cache/serving overhead that would only make the
SGT-QAT number larger still. Not apples-to-apples (standalone weights vs. full
serving), so not proof compression can't compete on memory, but a real signal
against assuming compression alone would have made SGT-QAT competitive on
memory even if vLLM supported loading it. Full write-up in `docs/findings.md`
2026-07-25 entry.

`plain_checkpoint_delta_bytes` came back `null` — the plain/decompressed
checkpoint from notebook 03 only ever existed on that session's local Colab
disk, never backed up to Drive, so this different session couldn't find it.
Would need a fresh decompression run or a Drive backup to get a real
same-methodology plain-vs-compressed number instead of extrapolating from disk
size.

## Notebook 04 — RUN, real number (2026-07-25)

Standalone VRAM footprint of the genuinely compressed checkpoint (no vLLM, no
decompression): **1.629 GiB**. Notable: this is *already bigger* than EAGLE-3's
full in-vLLM memory delta (1.047 GiB, weights + its own KV cache + serving
overhead) — even excluding KV cache/serving overhead that would only make the
SGT-QAT number larger still. Not apples-to-apples (standalone weights vs. full
serving), so not proof compression can't compete on memory, but a real signal
against assuming compression alone would have made SGT-QAT competitive on
memory even if vLLM supported loading it. Full write-up in `docs/findings.md`
2026-07-25 entry.

`plain_checkpoint_delta_bytes` came back `null` — the plain/decompressed
checkpoint from notebook 03 only ever existed on that session's local Colab
disk, never backed up to Drive, so this different session couldn't find it.
Would need a fresh decompression run or a Drive backup to get a real
same-methodology plain-vs-compressed number instead of extrapolating from disk
size.

## Notebook 03 — RUN, real numbers (2026-07-24)

SGT-QAT drafter (plain/decompressed, `checkpoints/qwen3-1.7b-sgt-qat-plain/`) vs.
notebook 02's baselines. **Headline finding: mean acceptance length 2.488, vs.
EAGLE-3's 2.023 — higher at every speculative position, roughly double at
positions 1-2.** Our drafter's predictions agree with the target far more than
EAGLE-3's. But throughput is 33.69 tok/s — *worse than no-spec-decode* (76.73
tok/s, 0.44x) — because running a full 1.7B fp16 model as drafter is
computationally heavy enough per step to outweigh the acceptance-rate benefit.
Direct consequence of being forced onto the plain checkpoint (see "RESOLVED"
section below) rather than a compressed/small one. Full numbers in
`docs/findings.md` "Phase 3" entry.

**`max_model_len` mismatch — fixed and verified (2026-07-24)**: notebook 03 used
`max_model_len=4096` (OOM mitigation), notebook 02 originally used the default
40960. Added `MAX_MODEL_LEN=4096` to notebook 02 and re-ran both its conditions.
Result: throughput moved <1% (76.29→76.73, 136.76→136.58 tok/s), acceptance
length unchanged (2.0234 both times) — **confirms `max_model_len` doesn't
meaningfully affect speed/acceptance at these sequence lengths, so the speed and
acceptance-rate comparisons are now on solid footing.** Memory remains not
directly comparable, but now for a *different* reason: `gpu_memory_used_bytes` is
an absolute whole-device reading, and the SGT-QAT run was a different Colab
session/VM instance than the (re-run) baselines — config matching alone can't
fix a cross-session absolute-memory comparison. User explicitly deprioritized
chasing a same-session 3-way re-run for this; `notebooks/04` targets a
session-independent angle instead (see below).

**Also fixed while processing results**: notebook 02's actual result JSON files
(`no_spec_decode_*.json`, `baseline_eagle3_*.json`) were never actually pushed to
GitHub — only referenced by filename in the write-up. Reconstructed the original
(2026-07-23, `max_model_len=40960`) ones from data pasted earlier in conversation,
plus the new 2026-07-24 `max_model_len=4096`-matched re-run of both, plus
`sgt_qat_drafter_*.json` — `results/` now actually has everything findings.md
references.

**Next**: `notebooks/04_compressed_checkpoint_memory.ipynb` — written, not run.
Measures the *genuinely compressed* checkpoint's standalone VRAM footprint (direct
`transformers` load, no vLLM) since notebook 03 couldn't answer that question at
all (forced onto the plain checkpoint), and sidesteps the cross-session issue by
being a self-contained standalone measurement rather than something meant to
diff against notebook 02/03's in-vLLM numbers directly. Not a perfect substitute
for an in-vLLM measurement (no KV cache/serving overhead included), but real
measured data, better than no memory story at all. Also measures the plain
checkpoint the same way, as an isolated compression-only comparison point.

## RESOLVED: notebook 03 draft_model loading (2026-07-24)

Root cause confirmed via the real traceback (obtained by temporarily setting
`VLLM_ENABLE_V1_MULTIPROCESSING=0` + `VLLM_LOGGING_LEVEL=DEBUG` to bypass vLLM's
subprocess-swallowed-crash + a Jupyter-specific `fileno()` incompatibility — both
red herrings on the way, see `docs/logs.md` for that detour):

```
ValueError: There is no module or parameter named 'layers.0.mlp.down_proj.weight_packed'
in Qwen3Model. The available parameters belonging to layers.0.mlp.down_proj
(RowParallelLinear) are: {'layers.0.mlp.down_proj.weight'}
```

**vLLM's `method="draft_model"` path (v0.25.1) cannot load compressed-tensors
packed checkpoints.** It builds the draft model's linear layers as plain,
unquantized `RowParallelLinear` (expecting an ordinary `.weight` tensor), then
fails because our checkpoint stores packed weights under `.weight_packed`. This
contradicts the source-level trace done earlier (`quant_config` *should* be
re-derived correctly for the draft model per `VllmConfig.__post_init__`) — that
trace was apparently incomplete; whatever the config-level resolution does, it
doesn't make it through to how the draft model's `nn.Module` gets built. Not
pursuing the exact internal reason further — the empirical fact (confirmed by a
real traceback, not speculation) is enough to act on.

**Fix applied** (`notebooks/03_sgt_qat_drafter_bench.ipynb`, committed): reload the
compressed checkpoint via `transformers.AutoModelForCausalLM.from_pretrained()`
(auto-decompresses through `compressed-tensors`' loading hooks) and re-save without
`save_compressed=True`, to `checkpoints/qwen3-1.7b-sgt-qat-plain/`. `SGT_QAT_DRAFTER`
now points at this plain checkpoint. No need to redo notebook 01's Stage 1/2 GPU
pipeline — this is a cheap reload-and-resave. Also reverted the two diagnostic env
vars back to vLLM's defaults now that they're no longer needed.

**Consequence, already written up in `docs/findings.md`**: the drafter actually
benchmarked will be a full ~3.4GB fp16 checkpoint, not the 1.18GB compressed one —
the memory-footprint comparison loses its compression story for this benchmark
specifically (tooling limitation, not a property of the SGT-QAT method — the
compressed export from notebook 01 remains valid evidence the export pipeline
itself works).

**Not yet tried**: this fix is committed but the notebook hasn't been re-run since.
**Resume here**: run notebook 03 fresh — should mount Drive, copy+decompress the
checkpoint, and load it into vLLM successfully this time.

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

## Next step (Phase 4: packaging + follow-ups)

1. ~~Fix the `max_model_len` mismatch~~ — **done 2026-07-24**, speed/acceptance
   comparison confirmed airtight (see above).
2. ~~Run `notebooks/04_compressed_checkpoint_memory.ipynb`~~ — **done 2026-07-25**,
   real number (1.629 GiB), written up in findings.md. Memory story now has real
   data but isn't a clean win either way — stays noted as non-apples-to-apples.
3. ~~Swap placeholder prompts~~ — **fully done 2026-07-25**
   (`bench_utils.load_benchmark_prompts`, mt-bench), both notebooks 02 and 03
   run and transcribed. Notebook 02: EAGLE-3 baseline materially better with
   real prompts (2.16x speedup, was 1.78x; mean AL 2.474, was 2.023).
   Notebook 03 (SGT-QAT drafter, real prompts): mean AL 2.443 — close to
   EAGLE-3's, basically a wash on acceptance quality — but speedup is
   **0.43x, an actual regression** vs. no speculation at all. Headline: full
   dense 1.7B drafting cost outweighs its (competent) acceptance rate; this
   is architectural (EAGLE-3's tiny shared-embedding head vs. a 28-layer
   dense model), not something quantization quality fixes. See findings.md
   "Real-prompt SGT-QAT drafter run: full 3-way comparison". Note: notebook
   03's own printed comparison table was initially wrong (pulled a stale
   placeholder-prompt EAGLE-3 result because the real one hadn't been pushed
   to GitHub before that session's reclone) — corrected by hand before
   transcribing; see `docs/logs.md` 2026-07-25 for the full mechanism.
   Mid-run, notebook 03 also hit a Colab session/environment corruption saga
   (wrong cwd → incomplete clone → a deleted-cwd bug → GitHub anonymous-clone
   auth failure → a stale `sys.path_importer_cache` entry) — full chain in
   `docs/logs.md` 2026-07-25, now resolved. **All five notebooks' Setup cells
   (01-05) were proactively updated to clone with `GITHUB_TOKEN` by default**
   as a result.
4. ~~Notebook 05~~ (aggressive quantization tradeoff) — **done 2026-07-25**.
   Clean answer: pushing to ~2.16 bits/weight (W3 protected/W2 rest) gets
   standalone memory under EAGLE-3's number (0.646 GiB vs. 1.047 GiB), but
   perplexity collapses getting there (188.66 combined PPL vs. the flagship's
   15.91, ~12x worse, even after QAT fine-tuning). A memory win over EAGLE-3
   is achievable, but not without breaking the quality the drafter approach
   depends on. Didn't run the expensive 8B/vLLM acceptance-rate check at this
   bit-width — the PPL collapse alone is disqualifying. See findings.md
   "Aggressive quantization tradeoff".
5. **All Phase 3 experiments are now landed with real data.** `docs/findings.md`
   has methods + numbers for every condition (no-spec / EAGLE-3 / SGT-QAT
   drafter, plus the aggressive-quant memory/quality tradeoff). Next real step
   is Phase 4 packaging: decide the paper's honest framing (quality-parity +
   architectural-cost story, not an EAGLE-parity chase — see the user's
   2026-07-25 discussion and the deferred "Project Y" idea in memory) and
   start drafting `docs/paper-draft.md` from what's already written up.

## Workflow note (2026-07-23)

Local machine (this Claude session) cannot push to GitHub — no credentials
configured here. User pushes manually from this local machine's `git push` after
each session. **Confirmed 2026-07-24: `origin/main` is currently several commits
behind this local machine** (stuck at `657051b`, missing all of notebook 03's
debugging fixes and the reconstructed `results/` files) — a push from here is
overdue. Also confirmed: notebook 02's actual result JSON files were never
pushed at all in an earlier session (only referenced by filename) — reconstructed
and committed locally 2026-07-24, still needs to reach GitHub too.

## Open questions / decisions pending

- Whether the 68.0% recovery result (notebook 01) replicates on a second run/seed,
  or was specific to this A100 run — flagged in findings.md, not yet re-checked.
- Exact vLLM version/commit to pin for reproducibility — deliberately deferred
  (plain `pip install vllm` used for notebook 02), revisit if it matters later.
- Prompts are real (mt-bench) for notebooks 02's no-spec/EAGLE-3 baseline as of
  2026-07-25; notebook 03's SGT-QAT-drafter run still needs to complete with
  real prompts before the 3-way comparison is final/paper-ready (see Next step #3).
- Memory comparison remains open (cross-session absolute-reading issue, not
  `max_model_len` anymore) — deliberately deprioritized, see Next step #2.
- Whether SGT-QAT's higher acceptance rate (2.488 vs. EAGLE-3's 2.023) would
  translate into an actual speedup win if the compressed checkpoint could be
  loaded — plausible given the quality signal, but unverified; blocked on vLLM
  gaining compressed-tensors draft-model support (or us building a workaround).
