# Logs

Loose, informal running log — whatever happened, in the moment, roughly
chronological. Lower bar than findings.md: half-formed observations, dead ends,
"tried X, didn't work, here's why" all belong here.

---

## 2026-07-22

Repo scaffolded (structure, CLAUDE.md, docs, notebook skeletons). Explored the prior
QAT project at `/mnt/windows/projects/quant research` — confirmed no checkpoint was
ever saved, everything ran ephemeral in Colab. Recipe to reproduce lives in that
repo's `notebooks/15_mixed_precision_guided_targeted_qat.ipynb`.

Cloned `vllm-project/vllm` (shallow) into `vendor/vllm`. Grepped for eagle/spec_decode
files. Found the generic drafter interface (`SpecDecodeBaseProposer` in
`llm_base_proposer.py`) and, importantly, `DraftModelProposer` in `draft_model.py` —
vLLM already has a first-class "arbitrary HF checkpoint as drafter" path, separate from
the EAGLE-specific code. Confirmed via `vllm/config/speculative.py` that
`method="draft_model"` is the *default* fallback when a model path doesn't look like an
EAGLE/ngram name — so plugging in the SGT-QAT checkpoint should mostly be a config
problem, not a new-code problem. Also spotted `qwen3_eagle3.py` — an EAGLE-3 head
already implemented specifically for Qwen3, good reference for the baseline notebook.
Found `examples/features/speculative_decoding/spec_decode_offline.py` in the vllm
checkout — a working example covering exactly both methods we need (`eagle3` and
`draft_model`), with the exact `speculative_config` dict shape for each, plus vLLM's
built-in acceptance-rate metrics (`vllm:spec_decode_num_drafts` etc. via
`llm.get_metrics()`). This effectively finishes Phase 1 orientation — no more
guesswork on the drafter interface or config wiring. Wall-clock and memory metrics
aren't covered by this example, so those still need custom instrumentation in our
harness. Full `propose()` internals still unread but no longer blocking — the example
shows the config-level API is all we need, not proposer internals.

## 2026-07-23

First real run attempt of notebook 01 (Colab, L4, 22GB VRAM). Stage 1 (GPTQ) and the
start of Stage 2 (QAT) ran fine — "Trainable parameters (still-W3 layers only): 1193.3M"
printed correctly — but hit `CUDA out of memory` on the very first `backward()` call.
20.33/22.03 GiB already allocated going into it. Likely two compounding causes: (1)
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only works if set before torch/CUDA
initializes in-process — if cells get re-run out of order after a mid-session edit
(which is what happened here, after adding the git-clone bootstrap cell), it silently
doesn't apply; a full Runtime restart + top-to-bottom re-run is the fix. (2)
`BATCH_TOKENS=1024` in fp32 with ~1.2B trainable params is just heavy for a single
22GB card — lowered the notebook's default to 256 and added an explicit
`gc.collect()`/`empty_cache()` right before Stage 2 starts to clear Stage-1
fragmentation. Not yet confirmed this actually fixes it — next attempt should confirm.

Second run attempt: switched to an A100-SXM4-40GB Colab instance instead, and ran the
notebook **as originally pulled** (before the 256/OOM-fix commit landed locally) — so
still `BATCH_TOKENS=1024`, matching the prior project's original recipe exactly. Worked
end to end. Real numbers: Stage 1 PPL 22.37, combined PPL 15.91, checkpoint 1184.8MB
compressed. See `docs/findings.md` for the full write-up — corrected recovery (68.0%)
came out higher than the prior project's own two runs of the same recipe (63.4%/65.8%),
flagged as unreplicated rather than trusted yet.

Nearly lost the output: the "Saved: results/..." print made it look like the file was
somewhere findable, but it actually only exists on the Colab VM's ephemeral local disk
(`/content/sgt-qat-draft/results/`), not Google Drive — user was looking in the *prior*
project's Drive `results/` folder, a completely different (and irrelevant) location.
Same ephemeral risk applies to the exported checkpoint itself, which is the more
valuable artifact. Backup (git push or Drive copy) still needs to happen before the
Colab runtime disconnects/recycles.

Also caught: `.gitignore` had `results/*.json` excluded from the start — a mistake,
since result JSONs are meant to be tracked (only `checkpoints/` should stay untracked,
for size reasons). Fixed. Reverted the notebook's `BATCH_TOKENS` default back to 1024
(from the 256 OOM workaround) since 1024 is what actually produced this validated
checkpoint on the A100 — 256 is now just a documented fallback for smaller GPUs.

Tried to `git push` the checkpoint from Colab — hit two blockers in sequence: (1)
`checkpoints/` is `.gitignore`'d (intentional, hadn't been resolved yet at that
point), (2) even after that, `git push` failed with `fatal: could not read Username
for 'https://github.com'` — no credentials configured for a non-interactive HTTPS
push. Fixed the auth problem with a `GITHUB_TOKEN` Colab secret (same pattern the
prior project's notebook 15 used for its own repo clone), wired into the remote URL
for that session only. That got the *results JSON* pushed fine. For the checkpoint
itself: `model.safetensors` turned out to be 1.2GB as a single file — over GitHub's
100MB-per-file limit and over GitHub's free LFS quota (1GB total), so git was never
going to work for it regardless of auth. Considered HF Hub (private repo, would let
vLLM load it by repo id like the EAGLE-3 baseline) vs. Google Drive; user chose
**Drive only**. Convention documented in context.md: Drive path
`MyDrive/sgt-qat-draft-checkpoints/qwen3-1.7b-sgt-qat/`, copy-to-local-disk before
vLLM load (don't read straight off the Drive mount).

First Drive copy attempt silently truncated — only `config.json` and
`generation_config.json` (tiny files) made it into the Drive folder, `model.safetensors`
(1.2GB) never landed. Cause: this Google account's Drive was 95% full (14.36/15GB),
only ~0.64GB free, not enough for the file. Freed space and retried; confirmed
byte-identical via `du -sh` on both the local and Drive copies (1.2G / 1.2G match).
Checkpoint backup is now actually durable, not just attempted.

Wrote notebook 03 (SGT-QAT drafter benchmark) fully, ahead of the user actually
running it, while they worked through notebook 02 — no reason to block on that
since the checkpoint already existed and the harness was stable. Mirrors notebook
02's structure, adds a Drive-mount-then-copy step with a size sanity check (given
the truncation incident above).

Ran notebook 02's EAGLE-3 baseline. First attempt: `ImportError: libcudart.so.13`
after `pip install vllm` (installed a CUDA-13-linked vLLM binary, but pip resolved
torch to a CUDA 12.8 build — versions disagreed). Fix: the cu13 runtime lib was
actually already present on disk (`.../nvidia/cu13/lib/libcudart.so.13`), just not
on the linker's search path — symlinked it into `/usr/lib/x86_64-linux-gnu/` and
ran `ldconfig`, then restarted the runtime. Worked.

Next: `TypeError: Object of type dtype is not JSON serializable` in
`bench_utils.save_result` — root cause: `LLM(...)` mutates the `speculative_config`
dict passed to it in place (resolves it into a full `SpeculativeConfig`, adding a
`torch.dtype` field along the way), and we were storing a reference to that same
dict rather than a snapshot. The actual generation had already succeeded (EAGLE-3
loaded and ran fine against Qwen3-8B — this answered the "does
`Tengyunw/qwen3_8b_eagle3` actually work" question cleanly). Fixed by snapshotting
`speculative_config` before passing it to `LLM()`, plus `default=str` on the
`json.dumps` call as a general safety net. Gave the user an in-session monkeypatch
(`save_result_fixed`) so they didn't have to redo the expensive generation step just
to save results that already existed in memory.

Then noticed the actual numbers looked wrong even once saving worked: `peak MB: 0`
for both conditions, and EAGLE-3 at 0.98x (slightly *slower* than no-spec) despite a
plausible acceptance length (2.07). Two real bugs, not noise:
1. `torch.cuda.max_memory_allocated()` only sees the calling process's CUDA
   allocations. vLLM's V1 engine runs model execution in separate worker
   subprocess(es), so the notebook kernel's own CUDA context stays near-empty
   regardless of actual GPU usage. Fixed: query `nvidia-smi` for whole-device memory
   instead (`_gpu_memory_used_mb`).
2. The 80 prompts were being submitted as one big `llm.generate(prompts)` batch.
   Speculative decoding's throughput benefit is a low-concurrency effect (helps when
   the GPU is memory-bandwidth-bound waiting on per-token weight loads; the benefit
   shrinks/reverses once the GPU is already compute-saturated by heavy batching) —
   so the batched setup was measuring the wrong thing entirely, not revealing a real
   EAGLE-3 weakness. Fixed: `run_benchmark()` now generates prompts sequentially,
   one at a time.

Git sync friction: local machine (this Claude session) has no GitHub push
credentials, so fixes made here couldn't reach Colab via `git pull` either (nothing
new to pull). Worked around by hand-patching `bench_utils.py` directly in Colab via
`%%writefile`. Also hit `git -C sgt-qat-draft pull` failing with "No such file or
directory" — red herring, the earlier `%cd sgt-qat-draft` cell already puts you
inside that directory, so `-C sgt-qat-draft` was looking for a nonexistent nested
copy; plain `git pull` was correct. User decided going forward: they'll handle all
GitHub pushes themselves from the local machine, no more routing through Colab
(that's also what caused the notebook 02 duplicate-cell bug earlier this session —
an editing mistake, not a Colab-vs-local sync issue, but avoiding Colab-side git
ops reduces surface area for that class of problem regardless).

After the low-concurrency + memory-metric fixes and a kernel restart (`importlib.reload`
wasn't attempted; went straight to a full restart), re-ran notebook 02 with real
numbers: **1.79x speedup** (76.29 → 136.76 tok/s), mean acceptance length 2.023, GPU
memory delta +1.30GiB. Ran the full 80 prompts despite the low compute-balance
warning (Colab flagged ~1 hour remaining at the time) — took about 4.5 + 2.5 minutes
for the two conditions, fit within budget. See `docs/findings.md` for the full
write-up. User bought more compute and proceeded to notebook 03.

Notebook 03's first real attempt (`method="draft_model"` loading
`checkpoints/qwen3-1.7b-sgt-qat/`) hit `ImportError: libcudart.so.13` again (same
CUDA mismatch as notebook 02, but on a *fresh* Colab session/VM — the earlier
symlink fix was a runtime-level change, never saved anywhere, so it didn't carry
over). Baked the fix permanently into both notebooks' setup cells this time
(auto-detects and symlinks the cu13 lib before anything imports torch/vllm) so this
shouldn't recur.

After that fix, hit a real, unresolved failure: `RuntimeError: Engine core
initialization failed. See root cause above. Failed core proc(s): {}` — vLLM's
spawned worker subprocess crashed, but the actual underlying traceback never made
it into the visible Colab cell output (a known rough edge with vLLM's multiprocess
V1 engine in notebooks). Investigated via source reading in `vendor/vllm` (free,
no compute cost) rather than blind retries:

- Confirmed via `checkpoints/qwen3-1.7b-sgt-qat/config.json` that the checkpoint
  really does carry a `quantization_config` block (`quant_method:
  compressed-tensors`, `format: pack-quantized`, mixed W3/W4 groups) — the
  compressed export from notebook 01 worked as intended.
- Initial hypothesis: `DraftModelProposer._create_draft_vllm_config()` sets
  `quant_config=None`, which looked like it might strip quantization awareness for
  the draft model entirely. **Traced this further and it's wrong** —
  `vllm/config/utils.py`'s `replace()` fully reconstructs the dataclass (calls
  `cls(**dict)`), which re-triggers `VllmConfig.__post_init__`, which re-derives
  `quant_config` fresh from `model_config` (the draft's own config, i.e. our
  checkpoint) whenever it's `None`. So quantization *should* still be correctly
  detected for the draft model — this is not the bug.
- Checked whether vLLM's compressed-tensors WNA16 scheme supports 3-bit weights
  (our checkpoint mixes num_bits=3 and num_bits=4 groups): `WNA16_SUPPORTED_TYPES_MAP`
  in `compressed_tensors_wNa16.py` does include `3: scalar_types.uint3b4` — so
  3-bit isn't nominally unsupported either, though didn't fully trace the kernel
  dispatch logic (`get_scheme`/`_get_scheme_from_parts`) far enough to rule out a
  narrower kernel-level restriction.

Neither hypothesis panned out cleanly from source reading alone — need the actual
subprocess traceback to make further progress rather than continuing to guess.
Found `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in `vllm/envs.py`) as a way to force the
engine in-process so a crash would surface directly instead of being swallowed by
the subprocess boundary. **Not yet tried** — user ran out of Colab compute before
the drafter cell even started executing (0.43 units wasn't enough headroom).
Paused here; resume with the `VLLM_ENABLE_V1_MULTIPROCESSING=0` env var set before
re-attempting, once compute is available again.

## 2026-07-24

User got more compute, re-ran notebook 03 with `VLLM_ENABLE_V1_MULTIPROCESSING=0`
baked in. Got a real traceback this time, but a *different* one:
`UnsupportedOperation: fileno` inside vLLM's `suppress_stdout()` helper
(`vllm/utils/system_utils.py`), called during distributed-group init
(`init_distributed_environment` → `suppress_stdout()` → `sys.stdout.fileno()`) —
happens even for a single-GPU TP=1 setup, since vLLM still sets up a `gloo`
process group for CPU-side coordination. Root cause: Jupyter/Colab replaces
`sys.stdout` with `ipykernel.iostream.OutStream`, which doesn't implement
`fileno()` (no real OS file descriptor backing it) — this only matters when vLLM's
engine runs *in-process* inside the notebook kernel, which is exactly what
disabling multiprocessing forced. The normal spawned-subprocess mode wouldn't hit
this, since each worker subprocess has its own genuine stdout.

So the multiprocessing-disable diagnostic traded the original hidden crash for a
new, self-inflicted, environment-specific one — informative in its own way (now we
know in-process mode doesn't work cleanly in this notebook environment at all) but
not yet the answer to the original question. Found a clean bypass:
`suppress_stdout()` has a built-in early-return when `VLLM_LOGGING_LEVEL == "DEBUG"`
that skips the `fileno()` call entirely. Set that too, on top of the multiprocessing
disable, to get past this and back to chasing the original mystery — untested as of
this entry, next attempt should show whether it actually reaches the real
underlying issue or surfaces yet another layer.

Re-ran with both env vars set. Got past the fileno() crash — target model
(Qwen3-8B, 15.26GiB, 5 shards) loaded fine, DEBUG logging showed every weight
tensor loading successfully. Then the draft model's weight loading started
(`Filesystem type for checkpoints: OVERLAY. Checkpoint size: 1.15 GiB`) and hit the
real error almost immediately:

```
ValueError: There is no module or parameter named 'layers.0.mlp.down_proj.weight_packed'
in Qwen3Model. The available parameters belonging to layers.0.mlp.down_proj
(RowParallelLinear) are: {'layers.0.mlp.down_proj.weight'}
```

This is the actual answer, finally. vLLM built the draft model's linear layers as
plain unquantized `RowParallelLinear`, then failed to find the ordinary `.weight`
tensor it expected because our checkpoint genuinely stores compressed-tensors
packed weights under `.weight_packed`. So the very first hypothesis from
2026-07-23 (quantization awareness not making it through for the draft model) was
right in outcome, even though the source trace showing `quant_config`
auto-re-derivation said it should work — apparently something downstream of that
config resolution doesn't correctly propagate to per-layer module construction for
draft models specifically. Not worth digging further into vLLM's internals to find
the exact spot; the empirical answer (a real ValueError, not speculation) is
enough to act on.

**Fix, no GPU re-training needed**: reload the compressed checkpoint via
`transformers.AutoModelForCausalLM.from_pretrained()` (compressed-tensors'
`transformers` integration decompresses transparently on load) and re-save without
`save_compressed=True`. Added this as a new cell in notebook 03 (decompress
Drive-sourced checkpoint to `checkpoints/qwen3-1.7b-sgt-qat-plain/` before handing
it to vLLM), added `compressed-tensors` to the pip installs (wasn't there before —
only `vllm` was installed, which may or may not pull it in transitively), and
pointed `SGT_QAT_DRAFTER` at the plain checkpoint. Also reverted the two
diagnostic env vars (`VLLM_ENABLE_V1_MULTIPROCESSING`, `VLLM_LOGGING_LEVEL`) back
to vLLM's defaults now that they're no longer needed for debugging — leaving DEBUG
logging on was also what caused the earlier truncated-paste headaches.

Real consequence for the paper: the drafter benchmarked going forward is a full
~3.4GB fp16 checkpoint, not the 1.18GB compressed one. Framing this honestly in
findings.md as a vLLM tooling limitation (draft-model quantization support), not a
failure of the SGT-QAT export itself — notebook 01's compressed checkpoint still
stands as valid evidence the export pipeline works, it's specifically vLLM's
speculative-decoding drafter loader that can't consume that format yet.

Not yet re-run with this fix — next step.

Ran it: decompression load worked, but `save_pretrained()` on the decompressed
model crashed with `AttributeError: 'NoneType' object has no attribute 'convert'`
inside `transformers`' `revert_weight_conversion` internals — a bug in this
transformers version's weight-name-reversion logic specific to
compressed-tensors-derived models. Fixed with `save_original_format=False` (skips
that step; we don't need original-format weight names for our purposes). Also
caught a related bug in our own code: the failed attempt had already run
`PLAIN_CHECKPOINT.mkdir()` before crashing, so the directory existed but was
empty/incomplete — the `if not PLAIN_CHECKPOINT.exists()` retry guard would have
silently skipped decompression on the next attempt. Changed to check for
`config.json` specifically. Not yet re-run with this second fix.

Ran it. New crash: `ValidationError: Invalid repository ID or local directory
specified: 'checkpoints/qwen3-1.7b-sgt-qat-plain'` -- vLLM's `SpeculativeConfig`
tried to resolve the relative path as a HF Hub repo id (401s) instead of
recognizing it as a local directory. Fixed with `.resolve()` to an absolute path.

Ran it again (user restarted the session to be safe). Got the *exact same*
`weight_packed` ValueError as the very first attempt, and the log showed
"Checkpoint size: 1.15 GiB" -- the compressed checkpoint's size, not the ~3.4GB
plain one. First suspected a stale notebook (thought `!git pull` on the cloned
repo folder might not sync the actually-open Colab notebook document), but user
confirmed the notebook did have the `.resolve()` fix. Real cause, once traced
properly: `AutoModelForCausalLM.from_pretrained()` on a compressed-tensors
checkpoint does NOT dequantize on load -- it builds `CompressedLinear` modules
that keep weights packed in memory, dequantizing only during `forward()` for
inference. So the "plain" checkpoint from the first decompression attempt was
never actually plain; `save_pretrained()` just re-serialized the same packed
weights, hence identical size and identical crash. The `AutoModelForCausalLM`
auto-decompress assumption was wrong from the start.

Fixed properly using `compressed_tensors.ModelCompressor.decompress()`, the
actual designed-for-this API. Also added a cheap pre-flight check (inspect the
saved checkpoint's real safetensors tensor names for `weight_packed`, no GPU
needed) so a similarly-broken "looks done, isn't" checkpoint can't silently pass
the `config.json`-exists check again and waste another expensive vLLM load
attempt finding out the hard way.

Tried `hf_quantizer.dequantize(model)` next (confirmed via introspection to
exist and be bound to the loaded model already) — raised
`NotImplementedError: QuantizationMethod.COMPRESSED_TENSORS has no
implementation of dequantize` from `transformers/quantizers/base.py`. So the
generic transformers-side dequantize path is just not implemented for this
quant method in the installed version — not something we can work around at
that level.

Went back to `ModelCompressor` — realized the second attempt had used wrong
method names (`from_pretrained`/`decompress`, guessed) when the *first*
introspection dump already had the real ones (`from_pretrained_model`,
`decompress_model`) sitting right there. Fixed to use those:
`ModelCompressor.from_pretrained_model(model)` then
`compressor.decompress_model(model)`. Not yet run.

Ran it: `decompress_model()` correctly unpacked `down_proj.weight` with the right
shape (confirmed before the crash) — real progress. But crashed again with a
*different* leftover: `ValueError: no module or parameter named
'layers.0.mlp.down_proj.weight_scale'`. Tried stripping known quantization buffer
names (`weight_scale`, `weight_zero_point`, `weight_shape`, `weight_g_idx`) from
each module's `_buffers` before saving — the pre-flight validity check (already in
place) caught that this didn't work either (`AssertionError`, cheaply, no wasted
vLLM load). Had the user dump the actual leftover keys rather than guessing again:
392 tensors still had `weight_scale`/`weight_shape`. Root cause: these modules
stay a custom quantized module type even after `decompress_model()` runs, with
their own `state_dict()`/serialization logic that keeps re-adding scale/shape
regardless of what's deleted from `_buffers` — not simple registered buffers.

**Final fix**: instead of fighting the wrapper module's serialization behavior,
explicitly replace every such module with a genuine `nn.Linear`, copying over the
already-correctly-dequantized weight and discarding the wrapper type entirely.
Ran it — worked. `Replaced N quantized modules with plain nn.Linear`, verification
passed, checkpoint saved clean.

**Notebook 03 ran end to end.** Real numbers:
`sgt_qat_drafter`: 33.69 tok/s (vs. no-spec 76.29, EAGLE-3 136.76 — **0.44x, worse
than no speculation at all**), but mean acceptance length **2.488** vs. EAGLE-3's
2.023 — higher at every speculative position, roughly double at positions 1-2.
Real, striking finding: our drafter predicts the target far more reliably than
EAGLE-3, but running a full 1.7B fp16 model as drafter is too computationally
heavy per step to turn that into a speedup. Direct consequence of being forced
onto the plain (uncompressed) checkpoint.

Caught a real methodology inconsistency while writing this up: notebook 03's run
used `llm_kwargs={"max_model_len": 4096}` (added earlier as an OOM mitigation)
while notebook 02's runs used vLLM's default (40960) — this is almost certainly
why SGT-QAT's GPU memory (37.26 GiB) read *lower* than even the no-spec baseline
(37.33 GiB) despite loading an extra ~3.4GB model: the much smaller KV cache
reservation more than offset the extra drafter weights. Flagged in findings.md as
invalidating the memory row of the 3-way comparison specifically — not yet fixed
(would need a matched re-run).

Also discovered while reconstructing results for the write-up: notebook 02's
actual `no_spec_decode_*.json`/`baseline_eagle3_*.json` files were never pushed to
GitHub in an earlier session (`git fetch` + `git ls-tree origin/main` confirmed
only the notebook 01 export JSON ever made it) — only their filenames were
referenced in findings.md. Reconstructed both files from the exact data the user
had pasted earlier in conversation and committed them, along with the new
`sgt_qat_drafter_*.json`, so `results/` actually has what the docs reference.

User asked to separately measure the *compressed* checkpoint's standalone memory
footprint (outside vLLM, since it can't load it as a drafter anyway) to at least
partially recover the memory story. Wrote `notebooks/04_compressed_checkpoint_memory.ipynb`
for this — loads the compressed checkpoint directly via `transformers` (no
decompression, no vLLM) and reads GPU memory before/after, plus the same for the
plain checkpoint as an isolated compression-only comparison point. Not yet run.

User wanted the speed comparison made airtight before moving on, deferring memory
(already bucketed as "tooling unsupported") to later. Added `MAX_MODEL_LEN=4096`
to notebook 02's config cell and passed it via `llm_kwargs` to both `run_benchmark()`
calls, matching notebook 03 exactly, then re-ran notebook 02.

Result: throughput moved <1% (no-spec 76.29→76.73 tok/s, EAGLE-3 136.76→136.58
tok/s), mean acceptance length identical (2.0234143449911084 both times, to full
float precision even) — clean confirmation that `max_model_len` doesn't
meaningfully affect speed or acceptance at these sequence lengths (unsurprising:
our prompts + 256 output tokens are way under even 4096). Speed/acceptance
comparison in findings.md is now solid.

Memory, however, turned out to have a *second*, independent problem beyond the
`max_model_len` mismatch: `gpu_memory_used_bytes` is an absolute whole-device
`nvidia-smi` reading, and the SGT-QAT run was measured in a completely different
Colab session (different day, different VM instance) than these baselines —
config matching alone doesn't make absolute memory readings across different
runtime instances comparable. Documented this clearly rather than pretend the
`max_model_len` fix solved memory too. Reconstructed and committed both new
result files (`no_spec_decode_2026-07-24T17-39-17...`,
`baseline_eagle3_2026-07-24T17-44-02...`) from the pasted `BenchResult` reprs,
same as before.

Ran a full doc/commit audit while notebook 04 was running in the background
(user's request, "cross check docs and commits theyre upto progress"). Findings:
git working tree clean, origin only 1 commit behind (caught up after this
session's push), every `results/` file findings.md references actually exists,
all notebooks valid JSON, no leftover duplicate headers. Two real staleness
issues found and fixed: `README.md`'s Status section still said "Phase 1 in
progress" (we're Phase 3 done / Phase 4 in progress), and `CLAUDE.md` still said
"no GitHub remote configured" (one's been set up since near the start —
`origin` → `Resh19S/sgt-qat-draft`).

## 2026-07-25

Notebook 04 ran. Real number: compressed checkpoint's standalone VRAM footprint
= 1.629 GiB. `plain_checkpoint_delta_bytes` came back `null` as expected — the
plain checkpoint only ever lived on notebook 03's local Colab disk, never backed
up to Drive, so this different session's fallback logic correctly skipped it
rather than erroring.

Noticed something worth flagging honestly rather than glossing over: even this
standalone, no-KV-cache, best-case number (1.629 GiB) is already bigger than
EAGLE-3's *entire* in-vLLM memory delta (1.047 GiB, from the matched-session
run) — which itself includes KV cache and serving overhead the SGT-QAT number
doesn't. Not a fair comparison as constructed, but real evidence against
assuming "if only vLLM supported compressed drafters, SGT-QAT would obviously
win on memory too." Wrote this up plainly in findings.md rather than let the
quality-win narrative imply a memory win that isn't demonstrated.

User asked whether we could theoretically push quantization further to close
the memory gap, and whether the paper could be framed as "SGT-QAT would
match/beat EAGLE-3 if only vLLM supported it." Did the bits-per-weight math:
current checkpoint averages 3.156 bits/weight at 1.629 GiB; getting to ~2
bits/weight would arithmetically land around 1.03 GiB, close to EAGLE-3's 1.047
GiB. But pushed back on the framing -- we have zero data below W3/W4, quality
could collapse well before reaching that bit-width, and there's a structural
difference (EAGLE-3 is small by architecture, reusing target embeddings; our
drafter is a fully independent 1.7B model) that quantization alone can't erase.
Recommended: don't write it as an implied conditional win, actually test it.

User agreed and asked for two things in parallel: (1) a new notebook to
actually test the aggressive-quantization hypothesis instead of leaving it as
math, and (2) finalize notebooks 02/03 with real prompts (mt-bench) instead of
the placeholder text everything so far was built on.

Wrote `notebooks/05_aggressive_quant_tradeoff.ipynb`: same recipe as notebook
01, shifted down one bit-width tier (protected W4->W3, rest W3->W2, ~2.15
bits/weight average). Deliberately kept cheap -- no 8B target, no vLLM, just
the 1.7B model (same cost class as notebook 01, chosen specifically so this
"wouldn't hurt compute balance" per the user's framing) -- uses WikiText-2 PPL
as a quality proxy instead of a full in-vLLM acceptance-rate re-run, and
notebook 04's standalone-memory-measurement pattern for the memory side.
Explicit non-finite-loss handling flags that a training collapse at 2-bit would
itself be a valid answer to the question, not a bug to route around. Backs up
to Drive immediately this time (learned from notebook 01's near-miss).

Added `bench_utils.load_benchmark_prompts()`: loads mt-bench via vLLM's own
`vllm.benchmarks.datasets.add_dataset_parser`/`get_samples` utilities, the same
convention `spec_decode_offline.py`'s own `--test` mode uses, reusing vLLM's
tested dataset-loading code rather than us re-implementing mt-bench parsing
ourselves. Raises loudly (not a silent fallback) if the prompt count comes back
wrong, consistent with the project's established "fail loud, don't silently
degrade" pattern. Wired into both notebooks 02 and 03's config cells,
replacing the placeholder text. User will run 02 then 03 in sequence while
notebook 05 runs in parallel.

Notebook 02 ran successfully with real prompts. Results were a real surprise:
EAGLE-3 speedup went from 1.78x (placeholder text) to **2.16x**, mean
acceptance length from 2.023 to **2.474**. Placeholder smoke-test strings were
apparently harder for EAGLE-3 to draft well on than realistic mt-bench prompts
-- the earlier baseline understated EAGLE-3. GPU memory delta between
conditions came back byte-identical to the earlier placeholder-prompt run
(1.047 GiB both times), confirming memory is driven by weights + KV cache
sizing, not prompt content, as expected. Wrote this up in findings.md.

Notebook 03 then hit a real debugging saga -- not a code bug in the usual
sense, a Colab session/environment corruption chain:
1. `ModuleNotFoundError: No module named 'bench_utils'` -- diagnosed via
   `!pwd`/`!ls`/`!git log` that cwd was `/content`, not `/content/sgt-qat-draft`
   (a runtime restart had reset cwd and the Setup cell wasn't re-run first).
2. Recovery attempt found the clone directory existed but was **incomplete**
   (`bench_utils.py present: False`) -- likely an earlier `os.system('git
   clone...')` that failed silently, since `os.system` doesn't raise on
   nonzero exit.
3. Gave a `subprocess.run(..., check=True)` wipe-and-reclone cell to force a
   loud failure instead of a silent one. It raised `CalledProcessError: exit
   status 128` -- real, but the actual git stderr wasn't visible in Colab's
   default output.
4. Requested `capture_output=True, text=True` to surface git's own stderr:
   `fatal: Unable to read current working directory: No such file or
   directory`. Root cause was **my own bug** -- the previous cell had
   `os.chdir()`'d into the repo dir, then this cell called
   `shutil.rmtree()` on that same directory while it was still the process's
   cwd (the classic Linux deleted-cwd issue, which breaks all subsequent
   subprocess calls regardless of target). Fixed by `os.chdir('/content')`
   before the `rmtree`.
5. That produced a *different*, real error: `fatal: could not read Username
   for 'https://github.com': No such device or address` -- anonymous clone
   auth/rate-limit failure, likely from Colab's shared IPs after many clone
   attempts that day.
6. Fixed by cloning with the user's existing `GITHUB_TOKEN` Colab secret
   embedded in the URL (`https://{token}@github.com/...`), the same pattern
   already used elsewhere in this project for pushing. Confirmed working:
   `bench_utils.py present: True`, `cwd: /content/sgt-qat-draft`.

Applied the token-based clone fix proactively to **all five** notebooks'
Setup cells (01-05), not just 03 -- they all had the same plain, token-less
`git clone` pattern and would eventually hit the same auth failure. Validated
all five are still valid JSON after the edits.

Wrote up the real-prompt no-spec/EAGLE-3 results in findings.md (2026-07-25
entry, "Real-prompt baseline re-run"). Notebook 03's actual SGT-QAT-drafter
real-prompt run is still pending -- user needs to restore the checkpoint
backup in their live session and re-run from "Run: SGT-QAT drafter" onward.
Notebook 05 (aggressive quant tradeoff) is still running in parallel, results
not yet in.

Notebook 03 hit one more snag after the reclone: `ModuleNotFoundError:
bench_utils` again, despite `%cd /content/sgt-qat-draft` confirming correct
cwd and `ls notebooks/common/` confirming the file genuinely existed on disk.
Diagnosed as a stale `sys.path_importer_cache` entry from an earlier
in-session failed import attempt (the same kernel had tried and failed to
import from that path before the directory was in its final state) --
`importlib.invalidate_caches()` is the standard fix for exactly this. Didn't
get final confirmation this specific fix worked, but the notebook did go on
to produce a real result shortly after, so it's presumed resolved.

Notebook 03's real-prompt SGT-QAT-drafter run landed:
`sgt_qat_drafter_2026-07-25T09-08-02...json`. First "Compare" cell output the
notebook itself printed was WRONG though -- its EAGLE-3 row silently pulled
the old placeholder-prompt result (136.6 tok/s, 1.78x, mean AL 2.02) instead
of the real one (165.4 tok/s, 2.16x, mean AL 2.474), because the real-prompt
EAGLE-3 JSON had only been committed locally in this Claude session, never
pushed to GitHub -- and this Colab session's `results/` dir came entirely
from a fresh clone (git wipe-and-reclone during the earlier debugging saga
had already erased whatever notebook 02 originally saved locally in that
session). Caught this before transcribing anything -- corrected the
comparison by hand using the actually-real `BenchResult` values instead of
trusting the notebook's own (stale) `summarize()` output. Real headline:
mean acceptance length is close between the two drafters (2.443 vs. 2.474,
basically a wash on quality) but SGT-QAT-as-drafter is an actual wall-clock
*regression* vs. no speculation at all (0.43x) -- a full dense 1.7B model's
per-step drafting cost outweighs whatever its (perfectly competent)
acceptance rate saves. This is architectural, not a quantization-quality
problem: EAGLE-3's speed comes from being a tiny few-layer head sharing the
target's embeddings, which quantizing a dense model can't replicate. Wrote
this up in findings.md as a real, if unflattering, Phase 3 result, and saved
a memory note about a possible future "Project Y" (build an EAGLE-style
architecture and QAT *that*) -- explicitly out of scope for this project,
just flagged for later.

Notebook 05 (aggressive quantization tradeoff) landed too:
`aggressive_quant_tradeoff_seed42_2026-07-25T09-51-46.json`. Clean answer to
the question it was written for -- pushing to ~2.16 bits/weight (W3
protected / W2 rest) DOES get standalone memory under EAGLE-3's number
(0.646 GiB vs. 1.047 GiB), but perplexity collapses getting there: 188.66
combined PPL vs. the flagship's 15.91, roughly 12x worse, even after QAT
fine-tuning (which helped, 454.31 -> 188.66, but nowhere near enough). Confirms
the "we might win we might lose" framing from earlier this session was the
right call rather than assuming a match/beat outcome -- the actual answer is
"a memory win is achievable, but not without breaking the quality the whole
approach depends on." Didn't bother running the expensive 8B-target vLLM
acceptance-rate benchmark at this bit-width -- the PPL result alone is
disqualifying, not worth the compute to confirm what's already clear.

Wrote up both as findings.md entries (real-prompt 3-way SGT-QAT comparison,
and the aggressive-quant tradeoff), created the two missing results/ JSON
files from the pasted data (they existed on the user's Colab/Drive but not
in this local repo checkout).

Wrote `docs/paper-draft.md` (first real draft, not a skeleton) and
`results/visual_metrics/` (6 SVG charts, hand-built since matplotlib isn't
available locally -- no pip on this machine either). Spot-checked several by
rendering to PNG via `convert` (ImageMagick, available locally) before
trusting the layout; caught and fixed two real bugs in the generator: (1)
tick labels used `{:.2g}` which produced ugly scientific notation for
values >=100, (2) multi-line bar/group labels containing literal `\n` were
being word-split instead of respecting the explicit line breaks, garbling
two labels. Both fixed in `generate_charts.py` itself, not by hand-editing
the SVGs, so the fix persists across regeneration.

User then asked to start an actual open-source contribution to vLLM about
the `draft_model`/compressed-tensors loading bug -- explicitly opted into
the "out of scope" category CLAUDE.md flags as a later phase. Deliberately
did NOT fabricate the two things a real bug report needs that only a live
Colab session can produce: `collect_env.py` output and a fresh `pip show
vllm` (the "vLLM 0.25.1" in findings.md was never a verified `pip show`
paste, and vLLM was never version-pinned across sessions, so reusing it
in a public bug report felt like the wrong call). Also decided against
reusing our actual mixed-precision Qwen3-1.7B/Qwen3-8B repro for the bug
report itself -- it depends on a private Drive checkpoint outside
maintainers' reach. Instead wrote a minimal, cheap, self-contained repro
(`notebooks/06_vllm_draft_model_compressed_tensors_bug_repro.ipynb`): tiny
public model, plain unmixed W4A16 GPTQ, no QAT -- isolates that the bug is
about compressed-tensors packed checkpoints in general, not our specific
recipe. Drafted the actual issue text in
`docs/vllm-bug-report-draft.md`, following vLLM's own `[Bug]:` template,
with explicit `<PASTE ... HERE>` placeholders (not fabricated filler) for
everything that needs live data, plus a "before filing" checklist at the
bottom. Nothing has been filed/submitted anywhere -- this is draft text
only, sitting in the repo until the user runs notebook 06 and fills in the
real data.

Ran notebook 06's section 1-3 (2026-07-26). First real surprise: forgot to
actually `!pip install vllm` in the notebook before importing it -- fixed
that (also added the known cu13/libcudart symlink fix from notebooks 02/03
proactively, since this is a fresh environment). Second, bigger surprise
once that was fixed: the minimal repro (plain, single-scheme W4A16, no
mixed precision) **loaded successfully in vLLM** -- no ValueError, full
engine init completed (191s). The "any compressed-tensors checkpoint fails"
framing in the original bug report draft is not what actually happened --
my simplification to "the smallest thing that's still compressed-tensors"
apparently also stripped out the actual trigger. Real remaining hypothesis:
the bug is specific to **mixed-precision** (`config_groups`, different
bit-widths per layer subset) checkpoints, since that's the one structural
difference between this (passing) minimal repro and our original (failing)
SGT-QAT checkpoint. Added section 5 to notebook 06 to test that directly
(same tiny model, but with two config_groups mirroring notebook 01's real
structure) before touching the bug report draft further. Marked
`docs/vllm-bug-report-draft.md` as blocked/do-not-file until section 5
actually confirms or disconfirms this -- didn't want a half-verified claim
sitting in a file that looks filing-ready. This is exactly why the
"minimal repro" step existed in the first place rather than just filing the
original observation as fact.

Section 5b confirmed the mixed-precision hypothesis: identical ValueError,
identical `layers.0.mlp.down_proj.weight_packed` module path, on the tiny
public model with a `config_groups` (W4/W3 split) recipe. Single-scheme
W4A16 loads fine; mixed-precision `config_groups` fails. Rewrote
`docs/vllm-bug-report-draft.md` around the now-confirmed, narrower, more
useful scope (mixed-precision specifically, not compressed-tensors in
general) and filled in the real traceback (including the
`vllm/model_executor/models/utils.py:395` `_load_module` frame, and the
adjacent `support_quantized_model_reload_from_hp_weights` decorator name as
a possible pointer for maintainers -- flagged honestly as "noticed, not
traced" rather than claiming we understand the mechanism). Still missing:
`collect_env`/`pip show vllm` output from section 1 -- only genuinely
blocking item left before this can be filed.

User pasted the real `pip show vllm` + `collect_env` output. Confirms this
reproduces on **vLLM 0.26.0** -- newer than the "0.25.1" originally noted
back on 2026-07-24, so the bug survived at least one version bump; good
thing that stale, never-actually-pip-show-verified number wasn't reused in
the filed report. Trimmed the pasted collect_env output slightly for length
(full pip freeze list, CPU vulnerability listing) -- noted explicitly in the
draft that this was a trim of real data, not a fabrication, and that
nothing omitted is relevant to the bug. `docs/vllm-bug-report-draft.md` is
now fully filled in -- every placeholder replaced with real, pasted data.
Only remaining checklist item is the user's own final read-through before
filing on GitHub; nothing left for this session to do here.

**2026-07-27** — user filed the issue: vllm-project/vllm#49893. Fast
response -- maintainer harjothkhara confirmed within hours, labeled
bug/quantization, and opened a fix PR (#49900) with a root-cause
explanation that lines up almost exactly with what we found ourselves: the
draft model loads under a `draft_model` weight prefix at runtime, which
breaks `config_groups`' exact-name/anchored-regex target matching --
single-scheme worked only because `Linear`-class-name matching happens to
be substring-based. Validates the whole "narrow the repro before filing"
detour from 2026-07-26 -- this is a precise, actionable bug report because
of that, not despite the extra step.

Maintainer asked for 4 confirmations against their fix branch (no GPU on
their end). Added section 6 to notebook 06: installs the fix branch
(`VLLM_USE_PRECOMPILED=1 pip install git+...`), then checks (a) mixed-
precision checkpoint loads, (b) single-scheme checkpoint still loads --
regression check, (c) an actual short generation runs (not just engine
init), and (d) in-serving memory delta for the compressed checkpoint vs.
the decompressed workaround (reuses the same ModelCompressor decompression
approach from notebook 03) -- confirms compression actually survives
loading now, not just that loading stopped crashing. Split into separate
restart-the-runtime sections per the same persistent-CUDA-context caution
as section 5b, since several of these instantiate a second `LLM()` in what
would otherwise be the same process. Not yet run. Updated
`docs/vllm-bug-report-draft.md`'s header to reflect FILED status and link
both the issue and the fix PR, kept the original filed text below for the
record rather than overwriting it.

Tried installing the fix branch per the maintainer's exact command --
hit a real, separate install-path bug, not our repro. First failure was
silent (`pip install -q` swallows subprocess build errors -- my own
mistake giving that flag originally). Without `-q`, still nothing useful
until `-v` forced streaming: revealed `VLLM_USE_PRECOMPILED=1` downloads a
prebuilt wheel for upstream vLLM main and layers this (Python-only) PR's
changes on top, auto-detecting a CUDA variant via `torch.version.cuda` --
but pip's *isolated build environment* pulled in an unpinned, newer
`torch==2.13.0` (cu13-associated) as its own build dependency, so the
detector saw "CUDA 13.0" and requested a `cu130` wheel that doesn't exist
for that commit (404), instead of matching the real system CUDA (12.8).
Checked `vendor/vllm/setup.py`'s `detect_system_cuda_variant()` directly to
confirm the mechanism and find the override (`VLLM_MAIN_CUDA_VERSION`)
rather than guessing at a workaround. Forcing that to 12.8 correctly
resolved the variant to `cu128` -- but the wheel fetch STILL 404s, for both
`cu128` and the unversioned default, at a commit hash that doesn't even
match the "upstream main latest commit" printed right next to it. This
looks like their nightly wheel index doesn't have anything published for
whatever `get_base_commit_in_main_branch()` resolves to right now -- a
real, separate bug in the PR's own install instructions, not something on
our end to keep working around.

Asked the user: report this and wait for the maintainer's reply, or spend
the compute on a full source build to bypass their wheel infra entirely.
User chose to report and wait, consistent with earlier compute-budget
awareness in this project (the original "89 compute unit balance" framing
from notebook 05). Drafted a PR comment with the exact commands/output,
gave it to the user to post (not something to post automatically -- GitHub
comments are visible, external, shared state). Logged status as BLOCKED in
`docs/vllm-bug-report-draft.md` pending their response; section 6 of
notebook 06 hasn't run yet and can't until install actually succeeds.

User decided not to wait -- ran a full source build instead (~1hr, no
`VLLM_USE_PRECOMPILED`), accepting the real compute cost rather than staying
blocked on the maintainer's wheel infra. This then cascaded into a genuine
environment fight, not a repro issue: the build pulled in a newer torch,
which broke torchaudio's import-time CUDA-version check (fixed by removing
torchaudio, unneeded here); then a plain `pip install llmcompressor` (to
rebuild the tiny checkpoints, since Colab's VM had silently recycled during
the long build, losing the ones from earlier) downgraded torch to satisfy
its own `<=2.12.0` pin, breaking torchvision AND corrupting `pyarrow` into a
mismatched-.so-vs-metadata state. Diagnosed the pyarrow break concretely
(`pyarrow.__file__`/`__version__` vs. `pip show`) rather than guessing,
found a clean purge fixed it, then restored torch==2.13.0 -- but recognized
this whack-a-mole would keep recurring as long as `llmcompressor` and the
source-built `vllm` shared one environment, since their torch pins are
flatly incompatible. Split into two Colab sessions (build vs. serve) handed
off via Google Drive, matching this project's existing checkpoint-storage
pattern. Hit and fixed a `cp` semantics bug of my own along the way (first
`cp -r src checkpoints/` flattens instead of nesting when `checkpoints/`
doesn't exist yet) that briefly looked like a Drive-copy failure but wasn't.

With a genuinely working environment, the mixed-precision checkpoint (first
the original flat 50/50 split, then a layer-boundary-aligned rebuild after
ruling out the split itself as the cause) got past the ORIGINAL bug
entirely -- point 1 confirmed, the fix works for the issue as filed. But hit
a new `AssertionError` in `vllm/model_executor/parameter.py:175`
(`load_merged_column_weight`) partway through loading. Used `%debug`
post-mortem to get real shape data (`param_data.shape=[3072,96]` vs.
`loaded_weight.shape=[3072,103]`) rather than guess at the mechanism from
the bare assert. Confirmed this isn't a repeat of the earlier
merged-weight-boundary confound (rebuilt with a clean layer-boundary split
specifically to rule that out, failed identically) -- this is a new, real,
separate finding. Drafted and had the user post a follow-up PR comment with
the exact numbers. Points 2-4 of the maintainer's original ask are now
blocked behind this new issue, not something more debugging on our end
would resolve.

User exported the live Colab notebook (`06_..._repro1.ipynb`, 782KB with all
debug-log outputs baked in) and pasted it into `notebooks/` so the repo's
canonical version could be reconciled with what actually happened.
Reconciled: replaced section 5's flat 50/50 split with the layer-boundary
version; updated section 5/6's markdown with the real 2026-07-27 results;
rewrote the install cell to show both the dead-end (`VLLM_USE_PRECOMPILED`)
and working (source build) paths, documented rather than deleted; added the
"6-prep" Drive-handoff section as first-class documentation (not just
something I said in chat); updated 6a with the new AssertionError finding
and `%debug` commands; updated section 7 to reflect the comment was posted.
Deliberately did NOT carry over cell 21 from the live notebook (an abandoned
attempt to rebuild the single-scheme checkpoint in the vllm session, before
the two-session split was discovered) -- dead code, no reason to preserve it
as runnable, though the failure mode it hit is captured in the new
troubleshooting note instead. The 782KB exported file itself
(`06_..._repro1.ipynb`) is untracked and not committed -- redundant now that
its useful content is folded into the canonical notebook; left in place for
the user to delete or keep as they prefer, not removed unilaterally.

**2026-07-29** — user pushed everything, asked whether 2 days of silence on
the issue/PR counts as a stalemate. Answered no -- normal maintainer pace,
not unusual for a volunteer who already engaged fast once; recommended
waiting ~1 week before a polite bump rather than reading silence as stuck.
User confirmed the plan: hold off on the decoupled point-2 (single-scheme
regression) check until they're actually ready to follow up on the issue,
not run it proactively now just because it's cheap and available. Noted this
explicitly in context.md so a future session doesn't jump ahead and run it
unprompted.

Maintainer actually replied well before the week mark -- precise, better
diagnosis than either of our own hypotheses. He traced the real cause: vLLM
now expects DENSE W3 packing (96 int32 columns, sub-byte-boundary packed --
landed in compressed-tensors 0.17.0), our checkpoint's W3 tensor has 103
columns (older whole-values-per-int32 layout). The two layouts coincide
exactly for 4-bit, which is why only W3 ever tripped anything. This isn't a
draft_model-path bug or a merged-weight bug at all -- both of our earlier
hypotheses were wrong, though the diagnostic legwork (exact shapes via
%debug, ruling out the layer-boundary confound) is clearly what let him
nail it this precisely instead of guessing himself.

Before jumping to code, discussed the "should we trace load_merged_column_
weight ourselves" question the user raised just before this reply landed --
moot now given his answer, glad we asked/planned first rather than spending
compute on a source-level trace that would've been superseded within
minutes.

He asked one question (exact library versions that produced our checkpoint
-- something we never captured, a real gap) and suggested two cheap
isolation checks (plain-model W3 load with no draft_model at all; W4/W4
config_groups drafter, unambiguous packing). Wrote `notebooks/07_w3_packing
_format_isolation_checks.ipynb` to answer all three: builds a uniform W3
checkpoint with compressed-tensors>=0.17.0 explicitly pinned this time
(captures pip show output immediately, unlike notebook 06's gap), prints
the packed tensor's actual shape directly via safetensors (no vLLM needed
to answer the version question), then isolation check 1 in the same cheap
session (stock pip install vllm, no draft_model, no fix-branch conflict --
simpler than anything in notebook 06's later sections), then isolation
check 2 in a separate fix-branch session via the same Drive-handoff pattern
as notebook 06.

User also said: no more Co-Authored-By tag on commits in this project going
forward -- saved as a feedback memory so future sessions don't need
reminding.

Notebook 07 sections 1-2 ran. Isolation check 1 confirmed the maintainer's
diagnosis exactly -- identical AssertionError, identical location, on a
completely plain model load with zero draft_model/speculative_config
involved. Clean, strong confirmation this is unrelated to the PR.

Real surprise: pinning compressed-tensors>=0.17.0 (resolved to 0.17.1) did
NOT change the packed shape -- still (3072, 103), the old layout, not the
dense 96-column one he said landed in 0.17.0. So the version-pin
remediation he suggested doesn't actually work as expected; either
llmcompressor 0.12.0 doesn't invoke the updated packer regardless of which
compressed-tensors is installed alongside it, or there's some other
setting/flag needed that neither of us knows about yet. Flagged this
explicitly in the reply rather than just reporting the isolation-check
success and leaving the version question looking resolved when it isn't.
Isolation check 2 (W4/W4 config_groups drafter, needs the fix-branch
session) still pending.

User asked (a) whether 1a-1c need re-running before 2/3, and (b) to add a
fix attempt for the packing-format surprise, then re-upload the notebook to
debug 2/3 live. Answered (a): no, only a cheap Drive-restore is needed (1a/1b
are the expensive quantization steps, already done and safely on Drive) --
added a "1-restore" cell for this instead of dropping section 1 outright,
since section 1's actual builds still need to exist somewhere. For (b):
added section 1d (retry building the W3 checkpoint with `llmcompressor`
upgraded, not just `compressed-tensors` -- the next lever to try after the
version-pin alone didn't work) and section 2b (re-test with that checkpoint).

Hit and fixed a real bug in my own NotebookEdit usage while doing this: the
plain `cell-N` labels the tool shows for cells without an explicit `id`
field are POSITIONAL, not stable -- they shift every time an earlier cell
is inserted. My inserts for 1d and 2b used cell-id targets computed from an
earlier snapshot, landed at the wrong (shifted) positions, and ended up
splitting section 1c's markdown from its own code cell. Caught this by
diffing the intended structure against a fresh full read (which showed the
actual current ids), rather than assuming my inserts landed where intended.
Fixed by rewriting the whole file in one clean pass with explicit stable
`id` fields on every cell -- avoids this class of bug recurring on any
future edit to this notebook.

Section 1d ran: `pip install -U llmcompressor "compressed-tensors>=0.17.0"`
resolved to `llmcompressor==0.12.0` (unchanged -- apparently already the
latest on PyPI, `-U` had nothing to do) and `compressed-tensors==0.17.0`
(actually DROPPED from the earlier 0.17.1 -- llmcompressor evidently pins
it tighter than our `>=0.17.0`). Packed shape: still `(3072, 103)`, same as
before. Second confirmed negative result -- we've now tried both obvious
levers (pin compressed-tensors>=0.17.0 alone; upgrade llmcompressor too)
and neither changes the packing. Skipped 2b (redundant -- the direct
safetensors shape check already answers the question, no need to spend a
vLLM load confirming what's already known). Drafted a reply asking the
maintainer directly whether a specific llmcompressor version/branch or a
recipe/config setting is needed, since we're out of things to try blindly.
Updated the notebook's intro and report-back sections to reflect this.

**2026-08-10** — user asked whether to send a follow-up bump (been ~2 weeks
since our last comment). Checked actual issue/PR status via the public
GitHub API (no gh CLI here, but curl + api.github.com works fine for public
repos). Findings: issue #49893 still open; PR #49900 still open/unmerged but
updated Aug 4 -- harjothkhara added a mixed-config_groups per-layer-scheme
test (Jul 30) then asked (Aug 4) for a maintainer to add the `ready` label,
because vLLM's pre-run-check gates CI behind 4+ merged PRs and they have 1.
So the fix is done (code + test) but merge-blocked on vLLM core maintainers,
not on us or harjothkhara. User first drafted a nicely-reframed "just
logging findings, no action expected" comment -- good posture -- but I
flagged the content was a duplicate of our already-posted 2026-07-30
comment (same versions, same 103-column result, same question), so posting
it would just be re-logging stale info. Agreed to DROP posting anything and
just wait, re-checking harjothkhara/PR activity after ~another week of
staleness (~2026-08-17+). If a comment ever becomes warranted, it should
carry genuinely new data (isolation check 2, the W4/W4 drafter end-to-end
confirmation) rather than repeat what's already in the thread. Also
confirmed via the API that the user's author_association is NONE -- they're
the issue reporter, not an official vLLM contributor (which requires a
merged commit; the fix PR is harjothkhara's, not theirs). Updated
context.md with the hold decision + recheck plan.

**2026-08-14** — a third person, `medhavee-upadhyaya`, commented on the
issue: independently traced the root cause on current `main`
(`DraftModelProposer._create_draft_vllm_config()` forces `quant_config=None`
while vLLM already exposes `get_draft_quant_config(vllm_config)` to resolve
the draft checkpoint's own quant metadata -- a crisper mechanism-level
diagnosis than we or harjothkhara had stated), offered to fix it, then an
hour later found PR #49900 already addresses it and gracefully bowed out
("will not open a competing PR... defer to the existing contributor's
work"). No action needed on our end -- nobody asked us anything, and they
self-resolved. Net status unchanged: still waiting on #49900, still
CI-gated. Positive signal though: the issue is drawing capable contributors
who independently confirm the same failure, which reflects well on the
report. Our hold-and-wait posture stands.
