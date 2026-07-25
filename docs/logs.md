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
