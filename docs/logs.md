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
