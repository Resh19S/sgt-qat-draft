# Findings

Formal, literature-style record of every metric and result produced in this project.
Each entry should note: date, method/configuration, exact numbers, and any caveats
needed to interpret them correctly. Write for a reader who wasn't in the room —
this is the raw material for `paper-draft.md`.

## 2026-07-22 — SGT-QAT checkpoint export (Qwen3-1.7B, seed 42)

**Method**: `notebooks/01_export_sgt_qat_checkpoint.ipynb`, adapting the prior
project's flagship recipe (`quant research/notebooks/15_mixed_precision_guided_targeted_qat.ipynb`)
unchanged: Stage 1 = single `llmcompressor.oneshot()` GPTQ call with `config_groups`
(sensitivity-ranked layers protected at W4, rest at W3, group_size=128, symmetric
int); Stage 2 = targeted QAT (custom `FakeQuantize` STE, 500 steps, batch=1024
tokens, lr=1e-5, AdamW8bit) fine-tuning only the still-W3 layers on WikiText-2 train.
Calibration: `allenai/c4`, 128 samples, seq_len 2048, seed 42. Hardware: A100-SXM4-40GB
(Colab Pro). New relative to notebook 15: a real compressed export
(`save_pretrained(..., save_compressed=True)`) — the prior project never persisted a
checkpoint at all.

**Metrics** (WikiText-2 test perplexity):
- Stage 1 (mixed precision only, before QAT): PPL 22.37
- Stage 1+2 (combined, final): PPL 15.91
- Protected layers: 53 of 196 (15.62% of quantized params, target was 15%)
- Trainable params in Stage 2: 1189.1M (still-W3 layers only)
- Exported checkpoint size: 1184.8 MB (vs. ~3.4 GB fp16 Qwen3-1.7B — confirms
  `save_compressed=True` produced a genuinely packed checkpoint, not a full-precision
  save)

**Comparison baseline(s)** (prior project, `quant research`, same seed=42):
- fp16 (no quantization): PPL 16.67
- fp16, finetuned on the same WikiText-2 data (recovery ceiling): PPL 9.99
- Pure GPTQ-W3 (no recovery): PPL 28.50
- Full-parameter QAT (all layers trained, no mixed precision): PPL 17.53

Using the prior project's "corrected recovery" formula,
`100 × (ptq_w3 − result) / (ptq_w3 − ceiling)`:
- This run's combined recovery: **68.0%**
- Prior project's own combined-method runs (same recipe, different seeds/hardware):
  63.4% (seed 42) / 65.8% (seed 123)

**Caveats / threats to validity**:
- Combined PPL (15.91) beats the raw fp16 PPL (16.67). This is *expected*, not a bug:
  Stage 2 fine-tunes on WikiText-2 train, so it's being compared against a fp16 model
  that never saw that data — the fair ceiling is the finetuned-fp16 PPL (9.99), which
  this result still sits well above.
- This run's recovery (68.0%) is higher than the prior project's own two runs of the
  identical recipe (63.4%/65.8%). Batch size, steps, and seed all match the original —
  only the hardware differs (A100-40GB here vs. whatever the prior runs used). Treat
  this as a promising but unreplicated data point, not yet a confirmed improvement —
  worth a second seed/run before leaning on it in the paper.
- `save_compressed=True` behavior (does it correctly re-quantize Stage-2-trained
  weights onto the grid implied by their stored scale/zero_point, vs. silently
  fudging it) has not been independently verified beyond the plausible file-size
  reduction — worth a spot-check (e.g. reload and inspect a few weight tensors) before
  trusting downstream benchmark numbers built on this checkpoint.

**Raw data**: `results/export_sgt_qat_checkpoint_seed42_2026-07-22T20-49-26.json`

## 2026-07-23 — Phase 2 baseline: no-spec-decode vs. EAGLE-3 (Qwen3-8B)

**Method**: `notebooks/02_baseline_eagle3.ipynb` via `notebooks/common/bench_utils.py`.
Target model `Qwen/Qwen3-8B`, EAGLE-3 drafter `Tengyunw/qwen3_8b_eagle3`
(`speculative_config={"method": "eagle3", "model": "Tengyunw/qwen3_8b_eagle3",
"num_speculative_tokens": 3}`). Hardware: A100-SXM4-40GB (Colab Pro). 80 prompts,
256 max output tokens, temperature 0, greedy. **Low-concurrency methodology**:
prompts submitted one at a time (sequential `llm.generate([prompt])` per request),
not batched — see caveats below for why this matters. vLLM 0.25.1
(`pip install vllm`, not pinned to the `vendor/vllm` commit from Phase 1
orientation). GPU memory measured via `nvidia-smi` at end-of-generation (total
device memory in use, not `torch.cuda.max_memory_allocated()` — see caveats).

**Metrics**:

| | no-spec-decode | EAGLE-3 |
|---|---|---|
| Throughput | 76.29 tok/s | 136.76 tok/s |
| Wall-clock (80 prompts) | 268.47s | 149.75s |
| **Speedup** | 1.00x | **1.79x** |
| GPU memory in use | 40,084,963,328 B (37.33 GiB) | 41,443,917,824 B (38.60 GiB) |
| Mean acceptance length | n/a | 2.023 |
| Per-position acceptance rate | n/a | [0.596, 0.283, 0.145] |
| Avg draft acceptance rate | n/a | 34.1% (10,359 accepted / 30,366 drafted) |

Memory delta (EAGLE-3 drafter's own weights + KV cache): **+1.30 GiB**
(1,358,954,496 bytes).

**Comparison baseline(s)**: no-speculative-decode run in the same notebook, same
prompts, same hardware, same session (back-to-back).

**Caveats / threats to validity**:
- **Prompts are still placeholder smoke-test text** — 3 sentences cycled to fill 80
  slots, not a real benchmark dataset (e.g. mt-bench). Directionally informative
  (real speedup, plausible acceptance rate) but not final numbers for the paper.
- **A first attempt at this same comparison, submitting all 80 prompts as one batch
  to `llm.generate()`, produced 0.98x (EAGLE-3 marginally slower than no-spec) with
  GPU memory reading 0 (a measurement bug, since fixed).** The 0.98x number is not
  included in the table above and should not be cited — it reflects vLLM's
  continuous batching saturating the GPU regardless of drafter, which is a real and
  known property of speculative decoding (its throughput benefit is a
  low-concurrency effect), not a finding about EAGLE-3 itself. Documented here so a
  future session doesn't rediscover the same confusion; see `docs/logs.md`
  2026-07-23 for the full debugging trail.
- GPU memory is measured via `nvidia-smi` (whole-device usage) because vLLM's V1
  engine runs model execution in separate worker subprocess(es) —
  `torch.cuda.max_memory_allocated()` in the notebook kernel process sees almost
  nothing regardless of actual usage. This is end-of-generation memory in use, not a
  true instantaneous peak.
- vLLM installed via plain `pip install vllm`, not pinned to the `vendor/vllm`
  commit read during Phase 1 orientation — a deliberate speed/reproducibility
  trade-off (see `docs/context.md`), revisit if exact-source reproducibility matters
  later.

**Raw data**: `results/no_spec_decode_2026-07-23T18-42-32.626431+00-00.json`,
`results/baseline_eagle3_2026-07-23T18-46-18.541328+00-00.json`

## 2026-07-24 — vLLM `draft_model` cannot load compressed-tensors checkpoints (finding, not a benchmark result)

**Method**: attempted to load `checkpoints/qwen3-1.7b-sgt-qat/` (the compressed
export from notebook 01) via `speculative_config={"method": "draft_model", "model":
"checkpoints/qwen3-1.7b-sgt-qat", "num_speculative_tokens": 3}` against Qwen3-8B,
vLLM 0.25.1. Diagnosed with `VLLM_ENABLE_V1_MULTIPROCESSING=0` +
`VLLM_LOGGING_LEVEL=DEBUG` after an initial attempt failed with an opaque
`RuntimeError: Engine core initialization failed` whose real cause was swallowed by
vLLM's spawned worker subprocess.

**Result**: confirmed failure, not a config or environment issue on our side:

```
ValueError: There is no module or parameter named 'layers.0.mlp.down_proj.weight_packed'
in Qwen3Model. The available parameters belonging to layers.0.mlp.down_proj
(RowParallelLinear) are: {'layers.0.mlp.down_proj.weight'}
```

vLLM constructs the draft model's linear layers as plain, unquantized
`RowParallelLinear` (expecting an ordinary `.weight` tensor), then fails to find it
because our checkpoint genuinely stores compressed-tensors packed weights under
`.weight_packed` (plus separate scale/zero-point buffers). Despite `VllmConfig`
correctly re-deriving `quant_config` for the draft model from its own
`config.json` at the config level (traced via source, see the 2026-07-23/24
`context.md` entries), that awareness does not make it through to how the draft
model's actual `nn.Module` layers get constructed. **vLLM's `method="draft_model"`
path, as of v0.25.1, does not support compressed-tensors packed draft checkpoints.**

**Fix adopted**: notebook 03 now reloads the compressed checkpoint and genuinely
decompresses it before re-saving without `save_compressed=True`, to
`checkpoints/qwen3-1.7b-sgt-qat-plain/`. This took several attempts to get right —
`AutoModelForCausalLM.from_pretrained()` alone does *not* dequantize (keeps weights
packed in `CompressedLinear` modules, dequantizing only at inference time), and
`hf_quantizer.dequantize()` raises `NotImplementedError` for this quant method in
the installed `transformers` version. The working approach:
`compressed_tensors.ModelCompressor.from_pretrained_model(model)` +
`.decompress_model(model)` to unpack weights, then explicitly replacing each
still-quantized-typed module with a genuine `nn.Linear` (the wrapper module's own
serialization logic kept re-adding `weight_scale`/`weight_shape` even after the
underlying buffers were deleted). Verified via direct inspection of the saved
checkpoint's safetensors tensor names (no leftover quantization-related keys)
before attempting the (expensive) vLLM load again. Full debugging trail in
`docs/logs.md` 2026-07-24. No re-running of notebook 01's Stage 1/2 GPU pipeline
required — this is a reload-and-resave of the already-exported checkpoint.

**Consequence for the memory-footprint comparison**: the drafter actually
benchmarked in notebook 03 is a full ~3.4GB fp16 checkpoint, not the 1.18GB
compressed one. The accuracy/QAT-training benefit of the SGT-QAT method is still
present (same trained weight values, just stored at full precision instead of
packed) — only the on-disk/VRAM compression story is lost for *this* benchmark.
This should be stated plainly in the paper as a current tooling limitation (vLLM
draft-model quantization support), not a property of the SGT-QAT method itself —
the compressed checkpoint from notebook 01 (1184.8MB, confirmed genuinely packed)
remains valid evidence that the *export* pipeline works; it's specifically vLLM's
speculative-decoding drafter loader that can't consume it yet.

**Raw data**: none (this is a negative/diagnostic result, not a benchmark run) —
see `docs/logs.md` 2026-07-24 for the full debugging trail including the two red
herrings (`fileno()` crash, quantization-config source trace) encountered along
the way.

## 2026-07-24 — Phase 3: SGT-QAT drafter vs. EAGLE-3 vs. no-spec-decode (Qwen3-8B)

**Method**: `notebooks/03_sgt_qat_drafter_bench.ipynb`, same harness and
low-concurrency methodology as the EAGLE-3 baseline. Drafter:
`checkpoints/qwen3-1.7b-sgt-qat-plain/` — the **decompressed** SGT-QAT checkpoint
(see the entry above; vLLM's `draft_model` path cannot load the genuinely
compressed 1.18GB checkpoint, so this is a plain ~3.4GB fp16 version of the same
trained weights). `speculative_config={"method": "draft_model", "model": <plain
checkpoint path>, "num_speculative_tokens": 3}`. 80 prompts (same placeholder set
as the other two conditions), 256 max output tokens, temperature 0,
`max_model_len=4096`. The no-spec/EAGLE-3 baselines were **re-run 2026-07-24
with `max_model_len=4096` to match** (the original 2026-07-23 baseline run used
vLLM's default, 40960 — see "Speed comparison" below for why this was fixed and
what it confirmed).

**Metrics** (all three conditions now share `max_model_len=4096`):

| | no-spec-decode | EAGLE-3 | SGT-QAT drafter |
|---|---|---|---|
| Throughput | 76.73 tok/s | 136.58 tok/s | 33.69 tok/s |
| Wall-clock (80 prompts) | 266.90s | 149.95s | 607.90s |
| **Speedup vs. no-spec** | 1.00x | 1.78x | **0.44x** |
| GPU memory in use | 36.75 GiB | 37.80 GiB | 37.26 GiB (still not directly comparable — see caveats) |
| Mean acceptance length | n/a | 2.023 | **2.488** |
| Per-position acceptance rate | n/a | [0.596, 0.283, 0.145] | **[0.689, 0.474, 0.325]** |
| Avg draft acceptance rate | n/a | 34.1% | **49.6%** (12,218 / 24,627) |

**Speed comparison — now airtight.** The no-spec/EAGLE-3 baselines were originally
measured with vLLM's default `max_model_len` (40960) while the SGT-QAT run used
4096 (an OOM mitigation) — a real methodological gap. Re-running the baselines
with `max_model_len=4096` to match moved throughput by <1% in both cases
(76.29→76.73 tok/s, 136.76→136.58 tok/s) and left mean acceptance length
unchanged (2.0234 both times) — confirming `max_model_len` has no meaningful
effect on speed/acceptance at these sequence lengths, as expected. **The speed and
acceptance-rate comparisons in this table are now on solid methodological
footing.**

**Headline finding**: the SGT-QAT drafter achieves a **substantially higher
acceptance rate than EAGLE-3 at every speculative position** — roughly double at
positions 1 and 2, and meaningfully higher at position 0. This is a genuinely
strong quality signal: the target model agrees with our drafter's predictions far
more often than with EAGLE-3's. However, wall-clock throughput is *worse than no
speculation at all* (0.44x) — running a full 1.7B fp16 model as the drafter is
computationally heavy enough per step that the compute cost outweighs the benefit
of the higher acceptance rate. This is the direct, expected consequence of being
forced to use the plain (uncompressed) checkpoint (see the entry above) rather
than a small or genuinely compressed drafter — the quality signal is real, but
this particular deployment path can't currently turn it into a speedup.

**Comparison baseline(s)**: no-spec-decode and EAGLE-3 runs from
`notebooks/02_baseline_eagle3.ipynb`, `max_model_len=4096`-matched re-run,
2026-07-24. Same target model, same prompts, same hardware — but a different
Colab session/VM instance than the SGT-QAT run (see caveats on GPU memory).

**Caveats / threats to validity**:
- **Drafter is the plain/uncompressed checkpoint**, not the genuinely compressed
  one — see the 2026-07-24 finding above. The acceptance-rate numbers are still
  valid (same trained weight values, decompression is lossless dequantization),
  but the speed/memory numbers reflect a full-precision 1.7B drafter, not what a
  real compressed deployment would look like.
- **GPU memory still not directly comparable, for a different reason than
  before**: `max_model_len` is now matched across all three conditions (fixed
  2026-07-24), which resolves the KV-cache-size confound. But `gpu_memory_used_bytes`
  is an *absolute whole-device* reading (via `nvidia-smi`), and the SGT-QAT run
  was measured in a **different Colab session/VM instance** than the no-spec/EAGLE-3
  baselines (different day, not back-to-back) — absolute memory readings across
  different runtime instances aren't guaranteed comparable (different baseline
  driver/OS overhead, etc.), independent of any config matching. Memory is
  explicitly deprioritized for now (see `docs/context.md`) — `notebooks/04` (see
  below) targets a different, session-independent angle on this instead of chasing
  a same-session 3-way re-run.
- Placeholder prompts throughout (see prior entries) — not a real benchmark
  dataset yet.
- See `docs/context.md`/`docs/logs.md` for the planned follow-up (`notebooks/04`):
  measuring the *genuinely compressed* checkpoint's standalone VRAM footprint
  (loaded directly via `transformers`, not through vLLM) to at least partially
  recover the memory story this benchmark couldn't answer.

**Raw data**: `results/sgt_qat_drafter_2026-07-24T17-12-10.285012+00-00.json`
(SGT-QAT drafter, `max_model_len=4096`),
`results/no_spec_decode_2026-07-24T17-39-17.319534+00-00.json`,
`results/baseline_eagle3_2026-07-24T17-44-02.392776+00-00.json` (baselines,
`max_model_len=4096`-matched re-run). The original 2026-07-23
`max_model_len=40960` baseline run (`results/no_spec_decode_2026-07-23T18-42-32...json`,
`results/baseline_eagle3_2026-07-23T18-46-18...json`) is superseded for this
comparison but kept for the record.

## 2026-07-25 — Standalone memory footprint: compressed SGT-QAT checkpoint

**Method**: `notebooks/04_compressed_checkpoint_memory.ipynb`. Loads
`checkpoints/qwen3-1.7b-sgt-qat/` (the genuinely compressed 1.18GB checkpoint —
the one vLLM's `draft_model` path can't load, see the 2026-07-24 finding above)
directly via `transformers.AutoModelForCausalLM.from_pretrained()`, deliberately
**without** decompressing, and reads `nvidia-smi` GPU memory before/after. No
vLLM, no target model, no KV cache, no speculative decoding — a standalone
weight-only measurement, session-independent by design (unlike the notebook
02/03 numbers, this doesn't depend on comparing absolute readings across
different Colab runtime instances for its own internal validity).

**Metric**: compressed checkpoint's own VRAM footprint: **1.629 GiB**
(1,749,024,768 bytes).

**Context, not a direct comparison**: EAGLE-3's full in-vLLM memory delta
(weights + its own KV cache + serving overhead, from the `max_model_len=4096`-matched
run) was **1.047 GiB**. The compressed SGT-QAT drafter's standalone weight-only
footprint is *already larger* than that — despite excluding KV cache/serving
overhead that would only add more. **Not an apples-to-apples comparison** (one
is standalone weights, the other is full serving), so this isn't proof SGT-QAT
loses on memory even when compressed — but it's a real, informative signal:
compression narrows the gap against the plain checkpoint (1.629 GiB vs. an
extrapolated ~4-5 GiB the plain ~3.4GB checkpoint would likely need standalone,
not measured this run), but doesn't obviously make a full 1.7B dense drafter
competitive with a purpose-built lightweight EAGLE-3-style head on memory alone.
Worth stating plainly in the paper rather than assuming compression would have
been a clean win if only vLLM supported it.

**Caveats / threats to validity**:
- **Plain checkpoint comparison point is missing** (`plain_checkpoint_delta_bytes:
  null` in the raw data) — the plain/decompressed checkpoint from notebook 03
  only ever existed on that session's local Colab disk, never backed up to
  Drive, so this (different) session couldn't find it to load. Would need
  either a fresh decompression run or a Drive backup of the plain checkpoint to
  fill this in with a real, same-methodology number instead of an extrapolation.
- Not comparable to notebook 02/03's `gpu_memory_used_bytes` numbers directly —
  different measurement context (standalone load vs. full vLLM serving) as well
  as the cross-session absolute-reading caveat already noted for those.
- Still doesn't answer the question that actually matters for the paper (would
  a genuinely compressed drafter be fast *and* memory-efficient inside vLLM) —
  that remains blocked on vLLM gaining compressed-tensors draft-model support.

**Raw data**: `results/compressed_checkpoint_memory_2026-07-25T07-50-21.json`

## 2026-07-25 — Real-prompt baseline re-run: no-spec vs. EAGLE-3 (supersedes placeholder-prompt numbers)

**Method**: `notebooks/02_baseline_eagle3.ipynb`, re-run with real mt-bench prompts
(`philschmid/mt-bench` via `bench_utils.load_benchmark_prompts()`, which wraps
vLLM's own `vllm.benchmarks.datasets` dataset utilities — the same convention as
`spec_decode_offline.py --test`), replacing the placeholder smoke-test text used
in every prior run. Same harness/methodology otherwise as the 2026-07-24
`max_model_len=4096`-matched entry above: target `Qwen/Qwen3-8B`,
`num_speculative_tokens=3`, 80 prompts, 256 max tokens, low-concurrency
(sequential single-request) benchmarking, `max_model_len=4096`.

**Metrics**:

| run | tok/s | speedup | mean acceptance length | GPU memory |
|---|---|---|---|---|
| no_spec_decode | 76.6 | 1.00x | — | 36.75 GiB (39,460,012,032 B) |
| baseline_eagle3 | 165.4 | **2.16x** | **2.474** | 37.80 GiB (40,584,085,504 B) |

Per-position acceptance rate (EAGLE-3): [0.714, 0.465, 0.296] for speculative
positions 1-3.

**This materially changes the EAGLE-3 baseline** versus the placeholder-prompt
run logged 2026-07-24: speedup went from 1.78x → **2.16x**, and mean acceptance
length from 2.023 → **2.474**. Placeholder text (short, repetitive smoke-test
strings) apparently made EAGLE-3's draft harder to accept than it is on
realistic conversational prompts — the earlier placeholder-prompt numbers
understated EAGLE-3's real performance. Any Phase 3 SGT-QAT-drafter comparison
against this baseline must also use real prompts (notebook 03, in progress) or
the two won't be comparable.

**Memory note**: the GPU memory delta between conditions
(40,584,085,504 − 39,460,012,032 = 1,124,073,472 B ≈ **1.047 GiB**) is *identical*
to the delta measured in the placeholder-prompt, `max_model_len`-matched
2026-07-24 run. Confirms memory usage in this harness is driven by model
weights + KV cache sizing (`max_model_len`), not prompt content — expected, but
good to have confirmed with real data rather than assumed.

**Caveats / threats to validity**: same as the 2026-07-24 entry above (GPU
memory is an absolute whole-device `nvidia-smi` reading; not guaranteed
comparable across different Colab VM instances if a future run isn't
back-to-back with this one). This run's two conditions were measured
back-to-back in the same session, so the memory delta between *them* is valid;
cross-session comparisons (e.g. against notebook 04's standalone numbers) still
carry that caveat.

**Raw data**: `results/no_spec_decode_2026-07-25T08-22-23.566257+00-00.json`,
`results/baseline_eagle3_2026-07-25T08-26-34.711075+00-00.json`

## Template for future entries

### [Date] — [Experiment name]

**Method**: model(s), configuration, hardware, vLLM version/commit, spec-decode
parameters (num speculative tokens, etc.).

**Metrics**:
- Acceptance rate:
- Wall-clock speedup (vs. no spec-decode):
- Memory footprint:

**Comparison baseline(s)**:

**Caveats / threats to validity**:

**Raw data**: `results/<filename>`
