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
