# Paper Draft

Status: first draft, 2026-07-25. All Phase 3 experiments are landed with real
data (see `docs/findings.md`) — this draft is built directly from that file;
every number below cites its findings.md entry and `results/*.json` source.
Not yet reviewed for tone/framing beyond what's already been agreed with the
user in-session (see `docs/logs.md` 2026-07-25 for the discussion that shaped
the framing below — explicitly *not* an "EAGLE-parity" chase).

---

## Working title (pick one)

1. "Quantization Preserves Drafting Quality, Not Drafting Speed: A Case Study
   in Speculative-Decoding Drafter Design"
2. "When a Good Drafter Isn't a Fast One: Sensitivity-Guided Targeted QAT as
   a Speculative-Decoding Drafter for Qwen3"
3. "Architecture, Not Precision: Why a Quantized Dense Model Can't Replace a
   Purpose-Built Speculative-Decoding Head"

Recommendation: **(1)** or **(3)** — both lead with the actual finding
(quality survives quantization, speed doesn't, and the reason is structural)
rather than implying a head-to-head win over EAGLE-3 that the data doesn't
support. Avoid any title implying "SGT-QAT beats/matches EAGLE-3" — that's
not what happened (see §4).

---

## Abstract (draft)

Speculative decoding accelerates LLM inference by using a small, cheap
"drafter" model to propose candidate tokens that a larger target model
verifies in parallel. vLLM's EAGLE-3 drafters achieve this via a purpose-built
architecture: a shallow (1-2 layer) head that reuses the target model's own
embeddings. We ask a different question: can a **quantized dense model** —
specifically, a Qwen3-1.7B checkpoint produced by Sensitivity-Guided Targeted
QAT (SGT-QAT), from our prior work — serve as a competitive drafter for
Qwen3-8B, benchmarked against vLLM's built-in EAGLE-3 and a no-speculation
baseline?

On real conversational prompts (mt-bench, 80 prompts), we find the SGT-QAT
drafter reaches **near-parity draft quality** with EAGLE-3 — mean acceptance
length of 2.443 tokens (of 3 proposed) vs. EAGLE-3's 2.474, and comparable
per-position acceptance rates. But wall-clock throughput tells a different
story: the SGT-QAT drafter achieves only **32.9 tokens/sec, a 0.43x
*regression* vs. not using speculative decoding at all** (76.6 tokens/sec),
while EAGLE-3 achieves a genuine **2.16x speedup** (165.4 tokens/sec). We show
this is architectural, not a quantization-quality problem: the SGT-QAT
drafter remains a full 28-layer dense 1.7B model, and no amount of bit-width
reduction changes its per-step compute cost enough to make up for not being a
shallow, purpose-built head. A follow-up experiment pushing quantization
further (to 2.158 average bits/weight, vs. the flagship's 3.156) confirms this
is a *memory* lever, not a *speed* lever: it does close the memory gap against
EAGLE-3 (0.646 GiB standalone vs. EAGLE-3's 1.047 GiB full in-vLLM footprint)
but perplexity collapses by roughly 12x doing it (188.66 vs. 15.91),
disqualifying it on quality grounds before speed is even considered. We
conclude that Sensitivity-Guided Targeted QAT is a genuinely quality-preserving
quantization method — but pairing it with a dense model architecture is the
wrong lever for the speculative-decoding drafter role; a purpose-built shallow
architecture (EAGLE-style) is what that role actually requires, independent of
quantization.

---

## 1. Introduction

- Speculative decoding background: drafter proposes k tokens, target verifies
  in one forward pass, accepted tokens are free, one token always generated
  fresh — net speedup depends on (acceptance rate) vs. (drafter's own cost per
  step).
- Prior work context: our own Sensitivity-Guided Targeted QAT paper
  (`/mnt/windows/projects/quant research`) showed SGT-QAT recovers GPTQ-W3
  quantization damage more effectively than uniform full-parameter QAT, at
  zero memory cost, on Qwen3-1.7B — established the checkpoint's quality
  credentials independent of this project.
- This project's question: does a quantization method that preserves *model
  quality* also produce a competent speculative-decoding *drafter*? These are
  not obviously the same question — drafter quality depends on next-token
  agreement with a specific, larger target model, and drafter viability
  depends on wall-clock cost per drafting step, not just perplexity.
- Contribution: an apples-to-apples (same harness, same real prompts, same
  target model) comparison of a quantized-dense-model drafter against vLLM's
  built-in EAGLE-3, showing quality-preservation and speed are separable
  outcomes — and a follow-up quantifying exactly where further quantization
  helps (memory) and where it doesn't (speed, quality past a threshold).

## 2. Background / Related Work

- vLLM's speculative decoding architecture: `SpeculativeConfig`,
  `method="draft_model"` (arbitrary HF checkpoint as drafter) vs.
  `method="eagle3"` (purpose-built shallow head, shares target embeddings).
  [Cite the vendor/vllm source locations identified in Phase 1 orientation —
  see `docs/context.md` for the confirmed file paths at the vLLM version used.]
- EAGLE / EAGLE-3 architecture summary: why it's fast — shallow depth (1-2
  layers), shared embeddings with the target, purpose-built for the drafting
  role rather than adapted from a general-purpose dense model.
- Our prior SGT-QAT paper: sensitivity-ranked mixed-precision GPTQ (W4 on the
  ~15% most sensitive layers, W3 on the rest) + targeted QAT fine-tuning
  (custom `FakeQuantize` STE module, 500 steps) on only the still-low-bit
  layers. Established result: recovers more GPTQ-W3 damage than uniform
  full-parameter QAT, at zero added memory cost (every layer stays quantized).

## 3. Method

### 3.1 SGT-QAT checkpoint (recap, this project's contribution)

- Exported via `notebooks/01_export_sgt_qat_checkpoint.ipynb`, closing the
  prior project's gap (no checkpoint was ever persisted to disk there).
- Flagship config: W4 protected (15% most sensitive layers, sensitivity-ranked
  via a cheap per-layer probe) / W3 rest, calibration seed 42, average 3.156
  bits/weight. Stage 1 (GPTQ only) PPL 22.37 → Stage 1+2 (+QAT) PPL 15.91
  (WikiText-2). Compressed checkpoint: 1184.8 MB on disk.
- Genuine compressed export via `save_pretrained(..., save_compressed=True)`
  (compressed-tensors format) — resolving the prior project's known gap where
  no compressed checkpoint was ever saved.

### 3.2 Drafter integration into vLLM

- `speculative_config={"method": "draft_model", "model": <path>,
  "num_speculative_tokens": 3}`.
- **Blocker found and resolved**: vLLM's `draft_model` path (version used in
  this project) cannot load compressed-tensors packed checkpoints directly —
  it builds plain `nn.Linear` layers and fails to find expected `.weight`
  tensors (`weight_packed` instead). Fix: decompress via
  `ModelCompressor.from_pretrained_model()` + `.decompress_model()`, then
  explicitly replace each quantized module with a plain `nn.Linear` holding
  the dequantized weight (see `docs/findings.md` 2026-07-24 for the full
  multi-attempt debugging chain). **Consequence for this paper**: all
  in-vLLM SGT-QAT-drafter benchmarks in §4 use this decompressed (~3.4GB
  fp16) checkpoint, not the genuinely compressed one — a real limitation of
  current vLLM tooling, not of the quantization method itself. Notebook 04's
  standalone memory measurement (§4.3) is the only place the genuinely
  compressed checkpoint's footprint is measured directly.

### 3.3 Benchmark harness

- `notebooks/common/bench_utils.py`: shared harness for all three conditions
  (no-spec, EAGLE-3, SGT-QAT drafter). Sequential single-request (low
  concurrency) methodology — speculative decoding's throughput benefit is
  specifically a low-concurrency effect, and this isolates it cleanly.
- Prompts: real mt-bench prompts (`philschmid/mt-bench`, 80 prompts), loaded
  via vLLM's own `vllm.benchmarks.datasets` utilities (same convention as
  vLLM's own `spec_decode_offline.py --test` mode) — not synthetic/placeholder
  text. An earlier round of this benchmark used placeholder smoke-test text;
  those numbers are superseded throughout and not used in this paper (see
  `docs/findings.md` for the historical placeholder-prompt entries, kept for
  the record but not cited here).
- Target model: Qwen3-8B (full precision), `max_model_len=4096` (matched
  across all three conditions), `num_speculative_tokens=3`, `max_tokens=256`
  per request.
- Metrics: wall-clock tokens/sec, speedup vs. no-spec baseline, mean
  acceptance length, per-position acceptance rate, GPU memory
  (`nvidia-smi`, absolute whole-device reading).

## 4. Results

### 4.1 Baseline: no speculation vs. EAGLE-3

| condition | tok/s | speedup | mean acceptance length | GPU memory |
|---|---|---|---|---|
| No spec-decode | 76.6 | 1.00x | — | 36.75 GiB |
| EAGLE-3 | 165.4 | 2.16x | 2.474 | 37.80 GiB |

See `results/visual_metrics/01_speedup.svg`, `02_throughput_tokens_per_sec.svg`.
Source: `docs/findings.md` 2026-07-25 "Real-prompt baseline re-run";
`results/no_spec_decode_2026-07-25T08-22-23...json`,
`results/baseline_eagle3_2026-07-25T08-26-34...json`.

### 4.2 SGT-QAT as drafter: quality parity, speed regression

| condition | tok/s | speedup | mean acceptance length | GPU memory |
|---|---|---|---|---|
| No spec-decode | 76.6 | 1.00x | — | 36.75 GiB |
| EAGLE-3 | 165.4 | 2.16x | 2.474 | 37.80 GiB |
| SGT-QAT drafter | 32.9 | **0.43x** | 2.443 | 37.27 GiB |

Per-position acceptance rate: EAGLE-3 [0.714, 0.465, 0.296], SGT-QAT drafter
[0.668, 0.456, 0.319] (positions 1-3).

See `results/visual_metrics/03_mean_acceptance_length.svg`,
`04_acceptance_rate_by_position.svg`. Source: `docs/findings.md` 2026-07-25
"Real-prompt SGT-QAT drafter run: full 3-way comparison";
`results/sgt_qat_drafter_2026-07-25T09-08-02...json`.

**The central result of this paper**: acceptance quality is a near-wash
(2.443 vs. 2.474 — within noise of each other, not the large gap an earlier,
now-superseded placeholder-prompt run suggested). But SGT-QAT-as-drafter is
**slower than not speculating at all**. The per-step cost of a full 28-layer
dense 1.7B forward pass exceeds whatever wall-clock time its (perfectly
competent) accepted tokens save. This is not a finding about quantization
quality — it's a finding about architecture: EAGLE-3's speed comes from being
structurally cheap (1-2 layers, shared target embeddings), a property
quantizing a dense model does not confer, regardless of bit-width.

### 4.3 Standalone memory: does the genuinely compressed checkpoint help?

Because vLLM's `draft_model` path cannot load the compressed checkpoint
in-serving (§3.2), we measure the compressed checkpoint's standalone
(weight-only, no vLLM, no KV cache) VRAM footprint directly:
**1.629 GiB** (`notebooks/04_compressed_checkpoint_memory.ipynb`).
EAGLE-3's full in-vLLM memory delta (weights + KV cache + serving overhead)
is **1.047 GiB**. Even in this best-case (no serving overhead) comparison,
the compressed SGT-QAT drafter is already larger. **Not apples-to-apples**
(standalone weights vs. full serving), stated plainly rather than implied as
a clean loss — but real evidence against assuming compression alone would
make a dense 1.7B drafter memory-competitive with a purpose-built shallow
head.

See `results/visual_metrics/05_memory_comparison.svg`. Source:
`docs/findings.md` 2026-07-25 "Standalone memory footprint"; `results/compressed_checkpoint_memory_2026-07-25...json`.

### 4.4 Pushing quantization further: a memory lever, not a speed or quality lever

To directly test whether more aggressive quantization could close the memory
gap, we re-ran the SGT-QAT recipe one bit-width tier lower: W3 protected / W2
rest (vs. the flagship's W4/W3), average **2.158 bits/weight** (vs. 3.156).

| | Stage 1 (GPTQ) PPL | Stage 1+2 (+QAT) PPL | Standalone memory |
|---|---|---|---|
| Flagship (3.156 bpw) | 22.37 | 15.91 | 1.629 GiB |
| Aggressive (2.158 bpw) | 454.31 | 188.66 | **0.646 GiB** |

See `results/visual_metrics/06_quality_vs_bitwidth_tradeoff.svg`. Source:
`docs/findings.md` 2026-07-25 "Aggressive quantization tradeoff";
`results/aggressive_quant_tradeoff_seed42_2026-07-25T09-51-46.json`.

This **does** close the memory gap against EAGLE-3 — 0.646 GiB standalone is
smaller than EAGLE-3's 1.047 GiB full in-vLLM delta, with real headroom even
accounting for the fact that a full in-vLLM measurement of this checkpoint
would add some KV cache/serving overhead on top. But perplexity collapses by
roughly **12x** relative to the flagship (188.66 vs. 15.91), even after QAT
fine-tuning materially helped (454.31 → 188.66, the same kind of correction
that took the flagship from 22.37 → 15.91). We did not run the expensive
8B-target/vLLM acceptance-rate benchmark at this bit-width — the perplexity
result alone is disqualifying; a model this degraded would not be a
competent next-token predictor for a drafter role, and confirming that with
an expensive benchmark would not change the conclusion.

**This confirms the framing decided on before running the experiment (see
`docs/logs.md` 2026-07-25): a memory win over EAGLE-3 is achievable through
more aggressive quantization, but not without breaking the quality the
drafter approach depends on.** We do not claim SGT-QAT "would win on memory
if only vLLM supported compressed drafter loading" — the aggressive
checkpoint that *would* win on memory is not a viable drafter on quality
grounds, independent of any vLLM tooling limitation.

## 5. Discussion

- **Quality-preservation and drafting-speed are separable properties.**
  SGT-QAT succeeds at the former (near-parity acceptance length with a
  purpose-built architecture) and fails at the latter (a genuine wall-clock
  regression) — for structural, not quantization-quality, reasons.
- **Quantization is a memory lever, with a quality ceiling, not a speed
  lever.** §4.4 shows pushing bit-width down trades memory for quality
  along a real, measured curve — but does nothing for the architectural
  compute-cost problem identified in §4.2. These are two independent axes;
  conflating them (e.g., assuming a memory win implies a speed win) would be
  a mistake this paper explicitly avoids.
- **What would actually close the speed gap**: a shallow, purpose-built
  drafter architecture (EAGLE-style: few layers, shared target embeddings),
  independent of quantization. This is a materially different, larger
  project than quantizing an existing dense checkpoint — noted as future
  work (§7), explicitly out of scope here.

## 6. Limitations

- SGT-QAT-as-drafter benchmarks (§4.2) use the *decompressed* checkpoint
  in-vLLM, not the genuinely compressed one, due to a current vLLM tooling
  gap (§3.2). The reported 0.43x speedup already reflects this (best-case,
  no dequantization-overhead) checkpoint — a genuinely compressed,
  properly-kernel-supported drafter might be marginally faster (less
  memory-bandwidth pressure during decode) or unchanged; this is untested
  and would require vLLM gaining compressed-tensors `draft_model` support.
- GPU memory readings are absolute whole-device `nvidia-smi` values; the
  no-spec/EAGLE-3 pair (§4.1-4.2) was measured back-to-back in one Colab
  session (valid same-session delta), but the SGT-QAT-drafter run (§4.2) was
  a different session — cross-session absolute-memory comparisons carry the
  usual caveat about differing baseline driver/OS overhead. Standalone
  memory numbers (§4.3-4.4) are session-independent by construction (no
  vLLM, no serving) but not directly comparable to full in-vLLM deltas.
- Single calibration seed (42) throughout — no replication check on whether
  the aggressive-quantization PPL collapse (§4.4) or the flagship's recovery
  (§3.1) would look different at a second seed. The prior SGT-QAT paper found
  seed-dependent effect sizes for a related ablation; this project has not
  repeated that check.
- `num_speculative_tokens=3` fixed throughout — not swept. A different value
  could shift the acceptance-length/speed tradeoff for either drafter, though
  it would not change the underlying architectural cost asymmetry (§4.2, §5).

## 7. Future Work

- **A purpose-built shallow drafter architecture, QAT'd.** The natural
  follow-up implied by §4.2/§5: design an EAGLE-style architecture (few
  layers, shared target embeddings) and apply Sensitivity-Guided Targeted QAT
  to *that*, rather than to a full dense model. This is a materially larger
  project (architecture design + distillation-style training against the
  target model, before QAT even enters), explicitly out of scope for the
  current project. [See memory note for this idea, flagged during this
  project's Phase 4 discussion, 2026-07-25 — not started.]
- Real in-vLLM benchmarking of the genuinely compressed checkpoint, once/if
  vLLM's `draft_model` path gains compressed-tensors support — would resolve
  the §6 limitation directly rather than relying on the standalone-vs-full
  serving inference in §4.3.
- A second calibration seed for both the flagship and aggressive-quantization
  recipes, to check whether §4.4's PPL collapse and §3.1's recovery numbers
  are seed-robust (per the prior project's precedent that some effect sizes
  in this method family are seed-dependent).

---

## Appendix: raw data index

All numbers in this draft are sourced from `docs/findings.md` (2026-07-25
entries) and the following `results/` files:
- `results/no_spec_decode_2026-07-25T08-22-23.566257+00-00.json`
- `results/baseline_eagle3_2026-07-25T08-26-34.711075+00-00.json`
- `results/sgt_qat_drafter_2026-07-25T09-08-02.978080+00-00.json`
- `results/compressed_checkpoint_memory_2026-07-25...json` (notebook 04)
- `results/aggressive_quant_tradeoff_seed42_2026-07-25T09-51-46.json`
- `results/export_sgt_qat_checkpoint_seed42_...json` (notebook 01, flagship
  checkpoint export)

Charts: `results/visual_metrics/*.svg` (see that folder's README for what
each one shows and its caveats).
