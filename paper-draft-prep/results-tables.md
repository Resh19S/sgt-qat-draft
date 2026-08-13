# Results Tables — SGT-QAT Drafter for Speculative Decoding

Consolidated, paper-ready tables. Every number here traces to a JSON file in
`raw-results/` and a dated entry in `findings.md`. All speed/acceptance numbers
are from the **real-prompt (mt-bench) runs** — earlier placeholder-prompt runs
are superseded and not included here (they remain in the main repo's `results/`
for the record).

**Common benchmark configuration** (Tables 1–3): target model Qwen3-8B (full
precision), drafter Qwen3-1.7B (SGT-QAT), mt-bench prompts (80 prompts,
256 max tokens), `num_speculative_tokens=3`, `max_model_len=4096`, greedy
decoding, low-concurrency (sequential single-request) methodology. GPU memory
is a whole-device `nvidia-smi` reading. Hardware: Colab A100-40GB.

---

## Table 1. Three-way speculative-decoding comparison (real prompts)

| Condition | tok/s | Speedup | Mean acceptance length | GPU memory |
|---|---|---|---|---|
| No spec-decode | 76.6 | 1.00× | — | 36.75 GiB |
| EAGLE-3 | 165.4 | **2.16×** | 2.474 | 37.80 GiB |
| SGT-QAT drafter | 32.9 | **0.43×** | 2.443 | 37.27 GiB |

**Headline**: acceptance quality is near-parity (2.443 vs. 2.474), but the
SGT-QAT drafter is a wall-clock *regression* (0.43×) — slower than no
speculative decoding at all. The per-step cost of a full 28-layer dense 1.7B
drafter outweighs the tokens its (competent) acceptance rate saves. This is
architectural, not a quantization-quality result.

Source: `raw-results/no_spec_decode_2026-07-25T08-22-23…json`,
`raw-results/baseline_eagle3_2026-07-25T08-26-34…json`,
`raw-results/sgt_qat_drafter_2026-07-25T09-08-02…json`.

**Caveats**: (1) the SGT-QAT drafter benchmarked is the *decompressed* (~3.4 GB
fp16) checkpoint, not the genuinely compressed one — vLLM's `draft_model` path
cannot load compressed-tensors packed checkpoints (see "Known limitations"
below). (2) GPU memory: no-spec/EAGLE-3 were measured back-to-back in one
session (valid delta between them); the SGT-QAT run was a different session, so
cross-session absolute-memory comparison carries the usual caveat.

---

## Table 2. Per-position acceptance rate (real prompts)

| Speculative position | EAGLE-3 | SGT-QAT drafter |
|---|---|---|
| 1 | 0.714 | 0.668 |
| 2 | 0.465 | 0.456 |
| 3 | 0.296 | 0.319 |

Both drafters degrade similarly with position; SGT-QAT is marginally lower at
positions 1–2 and marginally higher at position 3. Consistent with the
near-parity mean acceptance length in Table 1.

Source: same JSON files as Table 1.

---

## Table 3. Drafter memory footprint (NOT apples-to-apples)

| Drafter | Footprint | Measurement context |
|---|---|---|
| EAGLE-3 | 1.047 GiB | Full in-vLLM delta (weights + KV cache + serving overhead) |
| SGT-QAT flagship (W4/W3, compressed) | 1.629 GiB | Standalone weight-only load (no vLLM) |
| SGT-QAT aggressive (W3/W2, compressed) | 0.646 GiB | Standalone weight-only load (no vLLM) |

**Important**: the EAGLE-3 number is a full serving-context measurement; the
SGT-QAT numbers are standalone weight-only loads. These are different
measurement contexts and are not directly comparable — the table shows what
was measurable given vLLM's tooling limitation, not a clean head-to-head. Even
so: the flagship compressed SGT-QAT drafter (1.629 GiB, weights only) is
already larger than EAGLE-3's entire in-vLLM footprint (1.047 GiB). Pushing to
aggressive quantization (0.646 GiB) gets under EAGLE-3 — but at a catastrophic
quality cost (Table 4).

Source: `raw-results/compressed_checkpoint_memory_2026-07-25…json` (flagship
standalone, notebook 04), `raw-results/aggressive_quant_tradeoff_seed42…json`
(aggressive standalone, notebook 05), EAGLE-3 delta from Table 1.

---

## Table 4. Quantization aggressiveness vs. quality (WikiText-2 PPL)

| Recipe | Avg bits/weight | Stage 1 (GPTQ) PPL | Stage 1+2 (+QAT) PPL | Standalone memory |
|---|---|---|---|---|
| Flagship (W4 protected / W3 rest) | 3.156 | 22.37 | **15.91** | 1.629 GiB |
| Aggressive (W3 protected / W2 rest) | 2.158 | 454.31 | 188.66 | **0.646 GiB** |

Pushing bit-width down closes the memory gap against EAGLE-3 (0.646 < 1.047 GiB)
but perplexity collapses ~12× (15.91 → 188.66) even after QAT fine-tuning — a
memory win is achievable, but not without destroying the quality the drafter
approach depends on. QAT still helps at the aggressive tier (454.31 → 188.66)
but nowhere near enough. Model: Qwen3-1.7B, calibration seed 42, single seed.

Source: `raw-results/aggressive_quant_tradeoff_seed42_2026-07-25…json`
(notebook 05); flagship reference from `raw-results/export_sgt_qat_checkpoint_seed42…json`
(notebook 01).

---

## Table 5. SGT-QAT checkpoint export (flagship recipe)

| Property | Value |
|---|---|
| Model | Qwen3-1.7B |
| Recipe | Sensitivity-ranked mixed-precision GPTQ (W4 on ~15% most sensitive layers, W3 rest) + targeted QAT on still-W3 layers |
| Calibration seed | 42 |
| Avg bits/weight | 3.156 |
| Stage 1+2 WikiText-2 PPL | 15.91 |
| Compressed checkpoint size (disk) | 1184.8 MB |

Source: `raw-results/export_sgt_qat_checkpoint_seed42_2026-07-22…json`
(notebook 01).

---

## Known limitations (relevant to the memory story)

vLLM's `speculative_config method="draft_model"` path cannot load
compressed-tensors packed checkpoints as drafters — filed upstream as
[vllm-project/vllm#49893](https://github.com/vllm-project/vllm/issues/49893).
The originally-reported target-matching bug is fixed (PR #49900), but a
second, separate packing-format issue surfaced: vLLM expects **dense** W3
packing (`ceil(1024×3/32)=96` int32 columns, in compressed-tensors ≥0.17.0)
while our checkpoint uses the older whole-values-per-int32 layout (103
columns). W4 is unaffected (4-bit packs identically both ways). As of
2026-08-10 this is unresolved upstream. **Consequence for the paper**: the
in-vLLM SGT-QAT-drafter benchmark (Table 1) uses the *decompressed* checkpoint,
so its memory figure reflects a full-precision 1.7B drafter, not a compressed
deployment. The compressed footprint is only measurable standalone (Table 3),
which is why the memory comparison is stated as non-apples-to-apples rather
than as a clean result. See `methodology-context.md` and (in the main repo)
`docs/vllm-bug-report-draft.md` for the full trail.

---

## Figures

Poster/paper-ready SVG charts in `figures/` (regenerate via
`figures/generate_charts.py` if any number here changes — never hand-edit the
SVGs). See `figures/README.md` for what each shows and its caveats.

1. `01_speedup.svg` — Table 1 speedup column
2. `02_throughput_tokens_per_sec.svg` — Table 1 tok/s column
3. `03_mean_acceptance_length.svg` — Table 1 acceptance length (EAGLE-3 vs. SGT-QAT)
4. `04_acceptance_rate_by_position.svg` — Table 2
5. `05_memory_comparison.svg` — Table 3 (with the non-apples-to-apples caveat in the chart)
6. `06_quality_vs_bitwidth_tradeoff.svg` — Table 4
