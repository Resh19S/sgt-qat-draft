# Visual metrics

Poster/paper-ready charts generated from the real numbers in `docs/findings.md`
(2026-07-25 entries). SVG (scalable, print-quality) — no PNG conversion checked
in, since a poster tool should import the SVG directly for best print quality.
If you need PNG/PDF for a specific poster template, export from the SVG at
whatever DPI the template needs rather than asking for a raster file here.

**Source of truth is `docs/findings.md`, not these images.** If a number here
ever looks off, `generate_charts.py` is the thing to fix (it's a small,
dependency-free Python script — no matplotlib install needed) — never hand-edit
the `.svg` files directly, or they'll drift from findings.md.

Regenerate after any findings.md update:
```
python3 generate_charts.py
```

## Charts

1. **`01_speedup.svg`** — wall-clock speedup vs. no spec-decode, all three
   conditions (real prompts).
2. **`02_throughput_tokens_per_sec.svg`** — raw tokens/sec, same three
   conditions.
3. **`03_mean_acceptance_length.svg`** — EAGLE-3 vs. SGT-QAT drafter mean
   acceptance length (draft quality) — the "near-parity despite the speed gap"
   chart.
4. **`04_acceptance_rate_by_position.svg`** — per-speculative-position
   acceptance rate, both drafters.
5. **`05_memory_comparison.svg`** — EAGLE-3's full in-vLLM memory delta vs.
   SGT-QAT's standalone compressed-checkpoint footprint (flagship and
   aggressive quantization). **Not apples-to-apples** — the chart says so in
   its own subtitle/caption, keep that caveat if reusing this on a poster.
6. **`06_quality_vs_bitwidth_tradeoff.svg`** — WikiText-2 PPL, flagship
   (3.156 bits/weight) vs. aggressive (2.158 bits/weight) quantization, before
   and after QAT fine-tuning. This is the "you can win on memory, but quality
   collapses" chart.

## Caveats these charts inherit from findings.md

- Chart 5 mixes a full in-vLLM serving measurement (EAGLE-3) with standalone
  weight-only loads (SGT-QAT) — genuinely different measurement contexts, not
  fixed by prettier presentation. Keep the caveat text if this goes on a
  poster.
- All speed/acceptance numbers are from the real-prompt (mt-bench) runs, not
  the earlier (now-superseded) placeholder-text runs — see findings.md if you
  need the placeholder-prompt numbers for historical comparison.
