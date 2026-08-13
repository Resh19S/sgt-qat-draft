"""Generate poster/paper-ready SVG charts from the real numbers in docs/findings.md.

No external dependencies (no matplotlib) -- this machine doesn't have pip
available locally, and hand-built SVG is scalable/print-quality anyway, which
is what a poster needs. Every number here is transcribed directly from
docs/findings.md (2026-07-25 entries) / results/*.json -- do not hand-edit the
generated SVGs, edit this script and re-run it so the source of truth stays
findings.md, not a drifted image.

Run: python3 generate_charts.py   (from this directory, or any cwd -- writes
next to this script regardless of cwd)
"""

from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---- shared style ----------------------------------------------------------

FONT = "Helvetica, Arial, sans-serif"
COLOR_NO_SPEC = "#9aa0a6"
COLOR_EAGLE3 = "#4285f4"
COLOR_SGT_QAT = "#ea4335"
COLOR_SGT_QAT_AGGR = "#fbbc04"
COLOR_TEXT = "#202124"
COLOR_GRID = "#dadce0"
COLOR_BG = "#ffffff"


def _svg_header(width, height, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">\n'
        f'<rect width="{width}" height="{height}" fill="{COLOR_BG}"/>\n'
        f'<title>{title}</title>\n'
    )


def _svg_footer():
    return "</svg>\n"


def _fmt_tick(val):
    if val == 0:
        return "0"
    if val >= 100:
        return f"{val:,.0f}"
    if val >= 10:
        return f"{val:.0f}"
    if val >= 1:
        return f"{val:.1f}".rstrip("0").rstrip(".")
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _text(x, y, s, size=14, weight="normal", anchor="start", color=COLOR_TEXT):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{color}">{s}</text>\n'
    )


def bar_chart(
    filename,
    title,
    subtitle,
    bars,  # list of (label, value, color, value_label)
    y_max,
    y_label,
    source_note,
    width=760,
    height=520,
    log_scale=False,
):
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 100, 110
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n = len(bars)
    gap = plot_w / n
    bar_w = gap * 0.5

    svg = [_svg_header(width, height, title)]
    svg.append(_text(width / 2, 34, title, size=20, weight="bold", anchor="middle"))
    svg.append(_text(width / 2, 56, subtitle, size=13, anchor="middle", color="#5f6368"))

    def scale(v):
        if log_scale:
            import math

            lo = 1.0
            hi = y_max
            v = max(v, lo)
            frac = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            return plot_h * frac
        return plot_h * (v / y_max)

    # gridlines + y-axis ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        y = margin_top + plot_h - frac * plot_h
        if log_scale:
            import math

            val = 10 ** (math.log10(1.0) + frac * (math.log10(y_max) - math.log10(1.0)))
        else:
            val = frac * y_max
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{COLOR_GRID}" stroke-width="1"/>\n'
        )
        svg.append(_text(margin_left - 12, y + 4, _fmt_tick(val), size=11, anchor="end", color="#5f6368"))

    svg.append(
        _text(24, margin_top + plot_h / 2, y_label, size=12, color="#5f6368", anchor="middle")
        .replace("<text", f'<text transform="rotate(-90 24 {margin_top + plot_h / 2})"')
    )

    for i, (label, value, color, value_label) in enumerate(bars):
        bar_h = scale(value)
        x = margin_left + i * gap + (gap - bar_w) / 2
        y = margin_top + plot_h - bar_h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4"/>\n')
        svg.append(_text(x + bar_w / 2, y - 10, value_label, size=15, weight="bold", anchor="middle"))
        # respect explicit "\n" line breaks in the label; otherwise leave on one line
        lines = label.split("\n")
        for li, line in enumerate(lines):
            svg.append(_text(x + bar_w / 2, margin_top + plot_h + 22 + li * 18, line, size=13, anchor="middle"))

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" '
        f'y2="{margin_top + plot_h}" stroke="{COLOR_TEXT}" stroke-width="1.5"/>\n'
    )
    svg.append(_text(width / 2, height - 14, source_note, size=10, anchor="middle", color="#80868b"))
    svg.append(_svg_footer())

    (OUT_DIR / filename).write_text("".join(svg))
    print(f"wrote {filename}")


def grouped_bar_chart(
    filename,
    title,
    subtitle,
    groups,  # list of (group_label, [(series_label, value, color), ...])
    y_max,
    y_label,
    source_note,
    width=760,
    height=520,
):
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 110, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n_groups = len(groups)
    group_w = plot_w / n_groups
    n_series = len(groups[0][1])
    bar_w = (group_w * 0.7) / n_series

    svg = [_svg_header(width, height, title)]
    svg.append(_text(width / 2, 34, title, size=20, weight="bold", anchor="middle"))
    svg.append(_text(width / 2, 56, subtitle, size=13, anchor="middle", color="#5f6368"))

    def scale(v):
        return plot_h * (v / y_max)

    n_ticks = 5
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        y = margin_top + plot_h - frac * plot_h
        val = frac * y_max
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{COLOR_GRID}" stroke-width="1"/>\n'
        )
        svg.append(_text(margin_left - 12, y + 4, _fmt_tick(val), size=11, anchor="end", color="#5f6368"))

    for gi, (group_label, series) in enumerate(groups):
        group_x0 = margin_left + gi * group_w + group_w * 0.15
        for si, (series_label, value, color) in enumerate(series):
            bar_h = scale(value)
            x = group_x0 + si * bar_w
            y = margin_top + plot_h - bar_h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.85:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>\n')
            svg.append(_text(x + bar_w * 0.42, y - 8, f"{value:g}", size=12, weight="bold", anchor="middle"))
        gx = margin_left + gi * group_w + group_w / 2
        for li, line in enumerate(group_label.split("\n")):
            svg.append(_text(gx, margin_top + plot_h + 24 + li * 18, line, size=13, anchor="middle"))

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" '
        f'y2="{margin_top + plot_h}" stroke="{COLOR_TEXT}" stroke-width="1.5"/>\n'
    )

    # legend
    legend_x = margin_left
    legend_y = height - 50
    for si, (series_label, _, color) in enumerate(groups[0][1]):
        lx = legend_x + si * 190
        svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" fill="{color}" rx="2"/>\n')
        svg.append(_text(lx + 20, legend_y, series_label, size=12))

    svg.append(_text(width / 2, height - 14, source_note, size=10, anchor="middle", color="#80868b"))
    svg.append(_svg_footer())

    (OUT_DIR / filename).write_text("".join(svg))
    print(f"wrote {filename}")


def line_chart_positions(
    filename,
    title,
    subtitle,
    series,  # list of (label, [y0, y1, y2], color)
    source_note,
    width=760,
    height=480,
):
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 110, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n_points = 3
    x_positions = [margin_left + plot_w * (i / (n_points - 1)) for i in range(n_points)]
    y_max = 1.0

    svg = [_svg_header(width, height, title)]
    svg.append(_text(width / 2, 34, title, size=20, weight="bold", anchor="middle"))
    svg.append(_text(width / 2, 56, subtitle, size=13, anchor="middle", color="#5f6368"))

    n_ticks = 5
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        y = margin_top + plot_h - frac * plot_h
        val = frac * y_max
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{COLOR_GRID}" stroke-width="1"/>\n'
        )
        svg.append(_text(margin_left - 12, y + 4, f"{val:.0%}", size=11, anchor="end", color="#5f6368"))

    for i in range(n_points):
        svg.append(_text(x_positions[i], margin_top + plot_h + 26, f"position {i+1}", size=13, anchor="middle"))

    for si, (label, values, color) in enumerate(series):
        pts = " ".join(f"{x_positions[i]:.1f},{margin_top + plot_h - plot_h * values[i]:.1f}" for i in range(n_points))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>\n')
        # alternate label offset per series so close values don't overlap
        dy = -14 if si == 0 else 20
        for i in range(n_points):
            cx, cy = x_positions[i], margin_top + plot_h - plot_h * values[i]
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"/>\n')
            svg.append(_text(cx, cy + dy, f"{values[i]:.0%}", size=11, weight="bold", anchor="middle", color=color))

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" '
        f'y2="{margin_top + plot_h}" stroke="{COLOR_TEXT}" stroke-width="1.5"/>\n'
    )

    legend_y = height - 40
    for si, (label, _, color) in enumerate(series):
        lx = margin_left + si * 220
        svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" fill="{color}" rx="2"/>\n')
        svg.append(_text(lx + 20, legend_y, label, size=12))

    svg.append(_text(width / 2, height - 12, source_note, size=10, anchor="middle", color="#80868b"))
    svg.append(_svg_footer())

    (OUT_DIR / filename).write_text("".join(svg))
    print(f"wrote {filename}")


# ---- 1. Speedup vs no-spec-decode ------------------------------------------

bar_chart(
    "01_speedup.svg",
    "Wall-Clock Speedup vs. No Speculative Decoding",
    "Qwen3-8B target, real mt-bench prompts (80 prompts, 256 max tokens, max_model_len=4096)",
    bars=[
        ("No spec-decode", 1.00, COLOR_NO_SPEC, "1.00x"),
        ("EAGLE-3", 2.16, COLOR_EAGLE3, "2.16x"),
        ("SGT-QAT drafter", 0.43, COLOR_SGT_QAT, "0.43x"),
    ],
    y_max=2.5,
    y_label="Speedup (x)",
    source_note="Source: docs/findings.md 2026-07-25 \"Real-prompt SGT-QAT drafter run\" -- results/*2026-07-25T0[89]*.json",
)

# ---- 2. Raw throughput ------------------------------------------------------

bar_chart(
    "02_throughput_tokens_per_sec.svg",
    "Throughput: Tokens per Second",
    "Same real-prompt benchmark as above",
    bars=[
        ("No spec-decode", 76.6, COLOR_NO_SPEC, "76.6"),
        ("EAGLE-3", 165.4, COLOR_EAGLE3, "165.4"),
        ("SGT-QAT drafter", 32.9, COLOR_SGT_QAT, "32.9"),
    ],
    y_max=180,
    y_label="tokens/sec",
    source_note="Source: docs/findings.md 2026-07-25 \"Real-prompt SGT-QAT drafter run\" -- results/*2026-07-25T0[89]*.json",
)

# ---- 3. Mean acceptance length ---------------------------------------------

bar_chart(
    "03_mean_acceptance_length.svg",
    "Mean Acceptance Length (Draft Quality)",
    "Real prompts -- near-parity between drafters despite the huge speed gap above",
    bars=[
        ("EAGLE-3", 2.474, COLOR_EAGLE3, "2.474"),
        ("SGT-QAT drafter", 2.443, COLOR_SGT_QAT, "2.443"),
    ],
    y_max=3.0,
    y_label="tokens accepted per draft (of 3 proposed)",
    source_note="Source: docs/findings.md 2026-07-25 -- num_speculative_tokens=3 for both",
)

# ---- 4. Acceptance rate by speculative position ----------------------------

line_chart_positions(
    "04_acceptance_rate_by_position.svg",
    "Per-Position Acceptance Rate",
    "EAGLE-3 vs. SGT-QAT drafter, 3 speculative token positions",
    series=[
        ("EAGLE-3", [0.7142166928372992, 0.46466964609252326, 0.2955670974755405], COLOR_EAGLE3),
        ("SGT-QAT drafter", [0.6681373718101598, 0.45647507751013594, 0.31886477462437396], COLOR_SGT_QAT),
    ],
    source_note="Source: results/baseline_eagle3_2026-07-25T08-26-34...json, results/sgt_qat_drafter_2026-07-25T09-08-02...json",
)

# ---- 5. Memory comparison ---------------------------------------------------

bar_chart(
    "05_memory_comparison.svg",
    "Drafter Memory Footprint",
    "NOT apples-to-apples -- mixes standalone weight-only loads with full in-vLLM serving deltas (see caveat below)",
    bars=[
        ("EAGLE-3\n(full in-vLLM\ndelta)", 1.047, COLOR_EAGLE3, "1.047 GiB"),
        ("SGT-QAT flagship\n(standalone,\ncompressed)", 1.629, COLOR_SGT_QAT, "1.629 GiB"),
        ("SGT-QAT aggressive\n(standalone,\ncompressed)", 0.646, COLOR_SGT_QAT_AGGR, "0.646 GiB"),
    ],
    y_max=2.0,
    y_label="GiB",
    source_note="Caveat: EAGLE-3 bar = full vLLM serving delta (weights+KV cache); SGT-QAT bars = standalone weight-only load. See findings.md 2026-07-25.",
    height=560,
)

# ---- 6. Quality vs. bit-width tradeoff --------------------------------------

grouped_bar_chart(
    "06_quality_vs_bitwidth_tradeoff.svg",
    "Quantization Aggressiveness vs. Quality (WikiText-2 PPL)",
    "Flagship (3.156 bits/weight) vs. aggressive (2.158 bits/weight) -- lower PPL is better",
    groups=[
        ("Flagship\n(W4/W3, 3.156 bpw)", [
            ("Stage 1 (GPTQ only)", 22.37, "#c6dafc"),
            ("Stage 1+2 (+QAT)", 15.91, COLOR_EAGLE3),
        ]),
        ("Aggressive\n(W3/W2, 2.158 bpw)", [
            ("Stage 1 (GPTQ only)", 454.31, "#fdd7d3"),
            ("Stage 1+2 (+QAT)", 188.66, COLOR_SGT_QAT),
        ]),
    ],
    y_max=500,
    y_label="WikiText-2 perplexity (lower = better)",
    source_note="Source: docs/findings.md 2026-07-25 \"Aggressive quantization tradeoff\" -- results/aggressive_quant_tradeoff_seed42_*.json",
    width=780,
)

print("\nAll charts written to", OUT_DIR)
