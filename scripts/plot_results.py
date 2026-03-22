#!/usr/bin/env python3
"""Generate thesis-quality figures from eBPF design-space exploration results.

Usage:
    python3 scripts/plot_results.py [--results results/experiment_log.jsonl] [--output-dir results]
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    print("ERROR: matplotlib is not installed.")
    print("Install it with:  pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed.")
    print("Install it with:  pip install numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Global style – academic / thesis quality
# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Okabe-Ito)
CB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#F0E442", "#56B4E9", "#E69F00", "#000000"]
CB_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]

def _apply_style():
    """Set matplotlib rcParams for a clean, academic look."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "figure.figsize": (6, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def filter_data(data, **kwargs):
    """Return subset of data matching all key=value filters."""
    out = data
    for k, v in kwargs.items():
        out = [d for d in out if d.get(k) == v]
    return out


def sorted_unique(data, key):
    return sorted(set(d[key] for d in data))

# ---------------------------------------------------------------------------
# Figure 1: Scale bits vs Int Accuracy (grouped by hidden_size, layers=2)
# ---------------------------------------------------------------------------
def fig_scale_bits_accuracy(data, outdir):
    sub = filter_data(data, num_layers=2)
    hidden_sizes = sorted_unique(sub, "hidden_size")

    fig, ax = plt.subplots()
    for i, h in enumerate(hidden_sizes):
        pts = sorted(filter_data(sub, hidden_size=h), key=lambda d: d["scale_bits"])
        xs = [d["scale_bits"] for d in pts]
        ys = [d["int_accuracy"] for d in pts]
        ax.plot(xs, ys, marker=CB_MARKERS[i % len(CB_MARKERS)],
                color=CB_COLORS[i % len(CB_COLORS)], label=f"h={h}")

    # Annotate the b=16 threshold
    ax.axvline(x=16, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(16.3, 0.75, "b=16", fontsize=10, color="grey", va="center")

    ax.set_xlabel("Scale Bits (b)")
    ax.set_ylabel("Integer Accuracy")
    ax.set_title("Scale Bits vs Integer Accuracy (layers=2)")
    ax.set_xticks([8, 12, 16, 20])
    ax.legend(title="Hidden Size")
    fig.tight_layout()

    _save(fig, outdir, "fig_scale_bits_accuracy")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Hidden size vs Accuracy + Memory (layers=2, scale_bits=16)
# ---------------------------------------------------------------------------
def fig_hidden_size_tradeoff(data, outdir):
    sub = sorted(filter_data(data, num_layers=2, scale_bits=16),
                 key=lambda d: d["hidden_size"])
    xs = [d["hidden_size"] for d in sub]
    acc = [d["int_accuracy"] for d in sub]
    mem = [d["model_memory_bytes"] for d in sub]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Hidden Size")

    # Left axis – accuracy
    color_acc = CB_COLORS[0]
    ax1.set_ylabel("Integer Accuracy", color=color_acc)
    ln1 = ax1.plot(xs, acc, marker="o", color=color_acc, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(xs)

    # Right axis – memory
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)   # show right spine for dual axis
    color_mem = CB_COLORS[1]
    ax2.set_ylabel("Model Memory (bytes)", color=color_mem)
    ln2 = ax2.plot(xs, mem, marker="s", color=color_mem, linestyle="--",
                   label="Memory")
    ax2.tick_params(axis="y", labelcolor=color_mem)

    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    ax1.set_title("Hidden Size: Accuracy vs Memory (layers=2, b=16)")
    fig.tight_layout()

    _save(fig, outdir, "fig_hidden_size_tradeoff")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Pareto Front (accuracy vs instruction count, layers=2)
# ---------------------------------------------------------------------------
def _pareto_mask(costs):
    """Return boolean mask for Pareto-optimal rows.

    Each row of *costs* is [instruction_count, -int_accuracy] – we minimise
    both, so lower instruction_count + higher accuracy is better.
    """
    is_pareto = np.ones(len(costs), dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # A point dominates another if it is <= on all and < on at least one
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1) | \
                                    np.all(costs[is_pareto] == c, axis=1)
            is_pareto[i] = True
    return is_pareto


def fig_pareto_front(data, outdir):
    sub = [d for d in data
           if d["num_layers"] == 2 and d["instruction_count"] is not None]
    xs = np.array([d["instruction_count"] for d in sub])
    ys = np.array([d["int_accuracy"] for d in sub])
    labels = [f"(h={d['hidden_size']}, b={d['scale_bits']})" for d in sub]

    # Pareto: minimise instruction_count, maximise accuracy
    costs = np.column_stack([xs, -ys])
    mask = _pareto_mask(costs)

    fig, ax = plt.subplots()
    # All points
    ax.scatter(xs[~mask], ys[~mask], color=CB_COLORS[0], alpha=0.45,
               s=50, zorder=2, label="Dominated")
    # Pareto points
    ax.scatter(xs[mask], ys[mask], color=CB_COLORS[1], edgecolors="black",
               s=90, zorder=3, linewidths=0.8, label="Pareto-optimal")

    # Connect Pareto front with a line (sorted by instruction_count)
    pareto_idx = np.where(mask)[0]
    order = pareto_idx[np.argsort(xs[pareto_idx])]
    ax.plot(xs[order], ys[order], color=CB_COLORS[1], linewidth=1.2,
            linestyle="--", alpha=0.6, zorder=2)

    # Labels
    for idx in range(len(sub)):
        offset = (5, 5) if not mask[idx] else (5, -12)
        ax.annotate(labels[idx], (xs[idx], ys[idx]),
                    textcoords="offset points", xytext=offset,
                    fontsize=7.5, color="black", alpha=0.85)

    ax.set_xlabel("Instruction Count")
    ax.set_ylabel("Integer Accuracy")
    ax.set_title("Pareto Front: Accuracy vs Instruction Count (layers=2)")
    ax.legend()
    fig.tight_layout()

    _save(fig, outdir, "fig_pareto_front")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Quantization Loss Heatmap (hidden_size vs scale_bits, layers=2)
# ---------------------------------------------------------------------------
def fig_quantization_heatmap(data, outdir):
    sub = filter_data(data, num_layers=2)
    h_vals = sorted_unique(sub, "hidden_size")
    b_vals = sorted_unique(sub, "scale_bits")

    matrix = np.full((len(h_vals), len(b_vals)), np.nan)
    for d in sub:
        r = h_vals.index(d["hidden_size"])
        c = b_vals.index(d["scale_bits"])
        matrix[r, c] = d["accuracy_delta"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   origin="lower", interpolation="nearest")

    # Annotate cells
    for r in range(len(h_vals)):
        for c in range(len(b_vals)):
            val = matrix[r, c]
            text_color = "white" if val > 0.15 else "black"
            ax.text(c, r, f"{val:.4f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(b_vals)))
    ax.set_xticklabels([str(b) for b in b_vals])
    ax.set_yticks(range(len(h_vals)))
    ax.set_yticklabels([str(h) for h in h_vals])
    ax.set_xlabel("Scale Bits (b)")
    ax.set_ylabel("Hidden Size")
    ax.set_title("Quantization Accuracy Loss (layers=2)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy Delta (float − int)")
    fig.tight_layout()

    _save(fig, outdir, "fig_quantization_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Layer count effect (scale_bits=16)
# ---------------------------------------------------------------------------
def fig_layer_count(data, outdir):
    sub = filter_data(data, scale_bits=16)
    hidden_sizes = sorted_unique(sub, "hidden_size")

    fig, ax = plt.subplots()
    for i, h in enumerate(hidden_sizes):
        pts = sorted(filter_data(sub, hidden_size=h), key=lambda d: d["num_layers"])
        xs = [d["num_layers"] for d in pts]
        ys = [d["int_accuracy"] for d in pts]
        ax.plot(xs, ys, marker=CB_MARKERS[i % len(CB_MARKERS)],
                color=CB_COLORS[i % len(CB_COLORS)], label=f"h={h}")

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Integer Accuracy")
    ax.set_title("Layer Count Effect on Accuracy (b=16)")
    ax.set_xticks([1, 2, 3])
    ax.legend(title="Hidden Size")
    fig.tight_layout()

    _save(fig, outdir, "fig_layer_count")
    return fig


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------
def fig_summary(data, outdir):
    """2×3 subplot layout combining all five analyses."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ---- Panel (0,0): Scale bits vs accuracy (= Fig 1) ----
    ax = axes[0, 0]
    sub = filter_data(data, num_layers=2)
    for i, h in enumerate(sorted_unique(sub, "hidden_size")):
        pts = sorted(filter_data(sub, hidden_size=h), key=lambda d: d["scale_bits"])
        ax.plot([d["scale_bits"] for d in pts],
                [d["int_accuracy"] for d in pts],
                marker=CB_MARKERS[i], color=CB_COLORS[i], label=f"h={h}")
    ax.axvline(16, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Scale Bits")
    ax.set_ylabel("Int Accuracy")
    ax.set_title("(a) Scale Bits vs Accuracy")
    ax.set_xticks([8, 12, 16, 20])
    ax.legend(fontsize=8, title="Hidden", title_fontsize=8)

    # ---- Panel (0,1): Hidden size tradeoff (= Fig 2) ----
    ax1 = axes[0, 1]
    sub2 = sorted(filter_data(data, num_layers=2, scale_bits=16),
                  key=lambda d: d["hidden_size"])
    xs = [d["hidden_size"] for d in sub2]
    ln1 = ax1.plot(xs, [d["int_accuracy"] for d in sub2],
                   marker="o", color=CB_COLORS[0], label="Accuracy")
    ax1.set_xlabel("Hidden Size")
    ax1.set_ylabel("Int Accuracy", color=CB_COLORS[0])
    ax1.tick_params(axis="y", labelcolor=CB_COLORS[0])
    ax1.set_xticks(xs)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ln2 = ax2.plot(xs, [d["model_memory_bytes"] for d in sub2],
                   marker="s", color=CB_COLORS[1], ls="--", label="Memory")
    ax2.set_ylabel("Memory (B)", color=CB_COLORS[1])
    ax2.tick_params(axis="y", labelcolor=CB_COLORS[1])
    lns = ln1 + ln2
    ax1.legend(lns, [l.get_label() for l in lns], fontsize=8)
    ax1.set_title("(b) Accuracy vs Memory")

    # ---- Panel (0,2): Pareto front (= Fig 3) ----
    ax = axes[0, 2]
    sub3 = [d for d in data
            if d["num_layers"] == 2 and d["instruction_count"] is not None]
    pxs = np.array([d["instruction_count"] for d in sub3])
    pys = np.array([d["int_accuracy"] for d in sub3])
    costs = np.column_stack([pxs, -pys])
    pmask = _pareto_mask(costs)
    ax.scatter(pxs[~pmask], pys[~pmask], color=CB_COLORS[0], alpha=0.45, s=40)
    ax.scatter(pxs[pmask], pys[pmask], color=CB_COLORS[1],
               edgecolors="black", s=70, linewidths=0.7)
    for idx in range(len(sub3)):
        ax.annotate(f"({sub3[idx]['hidden_size']},{sub3[idx]['scale_bits']})",
                    (pxs[idx], pys[idx]), textcoords="offset points",
                    xytext=(4, 4), fontsize=6.5, alpha=0.8)
    ax.set_xlabel("Instructions")
    ax.set_ylabel("Int Accuracy")
    ax.set_title("(c) Pareto Front")

    # ---- Panel (1,0): Heatmap (= Fig 4) ----
    ax = axes[1, 0]
    sub4 = filter_data(data, num_layers=2)
    h_vals = sorted_unique(sub4, "hidden_size")
    b_vals = sorted_unique(sub4, "scale_bits")
    mat = np.full((len(h_vals), len(b_vals)), np.nan)
    for d in sub4:
        mat[h_vals.index(d["hidden_size"]), b_vals.index(d["scale_bits"])] = d["accuracy_delta"]
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", origin="lower")
    for r in range(len(h_vals)):
        for c in range(len(b_vals)):
            tc = "white" if mat[r, c] > 0.15 else "black"
            ax.text(c, r, f"{mat[r,c]:.3f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")
    ax.set_xticks(range(len(b_vals)))
    ax.set_xticklabels(b_vals)
    ax.set_yticks(range(len(h_vals)))
    ax.set_yticklabels(h_vals)
    ax.set_xlabel("Scale Bits")
    ax.set_ylabel("Hidden Size")
    ax.set_title("(d) Quantization Loss")

    # ---- Panel (1,1): Layer count (= Fig 5) ----
    ax = axes[1, 1]
    sub5 = filter_data(data, scale_bits=16)
    for i, h in enumerate(sorted_unique(sub5, "hidden_size")):
        pts = sorted(filter_data(sub5, hidden_size=h), key=lambda d: d["num_layers"])
        ax.plot([d["num_layers"] for d in pts],
                [d["int_accuracy"] for d in pts],
                marker=CB_MARKERS[i], color=CB_COLORS[i], label=f"h={h}")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Int Accuracy")
    ax.set_title("(e) Layer Count Effect")
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8, title="Hidden", title_fontsize=8)

    # ---- Panel (1,2): blank – turn off ----
    axes[1, 2].axis("off")

    fig.suptitle("Design Space Exploration Summary", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _save(fig, outdir, "fig_exploration_summary", formats=["png"])
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, outdir, stem, formats=None):
    if formats is None:
        formats = ["png", "pdf"]
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(outdir, f"{stem}.{fmt}")
        try:
            fig.savefig(path)
            print(f"  ✓ {path}")
        except Exception as e:
            print(f"  ✗ {path}  (skipped: {e})")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality figures from exploration results.")
    parser.add_argument("--results", default="results/experiment_log.jsonl",
                        help="Path to JSONL results file.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output figures.")
    args = parser.parse_args()

    if not os.path.isfile(args.results):
        print(f"ERROR: results file not found: {args.results}")
        sys.exit(1)

    _apply_style()
    data = load_data(args.results)
    print(f"Loaded {len(data)} experiment records from {args.results}\n")

    print("Generating figures...")
    fig_scale_bits_accuracy(data, args.output_dir)
    fig_hidden_size_tradeoff(data, args.output_dir)
    fig_pareto_front(data, args.output_dir)
    fig_quantization_heatmap(data, args.output_dir)
    fig_layer_count(data, args.output_dir)
    fig_summary(data, args.output_dir)

    print(f"\nDone – all figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
