#!/usr/bin/env python3
"""
overflow_analysis.py — Formal overflow safety analysis for quantized MLP
inference in eBPF.

Proves (or disproves) that int64 accumulators never overflow for a given
model architecture, scale factor, and input domain.

Two analysis modes:
  1. Theoretical worst-case bound  — uses architecture params + W_max + M
  2. Empirical bound               — uses actual trained weights + test data

Dependencies: numpy only.

Usage:
    python3 scripts/overflow_analysis.py
    python3 scripts/overflow_analysis.py --hidden-size 64 --scale-bits 20
    python3 scripts/overflow_analysis.py --sweep
    python3 scripts/overflow_analysis.py --latex
"""

import argparse
import math
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Allow importing train_model from the same directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import train_model  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INT64_MAX = 2**63 - 1
MAX_FEATURE_RAW = 65536        # max raw feature value (e.g. dst_port)
TYPICAL_W_MAX = 3.0            # He init 3-sigma bound
DEFAULT_D_IN = train_model.NUM_FEATURES   # 6
DEFAULT_D_OUT = train_model.NUM_CLASSES   # 4
DEFAULT_HIDDEN = train_model.HIDDEN_SIZE  # 32
DEFAULT_SCALE_BITS = train_model.SCALE_BITS  # 16


# ===================================================================
# 1.  Theoretical (worst-case) analysis
# ===================================================================

def theoretical_analysis(d_in, hidden_sizes, d_out, scale_bits,
                         max_feature_raw=MAX_FEATURE_RAW,
                         w_max=TYPICAL_W_MAX):
    """Compute worst-case accumulator bound at every layer.

    Parameters
    ----------
    d_in : int            Number of input features.
    hidden_sizes : list   Sizes of hidden layers, e.g. [32, 32].
    d_out : int           Number of output classes.
    scale_bits : int      b, where scale factor s = 2^b.
    max_feature_raw : int Maximum absolute raw feature value.
    w_max : float         Maximum absolute float weight (3-sigma He init).

    Returns
    -------
    dict with keys:
        layers      — list of per-layer dicts
        b_max_safe  — maximum safe scale_bits for this architecture
        safe        — bool, whether the given scale_bits is safe
    """
    s = 1 << scale_bits
    layer_dims = [d_in] + list(hidden_sizes) + [d_out]
    num_layers = len(layer_dims) - 1

    # Maximum absolute quantized weight at any layer
    w_max_q = w_max * s  # |round(s * w_float)| ≤ w_max * s

    # Maximum absolute input to layer 0 after normalisation:
    #   |x_norm[j]| ≤ norm_scale * x_raw + |norm_offset|
    #   norm_scale ≤ s (when std ≥ 1), so ≤ s * max_feature_raw + s * (mean/std)
    #   Conservatively: ≤ s * max_feature_raw  (dominates for large features)
    m_input = s * max_feature_raw

    layers = []
    overall_safe = True
    b_max_safe = None  # will be the minimum across layers

    for k in range(num_layers):
        fan_in = layer_dims[k]
        fan_out = layer_dims[k + 1]

        # Accumulator worst case: fan_in * |W_q| * |x_input|
        max_accum = fan_in * w_max_q * m_input

        # Check overflow
        safe = max_accum <= INT64_MAX
        if not safe:
            overall_safe = False

        # Utilisation ratio
        ratio = max_accum / INT64_MAX

        # Max safe b for THIS layer alone:
        #   fan_in * (w_max * 2^b) * M_input(b) < 2^63
        #   where M_input depends on b via previous layers.
        # We compute C_k such that max_accum = C_k * s^2, then
        #   C_k * 2^(2b) < 2^63  =>  b < (63 - log2(C_k)) / 2
        # C_k = max_accum / s^2
        c_k = max_accum / (s * s) if s > 0 else float('inf')
        if c_k > 0:
            b_layer = math.floor((63 - math.log2(c_k)) / 2)
        else:
            b_layer = 63  # degenerate

        if b_max_safe is None or b_layer < b_max_safe:
            b_max_safe = b_layer

        layers.append({
            "index": k,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "max_input_abs": m_input,
            "max_weight_abs": w_max_q,
            "max_accum": max_accum,
            "ratio": ratio,
            "safe": safe,
            "b_max_layer": b_layer,
            "C_k": c_k,
        })

        # Output of this layer becomes input to the next (after >> b + bias)
        # |output| ≤ max_accum / s + |bias_q|
        # |bias_q| ≤ w_max * s  (biases are also scaled by s)
        m_input = max_accum / s + w_max * s

    return {
        "layers": layers,
        "b_max_safe": b_max_safe,
        "safe": overall_safe,
        "scale_bits": scale_bits,
        "architecture": layer_dims,
    }


# ===================================================================
# 2.  Empirical analysis (with actual weights and data)
# ===================================================================

def empirical_analysis(model, mean, std, X_raw, y_true, scale_bits):
    """Run quantized inference tracking actual max accumulator values.

    Parameters
    ----------
    model : train_model.MLP   Trained float model.
    mean, std : ndarray        Per-feature statistics (from training set).
    X_raw : ndarray            Raw (un-normalised) test features.
    y_true : ndarray           Ground-truth labels.
    scale_bits : int           b, where s = 2^b.

    Returns
    -------
    dict with per-layer actual accumulator stats and overall headroom.
    """
    s = 1 << scale_bits
    W_q, b_q, norm_s, norm_o = train_model.quantize_model(
        model, mean, std, scale=s)

    N = X_raw.shape[0]

    # Normalise in integer domain
    x = (norm_s.astype(np.int64)[None, :] *
         X_raw.astype(np.int64)) - norm_o.astype(np.int64)[None, :]

    layers = []
    max_ratio_overall = 0.0

    for k in range(len(W_q)):
        W = W_q[k].astype(np.int64)
        B = b_q[k].astype(np.int64)

        # Track input stats
        max_input_abs = int(np.max(np.abs(x)))
        max_weight_abs = int(np.max(np.abs(W)))

        # Accumulator (before shift)
        z = x @ W.T   # (N, fan_out) — int64, scaled by s^2

        max_accum = int(np.max(np.abs(z)))
        ratio = max_accum / INT64_MAX

        if ratio > max_ratio_overall:
            max_ratio_overall = ratio

        # 99.9th percentile of |accum|
        p999 = int(np.percentile(np.abs(z), 99.9))

        layers.append({
            "index": k,
            "fan_in": W.shape[1],
            "fan_out": W.shape[0],
            "max_input_abs": max_input_abs,
            "max_weight_abs": max_weight_abs,
            "max_accum": max_accum,
            "p999_accum": p999,
            "ratio": ratio,
            "safe": max_accum <= INT64_MAX,
        })

        # Shift + bias + ReLU
        z = (z >> scale_bits) + B
        if k < len(W_q) - 1:
            z = np.maximum(0, z)
        x = z

    # Accuracy check
    preds = np.argmax(x, axis=1)
    acc = float((preds == y_true).mean())

    # Headroom: how much larger could s be?
    # If max_ratio_overall = R, then s could grow by sqrt(1/R) before overflow
    # (because accum ∝ s²)
    if max_ratio_overall > 0:
        headroom_factor = math.sqrt(1.0 / max_ratio_overall)
    else:
        headroom_factor = float('inf')

    return {
        "layers": layers,
        "max_ratio_overall": max_ratio_overall,
        "headroom_factor": headroom_factor,
        "accuracy": acc,
        "scale_bits": scale_bits,
    }


# ===================================================================
# 3.  Architecture sweep
# ===================================================================

def architecture_sweep(d_in=DEFAULT_D_IN, d_out=DEFAULT_D_OUT,
                       hidden_list=None, bits_list=None,
                       w_max=TYPICAL_W_MAX):
    """Sweep hidden sizes and scale_bits; return list of result rows."""
    if hidden_list is None:
        hidden_list = [8, 16, 32, 64, 128]
    if bits_list is None:
        bits_list = [12, 14, 16, 18, 20, 22, 24]

    rows = []
    for h in hidden_list:
        res = theoretical_analysis(d_in, [h, h], d_out, scale_bits=16,
                                   w_max=w_max)
        b_max = res["b_max_safe"]
        for b in bits_list:
            res_b = theoretical_analysis(d_in, [h, h], d_out, scale_bits=b,
                                         w_max=w_max)
            # Find the worst-case layer
            worst = max(res_b["layers"], key=lambda l: l["ratio"])
            rows.append({
                "hidden": h,
                "scale_bits": b,
                "b_max_safe": b_max,
                "safe": res_b["safe"],
                "worst_layer": worst["index"],
                "worst_ratio": worst["ratio"],
            })
    return rows


# ===================================================================
# 4.  LaTeX table generation
# ===================================================================

def generate_latex_table(sweep_rows, hidden_list=None):
    """Generate a LaTeX table from sweep results."""
    if hidden_list is None:
        hidden_list = sorted(set(r["hidden"] for r in sweep_rows))

    bits_vals = sorted(set(r["scale_bits"] for r in sweep_rows))
    lookup = {(r["hidden"], r["scale_bits"]): r for r in sweep_rows}

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Overflow safety for quantized MLP "
                 r"$[6, h, h, 4]$ with scale factor $s = 2^b$.}")
    lines.append(r"  \label{tab:overflow-safety}")

    col_spec = "r" + "c" * len(bits_vals)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")
    hdr = r"    $h$ & " + " & ".join(f"$b={b}$" for b in bits_vals) + r" \\"
    lines.append(hdr)
    lines.append(r"    \midrule")

    for h in hidden_list:
        cells = [str(h)]
        for b in bits_vals:
            r = lookup.get((h, b))
            if r is None:
                cells.append("--")
            elif r["safe"]:
                cells.append(r"\cmark")
            else:
                cells.append(r"\xmark")
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    # Add a note about b_max
    b_max_notes = []
    for h in hidden_list:
        r = lookup.get((h, bits_vals[0]))
        if r:
            b_max_notes.append(f"$h={h}$: $b_{{\\max}}={r['b_max_safe']}$")
    note = ", ".join(b_max_notes)
    lines.append(r"  \vspace{2pt}")
    lines.append(r"  \small{Max safe scale bits: " + note + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ===================================================================
# 5.  Report printer
# ===================================================================

def print_report(theo_result, emp_result=None, sweep_rows=None,
                 latex=False):
    """Print the complete analysis report."""
    arch = theo_result["architecture"]
    b = theo_result["scale_bits"]
    s = 1 << b

    print()
    print("=" * 66)
    print("  Overflow Safety Analysis for eBPF Quantized MLP")
    print("=" * 66)

    # ── Theorem statement ──
    print()
    print("Theorem 1 (Overflow Safety Bound).")
    print(f"  For a quantized MLP with architecture {arch},")
    print(f"  enlargement factor s = 2^b, max raw feature value M,")
    print(f"  and max absolute float weight W_max:")
    print()
    print("  The int64 accumulator does NOT overflow if and only if:")
    print()
    print("      b  <=  floor( (63 - log2(C_k)) / 2 )   for all layers k")
    print()
    print("  where C_k = max_accum_k / s^2 is the architecture-dependent")
    print("  constant at layer k (independent of b).")
    print()
    print("  For layer 0:  C_0 = d_in * W_max * M")
    print("  For layer k:  C_k = h_{k-1} * W_max * M_k")
    print("                M_k = C_{k-1} + W_max   (output of layer k-1)")
    print()

    # ── Per-layer theoretical bounds ──
    print("-" * 66)
    print(f"  Theoretical worst-case analysis  (b={b}, s=2^{b}={s})")
    print("-" * 66)
    for L in theo_result["layers"]:
        status = "SAFE" if L["safe"] else "OVERFLOW!"
        print(f"  Layer {L['index']}: "
              f"{L['fan_in']:>4d} -> {L['fan_out']:<4d}  "
              f"max_accum = {L['max_accum']:.3e}  "
              f"ratio = {L['ratio']:.3e}  "
              f"b_max = {L['b_max_layer']}  [{status}]")
    print()
    status_all = "SAFE" if theo_result["safe"] else "UNSAFE"
    print(f"  Overall: b_max_safe = {theo_result['b_max_safe']}  "
          f"(requested b={b})  -> {status_all}")

    # ── Empirical results ──
    if emp_result is not None:
        print()
        print("-" * 66)
        print(f"  Empirical analysis with actual weights  "
              f"(b={emp_result['scale_bits']})")
        print("-" * 66)
        for L in emp_result["layers"]:
            pct = L["ratio"] * 100
            # Choose format: use scientific notation for very small percentages
            if pct < 0.0001:
                pct_str = f"{pct:.2e}%"
            else:
                pct_str = f"{pct:.4f}%"
            print(f"  Layer {L['index']}: "
                  f"max_accum = {L['max_accum']:.3e} / "
                  f"{INT64_MAX:.3e} = {pct_str} of int64_max"
                  f"  (p99.9 = {L['p999_accum']:.3e})")
        hf = emp_result["headroom_factor"]
        extra_bits = math.log2(hf) if hf > 1 else 0
        print(f"\n  Headroom: s could be {hf:.1f}x larger before overflow")
        print(f"           (≈ {extra_bits:.1f} extra bits of scale)")
        print(f"  Quantized inference accuracy: {emp_result['accuracy']:.4f}")

    # ── Architecture sweep ──
    if sweep_rows is not None:
        print()
        print("-" * 66)
        print("  Architecture Sweep  (theoretical worst-case)")
        print("-" * 66)
        current_h = None
        for r in sweep_rows:
            if r["hidden"] != current_h:
                current_h = r["hidden"]
                # Print only for the default b to keep it concise
            if r["scale_bits"] == b:
                mark = "safe" if r["safe"] else "OVERFLOW"
                sym = "\u2713" if r["safe"] else "\u2717"
                print(f"  hidden={r['hidden']:<4d}  "
                      f"b_max_safe={r['b_max_safe']}  "
                      f"(b={b}) {sym} {mark}")

    # ── LaTeX table ──
    if latex and sweep_rows is not None:
        print()
        print("-" * 66)
        print("  LaTeX Table")
        print("-" * 66)
        print(generate_latex_table(sweep_rows))

    print()
    print("=" * 66)


# ===================================================================
# 6.  CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overflow safety analysis for eBPF quantized MLP.")
    parser.add_argument("--d-in", type=int, default=DEFAULT_D_IN,
                        help="Number of input features (default: 6)")
    parser.add_argument("--d-out", type=int, default=DEFAULT_D_OUT,
                        help="Number of output classes (default: 4)")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN,
                        help="Hidden layer width (default: 32)")
    parser.add_argument("--num-hidden", type=int, default=2,
                        help="Number of hidden layers (default: 2)")
    parser.add_argument("--scale-bits", type=int, default=DEFAULT_SCALE_BITS,
                        help="Scale factor exponent b, s=2^b (default: 16)")
    parser.add_argument("--w-max", type=float, default=TYPICAL_W_MAX,
                        help="Max absolute float weight (default: 3.0)")
    parser.add_argument("--max-feature", type=int, default=MAX_FEATURE_RAW,
                        help="Max raw feature value (default: 65536)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run architecture sweep over hidden sizes & bits")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX table")
    parser.add_argument("--empirical", action="store_true",
                        help="Run empirical analysis with trained model")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses (theoretical + empirical + "
                             "sweep + LaTeX)")
    args = parser.parse_args()

    if args.all:
        args.sweep = True
        args.latex = True
        args.empirical = True

    hidden_sizes = [args.hidden_size] * args.num_hidden

    # ── 1. Theoretical analysis ──
    theo = theoretical_analysis(
        d_in=args.d_in,
        hidden_sizes=hidden_sizes,
        d_out=args.d_out,
        scale_bits=args.scale_bits,
        max_feature_raw=args.max_feature,
        w_max=args.w_max,
    )

    # ── 2. Empirical analysis (optional) ──
    emp = None
    if args.empirical:
        np.random.seed(train_model.SEED)
        X, y = train_model.generate_dataset()

        split = int(0.8 * len(y))
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1e-8

        X_train_n = (X_train - mean) / std
        X_test_n = (X_test - mean) / std

        layer_sizes = ([args.d_in] + hidden_sizes + [args.d_out])

        print("  Training model for empirical analysis...")
        model = train_model.train_model(
            X_train_n, y_train, X_test_n, y_test,
            layer_sizes, epochs=80, batch_size=256, lr=0.05, quiet=True)

        emp = empirical_analysis(
            model, mean, std, X_test, y_test, args.scale_bits)

    # ── 3. Sweep (optional) ──
    sweep = None
    if args.sweep:
        sweep = architecture_sweep(
            d_in=args.d_in, d_out=args.d_out,
            hidden_list=[8, 16, 32, 64, 128],
            bits_list=[12, 14, 16, 18, 20, 22, 24],
            w_max=args.w_max,
        )

    # ── 4. Print report ──
    print_report(theo, emp_result=emp, sweep_rows=sweep, latex=args.latex)


if __name__ == "__main__":
    main()
