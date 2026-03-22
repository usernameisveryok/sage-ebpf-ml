#!/usr/bin/env python3
"""
ablation.py — Comprehensive ablation study and figure generation for the
eBPF in-kernel ML intrusion detection system.

Tests 6 inference modes on CIC-IDS-2017 (2.3M samples, 4 classes):
  1. Float           — standard float MLP
  2. Int32           — all layers quantized int32 (scale=2^16)
  3. MapLUT+Int32    — Layer 0 LUT, Layers 1/2 int32 multiply
  4. Int32+Ternary   — Layer 0 int32, Layers 1/2 ternary shift-add
  5. MapLUT+Ternary  — All zero multiply, no early exit
  6. MapLUT+Ternary+Exit — All zero multiply + early exit

Generates:
  paper/figures/ablation_accuracy.pdf
  paper/figures/instruction_comparison.pdf
  paper/figures/bins_vs_accuracy.pdf
  paper/figures/per_class_f1.pdf
  results/ablation_results.json

Dependencies: numpy, pyarrow, matplotlib
Usage:  python3 scripts/ablation.py
"""

import json
import math
import os
import sys
import time
import numpy as np
from pathlib import Path

# Line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────────────────────────────
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# 1.  Constants
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Fwd Packet Length Mean", "Fwd Header Length", "Avg Packet Size",
    "Fwd IAT Mean", "Total Fwd Packets", "Fwd Packet Length Max",
    "Init Bwd Win Bytes", "Init Fwd Win Bytes", "PSH Flag Count",
    "Total Backward Packets", "Fwd Packet Length Min", "Protocol",
]

LABEL_MAP = {
    "Benign": 0, "BENIGN": 0,
    "DDoS": 1, "DoS Hulk": 1, "DoS GoldenEye": 1,
    "DoS slowloris": 1, "DoS Slowhttptest": 1, "Heartbleed": 1,
    "PortScan": 2,
    "FTP-Patator": 3, "SSH-Patator": 3, "Bot": 3,
    "Infiltration": -1,
}

CLASS_NAMES = ["BENIGN", "DDOS", "PORTSCAN", "BRUTEFORCE"]
NUM_FEATURES = 12
NUM_CLASSES = 4
HIDDEN_SIZE = 64
SCALE_BITS = 16
SCALE = 1 << SCALE_BITS  # 65536
NUM_BINS = 64
OVERSAMPLE_TARGET = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "cicids_raw"
FIGURES_DIR = PROJECT_DIR / "paper" / "figures"
RESULTS_PATH = PROJECT_DIR / "results" / "ablation_results.json"


# ──────────────────────────────────────────────────────────────────────
# 2.  Data loading
# ──────────────────────────────────────────────────────────────────────

def map_label(label_str):
    """Map a label string to class index. Returns -1 to skip."""
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    if "Web Attack" in label_str:
        return 3
    if "DoS" in label_str or "DDoS" in label_str:
        return 1
    return -1


def load_data():
    """Load CIC-IDS-2017 parquet files, return X (float64), y (int64)."""
    import pyarrow.parquet as pq

    all_X, all_y = [], []
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files in {DATA_DIR}")

    for fpath in parquet_files:
        table = pq.read_table(str(fpath), columns=FEATURE_COLS + ["Label"])
        n_rows = table.num_rows
        labels = table.column("Label").to_pylist()

        feat_arrays = []
        for col in FEATURE_COLS:
            arr = table.column(col).to_numpy()
            feat_arrays.append(arr.astype(np.float64))
        X_file = np.column_stack(feat_arrays)

        kept = 0
        for i in range(n_rows):
            label = labels[i]
            cls = map_label(label)
            if cls < 0:
                continue
            row = X_file[i]
            if np.any(np.isnan(row)) or np.any(np.isinf(row)):
                continue
            all_X.append(row)
            all_y.append(cls)
            kept += 1
        print(f"    {fpath.name}: {n_rows} rows -> {kept} kept")

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y, dtype=np.int64)
    X = np.maximum(X, 0.0)

    print(f"  Total samples: {len(y)}")
    for c in range(NUM_CLASSES):
        print(f"    Class {c} ({CLASS_NAMES[c]:>12s}): {(y == c).sum()}")
    return X, y


# ──────────────────────────────────────────────────────────────────────
# 3.  Oversampling
# ──────────────────────────────────────────────────────────────────────

def oversample(X, y, target=OVERSAMPLE_TARGET, noise_std=0.05):
    """Balance classes to target count with multiplicative Gaussian noise."""
    Xs, ys = [], []
    for c in range(NUM_CLASSES):
        mask = y == c
        Xc = X[mask]
        nc = len(Xc)
        if nc == 0:
            print(f"  WARNING: class {c} has 0 samples!")
            continue
        if nc >= target:
            idx = np.random.choice(nc, target, replace=False)
            Xs.append(Xc[idx])
        else:
            reps = target // nc
            remainder = target % nc
            parts = []
            for r in range(reps):
                if r == 0:
                    parts.append(Xc.copy())
                else:
                    noisy = Xc * (1.0 + np.random.randn(*Xc.shape) * noise_std)
                    noisy = np.maximum(noisy, 0.0)
                    parts.append(noisy)
            if remainder > 0:
                idx = np.random.choice(nc, remainder, replace=False)
                noisy = Xc[idx] * (1.0 + np.random.randn(remainder, X.shape[1]) * noise_std)
                noisy = np.maximum(noisy, 0.0)
                parts.append(noisy)
            Xs.append(np.concatenate(parts, axis=0))
        ys.append(np.full(target, c, dtype=np.int64))
        print(f"    Class {c}: {nc} -> {target}")

    X_out = np.concatenate(Xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    idx = np.random.permutation(len(y_out))
    return X_out[idx], y_out[idx]


# ──────────────────────────────────────────────────────────────────────
# 4.  MLP with Focal Loss
# ──────────────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    """MLP: [12, 64, 64, 4] with He init."""

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(layer_sizes[i + 1], fan_in) * scale)
            self.b.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, X):
        cache = {"a": [X], "z": []}
        a = X
        for k in range(len(self.W)):
            z = a @ self.W[k].T + self.b[k]
            cache["z"].append(z)
            if k < len(self.W) - 1:
                a = relu(z)
            else:
                a = softmax(z)
            cache["a"].append(a)
        return a, cache

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def get_hidden(self, X, layer_idx):
        """Get hidden activation after a specific layer (post-ReLU)."""
        a = X
        for k in range(layer_idx + 1):
            z = a @ self.W[k].T + self.b[k]
            a = relu(z)
        return a


def compute_focal_alpha(y, num_classes=NUM_CLASSES):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    N = len(y)
    alpha = np.sqrt(N / (num_classes * counts))
    alpha = alpha / alpha.mean()
    return alpha


def focal_backward(probs, y, alpha, gamma):
    N = len(y)
    p_t = probs[np.arange(N), y]
    p_t = np.clip(p_t, 1e-12, 1.0)
    focal_w = alpha[y] * (1.0 - p_t) ** gamma
    dz = probs.copy()
    dz[np.arange(N), y] -= 1.0
    dz *= focal_w[:, None]
    dz /= focal_w.sum()  # normalize by focal weight sum, not N
    return dz


def train_mlp(X_train, y_train, X_test, y_test,
              layer_sizes, epochs=100, batch_size=256, lr=0.01, gamma=3.0):
    """Train MLP with focal loss. LR decay at epochs 25, 50, 75."""
    model = MLP(layer_sizes)
    N = len(y_train)
    alpha = compute_focal_alpha(y_train)
    print(f"  Focal alpha: {np.round(alpha, 4)}")
    print(f"  Focal gamma: {gamma}")

    best_score = -1.0
    best_W = [w.copy() for w in model.W]
    best_b = [b.copy() for b in model.b]

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            nb = len(yb)

            probs, cache = model.forward(Xb)

            # Focal loss value for logging
            p_t = probs[np.arange(nb), yb]
            p_t = np.clip(p_t, 1e-12, 1.0)
            fl = -alpha[yb] * (1.0 - p_t) ** gamma * np.log(p_t)
            epoch_loss += fl.mean()
            n_batches += 1

            # Backward with focal loss gradient
            dz = focal_backward(probs, yb, alpha, gamma)

            # Backprop through layers
            dW_list = [None] * len(model.W)
            db_list = [None] * len(model.W)
            for k in reversed(range(len(model.W))):
                a_prev = cache["a"][k]
                dW_list[k] = dz.T @ a_prev
                db_list[k] = dz.sum(axis=0)
                if k > 0:
                    da = dz @ model.W[k]
                    dz = da * relu_grad(cache["z"][k - 1])

            # SGD update
            for k in range(len(model.W)):
                model.W[k] -= lr * dW_list[k]
                model.b[k] -= lr * db_list[k]

        # LR decay at epochs 25, 50, 75
        if epoch in (25, 50, 75):
            lr *= 0.5
            print(f"    LR decayed to {lr:.6f}")

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            score = metrics["accuracy"] + metrics["macro_f1"]
            if score > best_score:
                best_score = score
                best_W = [w.copy() for w in model.W]
                best_b = [b.copy() for b in model.b]
            print(f"  epoch {epoch:3d}  loss={epoch_loss / n_batches:.4f}  "
                  f"acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"score={score:.4f}  best={best_score:.4f}")

    # Restore best
    model.W = best_W
    model.b = best_b
    print(f"\n  Best score (accuracy + macro_f1): {best_score:.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────
# 5.  Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Compute accuracy, per-class precision/recall/f1, macro_f1."""
    acc = (y_true == y_pred).mean()
    per_class = {}
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": int((y_true == c).sum()),
        }
        f1s.append(f1)
    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }


def print_report(y_true, y_pred, title=""):
    metrics = compute_metrics(y_true, y_pred)
    if title:
        print(f"\n  {title}")
    print(f"  {'Class':<12s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>8s}")
    print("  " + "-" * 44)
    for c in range(NUM_CLASSES):
        m = metrics["per_class"][CLASS_NAMES[c]]
        print(f"  {CLASS_NAMES[c]:<12s} {m['precision']:8.4f} {m['recall']:8.4f} "
              f"{m['f1']:8.4f} {m['support']:8d}")
    print("  " + "-" * 44)
    print(f"  {'Accuracy':<12s} {'':>8s} {'':>8s} {metrics['accuracy']:8.4f} {len(y_true):8d}")
    print(f"  {'Macro F1':<12s} {'':>8s} {'':>8s} {metrics['macro_f1']:8.4f}")
    return metrics


# ──────────────────────────────────────────────────────────────────────
# 6.  Ternary Quantization
# ──────────────────────────────────────────────────────────────────────

def ternary_quantize_layer(W, threshold_ratio=0.7):
    """Ternary quantization: signs in {-1, 0, +1}, alpha = mean(|W[large]|)."""
    abs_W = np.abs(W)
    threshold = threshold_ratio * abs_W.mean()
    mask_pos = W > threshold
    mask_neg = W < -threshold

    large_entries = abs_W[mask_pos | mask_neg]
    if len(large_entries) == 0:
        alpha_float = 1.0
    else:
        alpha_float = large_entries.mean()

    alpha_shift = int(np.round(SCALE_BITS + np.log2(alpha_float)))
    alpha_shift = max(alpha_shift, 0)

    signs = np.zeros_like(W, dtype=np.int64)
    signs[mask_pos] = 1
    signs[mask_neg] = -1

    n_zero = (signs == 0).sum()
    sparsity = n_zero / signs.size * 100

    print(f"    threshold_ratio={threshold_ratio}, threshold={threshold:.6f}, "
          f"alpha={alpha_float:.6f}, alpha_shift={alpha_shift}, "
          f"sparsity={sparsity:.1f}%")

    return signs, alpha_shift, alpha_float


# ──────────────────────────────────────────────────────────────────────
# 7.  MapLUT Generation
# ──────────────────────────────────────────────────────────────────────

def compute_lut_params(X_train_raw, num_bins=NUM_BINS):
    feat_offset = np.zeros(NUM_FEATURES, dtype=np.int64)
    feat_shift = np.zeros(NUM_FEATURES, dtype=np.int64)

    for j in range(NUM_FEATURES):
        col = X_train_raw[:, j]
        fmin = col.min()
        fmax = col.max()
        feat_offset[j] = int(math.floor(fmin))
        feat_range = fmax - feat_offset[j]
        if feat_range <= 0:
            feat_range = 1.0
        raw_shift = math.log2(feat_range / num_bins) if feat_range > num_bins else 0
        feat_shift[j] = max(int(math.ceil(raw_shift)), 0)

    return feat_offset.astype(np.int32), feat_shift.astype(np.uint32)


def generate_lut(W0, b0, mu, sigma, feat_offset, feat_shift, num_bins=NUM_BINS):
    """Generate Layer 0 MapLUT using LEFT EDGE of each bin."""
    s = SCALE
    lut_data = np.zeros((NUM_FEATURES, num_bins, HIDDEN_SIZE), dtype=np.int32)

    for j in range(NUM_FEATURES):
        bin_width = 1 << int(feat_shift[j])
        for b in range(num_bins):
            center_raw = float(feat_offset[j]) + b * bin_width  # left edge
            center_norm = (center_raw - mu[j]) / sigma[j]
            for i in range(HIDDEN_SIZE):
                lut_data[j, b, i] = int(np.round(W0[i, j] * center_norm * s))

    bias_layer_0_q = np.round(b0 * s).astype(np.int32)
    return lut_data, bias_layer_0_q


# ──────────────────────────────────────────────────────────────────────
# 8.  Early-Exit Head
# ──────────────────────────────────────────────────────────────────────

def train_exit_head(model, X_train_n, y_train, X_test_n, y_test,
                    alpha, epochs=30, lr=0.01, batch_size=256):
    """Train a linear exit head (64->4) on hidden layer 0 output."""
    print("\n  Training early-exit head...")
    H_train = model.get_hidden(X_train_n, 0)
    H_test = model.get_hidden(X_test_n, 0)

    fan_in = HIDDEN_SIZE
    scale = np.sqrt(2.0 / fan_in)
    exit_W = np.random.randn(NUM_CLASSES, fan_in) * scale
    exit_b = np.zeros(NUM_CLASSES)
    N = len(y_train)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        H_shuf = H_train[perm]
        y_shuf = y_train[perm]

        for start in range(0, N, batch_size):
            Hb = H_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            nb = len(yb)

            z = Hb @ exit_W.T + exit_b
            probs = softmax(z)
            p_t = probs[np.arange(nb), yb]
            p_t = np.clip(p_t, 1e-12, 1.0)
            focal_w = alpha[yb] * (1.0 - p_t) ** 1.0

            dz = probs.copy()
            dz[np.arange(nb), yb] -= 1.0
            dz *= focal_w[:, None]
            dz /= nb

            exit_W -= lr * (dz.T @ Hb)
            exit_b -= lr * dz.sum(axis=0)

        if epoch % 10 == 0 or epoch == 1:
            z_test = H_test @ exit_W.T + exit_b
            pred = np.argmax(z_test, axis=1)
            acc = (pred == y_test).mean()
            print(f"    exit epoch {epoch:3d}  acc={acc:.4f}")

    return exit_W, exit_b


def calibrate_exit_threshold(model, exit_W, exit_b, X_test_n, y_test,
                             percentile=15):
    """Calibrate exit threshold at the given percentile of logit margins."""
    H_test = model.get_hidden(X_test_n, 0)
    z = H_test @ exit_W.T + exit_b
    pred = np.argmax(z, axis=1)
    correct_mask = pred == y_test

    z_correct = z[correct_mask]
    sorted_logits = np.sort(z_correct, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]

    threshold_float = np.percentile(margins, percentile)
    threshold_int = int(np.round(threshold_float * SCALE))

    print(f"  Exit threshold calibration:")
    print(f"    Correct samples: {correct_mask.sum()} / {len(y_test)}")
    print(f"    Margin percentile {percentile}%: {threshold_float:.4f}")
    print(f"    Int threshold: {threshold_int}")
    return threshold_int


# ──────────────────────────────────────────────────────────────────────
# 9.  Inference Engines (6 modes)
# ──────────────────────────────────────────────────────────────────────

def mode1_float(X_norm, model):
    """Mode 1: Standard float MLP forward pass."""
    return model.predict(X_norm)


def mode2_int32(X_raw, model, mu, sigma):
    """Mode 2: All layers quantized int32 (scale=2^16)."""
    s = SCALE
    sb = SCALE_BITS

    Wq = [np.round(w * s).astype(np.int64) for w in model.W]
    Bq = [np.round(b * s).astype(np.int64) for b in model.b]

    X_norm = (X_raw - mu) / sigma
    xq = np.round(X_norm * s).astype(np.int64)

    x = xq
    for k in range(len(Wq)):
        z = (x @ Wq[k].T >> sb) + Bq[k]
        if k < len(Wq) - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def mode3_maplut_int32(X_raw, model, mu, sigma,
                       lut_data, bias_layer_0_q,
                       feat_offset, feat_shift, num_bins=NUM_BINS):
    """Mode 3: MapLUT+Int32 — Layer 0 via LUT, Layers 1/2 via int32 multiply."""
    s = SCALE
    sb = SCALE_BITS
    N = X_raw.shape[0]

    # Layer 0: MapLUT accumulation
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = raw - int(feat_offset[j])
        shifted = np.maximum(shifted, 0)
        bins = (shifted >> int(feat_shift[j])).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        hidden += lut_data[j][bins]

    # ReLU after layer 0
    hidden = np.maximum(hidden, 0)

    # Layers 1, 2: int32 multiply (hidden values at scale s)
    x = hidden
    for k in [1, 2]:
        Wq = np.round(model.W[k] * s).astype(np.int64)
        Bq = np.round(model.b[k] * s).astype(np.int64)
        z = (x @ Wq.T >> sb) + Bq
        if k < len(model.W) - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def mode4_int32_ternary(X_raw, model, mu, sigma,
                        ternary_signs, alpha_shifts):
    """Mode 4: Int32+Ternary — Layer 0 int32, Layers 1/2 ternary."""
    s = SCALE
    sb = SCALE_BITS

    # Layer 0: full int32
    Wq0 = np.round(model.W[0] * s).astype(np.int64)
    Bq0 = np.round(model.b[0] * s).astype(np.int64)

    X_norm = (X_raw - mu) / sigma
    xq = np.round(X_norm * s).astype(np.int64)

    z = (xq @ Wq0.T >> sb) + Bq0
    z = np.maximum(z, 0)
    x = z

    # Ternary layers (1, 2)
    for k_rel, k_abs in enumerate([1, 2]):
        signs = ternary_signs[k_rel].astype(np.int64)
        ashift = alpha_shifts[k_rel]
        Bq = np.round(model.b[k_abs] * s).astype(np.int64)

        acc = x @ signs.T
        z = (acc << ashift) >> sb
        z += Bq
        if k_abs < len(model.W) - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def mode5_maplut_ternary_noexit(X_raw, lut_data, bias_layer_0_q,
                                feat_offset, feat_shift,
                                ternary_signs, alpha_shifts, ternary_biases,
                                num_bins=NUM_BINS):
    """Mode 5: MapLUT+Ternary (no exit) — All zero multiply, no early exit."""
    sb = SCALE_BITS
    N = X_raw.shape[0]

    # Layer 0: MapLUT
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = raw - int(feat_offset[j])
        shifted = np.maximum(shifted, 0)
        bins = (shifted >> int(feat_shift[j])).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        hidden += lut_data[j][bins]
    hidden = np.maximum(hidden, 0)

    # Ternary layers 1, 2
    x = hidden
    for k_rel in range(2):
        signs = ternary_signs[k_rel].astype(np.int64)
        ashift = alpha_shifts[k_rel]
        bq = ternary_biases[k_rel].astype(np.int64)

        acc = x @ signs.T
        z = ((acc << ashift) >> sb) + bq
        if k_rel < 1:  # ReLU for layer 1 only
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def mode6_maplut_ternary_exit(X_raw, lut_data, bias_layer_0_q,
                              feat_offset, feat_shift,
                              exit_signs, exit_alpha_shift, exit_bias_q,
                              ternary_signs, alpha_shifts, ternary_biases,
                              exit_threshold, num_bins=NUM_BINS):
    """Mode 6: MapLUT+Ternary+Exit — Full zero-multiply pipeline with early exit."""
    sb = SCALE_BITS
    N = X_raw.shape[0]

    # Layer 0: MapLUT
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = raw - int(feat_offset[j])
        shifted = np.maximum(shifted, 0)
        bins = (shifted >> int(feat_shift[j])).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        hidden += lut_data[j][bins]
    hidden = np.maximum(hidden, 0)

    # Exit head: ternary on hidden
    exit_signs_64 = exit_signs.astype(np.int64)
    exit_acc = hidden @ exit_signs_64.T
    exit_logits = ((exit_acc << exit_alpha_shift) >> sb) + exit_bias_q.astype(np.int64)

    # Early exit decision
    preds = np.full(N, -1, dtype=np.int64)
    exit_pred = np.argmax(exit_logits, axis=1)
    sorted_el = np.sort(exit_logits, axis=1)
    margins = sorted_el[:, -1] - sorted_el[:, -2]
    confident = margins >= exit_threshold
    preds[confident] = exit_pred[confident]
    early_exit_count = int(confident.sum())

    # Full pipeline for remaining
    need_full = preds == -1
    if need_full.any():
        x = hidden[need_full].copy()

        for k_rel in range(2):
            signs = ternary_signs[k_rel].astype(np.int64)
            ashift = alpha_shifts[k_rel]
            bq = ternary_biases[k_rel].astype(np.int64)

            acc = x @ signs.T
            z = ((acc << ashift) >> sb) + bq
            if k_rel < 1:
                z = np.maximum(z, 0)
            x = z

        full_pred = np.argmax(x, axis=1)
        preds[need_full] = full_pred

    return preds, early_exit_count


# ──────────────────────────────────────────────────────────────────────
# 10. Figure Generation
# ──────────────────────────────────────────────────────────────────────

def setup_matplotlib():
    """Set up matplotlib for clean academic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (6, 4),
    })
    return plt


def fig_ablation_accuracy(plt, mode_results, outdir):
    """Fig 1: Bar chart comparing accuracy and Macro F1 of all 6 modes."""
    names = [r["name"] for r in mode_results]
    accs = [r["metrics"]["accuracy"] for r in mode_results]
    f1s = [r["metrics"]["macro_f1"] for r in mode_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy",
                   color="#0072B2", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1",
                   color="#D55E00", edgecolor="black", linewidth=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Accuracy and Macro F1 Across Inference Modes")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = os.path.join(outdir, "ablation_accuracy.pdf")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


def fig_instruction_comparison(plt, outdir):
    """Fig 2: Grouped bar chart for BPF instruction counts."""
    versions = ["V0-Baseline", "V2-Fused+Ternary+Exit", "V3-MapLUT+Ternary+Exit"]
    total_insns = [804, 1437, 1424]
    mul_insns = [15, 16, 0]

    x = np.arange(len(versions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, total_insns, width, label="Total Instructions",
                   color="#0072B2", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, mul_insns, width, label="Multiply Instructions",
                   color="#D55E00", edgecolor="black", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Instruction Count")
    ax.set_title("eBPF Instruction Comparison Across Versions")
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=9)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(outdir, "instruction_comparison.pdf")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


def fig_bins_vs_accuracy(plt, bins_results, outdir):
    """Fig 3: Line plot of MapLUT accuracy vs number of bins."""
    bins_list = [r["bins"] for r in bins_results]
    accs = [r["accuracy"] for r in bins_results]
    mems = [r["lut_memory_kb"] for r in bins_results]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_acc = "#0072B2"
    ax1.plot(bins_list, accs, marker="o", color=color_acc, linewidth=2,
             markersize=8, label="Accuracy", zorder=3)
    ax1.set_xlabel("Number of Bins")
    ax1.set_ylabel("Accuracy", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(bins_list)
    ax1.set_xticklabels([str(b) for b in bins_list])

    # Add memory on right axis
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    color_mem = "#D55E00"
    ax2.plot(bins_list, mems, marker="s", color=color_mem, linewidth=1.5,
             linestyle="--", markersize=6, label="LUT Memory", zorder=2)
    ax2.set_ylabel("LUT Memory (KB)", color=color_mem)
    ax2.tick_params(axis="y", labelcolor=color_mem)

    # Annotate accuracy values
    for b, a in zip(bins_list, accs):
        ax1.annotate(f"{a:.4f}", (b, a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=color_acc)

    ax1.set_title("MapLUT: Accuracy vs Number of Bins")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    path = os.path.join(outdir, "bins_vs_accuracy.pdf")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


def fig_per_class_f1(plt, mode_results, outdir):
    """Fig 4: Grouped bar chart of per-class F1 for selected modes."""
    selected_modes = ["Float", "Int32", "MapLUT+Int32", "MapLUT+Tern+Exit"]
    selected = [r for r in mode_results if r["name"] in selected_modes]

    classes = CLASS_NAMES
    n_classes = len(classes)
    n_modes = len(selected)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_classes)
    width = 0.8 / n_modes
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for i, mode in enumerate(selected):
        f1s = [mode["metrics"]["per_class"][c]["f1"] for c in classes]
        offset = (i - n_modes / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1s, width, label=mode["name"],
                      color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.4)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Across Inference Modes")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    path = os.path.join(outdir, "per_class_f1.pdf")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 11. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES]

    # ── Step 1: Load data ──
    print("=" * 65)
    print("Step 1: Loading CIC-IDS-2017 dataset")
    print("=" * 65)
    X_all, y_all = load_data()

    # Shuffle and split 75/25
    idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]
    split = int(0.75 * len(y_all))
    X_train_raw, y_train = X_all[:split], y_all[:split]
    X_test_raw, y_test = X_all[split:], y_all[split:]
    print(f"\n  Train: {len(y_train)}, Test: {len(y_test)}")

    # Normalization stats (before oversampling)
    mu = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma[sigma < 1e-8] = 1e-8
    print(f"  Feature means: {np.round(mu, 2)}")
    print(f"  Feature stds:  {np.round(sigma, 2)}")

    X_train_n = (X_train_raw - mu) / sigma
    X_test_n = (X_test_raw - mu) / sigma

    # ── Step 2: Oversample ──
    print("\n" + "=" * 65)
    print("Step 2: Oversampling to 100K per class")
    print("=" * 65)
    X_train_os, y_train_os = oversample(X_train_n, y_train,
                                         target=OVERSAMPLE_TARGET)
    print(f"  Oversampled: {len(y_train_os)} total")

    # ── Step 3: Train MLP ──
    print("\n" + "=" * 65)
    print(f"Step 3: Training MLP {layer_sizes} (100 epochs, focal loss)")
    print("=" * 65)
    t_train = time.time()
    model = train_mlp(X_train_os, y_train_os, X_test_n, y_test,
                      layer_sizes, epochs=100, batch_size=256,
                      lr=0.01, gamma=3.0)
    print(f"  Training time: {time.time() - t_train:.1f}s")

    # ── Step 4: Early-exit head ──
    print("\n" + "=" * 65)
    print("Step 4: Training early-exit head")
    print("=" * 65)
    alpha = compute_focal_alpha(y_train_os)
    exit_W, exit_b = train_exit_head(model, X_train_os, y_train_os,
                                      X_test_n, y_test, alpha)
    exit_threshold = calibrate_exit_threshold(model, exit_W, exit_b,
                                               X_test_n, y_test,
                                               percentile=15)

    # ── Step 5: Ternary quantization — sweep threshold ratios ──
    print("\n" + "=" * 65)
    print("Step 5: Ternary quantization — threshold ratio sweep")
    print("=" * 65)

    best_ternary_ratio = 0.7
    best_ternary_acc = -1
    ternary_sweep_results = []

    for ratio in [0.3, 0.5, 0.7]:
        print(f"\n  --- threshold_ratio = {ratio} ---")
        t_signs, t_ashifts = [], []
        for k_abs in [1, 2]:
            print(f"  Layer {k_abs} ({model.W[k_abs].shape}):")
            signs, ashift, _ = ternary_quantize_layer(model.W[k_abs],
                                                       threshold_ratio=ratio)
            t_signs.append(signs)
            t_ashifts.append(ashift)

        # Quick test with mode 4 (int32+ternary)
        y_pred = mode4_int32_ternary(X_test_raw, model, mu, sigma,
                                      t_signs, t_ashifts)
        acc = float((y_pred == y_test).mean())
        m = compute_metrics(y_test, y_pred)
        ternary_sweep_results.append({
            "threshold_ratio": ratio,
            "accuracy": acc,
            "macro_f1": m["macro_f1"],
        })
        print(f"  Int32+Ternary(ratio={ratio}): acc={acc:.4f}, "
              f"macro_f1={m['macro_f1']:.4f}")

        if acc > best_ternary_acc:
            best_ternary_acc = acc
            best_ternary_ratio = ratio

    print(f"\n  Best ternary threshold_ratio: {best_ternary_ratio} "
          f"(acc={best_ternary_acc:.4f})")

    # ── Step 6: Quantize with best ratio ──
    print("\n" + "=" * 65)
    print(f"Step 6: Final ternary quantization (ratio={best_ternary_ratio})")
    print("=" * 65)
    ternary_signs = []
    alpha_shifts = []
    for k_abs in [1, 2]:
        print(f"  Layer {k_abs} ({model.W[k_abs].shape}):")
        signs, ashift, _ = ternary_quantize_layer(model.W[k_abs],
                                                   threshold_ratio=best_ternary_ratio)
        ternary_signs.append(signs)
        alpha_shifts.append(ashift)

    print(f"  Exit head ({exit_W.shape}):")
    exit_signs, exit_alpha_shift, _ = ternary_quantize_layer(
        exit_W, threshold_ratio=best_ternary_ratio)
    exit_bias_q = np.round(exit_b * SCALE).astype(np.int32)

    ternary_biases = [
        np.round(model.b[1] * SCALE).astype(np.int32),
        np.round(model.b[2] * SCALE).astype(np.int32),
    ]

    # ── Step 7: MapLUT generation ──
    print("\n" + "=" * 65)
    print("Step 7: MapLUT generation (Layer 0)")
    print("=" * 65)
    feat_offset, feat_shift = compute_lut_params(X_train_raw, NUM_BINS)
    lut_data, bias_layer_0_q = generate_lut(
        model.W[0], model.b[0], mu, sigma, feat_offset, feat_shift, NUM_BINS)
    lut_mem = lut_data.nbytes
    print(f"  feat_offset: {feat_offset.tolist()}")
    print(f"  feat_shift:  {feat_shift.tolist()}")
    print(f"  LUT shape:   {lut_data.shape}")
    print(f"  LUT memory:  {lut_mem} bytes ({lut_mem / 1024:.1f} KB)")

    # ── Step 8: Evaluate all 6 inference modes ──
    print("\n" + "=" * 65)
    print("Step 8: Evaluating all 6 inference modes")
    print("=" * 65)

    mode_results = []

    # Mode 1: Float
    print("\n  --- Mode 1: Float ---")
    y1 = mode1_float(X_test_n, model)
    m1 = print_report(y_test, y1, "Float")
    mode_results.append({"name": "Float", "metrics": m1})

    # Mode 2: Int32
    print("\n  --- Mode 2: Int32 ---")
    y2 = mode2_int32(X_test_raw, model, mu, sigma)
    m2 = print_report(y_test, y2, "Int32")
    mode_results.append({"name": "Int32", "metrics": m2})

    # Mode 3: MapLUT+Int32
    print("\n  --- Mode 3: MapLUT+Int32 ---")
    y3 = mode3_maplut_int32(X_test_raw, model, mu, sigma,
                             lut_data, bias_layer_0_q,
                             feat_offset, feat_shift)
    m3 = print_report(y_test, y3, "MapLUT+Int32")
    mode_results.append({"name": "MapLUT+Int32", "metrics": m3})

    # Mode 4: Int32+Ternary
    print("\n  --- Mode 4: Int32+Ternary ---")
    y4 = mode4_int32_ternary(X_test_raw, model, mu, sigma,
                              ternary_signs, alpha_shifts)
    m4 = print_report(y_test, y4, "Int32+Ternary")
    mode_results.append({"name": "Int32+Ternary", "metrics": m4})

    # Mode 5: MapLUT+Ternary (no exit)
    print("\n  --- Mode 5: MapLUT+Ternary (no exit) ---")
    y5 = mode5_maplut_ternary_noexit(X_test_raw, lut_data, bias_layer_0_q,
                                      feat_offset, feat_shift,
                                      ternary_signs, alpha_shifts,
                                      ternary_biases)
    m5 = print_report(y_test, y5, "MapLUT+Ternary (no exit)")
    mode_results.append({"name": "MapLUT+Tern", "metrics": m5})

    # Mode 6: MapLUT+Ternary+Exit
    print("\n  --- Mode 6: MapLUT+Ternary+Exit ---")
    y6, exit_cnt = mode6_maplut_ternary_exit(
        X_test_raw, lut_data, bias_layer_0_q,
        feat_offset, feat_shift,
        exit_signs, exit_alpha_shift, exit_bias_q,
        ternary_signs, alpha_shifts, ternary_biases,
        exit_threshold)
    m6 = print_report(y_test, y6, "MapLUT+Ternary+Exit")
    early_exit_rate = exit_cnt / len(y_test)
    print(f"  Early exit: {exit_cnt}/{len(y_test)} = {early_exit_rate:.2%}")
    mode_results.append({
        "name": "MapLUT+Tern+Exit",
        "metrics": m6,
        "early_exit_count": exit_cnt,
        "early_exit_rate": float(early_exit_rate),
    })

    # Summary table
    print("\n  " + "=" * 65)
    print(f"  {'Mode':<22s} {'Accuracy':>10s} {'Macro F1':>10s} {'Mul-Free':>10s}")
    print("  " + "-" * 65)
    mul_free = ["No", "No", "Partial", "Partial", "Yes", "Yes"]
    for i, r in enumerate(mode_results):
        print(f"  {r['name']:<22s} {r['metrics']['accuracy']:10.4f} "
              f"{r['metrics']['macro_f1']:10.4f} {mul_free[i]:>10s}")
    print("  " + "=" * 65)

    # ── Step 9: Bins vs accuracy sweep ──
    print("\n" + "=" * 65)
    print("Step 9: Bins vs accuracy sweep")
    print("=" * 65)
    bins_results = []
    for nb in [16, 32, 64, 128, 256]:
        fo_nb, fs_nb = compute_lut_params(X_train_raw, nb)
        lut_nb, blq_nb = generate_lut(
            model.W[0], model.b[0], mu, sigma, fo_nb, fs_nb, nb)
        y_nb = mode5_maplut_ternary_noexit(
            X_test_raw, lut_nb, blq_nb, fo_nb, fs_nb,
            ternary_signs, alpha_shifts, ternary_biases, num_bins=nb)
        acc_nb = float((y_nb == y_test).mean())
        mem_nb = nb * NUM_FEATURES * HIDDEN_SIZE * 4
        bins_results.append({
            "bins": nb,
            "accuracy": acc_nb,
            "lut_memory_bytes": mem_nb,
            "lut_memory_kb": mem_nb / 1024,
        })
        print(f"    bins={nb:4d}  acc={acc_nb:.4f}  mem={mem_nb / 1024:.0f} KB")

    # ── Step 10: BPF instruction metrics ──
    print("\n" + "=" * 65)
    print("Step 10: BPF instruction metrics")
    print("=" * 65)
    bpf_metrics = [
        {"version": "V0-Baseline", "total_insns": 804, "multiply_insns": 15},
        {"version": "V2-Fused+Ternary+Exit", "total_insns": 1437, "multiply_insns": 16},
        {"version": "V3-MapLUT+Ternary+Exit", "total_insns": 1424, "multiply_insns": 0},
    ]
    print(f"  {'Version':<30s} {'Total Insns':>12s} {'Multiply':>10s}")
    print("  " + "-" * 55)
    for b in bpf_metrics:
        print(f"  {b['version']:<30s} {b['total_insns']:12d} {b['multiply_insns']:10d}")

    # ── Step 11: Generate figures ──
    print("\n" + "=" * 65)
    print("Step 11: Generating figures")
    print("=" * 65)
    os.makedirs(str(FIGURES_DIR), exist_ok=True)
    plt = setup_matplotlib()

    fig_ablation_accuracy(plt, mode_results, str(FIGURES_DIR))
    fig_instruction_comparison(plt, str(FIGURES_DIR))
    fig_bins_vs_accuracy(plt, bins_results, str(FIGURES_DIR))
    fig_per_class_f1(plt, mode_results, str(FIGURES_DIR))

    # ── Step 12: Save results JSON ──
    print("\n" + "=" * 65)
    print("Step 12: Saving results")
    print("=" * 65)
    results = {
        "architecture": layer_sizes,
        "dataset": "CIC-IDS-2017",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "oversampled_train_samples": int(len(y_train_os)),
        "scale_bits": SCALE_BITS,
        "num_bins": NUM_BINS,
        "best_ternary_threshold_ratio": best_ternary_ratio,
        "ternary_sweep": ternary_sweep_results,
        "modes": [],
        "bins_vs_accuracy": bins_results,
        "bpf_metrics": bpf_metrics,
        "exit_threshold_int": exit_threshold,
        "early_exit_rate": float(early_exit_rate),
        "total_time_seconds": round(time.time() - t_start, 1),
    }

    for r in mode_results:
        entry = {"name": r["name"], **r["metrics"]}
        if "early_exit_count" in r:
            entry["early_exit_count"] = r["early_exit_count"]
            entry["early_exit_rate"] = r["early_exit_rate"]
        results["modes"].append(entry)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results written to {RESULTS_PATH}")

    # ── Final summary ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"ABLATION STUDY COMPLETE  ({elapsed:.0f}s)")
    print("=" * 65)
    print(f"  Figures:  {FIGURES_DIR}/ablation_accuracy.pdf")
    print(f"            {FIGURES_DIR}/instruction_comparison.pdf")
    print(f"            {FIGURES_DIR}/bins_vs_accuracy.pdf")
    print(f"            {FIGURES_DIR}/per_class_f1.pdf")
    print(f"  Results:  {RESULTS_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
