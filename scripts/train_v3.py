#!/usr/bin/env python3
"""
train_v3.py — Train MLP [12,64,64,4] on CIC-IDS-2017 and generate all
artifacts for the V3 zero-multiply eBPF inference pipeline.

Operators:
  Layer 0:    MapLUT   (precomputed lookup table, zero multiply)
  Exit head:  TernaryShift  (ternary weights on hidden vector, zero multiply)
  Layer 1:    TernaryShift  [64→64]
  Layer 2:    TernaryShift  [64→4]

Outputs:
  include/model_params_v3.h   — ternary weights, biases, feat quantization
  data/lut_v3.bin             — Layer-0 LUT binary (192 KB)
  results/v3_results.json     — accuracy for all 4 inference modes

Dependencies: numpy, pyarrow
Usage:  python3 scripts/train_v3.py
"""

import json
import math
import os
import struct
import subprocess
import sys
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
    "Infiltration": -1,  # skip
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
HEADER_PATH = PROJECT_DIR / "include" / "model_params_v3.h"
LUT_BIN_PATH = PROJECT_DIR / "data" / "lut_v3.bin"
RESULTS_PATH = PROJECT_DIR / "results" / "v3_results.json"


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
    return -1  # skip unknown


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

    # Clamp negatives to 0
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
    """Alpha = sqrt(N / (num_classes * counts)), normalized to mean=1."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    N = len(y)
    alpha = np.sqrt(N / (num_classes * counts))
    alpha = alpha / alpha.mean()
    return alpha


def focal_backward(probs, y, alpha, gamma):
    """Focal loss gradient for softmax output."""
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
    """Train MLP with focal loss."""
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

        # LR decay
        if epoch % 30 == 0:
            lr *= 0.5
            print(f"    LR decayed to {lr:.6f}")

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            score = metrics["macro_f1"] + metrics["min_recall"]
            if score > best_score:
                best_score = score
                best_W = [w.copy() for w in model.W]
                best_b = [b.copy() for b in model.b]
            print(f"  epoch {epoch:3d}  loss={epoch_loss / n_batches:.4f}  "
                  f"acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"min_rec={metrics['min_recall']:.4f}  score={score:.4f}  "
                  f"best={best_score:.4f}")

    # Restore best
    model.W = best_W
    model.b = best_b
    print(f"\n  Best score (macro_f1 + min_recall): {best_score:.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────
# 5.  Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Compute accuracy, per-class precision/recall/f1, macro_f1, min_recall."""
    acc = (y_true == y_pred).mean()
    per_class = {}
    recalls = []
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {"precision": prec, "recall": rec, "f1": f1,
                                     "support": int((y_true == c).sum())}
        recalls.append(rec)
        f1s.append(f1)
    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1s)),
        "min_recall": float(min(recalls)),
        "per_class": per_class,
    }


def print_report(y_true, y_pred, title=""):
    """Print a classification report."""
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
    print(f"  {'Min Recall':<12s} {'':>8s} {'':>8s} {metrics['min_recall']:8.4f}")
    return metrics


# ──────────────────────────────────────────────────────────────────────
# 6.  Early-Exit Head
# ──────────────────────────────────────────────────────────────────────

def train_exit_head(model, X_train_n, y_train, X_test_n, y_test,
                    alpha, epochs=30, lr=0.01, batch_size=256):
    """Train a linear exit head (64->4) on hidden layer 0 output."""
    print("\n  Training early-exit head...")

    H_train = model.get_hidden(X_train_n, 0)  # (N, 64)
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
    """Calibrate exit threshold on correctly-classified test samples."""
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
    print(f"    Expected early-exit rate: ~{100 - percentile}%")

    return threshold_int


# ──────────────────────────────────────────────────────────────────────
# 7.  Ternary Quantization
# ──────────────────────────────────────────────────────────────────────

def ternary_quantize_layer(W, threshold_ratio=0.7):
    """
    Ternary quantization: signs in {-1, 0, +1}, alpha = mean(|W[large]|).
    alpha_shift = round(16 + log2(alpha)).
    """
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
    effective_alpha = 2.0 ** alpha_shift / SCALE

    signs = np.zeros_like(W, dtype=np.int64)
    signs[mask_pos] = 1
    signs[mask_neg] = -1

    n_zero = (signs == 0).sum()
    sparsity = n_zero / signs.size * 100

    print(f"    threshold={threshold:.6f}, alpha={alpha_float:.6f}, "
          f"alpha_shift={alpha_shift} (eff_alpha={effective_alpha:.4f}), "
          f"sparsity={sparsity:.1f}%")

    return signs, alpha_shift, alpha_float


# ──────────────────────────────────────────────────────────────────────
# 8.  MapLUT Generation
# ──────────────────────────────────────────────────────────────────────

def compute_lut_params(X_train_raw, num_bins=NUM_BINS):
    """
    Compute per-feature binning parameters from raw training data.
    Returns feat_offset (int32), feat_shift (uint32).
    """
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
        # shift such that (raw - offset) >> shift ∈ [0, num_bins)
        # We need (feat_range) >> shift <= num_bins - 1
        # So shift = ceil(log2(feat_range / num_bins))
        raw_shift = math.log2(feat_range / num_bins) if feat_range > num_bins else 0
        feat_shift[j] = max(int(math.ceil(raw_shift)), 0)

    return feat_offset.astype(np.int32), feat_shift.astype(np.uint32)


def generate_lut(W0, b0, mu, sigma, feat_offset, feat_shift, num_bins=NUM_BINS):
    """
    Generate Layer 0 MapLUT: lut_data[j][b][i] for j features, b bins, i neurons.
    Also returns bias_layer_0_q (int32, scaled by SCALE).
    """
    s = SCALE
    lut_data = np.zeros((NUM_FEATURES, num_bins, HIDDEN_SIZE), dtype=np.int32)

    for j in range(NUM_FEATURES):
        bin_width = 1 << int(feat_shift[j])  # width of each bin in raw units
        for b in range(num_bins):
            # Left edge of bin b in raw feature space
            # Using left edge instead of center: for right-skewed network
            # features, most data falls in bin 0, where the center (0.5*width)
            # is far from typical values (near 0). Left edge is more accurate.
            center_raw = float(feat_offset[j]) + b * bin_width
            # Normalize
            center_norm = (center_raw - mu[j]) / sigma[j]
            # Partial sum for each hidden neuron
            for i in range(HIDDEN_SIZE):
                lut_data[j, b, i] = int(np.round(W0[i, j] * center_norm * s))

    bias_layer_0_q = np.round(b0 * s).astype(np.int32)
    return lut_data, bias_layer_0_q


# ──────────────────────────────────────────────────────────────────────
# 9.  Inference Engines
# ──────────────────────────────────────────────────────────────────────

def float_inference(X_norm, model):
    """Mode 1: Standard float MLP forward pass."""
    return model.predict(X_norm)


def int32_inference(X_raw, model, mu, sigma):
    """Mode 2: All layers quantized with scale=2^16, integer arithmetic."""
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


def ternary_int_inference(X_raw, model, mu, sigma,
                          ternary_signs, alpha_shifts):
    """Mode 3: Layer 0 full int32, layers 1+2 ternary."""
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


def maplut_inference(X_raw, lut_data, bias_layer_0_q,
                     feat_offset, feat_shift,
                     exit_signs, exit_alpha_shift, exit_bias_q,
                     ternary_signs, alpha_shifts, ternary_biases,
                     exit_threshold=None, num_bins=NUM_BINS):
    """
    Mode 4: MapLUT-Int — zero-multiply inference.
      Layer 0:    MapLUT (lookup + add)
      Exit head:  TernaryShift
      Layer 1:    TernaryShift [64->64]
      Layer 2:    TernaryShift [64->4]
    Returns predictions (and early_exit_count if threshold given).
    """
    sb = SCALE_BITS
    N = X_raw.shape[0]

    # Layer 0: MapLUT accumulation
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))  # (N, H)
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = raw - int(feat_offset[j])
        shifted = np.maximum(shifted, 0)
        bins = (shifted >> int(feat_shift[j])).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        # Vectorized: gather lut_data[j][bins[n]] for each sample n
        hidden += lut_data[j][bins]  # (N, H) — ADD only, zero multiply

    # ReLU
    hidden = np.maximum(hidden, 0)

    # Exit head: ternary on hidden
    exit_signs_64 = exit_signs.astype(np.int64)
    exit_acc = hidden @ exit_signs_64.T
    exit_logits = ((exit_acc << exit_alpha_shift) >> sb) + exit_bias_q.astype(np.int64)

    # Compute early exit stats
    early_exit_count = 0
    preds = np.full(N, -1, dtype=np.int64)
    if exit_threshold is not None:
        exit_pred = np.argmax(exit_logits, axis=1)
        sorted_el = np.sort(exit_logits, axis=1)
        margins = sorted_el[:, -1] - sorted_el[:, -2]
        confident = margins >= exit_threshold
        preds[confident] = exit_pred[confident]
        early_exit_count = int(confident.sum())

    # Full pipeline for remaining (or all if no threshold)
    need_full = preds == -1
    if need_full.any():
        x = hidden[need_full].copy()

        # Layer 1: ternary [64->64]
        signs1 = ternary_signs[0].astype(np.int64)
        ashift1 = alpha_shifts[0]
        bq1 = ternary_biases[0].astype(np.int64)
        acc1 = x @ signs1.T
        z1 = ((acc1 << ashift1) >> sb) + bq1
        z1 = np.maximum(z1, 0)

        # Layer 2: ternary [64->4]
        signs2 = ternary_signs[1].astype(np.int64)
        ashift2 = alpha_shifts[1]
        bq2 = ternary_biases[1].astype(np.int64)
        acc2 = z1 @ signs2.T
        z2 = ((acc2 << ashift2) >> sb) + bq2

        full_pred = np.argmax(z2, axis=1)
        preds[need_full] = full_pred

    return preds, early_exit_count


# ──────────────────────────────────────────────────────────────────────
# 10. Header Generation
# ──────────────────────────────────────────────────────────────────────

def pack_ternary_row(signs_row, input_dim_padded):
    """Pack a row of ternary signs into uint32 words (16 per word, 2 bits each)."""
    n_words = input_dim_padded // 16
    words = []
    for w_idx in range(n_words):
        word = 0
        for k in range(16):
            idx = w_idx * 16 + k
            if idx < len(signs_row):
                s = signs_row[idx]
            else:
                s = 0
            if s > 0:
                bits = 0b01  # TERNARY_POS
            elif s < 0:
                bits = 0b10  # TERNARY_NEG
            else:
                bits = 0b00  # TERNARY_ZERO
            word |= (bits << (k * 2))
        words.append(word)
    return words


def c_array_2d_s32(name, arr):
    rows, cols = arr.shape
    lines = [f"static const __s32 {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(f"{int(v)}" for v in arr[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{ {vals} }}{comma}")
    lines.append("};")
    return "\n".join(lines)


def c_array_1d_s32(name, arr):
    vals = ", ".join(f"{int(v)}" for v in arr)
    return f"static const __s32 {name}[{len(arr)}] = {{ {vals} }};"


def c_array_1d_u32(name, arr):
    vals = ", ".join(f"{int(v)}" for v in arr)
    return f"static const __u32 {name}[{len(arr)}] = {{ {vals} }};"


def c_array_2d_u32(name, arr_2d):
    rows = len(arr_2d)
    cols = len(arr_2d[0])
    lines = [f"static const __u32 {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(f"0x{v:08X}" for v in arr_2d[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{ {vals} }}{comma}  /* neuron {r:2d} */")
    lines.append("};")
    return "\n".join(lines)


def generate_v3_header(model, bias_layer_0_q, feat_offset, feat_shift,
                       exit_signs, exit_alpha_shift, exit_bias_q,
                       ternary_signs, alpha_shifts,
                       exit_threshold, layer_sizes):
    """Generate include/model_params_v3.h."""
    s = SCALE
    sb = SCALE_BITS

    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("/*")
    lines.append(" * model_params_v3.h — Auto-generated by train_v3.py")
    lines.append(f" * Architecture: {layer_sizes}")
    lines.append(" * Zero-multiply pipeline: MapLUT + TernaryShift + EarlyExit")
    lines.append(f" * Scale: 2^{sb} = {s}")
    lines.append(" */")
    lines.append("")
    lines.append("#include <linux/types.h>")
    lines.append("")

    # Architecture constants
    lines.append(f"#define NUM_FEATURES        {NUM_FEATURES}")
    lines.append(f"#define HIDDEN_SIZE         {HIDDEN_SIZE}")
    lines.append(f"#define NUM_CLASSES          {NUM_CLASSES}")
    lines.append(f"#define SCALE_FACTOR_BITS   {sb}")
    lines.append(f"#define SCALE_FACTOR        {s}")
    lines.append(f"#define NUM_BINS            {NUM_BINS}")
    lines.append(f"#define EARLY_EXIT_THRESHOLD  {exit_threshold}")
    lines.append("")

    # Class labels
    lines.append("#define CLASS_BENIGN       0")
    lines.append("#define CLASS_DDOS         1")
    lines.append("#define CLASS_PORTSCAN     2")
    lines.append("#define CLASS_BRUTEFORCE   3")
    lines.append("")

    # Ternary encoding
    lines.append("#define TERNARY_ZERO   0")
    lines.append("#define TERNARY_POS    1")
    lines.append("#define TERNARY_NEG    2")
    lines.append("")

    # Per-feature quantization for MapLUT
    lines.append("/* Per-feature quantization for MapLUT binning */")
    lines.append(c_array_1d_s32("feat_offset", feat_offset))
    lines.append("")
    lines.append(c_array_1d_u32("feat_shift", feat_shift))
    lines.append("")

    # Layer 0 bias
    lines.append(f"/* Layer 0 bias [{HIDDEN_SIZE}] — added before LUT accumulation */")
    lines.append(c_array_1d_s32("bias_layer_0", bias_layer_0_q))
    lines.append("")

    # Exit head: ternary
    exit_in = exit_signs.shape[1]
    exit_in_padded = ((exit_in + 15) // 16) * 16
    n_words_exit = exit_in_padded // 16

    if exit_in_padded > exit_in:
        exit_signs_padded = np.zeros((NUM_CLASSES, exit_in_padded),
                                     dtype=exit_signs.dtype)
        exit_signs_padded[:, :exit_in] = exit_signs
    else:
        exit_signs_padded = exit_signs

    exit_packed = []
    for i in range(NUM_CLASSES):
        exit_packed.append(pack_ternary_row(exit_signs_padded[i], exit_in_padded))

    n_zero_exit = (exit_signs == 0).sum()
    sp_exit = n_zero_exit / exit_signs.size * 100

    lines.append(f"/* Exit Head: Ternary [{exit_in} -> {NUM_CLASSES}] */")
    lines.append(f"/* alpha = 2^{exit_alpha_shift}, sparsity = {sp_exit:.1f}% */")
    lines.append(f"#define EXIT_ALPHA_SHIFT  {exit_alpha_shift}")
    lines.append("")
    lines.append(c_array_2d_u32("exit_packed", exit_packed))
    lines.append("")
    lines.append(c_array_1d_s32("exit_bias", exit_bias_q))
    lines.append("")

    # Ternary layers 1, 2
    for k_rel, k_abs in enumerate([1, 2]):
        signs = ternary_signs[k_rel]
        ashift = alpha_shifts[k_rel]
        bias = np.round(model.b[k_abs] * s).astype(np.int32)

        out_size = signs.shape[0]
        in_size = signs.shape[1]
        in_padded = ((in_size + 15) // 16) * 16
        n_words = in_padded // 16

        if in_padded > in_size:
            signs_padded = np.zeros((out_size, in_padded), dtype=signs.dtype)
            signs_padded[:, :in_size] = signs
        else:
            signs_padded = signs

        packed = []
        for i in range(out_size):
            packed.append(pack_ternary_row(signs_padded[i], in_padded))

        n_zero = (signs == 0).sum()
        sparsity = n_zero / signs.size * 100

        lines.append(f"/* Layer {k_abs}: Ternary [{in_size} -> {out_size}] */")
        lines.append(f"/* alpha = 2^{ashift}, sparsity = {sparsity:.1f}% */")
        lines.append(f"#define ALPHA_SHIFT_LAYER_{k_abs}  {ashift}")
        lines.append("")
        lines.append(c_array_2d_u32(f"packed_w{k_abs}", packed))
        lines.append("")
        lines.append(c_array_1d_s32(f"bias_layer_{k_abs}", bias))
        lines.append("")

    header_text = "\n".join(lines)
    os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)
    with open(HEADER_PATH, "w") as f:
        f.write(header_text)
    print(f"\n  Header written to {HEADER_PATH} ({len(header_text)} bytes)")
    return header_text


def write_lut_binary(lut_data, path):
    """Write LUT as binary file: int32 lut_data[12][64][64]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        # Layout: lut_data[j][b][i] as int32, j=0..11, b=0..63, i=0..63
        for j in range(lut_data.shape[0]):
            for b in range(lut_data.shape[1]):
                for i in range(lut_data.shape[2]):
                    f.write(struct.pack("<i", int(lut_data[j, b, i])))
    size = os.path.getsize(path)
    print(f"  LUT binary written to {path} ({size} bytes = {size/1024:.1f} KB)")


# ──────────────────────────────────────────────────────────────────────
# 11. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES]

    # ── Step 1: Load data ──
    print("=" * 60)
    print("Step 1: Loading CIC-IDS-2017 dataset")
    print("=" * 60)
    X_all, y_all = load_data()

    # Shuffle and split 75/25
    idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]
    split = int(0.75 * len(y_all))
    X_train_raw, y_train = X_all[:split], y_all[:split]
    X_test_raw, y_test = X_all[split:], y_all[split:]
    print(f"\n  Train: {len(y_train)}, Test: {len(y_test)}")

    # Compute normalization stats on training data (before oversampling)
    mu = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma[sigma < 1e-8] = 1e-8
    print(f"  Feature means: {np.round(mu, 2)}")
    print(f"  Feature stds:  {np.round(sigma, 2)}")

    # Normalize
    X_train_n = (X_train_raw - mu) / sigma
    X_test_n = (X_test_raw - mu) / sigma

    # ── Step 2: Oversample ──
    print("\n" + "=" * 60)
    print("Step 2: Oversampling to 100K per class")
    print("=" * 60)
    X_train_os, y_train_os = oversample(X_train_n, y_train,
                                         target=OVERSAMPLE_TARGET)
    print(f"  Oversampled: {len(y_train_os)} total")
    for c in range(NUM_CLASSES):
        print(f"    Class {c}: {(y_train_os == c).sum()}")

    # ── Step 3: Train MLP ──
    print("\n" + "=" * 60)
    print(f"Step 3: Training MLP {layer_sizes}")
    print("=" * 60)
    model = train_mlp(X_train_os, y_train_os, X_test_n, y_test,
                      layer_sizes, epochs=100, batch_size=256,
                      lr=0.01, gamma=3.0)

    # ── Step 4: Float evaluation ──
    print("\n" + "=" * 60)
    print("Step 4: Float model evaluation")
    print("=" * 60)
    y_pred_float = float_inference(X_test_n, model)
    m_float = print_report(y_test, y_pred_float, "Float Model")

    # ── Step 5: Early-exit head training ──
    print("\n" + "=" * 60)
    print("Step 5: Training early-exit head")
    print("=" * 60)
    alpha = compute_focal_alpha(y_train_os)
    exit_W, exit_b = train_exit_head(model, X_train_os, y_train_os,
                                      X_test_n, y_test, alpha)
    exit_threshold = calibrate_exit_threshold(model, exit_W, exit_b,
                                               X_test_n, y_test,
                                               percentile=15)

    # ── Step 6: Ternary quantization (layers 1, 2, exit head) ──
    print("\n" + "=" * 60)
    print("Step 6: Ternary quantization (layers 1, 2, exit head)")
    print("=" * 60)
    ternary_signs = []
    alpha_shifts = []
    for k_abs in [1, 2]:
        print(f"  Layer {k_abs} ({model.W[k_abs].shape}):")
        signs, ashift, _ = ternary_quantize_layer(model.W[k_abs])
        ternary_signs.append(signs)
        alpha_shifts.append(ashift)

    print(f"  Exit head ({exit_W.shape}):")
    exit_signs, exit_alpha_shift, _ = ternary_quantize_layer(exit_W)
    exit_bias_q = np.round(exit_b * SCALE).astype(np.int32)

    # Quantized biases for ternary layers
    ternary_biases = [
        np.round(model.b[1] * SCALE).astype(np.int32),
        np.round(model.b[2] * SCALE).astype(np.int32),
    ]

    # ── Step 7: MapLUT generation ──
    print("\n" + "=" * 60)
    print("Step 7: MapLUT generation (Layer 0)")
    print("=" * 60)
    feat_offset, feat_shift = compute_lut_params(X_train_raw, NUM_BINS)
    lut_data, bias_layer_0_q = generate_lut(
        model.W[0], model.b[0], mu, sigma, feat_offset, feat_shift, NUM_BINS)

    print(f"  feat_offset: {feat_offset.tolist()}")
    print(f"  feat_shift:  {feat_shift.tolist()}")
    print(f"  LUT shape:   {lut_data.shape}")
    lut_mem = lut_data.nbytes
    print(f"  LUT memory:  {lut_mem} bytes ({lut_mem/1024:.1f} KB)")

    # ── Step 8: Evaluate all 4 inference modes ──
    print("\n" + "=" * 60)
    print("Step 8: Accuracy comparison — all 4 inference modes")
    print("=" * 60)

    # Mode 1: Float
    print("\n  --- Mode 1: Float ---")
    y_float = float_inference(X_test_n, model)
    m_float = print_report(y_test, y_float, "Float")

    # Mode 2: Int32
    print("\n  --- Mode 2: Int32 ---")
    y_int32 = int32_inference(X_test_raw, model, mu, sigma)
    m_int32 = print_report(y_test, y_int32, "Int32")

    # Mode 3: Ternary-Int
    print("\n  --- Mode 3: Ternary-Int ---")
    y_ternary = ternary_int_inference(X_test_raw, model, mu, sigma,
                                       ternary_signs, alpha_shifts)
    m_ternary = print_report(y_test, y_ternary, "Ternary-Int")

    # Mode 4: MapLUT-Int
    print("\n  --- Mode 4: MapLUT-Int (zero multiply) ---")
    y_maplut, exit_cnt = maplut_inference(
        X_test_raw, lut_data, bias_layer_0_q,
        feat_offset, feat_shift,
        exit_signs, exit_alpha_shift, exit_bias_q,
        ternary_signs, alpha_shifts, ternary_biases,
        exit_threshold=exit_threshold, num_bins=NUM_BINS)
    m_maplut = print_report(y_test, y_maplut, "MapLUT-Int")
    early_exit_rate = exit_cnt / len(y_test)
    print(f"  Early exit: {exit_cnt}/{len(y_test)} = {early_exit_rate:.2%}")

    # Summary table
    print("\n  " + "=" * 55)
    print(f"  {'Mode':<18s} {'Accuracy':>10s} {'Macro F1':>10s} {'Min Rec':>10s}")
    print("  " + "-" * 55)
    for name, m in [("Float", m_float), ("Int32", m_int32),
                    ("Ternary-Int", m_ternary), ("MapLUT-Int", m_maplut)]:
        print(f"  {name:<18s} {m['accuracy']:10.4f} {m['macro_f1']:10.4f} "
              f"{m['min_recall']:10.4f}")
    print("  " + "=" * 55)

    # ── Step 9: Bins vs accuracy sweep ──
    print("\n" + "=" * 60)
    print("Step 9: Bins vs accuracy sweep")
    print("=" * 60)
    bins_vs_accuracy = []
    for nb in [16, 32, 64, 128, 256]:
        fo_nb, fs_nb = compute_lut_params(X_train_raw, nb)
        lut_nb, blq_nb = generate_lut(
            model.W[0], model.b[0], mu, sigma, fo_nb, fs_nb, nb)
        y_nb, _ = maplut_inference(
            X_test_raw, lut_nb, blq_nb,
            fo_nb, fs_nb,
            exit_signs, exit_alpha_shift, exit_bias_q,
            ternary_signs, alpha_shifts, ternary_biases,
            exit_threshold=None, num_bins=nb)
        acc_nb = float((y_nb == y_test).mean())
        mem_nb = nb * NUM_FEATURES * HIDDEN_SIZE * 4
        bins_vs_accuracy.append({
            "bins": nb,
            "accuracy": acc_nb,
            "lut_memory_bytes": mem_nb,
        })
        print(f"    bins={nb:4d}  acc={acc_nb:.4f}  mem={mem_nb/1024:.0f} KB")

    # ── Step 10: Generate header ──
    print("\n" + "=" * 60)
    print("Step 10: Generating C header + LUT binary")
    print("=" * 60)
    generate_v3_header(model, bias_layer_0_q, feat_offset, feat_shift,
                       exit_signs, exit_alpha_shift, exit_bias_q,
                       ternary_signs, alpha_shifts,
                       exit_threshold, layer_sizes)

    write_lut_binary(lut_data, str(LUT_BIN_PATH))

    # Memory accounting
    mem_lut = lut_mem
    mem_feat = NUM_FEATURES * 4 * 2  # feat_offset + feat_shift
    mem_bias0 = HIDDEN_SIZE * 4
    mem_exit = NUM_CLASSES * 4 * 4 + NUM_CLASSES * 4  # packed + bias
    mem_ternary = 0
    for k_rel in range(2):
        signs = ternary_signs[k_rel]
        out_size = signs.shape[0]
        in_padded = ((signs.shape[1] + 15) // 16) * 16
        n_words = in_padded // 16
        mem_ternary += out_size * n_words * 4 + out_size * 4
    total_mem = mem_feat + mem_bias0 + mem_exit + mem_ternary
    total_mem_with_lut = total_mem + mem_lut
    print(f"\n  Model params memory (header): {total_mem} bytes ({total_mem/1024:.1f} KB)")
    print(f"  LUT memory:                   {mem_lut} bytes ({mem_lut/1024:.1f} KB)")
    print(f"  Total (params + LUT):         {total_mem_with_lut} bytes "
          f"({total_mem_with_lut/1024:.1f} KB)")

    # ── Step 11: Save results JSON ──
    print("\n" + "=" * 60)
    print("Step 11: Saving results")
    print("=" * 60)
    results = {
        "architecture": layer_sizes,
        "num_bins": NUM_BINS,
        "lut_memory_bytes": int(mem_lut),
        "model_params_bytes": int(total_mem),
        "total_memory_bytes": int(total_mem_with_lut),
        "float": m_float,
        "int32": m_int32,
        "ternary_int": m_ternary,
        "maplut_int": m_maplut,
        "early_exit_rate": float(early_exit_rate),
        "early_exit_threshold": exit_threshold,
        "exit_alpha_shift": exit_alpha_shift,
        "alpha_shifts": alpha_shifts,
        "bins_vs_accuracy": bins_vs_accuracy,
        "train_samples": int(len(y_train_os)),
        "test_samples": int(len(y_test)),
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results written to {RESULTS_PATH}")

    # ── Step 12: Verify eBPF compilation ──
    print("\n" + "=" * 60)
    print("Step 12: Verifying eBPF compilation")
    print("=" * 60)
    ebpf_src = PROJECT_DIR / "src" / "xdp_ml_v3.c"
    ebpf_out = "/tmp/xdp_ml_v3.o"
    if ebpf_src.exists():
        compile_cmd = (
            f"clang -O2 -g -target bpf -D__TARGET_ARCH_x86 "
            f"-I/usr/include/x86_64-linux-gnu "
            f"-c {ebpf_src} -o {ebpf_out}"
        )
        print(f"  {compile_cmd}")
        ret = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        if ret.returncode == 0:
            print("  ✓ Compilation succeeded!")
        else:
            print(f"  ✗ Compilation failed!")
            print(f"    stderr: {ret.stderr[:500]}")
    else:
        print(f"  {ebpf_src} not found, skipping compilation check.")

    print("\n" + "=" * 60)
    print("Done!  V3 zero-multiply pipeline artifacts generated.")
    print(f"  Header:  {HEADER_PATH}")
    print(f"  LUT:     {LUT_BIN_PATH}")
    print(f"  Results: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
