#!/usr/bin/env python3
"""
train_final.py — Train the best CIC-IDS-2017 model configuration
(h=64, L=2, γ=3.0, oversample=100K) and generate a C header for
eBPF deployment with ternary weight compression.

Dependencies: numpy, pyarrow
Usage:  python3 scripts/train_final.py
"""

import json
import os
import sys
import subprocess
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
OVERSAMPLE_TARGET = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "cicids_raw"
HEADER_PATH = PROJECT_DIR / "include" / "model_params_v2.h"
RESULTS_PATH = PROJECT_DIR / "results" / "final_model.json"


# ──────────────────────────────────────────────────────────────────────
# 2.  Data loading
# ──────────────────────────────────────────────────────────────────────

def map_label(label_str):
    """Map a label string to class index. Returns -1 to skip."""
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    # Fallback heuristics
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

        # Extract feature columns as numpy arrays
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
            # Skip NaN/Inf/None
            if np.any(np.isnan(row)) or np.any(np.isinf(row)):
                continue
            all_X.append(row)
            all_y.append(cls)
            kept += 1

        print(f"    {fpath.name}: {n_rows} rows → {kept} kept")

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
    """Balance classes to target count with Gaussian noise augmentation."""
    Xs, ys = [], []
    for c in range(NUM_CLASSES):
        mask = y == c
        Xc = X[mask]
        nc = len(Xc)
        if nc == 0:
            print(f"  WARNING: class {c} has 0 samples!")
            continue
        if nc >= target:
            # Subsample
            idx = np.random.choice(nc, target, replace=False)
            Xs.append(Xc[idx])
        else:
            # Duplicate with noise
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
        print(f"    Class {c}: {nc} → {target}")

    X_out = np.concatenate(Xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    # Shuffle
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
    alpha = alpha / alpha.mean()  # normalize to mean=1
    return alpha


def focal_backward(probs, y, alpha, gamma):
    """
    Focal loss gradient: FL = -alpha * (1-p_t)^gamma * log(p_t)
    Returns dz for softmax output layer.
    """
    N = len(y)
    p_t = probs[np.arange(N), y]
    p_t = np.clip(p_t, 1e-12, 1.0)

    # Focal weight per sample
    focal_w = alpha[y] * (1.0 - p_t) ** gamma

    # Gradient: for softmax + focal CE
    # dL/dz_j = focal_w * (p_j - 1{j=y}) * [gamma*(1-p_t)^(gamma-1)*p_t*log(p_t) + (1-p_t)^gamma]
    # Simplified practical gradient (standard approach):
    # dz = alpha_i * (1-p_t)^gamma * (p - one_hot)
    # This is the commonly-used approximation for focal loss backprop
    dz = probs.copy()
    dz[np.arange(N), y] -= 1.0
    dz *= focal_w[:, None]
    dz /= N

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
    n = len(y_true)
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
    """Train a linear exit head (64→4) on hidden layer 0 output."""
    print("\n  Training early-exit head...")

    # Get hidden activations from layer 0
    H_train = model.get_hidden(X_train_n, 0)  # (N, 64)
    H_test = model.get_hidden(X_test_n, 0)

    # Initialize exit head
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

            # Forward
            z = Hb @ exit_W.T + exit_b
            probs = softmax(z)

            # Focal-weighted CE gradient
            p_t = probs[np.arange(nb), yb]
            p_t = np.clip(p_t, 1e-12, 1.0)
            focal_w = alpha[yb] * (1.0 - p_t) ** 1.0  # lighter focal for exit head

            dz = probs.copy()
            dz[np.arange(nb), yb] -= 1.0
            dz *= focal_w[:, None]
            dz /= nb

            # Update
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
    """
    Calibrate exit threshold at the given percentile of logit margins
    on correctly-classified test samples.
    """
    H_test = model.get_hidden(X_test_n, 0)
    z = H_test @ exit_W.T + exit_b  # float logits

    pred = np.argmax(z, axis=1)
    correct_mask = pred == y_test

    # Compute logit margins on correct samples
    z_correct = z[correct_mask]
    sorted_logits = np.sort(z_correct, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]

    threshold_float = np.percentile(margins, percentile)

    # Convert to int32 domain (scale by SCALE)
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
    Apply ternary quantization to a weight matrix.
    Returns: signs (int array), alpha_shift (int), alpha_float

    alpha_shift is chosen so that (acc << alpha_shift) >> 16 ≈ acc * alpha,
    i.e. alpha_shift = round(16 + log2(alpha)).
    """
    abs_W = np.abs(W)
    threshold = threshold_ratio * abs_W.mean()
    mask_pos = W > threshold
    mask_neg = W < -threshold

    # Alpha = mean of |W| for entries exceeding threshold
    large_entries = abs_W[mask_pos | mask_neg]
    if len(large_entries) == 0:
        alpha_float = 1.0
    else:
        alpha_float = large_entries.mean()

    # Round alpha*SCALE to nearest power of 2 → alpha_shift
    # Formula: (acc << alpha_shift) >> 16 ≈ acc * alpha
    # So 2^alpha_shift / 2^16 = alpha  →  alpha_shift = 16 + log2(alpha)
    alpha_shift = int(np.round(SCALE_BITS + np.log2(alpha_float)))
    alpha_shift = max(alpha_shift, 0)  # ensure non-negative
    effective_alpha = 2.0 ** alpha_shift / SCALE

    # Build ternary signs
    signs = np.zeros_like(W, dtype=np.int64)
    signs[mask_pos] = 1
    signs[mask_neg] = -1

    # Sparsity
    n_zero = (signs == 0).sum()
    sparsity = n_zero / signs.size * 100

    print(f"    threshold={threshold:.6f}, alpha={alpha_float:.6f}, "
          f"alpha_shift={alpha_shift} (eff_alpha={effective_alpha:.4f}), "
          f"sparsity={sparsity:.1f}%")

    return signs, alpha_shift, alpha_float


# ──────────────────────────────────────────────────────────────────────
# 8.  Integer Inference Engines
# ──────────────────────────────────────────────────────────────────────

def int32_inference(X, model, mean, std):
    """Full int32 inference (all layers quantized with scale=2^16)."""
    s = SCALE
    sb = SCALE_BITS

    # Quantize all weights/biases
    Wq = [np.round(w * s).astype(np.int64) for w in model.W]
    Bq = [np.round(b * s).astype(np.int64) for b in model.b]

    # Normalize and quantize input
    X_norm = (X - mean) / std
    xq = np.round(X_norm * s).astype(np.int64)

    x = xq
    for k in range(len(Wq)):
        z = (x @ Wq[k].T >> sb) + Bq[k]
        if k < len(Wq) - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def ternary_int_inference(X, model, mean, std, ternary_signs, alpha_shifts):
    """
    Ternary-Int inference: layer 0 full int32, layers 1+2 ternary.

    For ternary layers:
      acc = x @ signs.T          (add/sub only, no multiply)
      z   = (acc << alpha_shift) >> 16  (scale by alpha)
      z  += Bq                   (add bias, still at scale s)
    """
    s = SCALE
    sb = SCALE_BITS

    # Layer 0: full int32
    Wq0 = np.round(model.W[0] * s).astype(np.int64)
    Bq0 = np.round(model.b[0] * s).astype(np.int64)

    X_norm = (X - mean) / std
    xq = np.round(X_norm * s).astype(np.int64)

    z = (xq @ Wq0.T >> sb) + Bq0
    z = np.maximum(z, 0)
    x = z

    # Ternary layers (1, 2)
    for k_rel, k_abs in enumerate([1, 2]):
        signs = ternary_signs[k_rel].astype(np.int64)
        ashift = alpha_shifts[k_rel]
        Bq = np.round(model.b[k_abs] * s).astype(np.int64)

        # acc = x @ signs.T  (add/sub only)
        acc = x @ signs.T
        # Scale: (acc << alpha_shift) >> 16 ≈ acc * alpha
        z = (acc << ashift) >> sb
        z += Bq
        if k_abs < len(model.W) - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


# ──────────────────────────────────────────────────────────────────────
# 9.  Header Generation
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
    """Format 2D int32 array as C."""
    rows, cols = arr.shape
    lines = [f"static const __s32 {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(f"{int(v)}" for v in arr[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{ {vals} }}{comma}")
    lines.append("};")
    return "\n".join(lines)


def c_array_1d_s32(name, arr):
    """Format 1D int32 array as C."""
    vals = ", ".join(f"{int(v)}" for v in arr)
    return f"static const __s32 {name}[{len(arr)}] = {{ {vals} }};"


def c_array_2d_u32(name, arr_2d):
    """Format 2D list-of-lists of uint32 as C."""
    rows = len(arr_2d)
    cols = len(arr_2d[0])
    lines = [f"static const __u32 {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(f"0x{v:08X}" for v in arr_2d[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{ {vals} }}{comma}  /* neuron {r:2d} */")
    lines.append("};")
    return "\n".join(lines)


def generate_header(model, mean_raw, std_raw, exit_W, exit_b,
                    ternary_signs, alpha_shifts, exit_threshold,
                    layer_sizes):
    """Generate the model_params_v2.h header file."""
    s = SCALE
    sb = SCALE_BITS

    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("/*")
    lines.append(" * model_params_v2.h — Auto-generated by train_final.py")
    lines.append(f" * Architecture: {layer_sizes}")
    lines.append(" * Three innovations: Fused Norm, Ternary Weights, Early-Exit")
    lines.append(f" * Scale: 2^{sb} = {s}")
    lines.append(" */")
    lines.append("")
    lines.append("#include <linux/types.h>")
    lines.append("")

    # Architecture constants
    lines.append(f"#define NUM_FEATURES        {NUM_FEATURES}")
    lines.append(f"#define HIDDEN_SIZE         {HIDDEN_SIZE}")
    lines.append(f"#define NUM_CLASSES          {NUM_CLASSES}")
    lines.append(f"#define NUM_LAYERS           {len(layer_sizes) - 1}")
    lines.append(f"#define SCALE_FACTOR_BITS   {sb}")
    lines.append(f"#define SCALE_FACTOR        {s}")
    lines.append(f"#define FUSED_SCALE_BITS    32")
    lines.append(f"#define EARLY_EXIT_THRESHOLD  {exit_threshold}")
    lines.append("")
    lines.append("#define CLASS_BENIGN       0")
    lines.append("#define CLASS_DDOS         1")
    lines.append("#define CLASS_PORTSCAN     2")
    lines.append("#define CLASS_BRUTEFORCE   3")
    lines.append("")
    lines.append("#define TERNARY_ZERO   0")
    lines.append("#define TERNARY_POS    1")
    lines.append("#define TERNARY_NEG    2")
    lines.append("")

    # ── Layer 0: Fused Norm-MatMul-ReLU ──
    # fused_w0[i][j] = round(W0[i][j] * s) * round(s / sigma[j]) // s
    W0 = model.W[0]  # (64, 12)
    b0 = model.b[0]  # (64,)

    norm_scale = np.round(s / std_raw)  # round(s / sigma[j])
    Wq0 = np.round(W0 * s)  # round(W0[i][j] * s)

    fused_w0 = np.zeros((HIDDEN_SIZE, NUM_FEATURES), dtype=np.int64)
    for i in range(HIDDEN_SIZE):
        for j in range(NUM_FEATURES):
            fused_w0[i, j] = int(Wq0[i, j]) * int(norm_scale[j]) // s

    # fused_offset[i] = sum_j(fused_w0[i][j] * round(mu[j])) >> 16
    mu_q = np.round(mean_raw)  # raw mean rounded
    fused_offset = np.zeros(HIDDEN_SIZE, dtype=np.int64)
    for i in range(HIDDEN_SIZE):
        acc = 0
        for j in range(NUM_FEATURES):
            acc += int(fused_w0[i, j]) * int(mu_q[j])
        fused_offset[i] = acc >> sb

    bias_layer_0 = np.round(b0 * s).astype(np.int64)

    lines.append(f"/* Layer 0: Fused Norm-MatMul-ReLU [{NUM_FEATURES} → {HIDDEN_SIZE}] */")
    lines.append(c_array_2d_s32("fused_w0", fused_w0.astype(np.int32)))
    lines.append("")
    lines.append(c_array_1d_s32("fused_offset", fused_offset.astype(np.int32)))
    lines.append("")
    lines.append(c_array_1d_s32("bias_layer_0", bias_layer_0.astype(np.int32)))
    lines.append("")

    # ── Early-Exit Head ──
    exit_Wq = np.round(exit_W * s).astype(np.int32)
    exit_bq = np.round(exit_b * s).astype(np.int32)
    lines.append(f"/* Early-Exit Head [{HIDDEN_SIZE} → {NUM_CLASSES}] */")
    lines.append(c_array_2d_s32("exit_weight", exit_Wq))
    lines.append("")
    lines.append(c_array_1d_s32("exit_bias", exit_bq))
    lines.append("")

    # ── Ternary Layers ──
    for k_rel, k_abs in enumerate([1, 2]):
        signs = ternary_signs[k_rel]
        ashift = alpha_shifts[k_rel]
        bias = np.round(model.b[k_abs] * s).astype(np.int32)

        out_size = signs.shape[0]
        in_size = signs.shape[1]
        # Pad input to next multiple of 16
        in_padded = ((in_size + 15) // 16) * 16
        n_words = in_padded // 16

        # Pad signs
        if in_padded > in_size:
            signs_padded = np.zeros((out_size, in_padded), dtype=signs.dtype)
            signs_padded[:, :in_size] = signs
        else:
            signs_padded = signs

        # Pack
        packed = []
        for i in range(out_size):
            packed.append(pack_ternary_row(signs_padded[i], in_padded))

        # Compute sparsity
        n_zero = (signs == 0).sum()
        sparsity = n_zero / signs.size * 100

        lines.append(f"/* Layer {k_abs}: Ternary [{in_size} → {out_size}] */")
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


# ──────────────────────────────────────────────────────────────────────
# 10. Main
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
    mean_raw = X_train_raw.mean(axis=0)
    std_raw = X_train_raw.std(axis=0)
    std_raw[std_raw < 1e-8] = 1e-8
    print(f"  Feature means: {np.round(mean_raw, 2)}")
    print(f"  Feature stds:  {np.round(std_raw, 2)}")

    # Normalize
    X_train_n = (X_train_raw - mean_raw) / std_raw
    X_test_n = (X_test_raw - mean_raw) / std_raw

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
    y_pred_float = model.predict(X_test_n)
    float_metrics = print_report(y_test, y_pred_float, "Float Model")

    # ── Step 5: Early-exit head ──
    print("\n" + "=" * 60)
    print("Step 5: Training early-exit head")
    print("=" * 60)
    alpha = compute_focal_alpha(y_train_os)
    exit_W, exit_b = train_exit_head(model, X_train_os, y_train_os,
                                      X_test_n, y_test, alpha)
    exit_threshold = calibrate_exit_threshold(model, exit_W, exit_b,
                                               X_test_n, y_test,
                                               percentile=15)

    # ── Step 6: Ternary quantization ──
    print("\n" + "=" * 60)
    print("Step 6: Post-hoc ternary quantization (layers 1, 2)")
    print("=" * 60)
    ternary_signs = []
    alpha_shifts = []
    for k_abs in [1, 2]:
        print(f"  Layer {k_abs} ({model.W[k_abs].shape}):")
        signs, ashift, alpha_f = ternary_quantize_layer(model.W[k_abs])
        ternary_signs.append(signs)
        alpha_shifts.append(ashift)

    # ── Step 7: Evaluate all inference modes ──
    print("\n" + "=" * 60)
    print("Step 7: Accuracy comparison")
    print("=" * 60)

    # Float
    print("\n  --- Float ---")
    y_float = model.predict(X_test_n)
    m_float = print_report(y_test, y_float, "Float")

    # Int32
    print("\n  --- Int32 ---")
    y_int32 = int32_inference(X_test_raw, model, mean_raw, std_raw)
    m_int32 = print_report(y_test, y_int32, "Int32")

    # Ternary-Int
    print("\n  --- Ternary-Int ---")
    y_ternary = ternary_int_inference(X_test_raw, model, mean_raw, std_raw,
                                       ternary_signs, alpha_shifts)
    m_ternary = print_report(y_test, y_ternary, "Ternary-Int")

    # Summary table
    print("\n  " + "=" * 50)
    print(f"  {'Mode':<15s} {'Accuracy':>10s} {'Macro F1':>10s} {'Min Rec':>10s}")
    print("  " + "-" * 50)
    for name, m in [("Float", m_float), ("Int32", m_int32), ("Ternary-Int", m_ternary)]:
        print(f"  {name:<15s} {m['accuracy']:10.4f} {m['macro_f1']:10.4f} "
              f"{m['min_recall']:10.4f}")
    print("  " + "=" * 50)

    # ── Step 8: Generate header ──
    print("\n" + "=" * 60)
    print("Step 8: Generating C header")
    print("=" * 60)
    generate_header(model, mean_raw, std_raw, exit_W, exit_b,
                    ternary_signs, alpha_shifts, exit_threshold,
                    layer_sizes)

    # Compute model memory
    # Layer 0: fused_w0 (64*12*4) + fused_offset (64*4) + bias_layer_0 (64*4)
    mem_layer0 = HIDDEN_SIZE * NUM_FEATURES * 4 + HIDDEN_SIZE * 4 + HIDDEN_SIZE * 4
    # Exit head: exit_weight (4*64*4) + exit_bias (4*4)
    mem_exit = NUM_CLASSES * HIDDEN_SIZE * 4 + NUM_CLASSES * 4
    # Ternary layers
    mem_ternary = 0
    for k_rel in range(2):
        signs = ternary_signs[k_rel]
        out_size = signs.shape[0]
        in_padded = ((signs.shape[1] + 15) // 16) * 16
        n_words = in_padded // 16
        mem_ternary += out_size * n_words * 4  # packed weights
        mem_ternary += out_size * 4  # biases
    total_mem = mem_layer0 + mem_exit + mem_ternary
    print(f"  Total model memory: {total_mem} bytes ({total_mem/1024:.1f} KB)")

    # ── Step 9: Save results JSON ──
    print("\n" + "=" * 60)
    print("Step 9: Saving results")
    print("=" * 60)
    results = {
        "architecture": layer_sizes,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 2,
        "gamma": 3.0,
        "oversample_target": OVERSAMPLE_TARGET,
        "scale_bits": SCALE_BITS,
        "early_exit_threshold": exit_threshold,
        "model_memory_bytes": total_mem,
        "float": m_float,
        "int32": m_int32,
        "ternary_int": m_ternary,
        "alpha_shifts": alpha_shifts,
        "train_samples": len(y_train_os),
        "test_samples": len(y_test),
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results written to {RESULTS_PATH}")

    # ── Step 10: Verify eBPF compilation ──
    print("\n" + "=" * 60)
    print("Step 10: Verifying eBPF compilation")
    print("=" * 60)
    ebpf_src = PROJECT_DIR / "src" / "xdp_ml_v2.c"
    ebpf_out = "/tmp/xdp_ml_v2.o"
    compile_cmd = (
        f"clang -O2 -g -target bpf -D__TARGET_ARCH_x86 "
        f"-I/usr/include/x86_64-linux-gnu "
        f"-c {ebpf_src} -o {ebpf_out}"
    )
    print(f"  {compile_cmd}")
    ret = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if ret.returncode == 0:
        print("  ✓ Compilation succeeded!")
        # Count BPF instructions
        objdump_cmd = f"llvm-objdump -d {ebpf_out}"
        ret2 = subprocess.run(objdump_cmd, shell=True, capture_output=True, text=True)
        if ret2.returncode == 0:
            insn_lines = [l for l in ret2.stdout.splitlines()
                          if l.strip() and not l.startswith("Disassembly")
                          and not l.startswith("/")
                          and not l.endswith(":") and ":" in l
                          and not l.startswith("file format")]
            print(f"  Total BPF instructions: ~{len(insn_lines)}")
            # Count per section
            current_section = None
            section_counts = {}
            for line in ret2.stdout.splitlines():
                if line.startswith("Disassembly of section"):
                    current_section = line.split("section ")[-1].rstrip(":")
                    section_counts[current_section] = 0
                elif current_section and ":" in line and line.strip() and \
                     not line.endswith(":"):
                    section_counts[current_section] = \
                        section_counts.get(current_section, 0) + 1
            for sec, cnt in section_counts.items():
                print(f"    {sec}: {cnt} instructions")
        else:
            print(f"  llvm-objdump failed: {ret2.stderr}")
    else:
        print(f"  ✗ Compilation failed!")
        print(f"    stdout: {ret.stdout}")
        print(f"    stderr: {ret.stderr}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
