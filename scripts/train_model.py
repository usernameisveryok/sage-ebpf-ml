#!/usr/bin/env python3
"""
train_model.py — Train a packet-classification MLP and export quantized
weights as a C header for eBPF inference.

Dependencies: numpy only  (no PyTorch / sklearn required)

Usage:
    python3 scripts/train_model.py

Outputs:
    include/model_params.h   — quantized int32 weights / biases / norm params
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────
# 1.  Synthetic dataset generation
# ──────────────────────────────────────────────────────────────────────
#
# Six features per sample:
#   pkt_len          — packet length           (64 – 1500)
#   hdr_len          — header length           (20 – 60)
#   dst_port         — destination port        (0 – 65535)
#   fwd_iat          — forward inter-arrival   (0 – 1_000_000 μs)
#   total_fwd_pkts   — total forward packets   (1 – 10_000)
#   fwd_pkt_len_max  — max fwd packet length   (64 – 1500)
#
# Four classes:
#   0 = BENIGN       normal traffic
#   1 = DDOS         high pkt rate, many small pkts
#   2 = PORTSCAN     many dst ports, few pkts per flow
#   3 = BRUTEFORCE   targets port 22/21, moderate rate

FEATURE_NAMES = [
    "pkt_len", "hdr_len", "dst_port",
    "fwd_iat", "total_fwd_pkts", "fwd_pkt_len_max",
]
CLASS_NAMES = ["BENIGN", "DDOS", "PORTSCAN", "BRUTEFORCE"]
NUM_FEATURES = len(FEATURE_NAMES)        # 6
NUM_CLASSES  = len(CLASS_NAMES)           # 4
HIDDEN_SIZE  = 32
N_SAMPLES    = 50_000                     # total samples
SAMPLES_PER_CLASS = N_SAMPLES // NUM_CLASSES


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_class_samples(label: int, n: int) -> np.ndarray:
    """Return (n, 6) float64 array with class-specific distributions."""
    X = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    if label == 0:  # BENIGN
        X[:, 0] = _clip(np.random.normal(600, 300, n), 64, 1500)      # pkt_len
        X[:, 1] = _clip(np.random.normal(40, 8, n), 20, 60)           # hdr_len
        # Common ports: 80, 443, 22, 53, 8080 with some noise
        common_ports = np.array([80, 443, 22, 53, 8080, 8443, 3306, 5432])
        X[:, 2] = np.random.choice(common_ports, n).astype(float) + \
                  np.random.randint(-5, 6, n)
        X[:, 2] = _clip(X[:, 2], 0, 65535)
        X[:, 3] = _clip(np.random.exponential(50000, n), 0, 1_000_000)  # fwd_iat
        X[:, 4] = _clip(np.random.lognormal(4, 1.5, n), 1, 10000)      # total_fwd_pkts
        X[:, 5] = _clip(np.random.normal(800, 350, n), 64, 1500)        # fwd_pkt_len_max

    elif label == 1:  # DDOS
        X[:, 0] = _clip(np.random.normal(120, 30, n), 64, 1500)       # small pkts
        X[:, 1] = _clip(np.random.normal(28, 4, n), 20, 60)           # minimal headers
        X[:, 2] = _clip(np.random.normal(80, 20, n), 0, 65535)        # target port ~80
        X[:, 3] = _clip(np.random.exponential(200, n), 0, 1_000_000)  # very low IAT
        X[:, 4] = _clip(np.random.lognormal(7, 1.0, n), 1, 10000)    # many pkts
        X[:, 5] = _clip(np.random.normal(150, 40, n), 64, 1500)       # small max

    elif label == 2:  # PORTSCAN
        X[:, 0] = _clip(np.random.normal(80, 15, n), 64, 1500)        # tiny pkts
        X[:, 1] = _clip(np.random.normal(30, 5, n), 20, 60)
        X[:, 2] = np.random.uniform(1, 65535, n)                       # random ports
        X[:, 3] = _clip(np.random.exponential(5000, n), 0, 1_000_000) # moderate IAT
        X[:, 4] = _clip(np.random.exponential(3, n) + 1, 1, 10000)    # very few pkts/flow
        X[:, 5] = _clip(np.random.normal(90, 15, n), 64, 1500)

    elif label == 3:  # BRUTEFORCE
        X[:, 0] = _clip(np.random.normal(200, 60, n), 64, 1500)
        X[:, 1] = _clip(np.random.normal(35, 6, n), 20, 60)
        # Targets port 22 (SSH) or 21 (FTP)
        X[:, 2] = np.where(np.random.rand(n) < 0.7, 22, 21).astype(float) + \
                  np.random.randint(-2, 3, n)
        X[:, 2] = _clip(X[:, 2], 0, 65535)
        X[:, 3] = _clip(np.random.normal(10000, 3000, n), 0, 1_000_000)  # medium IAT
        X[:, 4] = _clip(np.random.normal(50, 20, n), 1, 10000)           # moderate pkts
        X[:, 5] = _clip(np.random.normal(250, 80, n), 64, 1500)

    return X


def generate_dataset():
    """Return X (N, 6) and y (N,) with balanced classes."""
    Xs, ys = [], []
    for c in range(NUM_CLASSES):
        Xs.append(generate_class_samples(c, SAMPLES_PER_CLASS))
        ys.append(np.full(SAMPLES_PER_CLASS, c, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# ──────────────────────────────────────────────────────────────────────
# 2.  Numpy MLP — forward / backward / training
# ──────────────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(z.dtype)

def softmax(z):
    """Numerically stable softmax along last axis."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_loss(probs, y, class_weights=None):
    """Mean cross-entropy with optional per-class weights."""
    n = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-12)
    if class_weights is not None:
        w = class_weights[y]
        return -(w * log_p).sum() / w.sum()
    return -log_p.mean()


class MLP:
    """Simple 2-hidden-layer MLP: [6] → 32 → 32 → [4]."""

    def __init__(self, layer_sizes):
        """He initialisation for ReLU layers."""
        self.layer_sizes = layer_sizes  # e.g. [6, 32, 32, 4]
        self.W = []   # weights
        self.b = []   # biases
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale   = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(fan_out, fan_in) * scale)
            self.b.append(np.zeros(fan_out))

    # ---- forward ----
    def forward(self, X):
        """X: (N, features).  Returns (probs, cache)."""
        cache = {"a": [X]}  # activations (post-ReLU / input)
        a = X
        for k in range(len(self.W)):
            z = a @ self.W[k].T + self.b[k]       # (N, out)
            cache.setdefault("z", []).append(z)
            if k < len(self.W) - 1:                # hidden layers
                a = relu(z)
            else:                                   # output layer
                a = softmax(z)
            cache["a"].append(a)
        return a, cache                             # probs, cache

    # ---- backward ----
    def backward(self, y, cache, class_weights=None):
        """Compute gradients via backprop.  Returns (dW_list, db_list)."""
        N = len(y)
        probs = cache["a"][-1]                      # (N, C)
        # Gradient of softmax + cross-entropy
        dz = probs.copy()
        dz[np.arange(N), y] -= 1.0
        if class_weights is not None:
            w = class_weights[y]                     # (N,)
            dz *= w[:, None]                         # weight each sample
            dz /= w.sum()
        else:
            dz /= N                                  # (N, C)

        dW_list = [None] * len(self.W)
        db_list = [None] * len(self.W)

        for k in reversed(range(len(self.W))):
            a_prev = cache["a"][k]                   # (N, fan_in)
            dW_list[k] = dz.T @ a_prev               # (fan_out, fan_in)
            db_list[k] = dz.sum(axis=0)               # (fan_out,)
            if k > 0:
                da = dz @ self.W[k]                    # (N, fan_in)
                dz = da * relu_grad(cache["z"][k - 1]) # element-wise

        return dW_list, db_list

    # ---- SGD step ----
    def step(self, dW_list, db_list, lr):
        for k in range(len(self.W)):
            self.W[k] -= lr * dW_list[k]
            self.b[k] -= lr * db_list[k]

    # ---- predict ----
    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


def train_model(X_train, y_train, X_val, y_val,
                layer_sizes, epochs=80, batch_size=256, lr=0.01,
                quiet=False):
    """Train MLP with mini-batch SGD + simple LR decay."""
    model = MLP(layer_sizes)
    N = len(y_train)

    # Compute inverse-frequency class weights for imbalanced data
    n_classes = layer_sizes[-1]
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1)  # avoid div by zero
    class_weights = N / (n_classes * class_counts)
    if not quiet:
        print(f"  Class weights: {dict(enumerate(np.round(class_weights, 2)))}")

    best_val_acc = 0.0
    best_W = [w.copy() for w in model.W]
    best_b = [b.copy() for b in model.b]

    for epoch in range(1, epochs + 1):
        # Shuffle training data
        perm = np.random.permutation(N)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            probs, cache = model.forward(Xb)
            loss = cross_entropy_loss(probs, yb, class_weights)
            epoch_loss += loss
            n_batches += 1
            dW, db = model.backward(yb, cache, class_weights)
            model.step(dW, db, lr)

        # LR decay
        if epoch % 30 == 0:
            lr *= 0.5

        # Validation
        val_pred = model.predict(X_val)
        val_acc  = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_W = [w.copy() for w in model.W]
            best_b = [b.copy() for b in model.b]

        if not quiet and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  loss={epoch_loss / n_batches:.4f}  "
                  f"val_acc={val_acc:.4f}  lr={lr:.6f}")

    # Restore best
    model.W = best_W
    model.b = best_b
    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────
# 3.  Evaluation helpers (no sklearn needed)
# ──────────────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true, y_pred, class_names):
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    print("\n  Confusion matrix:")
    hdr = "            " + "".join(f"{c:>12s}" for c in class_names)
    print(hdr)
    for i, name in enumerate(class_names):
        row = f"  {name:>10s}" + "".join(f"{cm[i, j]:12d}" for j in range(n_classes))
        print(row)

    print(f"\n  {'Class':<12s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>8s}")
    print("  " + "-" * 44)
    f1s = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup  = cm[i, :].sum()
        f1s.append(f1)
        print(f"  {name:<12s} {prec:8.4f} {rec:8.4f} {f1:8.4f} {sup:8d}")
    macro_f1 = np.mean(f1s)
    acc = (y_true == y_pred).mean()
    print("  " + "-" * 44)
    print(f"  {'Accuracy':<12s} {'':>8s} {'':>8s} {acc:8.4f} {len(y_true):8d}")
    print(f"  {'Macro F1':<12s} {'':>8s} {'':>8s} {macro_f1:8.4f}")
    return acc, macro_f1


# ──────────────────────────────────────────────────────────────────────
# 4.  Quantization — "Enlargement Method" (DSN 2024)
# ──────────────────────────────────────────────────────────────────────
#
# Scale factor s = 2^b,  b = 16  →  s = 65536
#
# Standardisation (per-feature mean/std computed on training set):
#   x_norm[j] = round( s * (x[j] - mean[j]) / std[j] )        — int32
#
# For each layer k:
#   W_E[k] = round( s * W[k] )                                 — int32
#   B_E[k] = round( s * B[k] )                                 — int32
#
# Integer inference for layer k:
#   acc_i  = sum_j( W_E[k][i][j] * x[j] )     (int64 accum.)
#   y[i]   = (acc_i >> b) + B_E[k][i]          (int32 result, scaled by s)
#   if hidden: y[i] = max(0, y[i])             (ReLU)
#
# Final layer output is still scaled by s, but argmax is
# scale-invariant so we just pick the largest element.

SCALE_BITS = 16
SCALE      = 1 << SCALE_BITS   # 65536


def quantize_model(model, mean, std, scale=None):
    """Return quantized weights, biases, and normalisation params as int32."""
    # Normalisation: store (s / std[j]) and (s * mean[j] / std[j]) as int32
    # so that x_norm[j] = round( norm_scale[j] * x[j] - norm_offset[j] )
    #   where  norm_scale[j]  = s / std[j]
    #          norm_offset[j] = s * mean[j] / std[j]
    # Both stored as int32 (they are multiplied by raw int features).
    #
    # Actually, to keep integer precision we store them as int32:
    #   norm_scale  = round( s / std )
    #   norm_offset = round( s * mean / std )
    # Then:  x_norm[j] = norm_scale[j] * x_raw[j] - norm_offset[j]
    #
    # But x_raw can be up to 65535 (dst_port) and norm_scale up to s.
    # 65535 * 65536 = 2^32 - 65536 which fits in int64 but not int32 product.
    # So in eBPF we'll do the multiply in int64.  The stored params are int32.

    _s = scale if scale is not None else SCALE
    norm_scale  = np.round(_s / std).astype(np.int32)
    norm_offset = np.round(_s * mean / std).astype(np.int32)

    W_q = []
    b_q = []
    for k in range(len(model.W)):
        wq = np.round(_s * model.W[k]).astype(np.int32)
        bq = np.round(_s * model.b[k]).astype(np.int32)
        W_q.append(wq)
        b_q.append(bq)

    return W_q, b_q, norm_scale, norm_offset


def verify_quantized(model, W_q, b_q, norm_scale, norm_offset,
                     X_raw, y_true, scale_bits=None):
    """Run integer-only inference and compare accuracy to float model."""
    _sb = scale_bits if scale_bits is not None else SCALE_BITS
    N = X_raw.shape[0]
    # Standardise in integer domain
    # x_norm[j] = norm_scale[j] * x_raw[j] - norm_offset[j]   (all int64)
    x = (norm_scale.astype(np.int64)[None, :] *
         X_raw.astype(np.int64)) - norm_offset.astype(np.int64)[None, :]
    # x is (N, 6) int64, scaled by s

    for k in range(len(W_q)):
        W = W_q[k].astype(np.int64)   # (out, in)
        B = b_q[k].astype(np.int64)   # (out,)
        # Matrix multiply: (N, in) @ (in, out) → (N, out)
        z = x @ W.T                    # (N, out) — scaled by s^2
        z = (z >> _sb) + B              # (N, out) — scaled by s
        if k < len(W_q) - 1:
            z = np.maximum(0, z)        # ReLU
        x = z

    preds = np.argmax(x, axis=1)
    acc = (preds == y_true).mean()
    return acc, preds


# ──────────────────────────────────────────────────────────────────────
# 5.  C header export
# ──────────────────────────────────────────────────────────────────────

def _c_array_2d(name, arr):
    """Format a 2-D int32 array as a C static const definition."""
    rows, cols = arr.shape
    lines = [f"static const __s32 {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        vals = ", ".join(str(int(v)) for v in arr[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{ {vals} }}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _c_array_1d(name, arr):
    """Format a 1-D int32 array as a C static const definition."""
    size = arr.shape[0]
    vals = ", ".join(str(int(v)) for v in arr)
    return f"static const __s32 {name}[{size}] = {{ {vals} }};"


def export_header(path, W_q, b_q, norm_scale, norm_offset, layer_sizes,
                  scale_bits=None, scale=None):
    """Write the quantized model to a C header file."""
    _sb = scale_bits if scale_bits is not None else SCALE_BITS
    _s = scale if scale is not None else SCALE
    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("/*")
    lines.append(" * model_params.h — Auto-generated by train_model.py")
    lines.append(" *")
    lines.append(" * Quantized MLP parameters for eBPF packet classification.")
    lines.append(f" * Architecture: {layer_sizes}")
    lines.append(f" * Scale factor: s = 2^{_sb} = {_s}")
    lines.append(" *")
    lines.append(" * Integer inference per layer k:")
    lines.append(" *   acc  = sum_j( W[k][i][j] * x[j] )   // int64 accumulator")
    lines.append(f" *   y[i] = (acc >> {_sb}) + B[k][i]     // int32, scaled by s")
    lines.append(" *   y[i] = max(0, y[i])                  // ReLU (hidden layers only)")
    lines.append(" *")
    lines.append(" * Input standardisation:")
    lines.append(" *   x_norm[j] = (int64)norm_scale[j] * x_raw[j] - norm_offset[j]")
    lines.append(" */")
    lines.append("")
    lines.append("#include <linux/types.h>")
    lines.append("")

    # Architecture constants
    lines.append("/* ── Architecture constants ──────────────────────────── */")
    lines.append(f"#define NUM_FEATURES        {layer_sizes[0]}")
    lines.append(f"#define HIDDEN_SIZE         {layer_sizes[1]}")
    lines.append(f"#define NUM_CLASSES         {layer_sizes[-1]}")
    lines.append(f"#define NUM_LAYERS          {len(layer_sizes) - 1}")
    lines.append(f"#define SCALE_FACTOR_BITS   {_sb}")
    lines.append(f"#define SCALE_FACTOR        {_s}")
    lines.append("")

    # Normalisation
    lines.append("/* ── Input standardisation (int32) ─────────────────── */")
    lines.append(_c_array_1d("norm_scale", norm_scale))
    lines.append("")
    lines.append(_c_array_1d("norm_offset", norm_offset))
    lines.append("")

    # Per-layer weights and biases
    for k in range(len(W_q)):
        lines.append(f"/* ── Layer {k}  ({W_q[k].shape[1]} → {W_q[k].shape[0]}) "
                      f"──────────────────────────── */")
        lines.append(_c_array_2d(f"weight_layer_{k}", W_q[k]))
        lines.append("")
        lines.append(_c_array_1d(f"bias_layer_{k}", b_q[k]))
        lines.append("")

    # Class label mapping
    lines.append("/* ── Class labels ───────────────────────────────────── */")
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"#define CLASS_{name:<12s} {i}")
    lines.append("")

    header_text = "\n".join(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(header_text)
    print(f"\n  Header written to {path}  ({len(header_text)} bytes)")


# ──────────────────────────────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train packet-classification MLP and export quantized C header.")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--scale-bits", type=int, default=SCALE_BITS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-header", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    script_dir  = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    header_path = args.output_header or str(project_dir / "include" / "model_params.h")

    scale_bits = args.scale_bits
    scale = 1 << scale_bits

    layer_sizes = [NUM_FEATURES] + [args.hidden_size] * args.num_layers + [NUM_CLASSES]

    # ── Load or generate data ───────────────────────────────────────
    print("=" * 60)
    if args.data_dir:
        print("Step 1: Loading dataset from", args.data_dir)
        print("=" * 60)
        X_train = np.load(os.path.join(args.data_dir, 'cicids_train_X.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'cicids_train_y.npy'))
        X_test = np.load(os.path.join(args.data_dir, 'cicids_test_X.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'cicids_test_y.npy'))
    else:
        print("Step 1: Generating synthetic dataset")
        print("=" * 60)
        X, y = generate_dataset()
        print(f"  Total samples : {len(y)}")
        print(f"  Feature shape : {X.shape}")
        for c in range(NUM_CLASSES):
            print(f"  Class {c} ({CLASS_NAMES[c]:>12s}): {(y == c).sum()}")

        # ── Train / test split (80 / 20) ──────────────────────────────
        split = int(0.8 * len(y))
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

    # ── Standardise (fit on train only) ───────────────────────────
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std < 1e-8] = 1e-8   # avoid division by zero

    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    print(f"\n  Train samples : {len(y_train)}")
    print(f"  Test  samples : {len(y_test)}")
    print(f"  Feature means : {np.round(mean, 2)}")
    print(f"  Feature stds  : {np.round(std, 2)}")

    # ── Train ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Training MLP  ", layer_sizes)
    print("=" * 60)
    model = train_model(X_train_n, y_train, X_test_n, y_test,
                        layer_sizes, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        quiet=args.quiet)

    # ── Evaluate (float) ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Evaluation (float32 model)")
    print("=" * 60)
    y_pred = model.predict(X_test_n)
    acc, macro_f1 = classification_report(y_test, y_pred, CLASS_NAMES)

    # ── Quantize ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 4: Quantization  (s = 2^{scale_bits} = {scale})")
    print("=" * 60)
    W_q, b_q, norm_s, norm_o = quantize_model(model, mean, std, scale=scale)

    for k in range(len(W_q)):
        w = W_q[k]
        print(f"  Layer {k} weights: shape={w.shape}  "
              f"range=[{w.min()}, {w.max()}]")
    print(f"  norm_scale  : {norm_s}")
    print(f"  norm_offset : {norm_o}")

    # ── Verify integer-only inference ────────────────────────────
    print("\n  Verifying integer-only inference on test set...")
    q_acc, q_pred = verify_quantized(model, W_q, b_q, norm_s, norm_o,
                                     X_test, y_test, scale_bits=scale_bits)
    print(f"  Float   accuracy: {acc:.4f}")
    print(f"  Integer accuracy: {q_acc:.4f}")
    print(f"  Accuracy delta  : {abs(acc - q_acc):.4f}")

    # Show integer model classification report
    print("\n  Integer model classification report:")
    classification_report(y_test, q_pred, CLASS_NAMES)

    # ── Export header ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Exporting C header")
    print("=" * 60)
    export_header(header_path, W_q, b_q, norm_s, norm_o, layer_sizes,
                  scale_bits=scale_bits, scale=scale)

    # ── JSON output ──────────────────────────────────────────────
    if args.json_output:
        result = {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "scale_bits": args.scale_bits,
            "float_accuracy": float(acc),
            "int_accuracy": float(q_acc),
            "accuracy_delta": float(abs(acc - q_acc)),
            "macro_f1": float(macro_f1),
            "model_memory_bytes": sum(
                w.nbytes + b.nbytes for w, b in zip(W_q, b_q)),
        }
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  JSON results written to {args.json_output}")

    print("\n" + "=" * 60)
    print("Done!  Use the header in your eBPF program:")
    print(f'  #include "model_params.h"')
    print("=" * 60)




if __name__ == "__main__":
    main()
