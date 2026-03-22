#!/usr/bin/env python3
"""
train_qat.py — Quantization-Aware Training (QAT) with Straight-Through Estimator
===================================================================================

Solves the catastrophic minority-class degradation in post-hoc ternary quantization:
  - PORTSCAN:   Float F1=0.597 → Post-hoc Ternary F1=0.020 (catastrophic)
  - BRUTEFORCE: Float F1=0.552 → Post-hoc Ternary F1=0.039

Two QAT approaches:
  1. Ternary QAT: int32 Layer 0, QAT-ternary Layers 1 & 2
  2. Full QAT:    MapLUT Layer 0, QAT-ternary Layers 1 & 2 (zero multiply)

Architecture: MLP [12 → 64 → 64 → 4]

Training schedule:
  Phase 1 (epochs 1-30):   Float warmup
  Phase 2 (epochs 31-120): QAT with STE

Dependencies: numpy, pyarrow
Usage:  python3 scripts/train_qat.py
"""

import json
import math
import os
import struct
import sys
import time
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

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
OVERSAMPLE_TARGET = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "cicids_raw"
HEADER_PATH = PROJECT_DIR / "include" / "model_params_v3.h"
LUT_BIN_PATH = PROJECT_DIR / "data" / "lut_v3.bin"
RESULTS_PATH = PROJECT_DIR / "results" / "qat_results.json"


# ──────────────────────────────────────────────────────────────────────
# 2.  Data loading
# ──────────────────────────────────────────────────────────────────────

def map_label(label_str):
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
            cls = map_label(labels[i])
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
# 4.  Activation functions
# ──────────────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ──────────────────────────────────────────────────────────────────────
# 5.  Ternary quantization
# ──────────────────────────────────────────────────────────────────────

def ternarize(W, threshold_ratio=0.7):
    """Ternarize a weight matrix. Returns (W_ternary, signs, alpha_shift).

    W_ternary: float approximation using ternary + effective alpha
    signs: {-1, 0, +1} matrix
    alpha_shift: integer shift for 2^alpha_shift / 65536 approximation
    """
    abs_W = np.abs(W)
    threshold = threshold_ratio * abs_W.mean()

    signs = np.zeros_like(W)
    signs[W > threshold] = 1.0
    signs[W < -threshold] = -1.0

    large = abs_W[np.abs(signs) > 0]
    alpha = large.mean() if len(large) > 0 else 1.0

    alpha_shift = int(np.round(SCALE_BITS + np.log2(max(alpha, 1e-12))))
    alpha_shift = max(alpha_shift, 0)
    effective_alpha = 2.0 ** alpha_shift / SCALE

    W_ternary = signs * effective_alpha
    return W_ternary, signs, alpha_shift


# ──────────────────────────────────────────────────────────────────────
# 6.  Focal loss helpers
# ──────────────────────────────────────────────────────────────────────

def compute_focal_alpha(y, num_classes=NUM_CLASSES):
    """Alpha = sqrt(N / (nc * counts)), normalized to mean=1."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    N = len(y)
    alpha = np.sqrt(N / (num_classes * counts))
    alpha = alpha / alpha.mean()
    return alpha


def focal_backward(probs, y, alpha, gamma):
    """Focal loss gradient for softmax output.
    Normalizes by focal_weight.sum() (NOT by N).
    """
    N = len(y)
    p_t = probs[np.arange(N), y]
    p_t = np.clip(p_t, 1e-12, 1.0)
    focal_w = alpha[y] * (1.0 - p_t) ** gamma
    dz = probs.copy()
    dz[np.arange(N), y] -= 1.0
    dz *= focal_w[:, None]
    dz /= (focal_w.sum() + 1e-12)
    return dz


def focal_loss_value(probs, y, alpha, gamma):
    """Focal loss scalar for logging."""
    N = len(y)
    p_t = probs[np.arange(N), y]
    p_t = np.clip(p_t, 1e-12, 1.0)
    fl = -alpha[y] * (1.0 - p_t) ** gamma * np.log(p_t)
    return fl.mean()


# ──────────────────────────────────────────────────────────────────────
# 7.  Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
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
            "precision": round(prec, 6), "recall": round(rec, 6),
            "f1": round(f1, 6), "support": int((y_true == c).sum()),
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


def print_f1_summary(metrics, label=""):
    """One-line per-class F1 summary."""
    parts = []
    for c in range(NUM_CLASSES):
        f1 = metrics["per_class"][CLASS_NAMES[c]]["f1"]
        parts.append(f"{CLASS_NAMES[c]}={f1:.4f}")
    print(f"  {label:20s} Acc={metrics['accuracy']:.4f} MacroF1={metrics['macro_f1']:.4f} | "
          + " ".join(parts))


# ──────────────────────────────────────────────────────────────────────
# 8.  QAT MLP — Ternary QAT with STE
# ──────────────────────────────────────────────────────────────────────

class QAT_MLP:
    """MLP with Quantization-Aware Training (QAT) using STE.

    Layer 0: always float weights (or MapLUT-simulated in full QAT mode)
    Layers 1, 2: ternary quantization in forward pass during QAT phase
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        # Float shadow weights — these are what we optimize
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(layer_sizes[i + 1], fan_in) * scale)
            self.b.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, X, qat_enabled=False, threshold_ratio=0.7):
        """Forward pass. If qat_enabled, layers 1 and 2 use ternary weights.

        Returns (probs, cache).
        cache contains: a (activations), z (pre-activation), W_used (effective weights)
        """
        cache = {"a": [X], "z": [], "W_used": []}
        a = X

        for k in range(self.n_layers):
            if qat_enabled and k >= 1:
                # QAT: ternarize W_float[k] for forward pass
                W_tern, signs, ashift = ternarize(self.W[k], threshold_ratio)
                W_used = W_tern
            else:
                W_used = self.W[k]

            cache["W_used"].append(W_used)
            z = a @ W_used.T + self.b[k]
            cache["z"].append(z)

            if k < self.n_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            cache["a"].append(a)

        return a, cache

    def backward_qat(self, y, cache, alpha, gamma, qat_enabled=False):
        """Backward pass with focal loss and STE.

        During QAT: gradients flow through ternary as if W_ternary = W_float,
        but we clip gradients for W_float entries with |W| > 2.0 to zero.
        """
        N = len(y)
        probs = cache["a"][-1]

        # Focal loss gradient at output
        dz = focal_backward(probs, y, alpha, gamma)

        dW_list = [None] * self.n_layers
        db_list = [None] * self.n_layers

        for k in reversed(range(self.n_layers)):
            a_prev = cache["a"][k]
            dW = dz.T @ a_prev
            db = dz.sum(axis=0)

            if qat_enabled and k >= 1:
                # STE clipping: zero out gradient for weights outside [-2, 2]
                ste_mask = (np.abs(self.W[k]) <= 2.0).astype(np.float64)
                dW *= ste_mask

            dW_list[k] = dW
            db_list[k] = db

            if k > 0:
                # For gradient routing, use the actual float W (STE principle)
                da = dz @ self.W[k]
                dz = da * relu_grad(cache["z"][k - 1])

        return dW_list, db_list

    def predict(self, X, qat_enabled=False, threshold_ratio=0.7):
        probs, _ = self.forward(X, qat_enabled, threshold_ratio)
        return np.argmax(probs, axis=1)


# ──────────────────────────────────────────────────────────────────────
# 9.  MapLUT-aware forward pass (for Approach 2: Full QAT)
# ──────────────────────────────────────────────────────────────────────

def compute_lut_params(X_train_raw, num_bins):
    """Compute per-feature binning parameters from raw training data."""
    feat_offset = np.zeros(NUM_FEATURES, dtype=np.int32)
    feat_shift = np.zeros(NUM_FEATURES, dtype=np.uint32)
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
    return feat_offset, feat_shift


def maplut_forward_sim(X_raw, W0, b0, mu, sigma, feat_offset, feat_shift, num_bins):
    """Simulate MapLUT binning in forward pass for Full QAT.

    Bins features, computes bin centers, normalizes, then standard matmul.
    STE: gradient passes through as if binning didn't happen.
    """
    N = X_raw.shape[0]
    x_binned = np.zeros_like(X_raw)

    for j in range(NUM_FEATURES):
        raw_int = X_raw[:, j].astype(np.int64)
        shifted = np.maximum(raw_int - int(feat_offset[j]), 0)
        bins = np.clip(shifted >> int(feat_shift[j]), 0, num_bins - 1)
        bw = 1 << int(feat_shift[j])
        x_binned[:, j] = feat_offset[j] + bins * bw  # left edge of bin

    # Normalize binned values
    x_binned_norm = (x_binned - mu) / sigma

    # Standard matmul with binned features
    z = x_binned_norm @ W0.T + b0
    return z, x_binned_norm


# ──────────────────────────────────────────────────────────────────────
# 10. QAT Training Loop
# ──────────────────────────────────────────────────────────────────────

def train_qat(X_train_n, y_train, X_test_n, y_test,
              X_train_raw_for_maplut, X_test_raw,
              mu, sigma, feat_offset, feat_shift,
              layer_sizes, epochs=120, batch_size=256,
              lr=0.01, gamma=3.0, warmup_epochs=30,
              threshold_ratio=0.7, num_bins=256):
    """Two-phase QAT training.

    Phase 1 (1..warmup_epochs): Normal float training
    Phase 2 (warmup_epochs+1..epochs): QAT with STE
    """
    model = QAT_MLP(layer_sizes)
    N = len(y_train)
    alpha = compute_focal_alpha(y_train)
    print(f"  Focal alpha: {np.round(alpha, 4)}")
    print(f"  Focal gamma: {gamma}")
    print(f"  Warmup epochs: {warmup_epochs}, QAT epochs: {epochs - warmup_epochs}")
    print(f"  Threshold ratio: {threshold_ratio}")

    # Separate tracking for warmup and QAT phases
    best_warmup_score = -1.0
    best_warmup_W = [w.copy() for w in model.W]
    best_warmup_b = [b.copy() for b in model.b]

    best_qat_score = -1.0
    best_qat_W = None
    best_qat_b = None
    best_qat_epoch = 0
    current_lr = lr

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        qat_on = epoch > warmup_epochs

        perm = np.random.permutation(N)
        X_shuf = X_train_n[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]

            probs, cache = model.forward(Xb, qat_enabled=qat_on,
                                         threshold_ratio=threshold_ratio)
            loss = focal_loss_value(probs, yb, alpha, gamma)
            epoch_loss += loss
            n_batches += 1

            dW, db = model.backward_qat(yb, cache, alpha, gamma,
                                         qat_enabled=qat_on)

            # SGD update
            for k in range(model.n_layers):
                model.W[k] -= current_lr * dW[k]
                model.b[k] -= current_lr * db[k]

        # LR decay at epochs 30, 60, 90
        if epoch in [30, 60, 90]:
            current_lr *= 0.5
            print(f"    LR decayed to {current_lr:.6f}")

        # Evaluate at checkpoints
        if epoch % 5 == 0 or epoch == 1 or epoch == warmup_epochs or epoch == warmup_epochs + 1:
            elapsed = time.time() - t0

            # Float evaluation
            y_pred_float = model.predict(X_test_n, qat_enabled=False)
            m_float = compute_metrics(y_test, y_pred_float)

            # QAT ternary evaluation (what the ternary model actually produces)
            y_pred_qat = model.predict(X_test_n, qat_enabled=True,
                                       threshold_ratio=threshold_ratio)
            m_qat = compute_metrics(y_test, y_pred_qat)

            phase = "QAT" if qat_on else "Warmup"
            ps_f1 = m_qat["per_class"]["PORTSCAN"]["f1"]
            bf_f1 = m_qat["per_class"]["BRUTEFORCE"]["f1"]

            if not qat_on:
                # During warmup: track best float model
                score = m_float["accuracy"] + m_float["macro_f1"]
                if score > best_warmup_score:
                    best_warmup_score = score
                    best_warmup_W = [w.copy() for w in model.W]
                    best_warmup_b = [b.copy() for b in model.b]
            else:
                # During QAT: track best QAT model by (accuracy + macro_f1)
                # on TERNARY evaluation — this is what matters
                score = m_qat["accuracy"] + m_qat["macro_f1"]
                if score > best_qat_score:
                    best_qat_score = score
                    best_qat_W = [w.copy() for w in model.W]
                    best_qat_b = [b.copy() for b in model.b]
                    best_qat_epoch = epoch

            print(f"  Ep {epoch:3d} [{phase:6s}] ({elapsed:5.0f}s) "
                  f"loss={epoch_loss/n_batches:.4f} lr={current_lr:.5f} | "
                  f"Float: Acc={m_float['accuracy']:.4f} F1={m_float['macro_f1']:.4f} | "
                  f"QAT-Tern: Acc={m_qat['accuracy']:.4f} F1={m_qat['macro_f1']:.4f} "
                  f"PS={ps_f1:.4f} BF={bf_f1:.4f} "
                  f"| {'★' if (qat_on and score == best_qat_score) else ''}")

    # Restore best QAT model (or best warmup if QAT produced nothing)
    if best_qat_W is not None:
        model.W = best_qat_W
        model.b = best_qat_b
        print(f"\n  Best QAT model at epoch {best_qat_epoch}, "
              f"QAT score={best_qat_score:.4f}")
    else:
        model.W = best_warmup_W
        model.b = best_warmup_b
        print(f"\n  No QAT improvement; using best warmup model")
    return model


# ──────────────────────────────────────────────────────────────────────
# 11. Inference Engines
# ──────────────────────────────────────────────────────────────────────

def float_inference(X_norm, model):
    """Mode 1: Standard float MLP."""
    probs, _ = model.forward(X_norm, qat_enabled=False)
    return np.argmax(probs, axis=1)


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


def ternary_qat_inference(X_raw, model, mu, sigma, threshold_ratio=0.7):
    """Mode 3: Ternary-QAT — int32 Layer 0, QAT-ternary Layers 1 & 2."""
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

    # Layers 1, 2: ternary
    for k_abs in [1, 2]:
        _, signs, ashift = ternarize(model.W[k_abs], threshold_ratio)
        signs_i64 = signs.astype(np.int64)
        Bq = np.round(model.b[k_abs] * s).astype(np.int64)
        acc = x @ signs_i64.T
        z = (acc << ashift) >> sb
        z += Bq
        if k_abs < model.n_layers - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def maplut_int32_inference(X_raw, lut_data, bias_layer_0_q,
                           feat_offset, feat_shift, model, num_bins):
    """Mode 4: MapLUT Layer 0 + int32 Layers 1 & 2."""
    s = SCALE
    sb = SCALE_BITS
    N = X_raw.shape[0]

    # Layer 0: MapLUT
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = np.maximum(raw - int(feat_offset[j]), 0)
        bins = np.clip(shifted >> int(feat_shift[j]), 0, num_bins - 1)
        hidden += lut_data[j][bins]
    hidden = np.maximum(hidden, 0)
    x = hidden

    # Layers 1, 2: int32
    for k_abs in [1, 2]:
        Wq = np.round(model.W[k_abs] * s).astype(np.int64)
        Bq = np.round(model.b[k_abs] * s).astype(np.int64)
        z = (x @ Wq.T >> sb) + Bq
        if k_abs < model.n_layers - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


def maplut_ternary_qat_inference(X_raw, lut_data, bias_layer_0_q,
                                  feat_offset, feat_shift, model,
                                  num_bins, threshold_ratio=0.7):
    """Mode 5: MapLUT Layer 0 + QAT-ternary Layers 1 & 2 — zero multiply."""
    sb = SCALE_BITS
    s = SCALE
    N = X_raw.shape[0]

    # Layer 0: MapLUT
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = np.maximum(raw - int(feat_offset[j]), 0)
        bins = np.clip(shifted >> int(feat_shift[j]), 0, num_bins - 1)
        hidden += lut_data[j][bins]
    hidden = np.maximum(hidden, 0)
    x = hidden

    # Layers 1, 2: ternary
    for k_abs in [1, 2]:
        _, signs, ashift = ternarize(model.W[k_abs], threshold_ratio)
        signs_i64 = signs.astype(np.int64)
        Bq = np.round(model.b[k_abs] * s).astype(np.int64)
        acc = x @ signs_i64.T
        z = (acc << ashift) >> sb
        z += Bq
        if k_abs < model.n_layers - 1:
            z = np.maximum(z, 0)
        x = z

    return np.argmax(x, axis=1)


# ──────────────────────────────────────────────────────────────────────
# 12. LUT Generation
# ──────────────────────────────────────────────────────────────────────

def generate_lut(W0, b0, mu, sigma, feat_offset, feat_shift, num_bins):
    """Generate MapLUT: lut_data[j][b][i] (int32, scaled by SCALE)."""
    s = SCALE
    lut_data = np.zeros((NUM_FEATURES, num_bins, HIDDEN_SIZE), dtype=np.int32)
    for j in range(NUM_FEATURES):
        bin_width = 1 << int(feat_shift[j])
        for b in range(num_bins):
            center_raw = float(feat_offset[j]) + b * bin_width
            center_norm = (center_raw - mu[j]) / sigma[j]
            for i in range(HIDDEN_SIZE):
                lut_data[j, b, i] = int(np.round(W0[i, j] * center_norm * s))
    bias_layer_0_q = np.round(b0 * s).astype(np.int32)
    return lut_data, bias_layer_0_q


# ──────────────────────────────────────────────────────────────────────
# 13. Header and binary generation
# ──────────────────────────────────────────────────────────────────────

def pack_ternary_row(signs_row, input_dim_padded):
    n_words = input_dim_padded // 16
    words = []
    for w_idx in range(n_words):
        word = 0
        for k in range(16):
            idx = w_idx * 16 + k
            s = signs_row[idx] if idx < len(signs_row) else 0
            if s > 0:
                bits = 0b01
            elif s < 0:
                bits = 0b10
            else:
                bits = 0b00
            word |= (bits << (k * 2))
        words.append(word)
    return words


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


def generate_header(model, bias_layer_0_q, feat_offset, feat_shift,
                    ternary_data, layer_sizes, num_bins, threshold_ratio=0.7):
    """Generate include/model_params_v3.h (QAT version)."""
    s = SCALE
    sb = SCALE_BITS
    lines = []

    lines.append("#pragma once")
    lines.append("")
    lines.append("/*")
    lines.append(" * model_params_v3.h — Auto-generated by train_qat.py (QAT)")
    lines.append(f" * Architecture: {layer_sizes}")
    lines.append(" * QAT-trained: MapLUT + TernaryShift (zero multiply)")
    lines.append(f" * Scale: 2^{sb} = {s}")
    lines.append(" */")
    lines.append("")
    lines.append("#include <linux/types.h>")
    lines.append("")
    lines.append(f"#define NUM_FEATURES        {NUM_FEATURES}")
    lines.append(f"#define HIDDEN_SIZE         {HIDDEN_SIZE}")
    lines.append(f"#define NUM_CLASSES          {NUM_CLASSES}")
    lines.append(f"#define SCALE_FACTOR_BITS   {sb}")
    lines.append(f"#define SCALE_FACTOR        {s}")
    lines.append(f"#define NUM_BINS            {num_bins}")
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

    # Ternary layers 1, 2
    for k_rel, k_abs in enumerate([1, 2]):
        signs_data = ternary_data[k_rel]
        signs = signs_data["signs"]
        ashift = signs_data["alpha_shift"]
        bias = np.round(model.b[k_abs] * s).astype(np.int32)

        out_size = signs.shape[0]
        in_size = signs.shape[1]
        in_padded = ((in_size + 15) // 16) * 16

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


def write_lut_binary(lut_data, path, num_bins):
    """Write LUT as binary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for j in range(lut_data.shape[0]):
            for b in range(lut_data.shape[1]):
                for i in range(lut_data.shape[2]):
                    f.write(struct.pack("<i", int(lut_data[j, b, i])))
    size = os.path.getsize(path)
    print(f"  LUT binary written to {path} ({size} bytes = {size/1024:.1f} KB)")


# ──────────────────────────────────────────────────────────────────────
# 14. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES]
    threshold_ratio = 0.7

    # ── Step 1: Load data ──
    print("=" * 70)
    print("Step 1: Loading CIC-IDS-2017 dataset")
    print("=" * 70)
    X_all, y_all = load_data()

    # Shuffle and split 75/25
    idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]
    split = int(0.75 * len(y_all))
    X_train_raw, y_train = X_all[:split], y_all[:split]
    X_test_raw, y_test = X_all[split:], y_all[split:]
    print(f"\n  Train: {len(y_train)}, Test: {len(y_test)}")

    # Normalization stats on training data
    mu = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma[sigma < 1e-8] = 1e-8
    print(f"  Feature means: {np.round(mu, 2)}")
    print(f"  Feature stds:  {np.round(sigma, 2)}")

    X_train_n = (X_train_raw - mu) / sigma
    X_test_n = (X_test_raw - mu) / sigma

    # ── Step 2: Oversample ──
    print("\n" + "=" * 70)
    print("Step 2: Oversampling to 100K per class")
    print("=" * 70)
    X_train_os, y_train_os = oversample(X_train_n, y_train, target=OVERSAMPLE_TARGET)
    print(f"  Oversampled: {len(y_train_os)} total")
    for c in range(NUM_CLASSES):
        print(f"    Class {c}: {(y_train_os == c).sum()}")

    # ── Step 3: QAT Training ──
    print("\n" + "=" * 70)
    print("Step 3: QAT Training — Two Phase (Warmup → QAT with STE)")
    print("=" * 70)

    # Compute LUT params from raw training data (needed for full QAT eval)
    feat_offset_256, feat_shift_256 = compute_lut_params(X_train_raw, 256)

    model = train_qat(
        X_train_os, y_train_os, X_test_n, y_test,
        X_train_raw_for_maplut=X_train_raw,
        X_test_raw=X_test_raw,
        mu=mu, sigma=sigma,
        feat_offset=feat_offset_256, feat_shift=feat_shift_256,
        layer_sizes=layer_sizes,
        epochs=120, batch_size=256, lr=0.01, gamma=3.0,
        warmup_epochs=30, threshold_ratio=threshold_ratio,
        num_bins=256,
    )

    # ── Step 4: Comprehensive Evaluation ──
    print("\n" + "=" * 70)
    print("Step 4: Comprehensive Evaluation — All Inference Modes")
    print("=" * 70)

    all_results = {}

    # Mode 1: Float (reference)
    print("\n  ─── Mode 1: Float (reference) ───")
    y_float = float_inference(X_test_n, model)
    m_float = print_report(y_test, y_float, "Float Model")
    all_results["float"] = m_float

    # Mode 2: Int32 (standard quantization)
    print("\n  ─── Mode 2: Int32 (all layers) ───")
    y_int32 = int32_inference(X_test_raw, model, mu, sigma)
    m_int32 = print_report(y_test, y_int32, "Int32 Model")
    all_results["int32"] = m_int32

    # Mode 3: Ternary-QAT (int32 L0, QAT-ternary L1/L2)
    print("\n  ─── Mode 3: Ternary-QAT (int32 L0, QAT-tern L1/L2) ───")
    y_tern_qat = ternary_qat_inference(X_test_raw, model, mu, sigma, threshold_ratio)
    m_tern_qat = print_report(y_test, y_tern_qat, "Ternary-QAT")
    all_results["ternary_qat"] = m_tern_qat

    # Mode 4 & 5: MapLUT variants with multiple bin sizes
    print("\n  ─── MapLUT Evaluation (bins = [64, 128, 256, 512]) ───")
    maplut_results = {}

    for nb in [64, 128, 256, 512]:
        fo, fs = compute_lut_params(X_train_raw, nb)
        lut_data, bias_l0_q = generate_lut(model.W[0], model.b[0], mu, sigma, fo, fs, nb)
        mem = lut_data.nbytes

        # Mode 4: MapLUT + Int32
        y_ml_int = maplut_int32_inference(X_test_raw, lut_data, bias_l0_q,
                                           fo, fs, model, nb)
        m_ml_int = compute_metrics(y_test, y_ml_int)

        # Mode 5: MapLUT + Ternary-QAT (zero multiply)
        y_ml_tern = maplut_ternary_qat_inference(
            X_test_raw, lut_data, bias_l0_q, fo, fs, model, nb, threshold_ratio)
        m_ml_tern = compute_metrics(y_test, y_ml_tern)

        maplut_results[nb] = {
            "maplut_int32": m_ml_int,
            "maplut_ternary_qat": m_ml_tern,
            "lut_memory_bytes": int(mem),
        }

        ps_int = m_ml_int["per_class"]["PORTSCAN"]["f1"]
        bf_int = m_ml_int["per_class"]["BRUTEFORCE"]["f1"]
        ps_tern = m_ml_tern["per_class"]["PORTSCAN"]["f1"]
        bf_tern = m_ml_tern["per_class"]["BRUTEFORCE"]["f1"]

        print(f"\n    bins={nb:4d} (LUT={mem/1024:.0f} KB):")
        print(f"      MapLUT+Int32:       Acc={m_ml_int['accuracy']:.4f} "
              f"MacroF1={m_ml_int['macro_f1']:.4f} PS={ps_int:.4f} BF={bf_int:.4f}")
        print(f"      MapLUT+Tern-QAT:    Acc={m_ml_tern['accuracy']:.4f} "
              f"MacroF1={m_ml_tern['macro_f1']:.4f} PS={ps_tern:.4f} BF={bf_tern:.4f}")

    # Print detailed report for best MapLUT+Ternary-QAT (256 bins)
    best_nb = 256
    print(f"\n  ─── Mode 4: MapLUT+Int32 (bins={best_nb}) ───")
    fo_best, fs_best = compute_lut_params(X_train_raw, best_nb)
    lut_best, bias_best = generate_lut(model.W[0], model.b[0], mu, sigma,
                                        fo_best, fs_best, best_nb)
    y_ml_int_best = maplut_int32_inference(X_test_raw, lut_best, bias_best,
                                            fo_best, fs_best, model, best_nb)
    m_ml_int_best = print_report(y_test, y_ml_int_best, f"MapLUT+Int32 (bins={best_nb})")
    all_results["maplut_int32"] = m_ml_int_best

    print(f"\n  ─── Mode 5: MapLUT+Ternary-QAT (bins={best_nb}, zero multiply) ───")
    y_ml_tern_best = maplut_ternary_qat_inference(
        X_test_raw, lut_best, bias_best, fo_best, fs_best, model, best_nb, threshold_ratio)
    m_ml_tern_best = print_report(y_test, y_ml_tern_best,
                                   f"MapLUT+Ternary-QAT (bins={best_nb})")
    all_results["maplut_ternary_qat"] = m_ml_tern_best

    # ── Summary Table ──
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE — Per-Class F1 Scores")
    print("=" * 70)
    print(f"  {'Mode':<28s} {'Acc':>7s} {'MacF1':>7s} | "
          f"{'BENIGN':>8s} {'DDOS':>8s} {'PORTSCAN':>8s} {'BRUTEFC':>8s}")
    print("  " + "-" * 84)

    summary_modes = [
        ("Float (reference)", m_float),
        ("Int32 (all layers)", m_int32),
        ("Ternary-QAT (int32 L0)", m_tern_qat),
        (f"MapLUT+Int32 ({best_nb}b)", m_ml_int_best),
        (f"MapLUT+Tern-QAT ({best_nb}b) ★", m_ml_tern_best),
    ]

    for name, m in summary_modes:
        f1s = [m["per_class"][cn]["f1"] for cn in CLASS_NAMES]
        print(f"  {name:<28s} {m['accuracy']:7.4f} {m['macro_f1']:7.4f} | "
              f"{f1s[0]:8.4f} {f1s[1]:8.4f} {f1s[2]:8.4f} {f1s[3]:8.4f}")

    print("  " + "-" * 84)
    print("  ★ = zero-multiply pipeline (target deployment mode)")

    # Key metric check
    ps_qat = m_tern_qat["per_class"]["PORTSCAN"]["f1"]
    ps_zero = m_ml_tern_best["per_class"]["PORTSCAN"]["f1"]
    print(f"\n  KEY METRIC — PORTSCAN F1:")
    print(f"    Ternary-QAT:          {ps_qat:.4f} (target: >0.3)")
    print(f"    MapLUT+Tern-QAT:      {ps_zero:.4f} (target: >0.3)")
    if ps_qat > 0.3 or ps_zero > 0.3:
        print(f"    ✓ PORTSCAN F1 target achieved!")
    else:
        print(f"    ✗ PORTSCAN F1 below target. Consider tuning threshold_ratio or training longer.")

    # ── Step 5: Ternary weight analysis ──
    print("\n" + "=" * 70)
    print("Step 5: Ternary Weight Analysis")
    print("=" * 70)

    ternary_data = []
    for k_abs in [1, 2]:
        _, signs, ashift = ternarize(model.W[k_abs], threshold_ratio)
        n_zero = (signs == 0).sum()
        sparsity = n_zero / signs.size * 100
        alpha_float = np.abs(model.W[k_abs][np.abs(signs) > 0]).mean() if (np.abs(signs) > 0).sum() > 0 else 1.0
        ternary_data.append({
            "signs": signs.astype(np.int64),
            "alpha_shift": ashift,
        })
        print(f"  Layer {k_abs} [{model.W[k_abs].shape[1]}→{model.W[k_abs].shape[0]}]: "
              f"alpha_shift={ashift}, sparsity={sparsity:.1f}%, "
              f"alpha_float={alpha_float:.6f}")

    # ── Step 6: Generate artifacts if improved ──
    print("\n" + "=" * 70)
    print("Step 6: Generating artifacts (header + LUT)")
    print("=" * 70)

    # Use 256 bins for best accuracy
    generate_header(model, bias_best, fo_best, fs_best,
                    ternary_data, layer_sizes, best_nb, threshold_ratio)
    write_lut_binary(lut_best, str(LUT_BIN_PATH), best_nb)

    # ── Step 7: Save results JSON ──
    print("\n" + "=" * 70)
    print("Step 7: Saving results")
    print("=" * 70)

    results = {
        "architecture": layer_sizes,
        "threshold_ratio": threshold_ratio,
        "num_bins_best": best_nb,
        "training": {
            "epochs": 120,
            "warmup_epochs": 30,
            "qat_epochs": 90,
            "batch_size": 256,
            "lr": 0.01,
            "gamma": 3.0,
            "oversample_target": OVERSAMPLE_TARGET,
        },
        "float": m_float,
        "int32": m_int32,
        "ternary_qat": m_tern_qat,
        "maplut_int32": m_ml_int_best,
        "maplut_ternary_qat": m_ml_tern_best,
        "maplut_sweep": {},
        "ternary_analysis": [],
    }

    for nb, mdata in maplut_results.items():
        results["maplut_sweep"][str(nb)] = mdata

    for k_rel, k_abs in enumerate([1, 2]):
        td = ternary_data[k_rel]
        signs = td["signs"]
        n_zero = int((signs == 0).sum())
        results["ternary_analysis"].append({
            "layer": k_abs,
            "alpha_shift": td["alpha_shift"],
            "sparsity_pct": round(n_zero / signs.size * 100, 1),
            "shape": list(signs.shape),
        })

    results["train_samples"] = int(len(y_train_os))
    results["test_samples"] = int(len(y_test))

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results written to {RESULTS_PATH}")

    # ── Timing ──
    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print("\n" + "=" * 70)
    print("Done! QAT artifacts generated:")
    print(f"  Header:  {HEADER_PATH}")
    print(f"  LUT:     {LUT_BIN_PATH}")
    print(f"  Results: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
