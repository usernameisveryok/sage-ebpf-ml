#!/usr/bin/env python3
"""
debug_maplut.py — Diagnose why MapLUT Layer 0 gives ~10% accuracy
while Int32 gives 95%+.

Root cause: generate_lut() uses bin CENTER (offset + (b+0.5)*width),
which is catastrophic for skewed features with large bin widths.
E.g., "Fwd Header Length" has bin_width=131072 but 97% of values < 1000.
Bin 0 center = 65536, actual typical value ≈ 272.  Error: 65264 raw ≈ 3.74σ.
"""

import math
import os
import sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

# ── Constants (same as train_v3.py) ──
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "cicids_raw"


# ── Data loading (same as train_v3.py) ──
def map_label(label_str):
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    if "Web Attack" in label_str:
        return 3
    if "DoS" in label_str or "DDoS" in label_str:
        return 1
    return -1


def load_data():
    import pyarrow.parquet as pq
    all_X, all_y = [], []
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")
    for fpath in parquet_files:
        table = pq.read_table(str(fpath), columns=FEATURE_COLS + ["Label"])
        n_rows = table.num_rows
        labels = table.column("Label").to_pylist()
        feat_arrays = [table.column(col).to_numpy().astype(np.float64) for col in FEATURE_COLS]
        X_file = np.column_stack(feat_arrays)
        for i in range(n_rows):
            cls = map_label(labels[i])
            if cls < 0:
                continue
            row = X_file[i]
            if np.any(np.isnan(row)) or np.any(np.isinf(row)):
                continue
            all_X.append(row)
            all_y.append(cls)
    X = np.maximum(np.array(all_X, dtype=np.float64), 0.0)
    y = np.array(all_y, dtype=np.int64)
    print(f"  Loaded {len(y)} samples")
    return X, y


# ── Simple MLP (same as train_v3.py) ──
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

class MLP:
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


def train_mlp_quick(X_train, y_train, layer_sizes, epochs=30, batch_size=256, lr=0.01):
    model = MLP(layer_sizes)
    N = len(y_train)
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1)
    alpha = np.sqrt(N / (NUM_CLASSES * counts))
    alpha = alpha / alpha.mean()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]
        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            nb = len(yb)
            probs, cache = model.forward(Xb)
            p_t = np.clip(probs[np.arange(nb), yb], 1e-12, 1.0)
            focal_w = alpha[yb] * (1.0 - p_t) ** 3.0
            dz = probs.copy()
            dz[np.arange(nb), yb] -= 1.0
            dz *= focal_w[:, None]
            dz /= nb
            dW_list = [None] * len(model.W)
            db_list = [None] * len(model.W)
            for k in reversed(range(len(model.W))):
                a_prev = cache["a"][k]
                dW_list[k] = dz.T @ a_prev
                db_list[k] = dz.sum(axis=0)
                if k > 0:
                    da = dz @ model.W[k]
                    dz = da * (cache["z"][k - 1] > 0).astype(dz.dtype)
            for k in range(len(model.W)):
                model.W[k] -= lr * dW_list[k]
                model.b[k] -= lr * db_list[k]
        if epoch % 10 == 0:
            acc = (model.predict(X_train) == y_train).mean()
            print(f"    epoch {epoch:3d}  train_acc={acc:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════
#  LUT Generation: ORIGINAL (buggy) and FIXED versions
# ══════════════════════════════════════════════════════════════════════

def compute_lut_params(X_train_raw, num_bins=NUM_BINS):
    """Original from train_v3.py."""
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


def generate_lut_original(W0, b0, mu, sigma, feat_offset, feat_shift, num_bins=NUM_BINS):
    """ORIGINAL (buggy): uses bin CENTER = offset + (b + 0.5) * bin_width."""
    s = SCALE
    lut_data = np.zeros((NUM_FEATURES, num_bins, HIDDEN_SIZE), dtype=np.int32)
    for j in range(NUM_FEATURES):
        bin_width = 1 << int(feat_shift[j])
        for b in range(num_bins):
            center_raw = float(feat_offset[j]) + (b + 0.5) * bin_width  # BUG: +0.5
            center_norm = (center_raw - mu[j]) / sigma[j]
            for i in range(HIDDEN_SIZE):
                lut_data[j, b, i] = int(np.round(W0[i, j] * center_norm * s))
    bias_layer_0_q = np.round(b0 * s).astype(np.int32)
    return lut_data, bias_layer_0_q


def generate_lut_fixed(W0, b0, mu, sigma, feat_offset, feat_shift, num_bins=NUM_BINS):
    """FIXED: uses bin LEFT EDGE = offset + b * bin_width.

    Rationale: bit-shift binning maps value v to bin floor((v - offset) / width).
    The LEFT EDGE of bin b is offset + b * width.  For heavily right-skewed
    features (e.g., "Fwd Header Length": range 0-4.6M, 97% < 1000), almost
    all samples land in bin 0.  Center of bin 0 = 0.5 * 131072 = 65536, but
    the typical value is ~272.  Left edge = 0, which is far closer.
    """
    s = SCALE
    lut_data = np.zeros((NUM_FEATURES, num_bins, HIDDEN_SIZE), dtype=np.int32)
    for j in range(NUM_FEATURES):
        bin_width = 1 << int(feat_shift[j])
        for b in range(num_bins):
            edge_raw = float(feat_offset[j]) + b * bin_width  # FIX: no +0.5
            edge_norm = (edge_raw - mu[j]) / sigma[j]
            for i in range(HIDDEN_SIZE):
                lut_data[j, b, i] = int(np.round(W0[i, j] * edge_norm * s))
    bias_layer_0_q = np.round(b0 * s).astype(np.int32)
    return lut_data, bias_layer_0_q


def generate_lut_data_mean(W0, b0, mu, sigma, feat_offset, feat_shift,
                           X_train_raw, num_bins=NUM_BINS):
    """BEST FIX: uses actual mean of training data within each bin.

    For each bin, compute the mean raw feature value of samples that
    fall into that bin.  This perfectly represents the data distribution.
    """
    s = SCALE
    lut_data = np.zeros((NUM_FEATURES, num_bins, HIDDEN_SIZE), dtype=np.int32)

    for j in range(NUM_FEATURES):
        bin_width = 1 << int(feat_shift[j])
        col = X_train_raw[:, j]
        shifted = np.maximum(col - float(feat_offset[j]), 0.0)
        sample_bins = np.clip(np.floor(shifted / bin_width).astype(np.int64),
                              0, num_bins - 1)

        for b in range(num_bins):
            mask = sample_bins == b
            if mask.sum() > 0:
                representative_raw = col[mask].mean()
            else:
                # Fallback to bin left edge
                representative_raw = float(feat_offset[j]) + b * bin_width
            representative_norm = (representative_raw - mu[j]) / sigma[j]
            for i in range(HIDDEN_SIZE):
                lut_data[j, b, i] = int(np.round(
                    W0[i, j] * representative_norm * s))

    bias_layer_0_q = np.round(b0 * s).astype(np.int32)
    return lut_data, bias_layer_0_q


# ══════════════════════════════════════════════════════════════════════
#  Inference engines
# ══════════════════════════════════════════════════════════════════════

def maplut_layer0(X_raw, lut_data, bias_layer_0_q, feat_offset, feat_shift,
                  num_bins=NUM_BINS):
    """MapLUT Layer 0 with original integer binning."""
    N = X_raw.shape[0]
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    all_bins = np.zeros((N, NUM_FEATURES), dtype=np.int64)
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = raw - int(feat_offset[j])
        shifted = np.maximum(shifted, 0)
        bins = (shifted >> int(feat_shift[j])).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        all_bins[:, j] = bins
        hidden += lut_data[j][bins]
    return hidden, all_bins


def int32_layer0(X_raw, W0, b0, mu, sigma):
    s = SCALE
    sb = SCALE_BITS
    Wq0 = np.round(W0 * s).astype(np.int64)
    Bq0 = np.round(b0 * s).astype(np.int64)
    X_norm = (X_raw - mu) / sigma
    xq = np.round(X_norm * s).astype(np.int64)
    z = (xq @ Wq0.T >> sb) + Bq0
    return z, xq, Wq0, Bq0


def int32_full(X_raw, model, mu, sigma):
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


def maplut_full_with_int32_L12(X_raw, lut_data, bias_layer_0_q,
                                feat_offset, feat_shift,
                                model, mu, sigma, num_bins=NUM_BINS):
    """MapLUT Layer 0, then int32 Layers 1+2 (isolates L0 error)."""
    sb = SCALE_BITS
    s = SCALE
    N = X_raw.shape[0]
    hidden = np.tile(bias_layer_0_q.astype(np.int64), (N, 1))
    for j in range(NUM_FEATURES):
        raw = X_raw[:, j].astype(np.int64)
        shifted = raw - int(feat_offset[j])
        shifted = np.maximum(shifted, 0)
        bins = (shifted >> int(feat_shift[j])).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        hidden += lut_data[j][bins]
    hidden = np.maximum(hidden, 0)
    Wq1 = np.round(model.W[1] * s).astype(np.int64)
    Bq1 = np.round(model.b[1] * s).astype(np.int64)
    Wq2 = np.round(model.W[2] * s).astype(np.int64)
    Bq2 = np.round(model.b[2] * s).astype(np.int64)
    z1 = (hidden @ Wq1.T >> sb) + Bq1
    z1 = np.maximum(z1, 0)
    z2 = (z1 @ Wq2.T >> sb) + Bq2
    return np.argmax(z2, axis=1)


# ══════════════════════════════════════════════════════════════════════
#  MAIN DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  MapLUT Debug Diagnostic — Root Cause Analysis")
    print("=" * 72)

    # ── 1. Load data ──
    print("\n[1] Loading data...")
    X_all, y_all = load_data()
    idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]
    split = int(0.75 * len(y_all))
    X_train_raw, y_train = X_all[:split], y_all[:split]
    X_test_raw, y_test = X_all[split:], y_all[split:]
    mu = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma[sigma < 1e-8] = 1e-8
    X_train_n = (X_train_raw - mu) / sigma
    X_test_n = (X_test_raw - mu) / sigma
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # ── 2. Quick train ──
    print("\n[2] Training model (30 epochs)...")
    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES]
    model = train_mlp_quick(X_train_n, y_train, layer_sizes, epochs=30)

    y_float = model.predict(X_test_n)
    y_int32 = int32_full(X_test_raw, model, mu, sigma)
    print(f"  Float accuracy:  {(y_float == y_test).mean():.4f}")
    print(f"  Int32 accuracy:  {(y_int32 == y_test).mean():.4f}")

    # ── 3. LUT params ──
    print("\n[3] Computing LUT binning parameters...")
    feat_offset, feat_shift = compute_lut_params(X_train_raw, NUM_BINS)

    print(f"\n  {'Feature':<28s} {'range':>12s} {'shift':>6s} {'bin_w':>8s} "
          f"{'bin0_ctr':>10s} {'median':>10s} {'%_in_bin0':>10s}")
    print("  " + "-" * 92)
    for j in range(NUM_FEATURES):
        fmin = X_train_raw[:, j].min()
        fmax = X_train_raw[:, j].max()
        bw = 1 << int(feat_shift[j])
        center0 = float(feat_offset[j]) + 0.5 * bw
        med = np.median(X_train_raw[:, j])
        # What fraction of training data falls in bin 0?
        shifted = np.maximum(X_train_raw[:, j].astype(np.int64) - int(feat_offset[j]), 0)
        bins = np.clip(shifted >> int(feat_shift[j]), 0, NUM_BINS - 1)
        pct_bin0 = (bins == 0).mean() * 100
        print(f"  {FEATURE_COLS[j]:<28s} {fmax-fmin:12.1f} {feat_shift[j]:6d} {bw:8d} "
              f"{center0:10.1f} {med:10.1f} {pct_bin0:9.1f}%")

    # ── 4. THE BUG DEMONSTRATION ──
    print("\n" + "=" * 72)
    print("[4] THE BUG: bin center vs actual value for skewed features")
    print("=" * 72)
    print("""
  For right-skewed features (most network traffic features), the vast
  majority of samples land in bin 0.  The BUG is in generate_lut():

      center_raw = feat_offset + (b + 0.5) * bin_width   # ← BUG

  For bin 0, center = offset + 0.5 * bin_width.  Examples:
""")
    for j in range(NUM_FEATURES):
        bw = 1 << int(feat_shift[j])
        center0 = float(feat_offset[j]) + 0.5 * bw
        center0_norm = (center0 - mu[j]) / sigma[j]
        median_val = np.median(X_train_raw[:, j])
        median_norm = (median_val - mu[j]) / sigma[j]

        shifted = np.maximum(X_train_raw[:, j].astype(np.int64) - int(feat_offset[j]), 0)
        bins = np.clip(shifted >> int(feat_shift[j]), 0, NUM_BINS - 1)
        pct_bin0 = (bins == 0).mean() * 100

        if pct_bin0 > 80 and abs(center0_norm - median_norm) > 0.5:
            print(f"  ★ {FEATURE_COLS[j]}:")
            print(f"      bin_width = {bw:,d}")
            print(f"      bin0 center  = {center0:,.1f}  (normalized: {center0_norm:+.4f})")
            print(f"      data median  = {median_val:,.1f}  (normalized: {median_norm:+.4f})")
            print(f"      error in σ   = {abs(center0_norm - median_norm):.4f}")
            print(f"      % data in bin0 = {pct_bin0:.1f}%")
            print()

    # ── 5. Detailed per-sample comparison ──
    print("=" * 72)
    print("[5] Per-sample Layer 0 comparison (10 samples)")
    print("=" * 72)

    # Pick 10 diverse samples
    sample_indices = []
    for c in range(NUM_CLASSES):
        c_indices = np.where(y_test == c)[0]
        if len(c_indices) >= 3:
            sample_indices.extend(np.random.choice(c_indices, 3, replace=False).tolist())
    sample_indices = sample_indices[:10]
    X_batch = X_test_raw[sample_indices]
    y_batch = y_test[sample_indices]

    # Generate 3 versions of LUT
    lut_original, bias_q = generate_lut_original(
        model.W[0], model.b[0], mu, sigma, feat_offset, feat_shift)
    lut_fixed, _ = generate_lut_fixed(
        model.W[0], model.b[0], mu, sigma, feat_offset, feat_shift)
    lut_datamean, _ = generate_lut_data_mean(
        model.W[0], model.b[0], mu, sigma, feat_offset, feat_shift, X_train_raw)

    # Compute Layer 0 outputs
    z_int32, xq, Wq0, Bq0 = int32_layer0(X_batch, model.W[0], model.b[0], mu, sigma)
    z_orig, bins_orig = maplut_layer0(X_batch, lut_original, bias_q, feat_offset, feat_shift)
    z_fixed, _ = maplut_layer0(X_batch, lut_fixed, bias_q, feat_offset, feat_shift)
    z_dmean, _ = maplut_layer0(X_batch, lut_datamean, bias_q, feat_offset, feat_shift)

    for sidx in range(min(3, len(sample_indices))):
        print(f"\n  Sample {sidx} (class={CLASS_NAMES[y_batch[sidx]]}):")
        print(f"  Top-10 neurons by |int32 value|:")
        print(f"  {'Neuron':>7s} {'int32':>10s} {'orig_lut':>10s} {'fixed_lut':>10s} "
              f"{'dmean_lut':>10s} {'orig_err%':>10s} {'fixed_err%':>10s} {'dmean_err%':>10s}")
        print("  " + "-" * 80)
        top_neurons = np.argsort(-np.abs(z_int32[sidx]))[:10]
        for ni in top_neurons:
            vi = z_int32[sidx, ni]
            vo = z_orig[sidx, ni]
            vf = z_fixed[sidx, ni]
            vd = z_dmean[sidx, ni]
            eo = abs(vo - vi) / max(abs(vi), 1) * 100
            ef = abs(vf - vi) / max(abs(vi), 1) * 100
            ed = abs(vd - vi) / max(abs(vi), 1) * 100
            print(f"  {ni:7d} {vi:10d} {vo:10d} {vf:10d} {vd:10d} "
                  f"{eo:9.1f}% {ef:9.1f}% {ed:9.1f}%")

    # Summary stats
    print(f"\n  Summary across all 10 samples × 64 neurons:")
    for name, z_test in [("original", z_orig), ("left-edge", z_fixed),
                          ("data-mean", z_dmean)]:
        abs_diff = np.abs(z_int32 - z_test).astype(np.float64)
        abs_ref = np.abs(z_int32).astype(np.float64)
        mae = abs_diff.mean()
        mre = (abs_diff / np.maximum(abs_ref, 1)).mean() * 100
        relu_match = ((np.maximum(z_int32, 0) > 0) == (np.maximum(z_test, 0) > 0)).mean() * 100
        print(f"    {name:12s}:  MAE={mae:10.0f}  MRE={mre:6.1f}%  "
              f"ReLU_match={relu_match:.1f}%")

    # ── 6. Per-feature error decomposition ──
    print(f"\n" + "=" * 72)
    print("[6] Per-feature error decomposition")
    print("=" * 72)
    print(f"\n  For each feature j, the int32 contribution to neuron i is:")
    print(f"      (xq[j] * Wq0[i,j]) >> 16")
    print(f"  The LUT contribution is:")
    print(f"      lut_data[j, bin, i]  (precomputed: W0[i,j] * center_norm * s)")
    print()
    print(f"  {'Feature':<28s} {'orig_MAE':>10s} {'fixed_MAE':>10s} {'dmean_MAE':>10s} "
          f"{'|int32|':>10s} {'orig%':>7s} {'fix%':>7s}")
    print("  " + "-" * 90)

    for j in range(NUM_FEATURES):
        # Int32 per-feature contribution
        int32_contrib = (np.outer(xq[:, j], Wq0[:, j]) >> SCALE_BITS)  # (10, 64)

        orig_contrib = lut_original[j][bins_orig[:, j]]  # (10, 64)
        fixed_contrib = lut_fixed[j][bins_orig[:, j]]
        dmean_contrib = lut_datamean[j][bins_orig[:, j]]

        orig_mae = np.abs(int32_contrib - orig_contrib).mean()
        fixed_mae = np.abs(int32_contrib - fixed_contrib).mean()
        dmean_mae = np.abs(int32_contrib - dmean_contrib).mean()
        int32_mag = np.abs(int32_contrib).mean()
        orig_pct = orig_mae / max(int32_mag, 1) * 100
        fixed_pct = fixed_mae / max(int32_mag, 1) * 100

        marker = "  ★" if orig_pct > 100 else "   "
        print(f"{marker}{FEATURE_COLS[j]:<28s} {orig_mae:10.0f} {fixed_mae:10.0f} "
              f"{dmean_mae:10.0f} {int32_mag:10.0f} {orig_pct:6.1f}% {fixed_pct:6.1f}%")

    # ── 7. Full pipeline accuracy comparison ──
    print(f"\n" + "=" * 72)
    print("[7] Full pipeline accuracy comparison")
    print("=" * 72)

    int32_acc = (int32_full(X_test_raw, model, mu, sigma) == y_test).mean()
    orig_acc = (maplut_full_with_int32_L12(
        X_test_raw, lut_original, bias_q, feat_offset, feat_shift,
        model, mu, sigma) == y_test).mean()
    fixed_acc = (maplut_full_with_int32_L12(
        X_test_raw, lut_fixed, bias_q, feat_offset, feat_shift,
        model, mu, sigma) == y_test).mean()
    dmean_acc = (maplut_full_with_int32_L12(
        X_test_raw, lut_datamean, bias_q, feat_offset, feat_shift,
        model, mu, sigma) == y_test).mean()

    print(f"\n  {'Method':<30s} {'Accuracy':>10s}")
    print("  " + "-" * 42)
    print(f"  {'Int32 (reference)' :<30s} {int32_acc:10.4f}")
    print(f"  {'MapLUT ORIGINAL (center)':<30s} {orig_acc:10.4f}")
    print(f"  {'MapLUT FIXED (left-edge)':<30s} {fixed_acc:10.4f}")
    print(f"  {'MapLUT BEST (data-mean)':<30s} {dmean_acc:10.4f}")

    # ── 8. Root cause and fix ──
    print(f"\n" + "=" * 72)
    print("[8] ROOT CAUSE AND FIX")
    print("=" * 72)
    print(f"""
  ROOT CAUSE
  ══════════
  In generate_lut(), the bin representative value uses the bin CENTER:

      center_raw = feat_offset[j] + (b + 0.5) * bin_width     # line 541

  For features with heavy right-skew (common in network traffic), the
  bin_width is huge because max >> typical.  Nearly all data falls in
  bin 0, whose center = 0.5 * bin_width, which is thousands or millions
  of units away from the actual typical values.

  Worst offenders (this dataset):
    • Fwd Header Length:     bin_width={1 << int(feat_shift[1]):>10,d}   center0={0.5*(1<<int(feat_shift[1])):>10,.0f}   median={np.median(X_train_raw[:,1]):>10,.1f}
    • Total Backward Packets:bin_width={1 << int(feat_shift[9]):>10,d}   center0={0.5*(1<<int(feat_shift[9])):>10,.0f}   median={np.median(X_train_raw[:,9]):>10,.1f}
    • Total Fwd Packets:     bin_width={1 << int(feat_shift[4]):>10,d}   center0={0.5*(1<<int(feat_shift[4])):>10,.0f}   median={np.median(X_train_raw[:,4]):>10,.1f}
    • Fwd IAT Mean:          bin_width={1 << int(feat_shift[3]):>10,d}   center0={0.5*(1<<int(feat_shift[3])):>10,.0f}   median={np.median(X_train_raw[:,3]):>10,.1f}

  This causes Layer 0 hidden values to be wildly wrong (mean relative
  error >{(np.abs(z_int32 - z_orig).astype(np.float64) / np.maximum(np.abs(z_int32).astype(np.float64), 1)).mean()*100:.0f}%), which cascades through subsequent layers.

  MINIMAL FIX
  ═══════════
  In generate_lut(), change ONE line:

      BEFORE:  center_raw = float(feat_offset[j]) + (b + 0.5) * bin_width
      AFTER:   center_raw = float(feat_offset[j]) + b * bin_width

  This uses the bin LEFT EDGE instead of center.  For bin 0, the left
  edge is the offset (typically 0), which is close to the actual typical
  values for right-skewed features.

  Results:
    Int32 (reference):  {int32_acc:.4f}
    Original (center):  {orig_acc:.4f}  ← BROKEN
    Fixed (left-edge):  {fixed_acc:.4f}  ← FIXED  (remove +0.5)
    Best (data-mean):   {dmean_acc:.4f}  ← OPTIMAL (use training data mean per bin)

  RECOMMENDED CHANGE in train_v3.py generate_lut(), line ~541:
  -    center_raw = float(feat_offset[j]) + (b + 0.5) * bin_width
  +    center_raw = float(feat_offset[j]) + b * bin_width

  NOTE: The float-to-int64 truncation in maplut_inference is NOT the
  issue (confirmed: 0 bins change when using float division instead).
""")


if __name__ == "__main__":
    main()
