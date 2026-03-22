"""
Bins sweep: properly-trained float model + MapLUT+Int32 inference.
Uses int64 for LUT values to avoid overflow, then clamps to int32 range for inference.
"""
import numpy as np, pyarrow.parquet as pq, time, json, sys
from pathlib import Path

np.random.seed(42)

FEATURE_COLS = [
    "Fwd Packet Length Mean", "Fwd Header Length", "Avg Packet Size",
    "Fwd IAT Mean", "Total Fwd Packets", "Fwd Packet Length Max",
    "Init Bwd Win Bytes", "Init Fwd Win Bytes", "PSH Flag Count",
    "Total Backward Packets", "Fwd Packet Length Min", "Protocol",
]
LABEL_MAP = {
    "Benign": 0, "BENIGN": 0, "DDoS": 1, "DoS Hulk": 1, "DoS GoldenEye": 1,
    "DoS slowloris": 1, "DoS Slowhttptest": 1, "Heartbleed": 1,
    "PortScan": 2, "FTP-Patator": 3, "SSH-Patator": 3, "Bot": 3,
    "Infiltration": -1,
}
CN = ["BENIGN", "DDOS", "PORTSCAN", "BRUTEFORCE"]
S = 65536; SB = 16; H = 64; NF = 12; NC = 4

# ── Load data (vectorized) ──
print("Loading data...", flush=True)
t0 = time.time()
frames = []
for fp in sorted(Path("data/cicids_raw").glob("*.parquet")):
    t = pq.read_table(str(fp), columns=FEATURE_COLS + ["Label"])
    df = t.to_pandas()
    df["label_int"] = df["Label"].map(LABEL_MAP)
    df["label_int"] = __import__('pandas').to_numeric(df["label_int"], errors='coerce')
    df = df.dropna(subset=["label_int"])
    df = df[df["label_int"] >= 0]
    df["label_int"] = df["label_int"].astype(int)
    # Drop rows with non-finite feature values
    feat_df = df[FEATURE_COLS]
    mask = np.isfinite(feat_df.values).all(axis=1)
    df = df[mask]
    frames.append(df)

df_all = __import__("pandas").concat(frames, ignore_index=True)
X_all = df_all[FEATURE_COLS].values.astype(np.float64)
y_all = df_all["label_int"].values.astype(np.int64)

idx = np.random.permutation(len(y_all))
X_all, y_all = X_all[idx], y_all[idx]
split = int(0.75 * len(y_all))
Xtr, ytr = X_all[:split], y_all[:split]
Xte, yte = X_all[split:], y_all[split:]
print(f"Train: {len(ytr)}, Test: {len(yte)}, Time: {time.time()-t0:.1f}s", flush=True)

# ── Normalize ──
mu = Xtr.mean(0); sig = Xtr.std(0); sig[sig < 1e-8] = 1e-8
Xtr_n = (Xtr - mu) / sig
Xte_n = (Xte - mu) / sig

# ── Focal loss + oversampling training ──
def relu(x): return np.maximum(x, 0)
def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def focal_loss_grad(logits, y, gamma=3.0):
    p = softmax(logits)
    N = len(y)
    one_hot = np.zeros_like(p); one_hot[np.arange(N), y] = 1
    pt = (p * one_hot).sum(1, keepdims=True)
    focal_w = (1 - pt) ** gamma
    dz = focal_w * (p - one_hot)
    dz /= focal_w.sum()
    return dz

def oversample(X, y, target=100_000):
    Xo, yo = [X], [y]
    counts = np.bincount(y, minlength=NC)
    for c in range(NC):
        need = target - counts[c]
        if need > 0:
            idx_c = np.where(y == c)[0]
            rep = np.random.choice(idx_c, need, replace=True)
            Xo.append(X[rep]); yo.append(y[rep])
    X2 = np.concatenate(Xo); y2 = np.concatenate(yo)
    perm = np.random.permutation(len(y2))
    return X2[perm], y2[perm]

def train_float(Xtr_n, ytr, epochs=100, lr=0.001, gamma=3.0, os_target=100000):
    Xos, yos = oversample(Xtr_n, ytr, os_target)
    # Xavier init
    w0 = np.random.randn(NF, H) * np.sqrt(2.0/NF)
    b0 = np.zeros(H)
    w1 = np.random.randn(H, H) * np.sqrt(2.0/H)
    b1 = np.zeros(H)
    w2 = np.random.randn(H, NC) * np.sqrt(2.0/H)
    b2 = np.zeros(NC)
    BS = 512
    for ep in range(1, epochs+1):
        perm = np.random.permutation(len(yos))
        Xos, yos = Xos[perm], yos[perm]
        for i in range(0, len(yos), BS):
            xb = Xos[i:i+BS]; yb = yos[i:i+BS]
            # Forward
            z0 = xb @ w0 + b0; a0 = relu(z0)
            z1 = a0 @ w1 + b1; a1 = relu(z1)
            z2 = a1 @ w2 + b2
            # Backward
            dz2 = focal_loss_grad(z2, yb, gamma)
            dw2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ w2.T; dz1 = da1 * (z1 > 0)
            dw1 = a0.T @ dz1; db1 = dz1.sum(0)
            da0 = dz1 @ w1.T; dz0 = da0 * (z0 > 0)
            dw0 = xb.T @ dz0; db0 = dz0.sum(0)
            # Update
            w0 -= lr*dw0; b0 -= lr*db0
            w1 -= lr*dw1; b1 -= lr*db1
            w2 -= lr*dw2; b2 -= lr*db2
        if ep % 20 == 0:
            acc, mf1, pcf1 = evaluate_float(Xte_n, yte, w0, b0, w1, b1, w2, b2)
            print(f"  ep {ep:3d} acc={acc:.4f} mF1={mf1:.4f} PS={pcf1[2]:.4f} BF={pcf1[3]:.4f}", flush=True)
    return w0, b0, w1, b1, w2, b2

def evaluate_float(X, y, w0, b0, w1, b1, w2, b2):
    a0 = relu(X @ w0 + b0)
    a1 = relu(a0 @ w1 + b1)
    logits = a1 @ w2 + b2
    pred = logits.argmax(1)
    acc = (pred == y).mean()
    pcf1 = per_class_f1(y, pred)
    mf1 = np.mean(pcf1)
    return acc, mf1, pcf1

def per_class_f1(y, pred):
    f1s = []
    for c in range(NC):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        f1s.append(f1)
    return f1s

# ── Train ──
print("Training float model (100 epochs)...", flush=True)
t1 = time.time()
w0, b0, w1, b1, w2, b2 = train_float(Xtr_n, ytr)
print(f"Training done. Time: {time.time()-t1:.1f}s\n", flush=True)

# Float baseline
acc, mf1, pcf1 = evaluate_float(Xte_n, yte, w0, b0, w1, b1, w2, b2)
print(f"  Float   acc={acc:.4f} mF1={mf1:.4f} PS={pcf1[2]:.4f} BF={pcf1[3]:.4f}")

# Int32 baseline
def evaluate_int32(X, y, w0, b0, w1, b1, w2, b2):
    W0q = np.round(w0 * S).astype(np.int64)
    B0q = np.round(b0 * S).astype(np.int64)
    W1q = np.round(w1 * S).astype(np.int64)
    B1q = np.round(b1 * S).astype(np.int64)
    W2q = np.round(w2 * S).astype(np.int64)
    B2q = np.round(b2 * S).astype(np.int64)

    # Fixed-point features
    feat_offset = np.round(mu * S).astype(np.int64)
    feat_scale_inv = np.round((1.0/sig) * S).astype(np.int64)
    Xq = np.round(X * S).astype(np.int64)  # not normalized yet—wait, X is already normalized
    # Actually X is already normalized floats, so just quantize
    Xq = np.round(X * S).astype(np.int64)

    z0 = (Xq @ W0q) >> SB
    z0 = z0 + B0q
    a0 = np.maximum(z0, 0)
    z1 = (a0 @ W1q) >> SB
    z1 = z1 + B1q
    a1 = np.maximum(z1, 0)
    z2 = (a1 @ W2q) >> SB
    z2 = z2 + B2q
    pred = z2.argmax(1)
    acc = (pred == y).mean()
    pcf1 = per_class_f1(y, pred)
    return acc, np.mean(pcf1), pcf1

acc_i, mf1_i, pcf1_i = evaluate_int32(Xte_n, yte, w0, b0, w1, b1, w2, b2)
print(f"  Int32   acc={acc_i:.4f} mF1={mf1_i:.4f} PS={pcf1_i[2]:.4f} BF={pcf1_i[3]:.4f}")

# ── MapLUT+Int32 bins sweep ──
def compute_lut_int64(w0, b0, mu, sig, num_bins):
    """
    Compute LUT: for each feature j, bin b, hidden neuron h:
      lut[j][b][h] = round(w0[j,h] * center_of_bin(j,b) * S)
    where center_of_bin is the normalized value at the center of bin b.
    
    Returns int64 array, then we clamp to int32 for inference.
    """
    # For each feature j, compute bin edges in raw space
    # Bin b covers [lo + b*bw, lo + (b+1)*bw)
    # We use left edge: bin_center = lo + b * bw (matching the fixed bug)
    
    lut = np.zeros((NF, num_bins, H), dtype=np.int64)
    
    for j in range(NF):
        # Raw feature range from training data
        raw_min = Xtr[:, j].min()
        raw_max = Xtr[:, j].max()
        bw = (raw_max - raw_min) / num_bins
        
        for b in range(num_bins):
            # Left edge of bin in raw space
            raw_val = raw_min + b * bw
            # Normalize
            norm_val = (raw_val - mu[j]) / sig[j]
            # Quantize: contribution = w0[j,:] * norm_val * S
            contrib = np.round(w0[j, :] * norm_val * S).astype(np.int64)
            lut[j, b, :] = contrib
    
    return lut

def evaluate_maplut_int32(X_raw, X_norm, y, lut, b0, w1, b1, w2, b2, num_bins):
    """
    MapLUT Layer 0 + Int32 Layers 1,2.
    LUT values are int64, clamped to int32 range for accumulation.
    """
    N = len(y)
    INT32_MAX = 2**31 - 1
    INT32_MIN = -(2**31)
    
    B0q = np.round(b0 * S).astype(np.int64)
    W1q = np.round(w1 * S).astype(np.int64)
    B1q = np.round(b1 * S).astype(np.int64)
    W2q = np.round(w2 * S).astype(np.int64)
    B2q = np.round(b2 * S).astype(np.int64)
    
    # Compute bin indices for each sample
    hidden = np.zeros((N, H), dtype=np.int64)
    
    for j in range(NF):
        raw_min = Xtr[:, j].min()
        raw_max = Xtr[:, j].max()
        bw = (raw_max - raw_min) / num_bins
        if bw < 1e-15:
            bw = 1e-15
        
        # Bin index: floor((raw - raw_min) / bw), clamped
        bins = np.floor((X_raw[:, j] - raw_min) / bw).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        
        # LUT lookup and accumulate
        hidden += lut[j, bins, :]  # lut[j][bins[i]] for each sample
    
    # Add bias, clamp to int32, ReLU
    hidden = hidden + B0q
    hidden = np.clip(hidden, INT32_MIN, INT32_MAX)
    hidden = np.maximum(hidden, 0)
    
    # Layer 1: int32 matmul
    z1 = (hidden @ W1q) >> SB
    z1 = z1 + B1q
    z1 = np.clip(z1, INT32_MIN, INT32_MAX)
    a1 = np.maximum(z1, 0)
    
    # Layer 2: int32 matmul
    z2 = (a1 @ W2q) >> SB
    z2 = z2 + B2q
    
    pred = z2.argmax(1)
    acc = (pred == y).mean()
    pcf1 = per_class_f1(y, pred)
    return acc, np.mean(pcf1), pcf1

print("\n--- MapLUT+Int32 Bins Sweep ---", flush=True)
results = {}
for nb in [64, 128, 256, 512]:
    t2 = time.time()
    print(f"  Computing LUT for {nb} bins...", flush=True)
    lut = compute_lut_int64(w0, b0, mu, sig, nb)
    
    # Check LUT value range
    lut_min, lut_max = lut.min(), lut.max()
    print(f"    LUT range: [{lut_min}, {lut_max}]", flush=True)
    
    # Clamp LUT to int32 range for realistic eBPF inference
    INT32_MAX = 2**31 - 1
    INT32_MIN = -(2**31)
    lut_clamped = np.clip(lut, INT32_MIN, INT32_MAX)
    overflow_count = ((lut < INT32_MIN) | (lut > INT32_MAX)).sum()
    if overflow_count > 0:
        print(f"    WARNING: {overflow_count} LUT entries clamped to int32 range", flush=True)
    
    acc, mf1, pcf1 = evaluate_maplut_int32(Xte, Xte_n, yte, lut_clamped, b0, w1, b1, w2, b2, nb)
    elapsed = time.time() - t2
    print(f"    bins={nb:4d}  acc={acc:.4f}  mF1={mf1:.4f}  PS={pcf1[2]:.4f}  BF={pcf1[3]:.4f}  ({elapsed:.1f}s)", flush=True)
    results[str(nb)] = {
        "bins": nb,
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "per_class_f1": {CN[i]: float(pcf1[i]) for i in range(NC)},
        "lut_range": [int(lut_min), int(lut_max)],
        "overflow_count": int(overflow_count),
        "lut_memory_KB": int(NF * nb * H * 4 / 1024),
    }

# Also add float and int32 baselines
results["float"] = {
    "accuracy": float(acc), "macro_f1": float(mf1),
    "per_class_f1": {CN[i]: float(pcf1[i]) for i in range(NC)},
}
# Re-evaluate for correct float values
acc_f, mf1_f, pcf1_f = evaluate_float(Xte_n, yte, w0, b0, w1, b1, w2, b2)
results["float"] = {
    "accuracy": float(acc_f), "macro_f1": float(mf1_f),
    "per_class_f1": {CN[i]: float(pcf1_f[i]) for i in range(NC)},
}
acc_i, mf1_i, pcf1_i = evaluate_int32(Xte_n, yte, w0, b0, w1, b1, w2, b2)
results["int32"] = {
    "accuracy": float(acc_i), "macro_f1": float(mf1_i),
    "per_class_f1": {CN[i]: float(pcf1_i[i]) for i in range(NC)},
}

# Save
out_path = Path("results/bins_sweep_float_model.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
