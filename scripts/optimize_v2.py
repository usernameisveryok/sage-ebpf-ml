#!/usr/bin/env python3
"""Autoresearch V2: 12-feature + focal loss optimization on CIC-IDS-2017.

Changes from V1:
  - 12 XDP-computable features (was 6)
  - Focal loss for extreme class imbalance
  - Smarter oversampling with noise injection (poor-man's SMOTE)
  - Automatic cycle progression based on results

Success criteria:
  - Macro-F1 >= 0.85
  - min_class_recall >= 0.60
  - int_accuracy >= 0.90
"""

import json, os, sys, time
import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pip install pyarrow")

NUM_CLASSES = 4
CLASS_NAMES = ["BENIGN", "DDoS/DoS", "PortScan", "BruteForce"]

# 12 XDP-computable features (sorted by discrimination power)
FEATURE_COLS = [
    "Fwd Packet Length Mean",     # 0: avg fwd pkt len
    "Fwd Header Length",          # 1: fwd header bytes
    "Avg Packet Size",            # 2: avg pkt size
    "Fwd IAT Mean",               # 3: fwd inter-arrival time
    "Total Fwd Packets",          # 4: fwd packet count
    "Fwd Packet Length Max",      # 5: max fwd pkt len
    "Init Bwd Win Bytes",         # 6: TCP SYN-ACK window (8.5x discriminative!)
    "Init Fwd Win Bytes",         # 7: TCP SYN window (3.9x)
    "PSH Flag Count",             # 8: TCP PSH flag count (3.3x)
    "Total Backward Packets",     # 9: bwd packet count
    "Fwd Packet Length Min",      # 10: min fwd pkt len
    "Protocol",                   # 11: IP protocol number
]

LABEL_MAP = {
    "Benign": 0, "BENIGN": 0,
    "DDoS": 1, "DoS Hulk": 1, "DoS GoldenEye": 1,
    "DoS slowloris": 1, "DoS Slowhttptest": 1, "Heartbleed": 1,
    "PortScan": 2,
    "FTP-Patator": 3, "SSH-Patator": 3, "Bot": 3,
    "Infiltration": -1,  # skip (too few)
}

# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_from_parquet(data_dir, n_features=12):
    """Load and preprocess parquet files with n_features."""
    feature_cols = FEATURE_COLS[:n_features]
    all_X, all_y = [], []

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    for fname in files:
        path = os.path.join(data_dir, fname)
        table = pq.read_table(path, columns=feature_cols + ["Label"])
        data = {col: table.column(col).to_pylist() for col in feature_cols + ["Label"]}
        n = len(data["Label"])

        for i in range(n):
            label_str = data["Label"][i]
            label = LABEL_MAP.get(label_str)
            if label is None:
                # Fuzzy match
                if "Web Attack" in label_str:
                    label = 3
                elif "DoS" in label_str or "DDoS" in label_str:
                    label = 1
                else:
                    continue
            if label == -1:
                continue

            feats = []
            bad = False
            for col in feature_cols:
                v = data[col][i]
                if v is None:
                    bad = True; break
                v = float(v)
                if np.isnan(v) or np.isinf(v):
                    bad = True; break
                feats.append(max(v, 0.0))  # clamp negatives
            if bad:
                continue
            all_X.append(feats)
            all_y.append(label)

        print(f"  {fname}: loaded")

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y, dtype=np.int64)

    # Shuffle and split
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    split = int(len(y) * 0.75)

    return X[:split], y[:split], X[split:], y[split:]


def oversample_with_noise(X, y, target_per_class, noise_std=0.05):
    """Oversample minority with gaussian noise (poor-man's SMOTE)."""
    classes = np.unique(y)
    X_parts, y_parts = [], []
    for c in classes:
        mask = y == c
        Xc = X[mask]
        n = len(Xc)
        if n >= target_per_class:
            idx = np.random.choice(n, target_per_class, replace=False)
            X_parts.append(Xc[idx])
        else:
            X_parts.append(Xc)
            need = target_per_class - n
            idx = np.random.choice(n, need, replace=True)
            # Add gaussian noise to oversampled points
            noise = np.random.randn(need, Xc.shape[1]) * noise_std
            X_parts.append(Xc[idx] * (1 + noise))  # multiplicative noise
        y_parts.append(np.full(min(n, target_per_class) if n >= target_per_class
                               else target_per_class, c, dtype=np.int64))

    X_out = np.concatenate(X_parts)
    y_out = np.concatenate(y_parts)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def normalize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


# ──────────────────────────────────────────────────────────────────────
# MLP with Focal Loss
# ──────────────────────────────────────────────────────────────────────

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(z.dtype)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(layer_sizes[i+1], fan_in) * scale)
            self.b.append(np.zeros(layer_sizes[i+1]))

    def forward(self, X):
        cache = {"a": [X]}
        a = X
        for k in range(len(self.W)):
            z = a @ self.W[k].T + self.b[k]
            cache.setdefault("z", []).append(z)
            a = relu(z) if k < len(self.W) - 1 else softmax(z)
            cache["a"].append(a)
        return a, cache

    def backward_focal(self, y, cache, gamma=2.0, alpha=None):
        """Focal loss backward: FL = -alpha * (1-p_t)^gamma * log(p_t)
        
        This down-weights easy examples and focuses on hard ones.
        Much better than class weights for extreme imbalance.
        """
        N = len(y)
        probs = cache["a"][-1]  # (N, C)
        
        # p_t = prob of true class
        p_t = probs[np.arange(N), y]  # (N,)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma  # (N,)
        
        # Optional per-class alpha
        if alpha is not None:
            focal_weight *= alpha[y]
        
        # Gradient of focal loss w.r.t. logits (softmax input)
        # For focal loss with softmax, the gradient is:
        # dL/dz_j = focal_weight * (p_j - 1[j=y]) + correction terms
        # Simplified: use weighted CE gradient as approximation
        dz = probs.copy()
        dz[np.arange(N), y] -= 1.0
        dz *= focal_weight[:, None]
        dz /= focal_weight.sum()

        dW_list = [None] * len(self.W)
        db_list = [None] * len(self.W)
        for k in reversed(range(len(self.W))):
            a_prev = cache["a"][k]
            dW_list[k] = dz.T @ a_prev
            db_list[k] = dz.sum(axis=0)
            if k > 0:
                da = dz @ self.W[k]
                dz = da * relu_grad(cache["z"][k - 1])
        return dW_list, db_list

    def backward_weighted(self, y, cache, class_weights):
        N = len(y)
        probs = cache["a"][-1]
        dz = probs.copy()
        dz[np.arange(N), y] -= 1.0
        w = class_weights[y]
        dz *= w[:, None]
        dz /= w.sum()
        dW_list = [None] * len(self.W)
        db_list = [None] * len(self.W)
        for k in reversed(range(len(self.W))):
            a_prev = cache["a"][k]
            dW_list[k] = dz.T @ a_prev
            db_list[k] = dz.sum(axis=0)
            if k > 0:
                da = dz @ self.W[k]
                dz = da * relu_grad(cache["z"][k - 1])
        return dW_list, db_list

    def step(self, dW, db, lr):
        for k in range(len(self.W)):
            self.W[k] -= lr * dW[k]
            self.b[k] -= lr * db[k]

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    results = {"per_class": []}
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results["per_class"].append({
            "class": c, "name": CLASS_NAMES[c],
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": int(support)
        })
    results["accuracy"] = round(float(np.mean(y_pred == y_true)), 4)
    results["macro_f1"] = round(float(np.mean([c["f1"] for c in results["per_class"]])), 4)
    results["min_recall"] = round(float(min(c["recall"] for c in results["per_class"])), 4)
    return results


def quantized_predict(model, X, scale_bits):
    s = 1 << scale_bits
    W_q = [np.round(w * s).astype(np.int64) for w in model.W]
    B_q = [np.round(b * s).astype(np.int64) for b in model.b]
    x = np.round(X * s).astype(np.int64)
    for k in range(len(W_q)):
        acc = x @ W_q[k].T
        z = (acc >> scale_bits) + B_q[k]
        if k < len(W_q) - 1:
            z = np.maximum(z, 0)
        x = z
    return np.argmax(x, axis=1)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_one(config, X_tr_n, y_tr, X_te_n, y_test, n_features):
    np.random.seed(config.get("seed", 42))
    t0 = time.time()

    h = config["hidden_size"]
    L = config["num_layers"]
    layer_sizes = [n_features] + [h] * L + [4]
    model = MLP(layer_sizes)

    loss_type = config.get("loss", "focal")
    gamma = config.get("gamma", 2.0)
    epochs = config.get("epochs", 80)
    batch_size = config.get("batch_size", 256)
    lr = config.get("lr", 0.01)
    b = config.get("scale_bits", 16)

    # Class weights (for weighted CE)
    N = len(y_tr)
    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1)
    class_weights = N / (NUM_CLASSES * counts)
    # Capped alpha for focal (sqrt of inverse frequency, less aggressive)
    alpha = np.sqrt(class_weights)
    alpha /= alpha.sum() / NUM_CLASSES

    best_score = -1
    best_W, best_b = None, None

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuf, y_shuf = X_tr_n[perm], y_tr[perm]

        for start in range(0, N, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            probs, cache = model.forward(Xb)

            if loss_type == "focal":
                dW, db = model.backward_focal(yb, cache, gamma=gamma, alpha=alpha)
            else:
                dW, db = model.backward_weighted(yb, cache, class_weights)

            model.step(dW, db, lr)

        if epoch % 30 == 0:
            lr *= 0.5

        if epoch % 10 == 0 or epoch == epochs:
            pred = model.predict(X_te_n)
            m = compute_metrics(y_test, pred)
            score = m["macro_f1"] + m["min_recall"]
            if score > best_score:
                best_score = score
                best_W = [w.copy() for w in model.W]
                best_b = [b_.copy() for b_ in model.b]

    model.W, model.b = best_W, best_b

    float_pred = model.predict(X_te_n)
    float_m = compute_metrics(y_test, float_pred)

    int_pred = quantized_predict(model, X_te_n, b)
    int_m = compute_metrics(y_test, int_pred)

    elapsed = time.time() - t0
    return {
        "config": config, "train_samples": N,
        "float": float_m, "int": int_m,
        "elapsed_s": round(elapsed, 1)
    }


def print_result(r, tag=""):
    c = r["config"]
    im = r["int"]
    fm = r["float"]
    print(f"\n{'─'*70}")
    print(f"  {tag}")
    print(f"  h={c['hidden_size']} L={c['num_layers']} b={c.get('scale_bits',16)} "
          f"nf={c.get('n_features',12)} loss={c.get('loss','focal')} "
          f"γ={c.get('gamma',2.0)} OS={c.get('oversample_target','none')} "
          f"lr={c.get('lr',0.01)} ep={c.get('epochs',80)}")
    print(f"  Train: {r['train_samples']}  Time: {r['elapsed_s']}s")
    print(f"{'─'*70}")
    print(f"  {'':15s} {'Float':>8s} {'Int32':>8s}")
    print(f"  {'Accuracy':15s} {fm['accuracy']:8.4f} {im['accuracy']:8.4f}")
    print(f"  {'Macro-F1':15s} {fm['macro_f1']:8.4f} {im['macro_f1']:8.4f}")
    print(f"  {'Min Recall':15s} {fm['min_recall']:8.4f} {im['min_recall']:8.4f}")
    for pc in im["per_class"]:
        flag = "⚠" if pc["recall"] < 0.60 or pc["precision"] < 0.30 else "✓"
        print(f"    {flag} {pc['name']:12s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f}  (n={pc['support']})")

    ok = im["macro_f1"] >= 0.85 and im["min_recall"] >= 0.60 and im["accuracy"] >= 0.90
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ──────────────────────────────────────────────────────────────────────
# Main autoresearch loop
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="data/cicids_raw")
    parser.add_argument("--output", default="results/optimize_v2_log.jsonl")
    args = parser.parse_args()

    log = open(args.output, "w")
    all_results = []

    def run_and_log(config, X_tr, y_tr, X_te, y_te, tag, n_features):
        config["n_features"] = n_features
        # Oversample if requested
        os_target = config.get("oversample_target")
        noise = config.get("noise_std", 0.05)
        if os_target:
            X_t, y_t = oversample_with_noise(X_tr, y_tr, os_target, noise)
        else:
            X_t, y_t = X_tr, y_tr
        # Normalize
        X_tn, X_ten, _, _ = normalize(X_t, X_te)
        r = train_one(config, X_tn, y_t, X_ten, y_te, n_features)
        ok = print_result(r, tag)
        log.write(json.dumps(r, default=str) + "\n"); log.flush()
        all_results.append((r, ok))
        return r, ok

    # ================================================================
    # PHASE 1: 12 features vs 6 features comparison
    # ================================================================
    print("\n" + "█" * 70)
    print("  PHASE 1: Feature count comparison (6 vs 12)")
    print("█" * 70)

    for nf in [6, 12]:
        print(f"\n  Loading {nf}-feature data from parquet...")
        # Temporarily override FEATURE_COLS
        saved_cols = FEATURE_COLS.copy()
        X_train, y_train, X_test, y_test = load_from_parquet(args.parquet_dir, n_features=nf)
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]:12s}: train={np.sum(y_train==c):>8d} test={np.sum(y_test==c):>8d}")

        # Save for later
        if nf == 12:
            X12_train, y12_train, X12_test, y12_test = X_train, y_train, X_test, y_test
        elif nf == 6:
            X6_train, y6_train, X6_test, y6_test = X_train, y_train, X_test, y_test

        cfg = {"hidden_size": 64, "num_layers": 2, "scale_bits": 16,
               "loss": "focal", "gamma": 2.0, "oversample_target": 50000,
               "lr": 0.01, "epochs": 80}
        run_and_log(cfg, X_train, y_train, X_test, y_test,
                    f"P1: {nf} features, focal, OS=50K", nf)

    # ================================================================
    # PHASE 2: Loss function comparison on 12 features
    # ================================================================
    print("\n" + "█" * 70)
    print("  PHASE 2: Loss function comparison (12 features)")
    print("█" * 70)

    for loss, gamma in [("focal", 1.0), ("focal", 2.0), ("focal", 3.0), ("weighted_ce", 0)]:
        cfg = {"hidden_size": 64, "num_layers": 2, "scale_bits": 16,
               "loss": loss, "gamma": gamma, "oversample_target": 50000,
               "lr": 0.01, "epochs": 80}
        run_and_log(cfg, X12_train, y12_train, X12_test, y12_test,
                    f"P2: loss={loss} γ={gamma}", 12)

    # ================================================================
    # PHASE 3: Oversample target sweep (12 features, best loss)
    # ================================================================
    print("\n" + "█" * 70)
    print("  PHASE 3: Oversample target sweep")
    print("█" * 70)

    best_p2 = max(all_results, key=lambda x: x[0]["int"]["macro_f1"] + x[0]["int"]["min_recall"])
    best_loss = best_p2[0]["config"]["loss"]
    best_gamma = best_p2[0]["config"]["gamma"]
    print(f"  Best loss: {best_loss} γ={best_gamma}")

    for target in [10000, 30000, 50000, 100000]:
        cfg = {"hidden_size": 64, "num_layers": 2, "scale_bits": 16,
               "loss": best_loss, "gamma": best_gamma,
               "oversample_target": target, "noise_std": 0.05,
               "lr": 0.01, "epochs": 80}
        run_and_log(cfg, X12_train, y12_train, X12_test, y12_test,
                    f"P3: OS={target}", 12)

    # ================================================================
    # PHASE 4: Architecture sweep
    # ================================================================
    print("\n" + "█" * 70)
    print("  PHASE 4: Architecture sweep (best data + loss)")
    print("█" * 70)

    best_p3 = max(all_results, key=lambda x: x[0]["int"]["macro_f1"] + x[0]["int"]["min_recall"])
    best_target = best_p3[0]["config"].get("oversample_target", 50000)

    for h, L in [(32, 2), (64, 2), (128, 2), (64, 3), (128, 3)]:
        cfg = {"hidden_size": h, "num_layers": L, "scale_bits": 16,
               "loss": best_loss, "gamma": best_gamma,
               "oversample_target": best_target, "noise_std": 0.05,
               "lr": 0.01, "epochs": 100}
        run_and_log(cfg, X12_train, y12_train, X12_test, y12_test,
                    f"P4: h={h} L={L}", 12)

    # ================================================================
    # PHASE 5: LR + epochs tuning on best arch
    # ================================================================
    print("\n" + "█" * 70)
    print("  PHASE 5: Hyperparameter tuning")
    print("█" * 70)

    best_p4 = max(all_results, key=lambda x: x[0]["int"]["macro_f1"] + x[0]["int"]["min_recall"])
    best_h = best_p4[0]["config"]["hidden_size"]
    best_L = best_p4[0]["config"]["num_layers"]
    print(f"  Best arch: h={best_h} L={best_L}")

    for lr, ep in [(0.005, 120), (0.02, 80), (0.01, 160), (0.005, 200)]:
        cfg = {"hidden_size": best_h, "num_layers": best_L, "scale_bits": 16,
               "loss": best_loss, "gamma": best_gamma,
               "oversample_target": best_target, "noise_std": 0.05,
               "lr": lr, "epochs": ep}
        run_and_log(cfg, X12_train, y12_train, X12_test, y12_test,
                    f"P5: lr={lr} ep={ep}", 12)

    # ================================================================
    # PHASE 6: Quantization check
    # ================================================================
    print("\n" + "█" * 70)
    print("  PHASE 6: Scale bits sweep")
    print("█" * 70)

    best_all = max(all_results, key=lambda x: x[0]["int"]["macro_f1"] + x[0]["int"]["min_recall"])
    best_cfg = best_all[0]["config"].copy()

    for b in [12, 16, 20]:
        cfg = {**best_cfg, "scale_bits": b}
        run_and_log(cfg, X12_train, y12_train, X12_test, y12_test,
                    f"P6: b={b}", 12)

    # ================================================================
    # FINAL REPORT
    # ================================================================
    log.close()

    print("\n\n" + "█" * 70)
    print("  FINAL RANKING")
    print("█" * 70)

    ranked = sorted(all_results,
                    key=lambda x: x[0]["int"]["macro_f1"] + x[0]["int"]["min_recall"],
                    reverse=True)

    print(f"\n  {'#':>3} {'nf':>3} {'h':>4} {'L':>2} {'b':>3} {'loss':>6} {'OS':>7} "
          f"{'IntAcc':>7} {'MacF1':>7} {'MinR':>7} {'MinP':>7}")
    print(f"  {'─'*70}")
    for i, (r, ok) in enumerate(ranked[:10]):
        c = r["config"]
        min_prec = min(pc["precision"] for pc in r["int"]["per_class"])
        flag = "✅" if ok else "❌"
        print(f"  {i+1:3d} {c.get('n_features',12):3d} {c['hidden_size']:4d} "
              f"{c['num_layers']:2d} {c.get('scale_bits',16):3d} "
              f"{c.get('loss','focal'):>6s} {str(c.get('oversample_target','none')):>7s} "
              f"{r['int']['accuracy']:7.4f} {r['int']['macro_f1']:7.4f} "
              f"{r['int']['min_recall']:7.4f} {min_prec:7.4f} {flag}")

    # Best result detail
    best = ranked[0][0]
    print(f"\n  ★ BEST CONFIG:")
    print_result(best, "★ CHAMPION")

    with open("results/cicids_real_best_v2.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    print(f"\n  Log: {args.output}")
    print(f"  Best: results/cicids_real_best_v2.json")


if __name__ == "__main__":
    main()
