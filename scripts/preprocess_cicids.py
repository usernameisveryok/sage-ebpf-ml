#!/usr/bin/env python3
"""Preprocess CIC-IDS-2017 dataset for eBPF/XDP in-kernel ML inference.

Maps raw CIC-IDS-2017 CSV columns to the 6 features computable per-flow
inside an XDP program, cleans the data, and saves train/test splits as
numpy arrays.

Features (column order in output):
    0  pkt_len          — average forward packet length
    1  hdr_len          — average forward header length, clamped [20, 60]
    2  dst_port         — destination port, clamped [0, 65535]
    3  fwd_iat          — forward inter-arrival time (scaled)
    4  total_fwd_pkts   — total forward packets
    5  fwd_pkt_len_max  — max forward packet length

Labels:
    0  BENIGN
    1  DDoS / DoS
    2  PortScan
    3  Brute-force (FTP-Patator, SSH-Patator)

Usage:
    python3 scripts/preprocess_cicids.py --data-dir /path/to/cicids/csvs
    python3 scripts/preprocess_cicids.py --csv-file /path/to/single/file.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOWNLOAD_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

# Columns we need from the CIC-IDS-2017 CSVs (after stripping whitespace).
REQUIRED_COLS = [
    "Destination Port",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Header Length",
    "Fwd IAT Mean",
    "Total Fwd Packets",
    "Label",
]

# Label string → integer class mapping.
# Labels not in this map are skipped (rare attack types like Bot,
# Infiltration, Web Attack variants, Heartbleed).
LABEL_MAP = {
    "BENIGN":           0,
    "DDoS":             1,
    "DoS Hulk":         1,
    "DoS GoldenEye":    1,
    "DoS slowloris":    1,
    "DoS Slowhttptest": 1,
    "PortScan":         2,
    "FTP-Patator":      3,
    "SSH-Patator":      3,
}

# Days for time-based train/test split.
TRAIN_DAYS = {"monday", "tuesday", "wednesday"}
TEST_DAYS = {"thursday", "friday"}

CLASS_NAMES = {
    0: "BENIGN",
    1: "DDoS/DoS",
    2: "PortScan",
    3: "BruteForce",
}

# Feature clamping ranges: (min, max).
# Order: pkt_len, hdr_len, dst_port, fwd_iat, total_fwd_pkts, fwd_pkt_len_max
FEATURE_CLAMPS = [
    (0.0,  65535.0),   # pkt_len
    (20.0, 60.0),      # hdr_len
    (0.0,  65535.0),   # dst_port
    (0.0,  1e9),       # fwd_iat (µs) — cap at ~16 min
    (1.0,  1e7),       # total_fwd_pkts
    (0.0,  65535.0),   # fwd_pkt_len_max
]

FEATURE_NAMES = [
    "pkt_len", "hdr_len", "dst_port",
    "fwd_iat", "total_fwd_pkts", "fwd_pkt_len_max",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(s):
    """Convert a string to float, returning NaN for unparseable / Inf values."""
    s = s.strip()
    if not s:
        return float("nan")
    low = s.lower()
    if low in ("inf", "-inf", "+inf", "infinity", "-infinity", "nan"):
        return float("nan")
    try:
        v = float(s)
        # Reject infinities that survived parsing (e.g. very large exponents).
        if math.isinf(v):
            return float("nan")
        return v
    except ValueError:
        return float("nan")


def _day_from_filename(path):
    """Extract the lowercase day name from a CIC-IDS-2017 filename.

    Filenames look like 'Monday-WorkingHours.pcap_ISCX.csv'.
    Returns None if no day is found.
    """
    base = os.path.basename(path).lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday"):
        if day in base:
            return day
    return None


def _resolve_columns(header):
    """Map stripped header names to column indices.

    CIC-IDS-2017 headers often have leading/trailing whitespace.
    Also handles the 'Fwd Header Length.1' variant.

    Returns:
        (col_map, None) on success — col_map is {canonical_name: index}.
        (None, missing_name) on failure.
    """
    stripped = [h.strip() for h in header]
    col_map = {}

    for name in REQUIRED_COLS:
        if name in stripped:
            col_map[name] = stripped.index(name)
        elif name == "Fwd Header Length" and "Fwd Header Length.1" in stripped:
            # Some versions of the dataset rename this column.
            col_map[name] = stripped.index("Fwd Header Length.1")
        else:
            return None, name

    return col_map, None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_csv(filepath, stats):
    """Process a single CIC-IDS-2017 CSV file.

    Reads the file row-by-row (memory efficient), extracts and cleans
    the 6 features, and maps labels to integer classes.

    Args:
        filepath: Path to the CSV file.
        stats:    Dict accumulating skip-reason counters across files.

    Returns:
        (X, y) — numpy arrays with shapes (N, 6) float32 and (N,) int32,
        or (None, None) if the file is unusable.
    """
    print(f"  Processing: {os.path.basename(filepath)} ...", end=" ", flush=True)

    rows_X = []
    rows_y = []
    skipped = defaultdict(int)

    # Open with utf-8-sig to transparently strip a UTF-8 BOM if present.
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        # --- Header ---
        try:
            header = next(reader)
        except StopIteration:
            print("SKIP (empty file)")
            return None, None

        col_map, missing = _resolve_columns(header)
        if col_map is None:
            print(f"SKIP (missing column: {missing})")
            return None, None

        # Pre-lookup column indices for speed.
        idx_dst_port  = col_map["Destination Port"]
        idx_fwd_max   = col_map["Fwd Packet Length Max"]
        idx_fwd_mean  = col_map["Fwd Packet Length Mean"]
        idx_fwd_hdr   = col_map["Fwd Header Length"]
        idx_fwd_iat   = col_map["Fwd IAT Mean"]
        idx_total_fwd = col_map["Total Fwd Packets"]
        idx_label     = col_map["Label"]
        max_idx       = max(col_map.values())

        # --- Rows ---
        for row in reader:
            # Guard against truncated rows.
            if len(row) <= max_idx:
                skipped["short_row"] += 1
                continue

            # ── Label ──────────────────────────────────────────────
            label_str = row[idx_label].strip()
            if label_str not in LABEL_MAP:
                skipped[f"unknown_label:{label_str}"] += 1
                continue
            y_val = LABEL_MAP[label_str]

            # ── Parse raw numeric values ──────────────────────────
            dst_port  = _safe_float(row[idx_dst_port])
            fwd_max   = _safe_float(row[idx_fwd_max])
            fwd_mean  = _safe_float(row[idx_fwd_mean])
            fwd_hdr   = _safe_float(row[idx_fwd_hdr])
            fwd_iat   = _safe_float(row[idx_fwd_iat])
            total_fwd = _safe_float(row[idx_total_fwd])

            raw_vals = [dst_port, fwd_max, fwd_mean, fwd_hdr, fwd_iat,
                        total_fwd]

            # ── Validity checks ───────────────────────────────────
            if any(math.isnan(v) for v in raw_vals):
                skipped["nan_or_inf"] += 1
                continue
            if any(v < 0 for v in raw_vals):
                skipped["negative"] += 1
                continue
            if fwd_mean <= 0:
                skipped["fwd_mean_le0"] += 1
                continue
            if total_fwd <= 0:
                skipped["total_fwd_le0"] += 1
                continue

            # ── Compute 6 features ────────────────────────────────

            # 0: pkt_len ← Fwd Packet Length Mean
            pkt_len = fwd_mean

            # 1: hdr_len ← Fwd Header Length / Total Fwd Packets
            #    Clamped to [20, 60] (valid IP+TCP header range).
            hdr_len = fwd_hdr / total_fwd
            hdr_len = max(20.0, min(60.0, hdr_len))

            # 2: dst_port ← Destination Port (uint16 range)
            dst_port_feat = max(0.0, min(65535.0, dst_port))

            # 3: fwd_iat ← Fwd IAT Mean / 1000
            #    CIC-IDS-2017 stores IAT in microseconds.  Dividing by
            #    1000 keeps values in a compact range for the model while
            #    retaining sub-millisecond resolution.
            fwd_iat_feat = fwd_iat / 1000.0

            # 4: total_fwd_pkts ← Total Fwd Packets (as-is)
            total_fwd_pkts = total_fwd

            # 5: fwd_pkt_len_max ← Fwd Packet Length Max (as-is)
            fwd_pkt_len_max = fwd_max

            features = [pkt_len, hdr_len, dst_port_feat,
                        fwd_iat_feat, total_fwd_pkts, fwd_pkt_len_max]

            # ── Clamp to safe ranges ──────────────────────────────
            for i, (lo, hi) in enumerate(FEATURE_CLAMPS):
                features[i] = max(lo, min(hi, features[i]))

            rows_X.append(features)
            rows_y.append(y_val)

    n = len(rows_X)
    print(f"{n:,} rows kept")

    # Accumulate skip stats.
    for reason, cnt in skipped.items():
        stats["skipped"][reason] += cnt

    if n == 0:
        return None, None

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_distribution(name, y):
    """Print per-class sample counts and percentages."""
    total = len(y)
    print(f"\n  {name}: {total:,} samples")
    for cls_id in sorted(CLASS_NAMES.keys()):
        cnt = int(np.sum(y == cls_id))
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"    class {cls_id} ({CLASS_NAMES[cls_id]:>12s}): "
              f"{cnt:>10,}  ({pct:5.1f}%)")


def print_feature_stats(name, X):
    """Print min/max/mean per feature for a quick sanity check."""
    print(f"\n  {name} feature stats:")
    print(f"    {'feature':<18s} {'min':>12s} {'max':>12s} {'mean':>12s}")
    for i, fname in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f"    {fname:<18s} {col.min():>12.2f} {col.max():>12.2f} "
              f"{col.mean():>12.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CIC-IDS-2017 for eBPF/XDP ML inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Dataset download: {DOWNLOAD_URL}")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-dir",
        help="Directory containing one or more CIC-IDS-2017 CSV files.")
    group.add_argument(
        "--csv-file",
        help="Path to a single CIC-IDS-2017 CSV file.")

    parser.add_argument(
        "--output-dir", default="data",
        help="Directory for output .npy files (default: data).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the fallback 80/20 split (default: 42).")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Collect CSV file paths
    # ------------------------------------------------------------------
    csv_files = []

    if args.csv_file:
        if not os.path.isfile(args.csv_file):
            print(f"Error: file not found: {args.csv_file}\n")
            print("The CIC-IDS-2017 dataset can be downloaded from:")
            print(f"  {DOWNLOAD_URL}")
            sys.exit(1)
        csv_files.append(args.csv_file)
    else:
        if not os.path.isdir(args.data_dir):
            print(f"Error: directory not found: {args.data_dir}\n")
            print("The CIC-IDS-2017 dataset can be downloaded from:")
            print(f"  {DOWNLOAD_URL}")
            sys.exit(1)
        for fname in sorted(os.listdir(args.data_dir)):
            if fname.lower().endswith(".csv"):
                csv_files.append(os.path.join(args.data_dir, fname))

    if not csv_files:
        print(f"Error: no CSV files found in {args.data_dir}\n")
        print("Expected files like 'Monday-WorkingHours.pcap_ISCX.csv'.")
        print("Download the CIC-IDS-2017 dataset from:")
        print(f"  {DOWNLOAD_URL}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s).\n")

    # ------------------------------------------------------------------
    # Process each CSV
    # ------------------------------------------------------------------
    train_Xs, train_ys = [], []
    test_Xs, test_ys   = [], []
    unassigned_Xs, unassigned_ys = [], []
    stats = {"skipped": defaultdict(int)}

    # Use day-based split only when multiple files are present.
    use_day_split = len(csv_files) > 1

    for fpath in csv_files:
        X, y = process_csv(fpath, stats)
        if X is None:
            continue

        if use_day_split:
            day = _day_from_filename(fpath)
            if day and day in TRAIN_DAYS:
                train_Xs.append(X)
                train_ys.append(y)
            elif day and day in TEST_DAYS:
                test_Xs.append(X)
                test_ys.append(y)
            else:
                # Day not recognized — keep aside.
                unassigned_Xs.append(X)
                unassigned_ys.append(y)
        else:
            unassigned_Xs.append(X)
            unassigned_ys.append(y)

    # ------------------------------------------------------------------
    # Build train / test arrays
    # ------------------------------------------------------------------
    if use_day_split and train_Xs and test_Xs:
        print("\nUsing time-based split (Mon–Wed → train, Thu–Fri → test).")

        # Fold any day-unrecognised files into training.
        if unassigned_Xs:
            print(f"  ({len(unassigned_Xs)} file(s) with unknown day "
                  "added to training set)")
            train_Xs.extend(unassigned_Xs)
            train_ys.extend(unassigned_ys)

        X_train = np.concatenate(train_Xs, axis=0)
        y_train = np.concatenate(train_ys, axis=0)
        X_test  = np.concatenate(test_Xs,  axis=0)
        y_test  = np.concatenate(test_ys,  axis=0)
    else:
        # Fallback: merge everything and do a random 80/20 split.
        all_Xs = train_Xs + test_Xs + unassigned_Xs
        all_ys = train_ys + test_ys + unassigned_ys

        if not all_Xs:
            print("\nError: no valid data rows extracted from any file.")
            sys.exit(1)

        X_all = np.concatenate(all_Xs, axis=0)
        y_all = np.concatenate(all_ys, axis=0)

        print(f"\nUsing random 80/20 split (seed={args.seed}).")

        rng = np.random.RandomState(args.seed)
        n = len(y_all)
        indices = rng.permutation(n)
        split = int(0.8 * n)

        X_train = X_all[indices[:split]]
        y_train = y_all[indices[:split]]
        X_test  = X_all[indices[split:]]
        y_test  = y_all[indices[split:]]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print_distribution("Train", y_train)
    print_distribution("Test",  y_test)

    print("\n" + "-" * 60)
    print("FEATURE STATISTICS")
    print("-" * 60)
    print_feature_stats("Train", X_train)
    print_feature_stats("Test",  X_test)

    if stats["skipped"]:
        print("\n" + "-" * 60)
        print("SKIPPED ROWS")
        print("-" * 60)
        for reason, cnt in sorted(stats["skipped"].items(),
                                  key=lambda kv: -kv[1]):
            print(f"    {reason}: {cnt:,}")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    paths = {
        "train_X": os.path.join(args.output_dir, "cicids_train_X.npy"),
        "train_y": os.path.join(args.output_dir, "cicids_train_y.npy"),
        "test_X":  os.path.join(args.output_dir, "cicids_test_X.npy"),
        "test_y":  os.path.join(args.output_dir, "cicids_test_y.npy"),
    }

    np.save(paths["train_X"], X_train)
    np.save(paths["train_y"], y_train)
    np.save(paths["test_X"],  X_test)
    np.save(paths["test_y"],  y_test)

    print(f"\nSaved to {args.output_dir}/:")
    print(f"  {paths['train_X']}  shape={X_train.shape}  dtype={X_train.dtype}")
    print(f"  {paths['train_y']}  shape={y_train.shape}  dtype={y_train.dtype}")
    print(f"  {paths['test_X']}   shape={X_test.shape}   dtype={X_test.dtype}")
    print(f"  {paths['test_y']}   shape={y_test.shape}   dtype={y_test.dtype}")
    print("\nDone.")


if __name__ == "__main__":
    main()
