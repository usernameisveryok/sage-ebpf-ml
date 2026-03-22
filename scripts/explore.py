#!/usr/bin/env python3
"""
explore.py — Automated design-space exploration for eBPF-ML inference.

Sweeps over (hidden_size, num_layers, scale_bits) configurations, trains
a quantized MLP for each, optionally compiles the eBPF program (when
num_layers == 2), and records accuracy, memory, instruction count, and
overflow-safety analysis.

Outputs:
    results/experiment_log.jsonl     — one JSON object per experiment
    results/exploration_summary.json — aggregated summary + Pareto front

Dependencies: numpy, Python stdlib only.

Usage:
    python3 scripts/explore.py                  # synthetic data
    python3 scripts/explore.py --data-dir data  # real CIC-IDS data
    python3 scripts/explore.py --dry-run        # show plan without running
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Search space
# ──────────────────────────────────────────────────────────────────────
HIDDEN_SIZES = [8, 16, 32, 64]
NUM_LAYERS_OPTIONS = [1, 2, 3]
SCALE_BITS_OPTIONS = [8, 12, 16, 20]

# Fixed training hyper-parameters (ensure fair comparison)
EPOCHS = 80
LR = 0.05
BATCH_SIZE = 256


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def config_key(hidden_size: int, num_layers: int, scale_bits: int) -> str:
    """Canonical string key for a configuration (used for dedup)."""
    return f"h{hidden_size}_l{num_layers}_b{scale_bits}"


def load_completed(log_path: str) -> set:
    """Read the JSONL log and return the set of already-completed keys."""
    done = set()
    if not os.path.exists(log_path):
        return done
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = config_key(rec["hidden_size"],
                                 rec["num_layers"],
                                 rec["scale_bits"])
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def fmt_time(seconds: float) -> str:
    """Format seconds as Xm Ys."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ──────────────────────────────────────────────────────────────────────
# Overflow safety analysis
# ──────────────────────────────────────────────────────────────────────

# Conservative estimates for He-initialized + ReLU networks:
#   max |weight_float|   ≈ 3.0   (3σ of He init)
#   max |input_scaled|   ≈ 5 * 2^b  (5σ of standardized input, scaled)
#
# Worst-case accumulator at any hidden layer:
#   max_accum ≈ fan_in × (2^b × 3.0) × (2^b × 5.0)
#             = fan_in × 15 × 2^(2b)
#
# Must fit in int64:  fan_in × 15 × 2^(2b) < 2^63
# ⇒  2b < 63 − log2(15 × fan_in)
# ⇒  b  < (63 − log2(15 × fan_in)) / 2

MAX_ABS_WEIGHT_FLOAT = 3.0   # conservative 3σ bound
MAX_ABS_INPUT_SIGMA = 5.0     # 5σ for normalized input


def overflow_analysis(hidden_size: int, scale_bits: int):
    """Return (overflow_safe: bool, b_max_safe: int).

    overflow_safe — whether the given (hidden_size, scale_bits) is safe.
    b_max_safe    — maximum scale_bits that is overflow-safe for this
                    hidden_size (the bottleneck is the hidden→hidden layer
                    where fan_in = hidden_size).
    """
    # The bottleneck fan_in is max(NUM_FEATURES=6, hidden_size).
    # For hidden→hidden layers, fan_in = hidden_size.
    # For input→hidden, fan_in = 6 (always ≤ hidden_size for our range).
    fan_in = hidden_size

    coeff = fan_in * MAX_ABS_WEIGHT_FLOAT * MAX_ABS_INPUT_SIGMA
    # max_accum ≈ coeff × 2^(2b)
    # Safe if coeff × 2^(2b) < 2^63
    # ⇒ 2b < 63 − log2(coeff)
    if coeff <= 0:
        b_max = 31  # degenerate case
    else:
        b_max = int((63 - math.log2(coeff)) / 2)

    overflow_safe = scale_bits <= b_max
    return overflow_safe, b_max


# ──────────────────────────────────────────────────────────────────────
# BPF compilation + instruction counting
# ──────────────────────────────────────────────────────────────────────

def count_bpf_instructions(project_dir: str, header_path: str) -> int | None:
    """Compile xdp_ml.c against the given header and count BPF instructions.

    Returns the instruction count, or None if compilation fails.
    """
    src = os.path.join(project_dir, "src", "xdp_ml.c")
    if not os.path.exists(src):
        return None

    inc_dir = os.path.dirname(header_path)
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as tmp:
        obj_path = tmp.name

    try:
        # Compile
        cmd_compile = [
            "clang", "-O2", "-g", "-target", "bpf",
            "-D__TARGET_ARCH_x86",
            f"-I{inc_dir}",
            "-I/usr/include",
            "-I/usr/include/x86_64-linux-gnu",
            "-Wno-unused-value",
            "-Wno-compare-distinct-pointer-types",
            "-c", src, "-o", obj_path,
        ]
        result = subprocess.run(cmd_compile, capture_output=True, text=True,
                                timeout=60)
        if result.returncode != 0:
            print(f"    ⚠ BPF compile failed: {result.stderr.strip()[:200]}")
            return None

        # Count instructions via llvm-objdump
        cmd_objdump = ["llvm-objdump", "-d", obj_path]
        result = subprocess.run(cmd_objdump, capture_output=True, text=True,
                                timeout=30)
        if result.returncode != 0:
            return None

        # Each instruction line starts with whitespace then an address.
        # Pattern: lines with a colon after a hex address, e.g.
        #   "      18:	b7 02 00 00 00 00 00 00	r2 = 0"
        count = 0
        for line in result.stdout.splitlines():
            stripped = line.lstrip()
            # Instruction lines look like: "     18:\tb7 02 00 ..."
            if stripped and ":" in stripped:
                # Check that what's before the first colon is a hex address
                prefix = stripped.split(":")[0].strip()
                if prefix and all(c in "0123456789abcdef" for c in prefix):
                    count += 1
        return count if count > 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    ⚠ BPF instruction count failed: {e}")
        return None
    finally:
        if os.path.exists(obj_path):
            os.unlink(obj_path)


# ──────────────────────────────────────────────────────────────────────
# Single experiment runner
# ──────────────────────────────────────────────────────────────────────

def run_experiment(hidden_size: int, num_layers: int, scale_bits: int,
                   project_dir: str, data_dir: str | None = None) -> dict:
    """Train one configuration and return the result dict."""
    train_script = os.path.join(project_dir, "scripts", "train_model.py")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                     mode="w") as jf:
        json_path = jf.name

    # Use a temporary directory for the header so we don't clobber the
    # project's include/model_params.h during exploration.
    tmp_inc_dir = tempfile.mkdtemp(prefix="ebpf_explore_")
    header_path = os.path.join(tmp_inc_dir, "model_params.h")

    cmd = [
        sys.executable, train_script,
        "--hidden-size", str(hidden_size),
        "--num-layers", str(num_layers),
        "--scale-bits", str(scale_bits),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--batch-size", str(BATCH_SIZE),
        "--quiet",
        "--json-output", json_path,
        "--output-header", header_path,
    ]
    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"    ✗ train_model.py failed (exit {result.returncode})")
            stderr_preview = result.stderr.strip()[:300]
            if stderr_preview:
                print(f"      {stderr_preview}")
            return {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "scale_bits": scale_bits,
                "status": "train_failed",
                "error": result.stderr.strip()[:500],
                "elapsed_s": round(elapsed, 2),
            }

        # Parse JSON result from train_model.py
        with open(json_path, "r") as f:
            train_result = json.load(f)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "scale_bits": scale_bits,
            "status": "timeout",
            "elapsed_s": round(elapsed, 2),
        }
    finally:
        if os.path.exists(json_path):
            os.unlink(json_path)

    # ── BPF instruction count (only for 2-hidden-layer models) ──────
    instruction_count = None
    if num_layers == 2 and os.path.exists(header_path):
        instruction_count = count_bpf_instructions(project_dir, header_path)

    # ── Overflow analysis ───────────────────────────────────────────
    overflow_safe, b_max_safe = overflow_analysis(hidden_size, scale_bits)

    # ── Assemble record ─────────────────────────────────────────────
    record = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "scale_bits": scale_bits,
        "float_accuracy": train_result.get("float_accuracy"),
        "int_accuracy": train_result.get("int_accuracy"),
        "accuracy_delta": train_result.get("accuracy_delta"),
        "macro_f1": train_result.get("macro_f1"),
        "model_memory_bytes": train_result.get("model_memory_bytes"),
        "instruction_count": instruction_count,
        "overflow_safe": overflow_safe,
        "b_max_safe": b_max_safe,
        "status": "ok",
        "elapsed_s": round(elapsed, 2),
    }

    # Clean up temp include dir
    try:
        if os.path.exists(header_path):
            os.unlink(header_path)
        os.rmdir(tmp_inc_dir)
    except OSError:
        pass

    return record


# ──────────────────────────────────────────────────────────────────────
# Pareto analysis
# ──────────────────────────────────────────────────────────────────────

def compute_pareto(results: list[dict]) -> list[dict]:
    """Find Pareto-optimal configs (maximize accuracy, minimize instructions).

    A configuration is Pareto-dominated if another config has equal-or-better
    accuracy AND equal-or-fewer instructions, with strict improvement in at
    least one.  Configs without instruction counts are excluded from the
    Pareto front (they cannot be compared on the resource axis).
    """
    # Filter to configs with valid instruction counts
    candidates = [r for r in results
                  if r.get("instruction_count") is not None
                  and r.get("int_accuracy") is not None]

    pareto = []
    for r in candidates:
        dominated = False
        r_acc = r["int_accuracy"]
        r_ins = r["instruction_count"]
        for other in candidates:
            o_acc = other["int_accuracy"]
            o_ins = other["instruction_count"]
            if (o_acc >= r_acc and o_ins <= r_ins and
                    (o_acc > r_acc or o_ins < r_ins)):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    # Sort by accuracy descending
    pareto.sort(key=lambda x: x["int_accuracy"], reverse=True)
    return pareto


# ──────────────────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    """Print a formatted summary table of all experiments."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        print("\nNo successful experiments to summarize.")
        return

    # Header
    print("\n" + "=" * 105)
    print(f"{'Config':^20s} │ {'Float':>7s} {'Int':>7s} {'ΔAcc':>6s} "
          f"{'F1':>6s} │ {'Memory':>8s} {'Instrs':>7s} │ "
          f"{'Safe':>4s} {'bMax':>4s} │ {'Time':>6s}")
    print("─" * 20 + "─┼─" + "─" * 30 + "─┼─" + "─" * 16 + "─┼─"
          + "─" * 10 + "─┼─" + "─" * 6)

    # Sort by (num_layers, hidden_size, scale_bits) for readability
    ok.sort(key=lambda r: (r["num_layers"], r["hidden_size"], r["scale_bits"]))

    for r in ok:
        tag = f"h={r['hidden_size']:<3d} l={r['num_layers']} b={r['scale_bits']:<2d}"
        fa = f"{r['float_accuracy']:.4f}" if r.get("float_accuracy") else "  N/A "
        ia = f"{r['int_accuracy']:.4f}" if r.get("int_accuracy") else "  N/A "
        da = f"{r['accuracy_delta']:.3f}" if r.get("accuracy_delta") is not None else " N/A "
        f1 = f"{r['macro_f1']:.3f}" if r.get("macro_f1") else " N/A "
        mem = f"{r['model_memory_bytes']:>7d}B" if r.get("model_memory_bytes") else "    N/A "
        ins = f"{r['instruction_count']:>7d}" if r.get("instruction_count") else "    N/A"
        safe = " ✓" if r.get("overflow_safe") else " ✗"
        bmax = f"{r['b_max_safe']:>4d}" if r.get("b_max_safe") is not None else " N/A"
        elapsed = fmt_time(r.get("elapsed_s", 0))
        print(f" {tag:<19s} │ {fa} {ia} {da} {f1} │ {mem} {ins} │ "
              f"{safe} {bmax} │ {elapsed:>6s}")

    print("=" * 105)


def print_pareto(pareto: list[dict]):
    """Print the Pareto-optimal configurations."""
    if not pareto:
        print("\n  No Pareto-optimal configurations found (need num_layers=2 "
              "experiments with instruction counts).")
        return

    print(f"\n{'─' * 60}")
    print(f"  Pareto Front  ({len(pareto)} configuration(s))")
    print(f"{'─' * 60}")
    for i, r in enumerate(pareto):
        tag = (f"h={r['hidden_size']}, l={r['num_layers']}, "
               f"b={r['scale_bits']}")
        print(f"  {i + 1}. [{tag}]  "
              f"acc={r['int_accuracy']:.4f}  "
              f"instrs={r['instruction_count']}  "
              f"mem={r['model_memory_bytes']}B"
              f"{'  ⚠ overflow!' if not r.get('overflow_safe', True) else ''}")


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="eBPF-ML design-space exploration: sweep over MLP "
                    "architectures and measure accuracy vs. resource cost.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to preprocessed dataset (passed through "
                             "to train_model.py).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the experiment plan without running.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = str(script_dir.parent)
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "experiment_log.jsonl")
    summary_path = os.path.join(results_dir, "exploration_summary.json")

    # ── Build experiment plan ───────────────────────────────────────
    all_configs = []
    for h in HIDDEN_SIZES:
        for l in NUM_LAYERS_OPTIONS:
            for b in SCALE_BITS_OPTIONS:
                all_configs.append((h, l, b))

    completed = load_completed(log_path)
    todo = [(h, l, b) for h, l, b in all_configs
            if config_key(h, l, b) not in completed]

    print("=" * 60)
    print("  eBPF-ML Design Space Exploration")
    print("=" * 60)
    print(f"  Search space       : {len(HIDDEN_SIZES)} hidden × "
          f"{len(NUM_LAYERS_OPTIONS)} layers × "
          f"{len(SCALE_BITS_OPTIONS)} scale_bits "
          f"= {len(all_configs)} configs")
    print(f"  Already completed  : {len(completed)}")
    print(f"  Remaining          : {len(todo)}")
    if args.data_dir:
        print(f"  Data source        : {args.data_dir}")
    else:
        print(f"  Data source        : synthetic")
    print(f"  Fixed: epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Experiments that would run:\n")
        for i, (h, l, b) in enumerate(todo):
            safe, bmax = overflow_analysis(h, b)
            bpf = "compile+count" if l == 2 else "skip (layers≠2)"
            tag = "safe" if safe else f"⚠ overflow (bmax={bmax})"
            print(f"    {i + 1:3d}. hidden={h:3d}  layers={l}  "
                  f"scale_bits={b:2d}  BPF: {bpf:<20s}  {tag}")
        print(f"\n  Total: {len(todo)} experiment(s)")
        return

    if not todo:
        print("  All configurations already completed.")
    else:
        print(f"  Starting {len(todo)} experiment(s)...\n")

    # ── Run experiments ─────────────────────────────────────────────
    times = []
    for idx, (h, l, b) in enumerate(todo):
        label = f"[{idx + 1}/{len(todo)}]"
        key = config_key(h, l, b)

        # ETA calculation
        if times:
            avg = sum(times) / len(times)
            remaining = (len(todo) - idx) * avg
            eta_str = f"  ETA: {fmt_time(remaining)}"
        else:
            eta_str = ""

        print(f"  {label} h={h:<3d} l={l} b={b:<2d}  "
              f"{'(+BPF)' if l == 2 else '       '}{eta_str}")

        record = run_experiment(h, l, b, project_dir,
                                data_dir=args.data_dir)

        # Append to JSONL log
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        elapsed = record.get("elapsed_s", 0)
        times.append(elapsed)

        if record.get("status") == "ok":
            acc_str = (f"int_acc={record['int_accuracy']:.4f}"
                       if record.get("int_accuracy") is not None else "")
            ins_str = (f"instrs={record['instruction_count']}"
                       if record.get("instruction_count") is not None else "")
            safe_str = "safe" if record.get("overflow_safe") else "⚠ overflow"
            print(f"         ✓ {acc_str}  {ins_str}  {safe_str}  "
                  f"({fmt_time(elapsed)})")
        else:
            print(f"         ✗ {record.get('status', 'unknown')}  "
                  f"({fmt_time(elapsed)})")

    # ── Load all results and summarize ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Exploration Summary")
    print("=" * 60)

    all_results = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    ok_results = [r for r in all_results if r.get("status") == "ok"]
    failed = [r for r in all_results if r.get("status") != "ok"]

    print(f"\n  Total experiments : {len(all_results)}")
    print(f"  Successful        : {len(ok_results)}")
    print(f"  Failed/timeout    : {len(failed)}")

    if ok_results:
        # Best accuracy
        best_acc = max(ok_results, key=lambda r: r.get("int_accuracy", 0))
        print(f"\n  Best int accuracy : {best_acc['int_accuracy']:.4f}  "
              f"(h={best_acc['hidden_size']}, l={best_acc['num_layers']}, "
              f"b={best_acc['scale_bits']})")

        # Fewest instructions (among those with counts)
        with_instrs = [r for r in ok_results
                       if r.get("instruction_count") is not None]
        if with_instrs:
            fewest = min(with_instrs,
                         key=lambda r: r["instruction_count"])
            print(f"  Fewest BPF instrs : {fewest['instruction_count']}  "
                  f"(h={fewest['hidden_size']}, l={fewest['num_layers']}, "
                  f"b={fewest['scale_bits']})")

        # Smallest memory
        smallest = min(ok_results,
                       key=lambda r: r.get("model_memory_bytes", float("inf")))
        if smallest.get("model_memory_bytes"):
            print(f"  Smallest model    : {smallest['model_memory_bytes']}B  "
                  f"(h={smallest['hidden_size']}, l={smallest['num_layers']}, "
                  f"b={smallest['scale_bits']})")

        # Overflow safety summary
        safe_count = sum(1 for r in ok_results if r.get("overflow_safe"))
        print(f"\n  Overflow-safe     : {safe_count}/{len(ok_results)}")

    # ── Summary table ───────────────────────────────────────────────
    print_summary_table(ok_results)

    # ── Pareto front ────────────────────────────────────────────────
    pareto = compute_pareto(ok_results)
    print_pareto(pareto)

    # ── Save summary JSON ───────────────────────────────────────────
    summary = {
        "total_configs": len(all_configs),
        "completed": len(ok_results),
        "failed": len(failed),
        "search_space": {
            "hidden_sizes": HIDDEN_SIZES,
            "num_layers_options": NUM_LAYERS_OPTIONS,
            "scale_bits_options": SCALE_BITS_OPTIONS,
        },
        "training_config": {
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        },
        "best_accuracy": (
            {k: best_acc[k] for k in
             ["hidden_size", "num_layers", "scale_bits", "int_accuracy",
              "macro_f1", "model_memory_bytes", "instruction_count"]}
            if ok_results else None
        ),
        "pareto_front": [
            {k: r[k] for k in
             ["hidden_size", "num_layers", "scale_bits", "int_accuracy",
              "instruction_count", "model_memory_bytes", "overflow_safe"]}
            for r in pareto
        ],
        "all_results": ok_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {summary_path}")
    print(f"  Full log at        {log_path}")
    print()


if __name__ == "__main__":
    main()
