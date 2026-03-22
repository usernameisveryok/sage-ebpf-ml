#!/bin/bash
# benchmark.sh — Performance benchmarking for eBPF/XDP ML inference
#
# Usage: sudo ./scripts/benchmark.sh [interface] [duration_secs]
# Example: sudo ./scripts/benchmark.sh eth0 30

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────
INTERFACE="${1:-eth0}"
DURATION="${2:-30}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BPF_OBJ="$PROJECT_ROOT/build/xdp_ml.o"
LOADER="$PROJECT_ROOT/build/loader"
MODEL_HDR="$PROJECT_ROOT/include/model_params.h"
RESULTS_DIR="$PROJECT_ROOT/results"

# ── Colors ───────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Check prerequisites ──────────────────────────────
info "Checking prerequisites..."

# Root check
[[ $EUID -eq 0 ]] || die "This script must be run as root (sudo)."

# Interface check
if ! ip link show "$INTERFACE" &>/dev/null; then
    die "Interface '$INTERFACE' does not exist. Available interfaces:"
    ip -br link show
fi
ok "Interface $INTERFACE exists"

# bpftool (optional, for detailed stats)
HAS_BPFTOOL=false
if command -v bpftool &>/dev/null; then
    HAS_BPFTOOL=true
    ok "bpftool available"
else
    warn "bpftool not found — will use loader output for stats"
fi

# ── 2. Build if needed ──────────────────────────────────
if [[ ! -f "$BPF_OBJ" ]] || [[ ! -f "$LOADER" ]]; then
    info "Build artifacts not found, building..."
    cd "$PROJECT_ROOT"
    make bpf loader || die "Build failed"
    cd - >/dev/null
fi

[[ -f "$BPF_OBJ" ]]  || die "BPF object not found: $BPF_OBJ"
[[ -f "$LOADER" ]]    || die "Loader not found: $LOADER"
ok "Build artifacts present"

# ── 3. Pre-benchmark stats ──────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  eBPF/XDP ML Inference — Performance Benchmark"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Interface : $INTERFACE"
echo "  Duration  : ${DURATION}s"
echo "  BPF obj   : $BPF_OBJ"
echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

info "BPF program analysis:"

# Instruction count
if command -v llvm-objdump &>/dev/null; then
    TOTAL_INSN=$(llvm-objdump -d "$BPF_OBJ" | grep -c '^\s' || true)
    SECTION_COUNT=$(llvm-objdump -d "$BPF_OBJ" | grep -c 'Disassembly of section' || true)
    echo "  BPF instructions : $TOTAL_INSN"
    echo "  Program sections : $SECTION_COUNT"

    # Per-section breakdown
    while IFS= read -r section; do
        sec_name=$(echo "$section" | sed 's/.*section //;s/://')
        sec_count=$(llvm-objdump -d --section="$sec_name" "$BPF_OBJ" 2>/dev/null | grep -c '^\s' || true)
        echo "    $sec_name: $sec_count instructions"
    done < <(llvm-objdump -d "$BPF_OBJ" | grep 'Disassembly of section')
else
    warn "llvm-objdump not found — skipping instruction count"
fi

# Model memory estimate
if [[ -f "$MODEL_HDR" ]]; then
    MODEL_SIZE=$(stat -c %s "$MODEL_HDR" 2>/dev/null || stat -f %z "$MODEL_HDR" 2>/dev/null || echo "?")
    echo "  Model header size: ${MODEL_SIZE} bytes (approx)"
fi

BPF_OBJ_SIZE=$(stat -c %s "$BPF_OBJ" 2>/dev/null || stat -f %z "$BPF_OBJ" 2>/dev/null || echo "?")
echo "  BPF object size  : ${BPF_OBJ_SIZE} bytes"
echo ""

# ── 4. Capture pre-benchmark CPU stats ──────────────────
CPU_BEFORE=$(grep 'cpu ' /proc/stat 2>/dev/null || true)
SOFTIRQ_BEFORE=$(grep 'NET_RX\|NET_TX' /proc/softirqs 2>/dev/null || true)

# ── 5. Load XDP program and run benchmark ───────────────
info "Loading XDP program on $INTERFACE (SKB mode)..."

mkdir -p "$RESULTS_DIR"
BENCH_LOG="$RESULTS_DIR/benchmark_$(date '+%Y%m%d_%H%M%S').log"

# Run loader in background, capture output
$LOADER "$INTERFACE" -S > "$BENCH_LOG" 2>&1 &
LOADER_PID=$!

# Verify loader started
sleep 1
if ! kill -0 "$LOADER_PID" 2>/dev/null; then
    die "Loader failed to start. Check $BENCH_LOG for details."
fi
ok "Loader running (PID $LOADER_PID)"

info "Benchmarking for ${DURATION}s..."

# Progress display
ELAPSED=0
while [[ $ELAPSED -lt $DURATION ]]; do
    REMAINING=$((DURATION - ELAPSED))
    printf "\r  Time remaining: %3ds " "$REMAINING"

    # Show live packet count if bpftool available
    if $HAS_BPFTOOL; then
        PKT_COUNT=$(bpftool map dump name stats_map 2>/dev/null | grep -o '"value":.*' | head -1 || true)
        if [[ -n "$PKT_COUNT" ]]; then
            printf "| %s" "$PKT_COUNT"
        fi
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo ""

# ── 6. Collect results ──────────────────────────────────
info "Collecting results..."

# Capture post-benchmark CPU stats
CPU_AFTER=$(grep 'cpu ' /proc/stat 2>/dev/null || true)
SOFTIRQ_AFTER=$(grep 'NET_RX\|NET_TX' /proc/softirqs 2>/dev/null || true)

# Dump maps if bpftool available
LATENCY_DUMP=""
STATS_DUMP=""
if $HAS_BPFTOOL; then
    LATENCY_DUMP=$(bpftool map dump name latency_map 2>/dev/null || true)
    STATS_DUMP=$(bpftool map dump name stats_map 2>/dev/null || true)
fi

# ── 7. Unload XDP program ───────────────────────────────
info "Stopping loader and unloading XDP program..."

# Send SIGINT to let the loader clean up gracefully
kill -INT "$LOADER_PID" 2>/dev/null || true
sleep 2

# Make sure it's dead
if kill -0 "$LOADER_PID" 2>/dev/null; then
    kill -KILL "$LOADER_PID" 2>/dev/null || true
    sleep 1
fi

# Safety: force-detach XDP if still attached
if $HAS_BPFTOOL; then
    bpftool net detach xdp dev "$INTERFACE" 2>/dev/null || true
else
    ip link set dev "$INTERFACE" xdp off 2>/dev/null || true
fi

ok "XDP program unloaded"

# ── 8. Generate report ──────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  BENCHMARK RESULTS"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Test parameters:"
echo "    Interface     : $INTERFACE"
echo "    Duration      : ${DURATION}s"
echo "    Mode          : SKB (generic XDP)"
echo "    Date          : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Parse loader log for stats
echo "  Loader output summary:"
if [[ -f "$BENCH_LOG" ]]; then
    # Show the last few stats lines from the loader
    echo "  ──────────────────────────────────────"
    grep -E 'packets|pps|latency|drop|pass|class' "$BENCH_LOG" | tail -20 | while IFS= read -r line; do
        echo "    $line"
    done
    echo "  ──────────────────────────────────────"
fi

# bpftool-based detailed stats
if $HAS_BPFTOOL && [[ -n "$STATS_DUMP" ]]; then
    echo ""
    echo "  BPF map stats (bpftool):"
    echo "    $STATS_DUMP" | head -20
fi

if $HAS_BPFTOOL && [[ -n "$LATENCY_DUMP" ]]; then
    echo ""
    echo "  Latency map:"
    echo "    $LATENCY_DUMP" | head -20
fi

# CPU usage delta
if [[ -n "$CPU_BEFORE" ]] && [[ -n "$CPU_AFTER" ]]; then
    echo ""
    echo "  CPU stats delta (/proc/stat 'cpu' line):"
    echo "    Before: $CPU_BEFORE"
    echo "    After : $CPU_AFTER"
fi

if [[ -n "$SOFTIRQ_BEFORE" ]] && [[ -n "$SOFTIRQ_AFTER" ]]; then
    echo ""
    echo "  SoftIRQ (NET_RX/NET_TX) delta:"
    echo "    Before:"
    echo "$SOFTIRQ_BEFORE" | while IFS= read -r line; do echo "      $line"; done
    echo "    After:"
    echo "$SOFTIRQ_AFTER" | while IFS= read -r line; do echo "      $line"; done
fi

echo ""
echo "  BPF program info:"
echo "    Instructions  : ${TOTAL_INSN:-N/A}"
echo "    Sections      : ${SECTION_COUNT:-N/A}"
echo "    Object size   : ${BPF_OBJ_SIZE:-N/A} bytes"
echo "    Model header  : ${MODEL_SIZE:-N/A} bytes"
echo ""
echo "  Full log saved to: $BENCH_LOG"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Benchmark complete."
echo "════════════════════════════════════════════════════════════"
