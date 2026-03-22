# AGENTS.md — eBPF In-Kernel ML Inference Project

## Project Goal
Tsinghua CS academic master's thesis: "面向实时网络入侵检测的 eBPF 内核态量化神经网络推理方法研究"

## Build & Test
```bash
make all        # Train model → compile eBPF → compile loader
make test       # Run 6 inference tests (must all pass)
make explore    # Run 48-config design space exploration (~50 min)
make overflow   # Run overflow safety analysis
make benchmark IFACE=eth0 DURATION=30  # Performance test (needs root)
```

## Environment
- Ubuntu 22.04.5, Kernel 5.10.134
- Clang 14 (Ubuntu), libbpf 0.5.0
- Python 3.12 with numpy
- Uses `bpf_set_link_xdp_fd()` (older libbpf API)
- Extra include: `-I/usr/include/x86_64-linux-gnu` for asm/types.h

## Key Design Decisions
1. **NN-int32 over DT/int8**: Fixed structure, minimal memory (5.6KB), near-zero quantization loss
2. **Tail-call chain**: 4-stage pipeline solves 512B stack + 1M instruction limits
3. **Per-CPU scratch map**: Lock-free intermediate buffers between tail calls
4. **Scale factor s=2^16**: Empirically safe (12.4 bits headroom), formal analysis in overflow_analysis.py

## Coding Style
- eBPF C: Linux kernel style, `__always_inline`, bounded loops, explicit verifier hints
- Python: numpy-only (no PyTorch/sklearn), argparse CLI, backward-compatible imports
- All new scripts must preserve existing test compatibility (6/6 tests)

## Three Innovation Points
1. **理论**: Overflow safety formal analysis (scripts/overflow_analysis.py)
2. **系统**: Autoresearch-style design space exploration (scripts/explore.py, 48 configs)
3. **评测**: CIC-IDS-2017 + real XDP deployment (pending data download)

## File Roles
- `scripts/train_model.py` — Core: training + quantization. Imported by tests. DO NOT break backward compat.
- `src/xdp_ml.c` — eBPF program. Assumes HIDDEN_SIZE/NUM_FEATURES from header. Has latency instrumentation.
- `src/loader.c` — Loader. Reads stats_map + latency_map.
- `results/` — Generated experiment data. DO NOT edit manually.
