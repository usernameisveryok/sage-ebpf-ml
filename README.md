# eBPF/XDP In-Kernel ML Packet Classifier

Real-time network traffic classification using a quantized neural network
running entirely inside the Linux kernel via eBPF/XDP.

**Thesis**: 面向实时网络入侵检测的 eBPF 内核态量化神经网络推理方法研究

## Overview

This project implements a complete pipeline for in-kernel machine learning
inference on network packets, with automated design space exploration
inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

1. **Training** (Python) — Train a small MLP on packet features
2. **Quantization** — Convert float32 weights to int32 using the "Enlargement Method"
3. **eBPF Inference** (C/XDP) — Run the quantized model in the kernel at wire speed
4. **Design Space Exploration** — Automated sweep of 48 architecture configurations
5. **Overflow Safety Analysis** — Formal analysis of int64 accumulator bounds
6. **Monitoring** (C) — Userspace loader with live statistics + latency tracking

### Architecture

```
 ┌─────────────────────────────────────────────────────┐
 │                  Linux Kernel (XDP)                  │
 │                                                     │
 │  Packet → Parse → Flow Track → Feature Extract      │
 │                        │                            │
 │                   [tail call]                       │
 │                        ↓                            │
 │              Layer 0: [6→32] + ReLU                 │
 │                   [tail call]                       │
 │                        ↓                            │
 │              Layer 1: [32→32] + ReLU                │
 │                   [tail call]                       │
 │                        ↓                            │
 │              Layer 2: [32→4] → argmax               │
 │                        │                            │
 │              BENIGN → PASS  |  ATTACK → DROP        │
 └─────────────────────────────────────────────────────┘
```

### Model

- **Type**: Multi-Layer Perceptron (MLP)
- **Architecture**: [6, 32, 32, 4] (configurable)
- **Quantization**: Int32, enlargement factor s = 2^16
- **Accuracy**: ~99.6% (synthetic), pending CIC-IDS-2017
- **Memory**: ~5.6 KB model parameters
- **BPF Instructions**: 806 total across 4 programs
- **Inference time**: ~3-5 μs per flow (estimated)

### Classes

| Class | Action | Description |
|-------|--------|-------------|
| BENIGN | XDP_PASS | Normal traffic |
| DDOS | XDP_DROP | DDoS attack packets |
| PORTSCAN | XDP_DROP | Port scanning |
| BRUTEFORCE | XDP_DROP | Brute-force attacks |

## Three Innovation Points (创新点)

### 1. Overflow Safety Formal Analysis (理论)
Formal proof that int64 accumulators don't overflow during quantized inference.
- Worst-case bound: b_max_safe = 14 for [6,32,32,4]
- Empirical reality: only 3.2×10⁻⁶% of int64_max used (12.4 bits headroom)
- LaTeX table for thesis showing safety across architectures

### 2. Automated Design Space Exploration (系统)
Autoresearch-inspired experiment loop exploring 48 configurations:
- `hidden_size` ∈ {8, 16, 32, 64}
- `num_layers` ∈ {1, 2, 3}
- `scale_bits` ∈ {8, 12, 16, 20}

Key findings: b=16 is the critical threshold; h=16 is the sweet spot (Pareto optimal).

### 3. CIC-IDS-2017 Multi-class Evaluation (评测)
End-to-end evaluation with real dataset and XDP deployment benchmarks.

## Requirements

- Linux kernel ≥ 5.10
- clang/LLVM ≥ 14
- libbpf-dev, libelf-dev
- Python 3 with numpy (+ matplotlib for plots)

## Quick Start

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install clang llvm libbpf-dev libelf-dev python3-numpy

# Build everything (train model → compile eBPF → compile loader)
make all

# Run tests
make test

# Run design space exploration (48 configs, ~50 min)
make explore

# Run overflow analysis
make overflow

# Generate thesis figures
python3 scripts/plot_results.py

# Load on an interface (requires root)
cd build && sudo ./loader eth0 -S

# Run benchmark (requires root)
make benchmark IFACE=eth0 DURATION=30
```

## Project Structure

```
├── Makefile                      # Build system (all/bpf/loader/test/explore/...)
├── README.md                     # This file
├── RESEARCH_REPORT.md            # Literature survey
├── include/
│   └── model_params.h            # Auto-generated quantized model weights
├── scripts/
│   ├── train_model.py            # Model training & quantization (argparse CLI)
│   ├── preprocess_cicids.py      # CIC-IDS-2017 data preprocessing
│   ├── explore.py                # Automated design space exploration
│   ├── overflow_analysis.py      # Formal overflow safety analysis
│   ├── plot_results.py           # Thesis figure generation
│   ├── benchmark.sh              # Performance benchmarking script
│   └── program.md                # Experiment protocol (autoresearch style)
├── src/
│   ├── xdp_ml.c                  # eBPF/XDP kernel program (4 tail-call stages)
│   └── loader.c                  # Userspace loader/monitor with latency tracking
├── tests/
│   └── test_inference.py         # Integer inference verification (6/6 tests)
├── results/                      # Experiment outputs
│   ├── experiment_log.jsonl      # 48-config exploration results
│   ├── exploration_summary.json  # Summary with Pareto front
│   └── fig_*.png                 # Thesis figures
└── build/
    ├── xdp_ml.o                  # Compiled eBPF object
    └── loader                    # Userspace loader binary
```

## Design Space Exploration Results

| Hidden | Layers | Scale Bits | Int Accuracy | Instructions | Memory |
|--------|--------|------------|-------------|-------------|--------|
| 8 | 2 | 16 | 99.61% | 754 | 656B |
| **16** | **2** | **16** | **99.62%** | **754** | **1.8KB** |
| 32 | 2 | 16 | 99.62% | 806 | 5.6KB |
| 64 | 2 | 16 | 99.62% | 806 | 19.5KB |

**Pareto optimal**: h=16, layers=2, b=16 (same accuracy, minimal resources)

## Technical Details

### Enlargement Method Quantization

Scale factor: `s = 2^16 = 65536`
- Weights: `W_q = round(s × W)` → int32
- Inference: `y[i] = (Σ W_q[i][j] × x[j]) >> 16 + B_q[i]`

### eBPF Constraints Solved

| Constraint | Solution |
|-----------|----------|
| No floating point | Int32 fixed-point (enlargement method) |
| 512B stack limit | Tail calls + per-CPU scratch map |
| 1M instruction limit | 806 total instructions |
| No signed division | `safe_div()` using absolute values |
| No dynamic allocation | BPF maps for all dynamic storage |

### BPF Instruction Breakdown

| Program | Section | Instructions |
|---------|---------|-------------|
| xdp_entry | Packet parse + flow tracking | 309 |
| xdp_layer_0 | Normalize + Layer 0 + ReLU | 109 |
| xdp_layer_1 | Layer 1 + ReLU | 152 |
| xdp_layer_2 | Layer 2 + argmax + classify | 236 |
| **Total** | | **806** |

## References

1. Zhang et al., "Real-Time Intrusion Detection with NN in Kernel using eBPF", DSN 2024
2. Bachl et al., "A flow-based IDS using Machine Learning in eBPF", 2021
3. Osaki et al., "Dynamic Fixed-point Values in eBPF", AINTEC 2024
4. Hara et al., "On Practicality of Kernel Packet Processing", NoF 2023
5. Chen et al., "DDoS Detection via eBPF/XDP and Knowledge Distillation", ICCPR 2025
6. Karpathy, "autoresearch", 2026 (experiment methodology inspiration)

## License

GPL-2.0
