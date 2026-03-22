# eBPF-ML Design Space Exploration Protocol

## Objective

Systematically explore the design space of quantized MLP architectures for
eBPF/XDP in-kernel packet classification, identifying **Pareto-optimal
configurations** that balance classification accuracy against eBPF resource
constraints (instruction count, stack usage, and integer overflow safety).

The core research question is: *given the hard constraints of the eBPF
verifier — bounded stack, bounded instructions, no floating point — what
model architecture yields the best accuracy-vs-resource tradeoff for
real-time traffic classification?*

---

## Three Primitives

Inspired by Karpathy's *autoresearch* methodology, the exploration is
structured around three composable primitives that make the design-space
search systematic, reproducible, and incrementally refinable.

### 1. Editable Asset — Model Architecture Parameters

The editable asset is the MLP architecture specification, fully described by
three integers:

| Parameter       | Symbol | Range            | Description                                    |
|-----------------|--------|------------------|------------------------------------------------|
| `hidden_size`   | *h*    | {8, 16, 32, 64}  | Neurons per hidden layer                       |
| `num_layers`    | *L*    | {1, 2, 3}        | Number of hidden layers                        |
| `scale_bits`    | *b*    | {8, 12, 16, 20}  | Quantization bit-width (scale factor = 2^*b*)  |

The MLP topology is always `[6, h, h, …, 4]` — six input features (packet
length, header length, destination port, forward inter-arrival time, total
forward packets, max forward packet length) and four output classes (BENIGN,
DDOS, PORTSCAN, BRUTEFORCE).  All other training hyperparameters are held
constant to ensure a fair comparison:

| Hyperparameter  | Value  | Rationale                                        |
|-----------------|--------|--------------------------------------------------|
| Epochs          | 80     | Sufficient for convergence on this dataset        |
| Learning rate   | 0.05   | Tuned for the default [6,32,32,4] baseline       |
| Batch size      | 256    | Standard mini-batch SGD                          |
| LR decay        | ×0.5 every 30 epochs | Simple schedule, no per-config tuning |
| Seed            | 42     | Reproducibility                                  |

This creates a **48-point search grid** (4 × 3 × 4) — small enough for
exhaustive evaluation but large enough to reveal meaningful tradeoff curves.

### 2. Scalar Metrics — What We Measure

Each configuration produces a multi-dimensional quality vector:

| Metric                 | Type    | Source            | Purpose                            |
|------------------------|---------|-------------------|------------------------------------|
| `float_accuracy`       | float   | Training          | Baseline accuracy before quantization |
| `int_accuracy`         | float   | Quantized verify  | Accuracy after integer-only inference |
| `accuracy_delta`       | float   | Computed          | Quantization degradation           |
| `macro_f1`             | float   | Training          | Class-balanced quality metric      |
| `model_memory_bytes`   | int     | Weight arrays     | Total size of quantized parameters |
| `instruction_count`    | int     | BPF objdump       | Total BPF instructions in compiled program |
| `overflow_safe`        | bool    | Analytical        | Whether int64 accumulators are safe |
| `b_max_safe`           | int     | Analytical        | Maximum safe scale_bits for this hidden_size |

**Why these metrics?**

- **`int_accuracy`** is the primary quality metric — it reflects what the
  eBPF program will actually compute in the kernel.
- **`instruction_count`** is the primary cost metric — the eBPF verifier
  enforces a hard limit (1M instructions per program, but tail-call chains
  have per-program limits), and more instructions mean higher per-packet
  latency.
- **`model_memory_bytes`** captures the storage cost of embedding weights
  in the BPF object.
- **`overflow_safe`** ensures correctness — an unsafe configuration will
  produce silently wrong results due to int64 accumulator overflow.
- **`accuracy_delta`** quantifies the cost of quantization itself, separated
  from the architecture's inherent capacity.

### 3. Time-boxed Cycle — Fixed Training Budget

Every configuration is trained for exactly 80 epochs with identical
hyperparameters and the same random seed, ensuring:

1. **Fair comparison** — differences in accuracy are due to architecture,
   not training effort.
2. **Reproducibility** — any configuration can be re-run and will produce
   identical results.
3. **Bounded wall-clock time** — each experiment takes 5–30 seconds
   (depending on hidden_size), making the full 48-config sweep completable
   in under 30 minutes.

The cycle for each configuration is:

```
Train (80 epochs, fixed seed)
  → Quantize (scale = 2^b)
  → Verify integer accuracy
  → Export C header
  → [if L=2] Compile BPF → Count instructions
  → Compute overflow safety bound
  → Append to experiment log
```

---

## Search Space

The full search grid:

```
hidden_size  ∈ {8, 16, 32, 64}     — 4 values
num_layers   ∈ {1, 2, 3}            — 3 values
scale_bits   ∈ {8, 12, 16, 20}      — 4 values
                                      ─────────
                              Total:  48 configurations
```

### Why these ranges?

- **hidden_size**: 8 is the minimum for non-trivial representations; 64
  approaches the point where BPF instruction counts become problematic
  (O(h²) inner products).  Powers of two align with common SIMD widths.

- **num_layers**: 1 hidden layer (linear + one nonlinearity) tests whether
  the problem is linearly separable in feature space.  2 hidden layers is
  the standard "universal approximator" baseline.  3 hidden layers tests
  whether added depth helps despite eBPF's compilation constraints.

- **scale_bits**: 8 bits (s=256) gives coarse quantization — fast and small
  but may destroy accuracy.  20 bits (s=1,048,576) gives fine quantization
  but risks int64 overflow for large hidden sizes.  The interplay between
  scale_bits and hidden_size is a key finding of this exploration.

---

## Evaluation Protocol

### Step 1: Training

```bash
python3 scripts/train_model.py \
    --hidden-size $H --num-layers $L --scale-bits $B \
    --epochs 80 --lr 0.05 --batch-size 256 \
    --quiet --json-output /tmp/result.json \
    --output-header /tmp/model_params.h
```

The training script:
1. Generates (or loads) the dataset
2. Trains a numpy-only MLP with mini-batch SGD
3. Quantizes all weights and biases to int32 (scale = 2^*b*)
4. Verifies integer-only inference accuracy
5. Exports a C header with quantized parameters

### Step 2: BPF Compilation (num_layers = 2 only)

```bash
clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
    -I/tmp/ -I/usr/include -I/usr/include/x86_64-linux-gnu \
    -c src/xdp_ml.c -o /tmp/xdp_ml.o
```

The XDP C program uses `#define HIDDEN_SIZE`, `NUM_FEATURES`, etc. from the
generated header, so changing hidden_size automatically changes the compiled
loop bounds.  However, the program hardcodes three weight layers
(`weight_layer_0`, `weight_layer_1`, `weight_layer_2`), so only `num_layers
= 2` (which produces exactly 3 weight layers: input→hidden, hidden→hidden,
hidden→output) is compatible with BPF compilation.

### Step 3: Instruction Counting

```bash
llvm-objdump -d /tmp/xdp_ml.o | <count instruction lines>
```

Each disassembled instruction line starts with a hex address.  The total
count across all BPF program sections gives the instruction budget consumed.

### Step 4: Overflow Safety Analysis

For each configuration, we analytically compute whether int64 accumulators
are safe:

```
max_accum ≈ fan_in × max|W_q| × max|x_q|
          ≈ h × (3.0 × 2^b) × (5.0 × 2^b)
          = 15 × h × 2^(2b)
```

This must be less than 2^63 (int64 range).  Therefore:

```
b_max = floor((63 − log₂(15 × h)) / 2)
```

| hidden_size | b_max_safe | Scale bits that overflow |
|-------------|------------|-------------------------|
| 8           | 28         | None in our range        |
| 16          | 27         | None in our range        |
| 32          | 27         | None in our range        |
| 64          | 26         | None in our range        |

For the ranges we test ({8..20}), all configurations are analytically safe.
This validates the "enlargement method" for our scale range, but the
analysis framework would flag dangerous configurations if the search space
were extended (e.g., scale_bits=30 with hidden_size=64 would overflow).

### Step 5: Record Keeping

Each experiment appends one JSON line to `results/experiment_log.jsonl`:

```json
{
  "hidden_size": 32,
  "num_layers": 2,
  "scale_bits": 16,
  "float_accuracy": 0.9876,
  "int_accuracy": 0.9854,
  "accuracy_delta": 0.0022,
  "macro_f1": 0.9851,
  "model_memory_bytes": 9344,
  "instruction_count": 1847,
  "overflow_safe": true,
  "b_max_safe": 27,
  "status": "ok",
  "elapsed_s": 12.34
}
```

The append-only JSONL format supports incremental runs — if the sweep is
interrupted, it resumes from where it left off.

---

## Constraints (eBPF Verifier)

The eBPF verifier imposes hard constraints that make this design space
fundamentally different from standard ML model selection:

1. **No floating point** — All inference must use integer arithmetic.  The
   quantization scale factor 2^*b* determines the precision/overflow tradeoff.

2. **Bounded stack** — Each BPF program is limited to 512 bytes of stack.
   The XDP program uses tail-call chaining (one program per layer) to stay
   within this limit.  Per-CPU map scratch space holds the hidden vector
   between calls.

3. **Bounded instructions** — The verifier enforces a complexity limit per
   program section.  The inner-product loops have O(h²) instructions for
   hidden→hidden layers, making hidden_size the dominant cost factor.

4. **Bounded program count** — Tail-call chains are limited to 33 levels.
   Our 3-layer chain (entry + 3 NN stages) is well within this, but
   deeper networks would consume more tail-call slots.

5. **No dynamic allocation** — Array sizes must be compile-time constants.
   The weight arrays are `static const`, embedded directly in the BPF object.

6. **Deterministic termination** — All loops must have provably bounded
   iteration counts.  The `for (i = 0; i < HIDDEN_SIZE; i++)` loops satisfy
   this because `HIDDEN_SIZE` is a compile-time constant.

---

## Success Criteria

The exploration is successful if it:

1. **Identifies a Pareto front** — at least 2–3 configurations that are
   mutually non-dominated on the (accuracy, instruction_count) plane.

2. **Quantifies the quantization/overflow tradeoff** — shows how
   scale_bits affects accuracy and where the safety boundary lies.

3. **Reveals diminishing returns** — demonstrates that beyond some
   hidden_size, accuracy saturates while instruction count grows
   quadratically.

4. **Validates the baseline** — confirms that the default [6, 32, 32, 4]
   with scale_bits=16 is a reasonable choice (or identifies a better one).

5. **Produces reproducible data** — all results are logged in machine-readable
   format and can be exactly reproduced by re-running the sweep.

---

## Expected Outcomes (Hypotheses)

Based on the structure of the problem, we expect:

### H1: Accuracy saturates at moderate hidden sizes
The 4-class packet classification problem has limited inherent complexity
(the synthetic classes are generated from well-separated distributions).
We predict that `hidden_size=16` or `hidden_size=32` will achieve near-peak
accuracy, with `hidden_size=64` providing negligible improvement.

### H2: Scale bits have a "sweet spot"
Too few bits (b=8) will cause quantization noise to degrade accuracy.  Too
many bits (b=20) are wasteful and, for larger models, risk overflow.  We
expect b=12 or b=16 to be optimal for most hidden sizes.

### H3: Instruction count scales quadratically with hidden size
The hidden→hidden layer has an O(h²) inner product.  Doubling hidden_size
from 32 to 64 should roughly quadruple the instruction count for that layer,
making the cost curve steep.

### H4: The Pareto front will be small
Given the above, we expect 2–4 Pareto-optimal configurations, likely:
- A "tiny" model (h=8 or h=16, b=12) — low cost, acceptable accuracy
- The baseline (h=32, b=16) — good accuracy, moderate cost
- Possibly no configuration with h=64 on the front (dominated by h=32)

### H5: Single-layer models will underperform
With `num_layers=1`, the model is essentially linear (one hidden layer +
ReLU + output).  While the classes are reasonably separable, we expect the
2-layer models to achieve 2–5% higher accuracy due to the additional
nonlinear decision boundary.

---

## Reproducing the Exploration

```bash
# Full sweep with synthetic data
python3 scripts/explore.py

# Full sweep with real CIC-IDS2017 data
python3 scripts/explore.py --data-dir data

# Preview the plan without running
python3 scripts/explore.py --dry-run

# Results are incrementally saved — safe to interrupt and resume
```

All output is written to `results/experiment_log.jsonl` (per-experiment
records) and `results/exploration_summary.json` (aggregated analysis with
Pareto front).
