# eBPF/XDP ML Packet Classifier — Makefile
#
# Targets:
#   make model    — Train the model and export model_params.h
#   make bpf      — Compile the eBPF XDP program
#   make loader   — Compile the userspace loader
#   make all      — Build everything (model → bpf → loader)
#   make clean    — Remove build artifacts
#
# Usage after build:
#   sudo ./build/loader <interface> -S   (SKB mode)
#   sudo ./build/loader <interface> -u   (unload)

CLANG    ?= clang
LLC      ?= llc
CC       ?= gcc
PYTHON   ?= python3
STRIP    ?= llvm-strip

# Directories
SRC_DIR   = src
INC_DIR   = include
SCRIPT_DIR = scripts
BUILD_DIR  = build

# eBPF compilation flags
BPF_CFLAGS = -O2 -g -target bpf \
             -D__TARGET_ARCH_x86 \
             -I$(INC_DIR) \
             -I/usr/include \
             -I/usr/include/x86_64-linux-gnu \
             -Wall -Wno-unused-value -Wno-compare-distinct-pointer-types

# Userspace compilation flags
LOADER_CFLAGS  = -O2 -Wall -Wextra
LOADER_LDFLAGS = -lbpf -lelf -lz

.PHONY: all model bpf loader clean explore explore-cicids overflow benchmark count-instructions test thesis-data

all: model bpf loader

# ── Step 1: Train model & export C header ────────────────────────────

model: $(INC_DIR)/model_params.h

$(INC_DIR)/model_params.h: $(SCRIPT_DIR)/train_model.py
	@echo "=== Training model and exporting parameters ==="
	$(PYTHON) $(SCRIPT_DIR)/train_model.py

# ── Step 2: Compile eBPF XDP program ─────────────────────────────────

bpf: $(BUILD_DIR)/xdp_ml.o

$(BUILD_DIR)/xdp_ml.o: $(SRC_DIR)/xdp_ml.c $(INC_DIR)/model_params.h
	@mkdir -p $(BUILD_DIR)
	@echo "=== Compiling eBPF XDP program ==="
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@
	@echo "  → $@ (BPF object)"

# ── Step 3: Compile userspace loader ─────────────────────────────────

loader: $(BUILD_DIR)/loader

$(BUILD_DIR)/loader: $(SRC_DIR)/loader.c
	@mkdir -p $(BUILD_DIR)
	@echo "=== Compiling userspace loader ==="
	$(CC) $(LOADER_CFLAGS) -o $@ $< $(LOADER_LDFLAGS)
	@echo "  → $@ (userspace loader)"

# ── Clean ────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD_DIR)
	@echo "Build artifacts removed."

# ── Exploration ──────────────────────────────────────────
explore:
	@echo "=== Running Design Space Exploration ==="
	python3 scripts/explore.py

explore-cicids:
	@echo "=== Running Exploration on CIC-IDS-2017 ==="
	python3 scripts/explore.py --data-dir data

# ── Analysis ─────────────────────────────────────────────
overflow:
	@echo "=== Running Overflow Safety Analysis ==="
	python3 scripts/overflow_analysis.py --all

# ── Benchmarks ───────────────────────────────────────────
# Usage: make benchmark IFACE=eth0 DURATION=30
IFACE ?= eth0
DURATION ?= 30
benchmark: bpf loader
	@echo "=== Running Performance Benchmark ==="
	sudo bash scripts/benchmark.sh $(IFACE) $(DURATION)

# ── Instruction count ────────────────────────────────────
count-instructions: bpf
	@echo "=== BPF Instruction Count ==="
	@llvm-objdump -d build/xdp_ml.o | grep 'Disassembly of section' 
	@for section in $$(llvm-objdump -d build/xdp_ml.o | grep 'Disassembly of section' | sed 's/.*section //;s/://'); do \
		count=$$(llvm-objdump -d --section=$$section build/xdp_ml.o 2>/dev/null | grep -c '^\s'); \
		echo "  $$section: $$count instructions"; \
	done
	@total=$$(llvm-objdump -d build/xdp_ml.o | grep -c '^\s'); \
	echo "  TOTAL: $$total instructions"

# ── Tests ────────────────────────────────────────────────
test:
	@echo "=== Running Tests ==="
	python3 tests/test_inference.py

# ── Full pipeline ────────────────────────────────────────
thesis-data: model bpf overflow explore count-instructions
	@echo "=== All thesis data generated ==="
	@echo "Results in: results/"
