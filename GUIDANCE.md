# 🟢 Updated Guidance v4 (2026-03-23 04:30 CST)

## All Experiments Complete ✓

### Final Results Table

| Config | Source | Acc | Macro F1 | PS F1 | BF F1 | Zero-Mul? | BPF Insns |
|---|---|---|---|---|---|---|---|
| Float | Original | 97.64% | 0.770 | 0.597 | 0.552 | No | 804 |
| Int32 | Original | 97.64% | 0.770 | 0.597 | 0.552 | No | 804 |
| Post-hoc Ternary | Original | 95.02% | 0.482 | 0.020 | 0.039 | Partial | 1,437 |
| MapLUT+Int32 (64b) | Original | 90.39% | 0.508 | 0.029 | 0.220 | Partial | 1,016 |
| **Ternary-QAT** | **QAT** | **96.38%** | **0.742** | **0.635** | **0.432** | Partial | 1,424 |
| **MapLUT+Tern-QAT 256b** | **QAT** | **91.97%** | **0.691** | **0.629** | **0.383** | **YES** | 1,424 |
| **MapLUT+Tern-QAT 512b** | **QAT** | **95.78%** | **0.721** | **0.629** | **0.379** | **YES** | 1,424 |

### BPF Variants

| Version | File | Total Insns | Multiply Insns | Description |
|---------|------|------------|----------------|-------------|
| V0 | xdp_ml.c | 804 | 15 | Baseline int32 MLP |
| V2 | xdp_ml_v2.c | 1,437 | 16 | Fused+Ternary+Exit |
| V3 | xdp_ml_v3.c | 1,424 | 0 | MapLUT+Ternary+Exit |
| V3b | xdp_ml_v3b.c | 1,016 | 5 | MapLUT+Int32 (no ternary) |

### Key Insights
- **PortScan F1 improves** from 0.597 (float) to 0.629 (MapLUT+Ternary-QAT)
- QAT with focal loss forces model to focus on rare classes
- EarlyExit is harmful (74.88% acc) — deferred to future work
- Two models needed: Original for float/int32, QAT for ternary

### Paper Status
- Paper: `paper/main.tex` — 7 pages, compiles cleanly
- Slides: `slides/sage-presentation.html` + `.pdf` — 12 slides, updated with QAT
- Figures: confusion_matrix, per_class_f1_comparison, bins_sweep_qat, instruction_comparison

### Float Model Bins Sweep
- Running as background task — will provide MapLUT+Int32 on original model for completeness
- Not critical for paper narrative (QAT bins sweep is the main result)

---
*v4 — 2026-03-23 04:30 CST*
