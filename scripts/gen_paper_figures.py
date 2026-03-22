"""
Generate all paper figures with data from both original and QAT models.
"""
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CN_FULL = ["BENIGN", "DDOS", "PORTSCAN", "BRUTEFORCE"]
CN_DISP = ["BENIGN", "DDoS", "PortScan", "BruteForce"]
CN_SHORT = ["BEN", "DDoS", "PS", "BF"]

ablation = json.load(open("results/ablation_results.json"))
qat = json.load(open("results/qat_results.json"))

# Build a unified dict of all configs with per-class metrics
configs = {}
for m in ablation['modes']:
    configs[m['name']] = m
# Add QAT results
configs['Ternary-QAT'] = qat['ternary_qat']
configs['MapLUT+Tern-QAT\n256b'] = qat['maplut_ternary_qat']
if 'maplut_sweep' in qat and '512' in qat['maplut_sweep']:
    configs['MapLUT+Tern-QAT\n512b'] = qat['maplut_sweep']['512']['maplut_ternary_qat']

def get_f1(metrics, cls):
    return metrics['per_class'][cls]['f1']

def build_cm_from_metrics(metrics):
    """Build approximate confusion matrix from precision/recall/support."""
    NC = 4
    pc = metrics["per_class"]
    cm = np.zeros((NC, NC))
    
    for i, cn in enumerate(CN_FULL):
        sup = pc[cn]["support"]
        rec = pc[cn]["recall"]
        tp = round(rec * sup)
        fn = sup - tp
        cm[i][i] = tp
        # Distribute FN proportionally to other classes' support
        other_idx = [j for j in range(NC) if j != i]
        total_other = sum(pc[CN_FULL[j]]["support"] for j in other_idx)
        for j in other_idx:
            w = pc[CN_FULL[j]]["support"] / total_other if total_other > 0 else 1/len(other_idx)
            cm[i][j] = round(fn * w)
    return cm


# ══════════════════════════════════════════════════════════════
# Figure 1: 4-panel confusion matrix
# Original Float | Post-hoc Ternary | Ternary-QAT | MapLUT+Tern-QAT 512b
# ══════════════════════════════════════════════════════════════
cm_configs = [
    ("Float (Original)", configs['Float']),
    ("Post-hoc Ternary", configs['Int32+Ternary']),
    ("Ternary-QAT", configs['Ternary-QAT']),
    ("MapLUT+Tern-QAT\n512 bins", configs['MapLUT+Tern-QAT\n512b']),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for idx, (title, metrics) in enumerate(cm_configs):
    ax = axes[idx]
    cm = build_cm_from_metrics(metrics)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(4):
        for j in range(4):
            val = cm_norm[i][j]
            color = 'white' if val > 0.5 else 'black'
            if val > 0.005:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=8, color=color, fontweight='bold' if i==j else 'normal')
    
    ax.set_xticks(range(4)); ax.set_xticklabels(CN_SHORT, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(CN_SHORT, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    if idx == 0: ax.set_ylabel('True', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=9)
    
    acc = metrics['accuracy']
    mf1 = metrics['macro_f1']
    ax.text(1.5, 4.6, f'Acc={acc:.1%}  MF1={mf1:.3f}', ha='center', fontsize=7.5,
            style='italic', transform=ax.transData)

plt.tight_layout()
plt.savefig("paper/figures/confusion_matrix.pdf", bbox_inches='tight', dpi=300)
plt.savefig("paper/figures/confusion_matrix.png", bbox_inches='tight', dpi=150)
print("✓ confusion_matrix.pdf")

# ══════════════════════════════════════════════════════════════
# Figure 2: Per-class F1 bar chart — key configurations
# ══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 3.5))

bar_data = [
    ("Float\n(Original)", configs['Float'], '#2196F3'),
    ("Post-hoc\nTernary", configs['Int32+Ternary'], '#F44336'),
    ("Ternary\n-QAT", configs['Ternary-QAT'], '#4CAF50'),
    ("MapLUT+Tern\n-QAT 512b", configs['MapLUT+Tern-QAT\n512b'], '#FF9800'),
]

x = np.arange(4)
width = 0.8 / len(bar_data)

for idx, (label, metrics, color) in enumerate(bar_data):
    f1s = [get_f1(metrics, c) for c in CN_FULL]
    offset = (idx - len(bar_data)/2 + 0.5) * width
    bars = ax2.bar(x + offset, f1s, width * 0.9, label=label, color=color, alpha=0.85)
    for bar, v in zip(bars, f1s):
        if v > 0.03:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)

ax2.set_xticks(x)
ax2.set_xticklabels(CN_DISP, fontsize=10)
ax2.set_ylabel('F1 Score', fontsize=11)
ax2.set_ylim(0, 1.15)
ax2.legend(loc='upper right', fontsize=7.5, ncol=2)
ax2.grid(axis='y', alpha=0.3)
ax2.set_title('Per-Class F1: Post-hoc Quantization vs QAT', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("paper/figures/per_class_f1_comparison.pdf", bbox_inches='tight', dpi=300)
plt.savefig("paper/figures/per_class_f1_comparison.png", bbox_inches='tight', dpi=150)
print("✓ per_class_f1_comparison.pdf")

# ══════════════════════════════════════════════════════════════
# Figure 3: Bins sweep (QAT model — MapLUT+Ternary-QAT)
# ══════════════════════════════════════════════════════════════
bins_data = []
for bk in ['64', '128', '256', '512']:
    if bk in qat['maplut_sweep']:
        entry = qat['maplut_sweep'][bk]['maplut_ternary_qat']
        bins_data.append({
            'bins': int(bk),
            'acc': entry['accuracy'],
            'mf1': entry['macro_f1'],
            'ps_f1': entry['per_class']['PORTSCAN']['f1'],
            'bf_f1': entry['per_class']['BRUTEFORCE']['f1'],
            'mem_KB': 12 * int(bk) * 64 * 4 / 1024,
        })

if bins_data:
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(9, 3.5))
    
    bins = [d['bins'] for d in bins_data]
    accs = [d['acc'] for d in bins_data]
    mf1s = [d['mf1'] for d in bins_data]
    ps_f1s = [d['ps_f1'] for d in bins_data]
    mems = [d['mem_KB'] for d in bins_data]
    
    # Left: accuracy/F1 vs bins
    ax3a.plot(bins, [a*100 for a in accs], 'o-', color='#2196F3', linewidth=2, label='Accuracy (%)', markersize=8)
    ax3a.plot(bins, [f*100 for f in mf1s], 's-', color='#4CAF50', linewidth=2, label='Macro F1 (×100)', markersize=8)
    ax3a.plot(bins, [f*100 for f in ps_f1s], '^-', color='#FF9800', linewidth=2, label='PortScan F1 (×100)', markersize=7)
    
    # Add Ternary-QAT reference line (no MapLUT)
    tern_qat_acc = configs['Ternary-QAT']['accuracy'] * 100
    ax3a.axhline(tern_qat_acc, color='#F44336', linestyle='--', alpha=0.7, label=f'Ternary-QAT ({tern_qat_acc:.1f}%)')
    
    ax3a.set_xlabel('Number of Bins', fontsize=11)
    ax3a.set_ylabel('Score (×100)', fontsize=11)
    ax3a.set_xscale('log', base=2)
    ax3a.set_xticks(bins)
    ax3a.set_xticklabels([str(b) for b in bins])
    ax3a.set_ylim(20, 100)
    ax3a.legend(fontsize=8, loc='lower right')
    ax3a.grid(alpha=0.3)
    ax3a.set_title('MapLUT+Ternary-QAT: Bins Sweep', fontsize=10, fontweight='bold')
    
    # Right: memory vs bins
    ax3b.bar(range(len(bins)), mems, color=['#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5'], edgecolor='#1565C0')
    ax3b.set_xticks(range(len(bins)))
    ax3b.set_xticklabels([str(b) for b in bins])
    ax3b.set_xlabel('Number of Bins', fontsize=11)
    ax3b.set_ylabel('LUT Memory (KB per CPU)', fontsize=11)
    ax3b.set_title('LUT Memory Footprint', fontsize=10, fontweight='bold')
    for i, (b, m) in enumerate(zip(bins, mems)):
        ax3b.text(i, m + 10, f'{m:.0f}KB', ha='center', fontsize=9)
    ax3b.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("paper/figures/bins_sweep_qat.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("paper/figures/bins_sweep_qat.png", bbox_inches='tight', dpi=150)
    print("✓ bins_sweep_qat.pdf")

# ══════════════════════════════════════════════════════════════
# Figure 4: Instruction count comparison (updated)
# ══════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(7, 3))

versions = ['V0\nBaseline', 'V2\nFused+Ternary', 'V3\nMapLUT+Ternary', 'V3b\nMapLUT+Int32']
insns = [804, 1437, 1424, 1016]
muls = [15, 16, 0, 5]
colors = ['#E0E0E0', '#90CAF9', '#4CAF50', '#FF9800']

bars = ax4.bar(range(len(versions)), insns, color=colors, edgecolor='#333', linewidth=0.8)
for i, (bar, mul) in enumerate(zip(bars, muls)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{insns[i]} insns\n{mul} mul', ha='center', fontsize=8, fontweight='bold')

ax4.set_xticks(range(len(versions)))
ax4.set_xticklabels(versions, fontsize=9)
ax4.set_ylabel('BPF Instructions', fontsize=11)
ax4.set_ylim(0, 1700)
ax4.grid(axis='y', alpha=0.3)
ax4.set_title('eBPF Program Complexity Across Versions', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("paper/figures/instruction_comparison.pdf", bbox_inches='tight', dpi=300)
plt.savefig("paper/figures/instruction_comparison.png", bbox_inches='tight', dpi=150)
print("✓ instruction_comparison.pdf")

# ══════════════════════════════════════════════════════════════
# Figure 5: Summary accuracy table as figure (for slides)
# ══════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(8, 3))
ax5.axis('off')

table_data = [
    ['Float (Original)', '97.64%', '0.770', '0.597', '0.552', 'No', '804'],
    ['Int32 (Original)', '97.64%', '0.770', '0.597', '0.552', 'No', '804'],
    ['Post-hoc Ternary', '95.02%', '0.482', '0.020', '0.039', 'Partial', '1437'],
    ['Ternary-QAT', '96.38%', '0.742', '0.635', '0.432', 'Partial', '1424*'],
    ['MapLUT+Tern-QAT 256b', '91.97%', '0.691', '0.629', '0.383', '\\textbf{Yes}', '1424'],
    ['MapLUT+Tern-QAT 512b', '95.78%', '0.721', '0.629', '0.379', '\\textbf{Yes}', '1424'],
]
col_labels = ['Configuration', 'Accuracy', 'Macro F1', 'PS F1', 'BF F1', 'Zero-Mul?', 'Insns']

tbl = ax5.table(cellText=table_data, colLabels=col_labels, loc='center',
               cellLoc='center', colLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.3)

# Color header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
# Highlight zero-multiply rows
for i in [4, 5]:
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor('#E8F5E9')

plt.tight_layout()
plt.savefig("paper/figures/summary_table.pdf", bbox_inches='tight', dpi=300)
plt.savefig("paper/figures/summary_table.png", bbox_inches='tight', dpi=150)
print("✓ summary_table.pdf")

print("\nAll figures generated!")
