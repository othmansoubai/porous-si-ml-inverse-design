#!/usr/bin/env python3
"""
Paper 2 — Figure 2: Dataset Overview by Experimental Group
============================================================
Bar chart showing κ for all 19 geometries, organized and colored by group.
Each group's isolated variable is annotated.

Usage:
    python3 gen_fig2_paper2.py

Output: fig2_dataset_overview.png
"""

import os as _os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

# ================================================================
# DATA — 19 geometries in display order
# ================================================================
import re as _re
_txt = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "..", "ml", "ml_train_paper2.py")).read()
_ns = {}
exec(_re.search(r"^DATA\s*=\s*\[.*?^\]", _txt, _re.S | _re.M).group(0), _ns)

_SHORT = {
    "bulk_si":            "Bulk Si\n($\\varphi$=0%)",
    "P1_high_50pct":      "P1 high\n($\\varphi$=28%)",
    "A_S2_d2":            "S=2\n($\\varphi$=8.5%)",
    "A_S2.5_d2":          "S=2.5\n($\\varphi$=12.6%)",
    "A_S3_d2":            "S=3\n($\\varphi$=17.5%)",
    "A_S4.5_d2":          "S=4.5\n($\\varphi$=36.8%)",
    "B_S2_aligned_low":   "S=2, 0%\n($\\varphi$=8.0%)",
    "B_S2_d4_low":        "S=2, 50%\n($\\varphi$=8.6%)",
    "B_S4_aligned_high":  "S=4, 0%\n($\\varphi$=28.3%)",
    "B_S4_quarter_high":  "S=4, 25%\n($\\varphi$=29.6%)",
    "C_d0_aligned":       "d=0\nneck=5",
    "C_d1_12pct":         "d=1\nneck=4",
    "C_d3_37pct":         "d=3\nneck=2",
    "C_d4_50pct":         "d=4\nneck=2",
    "D_AR0.5_tall":       "AR=0.5\n(tall)",
    "D_AR2.0_wide":       "AR=2.0\n(wide)",
    "D_AR2.5_vwide":      "AR=2.5\n(v.wide)",
    "V_S4d3_37pct":       "S=4, d=3\n($\\varphi$=29.9%)",
}
_GRP = {"P1":"P1","A":"A","B":"B","C":"C","D":"D","V":"V"}

data = [(r["label"], _SHORT.get(r["label"], r["label"]), _GRP[r["group"]],
         r["phi"], r["kappa"], r["kappa_std"], "")
        for r in _ns["DATA"]]
print(f"Figure 2 built from {len(data)} entries")

labels      = [d[1] for d in data]
groups      = [d[2] for d in data]
kappas      = [d[4] for d in data]
kappa_stds  = [d[5] for d in data]

# Group colors
GROUP_COLORS = {
    "P1": "#E8706A",
    "A":  "#5BA3CF",
    "B":  "#3B9B6D",
    "C":  "#E8A048",
    "D":  "#9B6DB0",
    "V":  "#FF6B9D",
}

GROUP_LABELS = {
    "P1": "Reference points",
    "A":  "Group A: Porosity sweep",
    "B":  "Group B: Stagger and pore size",
    "C":  "Group C: Neck width sweep",
    "D":  "Group D: Aspect ratio sweep",
    "V":  "Validation (ML-suggested)",
}

# Variable isolated per group
C_GRAY = '#4A5568'

GROUP_VARS = {
    "P1": "Reference\npoints",
    "A":  "Variable:\nPorosity φ",
    "B":  "Variable:\nStagger + pore size",
    "C":  "Variable:\nNeck width",
    "D":  "Variable:\nAspect ratio",
    "V":  "ML\nvalidation",
}

# ================================================================
# FIGURE
# ================================================================
fig, ax = plt.subplots(figsize=(18, 7))

x = np.arange(len(data))
colors = [GROUP_COLORS[g] for g in groups]

bars = ax.bar(x, kappas, yerr=kappa_stds, color=colors,
              edgecolor='white', linewidth=1.2, capsize=3,
              width=0.75, zorder=3, alpha=0.9)

# Value labels on bars
for i, (k, s) in enumerate(zip(kappas, kappa_stds)):
    y_pos = k + s + 0.25
    ax.text(i, y_pos, f'{k:.2f}', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold', color=colors[i])

# X-axis labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5, rotation=0, ha='center')

# Group separators and headers
group_boundaries = []
current_group = groups[0]
start = 0
for i, g in enumerate(groups):
    if g != current_group:
        group_boundaries.append((start, i-1, current_group))
        current_group = g
        start = i
group_boundaries.append((start, len(groups)-1, current_group))

for (s, e, grp) in group_boundaries:
    mid = (s + e) / 2

    # Group header above bars
    max_k_in_group = max(kappas[s:e+1]) + max(kappa_stds[s:e+1])
    y_header = min(max_k_in_group + 1.5, 14)

    ax.text(mid, 14.2, GROUP_VARS[grp], ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=GROUP_COLORS[grp],
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=GROUP_COLORS[grp], alpha=0.9, lw=1.5))

    # Bracket underneath
    ax.plot([s - 0.4, s - 0.4, e + 0.4, e + 0.4], [13.8, 14.0, 14.0, 13.8],
            color=GROUP_COLORS[grp], lw=1.5, clip_on=False)

    # Vertical separator (except before first group)
    if s > 0:
        ax.axvline(s - 0.5, color='gray', linestyle=':', lw=0.8, alpha=0.5, zorder=1)

# Reference lines
ax.axhline(12.03, color='gray', linestyle='--', lw=1, alpha=0.4, zorder=1)
ax.text(len(data) - 0.5, 12.3, 'Bulk Si baseline', ha='right', va='bottom',
        fontsize=8, color='gray', fontstyle='italic')

# Styling
ax.set_ylabel('Thermal Conductivity κ (W/(m·K))', fontsize=13, fontweight='bold')
ax.set_xlabel('Geometry', fontsize=12)
ax.set_ylim(0, 16.5)
ax.set_xlim(-0.6, len(data) - 0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_patches = [mpatches.Patch(facecolor=GROUP_COLORS[g], edgecolor='white',
                                  label=GROUP_LABELS[g])
                  for g in ["P1", "A", "B", "C", "D", "V"]]
# Put the legend slightly above the plotting area so it does not hide
# the group header/bracket annotations.
ax.legend(handles=legend_patches,
          loc='upper right',
          bbox_to_anchor=(0.99, 1.075),
          fontsize=9,
          framealpha=0.95,
          edgecolor='gray',
          ncol=2,
          borderaxespad=0.0)

# Key insight annotation

fig.tight_layout(rect=[0, 0, 1, 0.93])
path = os.path.join(OUTDIR, "fig2_dataset_overview_legend_fixed.png")
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(path.replace('.png','.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✅ Saved: {path}")
