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
data = [
    # label, short_name, group, phi, kappa, kappa_std, variable_annotation
    ("bulk_si",          "Bulk Si\n(φ=0%)",         "P1", 0.00,  12.03, 0.29, ""),
    ("P1_low_50pct",     "Low\n(φ=7%)",             "P1", 6.84,  7.63,  0.99, ""),
    ("P1_high_50pct",    "High\n(φ=28%)",           "P1", 28.27, 2.59,  0.02, ""),

    ("A_S2_d2",          "S=2\n(φ=9%)",             "A",  8.54,  4.56,  0.10, ""),
    ("A_S2.5_d2",        "S=2.5\n(φ=13%)",          "A",  12.63, 4.02,  0.22, ""),
    ("A_S3_d2",          "S=3\n(φ=18%)",            "A",  17.50, 3.16,  0.21, ""),
    ("A_S4.5_d2",        "S=4.5\n(φ=37%)",          "A",  36.78, 1.28,  0.02, ""),

    ("B_S4_aligned_low", "0%\n(φ≈6%)",              "B",  6.25,  7.06,  1.14, ""),
    ("B_S4_quarter_low", "25%\n(φ≈7%)",             "B",  7.29,  4.39,  0.14, ""),
    ("B_S4_aligned_high","0%\n(φ≈28%)",             "B",  28.32, 3.74,  0.20, ""),
    ("B_S4_quarter_high","25%\n(φ≈30%)",            "B",  29.57, 1.82,  0.04, ""),

    ("C_d0_aligned",     "d=0\nneck=5",             "C",  17.50, 4.68,  0.64, ""),
    ("C_d1_12pct",       "d=1\nneck=4",             "C",  17.50, 3.83,  0.57, ""),
    ("C_d3_37pct",       "d=3\nneck=2",             "C",  17.50, 2.85,  0.35, ""),
    ("C_d4_50pct",       "d=4\nneck=2",             "C",  17.50, 2.91,  0.16, ""),

    ("D_AR0.5_tall",     "AR=0.5\n(tall)",          "D",  17.50, 3.53,  0.10, ""),
    ("D_AR2.0_wide",     "AR=2.0\n(wide)",          "D",  17.50, 3.22,  0.18, ""),
    ("D_AR2.5_vwide",    "AR=2.5\n(v.wide)",        "D",  17.50, 2.69,  0.10, ""),

    ("V_S4d3_37pct",     "S=4,d=3\n(φ=30%)",        "V",  29.93, 1.39,  0.00, ""),
]

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
    "B":  "Group B: Stagger sweep",
    "C":  "Group C: Neck width sweep",
    "D":  "Group D: Aspect ratio sweep",
    "V":  "Validation (ML-suggested)",
}

# Variable isolated per group
C_GRAY = '#4A5568'

GROUP_VARS = {
    "P1": "Reference\npoints",
    "A":  "Variable:\nPorosity φ",
    "B":  "Variable:\nStagger %",
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
ax.set_title('NEMD Dataset: 19 Porous Silicon Geometries Organized by Experimental Group',
             fontsize=14, fontweight='bold', pad=40)
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
ax.annotate('κ spans nearly\n1 order of magnitude\n(1.28 – 12.03 W/(m·K))',
            xy=(6, 1.28), xytext=(14.5, 6.5),
            fontsize=10, ha='center', fontweight='bold', color=C_GRAY,
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                            connectionstyle='arc3,rad=0.2'))

fig.tight_layout(rect=[0, 0, 1, 0.93])
path = os.path.join(OUTDIR, "fig2_dataset_overview_legend_fixed.png")
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✅ Saved: {path}")
