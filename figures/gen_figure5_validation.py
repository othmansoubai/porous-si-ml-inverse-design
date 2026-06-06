#!/usr/bin/env python3
"""
Paper 2 — Figure 5 (validation) — publication-ready labels
============================================================
Renames internal job labels (geo3_rerun, val1_S4d3) to descriptive
publication names that clearly identify each ML candidate.

Usage:
    python3 gen_fig_validation_v2.py

Output: fig_validation.png (overwrites old)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.expanduser("~/phase2_ml")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

# ================================================================
# DATA — verified from kappa_results_validation.csv
# ================================================================

# Display order: from low κ to high κ (left to right)
candidates = [
    {
        'short':       'Candidate 1\n(target κ = 1.5)',  # was val1_S4d3
        'desc':        'S=4, d=3 (37.5% stagger), φ ≈ 30%',
        'nemd_mean':   1.39,
        'nemd_std':    0.00,
        'ml_mean':     1.52,
        'ml_std':      0.20,
        'seed_kappa': [1.40, 1.39, 1.39],   # 3 seeds: 12345, 34567, 56789
        'color':       '#3B9B6D',
    },
    {
        'short':       'Candidate 2\n(target κ = 3.0)',  # was geo3_rerun
        'desc':        'S=3, d=2 (25% stagger), φ ≈ 17.5%',
        'nemd_mean':   3.03,
        'nemd_std':    0.14,
        'ml_mean':     3.02,
        'ml_std':      0.36,
        'seed_kappa': [3.13, 2.93],          # 2 new seeds: 91234, 11111
        'color':       '#E8A048',
    },
]

# ================================================================
# FIGURE
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("ML Validation: NEMD vs Predicted κ",
             fontsize=14, fontweight='bold')

# ----------------------------------------------------------------
# Panel (a): bar chart NEMD vs ML
# ----------------------------------------------------------------
ax = axes[0]
n = len(candidates)
x = np.arange(n)
w = 0.35

nemd_vals = [c['nemd_mean'] for c in candidates]
nemd_errs = [c['nemd_std']  for c in candidates]
ml_vals   = [c['ml_mean']   for c in candidates]
ml_errs   = [c['ml_std']    for c in candidates]

ax.bar(x - w/2, nemd_vals, w, yerr=nemd_errs, color='#E8706A',
       edgecolor='white', linewidth=1, capsize=5,
       label='NEMD (actual)', zorder=3)
ax.bar(x + w/2, ml_vals, w, yerr=ml_errs, color='#5BA3CF',
       edgecolor='white', linewidth=1, capsize=5,
       label='ML prediction', zorder=3)

# Value labels on bars
for i, c in enumerate(candidates):
    ax.text(i - w/2, c['nemd_mean'] + c['nemd_std'] + 0.1,
            f"{c['nemd_mean']:.2f}", ha='center', fontsize=9,
            fontweight='bold', color='#9B2C2C')
    ax.text(i + w/2, c['ml_mean'] + c['ml_std'] + 0.1,
            f"{c['ml_mean']:.2f}", ha='center', fontsize=9,
            fontweight='bold', color='#1E40AF')

# X-axis with descriptive labels
ax.set_xticks(x)
ax.set_xticklabels([c['short'] for c in candidates], fontsize=10)
ax.tick_params(axis='x', pad=3)

# Add geometry description below labels
# Put the italic description clearly below the 2-line x tick label.
for i, c in enumerate(candidates):
    ax.text(i, -0.18, c['desc'],
            transform=ax.get_xaxis_transform(),
            ha='center', va='top',
            fontsize=8.5, style='italic', color='#4A5568',
            clip_on=False)

ax.set_ylabel('κ (W/(m·K))', fontsize=12)
ax.set_title("(a) NEMD vs ML Prediction", fontsize=12, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left', bbox_to_anchor=(0.01, 0.99),
          framealpha=0.95, borderpad=0.4)
ax.set_ylim(0, max(max(nemd_vals), max(ml_vals)) * 1.3)
ax.grid(True, alpha=0.25, linestyle=':', axis='y')

# Within-1σ status annotation
ax.text(0.5, 0.98, '✓ Both predictions within 1σ',
        transform=ax.transAxes, ha='center', va='top', fontsize=10,
        fontweight='bold', color='#2B7A3E',
        bbox=dict(boxstyle='round,pad=0.3', fc='#F0FFF4',
                  ec='#2B7A3E', alpha=0.9))

# ----------------------------------------------------------------
# Panel (b): per-seed scatter
# ----------------------------------------------------------------
ax = axes[1]

# Set up x positions: seeds for cand 1, then seeds for cand 2
all_x, all_y, all_colors = [], [], []
xtick_positions = []
xtick_labels = []

run_idx = 0
for i, c in enumerate(candidates):
    seeds = c['seed_kappa']
    n_seeds = len(seeds)
    # plot seeds for this candidate
    seed_x = np.arange(run_idx, run_idx + n_seeds)
    ax.scatter(seed_x, seeds, color=c['color'], s=80,
               edgecolors='white', linewidth=1, zorder=5,
               label=c['short'].replace('\n', ' '))

    # ML prediction line
    ax.axhline(c['ml_mean'], color=c['color'], linestyle='--',
               alpha=0.5, lw=1.5, xmin=run_idx/(n+sum(len(c['seed_kappa']) for c in candidates)),
               xmax=1)

    # ±1σ shaded band for ML prediction
    x_band_start = run_idx - 0.4
    x_band_end = run_idx + n_seeds - 0.6
    ax.fill_between([x_band_start, x_band_end],
                     [c['ml_mean'] - c['ml_std']]*2,
                     [c['ml_mean'] + c['ml_std']]*2,
                     color=c['color'], alpha=0.15, zorder=2)

    # Group label below x-axis
    group_center = run_idx + (n_seeds - 1) / 2
    xtick_positions.append(group_center)
    xtick_labels.append(c['short'])

    run_idx += n_seeds + 1  # gap between groups

# Vertical separator between candidate groups
sep_x = candidates[0]['seed_kappa'].__len__() + 0.0
ax.axvline(sep_x, color='gray', linestyle=':', lw=0.8, alpha=0.5)

ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, fontsize=10)
ax.set_ylabel('κ (W/(m·K))', fontsize=12)
ax.set_title("(b) Per-Seed Results with ML ±1σ Band",
             fontsize=12, fontweight='bold')

# Custom legend for ML prediction line
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3B9B6D',
           markersize=10, label='Candidate 1 seeds'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E8A048',
           markersize=10, label='Candidate 2 seeds'),
    Line2D([0], [0], color='gray', linestyle='--', lw=1.5,
           label='ML prediction'),
]
ax.legend(handles=custom_lines, fontsize=9, loc='center right')

ax.grid(True, alpha=0.25, linestyle=':')

fig.tight_layout(rect=[0, 0.12, 1, 0.95])
out = os.path.join(OUTDIR, "fig_validation.png")
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✅ Saved: {out}")
print("Renamed: geo3_rerun → 'Candidate 2 (target κ = 3.0)'")
print("Renamed: val1_S4d3 → 'Candidate 1 (target κ = 1.5)'")
