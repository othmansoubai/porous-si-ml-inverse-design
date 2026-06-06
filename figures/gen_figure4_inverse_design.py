#!/usr/bin/env python3
"""
Paper 2 — Figure 4 v2: Fixed annotation overlap
=================================================
Fix: annotations for κ=1.5 and κ=2.0 candidates no longer overlap.
- κ=1.5 annotation moved to LEFT of its marker
- κ=2.0 annotation stays to the RIGHT but lower
- Added connector arrows so label-to-point association is clear

Usage:
    python3 gen_fig4_paper2_v2.py

Output: fig_inverse_design.png (overwrites old)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

# ================================================================
# DATA — Inverse design candidates from ml_train_paper2.py output
# ================================================================
# From the v3 run (n=19): these are the 4 candidate predictions
candidates = [
    # (target, predicted, sigma, phi, S, stagger, neck_uc, AR, color)
    (1.5, 1.50, 0.22, 30.9,  4.1, 30.5,  1.5, 1.01, '#3182CE'),
    (2.0, 2.00, 0.24, 28.7,  4.0, 24.8,  2.1, 0.93, '#DD6B20'),
    (3.0, 3.00, 0.19, 17.6,  3.0, 29.8,  2.7, 1.47, '#38A169'),
    (5.0, 5.00, 0.22,  9.2,  2.4, 27.0,  4.1, 0.96, '#E53E3E'),
]

# ================================================================
# FIGURE
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Inverse Design: ML-Suggested Geometries for NEMD Validation",
             fontsize=14, fontweight='bold')

# ----------------------------------------------------------------
# (a) Target vs Prediction — FIXED annotation placement
# ----------------------------------------------------------------
ax = axes[0]

targets = [c[0] for c in candidates]
preds   = [c[1] for c in candidates]
sigmas  = [c[2] for c in candidates]

# Error bar plot
ax.errorbar(targets, preds, yerr=sigmas, fmt='o', markersize=14,
            capsize=5, color='#38A169', markeredgecolor='white',
            markeredgewidth=1.5, zorder=5, elinewidth=1.5)

# Diagonal reference line
lim = [0, 6.3]
ax.plot(lim, lim, '--', color='gray', lw=1.5, alpha=0.5, zorder=1)

# ── FIXED ANNOTATIONS ──
# For each candidate, place annotation using dynamic offset logic
# to avoid collisions between adjacent points

annotation_configs = [
    # (target, dx, dy, ha) — offset direction for each annotation
    (1.5, -0.5, -0.5, 'right'),   # LEFT side, below — moved away from κ=2.0
    (2.0,  0.5, -0.3, 'left'),    # RIGHT side, slightly below
    (3.0,  0.4,  0.4, 'left'),    # upper-right
    (5.0, -0.4,  0.4, 'right'),   # upper-left
]

for (target, dx, dy, ha) in annotation_configs:
    c = next(c for c in candidates if c[0] == target)
    _, pred, sigma, phi, S, stagger, neck, AR, _ = c
    text = f'φ = {phi:.0f}%\nneck = {neck:.1f}'
    ax.annotate(text,
                xy=(target, pred),
                xytext=(target + dx, pred + dy),
                fontsize=9.5,
                ha=ha,
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='#4A5568', alpha=0.95, lw=0.8),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.6),
                zorder=6)

ax.set_xlabel("Target κ (W/m·K)", fontsize=12)
ax.set_ylabel("Predicted κ (W/m·K)", fontsize=12)
ax.set_title("(a) Target vs Prediction", fontsize=12, fontweight='bold')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.grid(True, alpha=0.25, linestyle=':')

# ----------------------------------------------------------------
# (b) Candidate Feature Profiles
# ----------------------------------------------------------------
ax = axes[1]

# Feature bounds for normalization (matching the inverse design bounds)
bounds = [(5, 40), (2, 5), (0, 50), (1.5, 5), (0.5, 3)]  # phi, S, stagger, neck, AR
feat_names = ['φ', 'S', 'stagger', 'neck', 'AR']
x_pos = np.arange(len(feat_names))

for c in candidates:
    _, pred, sigma, phi, S, stagger, neck, AR, color = c
    vals = [phi, S, stagger, neck, AR]
    # Normalize to [0, 1]
    normed = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(vals, bounds)]
    ax.plot(x_pos, normed, '-o', markersize=9, lw=2.5, color=color, alpha=0.85,
            label=f'κ = {c[0]:.1f} → {pred:.2f} ± {sigma:.2f}',
            markeredgecolor='white', markeredgewidth=1)

ax.set_xticks(x_pos)
ax.set_xticklabels(feat_names, fontsize=11)
ax.set_ylabel("Normalized feature value", fontsize=12)
ax.set_ylim(-0.05, 1.1)
ax.set_title("(b) Candidate Feature Profiles", fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle=':')

# Add interpretive annotation
ax.text(0.02, 0.97, 'Lower κ target →\nhigher φ, smaller neck',
        transform=ax.transAxes, fontsize=9, style='italic',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBEB',
                  ec='#D69E2E', alpha=0.9, lw=0.8))

fig.tight_layout(rect=[0, 0, 1, 0.94])
path = os.path.join(OUTDIR, "fig_inverse_design.png")
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✅ Saved: {path}")
print("Fix applied: annotations for κ=1.5 and κ=2.0 no longer overlap")
