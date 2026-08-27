#!/usr/bin/env python3
"""
Paper 2 — Figure 1 v2: Fixed Pore A label visibility
======================================================
Fix: moved stagger annotation (d) outside the pore box so Pore A
label is fully visible. Also improved aspect ratio annotation placement.

Usage:
    python3 gen_fig1_paper2_v2.py

Output: fig1_methodology.png (overwrites old)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

# Colors
C_BLUE   = '#2B6CB0'
C_GREEN  = '#2B7A3E'
C_RED    = '#C53030'
C_ORANGE = '#DD6B20'
C_PURPLE = '#6B46C1'
C_GRAY   = '#4A5568'
C_PORE   = '#FEB2B2'
C_SI     = '#C6F6D5'
C_NECK   = '#FEFCBF'

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.08)

# ================================================================
# (a) WORKFLOW (unchanged - was working well)
# ================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('(a) Inverse Design Workflow', fontsize=14, fontweight='bold', pad=15)

boxes = [
    (5, 9.0, 8.0, 1.2, ['NEMD Dataset Generation', '18 porous Si geometries × 3 seeds'], C_BLUE, 'white'),
    (5, 7.0, 8.0, 1.2, ['Feature Engineering', 'φ, S, stagger, neck width, AR → κ'], C_PURPLE, 'white'),
    (5, 5.0, 8.0, 1.2, ['ML Training (LOOCV)', 'Random Forest + Gaussian Process'], C_GREEN, 'white'),
    (5, 3.0, 8.0, 1.2, ['GP Inverse Design', 'Target κ → Optimal geometry (φ*, S*, d*, neck*, AR*)'], C_ORANGE, 'white'),
    (5, 1.0, 8.0, 1.2, ['NEMD Validation', '3 out-of-sample geometries validated'], C_RED, 'white'),
]

for (cx, cy, w, h, texts, fc, tc) in boxes:
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.15", facecolor=fc, edgecolor='white',
                          linewidth=2, alpha=0.92, zorder=3)
    ax1.add_patch(box)
    ax1.text(cx, cy + 0.15, texts[0], ha='center', va='center',
             fontsize=12, fontweight='bold', color=tc, zorder=4)
    if len(texts) > 1:
        ax1.text(cx, cy - 0.25, texts[1], ha='center', va='center',
                 fontsize=9.5, color=tc, alpha=0.9, zorder=4)

for y_start, y_end in [(8.4, 7.6), (6.4, 5.6), (4.4, 3.6), (3.4, 1.6)]:
    ax1.annotate('', xy=(5, y_end), xytext=(5, y_start),
                 arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=2.5))

side_notes = [
    (9.5, 9.0, 'LAMMPS\nMüller-Plathe\n500 ps × 3 seeds', C_BLUE),
    (9.5, 7.0, '5 features\n1 target (κ)', C_PURPLE),
    (9.5, 5.0, 'n = 18\nMAE ≈ 0.39\nR² ≈ 0.94', C_GREEN),
    (9.5, 3.0, 'Constrained\ngrid search\n+ uncertainty', C_ORANGE),
    (9.5, 1.0, '1.39 vs 1.34\n2.09 vs 2.12\n5.35 vs 4.73', C_RED),
]
for (x, y, text, color) in side_notes:
    ax1.text(x, y, text, ha='center', va='center', fontsize=8,
             color=color, fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.7, lw=0.8))

ax1.annotate('', xy=(1.0, 8.6), xytext=(1.0, 1.4),
             arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.5, linestyle='--'))
ax1.text(0.5, 5.0, 'Feedback\nloop', ha='center', va='center', fontsize=8,
         color=C_GRAY, rotation=90, fontstyle='italic')


# ================================================================
# (b) PORE GEOMETRY — FIXED ANNOTATION LAYOUT
# ================================================================
ax2 = fig.add_subplot(gs[1])
# Expanded limits to give more room for annotations
ax2.set_xlim(-2.5, 11.5)
ax2.set_ylim(-3.5, 10.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('(b) Pore Geometry: 5 Design Variables', fontsize=14, fontweight='bold', pad=15)

a = 1.0
box_size = 8 * a

# Si background
si_bg = mpatches.Rectangle((0, 0), box_size, box_size, facecolor=C_SI,
                             edgecolor='black', linewidth=2, zorder=1)
ax2.add_patch(si_bg)

# Grid lines
for i in range(9):
    ax2.plot([i * a, i * a], [0, box_size], color='gray', lw=0.3, alpha=0.4, zorder=1.5)
    ax2.plot([0, box_size], [i * a, i * a], color='gray', lw=0.3, alpha=0.4, zorder=1.5)

# Pore A: [0,3] × [0,3] (S=3)
pA_x, pA_y, pA_w, pA_h = 0, 0, 3*a, 3*a
poreA = mpatches.Rectangle((pA_x, pA_y), pA_w, pA_h, facecolor=C_PORE,
                             edgecolor=C_RED, linewidth=2, alpha=0.7, zorder=2)
ax2.add_patch(poreA)
# Pore A label — CENTERED inside pore, high z-order
ax2.text(pA_x + pA_w/2, pA_y + pA_h/2, 'Pore A', ha='center', va='center',
         fontsize=12, fontweight='bold', color=C_RED, zorder=10)

# Pore B: [2,5] × [2,5]
d = 2
pB_x, pB_y, pB_w, pB_h = d*a, d*a, 3*a, 3*a
poreB = mpatches.Rectangle((pB_x, pB_y), pB_w, pB_h, facecolor='#FED7D7',
                             edgecolor=C_RED, linewidth=2, alpha=0.7,
                             linestyle='--', zorder=2)
ax2.add_patch(poreB)
ax2.text(pB_x + pB_w/2, pB_y + pB_h/2, 'Pore B', ha='center', va='center',
         fontsize=12, fontweight='bold', color=C_RED, zorder=10)

# Neck channels (yellow highlights)
neck_patch = mpatches.Rectangle((5*a, 0), 3*a, 2*a, facecolor=C_NECK,
                                 edgecolor=None, alpha=0.5, zorder=1.8)
ax2.add_patch(neck_patch)
neck_patch2 = mpatches.Rectangle((0, 5*a), 2*a, 3*a, facecolor=C_NECK,
                                  edgecolor=None, alpha=0.5, zorder=1.8)
ax2.add_patch(neck_patch2)

# ── ANNOTATION 1: Pore size S ──
# Moved BELOW the figure (further down) to avoid cluttering
ax2.annotate('', xy=(3*a, -1.0), xytext=(0, -1.0),
             arrowprops=dict(arrowstyle='<->', color=C_BLUE, lw=2))
ax2.text(1.5*a, -1.7, 'S = 3 u.c.', ha='center', va='center',
         fontsize=11, fontweight='bold', color=C_BLUE,
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_BLUE, alpha=0.95))

# ── ANNOTATION 2: Stagger offset d ──
# MOVED: arrow stays in original position, but label moved OUTSIDE box (to the left)
ax2.annotate('', xy=(2*a, 0.5*a), xytext=(0, 0.5*a),
             arrowprops=dict(arrowstyle='<->', color=C_ORANGE, lw=2.5))
# Label now to the LEFT of the box, with arrow pointing to the offset region
ax2.text(-1.6, 0.5*a, 'd = 2 u.c.\n(25%)', ha='center', va='center',
         fontsize=10, fontweight='bold', color=C_ORANGE,
         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_ORANGE, alpha=0.95))
# Small connector line
ax2.annotate('', xy=(0, 0.5*a), xytext=(-0.9, 0.5*a),
             arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=1, alpha=0.6))

# ── ANNOTATION 3: Neck width ──
ax2.annotate('', xy=(8*a, 3.5*a), xytext=(5*a, 3.5*a),
             arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=2))
ax2.text(6.5*a, 4.2*a, 'Neck = 3 u.c.\n(16.3 Å)', ha='center', va='center',
         fontsize=9, fontweight='bold', color=C_GREEN,
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_GREEN, alpha=0.95))

# ── ANNOTATION 4: Aspect ratio (H and W) ──
# H arrow (left side of Pore A)
ax2.annotate('', xy=(-0.3, 3*a), xytext=(-0.3, 0),
             arrowprops=dict(arrowstyle='<->', color=C_PURPLE, lw=1.5))
ax2.text(-0.8, 1.5*a, 'H', ha='center', va='center',
         fontsize=11, fontweight='bold', color=C_PURPLE)

# W arrow (below, distinct from S arrow)
ax2.annotate('', xy=(3*a, -2.4), xytext=(0, -2.4),
             arrowprops=dict(arrowstyle='<->', color=C_PURPLE, lw=1.5))
ax2.text(1.5*a, -3.0, 'W    →    AR = W/H', ha='center', va='center',
         fontsize=10, fontweight='bold', color=C_PURPLE,
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_PURPLE, alpha=0.95))

# ── ANNOTATION 5: Porosity φ ──
ax2.text(7.0*a, 7.5*a, 'φ = removed\n     atoms / total', ha='center', va='center',
         fontsize=9, fontweight='bold', color=C_GRAY,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GRAY, alpha=0.9))

# Heat flow arrow
ax2.annotate('', xy=(9.0, 4*a), xytext=(9.0, 0.5*a),
             arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
ax2.text(9.9, 2.5*a, 'Heat\nflow\n(z)', ha='center', va='center',
         fontsize=10, fontweight='bold', color='black')

# COLD / HOT labels
ax2.text(4*a, -0.2, 'COLD (z = 0)', ha='center', va='top', fontsize=9,
         color='blue', fontweight='bold')
ax2.text(4*a, 8.3, 'HOT (z = Lz/2)', ha='center', va='bottom', fontsize=9,
         color='red', fontweight='bold')

ax2.text(8*a, -0.2, '8 × 8 u.c.\nx–y cross-section', ha='right', va='top',
         fontsize=8, color=C_GRAY, fontstyle='italic')

# Legend
legend_items = [
    mpatches.Patch(facecolor=C_SI, edgecolor='gray', label='Silicon'),
    mpatches.Patch(facecolor=C_PORE, edgecolor=C_RED, label='Pore (vacuum)'),
    mpatches.Patch(facecolor=C_NECK, edgecolor='gray', label='Neck channel'),
]
ax2.legend(handles=legend_items, loc='upper left', fontsize=9,
           framealpha=0.9, edgecolor='gray')

# Save
path = os.path.join(OUTDIR, "fig1_methodology.png")
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(path.replace('.png','.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✅ Saved: {path}")
print("Fix applied: Pore A label now fully visible (stagger annotation moved left)")
