#!/usr/bin/env python3
"""
Paper 2 — Supplementary Figure S1: Temperature Profiles of Validation Runs
============================================================================
Shows NEMD temperature profiles for the two validation geometries
(val1_S4d3 and geo3_rerun) to demonstrate steady-state convergence
and well-defined temperature gradients.

Usage:
    python3 gen_figS1_validation_profiles.py

Output: figS1_validation_profiles.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, glob

RESULTS_DIR = os.path.expanduser("~/phase2_ml/results")
OUTDIR      = os.path.expanduser("~/phase2_ml")

# ================================================================
# Read temperature profile data
# ================================================================

def read_all_blocks(filepath):
    """Read LAMMPS ave/chunk output — one block per timestep dump."""
    blocks, current = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) == 3:  # header line
                if current:
                    blocks.append(np.array(current))
                current = []
            else:
                current.append([float(x) for x in parts])
    if current:
        blocks.append(np.array(current))
    return blocks


def get_seed_profile(label, seed):
    """Load temp profile for given geometry+seed, average last 80% of timesteps."""
    path = os.path.join(RESULTS_DIR, f"temp_profile_{label}_seed{seed}.dat")
    if not os.path.exists(path):
        print(f"⚠ Missing: {path}")
        return None, None
    blocks = read_all_blocks(path)
    if not blocks:
        return None, None
    n_skip = len(blocks) // 5              # drop first 20% (equilibration)
    useful = blocks[n_skip:]
    coords = blocks[0][:, 1]               # z-position (Å)
    temps  = np.array([b[:, 3] for b in useful])  # temperature column
    return coords, temps.mean(axis=0)


def get_all_seeds(label, seeds):
    """Get average profile + std across seeds."""
    all_profiles = []
    coords_ref   = None
    for s in seeds:
        coords, profile = get_seed_profile(label, s)
        if profile is not None:
            all_profiles.append(profile)
            if coords_ref is None:
                coords_ref = coords
    if not all_profiles:
        return None, None, None
    arr = np.array(all_profiles)
    return coords_ref, arr.mean(axis=0), arr.std(axis=0)


# ================================================================
# FIGURE
# ================================================================

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, wspace=0.3)

# Styling
C_GREEN  = '#38A169'
C_ORANGE = '#DD6B20'
C_GRAY   = '#4A5568'

# ----------------------------------------------------------------
# Panel (a): val1_S4d3 — ML-designed geometry (target κ = 1.5)
# ----------------------------------------------------------------
ax = fig.add_subplot(gs[0])
coords, mean_t, std_t = get_all_seeds("val1_S4d3", [12345, 34567, 56789])

if coords is not None:
    ax.plot(coords, mean_t, '-o', color=C_GREEN, ms=12, lw=2.5,
            label='Mean (3 seeds)', zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(coords, mean_t - std_t, mean_t + std_t,
                    alpha=0.25, color=C_GREEN, label='±1σ spread', zorder=3)
    ax.axhline(300, color='gray', linestyle='--', lw=1.2, alpha=0.6, label='Target (300 K)')

    # Mark cold and hot slabs
    ax.axvline(coords[0], color='blue', linestyle=':', lw=1.2, alpha=0.5)
    ax.axvline(coords[len(coords)//2], color='red', linestyle=':', lw=1.2, alpha=0.5)

    # Slab labels — positioned INSIDE the plot area
    # HOT is at high temperature, so label goes BELOW and to the right of the peak
    # COLD is at low temperature, so label goes ABOVE and to the right of the trough
    ax.annotate('COLD\n(z = 0)',
                xy=(coords[0], mean_t[0]),
                xytext=(coords[0] + 30, mean_t[0] + 6),
                fontsize=9, color='blue', fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='blue', alpha=0.85))

    ax.annotate('HOT\n(z = Lz/2)',
                xy=(coords[len(coords)//2-1], mean_t[len(coords)//2-1]),
                xytext=(coords[len(coords)//2-1] + 30, mean_t[len(coords)//2-1] - 6),
                fontsize=9, color='red', fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.85))

    dT = mean_t.max() - mean_t.min()
    ax.text(0.02, 0.97,
            f'φ = 29.9% (S=4, d=3)\nκ = 1.39 ± 0.00 W/(m·K)\nΔT = {dT:.1f} K',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.3, linespacing=0.95,
            bbox=dict(boxstyle='round,pad=0.22',
                      fc='#FFFBEB', ec='#D69E2E',
                      alpha=0.78, lw=0.9))

ax.set_xlabel('z position (Å)', fontsize=12)
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title('(a) val1_S4d3 — ML-designed geometry (target κ = 1.5)',
             fontsize=11, fontweight='bold')
ax.legend(loc='lower right',
          fontsize=7.6,
          framealpha=0.78,
          borderpad=0.25,
          labelspacing=0.25,
          handlelength=1.35,
          handletextpad=0.45,
          markerscale=0.65,
          fancybox=True)
ax.grid(True, alpha=0.25, linestyle=':')

# ----------------------------------------------------------------
# Panel (b): geo3_rerun — validation at target κ = 3.0
# ----------------------------------------------------------------
ax = fig.add_subplot(gs[1])
# geo3_rerun uses seeds 91234 and 11111 (the 2 new seeds)
coords, mean_t, std_t = get_all_seeds("geo3_rerun", [91234, 11111])

if coords is not None:
    ax.plot(coords, mean_t, '-s', color=C_ORANGE, ms=12, lw=2.5,
            label='Mean (2 seeds)', zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(coords, mean_t - std_t, mean_t + std_t,
                    alpha=0.25, color=C_ORANGE, label='±1σ spread', zorder=3)
    ax.axhline(300, color='gray', linestyle='--', lw=1.2, alpha=0.6, label='Target (300 K)')

    ax.axvline(coords[0], color='blue', linestyle=':', lw=1.2, alpha=0.5)
    ax.axvline(coords[len(coords)//2], color='red', linestyle=':', lw=1.2, alpha=0.5)

    ax.annotate('COLD\n(z = 0)',
                xy=(coords[0], mean_t[0]),
                xytext=(coords[0] + 30, mean_t[0] + 4),
                fontsize=9, color='blue', fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='blue', alpha=0.85))

    ax.annotate('HOT\n(z = Lz/2)',
                xy=(coords[len(coords)//2-1], mean_t[len(coords)//2-1]),
                xytext=(coords[len(coords)//2-1] + 30, mean_t[len(coords)//2-1] - 4),
                fontsize=9, color='red', fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.85))

    dT = mean_t.max() - mean_t.min()
    ax.text(0.02, 0.97,
            f'φ = 17.5% (S=3, d=2)\nκ = 3.03 ± 0.14 W/(m·K)\nΔT = {dT:.1f} K',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.3, linespacing=0.95,
            bbox=dict(boxstyle='round,pad=0.22',
                      fc='#FFFBEB', ec='#D69E2E',
                      alpha=0.78, lw=0.9))

ax.set_xlabel('z position (Å)', fontsize=12)
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title('(b) geo3_rerun — reference geometry (target κ = 3.0)',
             fontsize=11, fontweight='bold')
ax.legend(loc='lower right',
          fontsize=7.6,
          framealpha=0.78,
          borderpad=0.25,
          labelspacing=0.25,
          handlelength=1.35,
          handletextpad=0.45,
          markerscale=0.65,
          fancybox=True)
ax.grid(True, alpha=0.25, linestyle=':')

# Overall title
fig.suptitle('Figure S1. NEMD Temperature Profiles for ML Validation Geometries',
             fontsize=13, fontweight='bold', y=1.01)

# Save
fig.tight_layout()
out = os.path.join(OUTDIR, "figS1_validation_profiles.png")
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"✅ Saved: {out}")
print("This figure shows the temperature profiles proving steady-state convergence")
print("for the two validation geometries.")
