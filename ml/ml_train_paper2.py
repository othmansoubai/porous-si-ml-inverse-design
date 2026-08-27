#!/usr/bin/env python3
"""
Paper 2 — ML Training Pipeline for Porous Si Inverse Design
=============================================================
Self-contained: all 18 verified NEMD data points hardcoded with
full geometric feature vectors.

Usage on MARWAN:
    module purge
    module load GCCcore/12.3.0 Python/3.11.3-GCCcore-12.3.0
    source ~/phase2_ml/analysis_env/bin/activate
    python3 ml_train_paper2.py
    deactivate

Outputs:
    ml_dataset_paper2.csv          — consolidated dataset
    ml_loocv_results.csv           — LOOCV predictions for each model
    ml_feature_importance.csv      — RF permutation importance
    ml_inverse_candidates.csv      — inverse design candidates
    fig_ml_results.png             — 6-panel results figure
    fig_inverse_design.png         — inverse design figure

Requirements: numpy, pandas, matplotlib, scikit-learn, scipy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.optimize import differential_evolution
import os, warnings
warnings.filterwarnings('ignore')

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# 1. DATASET — 18 geometries, all values verified from LAMMPS
# ================================================================
#
# Features:
#   phi       — actual porosity (%), computed as (24576 - N_remaining)/24576 * 100
#   S         — pore side length (unit cells, each u.c. = 5.431 Å)
#   stagger   — stagger offset as % of 8 u.c. box width (0 = aligned, 25 = quarter, 50 = full)
#   neck_uc   — minimum Si wall thickness between pore boundaries (unit cells)
#   AR        — pore aspect ratio (width / height in x-y cross-section)
#
# Target:
#   kappa     — thermal conductivity (W/m·K) from NEMD Müller-Plathe, mean of 3 seeds
#   kappa_std — seed-to-seed standard deviation
#
# Notes:
#   - Bulk Si encoded as S=0, stagger=0, neck=8 (full box), AR=1
#   - Group E excluded (degenerate thermal barriers, κ → 0)
#   - geo3 (A, S=3) flagged for high variance (kappa_std = 1.02)

DATA = [
    # Paper 1 reference points (Nz=48, 50% stagger)
    {"label": "bulk_si",             "group": "P1", "phi":  0.00, "S": 0.0, "stagger":  0.0, "neck_uc": 8.0, "AR": 1.0, "kappa": 12.03, "kappa_std": 0.29},
    # EXCLUDED (revision, R#3): phi=6.84% at S=3 is not reproducible by the
    # 12-slice parametric builder; z-architecture differs from all other rows.
    #{"label": "P1_low_50pct",        "group": "P1", "phi":  6.84, "S": 3.0, "stagger": 50.0, "neck_uc": 5.0, "AR": 1.0, "kappa":  7.63, "kappa_std": 0.99},
    {"label": "P1_high_50pct",       "group": "P1", "phi": 28.27, "S": 4.0, "stagger": 50.0, "neck_uc": 4.0, "AR": 1.0, "kappa":  2.59, "kappa_std": 0.02},

    # Group A — porosity sweep at 25% stagger
    {"label": "A_S2_d2",             "group": "A",  "phi":  8.54, "S": 2.0, "stagger": 25.0, "neck_uc": 4.0, "AR": 1.0, "kappa":  4.56, "kappa_std": 0.10},
    {"label": "A_S2.5_d2",           "group": "A",  "phi": 12.63, "S": 2.5, "stagger": 25.0, "neck_uc": 3.5, "AR": 1.0, "kappa":  4.02, "kappa_std": 0.22},
    {"label": "A_S3_d2",             "group": "A",  "phi": 17.50, "S": 3.0, "stagger": 25.0, "neck_uc": 3.0, "AR": 1.0, "kappa":  3.46, "kappa_std": 0.28},  # re-run, 3 fresh seeds, none excluded
    {"label": "A_S4.5_d2",           "group": "A",  "phi": 36.78, "S": 4.5, "stagger": 25.0, "neck_uc": 1.5, "AR": 1.0, "kappa":  1.28, "kappa_std": 0.02},

    # Group B — stagger offset sweep
    {"label": "B_S2_aligned_low",    "group": "B",  "phi":  8.01, "S": 2.0, "stagger":  0.0, "neck_uc": 6.0, "AR": 1.0, "kappa":  7.06, "kappa_std": 1.14},
    {"label": "B_S2_d4_low",         "group": "B",  "phi":  8.59, "S": 2.0, "stagger": 50.0, "neck_uc": 2.0, "AR": 1.0, "kappa":  4.39, "kappa_std": 0.14},
    {"label": "B_S4_aligned_high",   "group": "B",  "phi": 28.32, "S": 4.0, "stagger":  0.0, "neck_uc": 4.0, "AR": 1.0, "kappa":  3.74, "kappa_std": 0.20},
    {"label": "B_S4_quarter_high",   "group": "B",  "phi": 29.57, "S": 4.0, "stagger": 25.0, "neck_uc": 2.0, "AR": 1.0, "kappa":  1.82, "kappa_std": 0.04},

    # Group C — neck width sweep (S=3, φ≈17.5%)
    {"label": "C_d0_aligned",        "group": "C",  "phi": 17.50, "S": 3.0, "stagger":  0.0, "neck_uc": 5.0, "AR": 1.0, "kappa":  4.68, "kappa_std": 0.64},
    {"label": "C_d1_12pct",          "group": "C",  "phi": 17.50, "S": 3.0, "stagger": 12.5, "neck_uc": 4.0, "AR": 1.0, "kappa":  3.83, "kappa_std": 0.57},
    {"label": "C_d3_37pct",          "group": "C",  "phi": 17.50, "S": 3.0, "stagger": 37.5, "neck_uc": 2.0, "AR": 1.0, "kappa":  2.85, "kappa_std": 0.35},
    {"label": "C_d4_50pct",          "group": "C",  "phi": 17.50, "S": 3.0, "stagger": 50.0, "neck_uc": 2.0, "AR": 1.0, "kappa":  2.91, "kappa_std": 0.16},

    # Group D — aspect ratio sweep (S=3, d=2, φ≈17.5%)
    {"label": "D_AR0.5_tall",        "group": "D",  "phi": 17.50, "S": 3.0, "stagger": 25.0, "neck_uc": 3.0, "AR": 0.5, "kappa":  3.53, "kappa_std": 0.10},
    {"label": "D_AR2.0_wide",        "group": "D",  "phi": 17.50, "S": 3.0, "stagger": 25.0, "neck_uc": 3.0, "AR": 2.0, "kappa":  3.22, "kappa_std": 0.18},
    {"label": "D_AR2.5_vwide",       "group": "D",  "phi": 17.50, "S": 3.0, "stagger": 25.0, "neck_uc": 3.0, "AR": 2.5, "kappa":  2.69, "kappa_std": 0.10},

    # Validation — ML-suggested geometry
    {"label": "V_S4d3_37pct",        "group": "V",  "phi": 29.93, "S": 4.0, "stagger": 37.5, "neck_uc": 1.0, "AR": 1.0, "kappa":  1.39, "kappa_std": 0.01},
]

FEAT_COLS = ["phi", "S", "stagger", "neck_uc", "AR"]


def load_data():
    df = pd.DataFrame(DATA)
    print(f"Dataset: {len(df)} geometries, {len(FEAT_COLS)} features")
    print(f"  κ range: {df['kappa'].min():.2f} – {df['kappa'].max():.2f} W/(m·K)")
    for c in FEAT_COLS:
        print(f"  {c:>10s}: [{df[c].min():.1f}, {df[c].max():.1f}]")
    return df


# ================================================================
# 2. MODEL TRAINING WITH LOOCV
# ================================================================

def loocv_rf(X, y, use_log=False, n_est=500):
    """Random Forest with leave-one-out CV."""
    loo = LeaveOneOut()
    yt = np.log(y) if use_log else y.copy()
    y_pred = np.zeros(len(y))

    for tr, te in loo.split(X):
        m = RandomForestRegressor(n_estimators=n_est, random_state=42,
                                  max_features='sqrt', min_samples_leaf=2)
        m.fit(X[tr], yt[tr])
        p = m.predict(X[te])
        y_pred[te] = np.exp(p) if use_log else p

    # full model for importance
    rf = RandomForestRegressor(n_estimators=n_est, random_state=42,
                               max_features='sqrt', min_samples_leaf=2)
    rf.fit(X, yt)
    perm = permutation_importance(rf, X, yt, n_repeats=50, random_state=42)
    return y_pred, rf, perm


def loocv_gp(X, y, use_log=False):
    """Gaussian Process with leave-one-out CV."""
    yt = np.log(y) if use_log else y.copy()

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(X.shape[1]),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1)))

    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    y_std  = np.zeros(len(y))

    for tr, te in loo.split(X):
        # refit standardisation inside each fold (no information from held-out point)
        sc_f = StandardScaler().fit(X[tr])
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                       random_state=42, normalize_y=True)
        gp.fit(sc_f.transform(X[tr]), yt[tr])
        p, s = gp.predict(sc_f.transform(X[te]), return_std=True)
        if use_log:
            y_pred[te] = np.exp(p + s**2 / 2)
            y_std[te]  = y_pred[te] * np.sqrt(np.exp(s**2) - 1)
        else:
            y_pred[te] = p
            y_std[te]  = s

    # full model
    scaler  = StandardScaler().fit(X)
    gp_full = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                        random_state=42, normalize_y=True)
    gp_full.fit(scaler.transform(X), yt)
    return y_pred, y_std, gp_full, scaler


def report(name, y, yp):
    mae  = mean_absolute_error(y, yp)
    r2   = r2_score(y, yp)
    rmse = np.sqrt(mean_squared_error(y, yp))
    print(f"  {name:<25s}  MAE={mae:.3f}  R²={r2:.3f}  RMSE={rmse:.3f}")
    return mae, r2, rmse


# ================================================================
# 3. INVERSE DESIGN
# ================================================================

def inverse_design(gp, scaler, target_kappa, use_log=False):
    """Find geometry that produces target κ using GP surrogate."""
    target_val = np.log(target_kappa) if use_log else target_kappa

    bounds = [
        (5.0, 40.0),   # phi
        (2.0, 5.0),    # S
        (0.0, 50.0),   # stagger
        (1.5, 5.0),    # neck_uc
        (0.5, 3.0),    # AR
    ]

    def objective(x):
        xs = scaler.transform(x.reshape(1, -1))
        p, s = gp.predict(xs, return_std=True)
        return (p[0] - target_val)**2 + 0.3 * s[0]**2

    candidates = []
    for seed in range(20):
        res = differential_evolution(objective, bounds, seed=seed,
                                      maxiter=500, tol=1e-8, polish=True)
        x = res.x
        xs = scaler.transform(x.reshape(1, -1))
        p, s = gp.predict(xs, return_std=True)
        if use_log:
            kp = np.exp(p[0] + s[0]**2 / 2)
            ks = kp * np.sqrt(np.exp(s[0]**2) - 1)
        else:
            kp, ks = p[0], s[0]
        candidates.append(dict(phi=x[0], S=x[1], stagger=x[2],
                                neck_uc=x[3], AR=x[4],
                                kappa_pred=kp, kappa_std=ks, loss=res.fun))

    candidates.sort(key=lambda c: c['loss'])
    # deduplicate
    unique = [candidates[0]]
    for c in candidates[1:]:
        if not any(all(abs(c[f]-u[f]) / max(abs(u[f]), 0.01) < 0.1
                       for f in FEAT_COLS) for u in unique):
            unique.append(c)
        if len(unique) >= 3:
            break
    return unique


def candidates_to_lammps(candidates, target_kappa):
    """Convert ML candidates to LAMMPS-ready pore coordinates."""
    lines = []
    lines.append(f"\n  Target κ = {target_kappa:.1f} W/(m·K):")
    for i, c in enumerate(candidates):
        S  = round(c['S'])
        d  = round(c['stagger'] / 100 * 8)
        # handle AR
        if c['AR'] > 1.3:
            W = min(round(S * c['AR']), 7)
            H = max(round(S / c['AR']), 1)
        elif c['AR'] < 0.7:
            W = S
            H = min(round(S / c['AR']), 7)
        else:
            W, H = S, S

        neck_est = max(8 - max(W, H) - d, 0)
        lines.append(f"    Candidate {i+1}: κ_pred = {c['kappa_pred']:.2f} ± {c['kappa_std']:.2f} W/(m·K)")
        lines.append(f"      φ≈{c['phi']:.0f}%, S≈{S}, d≈{d} u.c., AR≈{c['AR']:.1f}")
        lines.append(f"      -var pAxlo 0 -var pAxhi {H} -var pAylo 0 -var pAyhi {W}")
        lines.append(f"      -var pBxlo {d} -var pBxhi {d+H} -var pBylo {d} -var pByhi {d+W}")
        lines.append(f"      Expected neck: ~{neck_est} u.c. ({neck_est*5.431:.1f} Å)")
    return "\n".join(lines)


# ================================================================
# 4. FIGURES
# ================================================================

def make_figures(df, y, y_err, results, perm, feat_names, inv_cands, use_log):
    """Generate publication figures."""

    GRP_COLORS = {"P1": "#E8706A", "A": "#5BA3CF", "B": "#3B9B6D",
                  "C": "#E8A048", "D": "#9B6DB0", "V": "#FF6B9D"}

    # ---- Figure 1: 6-panel ML results ----
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)

    # (a) RF predicted vs actual
    ax = fig.add_subplot(gs[0, 0])
    yp = results['rf']['pred']
    for g, col in GRP_COLORS.items():
        m = df["group"] == g
        ax.errorbar(y[m], yp[m], xerr=y_err[m], fmt='o', color=col, ms=9,
                     capsize=3, label=f"Group {g}", mec='white', mew=0.5)
    lim = [0, max(y.max(), yp.max()) + 1]
    ax.plot(lim, lim, '--', color='gray', lw=1.5, alpha=0.6)
    ax.set_xlabel("Actual κ (W/m·K)"); ax.set_ylabel("Predicted κ (W/m·K)")
    r2_rf = results['rf']['r2']
    mae_rf = results['rf']['mae']
    ax.set_title(f"(a) Random Forest LOOCV\nMAE = {mae_rf:.2f}, R² = {r2_rf:.3f}",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(lim); ax.set_ylim(lim)

    # (b) GP predicted vs actual with uncertainty
    ax = fig.add_subplot(gs[0, 1])
    yp = results['gp']['pred']
    ys = results['gp']['std']
    for g, col in GRP_COLORS.items():
        m = df["group"] == g
        ax.errorbar(y[m], yp[m], xerr=y_err[m], yerr=ys[m], fmt='s', color=col,
                     ms=9, capsize=3, label=f"Group {g}", mec='white', mew=0.5)
    ax.plot(lim, lim, '--', color='gray', lw=1.5, alpha=0.6)
    ax.set_xlabel("Actual κ (W/m·K)"); ax.set_ylabel("Predicted κ (W/m·K)")
    r2_gp = results['gp']['r2']
    mae_gp = results['gp']['mae']
    ax.set_title(f"(b) Gaussian Process LOOCV\nMAE = {mae_gp:.2f}, R² = {r2_gp:.3f}",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(lim); ax.set_ylim(lim)

    # (c) Feature importance
    ax = fig.add_subplot(gs[0, 2])
    si = perm.importances_mean.argsort()
    imp = perm.importances_mean[si]
    imp_s = perm.importances_std[si]
    _PRETTY = {"phi":"$\\varphi$","S":"S","stagger":"stagger","neck_uc":"neck width","AR":"AR"}
    names = [_PRETTY.get(feat_names[i], feat_names[i]) for i in si]
    cols = ['#5BA3CF'] * len(names)
    cols[-1] = '#E8706A'
    ax.barh(names, imp, xerr=imp_s, color=cols, edgecolor='white', capsize=3)
    ax.set_xlabel("Permutation Importance")
    log_label = "log(κ)" if use_log else "κ"
    ax.set_title(f"(c) Feature Importance (target: {log_label})",
                 fontsize=10, fontweight='bold')

    # (d) κ vs porosity colored by stagger
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(df["phi"], y, c=df["stagger"], cmap='viridis', s=80,
                     edgecolors='white', linewidth=0.5, zorder=5)
    ax.errorbar(df["phi"], y, yerr=y_err, fmt='none', ecolor='gray', capsize=2, alpha=0.4)
    plt.colorbar(sc, ax=ax, label='Stagger offset (%)')
    ax.set_xlabel("Porosity φ (%)"); ax.set_ylabel("κ (W/m·K)")
    ax.set_title("(d) κ vs Porosity", fontsize=10, fontweight='bold')

    # (e) κ vs neck width colored by porosity
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(df["neck_uc"], y, c=df["phi"], cmap='plasma', s=80,
                     edgecolors='white', linewidth=0.5, zorder=5)
    ax.errorbar(df["neck_uc"], y, yerr=y_err, fmt='none', ecolor='gray', capsize=2, alpha=0.4)
    plt.colorbar(sc, ax=ax, label='Porosity φ (%)')
    ax.set_xlabel("Neck width (unit cells)"); ax.set_ylabel("κ (W/m·K)")
    ax.set_title("(e) κ vs Neck Width", fontsize=10, fontweight='bold')

    # (f) GP uncertainty coverage
    ax = fig.add_subplot(gs[1, 2])
    si2 = np.argsort(y)
    ys_sorted = y[si2]
    yp_sorted = results['gp']['pred'][si2]
    ystd_sorted = results['gp']['std'][si2]
    xp = range(len(y))
    ax.fill_between(xp, yp_sorted - 2*ystd_sorted, yp_sorted + 2*ystd_sorted,
                     alpha=0.15, color='#5BA3CF', label='±2σ')
    ax.fill_between(xp, yp_sorted - ystd_sorted, yp_sorted + ystd_sorted,
                     alpha=0.3, color='#5BA3CF', label='±1σ')
    ax.plot(xp, yp_sorted, '-', color='#5BA3CF', lw=2, label='GP mean')
    ax.scatter(xp, ys_sorted, color='#E8706A', s=50, zorder=5,
               edgecolors='white', linewidth=0.5, label='Actual')
    ax.set_xlabel("Geometry (sorted by κ)"); ax.set_ylabel("κ (W/m·K)")
    ax.set_title("(f) GP Prediction Intervals", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    path1 = os.path.join(OUTDIR, "fig_ml_results.png")
    fig.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path1.replace('.png','.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ {path1}")

    # ---- Figure 2: Inverse design ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) predictions vs targets
    ax = axes2[0]
    targs = sorted(inv_cands.keys())
    for i, t in enumerate(targs):
        cs = inv_cands[t]
        if cs:
            c = cs[0]
            ax.errorbar(t, c['kappa_pred'], yerr=c['kappa_std'], fmt='o', ms=12,
                        capsize=5, color='#3B9B6D', mec='white', mew=1, zorder=5)
            ax.text(t + 0.12, c['kappa_pred'] + 0.2,
                    f"φ={c['phi']:.0f}%\nneck={c['neck_uc']:.1f}",
                    fontsize=8, color='#444')
    tlim = [0, max(targs) + 1.5]
    ax.plot(tlim, tlim, '--', color='gray', lw=1.5, alpha=0.5)
    ax.set_xlabel("Target κ (W/m·K)"); ax.set_ylabel("Predicted κ (W/m·K)")
    ax.set_title("(a) Target vs Prediction", fontsize=11, fontweight='bold')
    ax.set_xlim(tlim); ax.set_ylim(tlim)

    # (b) feature values of candidates
    ax = axes2[1]
    bounds_norm = [(5, 40), (2, 5), (0, 50), (1.5, 5), (0.5, 3)]
    for i, t in enumerate(targs):
        cs = inv_cands[t]
        if cs:
            c = cs[0]
            vals = [(c[f] - lo) / (hi - lo) for f, (lo, hi) in zip(FEAT_COLS, bounds_norm)]
            ax.plot(range(len(FEAT_COLS)), vals, '-o', ms=8, lw=2, alpha=0.8,
                    label=f"κ={t:.1f} → {c['kappa_pred']:.2f}±{c['kappa_std']:.2f}")
    ax.set_xticks(range(len(FEAT_COLS)))
    ax.set_xticklabels([_PRETTY.get(f, f) for f in FEAT_COLS], fontsize=10)
    ax.set_ylabel("Normalized value"); ax.set_ylim(-0.1, 1.1)
    ax.set_title("(b) Candidate Feature Profiles", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    path2 = os.path.join(OUTDIR, "fig_inverse_design.png")
    fig2.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(path2.replace('.png','.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  ✅ {path2}")


# ================================================================
# 5. MAIN
# ================================================================

def main():
    print("=" * 70)
    print("  Paper 2 — ML Training Pipeline")
    print("=" * 70)

    df = load_data()
    X  = df[FEAT_COLS].values
    y  = df["kappa"].values
    ye = df["kappa_std"].values

    # Save consolidated dataset
    df.to_csv(os.path.join(OUTDIR, "ml_dataset_paper2.csv"), index=False)

    # ------------------------------------------------------------------
    # Compare raw vs log(κ) target for both RF and GP
    # ------------------------------------------------------------------
    print(f"\n{'Model':<12} {'Target':<8} {'MAE':>8} {'R²':>8} {'RMSE':>8}")
    print("-" * 50)

    all_results = {}
    best_r2, best_key = -np.inf, None

    for use_log in [False, True]:
        tag = "log" if use_log else "raw"

        # RF
        yp_rf, rf_model, perm_rf = loocv_rf(X, y, use_log=use_log)
        mae, r2, rmse = report(f"RF ({tag})", y, yp_rf)
        key = f"RF_{tag}"
        all_results[key] = dict(pred=yp_rf, mae=mae, r2=r2, rmse=rmse,
                                model=rf_model, perm=perm_rf, log=use_log)
        if r2 > best_r2:
            best_r2, best_key = r2, key

        # GP
        yp_gp, ys_gp, gp_model, gp_scaler = loocv_gp(X, y, use_log=use_log)
        mae, r2, rmse = report(f"GP ({tag})", y, yp_gp)
        key = f"GP_{tag}"
        all_results[key] = dict(pred=yp_gp, std=ys_gp, mae=mae, r2=r2, rmse=rmse,
                                model=gp_model, scaler=gp_scaler, log=use_log)
        if r2 > best_r2:
            best_r2, best_key = r2, key

    print(f"\n  ★ Best model: {best_key}  (R² = {best_r2:.3f})")
    best = all_results[best_key]
    use_log_best = best['log']

    # ------------------------------------------------------------------
    # Feature importance from best RF
    # ------------------------------------------------------------------
    rf_key = f"RF_{'log' if use_log_best else 'raw'}"
    perm = all_results[rf_key]['perm']
    print(f"\nFeature Importance (RF, {'log(κ)' if use_log_best else 'κ'}):")
    si = perm.importances_mean.argsort()[::-1]
    fi_rows = []
    for i in si:
        print(f"  {FEAT_COLS[i]:>10s}: {perm.importances_mean[i]:.4f} ± {perm.importances_std[i]:.4f}")
        fi_rows.append(dict(feature=FEAT_COLS[i],
                            importance=perm.importances_mean[i],
                            std=perm.importances_std[i]))
    pd.DataFrame(fi_rows).to_csv(os.path.join(OUTDIR, "ml_feature_importance.csv"), index=False)

    # GP length scales
    gp_key = f"GP_{'log' if use_log_best else 'raw'}"
    gp_m = all_results[gp_key]['model']
    k = gp_m.kernel_
    print(f"\nGP kernel: {k}")
    if hasattr(k, 'k1') and hasattr(k.k1, 'k2') and hasattr(k.k1.k2, 'length_scale'):
        ls = k.k1.k2.length_scale
        print("GP length scales (smaller = more important):")
        for name, l in zip(FEAT_COLS, ls):
            print(f"  {name:>10s}: {l:.3f}")

    # ------------------------------------------------------------------
    # LOOCV results CSV
    # ------------------------------------------------------------------
    loocv_df = df[["label", "group", "kappa", "kappa_std"]].copy()
    for key, res in all_results.items():
        loocv_df[f"pred_{key}"] = res['pred']
        if 'std' in res:
            loocv_df[f"std_{key}"] = res['std']
    loocv_df.to_csv(os.path.join(OUTDIR, "ml_loocv_results.csv"), index=False)

    # ------------------------------------------------------------------
    # Inverse design
    # ------------------------------------------------------------------
    gp_inv = all_results[gp_key]['model']
    sc_inv = all_results[gp_key]['scaler']

    targets = [1.5, 2.0, 3.0, 5.0]
    inv_cands = {}
    inv_rows = []

    print(f"\n{'='*70}")
    print("INVERSE DESIGN")
    print(f"{'='*70}")
    for t in targets:
        cands = inverse_design(gp_inv, sc_inv, t, use_log=use_log_best)
        inv_cands[t] = cands
        for i, c in enumerate(cands):
            print(f"  κ_target={t:.1f}  C{i+1}: φ={c['phi']:.1f}%, S={c['S']:.1f}, "
                  f"stagger={c['stagger']:.1f}%, neck={c['neck_uc']:.1f}, AR={c['AR']:.2f} "
                  f"→ {c['kappa_pred']:.2f}±{c['kappa_std']:.2f}")
            inv_rows.append(dict(target_kappa=t, candidate=i+1, **c))

    pd.DataFrame(inv_rows).to_csv(os.path.join(OUTDIR, "ml_inverse_candidates.csv"), index=False)

    # ------------------------------------------------------------------
    # LAMMPS specs
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LAMMPS GEOMETRY SPECS FOR VALIDATION RUNS")
    print("(use with in.nemd_porous_parametric)")
    print(f"{'='*70}")
    for t in targets:
        print(candidates_to_lammps(inv_cands[t], t))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures...")
    res_plot = {
        'rf': dict(pred=all_results[rf_key]['pred'],
                   r2=all_results[rf_key]['r2'],
                   mae=all_results[rf_key]['mae']),
        'gp': dict(pred=all_results[gp_key]['pred'],
                   std=all_results[gp_key].get('std', np.zeros(len(y))),
                   r2=all_results[gp_key]['r2'],
                   mae=all_results[gp_key]['mae']),
    }
    make_figures(df, y, ye, res_plot, perm, FEAT_COLS, inv_cands, use_log_best)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OUTPUT FILES")
    print(f"{'='*70}")
    for f in ["ml_dataset_paper2.csv", "ml_loocv_results.csv",
              "ml_feature_importance.csv", "ml_inverse_candidates.csv",
              "fig_ml_results.png", "fig_inverse_design.png"]:
        print(f"  {f}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("  1. Review fig_ml_results.png — check R² and feature importance")
    print("  2. Pick 2-3 inverse design candidates from ml_inverse_candidates.csv")
    print("  3. Run NEMD validation using the LAMMPS specs above")
    print("  4. Compare NEMD κ to ML prediction → Paper 2 validation")
    print("  5. If R² < 0.7, consider rerunning geo3 (A_S3) with extra seeds")
    print("  6. Set up almaBTE cross-validation for the best candidate")


if __name__ == "__main__":
    main()
