"""
Figure 4 - Constrained inverse design over the buildable grid.

(a) GP-predicted kappa across the (S, d) design grid, with sampled cells
    marked and the three selected candidates highlighted.
(b) Predicted vs NEMD-measured kappa for the three validated candidates.

Reads DATA from ml_train_paper2.py so it cannot drift from Table 1.
"""

import os, re, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- load the corrected dataset ------------------------------------
txt = open(os.path.join(OUT, "..", "ml", "ml_train_paper2.py")).read()
ns = {}
exec(re.search(r"^DATA\s*=\s*\[.*?^\]", txt, re.S | re.M).group(0), ns)
df = pd.DataFrame(ns["DATA"])
F = ["phi", "S", "stagger", "neck_uc", "AR"]

# ---- porosity calibration phi(S) -----------------------------------
por = df[df.S > 0]
A = np.c_[por.S**2, por.S]
coef, *_ = np.linalg.lstsq(A, por.phi.values, rcond=None)
phi_of = lambda S: coef[0]*S**2 + coef[1]*S
print(f"phi(S) = {coef[0]:.3f} S^2 + {coef[1]:.3f} S")

# ---- fit the GP on all data ----------------------------------------
X, y = df[F].values, df["kappa"].values
sc = StandardScaler().fit(X)
kern = (ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(np.ones(5), (1e-2, 1e2), nu=2.5)
        + WhiteKernel(0.1, (1e-4, 1e1)))
gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=10,
                              random_state=42, normalize_y=True)
gp.fit(sc.transform(X), np.log(y))

# ---- evaluate the buildable grid -----------------------------------
S_vals = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
d_vals = [0, 1, 2, 3, 4]
grid = np.full((len(d_vals), len(S_vals)), np.nan)

for i, d in enumerate(d_vals):
    for j, S in enumerate(S_vals):
        neck = 8 - S - d
        if neck < 1.0:
            continue
        x = [[phi_of(S), S, d*12.5, neck, 1.0]]
        mu, sd = gp.predict(sc.transform(x), return_std=True)
        grid[i, j] = float(np.exp(mu[0] + sd[0]**2 / 2))

# cells already present in the training set
sampled = {(round(r.S, 2), int(round(r.stagger / 12.5)))
           for r in df.itertuples() if r.S > 0}

# the three validated candidates
cands = [
    dict(S=4.0, d=3, tgt=1.5, gp=1.341, gps=0.125, nemd=1.390, ns=0.010),
    dict(S=3.5, d=3, tgt=2.0, gp=2.116, gps=0.131, nemd=2.091, ns=0.048),
    dict(S=2.5, d=1, tgt=5.0, gp=4.728, gps=0.292, nemd=5.079, ns=0.483),
]

# ====================================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6),
                       gridspec_kw={'width_ratios': [1.35, 1]})

# ---- (a) design-space map ------------------------------------------
a = ax[0]
im = a.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
              extent=[-0.5, len(S_vals)-0.5, -0.5, len(d_vals)-0.5])

for i, d in enumerate(d_vals):
    for j, S in enumerate(S_vals):
        if np.isnan(grid[i, j]):
            a.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                      facecolor='#dddddd', edgecolor='white'))
            continue
        if (round(S, 2), d) in sampled:
            a.plot(j, i, 'x', color='white', ms=9, mew=2.2)
        a.text(j, i-0.34, f"{grid[i,j]:.1f}", ha='center', va='center',
               fontsize=7, color='white', alpha=0.85)

for c in cands:
    j = S_vals.index(c['S']); i = d_vals.index(c['d'])
    prior = (round(c['S'],2), c['d']) in sampled
    col = '#F5A623' if prior else 'crimson'
    a.plot(j, i, 'o', mfc='none', mec=col, ms=20, mew=2.6)
    a.text(j, i+0.36, f"$\\kappa$ = {c['tgt']:.1f}",
           ha='center', va='center', fontsize=8,
           color=col, fontweight='bold')

a.set_xticks(range(len(S_vals)))
a.set_xticklabels([f"{s}" for s in S_vals])
a.set_yticks(range(len(d_vals)))
a.set_yticklabels([f"{d}" for d in d_vals])
a.set_xlabel("Pore size $S$ (u.c.)")
a.set_ylabel("Stagger offset $d$ (u.c.)")
a.set_title("(a) GP-predicted $\\kappa$ over the buildable grid",
            fontweight='bold', fontsize=11)
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
cb.set_label("$\\kappa$ (W m$^{-1}$K$^{-1}$)")

a.plot([], [], 'x', color='k', ms=8, mew=2, label='already sampled')
a.plot([], [], 'o', mfc='none', mec='crimson', ms=11, mew=2,
       label='new candidate (unsampled)')
a.plot([], [], 'o', mfc='none', mec='#F5A623', ms=11, mew=2,
       label='validated in original submission')
a.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor='#dddddd',
                          label='neck $<$ 1 u.c.'))
a.set_xlim(-0.5, len(S_vals)-0.5); a.set_ylim(-0.5, len(d_vals)-0.5)
a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)

# ---- (b) predicted vs measured -------------------------------------
b = ax[1]
lim = [0, 6.4]
b.plot(lim, lim, '--', color='gray', lw=1)
for c in cands:
    b.errorbar(c['gp'], c['nemd'], xerr=c['gps'], yerr=c['ns'],
               fmt='o', ms=9, capsize=4, color='#2E86AB',
               ecolor='#2E86AB', mec='k', mew=0.6)
    b.annotate(f"$\\kappa_{{\\rm target}}$ = {c['tgt']}",
               (c['gp'], c['nemd']), textcoords='offset points',
               xytext=(9, -13), fontsize=8.5)
b.set_xlim(lim); b.set_ylim(lim)
b.set_xlabel("GP prediction (W m$^{-1}$K$^{-1}$)")
b.set_ylabel("NEMD (W m$^{-1}$K$^{-1}$)")
b.set_title("(b) Out-of-sample validation", fontweight='bold', fontsize=11)
b.grid(alpha=0.25, ls=':')

fig.tight_layout()
fig.savefig(f"{OUT}/fig_inverse_design.png", dpi=300,
            bbox_inches='tight', facecolor='white')
fig.savefig(f"{OUT}/fig_inverse_design.pdf",
            bbox_inches='tight', facecolor='white')
print("saved fig_inverse_design.png + .pdf")
print(f"sampled cells: {len(sampled)}  |  grid cells evaluated: "
      f"{int(np.isfinite(grid).sum())}")
