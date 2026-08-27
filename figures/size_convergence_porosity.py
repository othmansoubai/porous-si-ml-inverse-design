"""
Figure 7 - Porosity dependence of the finite-size correction.

(a) kappa versus cell length for three porosities.
(b) Schelling-Phillpot-Keblinski extrapolation, 1/kappa versus 1/Lz.
(c) Fractional correction from Nz=48 to the extrapolated bulk limit,
    plotted against porosity.

Nz=48 values are the dataset entries; Nz=72 and 96 are the additional
convergence runs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
OUT = os.path.dirname(os.path.abspath(__file__))
A0 = 5.431  # lattice constant, Angstrom

# --------------------------------------------------------------------
# structure : porosity, and (Nz -> mean, std, n)
# --------------------------------------------------------------------
SETS = [
    # excluded: only 1 seed at Nz=96, non-monotonic, extrapolation unconstrained
    # dict(tag="S=2, d=2",   phi=8.54,  color="#2E86AB", marker="o",
    #     data={48: (4.56, 0.10, 3), 72: (6.767, 0.314, 2), 96: (6.569, 0.000, 1)}),
    dict(tag="S=3, d=2",   phi=17.50, color="#8FBF6F", marker="s",
         data={48: (3.463, 0.275, 3), 72: (3.930, 0.244, 3), 96: (4.353, 0.078, 3)}),
    dict(tag="S=4.5, d=2", phi=36.78, color="#C44E52", marker="^",
         data={48: (1.28, 0.02, 3), 72: (1.357, 0.006, 2), 96: (1.418, 0.025, 2)}),
]

print(f"{'structure':<12}{'phi':>7}{'k48':>8}{'k96':>8}{'k_inf':>9}"
      f"{'Lambda':>9}{'R2':>7}{'corr':>8}")
print("-" * 68)

for s in SETS:
    Nz = np.array(sorted(s["data"]))
    L  = Nz * A0 / 10.0                      # nm
    k  = np.array([s["data"][n][0] for n in Nz])
    e  = np.array([s["data"][n][1] for n in Nz])
    s.update(L=L, k=k, e=e)

    x, y = 1 / L, 1 / k
    M = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(M, y, rcond=None)
    r2 = 1 - ((y - (m * x + b)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    if b > 0:
        kinf, lam = 1 / b, m / b
        corr = (kinf - k[0]) / kinf * 100
    else:
        kinf = lam = corr = np.nan

    s.update(m=m, b=b, kinf=kinf, lam=lam, r2=r2, corr=corr)
    print(f"{s['tag']:<12}{s['phi']:>7.1f}{k[0]:>8.2f}{k[-1]:>8.2f}"
          f"{kinf:>9.2f}{lam:>9.1f}{r2:>7.3f}{corr:>7.0f}%")

# --------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))

# (a) kappa vs Lz
for s in SETS:
    ax[0].errorbar(s["L"], s["k"], yerr=s["e"], fmt=s["marker"] + "-",
                   color=s["color"], ms=6, capsize=4, lw=1.4,
                   label=fr"{s['tag']}  ($\varphi$={s['phi']:.1f}%)")
ax[0].set_xlabel("Cell length $L_z$ (nm)")
ax[0].set_ylabel(r"$\kappa$ (W m$^{-1}$K$^{-1}$)")
ax[0].set_title("(a) Size dependence", fontweight="bold", fontsize=11)
ax[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
ax[0].grid(alpha=.25, ls=":")
ax[0].set_ylim(bottom=0)

# (b) SPK plot
for s in SETS:
    x, y = 1 / s["L"], 1 / s["k"]
    xf = np.linspace(0, x.max() * 1.15, 40)
    ax[1].errorbar(x, y, yerr=s["e"] / s["k"] ** 2, fmt=s["marker"],
                   color=s["color"], ms=6, capsize=3)
    ax[1].plot(xf, s["m"] * xf + s["b"], "-", color=s["color"], lw=1.3)
    ax[1].plot(0, s["b"], "*", color=s["color"], ms=13)
ax[1].set_xlabel(r"$1/L_z$ (nm$^{-1}$)")
ax[1].set_ylabel(r"$1/\kappa$ (m K W$^{-1}$)")
ax[1].set_title("(b) SPK extrapolation", fontweight="bold", fontsize=11)
ax[1].set_xlim(left=0)
ax[1].grid(alpha=.25, ls=":")


fig.tight_layout()
fig.savefig(f"{OUT}/fig_size_convergence.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_size_convergence.pdf",
            bbox_inches="tight", facecolor="white")
print("\nsaved fig_size_convergence.png + .pdf")
