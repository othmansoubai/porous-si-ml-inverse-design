import os, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
OUT = os.path.dirname(os.path.abspath(__file__))

txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "ml_train_paper2.py")).read()
ns = {}; exec(re.search(r"^DATA\s*=\s*\[.*?^\]", txt, re.S | re.M).group(0), ns)
df = pd.DataFrame(ns["DATA"])
F  = ["phi", "S", "stagger", "neck_uc", "AR"]
LB = [r"$\varphi$", "S", "stagger", "neck", "AR"]

por = df[df.S > 0]
C   = por[F].corr().values
vif = np.diag(inv(C))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

im = ax[0].imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(LB)
ax[0].set_yticks(range(5)); ax[0].set_yticklabels(LB)
for i in range(5):
    for j in range(5):
        ax[0].text(j, i, f"{C[i,j]:.2f}", ha='center', va='center',
                   fontsize=8.5,
                   fontweight='bold' if abs(C[i,j])>0.9 and i!=j else 'normal',
                   color='white' if abs(C[i,j])>0.6 else 'black')
ax[0].add_patch(plt.Rectangle((-.5,.5),1,1, fill=False, ec='k', lw=2))
ax[0].add_patch(plt.Rectangle((.5,-.5),1,1, fill=False, ec='k', lw=2))
ax[0].set_title("(a) Descriptor correlation", fontweight='bold', fontsize=11)
fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04)

S_ = np.linspace(1.8, 4.8, 100)
A  = np.c_[por.S**2, por.S]
co, *_ = np.linalg.lstsq(A, por.phi.values, rcond=None)
ax[1].scatter(por.S, por.phi, s=55, c='#2E86AB', ec='k', lw=.5, zorder=3)
ax[1].plot(S_, co[0]*S_**2 + co[1]*S_, 'r-', lw=1.5,
           label=fr"$\varphi$ = {co[0]:.3f}$S^2$ + {co[1]:.3f}$S$")
ax[1].set_xlabel("Pore size S (u.c.)")
ax[1].set_ylabel(r"Porosity $\varphi$ (%)")
ax[1].set_title(fr"(b) Coupling of $\varphi$ and S (r = {C[0,1]:.3f})",
                fontweight='bold', fontsize=11)
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.25, ls=':')

y = np.log(df.kappa.values)
sets = [("all five", F), ("no neck", ["phi","S","stagger","AR"]),
        ("no S", ["phi","stagger","neck_uc","AR"])]
res = {}
for tag, fs in sets:
    rf = RandomForestRegressor(500, max_features='sqrt', min_samples_leaf=2,
                               random_state=42).fit(df[fs].values, y)
    pi = permutation_importance(rf, df[fs].values, y, n_repeats=50, random_state=42)
    res[tag] = dict(zip(fs, pi.importances_mean))
w = .26
for k, (tag, _) in enumerate(sets):
    vals = [res[tag].get(f, 0) for f in F]
    ax[2].bar(np.arange(5) + (k-1)*w, vals, w, label=tag,
              color=['#2E86AB','#8FBF6F','#E8A048'][k], ec='k', lw=.4)
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(LB)
ax[2].set_ylabel("Permutation importance")
ax[2].set_title("(c) Ranking under ablation", fontweight='bold', fontsize=11)
ax[2].legend(fontsize=8.5); ax[2].grid(alpha=.25, ls=':', axis='y')

fig.tight_layout()
fig.savefig(f"{OUT}/fig_collinearity.png", dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f"{OUT}/fig_collinearity.pdf", bbox_inches='tight', facecolor='white')
print("VIF:", {f: round(v,1) for f, v in zip(F, vif)})
print("saved fig_collinearity.png + .pdf")
