import re, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

txt = open("ml_train_paper2.py").read()
DATA = {}
exec(re.search(r"^DATA\s*=\s*\[.*?^\]", txt, re.S|re.M).group(0), DATA)
df = pd.DataFrame(DATA["DATA"])
F = ["phi","S","stagger","neck_uc","AR"]

def loocv(X, y, log=True):
    yt = np.log(y) if log else y
    pr = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        k = (ConstantKernel(1.,(1e-3,1e3))*Matern(np.ones(X.shape[1]),(1e-2,1e2),nu=2.5)
             + WhiteKernel(0.1,(1e-4,1e1)))
        g = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10,
                                     random_state=42, normalize_y=True).fit(sc.transform(X[tr]), yt[tr])
        m, s = g.predict(sc.transform(X[te]), return_std=True)
        pr[te] = np.exp(m + s**2/2) if log else m
    return mean_absolute_error(y, pr), r2_score(y, pr)

print("=== R#3: surrogate evaluation vs inverse-design validation ===")
for name, sub in [("pre-Candidate-1 (n=17)", df[df.group!="V"]), ("with Candidate 1 (n=18)", df)]:
    mae, r2 = loocv(sub[F].values, sub["kappa"].values)
    print(f"  {name:<26s} n={len(sub)}  MAE={mae:.3f}  R2={r2:.3f}")

print("\n=== R#4: descriptor correlation (Pearson) ===")
p = df[df.S > 0]
print(p[F].corr().round(3).to_string())

print("\n=== R#4: VIF ===")
from numpy.linalg import inv
C = p[F].corr().values
for f, v in zip(F, np.diag(inv(C))):
    print(f"  {f:>8s}: {v:6.2f}")

print("\n=== R#4: RF importance, full vs decorrelated subsets ===")
for tag, feats in [("all 5", F), ("drop neck", ["phi","S","stagger","AR"]),
                   ("drop S", ["phi","stagger","neck_uc","AR"])]:
    X, y = df[feats].values, np.log(df["kappa"].values)
    rf = RandomForestRegressor(500, max_features='sqrt', min_samples_leaf=2,
                               random_state=42).fit(X, y)
    imp = permutation_importance(rf, X, y, n_repeats=50, random_state=42)
    order = np.argsort(-imp.importances_mean)
    print(f"  [{tag}] " + ", ".join(f"{feats[i]}={imp.importances_mean[i]:.3f}" for i in order))
