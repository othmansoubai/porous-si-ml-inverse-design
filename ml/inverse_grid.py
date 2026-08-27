import re
import numpy as np, itertools, importlib.util
spec = importlib.util.spec_from_file_location("m", "ml_train_paper2.py")
m = importlib.util.module_from_spec(spec)
import sys; sys.modules["m"] = m
txt = open("ml_train_paper2.py").read()
blk = re.search(r"^DATA\s*=\s*\[.*?^\]", txt, re.S | re.M).group(0)
ns = {}; exec(blk, ns); DATA = ns["DATA"]

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

df = pd.DataFrame(DATA)
F = ["phi","S","stagger","neck_uc","AR"]
X, y = df[F].values, df["kappa"].values

# calibrate phi(S) from measured porous rows
p = df[df.S > 0]
A = np.c_[p.S**2, p.S]
coef, *_ = np.linalg.lstsq(A, p.phi.values, rcond=None)
phi_of = lambda S: coef[0]*S**2 + coef[1]*S
print(f"phi(S) = {coef[0]:.4f}*S^2 + {coef[1]:.4f}*S")
for S in [2,2.5,3,4,4.5]: print(f"   S={S}: pred {phi_of(S):.2f}")

sc = StandardScaler(); Xs = sc.fit_transform(X)
k = (ConstantKernel(1.,(1e-3,1e3))*Matern(np.ones(5),(1e-2,1e2),nu=2.5)
     + WhiteKernel(0.1,(1e-4,1e1)))
gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10,
                              random_state=42, normalize_y=True)
gp.fit(Xs, np.log(y))

sampled = {(round(r.S,2), round(r.stagger/12.5), round(r.AR,2))
           for r in df.itertuples() if r.S > 0}

rows = []
for S, d, AR in itertools.product([2,2.5,3,3.5,4,4.5],[0,1,2,3,4],[1.0]):
    neck = 8 - S - d
    if neck < 1.0: continue
    phi = phi_of(S)
    if not (5 <= phi <= 40): continue
    new = (round(S,2), d, round(AR,2)) not in sampled
    mu, sd = gp.predict(sc.transform([[phi,S,d*12.5,neck,AR]]), return_std=True)
    rows.append(dict(S=S, d=d, AR=AR, phi=round(phi,2), neck=neck,
                     kappa=round(float(np.exp(mu[0]+sd[0]**2/2)),3),
                     std=round(float(np.exp(mu[0])*np.sqrt(np.exp(sd[0]**2)-1)),3),
                     NEW=new))
g = pd.DataFrame(rows)
g.to_csv("grid_predictions.csv", index=False)

for tgt in [2.0, 5.0]:
    c = g[g.NEW].copy(); c["err"] = (c.kappa-tgt).abs()
    print(f"\n=== target {tgt} — unsampled only ===")
    print(c.nsmallest(4,"err").to_string(index=False))
    b = c.nsmallest(1,"err").iloc[0]
    lo = int(b.d)
    print(f"  -var pAxlo 0 -var pAxhi {b.S} -var pAylo 0 -var pAyhi {b.S}")
    print(f"  -var pBxlo {lo} -var pBxhi {b.S+lo} -var pBylo {lo} -var pByhi {b.S+lo}")
