import numpy as np
from analyze_kappa import compute_kappa

runs = ["R1_S3.5_d3","R2_S2.5_d1","SZ_Nz48","SZ_Nz72","SZ_Nz96","SZhi_S4.5_Nz72","SZhi_S4.5_Nz96","SZlo_S2_Nz72","SZlo_S2_Nz96"]
print(f"{'label':<16}{'seed':>8}{'kappa':>10}")
print("-"*36)
summary={}
for lab in runs:
    vals=[]
    for s in (12345,34567,56789,78901,91234,11111):
        try:
            r = compute_kappa(lab, s)
            k = r["kappa"] if isinstance(r,dict) else (r[0] if isinstance(r,(tuple,list)) else r)
            vals.append(float(k)); print(f"{lab:<16}{s:>8}{float(k):>10.3f}")
        except Exception as e:
            print(f"{lab:<16}{s:>8}   -- ({type(e).__name__})")
    if vals: summary[lab]=(np.mean(vals), np.std(vals), len(vals))

print("\n=== SUMMARY ===")
for k,(m,s,n) in summary.items():
    print(f"  {k:<14} {m:6.3f} +/- {s:5.3f}  (n={n})")
