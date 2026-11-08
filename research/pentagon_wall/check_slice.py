import json, sys
from fractions import Fraction as F
import flint, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = sys.argv[1]
sheets = {}
for name in ("012", "12"):
    sheets[name] = {tuple(map(int, k.split(","))): int(v) for k, v in json.load(open(f"{S}/Phi_{name}_s10.json")).items()}
def real_roots(coef, tv):
    co = [sum(F(c) * tv**i for (i, jj), c in coef.items() if jj == j) for j in range(9)]
    p = flint.fmpq_poly([flint.fmpq(c.numerator, c.denominator) for c in co]).numer()
    return sorted(float(r.real.mid()) for r, _ in p.complex_roots() if abs(float(r.imag.mid())) < 1e-12)
cache = json.load(open("notebooks/data/cp2_2_thresholds.json"))
print(" t      cached m*     sheet  |root - m*|   (other sheet)")
pts = []
for k in range(1, 18):
    tv = F(k, 20)
    key = f"1/10|{tv}" if tv >= F(1, 10) else f"{tv}|1/10"
    rec = cache.get(key)
    c0, c1, c2 = rec["line"]
    if tv < F(1, 10): c1, c2 = c2, c1
    Vn = [(0, 0), (0.9, 0), (0.9, 0.1), (float(tv), 1 - float(tv)), (0, 1 - float(tv))]
    pos = {i for i, (x, y) in enumerate(Vn) if c0 + c1 * x + c2 * y > 1e-9}
    ch = "012" if pos in ({0, 1, 2}, {3, 4}) else "12" if pos in ({1, 2}, {0, 3, 4}) else str(pos)
    d = {n: min(abs(r - rec["m_star"]) for r in real_roots(cf, tv)) for n, cf in sheets.items()}
    other = "12" if ch == "012" else "012"
    pts.append((float(tv), rec["m_star"], ch))
    print(f"{float(tv):.2f}  {rec['m_star']:12.8f}   {ch:4s}   {d.get(ch, float('nan')):.1e}      ({d[other]:.1e})")
fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
cols = {"012": "#c0392b", "12": "#2471a3"}
for name, cf in sheets.items():
    for tv in np.linspace(0.003, 0.897, 450):
        rr = [r for r in real_roots(cf, F(tv).limit_denominator(10**4)) if 0 < r < 35]
        ax.plot([tv] * len(rr), rr, ".", ms=1.2, color=cols[name], alpha=0.55)
for tv, ms, ch in pts:
    ax.plot(tv, ms, "o", mfc="none", mec="k", ms=7, mew=1.1)
ax.plot([], [], ".", color=cols["012"], label="Φ₀₁₂ = 0 (crease has (0,0) on the side of the cut corner)")
ax.plot([], [], ".", color=cols["12"], label="Φ₁₂ = 0")
ax.plot([], [], "o", mfc="none", mec="k", label="m*(1/10, t) from the exact root-finder")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, 35); ax.set_xlim(0, 0.9); ax.set_xlabel("t"); ax.set_ylabel("m")
ax.set_title("s = 1/10: real points of the two wall sheets", fontsize=10)
fig.savefig(f"{S}/phi_s10.png", dpi=130)
