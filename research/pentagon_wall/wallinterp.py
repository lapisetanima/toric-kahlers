"""Interpolate Phi(t, m) = sum_{j<=8} p_j(t) m^j from monic samples: p_j(t_k) - p_8(t_k) a_{j,k} = 0.

usage: wallinterp.py out_prefix sample1.json [sample2.json ...]
"""
import json, math, sys, time
from fractions import Fraction as F
import flint

T = time.time()
log = lambda *a: print(*a, f"[{time.time() - T:.1f}s]", flush=True)
prefix, files = sys.argv[1], sys.argv[2:]
samples = [r for f in files for r in json.load(open(f)) if "monic" in r]
pts = [(F(r["t"]), [F(a) for a in r["monic"]]) for r in samples]
log(len(pts), "samples")
DM = 8


def system(d, pts, mod=None):
    rows = []
    for tk, a in pts:
        pw = [tk**i for i in range(d + 1)]
        for j in range(DM):
            row = [F(0)] * ((DM + 1) * (d + 1))
            for i in range(d + 1):
                row[j * (d + 1) + i] = pw[i]
                row[DM * (d + 1) + i] = -pw[i] * a[j]
            rows.append(row)
    return rows


def nullity_mod(d, pts, p=2**61 - 1):
    rows = system(d, pts)
    M = flint.nmod_mat([[ (x.numerator % p) * pow(x.denominator % p, -1, p) % p for x in r] for r in rows], p)
    return M.ncols() - M.rank()


# degree search mod p: nullity = d - d_true + 1 when enough equations
n = len(pts)
dmax = (DM * n) // (DM + 1) - 2
lo = None
for d in (20, 40, 60, 80, 100, 120, 140, 160, dmax):
    if d > dmax:
        continue
    k = nullity_mod(d, pts)
    log(f"d = {d}: nullity {k}")
    if k > 0:
        lo = d - k + 1
        break
if lo is None:
    sys.exit("degree in t exceeds what the samples determine")
log("degree of Phi in t:", lo)
d = lo
assert nullity_mod(d, pts) == 1
# exact solve: fix the coefficient of t^0 m^8 ... use fmpq_mat nullspace on just enough rows
use = pts[: (DM + 1) * (d + 1) // DM + 4]
rows = system(d, use)
M = flint.fmpq_mat([[flint.fmpq(x.numerator, x.denominator) for x in r] for r in rows])
log("exact rref of", M.nrows(), "x", M.ncols())
R, rank = M.rref()
log("rank", rank, "of", M.ncols())
# nullspace vector from rref: one free column
piv, r = [], 0
for c in range(M.ncols()):
    if r < rank and R[r, c] != 0:
        piv.append(c); r += 1
free = [c for c in range(M.ncols()) if c not in piv]
assert len(free) == 1, free
vec = [flint.fmpq(0)] * M.ncols()
vec[free[0]] = flint.fmpq(1)
for r, c in enumerate(piv):
    vec[c] = -R[r, free[0]]
den = 1

for v in vec:
    den = den * int(v.q) // math.gcd(den, int(v.q))
ints = [int(v.p) * (den // int(v.q)) for v in vec]
g = 0
for x in ints:
    g = math.gcd(g, x)
ints = [x // g for x in ints]
coef = {(i, j): ints[j * (d + 1) + i] for j in range(DM + 1) for i in range(d + 1) if ints[j * (d + 1) + i]}
log("Phi:", len(coef), "nonzero terms; largest coefficient", max(len(str(abs(c))) for c in coef.values()), "digits")
# verify at all samples (exactly)
bad = 0
for tk, a in pts:
    p = [sum(F(coef.get((i, j), 0)) * tk**i for i in range(d + 1)) for j in range(DM + 1)]
    if any(p[j] != p[DM] * a[j] for j in range(DM)):
        bad += 1
log(f"exact check at all {len(pts)} samples: {len(pts) - bad} agree")
json.dump({f"{i},{j}": str(c) for (i, j), c in coef.items()}, open(prefix + ".json", "w"))
# factor content in t of each m-coefficient (structure)
import sympy as sp
tt, mm = sp.symbols("t m")
Phi = sp.Poly(sum(c * tt**i * mm**j for (i, j), c in coef.items()), mm, tt)
lc = sp.Poly(sum(c * tt**i for (i, j), c in coef.items() if j == DM), tt)
log("leading coefficient in m factors as:", sp.factor_list(lc.as_expr()))
