"""Step 2: at fixed rational (s, t), eliminate the crease c from grad_c (N_rest + m N_hyp) = 0.

usage: wallpoly.py s t      (s, t as fractions; uses the cached balanced crease to pick chamber/factors)
"""
import json, os, sys, time
from fractions import Fraction as F
import flint
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from creasealg import family, A_ext, phis, C, s, t, m, w

sv, tv = F(sys.argv[1]), F(sys.argv[2])
T = time.time()
rec = json.load(open("notebooks/data/cp2_2_thresholds.json"))[f"{sv}|{tv}"]
mstar, line = rec["m_star"], rec["line"]
V, sig = family()
ph = phis(V, sig, A_ext(V, sig))
sub = {s: sp.Rational(sv), t: sp.Rational(tv), w[0]: 1, w[1]: 1, w[2]: m, w[3]: 1, w[4]: 1}
Vn = [(float(x.subs(sub)), float(y.subs(sub))) for x, y in V]
lv = [line[0] + line[1] * x + line[2] * y for x, y in Vn]
Sset = [i for i in range(5) if lv[i] > 1e-9] if len(sys.argv) < 4 else [int(c) for c in sys.argv[3].split(",")]
print(f"(s, t) = ({sv}, {tv}); cached m* = {mstar:.10f}; balanced line {np.round(line, 6)}; l(v) = {np.round(lv, 4)}; chamber S = {Sset}")

LS = sp.together(sum(ph[i].subs(sub) for i in Sset))
num, den = sp.fraction(LS)
assert not den.has(m)
# chart: divide by the larger of c1, c2
k = 1 if abs(line[1]) >= abs(line[2]) else 2
other = 3 - k
u = C[other]
names = ("c0", str(u), "m")
ctx = flint.fmpz_mpoly_ctx.get(names, "lex")
P = sp.Poly(sp.expand(num.subs(C[k], 1)), C[0], u, m)
P = P.clear_denoms()[1]
N = ctx.from_dict({mon: int(cf) for mon, cf in P.terms()})
g0, gu, gm = ctx.gens()
Nh = N.derivative("m")
Nr = N - gm * Nh
assert Nr.derivative("m").is_zero()
G = Nr.gcd(Nh)
Nr, Nh = Nr // G, Nh // G
print("deg N_rest, N_hyp (c0, u):", Nr.degrees()[:2], Nh.degrees()[:2], " common factor degree:", G.total_degree())

E0 = Nr * Nh.derivative("c0") - Nh * Nr.derivative("c0")
Eu = Nr * Nh.derivative(str(u)) - Nh * Nr.derivative(str(u))
H = E0.gcd(Eu)
E0, Eu = E0 // H, Eu // H
print("critical equations: degrees", E0.degrees()[:2], Eu.degrees()[:2], " removed common factor of degree", H.total_degree(), f"[{time.time()-T:.1f}s]", flush=True)

R = E0.resultant(Eu, "c0")
print(f"resultant in u: total degree {R.total_degree()} [{time.time()-T:.1f}s]", flush=True)
ustar = line[other] / line[k]
c0star = line[0] / line[k]

def uni_roots(p, var_index):
    d = p.to_dict()
    deg = max(e[var_index] for e in d)
    co = [0] * (deg + 1)
    for e, c in d.items():
        co[deg - e[var_index]] += int(c)
    if deg == 0:
        return np.array([])
    rts = flint.fmpz_poly(list(reversed(co))).complex_roots()
    return np.array([complex(r.real.mid()) + 1j * float(r.imag.mid()) for r, _ in rts])

c, facs = R.factor()
best = None
for f, e in facs:
    rts = uni_roots(f, 1)
    if len(rts) == 0: continue
    dist = min(abs(rts - ustar))
    print(f"   factor of degree {f.degrees()[1]} (mult {e}): closest root to u* = {ustar:.6f} at distance {dist:.2e}")
    if best is None or dist < best[1]:
        best = (f, dist)
fu = best[0]
print(f"slope factor: degree {fu.degrees()[1]} in u", flush=True)

Fm = Nr + gm * Nh
Hm = E0.resultant(Fm, "c0")
Rm = fu.resultant(Hm, str(u))
c, facs = Rm.factor()
best = None
for f, e in facs:
    rts = uni_roots(f, 2)
    if len(rts) == 0: continue
    dist = min(abs(rts - mstar))
    print(f"   m-factor of degree {f.degrees()[2]}: closest root to m* at distance {dist:.2e}")
    if best is None or dist < best[1]:
        best = (f, dist)
fm = best[0]
print(f"\nminimal polynomial of m*({sv}, {tv}): degree {fm.degrees()[2]}  [{time.time()-T:.1f}s]")
d = fm.to_dict()
print("    largest coefficient has", max(len(str(abs(int(c)))) for c in d.values()), "digits")
rts = uni_roots(fm, 2)
print("    roots:", ", ".join(f"{r.real:.6f}" + (f"{r.imag:+.3f}i" if abs(r.imag) > 1e-9 else "") for r in sorted(rts, key=lambda z: (abs(z.imag) > 1e-9, z.real))))
