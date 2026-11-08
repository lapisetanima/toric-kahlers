"""Sample the wall sheet of chamber S: at rational parameter values, the degree-8 minimal polynomial
of the critical value m (made monic), by exact elimination. Feeds wallinterp.py.

usage: wallsample.py S s_or_sym t_values_spec out.json
  t values: k/D for k in range(a, b)   given as "a:b:D"; with s = sym, s values likewise via env S_SPEC
"""
import json, math, os, sys, time
from fractions import Fraction as F
from multiprocessing import Pool
import flint
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from creasealg import family, A_ext, phis, C, s, t, m, w

NAMES = ("c0", "c2", "m", "s", "t")


def symbolic(Sset):
    V, sig = family()
    ph = phis(V, sig, A_ext(V, sig))
    sub = {w[0]: 1, w[1]: 1, w[2]: m, w[3]: 1, w[4]: 1, C[1]: 1}
    num, _ = sp.fraction(sp.together(sum(ph[i] for i in Sset).subs(sub)))
    P = sp.Poly(sp.expand(num), C[0], C[2], m, s, t).clear_denoms()[1]
    return {mon: int(cf) for mon, cf in P.terms()}


def specialize(Nd, sv, tv):
    """N(c0, c2, m) at s = sv, t = tv, as an integer polynomial in (c0, c2, m)."""
    ctx = flint.fmpz_mpoly_ctx.get(("c0", "c2", "m"), "lex")
    acc = {}
    for (a, b, c, i, j), cf in Nd.items():
        v = cf * sv**i * tv**j
        acc[(a, b, c)] = acc.get((a, b, c), 0) + v
    den = 1
    for v in acc.values():
        den = den * v.denominator // math.gcd(den, v.denominator)
    return ctx, ctx.from_dict({k: int(v * den) for k, v in acc.items() if v})


def univ(p, idx):
    d = p.to_dict()
    deg = max(e[idx] for e in d)
    co = [0] * (deg + 1)
    for e, c in d.items():
        co[e[idx]] += int(c)
    return flint.fmpz_poly(co)


def sample(args):
    Nd, sv, tv = args
    ctx, N = specialize(Nd, sv, tv)
    gm = ctx.gens()[2]
    Nh = N.derivative("m")
    Nr = N - gm * Nh
    G = Nr.gcd(Nh)
    Nr, Nh = Nr // G, Nh // G
    E0 = Nr * Nh.derivative("c0") - Nh * Nr.derivative("c0")
    Eu = Nr * Nh.derivative("c2") - Nh * Nr.derivative("c2")
    H = E0.gcd(Eu)
    E0, Eu = E0 // H, Eu // H
    R = E0.resultant(Eu, "c0")
    _, facs = R.factor()
    f8 = [f for f, e in facs if f.degrees()[1] == 8]
    if len(f8) != 1:
        return {"s": str(sv), "t": str(tv), "bad": [[int(x) for x in f.degrees()] for f, e in facs]}
    Hm = E0.resultant(Nr + gm * Nh, "c0")
    Rm = f8[0].resultant(Hm, "c2")
    _, mf = Rm.factor()
    g8 = [g for g, e in mf if g.degrees()[2] == 8]
    if len(g8) != 1:
        return {"s": str(sv), "t": str(tv), "bad_m": [[int(x) for x in g.degrees()] for g, e in mf]}
    co = univ(g8[0], 2).coeffs()          # ascending
    lead = F(int(co[-1]))
    return {"s": str(sv), "t": str(tv), "monic": [str(F(int(c)) / lead) for c in co[:-1]],
            "slope": [str(int(c)) for c in univ(f8[0], 1).coeffs()]}


def rng(spec):
    a, b, D = (int(x) for x in spec.split(":"))
    return [F(k, D) for k in range(a, b)]


if __name__ == "__main__":
    Sset = [int(c) for c in sys.argv[1].split(",")]
    svals = rng(os.environ["S_SPEC"]) if sys.argv[2] == "sym" else [F(sys.argv[2])]
    tvals = rng(sys.argv[3])
    out = sys.argv[4]
    T = time.time()
    Nd = symbolic(Sset)
    print(f"symbolic N: {len(Nd)} terms [{time.time() - T:.1f}s]", flush=True)
    jobs = [(Nd, sv, tv) for sv in svals for tv in tvals if tv != sv]
    with Pool(19) as p:
        res = p.map(sample, jobs, chunksize=1)
    json.dump(res, open(out, "w"))
    bad = [r for r in res if "monic" not in r]
    print(f"{len(res)} samples, {len(bad)} degenerate [{time.time() - T:.1f}s]", flush=True)
    for r in bad[:5]:
        print("  ", r)
