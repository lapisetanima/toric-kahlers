"""Symbolic crease functional on the CP2#2 family: L(c; s, t, w) as a Brion-Lawrence vertex sum.

l = c0 + c1 x + c2 y. For a vertex v with edge vectors e1, e2 to its neighbours (actual vectors),
d_i = c1 e_ix + c2 e_iy = l(nbr_i) - l(v), b_i = A(nbr_i) - A(v), sigma_i = w_i * lattice length:
  phi_v = -sum_i sigma_i l(v)^2 / (2 d_i) - |det(e1,e2)|/(d1 d2) * (A(v) l(v)^3/6 - (b1/d1 + b2/d2) l(v)^4/24)
and L(c) = sum over v with l(v) > 0 of phi_v.
"""
import sympy as sp

c0, c1, c2 = C = sp.symbols("c0 c1 c2")
s, t, m = sp.symbols("s t m")
w = sp.symbols("w0:5")


def family(s=s, t=t, w=w):
    V = [(0, 0), (1 - s, 0), (1 - s, s), (t, 1 - t), (0, 1 - t)]
    lengths = [1 - s, s, 1 - s - t, t, 1 - t]          # edge i: V[i] -> V[i+1]
    return [tuple(map(sp.sympify, v)) for v in V], [w[i] * lengths[i] for i in range(5)]


def moments(V):
    """∫ 1, x, y, x^2, xy, y^2 over the polygon (CCW), by Green's theorem."""
    M = [0] * 6
    n = len(V)
    for i in range(n):
        (x0, y0), (x1, y1) = V[i], V[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        M[0] += cr / 2
        M[1] += cr * (x0 + x1) / 6
        M[2] += cr * (y0 + y1) / 6
        M[3] += cr * (x0**2 + x0 * x1 + x1**2) / 12
        M[4] += cr * (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1) / 24
        M[5] += cr * (y0**2 + y0 * y1 + y1**2) / 12
    return [sp.factor(sp.expand(q)) for q in M]


def A_ext(V, sig):
    """(a0, a1, a2) with A = a0 + a1 x + a2 y and L_A(1) = L_A(x) = L_A(y) = 0, as (numerators, Delta):
    Delta = det of the moment matrix (a polynomial in s, t), numerators polynomial in s, t, linear in w."""
    M = moments(V)
    G = sp.Matrix([[M[0], M[1], M[2]], [M[1], M[3], M[4]], [M[2], M[4], M[5]]])
    n = len(V)
    rhs = sp.Matrix([sum(sig[i] * f(V[i], V[(i + 1) % n]) for i in range(n))
                     for f in (lambda p, q: 1, lambda p, q: (p[0] + q[0]) / 2, lambda p, q: (p[1] + q[1]) / 2)])
    delta = sp.expand(G.det(method="berkowitz"))
    num = G.adjugate(method="berkowitz") * rhs
    return [sp.expand(q) for q in num], delta


def phis(V, sig, A):
    a, delta = A
    Af = lambda p: (a[0] + a[1] * p[0] + a[2] * p[1]) / delta
    n = len(V)
    out = []
    for i in range(n):
        v, nb = V[i], (V[(i + 1) % n], V[i - 1])
        sg = (sig[i], sig[i - 1])
        e = [(q[0] - v[0], q[1] - v[1]) for q in nb]
        det = e[1][0] * e[0][1] - e[1][1] * e[0][0]
        det = -det if det.subs({s: sp.Rational(1, 7), t: sp.Rational(1, 5)}) < 0 else det
        lv = c0 + c1 * v[0] + c2 * v[1]
        d = [c1 * ee[0] + c2 * ee[1] for ee in e]
        b = [Af(q) - Af(v) for q in nb]
        bd = -sum(sg[j] * lv**2 / (2 * d[j]) for j in range(2))
        inter = det / (d[0] * d[1]) * (Af(v) * lv**3 / 6 - (b[0] / d[0] + b[1] / d[1]) * lv**4 / 24)
        out.append(bd - inter)
    return out


if __name__ == "__main__":
    import random, time
    from fractions import Fraction as F
    from extremal.models import cp2_two_blowups
    T = time.time()
    V, sig = family()
    A = A_ext(V, sig)
    print("A_ext computed", round(time.time() - T, 1), "s", flush=True)
    ph = phis(V, sig, A)
    random.seed(2)
    for trial in range(4):
        sv, tv = F(random.randint(1, 40), 100), F(random.randint(1, 40), 100)
        kv = F(1, random.randint(2, 12))
        P = cp2_two_blowups(sv, tv, kv)
        sub = {s: sp.Rational(sv), t: sp.Rational(tv), w[0]: 1, w[1]: 1, w[2]: 1 / sp.Rational(kv), w[3]: 1, w[4]: 1}
        fr = [sp.fraction(sp.together(p.subs(sub))) for p in ph]
        fr = [(sp.Poly(nu, *C), sp.Poly(de, *C)) for nu, de in fr]
        tot = sp.Poly(sp.together(sum(nu.as_expr() / de.as_expr() for nu, de in fr)).as_numer_denom()[0], *C)
        worst = 0
        for _ in range(100):
            x1, x2 = F(random.randint(-50, 50), 17), F(random.randint(-50, 50), 13)
            if x1 == 0 or x2 == 0 or x1 == x2: continue
            cc = -(x1 * F(random.randint(0, 100), 100) + x2 * F(random.randint(0, 100), 100))
            ex = P.L_crease(x1, x2, cc)
            pt = [sp.Rational(cc), sp.Rational(x1), sp.Rational(x2)]
            form = sum(nu.eval(dict(zip(C, pt))) / de.eval(dict(zip(C, pt)))
                       for i, (nu, de) in enumerate(fr) if x1 * P.vertices[i][0] + x2 * P.vertices[i][1] + cc > 0)
            worst = max(worst, abs(sp.Rational(ex.numerator, ex.denominator) - form))
        print(f"s={sv}, t={tv}, k={kv}: max diff over random creases {worst};  sum of all phi_v is 0: {tot.is_zero}", flush=True)
    print("total", round(time.time() - T, 1), "s")
