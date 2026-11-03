"""Figures for talk.tex, generated from the `extremal` package.

    python make_figures.py            # all figures
    python make_figures.py q12_optimal_crease ...

Each figure is a function decorated with @figure returning a matplotlib Figure, saved as
figures/<name>.pdf (vector, for Beamer). Figures are drawn at the size they appear on the slide
(the 16:9 text width is about 5.5 in), so 9-10 pt here is 9-10 pt on the slide. Diagrams that are
mostly typography (the mechanics triangle, the Kähler Venn diagram, the GIT cartoon, pipelines)
are TikZ in talk.tex instead.
"""

import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import mpmath  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Polygon as Patch  # noqa: E402

from extremal import Polygon, viz  # noqa: E402
from extremal.crease_exact import crease_minimum, optimal_crease  # noqa: E402
from extremal.models import wang_zhou_nonagon  # noqa: E402
from extremal.solver import ExtremalSolver  # noqa: E402

OUT = Path(__file__).parent / "figures"
FIGURES = {}

INK, INK_2, ACCENT, BLUE = viz.INK, viz.INK_2, viz.ACCENT, "#3987e5"
FS, FS_SMALL, FS_TITLE = 9, 8, 10
LW, LW_THIN = 1.3, 0.8
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS_TITLE, "savefig.bbox": "tight", "savefig.transparent": True,
    "savefig.pad_inches": 0.02, "mathtext.fontset": "cm", "text.color": INK, "axes.labelcolor": INK,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
})

LAM_STAR = mpmath.mpf("4.3175690953283481388139923642925619104856853")
LAM_STAR2 = 7.90742


def Q(k):
    """The labelled quadrilateral Q_k of notebook 03 (= P(5) of notebook 04 for k = 12)."""
    return Polygon([(0, 0), (5, 0), (4, 1), (0, 2)], labels=(1, k, k, k))


def P_lam(lam, k=12):
    lam = Fraction(lam)
    yv = (8 - lam) / 3
    return Polygon([(0, 0), (lam, 0), (lam - yv, yv), (0, 2)], labels=(1, k, k, k))


Q12 = Q(12)


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


def bare(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def plain_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def outline(ax, V, fill=None, lw=LW, color=INK, ls="-", z=3):
    V = np.asarray(V, float)
    if fill:
        ax.add_patch(Patch(V, closed=True, color=fill, lw=0, alpha=0.25, zorder=1))
    W = np.vstack([V, V[:1]])
    ax.plot(W[:, 0], W[:, 1], color=color, lw=lw, ls=ls, zorder=z, solid_joinstyle="round")


def crease_segment(P, a, b, s):
    """The chord {a x + b y = s} ∩ P (its two endpoints) and the piece where a x + b y ≥ s, in floats."""
    V = [np.array([float(c) for c in v]) for v in P.vertices]
    piece, on = [], []
    n = len(V)
    for i in range(n):
        p, q = V[i], V[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] - s, a * q[0] + b * q[1] - s
        if fp >= 0:
            piece.append(p)
        if fp * fq < 0:
            x = p + fp / (fp - fq) * (q - p)
            piece.append(x)
            on.append(x)
    return np.array(on), np.array(piece)


def _verts(P):
    return np.array([[float(c) for c in v] for v in P.vertices])


# ---- Topic 2: geodesics and wavefronts ---------------------------------------------------------

@figure
def eikonal():
    """Hyperbolic half-plane: geodesics from i (semicircles) are the characteristics of |∇S| = 1,
    whose level sets (hyperbolic circles: centre i cosh r, radius sinh r) are the wavefronts."""
    fig, ax = plt.subplots(figsize=(3.0, 1.75))
    bare(ax)
    t = np.linspace(0, 2 * np.pi, 400)
    for r in np.linspace(0.3, 2.1, 7):
        ax.plot(np.sinh(r) * np.cos(t), np.cosh(r) + np.sinh(r) * np.sin(t), color=BLUE, lw=LW_THIN)
    for ang in np.linspace(-80, 80, 9):
        th = np.radians(ang)
        if abs(ang) < 1e-9:
            ax.plot([0, 0], [0.02, 4.4], color=ACCENT, lw=LW)
            continue
        c = 1 / np.tan(th)  # the geodesic through i with direction (sin th, cos th): circle centred at (c, 0)
        R = np.hypot(c, 1)
        s = np.linspace(0, np.pi, 1200)
        X, Y = c + R * np.cos(s), R * np.sin(s)
        keep = (np.abs(X) < 4.4) & (Y < 4.4)
        ax.plot(X[keep], Y[keep], color=ACCENT, lw=LW)
    ax.plot([-4.4, 4.4], [0, 0], color=INK, lw=LW_THIN)
    ax.plot([0], [1], "o", color=INK, ms=3.5, zorder=5)
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-0.05, 4.4)
    return fig


# ---- Topic 4: moment maps of the sphere and the cigar ----------------------------------------

def _orbit(ax, x0, z, R):
    """A horizontal circle seen slightly from above: front half solid, back half dashed."""
    t = np.linspace(0, np.pi, 200)
    ax.plot(x0 + R * np.cos(t), z - 0.18 * R * np.sin(t), color=INK_2, lw=LW_THIN)
    ax.plot(x0 + R * np.cos(t), z + 0.18 * R * np.sin(t), color=INK_2, lw=LW_THIN, ls=(0, (2, 2)))


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.6, ls=(0, (3, 2))))


@figure
def moment_sphere_cigar():
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    bare(ax)
    # sphere: the moment map of rotation is the height (Archimedes)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=LW)
    zs = np.linspace(-0.8, 0.8, 5)
    for z in zs:
        _orbit(ax, 0, z, np.sqrt(1 - z * z))
        _arrow(ax, (np.sqrt(1 - z * z) + 0.08, z), (2.1, z))
    ax.plot([2.2, 2.2], [-1, 1], color=ACCENT, lw=3, solid_capstyle="round")
    for z in (-1, 1):
        ax.plot([0], [z], "o", color=ACCENT, ms=4, zorder=5)
        ax.plot([2.2], [z], "o", color=ACCENT, ms=5, zorder=5)
    ax.text(0, -1.5, r"$S^2$", ha="center", fontsize=FS_TITLE + 1)
    ax.text(2.2, -1.5, r"$[-1,1]$", ha="center", fontsize=FS_TITLE)
    # cigar: ds² = dr² + tanh²r dθ² as a surface of revolution; moment map log cosh r
    x0 = 5.0
    r = np.linspace(0, 2.6, 400)
    rad = np.tanh(r)
    dz = np.sqrt(np.clip(1 - (1 / np.cosh(r)) ** 4, 0, None))
    z = np.concatenate([[0], np.cumsum((dz[1:] + dz[:-1]) / 2 * np.diff(r))]) - 1
    ax.plot(x0 + rad, z, color=INK, lw=LW)
    ax.plot(x0 - rad, z, color=INK, lw=LW)
    ax.plot([x0], [z[0]], "o", color=ACCENT, ms=4, zorder=5)
    for rr in (0.35, 0.8, 1.3, 1.9):
        k = np.searchsorted(r, rr)
        _orbit(ax, x0, z[k], rad[k])
        _arrow(ax, (x0 + rad[k] + 0.08, z[k]), (x0 + 2.1, -1 + np.log(np.cosh(rr))))
    top = z[-1] + 0.1
    ax.annotate("", xy=(x0 + 2.2, top + 0.15), xytext=(x0 + 2.2, -1),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3, mutation_scale=14))
    ax.plot([x0 + 2.2], [-1], "o", color=ACCENT, ms=5, zorder=5)
    ax.text(x0, -1.5, "cigar", ha="center", fontsize=FS_TITLE)
    ax.text(x0 + 2.2, -1.5, r"$[0,\infty)$", ha="center", fontsize=FS_TITLE)
    ax.set_xlim(-1.3, x0 + 3)
    ax.set_ylim(-1.7, top + 0.3)
    return fig


# ---- Topic 5: Delzant polygons --------------------------------------------------------------

TRI = [(0, 0), (3, 0), (0, 3)]
BL1 = [(1, 0), (3, 0), (0, 3), (0, 1)]
BL2 = [(1, 0), (2, 0), (2, 1), (0, 3), (0, 1)]
SQ = [(0, 0), (2, 0), (2, 2), (0, 2)]
SQ1 = [(0, 0), (2, 0), (2, 1), (1, 2), (0, 2)]


def _polygon_panel(ax, V, ghost=None, title=None):
    bare(ax)
    if ghost is not None:
        outline(ax, ghost, lw=LW_THIN, color=INK_2, ls=(0, (3, 3)), z=2)
    outline(ax, V, fill=BLUE)
    V = np.asarray(V, float)
    ax.plot(V[:, 0], V[:, 1], "o", color=INK, ms=3, zorder=4)
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.3, 3.3)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, pad=4)


def _right_arrow(ax):
    ax.annotate("", xy=(1.17, 0.45), xytext=(1.0, 0.45), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.0))


@figure
def delzant_cp2():
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    _polygon_panel(ax, TRI)
    return fig


@figure
def delzant_blowups():
    fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.85))
    _polygon_panel(axes[0], TRI, title=r"$\mathbb{CP}^2$")
    _polygon_panel(axes[1], BL1, ghost=TRI, title=r"$\mathbb{CP}^2\#\overline{\mathbb{CP}}{}^2$")
    _polygon_panel(axes[2], BL2, ghost=TRI, title=r"$\mathbb{CP}^2\#2\overline{\mathbb{CP}}{}^2$")
    for a in axes[:2]:
        _right_arrow(a)
    return fig


@figure
def delzant_square():
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.9))
    _polygon_panel(axes[0], SQ, title=r"$\mathbb{CP}^1\times\mathbb{CP}^1$")
    _polygon_panel(axes[1], SQ1, ghost=SQ, title=r"$(\mathbb{CP}^1\times\mathbb{CP}^1)\#\overline{\mathbb{CP}}{}^2$")
    _right_arrow(axes[0])
    return fig


# ---- Topic 6/7: potentials and the Legendre transform ---------------------------------------

def _xlogx(t):
    return np.where(t > 0, t * np.log(np.where(t > 0, t, 1)), 0.0)


@figure
def guillemin_surface():
    """u = ½ Σ ℓ log ℓ over the CP² triangle {x, y ≥ 0, x + y ≤ 1}, drawn as a graph over the
    triangle, which sits on a floor plane below it."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def uG(x, y):
        return 0.5 * (_xlogx(x) + _xlogx(y) + _xlogx(np.clip(1 - x - y, 0, None)))

    s = np.linspace(0, 1, 140)
    X, Y = (g.ravel() for g in np.meshgrid(s, s))
    keep = X + Y <= 1 + 1e-12
    X, Y = X[keep], Y[keep]
    floor = -1.05
    corners = [(0, 0), (1, 0), (0, 1)]
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False  # draw in the order below: floor, drop lines, surface, rim
    ax.add_collection3d(Poly3DCollection([[(x, y, floor) for x, y in corners]], facecolor=BLUE, alpha=0.18,
                                         lw=0, zorder=1))
    edges = list(zip(corners, corners[1:] + corners[:1]))
    for (x0, y0), (x1, y1) in edges:
        ax.plot([x0, x1], [y0, y1], [floor, floor], color=INK, lw=LW, zorder=1)
    for x, y in corners:
        ax.plot([x, x], [y, y], [floor, 0], color=INK_2, lw=LW_THIN, ls=(0, (3, 2)), zorder=2)
    ax.plot_trisurf(mtri.Triangulation(X, Y), uG(X, Y), cmap=viz.SEQUENTIAL.reversed(), linewidth=0, alpha=1.0,
                    zorder=3)
    for (x0, y0), (x1, y1) in edges:
        t = np.linspace(0, 1, 200)
        xx, yy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        ax.plot(xx, yy, uG(xx, yy), color=INK, lw=LW, zorder=4)
    ax.text(0.3, 0.3, floor, r"$P$", color=INK, fontsize=FS_TITLE, ha="center", va="center", zorder=5)
    ax.text(1.0, 0.0, 0.08, r"$u_G$", color=INK, fontsize=FS_TITLE, ha="left", zorder=5)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(floor, 0)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((1, 1, 1.0), zoom=1.0)
    ax.set_axis_off()
    return fig


def _legendre_1d():
    """CP¹ of length 1: u(ξ) = ½(ξ log ξ + (1−ξ) log(1−ξ)), t = u'(ξ), φ(t) = ξ t − u(ξ)."""
    xi = np.linspace(1e-6, 1 - 1e-6, 4000)
    u = 0.5 * (xi * np.log(xi) + (1 - xi) * np.log(1 - xi))
    t = 0.5 * np.log(xi / (1 - xi))
    return xi, u, t, xi * t - u


@figure
def legendre_toric():
    xi, u, t, phi = _legendre_1d()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 1.9))
    k = np.argmin(np.abs(xi - 0.75))
    m = np.abs(t) < 4
    a1.plot(t[m], phi[m], color=BLUE, lw=1.8)
    tt = np.array([-1.5, 4])
    a1.plot(tt, phi[k] + xi[k] * (tt - t[k]), color=ACCENT, lw=LW_THIN + 0.2)
    a1.plot([t[k]], [phi[k]], "o", color=ACCENT, ms=4)
    a1.set_title(r"Kähler potential $\varphi(t)$, $t\in\mathbb{R}$")
    a1.text(-3.9, 2.9, r"slope $\xi=\varphi'(t)\in(0,1)$", color=ACCENT, fontsize=FS)
    a2.plot(xi, u, color=BLUE, lw=1.8)
    xx = np.array([0.45, 1.0])
    a2.plot(xx, u[k] + t[k] * (xx - xi[k]), color=ACCENT, lw=LW_THIN + 0.2)
    a2.plot([xi[k]], [u[k]], "o", color=ACCENT, ms=4)
    a2.set_title(r"symplectic potential $u(\xi)$, $\xi\in[0,1]$")
    a2.text(0.03, -0.07, r"slope $t=u'(\xi)\in\mathbb{R}$", color=ACCENT, fontsize=FS)
    a2.set_xlim(-0.05, 1.05)
    for a in (a1, a2):
        plain_axes(a)
    fig.tight_layout(w_pad=2)
    return fig


# ---- Topics 8, 10, 11: creases, the optimal destabilizer, the code ---------------------------

_CACHE = {}


def opt_q12():
    if "q12" not in _CACHE:
        _CACHE["q12"] = optimal_crease(Q12)
    return _CACHE["q12"]


def _crease_floats(opt):
    a, b = (float(v) for v in opt.crease.normal)
    return a, b, float(opt.crease.s)


def _draw_crease(ax, P, crease, lw=2.0, shade=False):
    on, piece = crease_segment(P, *crease)
    if shade:
        ax.add_patch(Patch(piece, closed=True, color=ACCENT, alpha=0.2, lw=0, zorder=1))
    ax.plot(on[:, 0], on[:, 1], color=ACCENT, lw=lw, zorder=6, solid_capstyle="round")


def _field(ax, P, values_fn, amp, resolution=140):
    """Colour P by a signed field on a fixed symmetric scale ±amp (values beyond it saturate)."""
    tri = viz.sample_mesh(P, resolution)
    vals = np.asarray(values_fn(tri.x, tri.y), float)
    art = ax.tripcolor(tri, np.clip(vals, -amp, amp), cmap=viz.DIVERGING, norm=TwoSlopeNorm(0, -amp, amp),
                       shading="gouraud", rasterized=True)
    outline(ax, _verts(P), lw=LW_THIN + 0.2)
    bare(ax)
    return art


@figure
def q12_optimal_crease():
    fig, ax = plt.subplots(figsize=(2.8, 1.3))
    bare(ax)
    outline(ax, _verts(Q12))
    _draw_crease(ax, Q12, _crease_floats(opt_q12()), shade=True)
    return fig


@figure
def q12_pieces():
    """f* on Q12, the optimal destabilizer: A − f* is the extremal affine function of each piece."""
    opt = opt_q12()
    fig, ax = plt.subplots(figsize=(3.6, 1.6))
    art = _field(ax, Q12, opt, 0.27)
    _draw_crease(ax, Q12, _crease_floats(opt))
    ax.text(1.3, 0.75, r"$P_1$", fontsize=13, ha="center", va="center")
    ax.text(3.65, 0.7, r"$P_2$", fontsize=13, ha="center", va="center")
    cb = fig.colorbar(art, ax=ax, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.set_label(r"$f^*$", fontsize=FS)
    return fig


@figure
def crease_tent():
    """The crease function max(0, ℓ) over Q12, as a surface: flat on one piece, a ramp on the other."""
    a, b, s = _crease_floats(opt_q12())
    on, piece = crease_segment(Q12, a, b, s)
    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_subplot(projection="3d")
    V = _verts(Q12)
    flat = [V[i] for i in range(len(V)) if a * V[i, 0] + b * V[i, 1] <= s]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # the zero piece, ordered around the polygon: the vertices on the low side plus the chord
    low = np.array(flat + [on[1], on[0]])
    c = low.mean(axis=0)
    low = low[np.argsort(np.arctan2(low[:, 1] - c[1], low[:, 0] - c[0]))]
    ax.add_collection3d(Poly3DCollection([np.c_[low, np.zeros(len(low))]], facecolor=BLUE, alpha=0.25, lw=0))
    hc = piece.mean(axis=0)
    hp = piece[np.argsort(np.arctan2(piece[:, 1] - hc[1], piece[:, 0] - hc[0]))]
    hz = np.maximum(0, a * hp[:, 0] + b * hp[:, 1] - s)
    ax.add_collection3d(Poly3DCollection([np.c_[hp, hz]], facecolor=ACCENT, alpha=0.55, edgecolor=ACCENT, lw=0.6))
    for p in hp:  # vertical drop lines from the ramp to the polygon
        z = max(0, a * p[0] + b * p[1] - s)
        if z > 1e-9:
            ax.plot([p[0], p[0]], [p[1], p[1]], [0, z], color=ACCENT, lw=0.6, ls=(0, (2, 2)))
    W = np.vstack([V, V[:1]])
    ax.plot(W[:, 0], W[:, 1], 0 * W[:, 0], color=INK, lw=LW)
    ax.plot(on[:, 0], on[:, 1], [0, 0], color=ACCENT, lw=2)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(0, 1.2)
    ax.set_box_aspect((5, 3, 1.6), zoom=1.15)
    ax.view_init(elev=28, azim=-62)
    ax.set_axis_off()
    return fig


@figure
def nonagon():
    P = wang_zhou_nonagon(10000)
    cr = crease_minimum(P).crease
    a, b = (float(v) for v in cr.normal)
    V = _verts(P)
    on, piece = crease_segment(P, a, b, float(cr.s))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.1, 2.0), gridspec_kw={"width_ratios": [1, 3]})
    for ax, ylim in ((a1, (-10300, 10300)), (a2, (9840, 10030))):
        ax.add_patch(Patch(piece, closed=True, color=ACCENT, alpha=0.2, lw=0, zorder=1))
        W = np.vstack([V, V[:1]])
        ax.plot(W[:, 0], W[:, 1], color=INK, lw=LW, zorder=3)
        ax.plot(on[:, 0], on[:, 1], color=ACCENT, lw=1.8, zorder=4)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, 7.5)
        ax.set_axis_off()
    a1.add_patch(plt.Rectangle((-0.3, 9800), 7.6, 260, fill=False, ec=INK_2, lw=0.6, ls=(0, (3, 2))))
    a2.plot(V[:, 0], V[:, 1], "o", color=INK, ms=2.5, zorder=5)
    a1.text(3.5, -10600, "not to\nscale", ha="center", va="top", fontsize=FS_SMALL, color=INK_2)
    a2.text(3.5, 9832, "the top, magnified", ha="center", va="top", fontsize=FS_SMALL, color=INK_2)
    return fig


@figure
def labels_q4_q12():
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 1.55))
    for ax, k in zip(axes, (4, 12)):
        bare(ax)
        outline(ax, _verts(Q(k)), fill=BLUE if k == 4 else None)
        for e, lab in zip(Q(k).edges, (1, k, k, k)):
            mid = (np.array(e.start, float) + np.array(e.end, float)) / 2
            nrm = np.array(e.normal, float)
            pos = mid - 0.33 * nrm / np.linalg.norm(nrm)
            ax.text(pos[0], pos[1], str(lab), ha="center", va="center", fontsize=FS_TITLE,
                    color=ACCENT if lab != 1 else INK)
        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(-0.6, 2.6)
    _draw_crease(axes[1], Q12, _crease_floats(opt_q12()), shade=True)
    axes[0].set_title("extremal metric exists")
    axes[1].set_title("unstable")
    return fig


@figure
def solver_haywire():
    """S − A_ext of the solver's polynomial potential (degree 14) on Q4 and Q12, and −f*, on one colour scale."""
    amp = 0.27  # max |f*| on Q12
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.25))
    runs = ((Q(4), {}), (Q12, dict(quad_scheme="edges", grading=2, quad_order=54)))
    for ax, (P, kw) in zip(axes[:2], runs):
        r = ExtremalSolver(P, degree=14, **kw).solve(max_iter=500)
        _field(ax, P, r.metric.extremal_defect, amp)
    opt = opt_q12()
    art = _field(axes[2], Q12, lambda x, y: -opt(x, y), amp)
    _draw_crease(axes[2], Q12, _crease_floats(opt), lw=1.4)
    for ax, title in zip(axes, (r"$Q_4$: solver's $S-A$", r"$Q_{12}$: solver's $S-A$", r"$Q_{12}$: the limit $-f^*$")):
        ax.set_title(title, fontsize=FS)
    cb = fig.colorbar(art, ax=axes, shrink=0.8, pad=0.02, extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, colors=INK_2)
    return fig


@figure
def threshold():
    """Stability margin of P(λ) (the exact crease minimum) against λ: it changes sign at λ* and λ**."""
    lams = sorted(set(np.round(np.linspace(2.4, 7.95, 38), 4)) | {4.2, 4.3, 4.35, 7.85, 7.9, 7.93})
    vals = np.array([float(crease_minimum(P_lam(Fraction(f"{lam:.4f}"))).value) for lam in lams])
    fig, ax = plt.subplots(figsize=(4.0, 1.6))
    ax.axhline(0, color=INK_2, lw=LW_THIN)
    ax.axvspan(float(LAM_STAR), LAM_STAR2, color=ACCENT, alpha=0.12, lw=0)
    ax.plot(lams, vals, color=INK, lw=LW)
    ax.plot([float(LAM_STAR)], [0], "o", color=ACCENT, ms=4.5, zorder=5)
    ax.text(float(LAM_STAR) - 0.08, 0.08, r"$\lambda^*$", fontsize=FS_TITLE + 1, ha="right", va="bottom")
    ax.text(5.0, -1.6, "unstable", color=ACCENT, fontsize=FS, ha="center")
    ax.text(3.3, -1.6, "stable", color=BLUE, fontsize=FS, ha="center")
    ax.set_xlabel(r"$\lambda$", labelpad=1)
    ax.set_ylabel("stability margin", fontsize=FS_SMALL)
    plain_axes(ax)
    for lam, x0 in (("3", 0.02), ("4.3176", 0.36), ("6.5", 0.70)):
        ins = ax.inset_axes([x0, 1.06, 0.26, 0.42])
        outline(ins, _verts(P_lam(Fraction(lam))), fill=BLUE, lw=LW_THIN)
        bare(ins)
        ins.set_xlim(-0.2, 8)
        ins.set_ylim(-0.2, 2.2)
        ins.text(4, -0.4, rf"$\lambda={lam}$" if lam != "4.3176" else r"$\lambda=\lambda^*$",
                 ha="center", va="top", fontsize=FS_SMALL)
    return fig


def main(names):
    OUT.mkdir(exist_ok=True)
    for name in names or FIGURES:
        fig = FIGURES[name]()
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
        print(f"wrote figures/{name}.pdf", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
