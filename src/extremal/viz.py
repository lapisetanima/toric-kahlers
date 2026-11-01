"""Matplotlib plots of Delzant polygons and of functions on them.

Magnitudes (S, |Ric|^2, ...) use a one-hue sequential blue ramp; signed quantities centred
at zero (S − Ŝ, S − A_ext) use a blue ↔ red diverging map with a neutral grey midpoint.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .exact import as_fraction
from .integrate import clip_halfplane
from .metric import ToricMetric
from .polygon import Polygon

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
ACCENT = "#eb6834"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "extremal_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "extremal_div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8c1c1c"]
)


def _float_vertices(P: Polygon) -> np.ndarray:
    return np.array([[float(c) for c in v] for v in P.vertices])


def _new_axes(ax, size=5.0):
    if ax is None:
        fig, ax = plt.subplots(figsize=(size, size))
        fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    return ax


def sample_mesh(P: Polygon, resolution: int = 120, inset: float = 1e-3) -> mtri.Triangulation:
    """A triangulation filling P up to a thin inset (fields have removable singularities on ∂P)."""
    V = _float_vertices(P)
    centre = np.array([float(c) for c in P.area_barycenter])
    W = centre + (1 - inset) * (V - centre)  # slightly shrunk copy, strictly inside P
    lo, hi = W.min(axis=0), W.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], resolution), np.linspace(lo[1], hi[1], resolution))
    gx, gy = gx.ravel(), gy.ravel()
    inside = np.ones_like(gx, dtype=bool)
    for i in range(len(W)):
        a, b = W[i], W[(i + 1) % len(W)]
        inside &= (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0]) > 1e-12
    step = (hi - lo).max() / resolution
    bx, by = [], []
    for i in range(len(W)):
        a, b = W[i], W[(i + 1) % len(W)]
        k = max(int(np.linalg.norm(b - a) / step), 2)
        t = np.linspace(0, 1, k, endpoint=False)
        bx.append(a[0] + t * (b[0] - a[0]))
        by.append(a[1] + t * (b[1] - a[1]))
    x = np.concatenate([gx[inside], *bx])
    y = np.concatenate([gy[inside], *by])
    return mtri.Triangulation(x, y)


def plot_polygon(
    P: Polygon, ax=None, normals: bool = True, labels: bool = True, vertex_labels: bool = True, lw: float = 1.6
):
    """Draw P with its vertices, inward normals u_i and edge labels (normal, lattice length)."""
    ax = _new_axes(ax)
    V = _float_vertices(P)
    closed = np.vstack([V, V[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=INK, lw=lw, zorder=3)
    ax.scatter(V[:, 0], V[:, 1], s=36, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    span = np.ptp(V, axis=0).max()
    centre = V.mean(axis=0)
    if vertex_labels:
        for i, (x, y) in enumerate(V):
            off = np.array([x, y]) - centre
            off = off / (np.linalg.norm(off) or 1) * 0.05 * span
            ax.annotate(f"v{i}", (x, y), xytext=(x + off[0], y + off[1]), color=INK_2, fontsize=8,
                        ha="center", va="center")
    for e in P.edges:
        a, b = np.array([float(c) for c in e.start]), np.array([float(c) for c in e.end])
        mid = (a + b) / 2
        u = np.array(e.normal, float)
        u_hat = u / np.linalg.norm(u)
        if normals:
            ax.annotate(
                "", xy=mid + 0.09 * span * u_hat, xytext=mid,
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.0, shrinkA=0, shrinkB=0), zorder=4,
            )
        if labels:
            pos = mid - 0.09 * span * u_hat
            text = f"u={e.normal}\nL={e.lattice_length}" + (f"\nm={e.label}" if P.is_labelled else "")
            ax.text(pos[0], pos[1], text, color=INK_2, fontsize=7, ha="center", va="center")
    pad = (0.25 if labels else 0.1) * span
    ax.set_xlim(V[:, 0].min() - pad, V[:, 0].max() + pad)
    ax.set_ylim(V[:, 1].min() - pad, V[:, 1].max() + pad)
    if P.name:
        ax.set_title(P.name, color=INK, fontsize=11, loc="left")
    return ax


def plot_function(
    P: Polygon,
    fn,
    label: str = "",
    ax=None,
    diverging: bool = False,
    resolution: int = 120,
    colorbar: bool = True,
    outline: bool = True,
    clip_percentile: float | None = 98.0,
):
    """Colour P by fn(x, y). `diverging=True` centres the colour scale at 0.

    Many fields blow up or vary fast near ∂P, which would wash out the interior. The colour
    scale therefore covers the central `clip_percentile` % of values; values beyond it are
    shown in the end colours (marked by the arrow ends on the colour bar). Pass None to use
    the full range.
    """
    ax = _new_axes(ax)
    tri = sample_mesh(P, resolution)
    values = np.asarray(fn(tri.x, tri.y), dtype=float)
    q = clip_percentile if clip_percentile is not None else 100.0
    constant = np.nanmax(values) - np.nanmin(values) < 1e-5 * max(1.0, float(np.nanmax(np.abs(values))))
    if constant:  # numerically constant: show its value instead of amplified round-off
        mean = float(np.nanmean(values))
        mean = 0.0 if abs(mean) < 1e-6 else mean
        values = np.full_like(values, mean)
        ax.text(0.98, 0.98, f"≡ {mean:.6g}", transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=10)
    if diverging:
        amp = float(np.nanpercentile(np.abs(values), q))
        amp = amp if amp > 1e-9 else 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-amp, vmax=amp)
        clipped = np.nanmax(np.abs(values)) > amp
        art = ax.tripcolor(tri, values, cmap=DIVERGING, norm=norm, shading="gouraud", rasterized=True)
    else:
        lo, hi = (float(v) for v in np.nanpercentile(values, [100 - q, q]))
        if constant or hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        clipped = np.nanmin(values) < lo or np.nanmax(values) > hi
        art = ax.tripcolor(tri, values, cmap=SEQUENTIAL, vmin=lo, vmax=hi, shading="gouraud", rasterized=True)
    if outline:
        plot_polygon(P, ax=ax, normals=False, labels=False, vertex_labels=False, lw=1.2)
    if colorbar:
        cb = ax.figure.colorbar(art, ax=ax, shrink=0.8, extend="both" if clipped else "neither")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(label, color=INK, fontsize=11, loc="left")
    return ax


def plot_field(metric: ToricMetric, name: str = "S", ax=None, resolution: int = 120, **kwargs):
    """Plot a named curvature field of a metric (see ToricMetric.FIELDS)."""
    fn, label = metric.field(name)
    diverging = kwargs.pop("diverging", name in ("S - S_hat", "S - A_ext"))
    return plot_function(metric.polygon, fn, label=label, ax=ax, diverging=diverging, resolution=resolution, **kwargs)


def plot_fields(metric: ToricMetric, names=("S", "S - S_hat", "S - A_ext", "|Ric|^2"), resolution: int = 100):
    """Small multiples: one panel (with its own colour scale) per field."""
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes[0], names):
        plot_field(metric, name, ax=ax, resolution=resolution)
    fig.suptitle(f"{metric.name} metric on {metric.polygon.name or 'polygon'}", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_A_ext(P: Polygon, ax=None, levels: int = 9, resolution: int = 120):
    """Level lines of the extremal affine function A_ext."""
    ax = _new_axes(ax) if ax is None else ax
    tri = sample_mesh(P, resolution)
    values = P.A_ext(tri.x, tri.y)
    if np.ptp(values) < 1e-12:
        ax.text(0.02, 0.02, f"A_ext ≡ {P.A_ext.c0} (constant)", transform=ax.transAxes, color=INK_2, fontsize=8)
        return ax
    cs = ax.tricontour(tri, values, levels=levels, colors=INK_2, linewidths=0.8, linestyles="--")
    ax.clabel(cs, fontsize=7, fmt="%.3g")
    return ax


def _line_axes(ax, figsize=(6.4, 3.4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    return ax


def plot_crease_profile(profile, ax=None, samples: int = 400, normalized: bool = False):
    """L_A(max(0, n·x − s)) as a function of the offset s, for one direction n.

    normalized=True plots L(f) / ∫_∂P f dσ (f vanishing at the barycenter) instead.
    """
    ax = _line_axes(ax)
    lo, hi = (float(v) for v in profile.t_range)
    s = np.linspace(lo, hi, samples + 2)[1:-1]
    L, ratio = profile.stability_ratio(s)
    y = ratio if normalized else L
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.plot(s, y, color="#2a78d6", lw=2)
    if normalized:
        ax.axvline(profile.barycenter_level, color=INK_2, lw=0.8, ls=":")
        ax.text(profile.barycenter_level, ax.get_ylim()[1], " n·(barycenter)", color=INK_2, fontsize=8, va="top")
    i = int(np.nanargmin(y))
    if normalized or y[i] < 0:  # the unnormalized minimum is otherwise a trivial ~0 at an end
        ax.scatter([s[i]], [y[i]], s=40, color="#2a78d6", edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"min {y[i]:.4g} at s = {s[i]:.4g}", (s[i], y[i]), xytext=(6, -12),
                    textcoords="offset points", color=INK, fontsize=8)
    n = tuple(float(c) for c in profile.n)
    ax.set_xlabel("offset s  (crease n·x = s)", color=INK_2, fontsize=9)
    ax.set_title(("L(f) / ∫∂P f dσ" if normalized else "L_A(max(0, n·x − s))") + f"   n = ({n[0]:.4g}, {n[1]:.4g})",
                 color=INK, fontsize=10, loc="left")
    return ax


def plot_crease_margin(result, ax=None):
    """The normalized crease margin min_s L(f)/∫_∂P f dσ against the crease direction θ."""
    ax = _line_axes(ax)
    deg = np.degrees(result.theta)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.plot(deg, result.margin, color="#2a78d6", lw=2)
    best = np.degrees(result.best_theta)
    ax.scatter([best], [result.best_margin], s=40, color=ACCENT if result.unstable else "#2a78d6",
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    label = "destabilizing" if result.unstable else "worst direction"
    ax.annotate(f"{label}: {result.best_margin:.3g} at θ = {best:.2f}°", (best, result.best_margin),
                xytext=(6, 8), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    # a few nearly-degenerate directions can spike; keep the bulk (and the minimum) readable
    top = float(np.nanpercentile(result.margin, 95))
    bottom = min(0.0, float(np.nanmin(result.margin)), result.best_margin)
    pad = 0.15 * (top - bottom or 1.0)
    ax.set_ylim(bottom - pad, top + pad)
    ax.set_xlabel("direction θ of the crease normal (degrees)", color=INK_2, fontsize=9)
    ax.set_title("crease stability margin  min_s L(f) / ∫∂P f dσ", color=INK, fontsize=10, loc="left")
    return ax


def plot_crease(P: Polygon, a, b, c, ax=None):
    """Draw the crease line a x + b y + c = 0 and lightly shade where f = max(0, ·) is positive."""
    a, b, c = (as_fraction(v) for v in (a, b, c))
    ax = plot_polygon(P, ax=ax, normals=False, labels=False, vertex_labels=False)
    piece = clip_halfplane(list(P.vertices), a, b, c)
    if len(piece) >= 3:
        pts = np.array([[float(q) for q in v] for v in piece])
        ax.fill(pts[:, 0], pts[:, 1], color=ACCENT, alpha=0.18, lw=0, zorder=1)
    on_line = [v for v in piece if a * v[0] + b * v[1] + c == 0]
    if len(on_line) >= 2:
        pts = np.array([[float(q) for q in v] for v in on_line])
        ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=2, zorder=5)
    value = P.L_crease(a, b, c)
    shown = str(value) if len(str(value)) <= 24 else f"{float(value):.6g}"
    c_shown = str(c) if len(str(c)) <= 16 else f"{float(c):.6g}"
    ax.set_title(f"crease max(0, {a}x + {b}y + {c_shown}):  L = {shown}", color=INK, fontsize=10, loc="left")
    return ax
