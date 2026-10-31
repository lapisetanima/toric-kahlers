# extremal

Toric Kähler geometry on Delzant polygons. Give the vertices of a polygon; get exact invariants
(A_ext, the Futaki invariant, Donaldson's functional L), metrics from symplectic potentials, their
curvature as functions of the momentum coordinates, and plots.

Formulas and normalizations: [CONVENTIONS.md](CONVENTIONS.md).

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Open [notebooks/01_explore_polygon.ipynb](notebooks/01_explore_polygon.ipynb) and select the `.venv` interpreter as the kernel.

## Quick tour

```python
from extremal import Polygon, ToricMetric, x, y
from extremal import viz

P = Polygon([(1, 0), (3, 0), (0, 3), (0, 1)], name="CP² blown up at a point")
P.is_delzant, P.S_hat, P.A_ext, P.futaki      # exact rationals
P.L(x**2 + y**2)                              # L_{A_ext}(f) for polynomial f
P.L_crease(1, 1, -2)                          # L_{A_ext}(max(0, x + y - 2))

g = ToricMetric(P, h=x**3 / 30)               # u = 1/2 Σ l log l + h
g.scalar_curvature(1.0, 1.0)
viz.plot_fields(g, names=("S", "S - A_ext", "|Ric|^2"))
```

## Layout

| module | contents |
|---|---|
| `exact.py` | rational helpers, exact polynomial integration on segments, exact linear solve |
| `integrate.py` | exact polynomial integrals over polygons (Green's theorem), half-plane clipping |
| `polygon.py` | `Polygon` (lazy, cached invariants), `Edge`, `Affine` |
| `models.py` | CP², Hirzebruch F_n, CP¹×CP¹, the Wang–Zhou nonagon, random Delzant polygons by blow-ups |
| `metric.py` | `ToricMetric`: S, \|Ric\|², Futaki kernel S − Ŝ, extremal defect S − A_ext |
| `quadrature.py` | Gauss quadrature on polygons, for checking identities numerically |
| `stability.py` | exact crease profiles `CreaseProfile`, `crease_search` with exact rational certificates |
| `viz.py` | polygon and field plots |
