# Conventions

All formulas in the code follow this page. Coordinates on the polygon are `x = (x1, x2)`
(written `x, y` in code).

## Polygon

* `P = { x : l_i(x) = <u_i, x> - lambda_i >= 0 }`, `u_i` the **primitive inward** normal of edge `i`.
* Vertices are stored counter-clockwise, as exact `Fraction`s. Edge `i` runs from vertex `i` to vertex `i+1`.
* Edge `i` has primitive direction `e_i` (integer, coprime) and **lattice length** `L_i`,
  so `v_{i+1} - v_i = L_i e_i`. For a CCW polygon, `u_i = rot90(e_i) = (-e_i[1], e_i[0])`.
* **Delzant**: at every vertex, `|det(e_{i-1}, e_i)| = 1` (equivalently `|det(u_{i-1}, u_i)| = 1`).

## Measures

* `dmu` = Lebesgue measure on P.
* `dsigma` on edge `i` is defined by `dl_i ∧ dsigma = ± dmu`. It equals the lattice-length measure:
  `∫_{edge i} f dsigma = L_i ∫_0^1 f(v_i + t (v_{i+1} - v_i)) dt`.

## Metrics

* Symplectic potential `u = u_G + h`, with Guillemin's potential `u_G = 1/2 Σ l_i log l_i`,
  `h` smooth on the closed polygon, `H = Hess(u) > 0` in the interior.
* Riemannian metric on the open dense torus orbit, in action–angle coordinates `(x, θ)`, `θ_i ∈ [0, 2π)`:

  `g = H_ij dx_i dx_j + (H^{-1})^{ij} dθ_i dθ_j`.

  With the factor `1/2` in `u_G` the metric closes up smoothly over the edges with angle period `2π`.

## Curvature

Write `K = H^{-1}`, `V^b = -Σ_j ∂_j K^{bj}`, and the 2×2 matrix `M_{eb} = ∂_e V^b`.

* **Riemannian scalar curvature** `s = tr M = -Σ_{ij} ∂_i ∂_j K^{ij}`.
* **Scalar curvature `S` (Abreu normalization)**, used everywhere else in the code:
  `S = s / 2 = -1/2 Σ_{ij} ∂_i ∂_j K^{ij}`.
* **Ricci tensor**: `Ric = R (dξ² + dθ²)` in log-complex coordinates, and
  `|Ric|²_g = 1/2 tr(M²)` (Riemannian norm over all 4 real dimensions).

Sanity check: for the standard simplex (CP², Fubini–Study) `M = 6·Id`, so `S = 6`, `s = 12`,
`Ric = 3g`, and `|Ric|² = 36`.

## Donaldson's functional and the invariants

* `L_A(f) = ∫_{∂P} f dsigma - ∫_P A f dmu`.
* Integration by parts (Donaldson): for any symplectic potential `u` as above,

  `1/2 ∫_P K^{ij} f_ij dmu = L_{S(u)}(f)`.

* **Average scalar curvature** `Ŝ = sigma(∂P) / mu(P)` (that is, `L_Ŝ(1) = 0`).
* **Extremal affine function** `A_ext`: the unique affine `A` with `L_A(f) = 0` for every affine `f`.
* **Futaki invariant**: `F(f) = L_Ŝ(f)` for affine `f`; it depends only on the linear part of `f`.
  The code reports the vector `(F(x1), F(x2))`. Equivalently
  `F = sigma(∂P) · (boundary barycenter − area barycenter)`.
* **Futaki kernel** of a metric: `S(x) − Ŝ`. For every metric, `F(f) = ∫_P (S − Ŝ) f dmu`.
* A convex piecewise-linear `f` with `L_{A_ext}(f) < 0` certifies that `P` is relatively K-unstable,
  so no extremal metric exists.

## Stability via simple creases (`stability.py`)

* A **simple crease** is `f = max(0, n·x − s)`.
* **Creases suffice in dimension 2.** Wang–Zhou (Adv. Math. 2011; Thm 3.4 and Cor. 3.5 in B. Zhou's ANU thesis, 2010):
  for any convex polygon and any affine `A` (no sign condition), if `L_A(u) < 0` for some convex `u` then
  `L_A(f) < 0` for some simple crease; and if `L_A >= 0` on convex functions but vanishes on a non-affine one,
  it vanishes on a simple crease. Donaldson (JDG 2002, Prop. 5.3.1) had the latter for `A > 0`.
  Donaldson (2002, Prop. 6.1): a crease with `L <= 0` can be replaced by a **rational** one.
* **One-variable reduction.** For fixed `n`, `L(s) = ∫_s^{t_max} (t − s) dρ(t)`, where on `[t_min, t_max]`
  (`t = n·x`) the measure `ρ` has density `Σ_{edges crossed} L_i/|Δt_i| − w(t)(A(p) + A(q))/2`
  (`p, q` the ends of the chord `{n·x = t}`, `w = |q − p|/|n|`), plus mass `L_i` at `t_max` for an edge
  parallel to the crease. `ρ` is quadratic between vertex values, so `L` is piecewise quartic in `s`.
  `ρ` has zero mass and zero first moment, so `L(t_min) = L(t_max) = 0`.
* **Margin.** For each direction, `min_s L(f) / ∫_∂P f dσ`, with `f` the side of the crease vanishing at the
  area barycenter (Donaldson's normalization). It is scale-invariant and jumps at `s = n·c`.
* **Certificates.** Instability is reported only for a rational crease whose `L` was computed exactly.
