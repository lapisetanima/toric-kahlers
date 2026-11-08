# The stability wall of the pentagons P(s, t)

Scripts for [docs/pentagon_walls.md](../../docs/pentagon_walls.md). Run them from the repository root; they need python-flint.

| script | what it does |
|---|---|
| `creasealg.py` | Donaldson's functional on simple creases as a vertex sum, symbolic in (s, t, weights); run directly to check it exactly against `Polygon.L_crease` |
| `wallpoly.py s t [S]` | at one shape: eliminate the crease, print the minimal polynomial of the threshold label m* (chamber S from the cached balanced crease, or given as e.g. `0,1,2`) |
| `wallsample.py S s a:b:D out.json` | exact monic degree-8 polynomials of the critical value at t = k/D, a ≤ k < b, in parallel |
| `wallinterp.py prefix samples.json ...` | interpolate Φ(t, m) from the samples (degree in t found mod p, then exact), verify at every sample, save to `prefix.json` |
| `check_slice.py research/pentagon_wall/data` | compare both sheets at s = 1/10 with the cached thresholds and draw the figure |

`data/` contains the samples and the two sheets at s = 1/10: `Phi_012_s10.json` and `Phi_12_s10.json`.
The key `"i,j"` is the integer coefficient of tⁱ mʲ. The reference thresholds are in
`notebooks/data/cp2_2_thresholds.json`, computed by `extremal.survey.thresholds`.
