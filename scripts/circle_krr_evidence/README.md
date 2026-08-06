# Ambient-circle KRR figure study

The study is intentionally figure-only. Run it from the repository root:

```bash
python scripts/circle_krr_evidence/circle_krr.py
```

The default writes the nine PNGs used by the note to
`scripts/circle_krr_evidence/runs/ambient_circle_centered_composite_l2/`.
`--quick` runs a small smoke protocol, `--workers` is capped at four, and
`--output-dir` selects another figure directory.

Both diffusion-map (DM) and RBF KRR use ambient Euclidean chord distances in
`R^2`. The semicircle has 1,024 training points and midpoint validation; the
full circle has 13 periodic training points and half-shifted validation. The
continuum norm uses composite Gauss-Legendre quadrature on the semicircle and
a periodic trapezoidal rule on the full circle.

Each KRR map is independently tuned by a 9-by-9 logarithmic bandwidth/ridge
grid followed by multi-start Nelder-Mead. RBF additionally checks the fixed
ambient length scale `ell=0.2` over 65 ridge values. The selected maps are used
directly to compute the orthogonal identity `E^2 = B^2 + L^2`, the family
crossover curves, and the kernel-mode principal angles shown in the figures.

The 12 LB, 12 RBF, and 12 full-circle label coefficient vectors are literal
data in `label_coefficients.json`. The four semicircle RBF modes also receive a
deterministic sign convention, so the same coefficients define the same
labels across eigensolver sign choices and operating systems. Tuning uses seed
zero.

The static LaTeX source remains at
`output/pdf/circle_krr_ambient/circle_krr_ambient_study_note.tex`; there is no
Python LaTeX generator. Compile it after regenerating the figures with:

```bash
cd output/pdf/circle_krr_ambient
pdflatex circle_krr_ambient_study_note.tex
pdflatex circle_krr_ambient_study_note.tex
```
