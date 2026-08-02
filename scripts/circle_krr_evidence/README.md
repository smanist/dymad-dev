# Ambient-Euclidean semicircle and full-circle KRR evidence

Run from the repository root:

```shell
python scripts/circle_krr_evidence/circle_krr_evidence_cli.py
```

The intrinsic coordinate only indexes the curve and the uniform integration
measure. DM means diffusion map. Both DM and RBF receive the ambient embedding `(cos(theta),
sin(theta))`, so every kernel uses its Euclidean chord distance. There is no
arc-distance kernel and no bandwidth rescaling.

The default uses 1,024 equispaced semicircle training points with 1,023
midpoints for validation. The full circle
uses 13 periodic training points and 13 half-shifted validation points: this
places the DM endpoint error just below `1e-10`, avoiding the precision floor
seen at 1,024 points. Both geometries use 65,536 seeded uniform-random test
points. Every target is normalized on that dense test rule. DM uses a 9-by-9 log grid followed by four-start
Nelder--Mead. Its bandwidth/length-scale range is `1e-4` to `1e2` and its
ridge range is `1e-16` to `1e1`. RBF takes the better of that search and an
additional fixed ambient `ell=0.2` sweep over 65 ridges from `1e-16` to
`1e-8`.

For the semicircle, the script compares 12 LB and 12 reflection-even ambient
RBF endpoints, then independently tunes every method on all 12 paired
LB--RBF families at 38 values of `s`. Its exact test-rule
decomposition is `E^2=B^2+L^2`, with an in-class term `B` and leakage `L`.
For the full circle, it audits the common periodic target space and does not
construct a degenerate family: both methods are compared only on the common
12-direction LB ensemble. The `krr_mode_angles.csv` artifact compares
principal angles among the LB/RBF target spaces and genuine finite-sample
kernel modes: eigenvectors of the selected kernel Gram matrix, extended to
the quadrature rule by the Nyström formula. It does not use singular vectors
of the regularized interpolation map. The `kernel_mode_comparison.png`
artifact plots the compared modes directly. Angles remain a geometric
diagnostic; the exact realized-fit decomposition is needed to assess leakage.

The implementation is deliberately small: `ambient_circle_study.py` contains
the target construction, tuning, and diagnostics;
`ambient_circle_plots.py` contains exactly the five report figures;
`ambient_circle_report.py` contains the LaTeX note; and
`circle_krr_evidence_cli.py` is only the command-line adapter. Generic KRR,
kernel, kernel-eigensystem, and tuning operations use DyMAD's public APIs.

The run directory contains `summary.json`, selected-model and tuning CSVs,
decomposition/family-diagnostic/KRR-mode-angle CSVs, and only the five figures
included in the report: `target_ensembles.png`,
`kernel_mode_comparison.png`, `semicircle_endpoints.png`,
`semicircle_family_focus_and_summary.png`, and
`fullcircle_lb_endpoints.png`.
The default writes a LaTeX source and its compiled PDF note to
`output/pdf/circle_krr_ambient/circle_krr_ambient_study_note.tex` and
`output/pdf/circle_krr_ambient/circle_krr_ambient_study_note.pdf`.  The report
requires a local `latexmk`/LaTeX installation. Use
`--no-report` or `--no-plot` for a fast workflow check. Generated run artifacts
live under `scripts/circle_krr_evidence/runs/` and are ignored by Git.

Use `--semi-n-train`/`--semi-n-valid` and
`--full-n-train`/`--full-n-valid` to change one geometry without changing the
other. `--n-train` remains a shorthand that sets both training counts.
