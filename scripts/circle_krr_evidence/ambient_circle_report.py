"""LaTeX report builder for the ambient-circle KRR study."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _math_number(value: float, significant_digits: int = 4) -> str:
    """Format a number for LaTeX, using powers of ten instead of e notation."""

    numeric = float(value)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    if 1.0e-3 <= magnitude < 1.0e3:
        return f"{numeric:.{significant_digits}g}"
    exponent = int(math.floor(math.log10(magnitude)))
    mantissa = numeric / 10.0**exponent
    return rf"{mantissa:.{significant_digits}g}\times10^{{{exponent}}}"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"${_math_number(float(value))}$"


def _power_of_ten(value: float) -> str:
    """Format an exact power-of-ten tuning bound for a LaTeX math cell."""

    exponent = int(round(math.log10(value)))
    if math.isclose(value, 10.0**exponent, rel_tol=1.0e-12, abs_tol=0.0):
        return rf"10^{{{exponent}}}"
    return _math_number(value)


def _figure(path: Path, *, width: str = r"\textwidth") -> str:
    """Return an includegraphics command using the artifact's absolute path."""

    return rf"\includegraphics[width={width}]{{{path.resolve().as_posix()}}}"


def _principal_angle_rows(mode_angles: dict[str, Any]) -> str:
    """Render principal-angle comparisons for LaTeX."""

    return "\n".join(
        row["comparison"]
        + " & $"
        + r",\; ".join(
            rf"{{{_math_number(angle, significant_digits=3)}}}^\circ"
            for angle in row["principal_angles_degrees"]
        )
        + r"$ \\"
        for row in mode_angles["comparisons"].values()
    )


def _maximum_angle(mode_angles: dict[str, Any], phrase: str) -> float:
    for row in mode_angles["comparisons"].values():
        if phrase in row["comparison"]:
            return float(row["maximum_angle_degrees"])
    raise KeyError(f"no principal-angle comparison contains {phrase!r}")


def _compile_latex(source_path: Path, output_path: Path) -> None:
    """Compile the report in an isolated temporary directory."""

    with tempfile.TemporaryDirectory(
        prefix="ambient_circle_latex_", dir=output_path.parent
    ) as build:
        result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build}",
                str(source_path.resolve()),
            ],
            cwd=source_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        generated_pdf = Path(build) / source_path.with_suffix(".pdf").name
        if result.returncode != 0 or not generated_pdf.is_file():
            output = (result.stdout + "\n" + result.stderr).strip()
            raise RuntimeError(f"LaTeX compilation failed:\n{output[-6000:]}")
        shutil.copy2(generated_pdf, output_path)


def write_ambient_circle_report(
    *, output_path: Path, run_dir: Path, summary: dict[str, Any]
) -> None:
    """Write a self-contained LaTeX source and compile the study note."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = output_path.with_suffix(".tex")
    protocol = summary["protocol"]
    semi = summary["semicircle"]
    full = summary["full_circle"]
    semi_lb = semi["endpoints"]["lb"]
    semi_rbf = semi["endpoints"]["rbf"]
    family = semi["families"]
    full_values = full["endpoints"]
    audit = summary["audit"]

    source = r"""\documentclass[11pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,caption,placeins,microtype}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Ambient Euclidean diffusion-map versus RBF KRR on circles}}
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\title{Tuned ambient-Euclidean diffusion-map versus RBF KRR\\
on the semicircle and full circle}
\date{}
\begin{document}
\maketitle

\begin{abstract}
For a finite-sample kernel-ridge-regression (KRR) map $S_A$, with $A$ denoting either a
diffusion-map (DM) kernel or a Gaussian radial-basis-function (RBF) kernel, and a target
$f$ in a chosen function space $W$, orthogonal projection gives
$\|f-S_Af\|^2=B_{A,W}^2+L_{A,W}^2$.  Here $B_{A,W}$ is error remaining inside $W$ and
$L_{A,W}$ is leakage of the fitted function into $W^\perp$.  Consequently a method can
overtake another along a target family when its leakage collapses while its in-class term
stays bounded, so that the complete squared-error stack becomes smaller.  Kernel-mode
alignment provides a geometric precursor, not a bound: a kernel eigenspace close to $W$
offers directions with little geometric leakage, while the realized ridge filter and all
higher modes determine the actual $L$.

This mechanism is tested on a semicircle and a full circle, with both kernels using ambient
Euclidean chord distance in $\mathbb R^2$.  On the semicircle LB endpoint, the maximum
principal angle is <<SEMI_DM_LB_ANGLE>> degrees for DM and <<SEMI_RBF_LB_ANGLE>> degrees for
RBF; DM has the smaller median leakage and wins all 12 targets.  At the RBF endpoint, the
corresponding even-mode angles are <<SEMI_DM_RBF_ANGLE>> and <<SEMI_RBF_RBF_ANGLE>> degrees;
RBF has the smaller median leakage and wins all 12 targets.  Along the paired families,
<<CROSSINGS>> of <<FAMILY_COUNT>> paths cross and the change in RBF total error follows its
leakage collapse.  On the full circle, rotational symmetry makes both first-eight
nonconstant kernel-mode spaces coincide with the periodic LB space to within
<<FULL_MAX_ANGLE>> degrees.  There the RBF advantage is not a space-mismatch transition:
RBF wins <<FULL_RBF_WINS>> of <<FULL_COUNT>> targets because both its in-class and leakage
norms are smaller at the deliberately resolved 13-point design.
\end{abstract}

\section{Setup and independently tuned KRR maps}
Let $x(\theta)=(\cos\theta,\sin\theta)$.  For method $A$, training points $X$, selected
kernel scale, and selected ridge $\lambda_A$, the realized map is
\[
  (S_Af)(x)=k_A(x,X)(K_A+\lambda_A I)^{-1}f(X).
\]
Thus $S_A$ includes the sample and the independently validation-selected hyperparameters.
The intrinsic coordinate $\theta$ is used only to sample the curve and evaluate the uniform
$L^2$ measure; neither kernel receives it.  RBF uses
\[
 k_{\rm RBF}(\theta,\phi)=
 \exp\!\left[-\frac{\|x(\theta)-x(\phi)\|^2}{2\ell^2}\right].
\]
The DM raw affinity is $\exp[-\|x(\theta)-x(\phi)\|^2/(4\varepsilon)]$.
Both methods therefore use the same ambient distance, but only DM estimates normalizing
factors from the finite sample.

\begin{center}
\begin{tabular}{lcc}
\toprule
quantity & semicircle & full circle \\
\midrule
training points & <<SEMI_TRAIN>> & <<FULL_TRAIN>> \\
validation points & <<SEMI_VALID>> & <<FULL_VALID>> \\
dense test points & <<TEST_COUNT>> & <<TEST_COUNT>> \\
shared search & \multicolumn{2}{c}{<<SEARCH>>} \\
shared scale range & \multicolumn{2}{c}{$\varepsilon,\ell\in[<<BANDWIDTH_LOW>>,<<BANDWIDTH_HIGH>>]$} \\
shared ridge range & \multicolumn{2}{c}{$\lambda\in[<<RIDGE_LOW>>,<<RIDGE_HIGH>>]$} \\
additional RBF-only sweep & \multicolumn{2}{c}{$\ell=<<ELL>>$, <<RIDGES>> ridges in
$[<<FIXED_RIDGE_LOW>>,<<FIXED_RIDGE_HIGH>>]$} \\
\bottomrule
\end{tabular}
\end{center}

The semicircle uses 1024 endpoints-inclusive equispaced training points and 1023 midpoints
for validation.  The full circle is far easier because it is periodic and translation
invariant: at 1024 points both methods reach a floating-point floor that hides their
difference.  Thirteen periodic training points and 13 half-shifted validation points place
the tuned DM error just below $10^{-10}$, keeping the comparison numerically resolved.  The
common search is a $9\times9$ logarithmic grid followed by a four-start Nelder--Mead
refinement.  The fixed $\ell=0.2$ sweep is additional RBF tuning only.  It protects against
a narrow ridge optimum at the same length scale used to construct the RBF target space; DM
uses only the common search.  In each case the lower validation error selects the map before
the independent test set is evaluated.

\section{Target spaces, kernel modes, and exact error identity}

\paragraph{Target-family construction.}
The target spaces separate boundary-compatible Laplace--Beltrami (LB) behavior from modes
native to an ambient Gaussian integral operator.  On the semicircle,
\[
 W_{\rm LB}=\operatorname{span}\{\sqrt2\cos\theta,\sqrt2\cos3\theta,
 \sqrt2\cos5\theta,\sqrt2\cos7\theta\}.
\]
The eight-dimensional $W_{\rm RBF}$ consists of the first eight reflection-even modes
$\psi_j$ of the continuum ambient-RBF operator at $\ell=0.2$.  A seeded unit direction in
$W_{\rm LB}$ defines $u_i$.  To define the RBF direction, choose the mode window $J_i$ and
write $v_i=\sum_{j\in J_i}c_{ij}\psi_j$.  In endpoint order, the twelve windows are
\[
\begin{gathered}
(1,3,5),\ (1,3,5,7),\ (1,3,5,7,9),\ (3,5,7),\\
(3,5,7,9),\ (3,5,7,9,11),\ (5,7,9),\ (5,7,9,11),\\
(5,7,9,11,13),\ (7,9,11),\ (7,9,11,13),\ (7,9,11,13,15).
\end{gathered}
\]
The coefficient vector is the unit right
nullvector of the boundary-jet matrix
\[
 \left[\psi_j^{(2r-1)}(0)\right]_{
  r=1,\ldots,|J_i|-1;\ j\in J_i},
\]
oriented to make $v_i(0)>0$; $v_i$ is then continuum-$L^2$ normalized.  Thus the first
$|J_i|-1$ odd boundary derivatives vanish.  All endpoints are subsequently normalized with
the dense-test norm
\[
 \|g\|_{\mathcal T}=\left(\frac1{N_{\mathcal T}}
 \sum_{q=1}^{N_{\mathcal T}}g(\theta_q)^2\right)^{1/2},
 \qquad N_{\mathcal T}=65{,}536,
\]
where the $\theta_q$ are the fixed uniform-random test points.  With this normalization,
\[
 f_i(s)=\frac{\cos(\pi s/2)u_i+\sin(\pi s/2)v_i}
 {\|\cos(\pi s/2)u_i+\sin(\pi s/2)v_i\|_{\mathcal T}},\qquad 0\le s\le1.
\]
Reflection parity makes $W_{\rm LB}$ odd and $W_{\rm RBF}$ even, so their four principal
angles are exactly $90^\circ$ and the family plane is nondegenerate.  On the full circle,
$W_{\rm LB}$ contains the first four periodic sine/cosine pairs.  Rotational invariance gives
the ambient RBF operator the same Fourier eigenspaces, so an LB--RBF family would be
degenerate and is omitted.

\begin{figure}[!htbp]
\centering
<<FIGURE_ONE>>
\caption{The normalized targets used in the study: 12 semicircle LB endpoints, 12
semicircle RBF endpoints, one paired family at $s=0.8$, and 12 full-circle LB targets.}
\label{fig:targets}
\end{figure}
\FloatBarrier

\paragraph{Subspace alignment.}
Before attributing errors to leakage, the kernel modes must be compared with the target
spaces using the correct spectral object.  For a typical endpoint-selected bandwidth, let
\[
 K_Au_j=\mu_j u_j,\qquad
 \psi_j(z)=\frac{k_A(z,X)u_j}{\mu_j}.
\]
These Nystr\"om-extended Gram-matrix eigenvectors are the finite-sample DM/RBF kernel modes.
The typical bandwidth is the exponentiated median log bandwidth among the 12 relevant
endpoint fits.  Ridge values are recorded but do not enter the kernel eigenproblem.

For semicircle LB targets, the first eight unrestricted kernel modes are compared with the
four-dimensional $W_{\rm LB}$; those eight contain the four relevant reflection-odd modes.
For RBF targets, the first eight reflection-even kernel modes are compared with
$W_{\rm RBF}$.  For the full circle, the constant is removed and the first eight
nonconstant modes are compared with the periodic eight-dimensional LB space.  With weighted
orthonormal bases $Q_1,Q_2$, the principal angles are
\[
 \theta_j=\arccos\sigma_j(Q_1^\top W_{\mathcal Q}Q_2).
\]

\begin{center}
\scriptsize
\begin{tabular}{lp{4.05in}}
\toprule
semicircle comparison & principal angles in degrees (ascending) \\
\midrule
<<SEMI_ANGLE_ROWS>>
\bottomrule
\end{tabular}

\vspace{0.8em}
\begin{tabular}{lp{4.05in}}
\toprule
full-circle comparison & principal angles in degrees (ascending) \\
\midrule
<<FULL_ANGLE_ROWS>>
\bottomrule
\end{tabular}
\normalsize
\end{center}

\begin{figure}[!htbp]
\centering
<<FIGURE_MODES>>
\caption{Direct mode check.  Dashed curves are target-space modes and solid curves are
Nystr\"om-extended finite-sample kernel modes.  For readability, the kernel basis is rotated
only within its own subspace to its closest target basis and the modes are vertically offset;
this rotation leaves all principal angles unchanged.  The figure shows the four target-facing
canonical pairs for semicircle LB and all eight pairs for the other comparisons.}
\label{fig:modes}
\end{figure}
\FloatBarrier

\paragraph{Error decomposition.}
The mode angles are only geometric diagnostics, so the claimed mechanism is tested with the
realized fitted functions.  On the independent dense test rule $\mathcal T$, let $P_W$ be
orthogonal projection onto the endpoint class or, for a family, onto its two-dimensional
plane.  Since $f\in W$,
\[
 f-S_Af=\underbrace{f-P_WS_Af}_{e^{\rm in}_{A,W}\in W}
       -\underbrace{(I-P_W)S_Af}_{\ell_{A,W}\in W^\perp}.
\]
Therefore, with
\[
 E_A=\|f-S_Af\|_{\mathcal T},\qquad
 B_{A,W}=\|e^{\rm in}_{A,W}\|_{\mathcal T},\qquad
 L_{A,W}=\|\ell_{A,W}\|_{\mathcal T},
\]
the exact test-rule identity is
\[
 \boxed{E_A^2=B_{A,W}^2+L_{A,W}^2}.
\]
The code evaluates the direct residual as an independent certificate and plots the
orthogonally recomposed $E=(B^2+L^2)^{1/2}$.  Across all fits, the largest absolute
squared-identity defect is <<MAX_DEFECT>>, the largest target projection defect is
<<MAX_PROJECTION_DEFECT>>, and the largest observed recomposed $L/E$ is
<<MAX_LEAKAGE_RATIO>>; there are <<LEAKAGE_VIOLATIONS>> cases with $L>E$.

\section{Semicircle endpoint evidence}
Figure~\ref{fig:semi-endpoints} separates magnitude from composition.  The upper panels show
only total error $E$; the lower panels show the bounded leakage-energy fraction $L^2/E^2$.
Its complement is $B^2/E^2$.  This avoids the misleading visual overlap produced when $E$
and an almost equal $L$ are drawn with different markers on the same logarithmic axis.

DM wins all 12 LB endpoints.  Its median total error and leakage norm are <<SEMI_LB_DM_E>> and
<<SEMI_LB_DM_L>>, versus <<SEMI_LB_RBF_E>> and <<SEMI_LB_RBF_L>> for RBF.  The median squared
leakage shares are <<SEMI_LB_DM_SHARE>> for DM and <<SEMI_LB_RBF_SHARE>> for RBF.  This agrees
with the mode diagnostic: the DM first-eight space is closer to the LB target space, especially
in its largest principal angle.

At the RBF endpoint, RBF wins all 12 targets.  Its median total error and leakage norm are
<<SEMI_RBF_RBF_E>> and <<SEMI_RBF_RBF_L>>, versus <<SEMI_RBF_DM_E>> and <<SEMI_RBF_DM_L>> for
DM; the median squared leakage shares are <<SEMI_RBF_RBF_SHARE>> and <<SEMI_RBF_DM_SHARE>>.
Here the RBF even-mode space is much closer to $W_{\rm RBF}$ than the DM even-mode space.
Thus the endpoint ordering and the absolute leakage ordering both reverse with the target
space, providing the two ends needed for a crossover.

The RBF-on-RBF endpoint median is itself at the double-precision floor, so its last digits and
its leakage share should not be interpreted quantitatively.  The robust conclusion is the
ordering against the DM median, which is roughly two orders of magnitude larger, together
with the resolved leakage collapse before the endpoint.

\begin{figure}[!htbp]
\centering
<<FIGURE_TWO>>
\caption{Semicircle endpoints.  Top: independently tuned total errors $E$.  Bottom:
leakage-energy shares $L^2/E^2\in[0,1]$; the complementary share is $B^2/E^2$.}
\label{fig:semi-endpoints}
\end{figure}
\FloatBarrier

\section{Semicircle LB--RBF crossover mechanism}
For family $i$, RBF is better exactly when
\[
 B_{{\rm RBF},i}(s)^2+L_{{\rm RBF},i}(s)^2
 <B_{{\rm DM},i}(s)^2+L_{{\rm DM},i}(s)^2.
\]
Figure~\ref{fig:families} shows family <<REPRESENTATIVE_FAMILY>>, selected because its leakage
reduction is nearest the geometric-median reduction among crossing families, not because it
has the largest effect.  The vertical line is the zero of
$E_{\rm RBF}(s)-E_{\rm DM}(s)$, linearly interpolated within the sampled interval that brackets
the sign change; here $s_\star=<<REPRESENTATIVE_CROSSING>>$.

Across all families, <<CROSSINGS>> of <<FAMILY_COUNT>> cross.  The median
$\operatorname{corr}(\log E_{\rm RBF},\log L_{\rm RBF})$ is <<CORRELATION>>, and the median
leakage reduction from $s=0.975$ to $s=1$ is <<COLLAPSE>>-fold.  The heat map condenses every
path as $\log_{10}(E_{\rm RBF}/E_{\rm DM})$.  These are the numerical signatures of the
theoretical explanation: as the target rotates into the well-aligned RBF space, RBF leakage
falls rapidly while the orthogonal in-class contribution supplies a residual floor; the
quadrature sum then passes the DM error.  The principal angles explain why the two endpoints
favor different spaces, while the realized decomposition identifies the actual moving term.

\begin{figure}[!htbp]
\centering
<<FIGURE_THREE>>
\caption{One representative family: total and orthogonal component norms on the full path
and near $s=1$, followed by the all-family signed error-ratio heat map.}
\label{fig:families}
\end{figure}
\FloatBarrier

\section{Full circle: a common periodic LB space}
On the full circle, both ambient kernels are rotationally invariant on the equispaced sample.
Their first eight nonconstant Nystr\"om mode spaces agree with the common periodic LB target
space to nearly numerical precision, as Figure~\ref{fig:modes} confirms.  There is therefore
no meaningful LB--RBF family and no subspace crossover to explain.

At 13 training and 13 validation points, RBF wins <<FULL_RBF_WINS>> of <<FULL_COUNT>> targets.
Its median total error is <<FULL_RBF_ERROR>>, compared with <<FULL_DM_ERROR>> for DM.  The
median in-class norms are <<FULL_RBF_IN>> and <<FULL_DM_IN>>, and the median leakage norms are
<<FULL_RBF_LEAK>> and <<FULL_DM_LEAK>>, respectively.  Thus RBF's absolute $E$, $B$, and $L$
are all smaller, as Figure~5 shows directly.  The close mode-space alignment means the
remaining difference comes from the finite-sample realized maps---including the DM
normalization estimate and the independently selected ridge filters---rather than from an
LB/RBF target-space mismatch.  Once geometric leakage is small for both, the lower absolute
in-class and leakage levels decide the comparison, and RBF is better.

\begin{figure}[!htbp]
\centering
<<FIGURE_FOUR>>
\caption{Full-circle common-LB endpoints.  For each method, the solid line is total error
$E$, the dashed line is in-class error $B$, and the dotted line is leakage $L$.}
\end{figure}

\end{document}
"""
    fixed_bounds = protocol["tuning"]["rbf_fixed_sweep"]["ridge_bounds"]
    replacements = {
        "<<SEMI_TRAIN>>": str(protocol["semi_circle"]["n_train"]),
        "<<FULL_TRAIN>>": str(protocol["full_circle"]["n_train"]),
        "<<SEMI_VALID>>": str(protocol["semi_circle"]["n_valid"]),
        "<<FULL_VALID>>": str(protocol["full_circle"]["n_valid"]),
        "<<TEST_COUNT>>": str(protocol["test_count"]),
        "<<ELL>>": f"{protocol['tuning']['rbf_fixed_sweep']['ambient_lengthscale']:.2g}",
        "<<RIDGES>>": str(protocol["tuning"]["rbf_fixed_sweep"]["ridge_count"]),
        "<<SEARCH>>": protocol["tuning"]["shared"].replace("--", "-{}-"),
        "<<BANDWIDTH_LOW>>": _power_of_ten(summary["config"]["bandwidth_bounds"][0]),
        "<<BANDWIDTH_HIGH>>": _power_of_ten(summary["config"]["bandwidth_bounds"][1]),
        "<<RIDGE_LOW>>": _power_of_ten(summary["config"]["ridge_bounds"][0]),
        "<<RIDGE_HIGH>>": _power_of_ten(summary["config"]["ridge_bounds"][1]),
        "<<FIXED_RIDGE_LOW>>": _power_of_ten(fixed_bounds[0]),
        "<<FIXED_RIDGE_HIGH>>": _power_of_ten(fixed_bounds[1]),
        "<<FAMILY_COUNT>>": str(family["count"]),
        "<<SEMI_ANGLE_ROWS>>": _principal_angle_rows(semi["krr_mode_angles"]),
        "<<FULL_ANGLE_ROWS>>": _principal_angle_rows(full["krr_mode_angles"]),
        "<<SEMI_DM_LB_ANGLE>>": _number(
            _maximum_angle(semi["krr_mode_angles"], "DM (LB-tuned)")
        ),
        "<<SEMI_RBF_LB_ANGLE>>": _number(
            _maximum_angle(semi["krr_mode_angles"], "RBF (LB-tuned)")
        ),
        "<<SEMI_DM_RBF_ANGLE>>": _number(
            _maximum_angle(semi["krr_mode_angles"], "DM (RBF-tuned)")
        ),
        "<<SEMI_RBF_RBF_ANGLE>>": _number(
            _maximum_angle(semi["krr_mode_angles"], "RBF (RBF-tuned)")
        ),
        "<<FULL_MAX_ANGLE>>": _number(
            max(
                _maximum_angle(full["krr_mode_angles"], "DM,"),
                _maximum_angle(full["krr_mode_angles"], "RBF,"),
            )
        ),
        "<<SEMI_LB_DM_E>>": _number(semi_lb["dm_median_population_error"]),
        "<<SEMI_LB_RBF_E>>": _number(semi_lb["rbf_median_population_error"]),
        "<<SEMI_LB_DM_L>>": _number(semi_lb["dm_median_leakage"]),
        "<<SEMI_LB_RBF_L>>": _number(semi_lb["rbf_median_leakage"]),
        "<<SEMI_LB_DM_SHARE>>": f"{100.0 * semi_lb['dm_median_leakage_share']:.3f}\\%",
        "<<SEMI_LB_RBF_SHARE>>": f"{100.0 * semi_lb['rbf_median_leakage_share']:.3f}\\%",
        "<<SEMI_RBF_DM_E>>": _number(semi_rbf["dm_median_population_error"]),
        "<<SEMI_RBF_RBF_E>>": _number(semi_rbf["rbf_median_population_error"]),
        "<<SEMI_RBF_DM_L>>": _number(semi_rbf["dm_median_leakage"]),
        "<<SEMI_RBF_RBF_L>>": _number(semi_rbf["rbf_median_leakage"]),
        "<<SEMI_RBF_DM_SHARE>>": f"{100.0 * semi_rbf['dm_median_leakage_share']:.3f}\\%",
        "<<SEMI_RBF_RBF_SHARE>>": f"{100.0 * semi_rbf['rbf_median_leakage_share']:.3f}\\%",
        "<<REPRESENTATIVE_FAMILY>>": str(family["representative_family_index"]),
        "<<REPRESENTATIVE_CROSSING>>": (
            "n/a"
            if family["representative_exact_crossing"] is None
            else f"{float(family['representative_exact_crossing']):.6f}"
        ),
        "<<CROSSINGS>>": str(family["crossing_count"]),
        "<<CORRELATION>>": f"{family['median_rbf_error_leakage_log_correlation']:.6f}",
        "<<COLLAPSE>>": _number(family["median_leakage_collapse"]),
        "<<FULL_RBF_WINS>>": str(full_values["rbf_win_count"]),
        "<<FULL_COUNT>>": str(full_values["count"]),
        "<<FULL_RBF_ERROR>>": _number(full_values["rbf_median_population_error"]),
        "<<FULL_DM_ERROR>>": _number(full_values["dm_median_population_error"]),
        "<<FULL_RBF_IN>>": _number(full_values["rbf_median_in_class_error"]),
        "<<FULL_DM_IN>>": _number(full_values["dm_median_in_class_error"]),
        "<<FULL_RBF_LEAK>>": _number(full_values["rbf_median_leakage"]),
        "<<FULL_DM_LEAK>>": _number(full_values["dm_median_leakage"]),
        "<<MAX_DEFECT>>": _number(audit["maximum_decomposition_defect"]),
        "<<MAX_PROJECTION_DEFECT>>": _number(audit["maximum_target_projection_defect"]),
        "<<MAX_LEAKAGE_RATIO>>": f"{audit['maximum_leakage_to_total_ratio']:.12f}",
        "<<LEAKAGE_VIOLATIONS>>": str(audit["leakage_exceeds_total_count"]),
        "<<FIGURE_ONE>>": _figure(run_dir / "target_ensembles.png"),
        "<<FIGURE_MODES>>": _figure(run_dir / "kernel_mode_comparison.png"),
        "<<FIGURE_TWO>>": _figure(run_dir / "semicircle_endpoints.png"),
        "<<FIGURE_THREE>>": _figure(run_dir / "semicircle_family_focus_and_summary.png"),
        "<<FIGURE_FOUR>>": _figure(
            run_dir / "fullcircle_lb_endpoints.png", width=r"0.82\textwidth"
        ),
    }
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    source_path.write_text(source, encoding="utf-8")
    _compile_latex(source_path, output_path)
