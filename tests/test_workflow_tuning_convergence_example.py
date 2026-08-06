import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest


def _run_tuning_convergence_cli(
    tmp_path: Path,
    case_name: str,
    extra_args: list[str],
    *,
    levels: str = "8,16",
    n_val: int = 16,
    n_test: int = 32,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_root = tmp_path / case_name
    output_dir = output_root / "laplace_neumann_m2_k2"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_root),
            "--target",
            "laplace_neumann_m2_k2",
            "--levels",
            levels,
            "--trials",
            "1",
            "--n-val",
            str(n_val),
            "--n-test",
            str(n_test),
            "--initial-budget",
            "2",
            "--max-workers",
            "2",
            "--no-restart",
            *extra_args,
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    return output_dir, result


def _assert_tuning_run_artifacts(output_dir: Path, *, expected_runs: int) -> list[dict[str, str]]:
    assert (output_dir / "raw_results.csv").is_file()
    assert (output_dir / "convergence_rates.json").is_file()
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == expected_runs
    with (output_dir / "convergence_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"median", "q25", "q75", "stderr"} <= set(rows[0])
    return rows


def test_cli_writes_complete_small_convergence_run_with_improving_error(tmp_path) -> None:
    output_dir, result = _run_tuning_convergence_cli(
        tmp_path,
        "complete-study",
        [
            "--resampling-mode",
            "legacy",
            "--validation-mode",
            "holdout",
            "--refinement-budget",
            "0",
        ],
        levels="8,16,32",
        n_test=64,
    )
    assert "Wrote convergence artifacts" in result.stdout
    rows = _assert_tuning_run_artifacts(output_dir, expected_runs=6)
    assert (output_dir / "convergence.png").is_file()
    assert len(list((output_dir / "tuning").glob("*/tuning_search.png"))) == 6
    assert len(list((output_dir / "median_predictions").glob("*.png"))) == 6
    for method in ("rbf_krr", "dm_krr"):
        errors = {
            int(row["refinement"]): float(row["median"])
            for row in rows
            if row["method"] == method and row["metric"] == "error"
        }
        assert set(errors) == {8, 16, 32}
        assert np.isfinite(list(errors.values())).all()


def test_target_registry_includes_requested_unit_disk_targets() -> None:
    sys.path.insert(0, str(Path(os.getcwd()) / "scripts/tuning_convergence"))
    from cartesian_high_freq_krr_targets import TARGETS

    points = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.25, 0.25], [-0.4, 0.3]])
    assert tuple(TARGETS) == ("laplace_neumann_m2_k2", "localized_bump", "oscillatory")
    for target_name in TARGETS:
        values = TARGETS[target_name](points)
        assert values.shape == (len(points), 1)
        assert np.isfinite(values).all()


def test_cli_defaults_match_ifblock_reference_configuration(monkeypatch) -> None:
    sys.path.insert(0, str(Path(os.getcwd()) / "scripts/tuning_convergence"))
    from cartesian_high_freq_krr_cli import config_from_args, default_levels_for_target, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["cartesian_high_freq_krr_cli.py", "--target", "laplace_neumann_m2_k2"],
    )
    config = config_from_args(parse_args())

    assert config.output_dir == Path("runs") / "laplace_neumann_m2_k2"
    assert config.levels == (512, 1024, 2048, 4096)
    assert default_levels_for_target("oscillatory") == (512, 1024, 2048, 4096)
    assert config.trials == 5
    assert config.n_val == 1024
    assert config.n_test == 4096
    assert config.initial_budget == (9, 9)
    assert config.refinement_strategy == "multi_start_nelder_mead"
    assert config.refinement_budget == 64
    assert config.max_workers == 4
    assert config.resampling_mode == "nested-fixed-test"
    assert config.validation_mode == "train-valid-count"
    assert config.validation_size == 1024
    assert config.pool_multiplier == 2
    assert config.restart is True


@pytest.mark.parametrize(
    ("case_name", "extra_args"),
    [
        pytest.param(
            "nested-kfold-grid",
            [
                "--resampling-mode",
                "nested-fixed-test",
                "--validation-mode",
                "kfold",
                "--k-folds",
                "2",
                "--refinement-budget",
                "0",
                "--no-plot",
            ],
            id="nested-kfold-no-refinement",
        ),
        pytest.param(
            "nested-train-valid-batch",
            [
                "--resampling-mode",
                "nested-fixed-test",
                "--validation-mode",
                "train-valid-count",
                "--validation-size",
                "4",
                "--pool-multiplier",
                "2",
                "--refinement-strategy",
                "batch_pattern_search",
                "--refinement-budget",
                "2",
                "--no-plot",
                "--no-prediction-plots",
            ],
            id="nested-train-valid-batch-pattern",
        ),
        pytest.param(
            "legacy-holdout-nelder",
            [
                "--resampling-mode",
                "legacy",
                "--validation-mode",
                "holdout",
                "--refinement-strategy",
                "nelder_mead_like",
                "--refinement-budget",
                "2",
                "--no-plot",
                "--no-prediction-plots",
            ],
            id="legacy-holdout-nelder-mead",
        ),
    ],
)
def test_cli_runs_resampling_validation_and_refinement_mode_combinations(
    tmp_path, case_name: str, extra_args: list[str]
) -> None:
    output_dir, result = _run_tuning_convergence_cli(
        tmp_path,
        case_name,
        extra_args,
        n_test=16,
    )
    assert "Wrote convergence artifacts" in result.stdout
    _assert_tuning_run_artifacts(output_dir, expected_runs=4)


def test_ifblock_entrypoint_writes_tuning_and_prediction_artifacts(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_dir = tmp_path / "runs" / "localized_bump"
    script_path = os.path.join(os.getcwd(), "scripts/tuning_convergence/cartesian_high_freq_krr.py")
    script_dir = str(Path(script_path).parent)
    script_source = Path(script_path).read_text(encoding="utf-8")
    replacements = {
        "LEVELS = (512, 1024, 2048, 4096)": "LEVELS = (8, 10)",
        'TARGET_NAME = "oscillatory"': 'TARGET_NAME = "localized_bump"',
        "TRIALS = 5": "TRIALS = 1",
        "N_VAL = 1024": "N_VAL = 8",
        "N_TEST = 4096": "N_TEST = 16",
        "INITIAL_BUDGET = (9, 9)": "INITIAL_BUDGET = 2",
        'REFINEMENT_STRATEGY = "multi_start_nelder_mead"': (
            'REFINEMENT_STRATEGY = "batch_pattern_search"'
        ),
        "REFINEMENT_BUDGET = 64": "REFINEMENT_BUDGET = 2",
        "MAX_WORKERS = 4": "MAX_WORKERS = 2",
        'VALIDATION_MODE = "train-valid-count"': 'VALIDATION_MODE = "kfold"',
        "VALIDATION_SIZE = 1024": "VALIDATION_SIZE = None",
        "K_FOLDS = 4": "K_FOLDS = 2",
    }
    for old, new in replacements.items():
        assert old in script_source
        script_source = script_source.replace(old, new)
    command = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {script_dir!r})
        namespace = {{"__file__": {script_path!r}, "__name__": "__main__"}}
        exec(compile({script_source!r}, {script_path!r}, "exec"), namespace)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    assert "Wrote convergence artifacts" in result.stdout
    assert (output_dir / "raw_results.csv").is_file()
    assert (output_dir / "convergence_rates.json").is_file()
    assert (output_dir / "convergence.png").is_file()
    with (output_dir / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        expected_count = len(list(csv.DictReader(handle)))
    tuning_results = list((output_dir / "tuning").glob("*/tuning_result.json"))
    assert len(tuning_results) == expected_count
    for tuning_result in tuning_results:
        payload = json.loads(tuning_result.read_text(encoding="utf-8"))
        assert payload["policy"]["refinement_strategy"] == "batch_pattern_search"
    assert len(list((output_dir / "tuning").glob("*/tuning_search.png"))) == expected_count
    assert len(list((output_dir / "median_predictions").glob("*.png"))) == expected_count
