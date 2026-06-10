import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np


def test_cartesian_high_freq_tuning_convergence_example_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_root = tmp_path / "study"
    output_dir = output_root / "smooth_radial"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_root),
            "--target",
            "smooth_radial",
            "--levels",
            "8,10",
            "--trials",
            "1",
            "--n-val",
            "8",
            "--n-test",
            "16",
            "--resampling-mode",
            "legacy",
            "--validation-mode",
            "holdout",
            "--initial-budget",
            "2",
            "--refinement-budget",
            "0",
            "--max-workers",
            "2",
            "--no-plot",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )

    assert "Wrote convergence artifacts" in result.stdout
    assert (output_dir / "raw_results.csv").is_file()
    assert (output_dir / "convergence_rates.json").is_file()
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == 4
    assert len(list((output_dir / "tuning").glob("*/tuning_search.png"))) == 4
    assert len(list((output_dir / "median_predictions").glob("*.png"))) == 4


def test_cartesian_high_freq_requested_mode_targets() -> None:
    sys.path.insert(0, str(Path(os.getcwd()) / "scripts/tuning_convergence"))
    from cartesian_high_freq_krr_targets import TARGETS

    points = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.25, 0.25], [-0.4, 0.3]])
    for target_name in ("laplace_neumann_m2_k2", "localized_bump", "rbf_eigen_m2_k2"):
        values = TARGETS[target_name](points)
        assert values.shape == (len(points), 1)
        assert np.isfinite(values).all()


def test_cartesian_high_freq_cli_defaults_match_reference(monkeypatch) -> None:
    sys.path.insert(0, str(Path(os.getcwd()) / "scripts/tuning_convergence"))
    from cartesian_high_freq_krr_cli import config_from_args, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["cartesian_high_freq_krr_cli.py", "--target", "rbf_eigen_m2_k2"],
    )
    config = config_from_args(parse_args())

    assert config.output_dir == Path("runs") / "rbf_eigen_m2_k2"
    assert config.levels == (512, 1024, 2048, 4096)
    assert config.trials == 5
    assert config.n_val == 1024
    assert config.n_test == 4096
    assert config.initial_budget == (9, 9)
    assert config.refinement_strategy == "batch_pattern_search"
    assert config.refinement_budget == 64
    assert config.max_workers == 4
    assert config.resampling_mode == "nested-fixed-test"
    assert config.validation_mode == "train-valid-count"
    assert config.validation_size == 1024
    assert config.pool_multiplier == 2
    assert config.restart is True


def test_cartesian_high_freq_tuning_convergence_nested_mode_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_root = tmp_path / "nested-study"
    output_dir = output_root / "oscillatory"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_root),
            "--target",
            "oscillatory",
            "--levels",
            "8,10",
            "--trials",
            "1",
            "--n-test",
            "16",
            "--initial-budget",
            "2",
            "--refinement-budget",
            "0",
            "--resampling-mode",
            "nested-fixed-test",
            "--validation-mode",
            "kfold",
            "--k-folds",
            "2",
            "--max-workers",
            "1",
            "--no-plot",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )

    assert "Wrote convergence artifacts" in result.stdout
    assert (output_dir / "raw_results.csv").is_file()
    assert (output_dir / "convergence_rates.json").is_file()
    with (output_dir / "convergence_summary.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert {"median", "q25", "q75", "stderr"} <= set(header)
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == 4
    assert len(list((output_dir / "median_predictions").glob("*.png"))) == 4


def test_cartesian_high_freq_tuning_convergence_train_valid_count_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_root = tmp_path / "train-valid-count-study"
    output_dir = output_root / "oscillatory"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_root),
            "--target",
            "oscillatory",
            "--levels",
            "8,10",
            "--trials",
            "1",
            "--n-test",
            "16",
            "--initial-budget",
            "2",
            "--refinement-budget",
            "0",
            "--resampling-mode",
            "nested-fixed-test",
            "--validation-mode",
            "train-valid-count",
            "--validation-size",
            "4",
            "--pool-multiplier",
            "2",
            "--max-workers",
            "1",
            "--no-plot",
            "--no-prediction-plots",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )

    assert "Wrote convergence artifacts" in result.stdout
    assert (output_dir / "raw_results.csv").is_file()
    assert (output_dir / "convergence_rates.json").is_file()
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == 4


def test_cartesian_high_freq_tuning_convergence_ifblock_example_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_dir = tmp_path / "runs" / "rbf_eigen_m2_k2"
    script_path = os.path.join(os.getcwd(), "scripts/tuning_convergence/cartesian_high_freq_krr.py")
    script_dir = str(Path(script_path).parent)
    script_source = Path(script_path).read_text(encoding="utf-8")
    replacements = {
        "LEVELS = (512, 1024, 2048, 4096, 8192)": "LEVELS = (8, 10)",
        "LEVELS = (512, 1024, 2048, 4096)": "LEVELS = (8, 10)",
        "TRIALS = 5": "TRIALS = 1",
        "N_VAL = 1024": "N_VAL = 8",
        "N_TEST = 4096": "N_TEST = 16",
        "INITIAL_BUDGET = (9, 9)": "INITIAL_BUDGET = 2",
        'REFINEMENT_BUDGET = 64 if REFINEMENT_STRATEGY == "batch_pattern_search" else 20': (
            "REFINEMENT_BUDGET = 2"
        ),
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
