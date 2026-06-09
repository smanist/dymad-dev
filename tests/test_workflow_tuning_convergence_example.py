import csv
import os
import subprocess
import sys


def test_cartesian_high_freq_tuning_convergence_example_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_dir = tmp_path / "study"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_dir),
            "--levels",
            "8,10",
            "--trials",
            "1",
            "--n-val",
            "8",
            "--n-test",
            "16",
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


def test_cartesian_high_freq_tuning_convergence_nested_mode_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output_dir = tmp_path / "nested-study"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_dir),
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
    output_dir = tmp_path / "train-valid-count-study"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tuning_convergence/cartesian_high_freq_krr_cli.py",
            "--workdir",
            str(output_dir),
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
    env.update(
        {
            "DYMAD_TUNING_LEVELS": "8,10",
            "DYMAD_TUNING_TRIALS": "1",
            "DYMAD_TUNING_N_VAL": "8",
            "DYMAD_TUNING_N_TEST": "16",
            "DYMAD_TUNING_INITIAL_BUDGET": "2",
            "DYMAD_TUNING_REFINEMENT_BUDGET": "0",
            "DYMAD_TUNING_MAX_WORKERS": "1",
            "DYMAD_TUNING_RESAMPLING_MODE": "nested-fixed-test",
            "DYMAD_TUNING_VALIDATION_MODE": "kfold",
            "DYMAD_TUNING_K_FOLDS": "2",
        }
    )
    output_dir = tmp_path / "runs"
    script_path = os.path.join(os.getcwd(), "scripts/tuning_convergence/cartesian_high_freq_krr.py")

    result = subprocess.run(
        [sys.executable, script_path],
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
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == expected_count
    assert len(list((output_dir / "tuning").glob("*/tuning_search.png"))) == expected_count
    assert len(list((output_dir / "median_predictions").glob("*.png"))) == expected_count
