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
            "0",
            "--n-val",
            "8",
            "--n-test",
            "16",
            "--initial-budget",
            "2",
            "--refinement-budget",
            "0",
            "--no-plot",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
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
            "DYMAD_CARTESIAN_WORKDIR": str(tmp_path / "study_ifblocks"),
            "DYMAD_CARTESIAN_LEVELS": "8,10",
            "DYMAD_CARTESIAN_TRIALS": "0",
            "DYMAD_CARTESIAN_N_VAL": "8",
            "DYMAD_CARTESIAN_N_TEST": "16",
            "DYMAD_CARTESIAN_INITIAL_BUDGET": "2",
            "DYMAD_CARTESIAN_REFINEMENT_BUDGET": "0",
            "DYMAD_CARTESIAN_IFPLT": "0",
        }
    )
    output_dir = tmp_path / "study_ifblocks"

    result = subprocess.run(
        [sys.executable, "scripts/tuning_convergence/cartesian_high_freq_krr.py"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )

    assert "Wrote convergence artifacts" in result.stdout
    assert (output_dir / "raw_results.csv").is_file()
    assert (output_dir / "convergence_rates.json").is_file()
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == 4
