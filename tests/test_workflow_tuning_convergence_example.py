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


def test_cartesian_high_freq_tuning_convergence_ifblock_example_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
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
    assert len(list((output_dir / "tuning").glob("*/tuning_result.json"))) == 8
    assert len(list((output_dir / "tuning").glob("*/tuning_search.png"))) == 8
    assert len(list((output_dir / "median_predictions").glob("*.png"))) == 8
