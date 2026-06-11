from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_installer() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "skills" / "install_dymad_skills.py"
    spec = importlib.util.spec_from_file_location("install_dymad_skills", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dymad_train_eval_skill_staging_files_exist() -> None:
    skill_root = Path(__file__).resolve().parents[1] / "skills" / "dymad-train-eval-workflow"
    skill_body = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    openai_yaml = (skill_root / "agents" / "openai.yaml").read_text(encoding="utf-8")

    assert "register_dataset_file" in skill_body
    assert "inspect_dataset" in skill_body
    assert "describe_training_capability" in skill_body
    assert "list_evaluation_capabilities" in skill_body
    assert "compile_training_request" in skill_body
    assert "start_training_run" in skill_body
    assert "describe_training_run" in skill_body
    assert "read_training_run_log" in skill_body
    assert "evaluate_checkpoint" in skill_body
    assert "dymad config validate CONFIG --out RUN_DIR" in skill_body
    assert "dymad train --config CONFIG [--out RUN_DIR]" in skill_body
    assert "Export Reproducible CLI Run" in skill_body
    assert "dymad-run.json" in skill_body
    assert "overrides.cv" in skill_body
    assert "cv_results_path" in skill_body
    assert "supported_metrics" in skill_body
    assert "free text" in skill_body
    assert "model.name" in skill_body
    assert "runtime_owned_override_paths" in skill_body
    assert "DyMAD Train/Eval Workflow" in openai_yaml
    assert "start_training_run" in openai_yaml
    assert "train_compiled_request" not in openai_yaml
    assert "CLI config" in openai_yaml


def test_dymad_tuning_convergence_skill_staging_files_exist() -> None:
    skill_root = Path(__file__).resolve().parents[1] / "skills" / "dymad-tuning-convergence-study"
    skill_body = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    openai_yaml = (skill_root / "agents" / "openai.yaml").read_text(encoding="utf-8")

    assert "ArrayRegressionProblem" in skill_body
    assert "ArrayRegressionStudyConfig" in skill_body
    assert "run_array_regression_study" in skill_body
    assert "ParameterSpec" in skill_body
    assert "TuningSpec" in skill_body
    assert "tests/test_workflow_<study>_tuning_convergence.py" in skill_body
    assert "Wrote convergence artifacts" in skill_body
    assert "context_results/" in skill_body
    assert "median_predictions/" in skill_body
    assert "tuning/" in skill_body
    assert "prediction_plotter" in skill_body
    assert "truth on the test/sample points" in skill_body
    assert "white" in skill_body
    assert "zero error" in skill_body
    assert "do not add MCP exposure" in skill_body
    assert "DyMAD Tuning Convergence" in openai_yaml
    assert "$dymad-tuning-convergence-study" in openai_yaml


def test_dymad_train_eval_skill_installer_copies_repo_skill_to_codex_home(tmp_path) -> None:
    installer = _load_installer()

    result = installer.install_skill(codex_home=tmp_path)

    installed_root = tmp_path / "skills" / "dymad-train-eval-workflow"
    assert result.changed is True
    assert result.destination_dir == installed_root
    assert (installed_root / "SKILL.md").read_text(encoding="utf-8") == (
        installer.DEFAULT_SOURCE_DIR / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert (installed_root / "agents" / "openai.yaml").is_file()
    assert installer.install_skill(codex_home=tmp_path, check=True).changed is False


def test_dymad_skill_installer_copies_all_repo_skills_to_codex_home(tmp_path) -> None:
    installer = _load_installer()

    results = installer.install_repo_skills(codex_home=tmp_path)

    assert {result.destination_dir.name for result in results} == set(installer.REPO_SKILL_NAMES)
    for skill_name in installer.REPO_SKILL_NAMES:
        installed_root = tmp_path / "skills" / skill_name
        source_root = Path(__file__).resolve().parents[1] / "skills" / skill_name
        assert (installed_root / "SKILL.md").read_text(encoding="utf-8") == (
            source_root / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert (installed_root / "agents" / "openai.yaml").is_file()

    assert not any(result.changed for result in installer.install_repo_skills(codex_home=tmp_path))


def test_dymad_train_eval_skill_installer_check_detects_drift(tmp_path) -> None:
    installer = _load_installer()
    installer.install_skill(codex_home=tmp_path)
    installed_skill = tmp_path / "skills" / "dymad-train-eval-workflow" / "SKILL.md"
    installed_skill.write_text("stale skill\n", encoding="utf-8")

    result = installer.install_skill(codex_home=tmp_path, check=True)

    assert result.changed is True
