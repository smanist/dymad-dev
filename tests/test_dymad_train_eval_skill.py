from __future__ import annotations

from pathlib import Path


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
    assert "overrides.cv" in skill_body
    assert "cv_results_path" in skill_body
    assert "supported_metrics" in skill_body
    assert "free text" in skill_body
    assert "DyMAD Train/Eval Workflow" in openai_yaml
