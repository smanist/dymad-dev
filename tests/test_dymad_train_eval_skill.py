from __future__ import annotations

from pathlib import Path


def test_phase1_skill_staging_files_exist() -> None:
    skills_root = Path(__file__).resolve().parents[1] / "skills"

    train_root = skills_root / "dymad-train-model"
    train_body = (train_root / "SKILL.md").read_text(encoding="utf-8")
    train_openai_yaml = (train_root / "agents" / "openai.yaml").read_text(encoding="utf-8")

    eval_root = skills_root / "dymad-evaluate-model"
    eval_body = (eval_root / "SKILL.md").read_text(encoding="utf-8")
    eval_openai_yaml = (eval_root / "agents" / "openai.yaml").read_text(encoding="utf-8")

    assert "list_model_families" in train_body
    assert "resolve_training_intent" in train_body
    assert "intent.accepted_inputs" in train_body
    assert "suggested_validate_request" in train_body
    assert (
        "Do not inspect or modify repo code unless an MCP tool reports that the requested workflow is unsupported."
        in train_body
    )
    assert "validate_training_config" in train_body
    assert "materialize_training_config" in train_body
    assert "train_model" in train_body
    assert "free text" in train_body
    assert "DyMAD Train Model" in train_openai_yaml
    assert "intent.accepted_inputs" in train_openai_yaml
    assert "suggested_validate_request" in train_openai_yaml

    assert "register_checkpoint" in eval_body
    assert "validate_dataset_compatibility" in eval_body
    assert "evaluate_model" in eval_body
    assert "rollout_rmse" in eval_body
    assert "DyMAD Evaluate Model" in eval_openai_yaml
