from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from dymad.agent.app import MANIFEST_FILENAME, CLIWorkflowService
from dymad.agent.exec.context import build_default_context
from dymad.agent.mcp import UserTools
from tests.test_agent_mcp_train_eval_tools import (
    _configure_worker_bootstrap,
    _write_regular_dataset,
)


def _write_cli_config(
    path: Path,
    *,
    train_path: Path,
    run_name: str,
    overrides: dict[str, Any],
) -> None:
    payload = {
        "version": 1,
        "model_key": "kbf",
        "data": {
            "train": {
                "path": train_path.name,
            }
        },
        "overrides": overrides,
        "run": {
            "name": run_name,
            "seed": 123,
            "device": "cpu",
            "max_workers": 2,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _wait_for_cli_training(run_dir: Path) -> dict[str, Any]:
    service = CLIWorkflowService()
    deadline = time.time() + 5.0
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        latest = service.status(run_dir=run_dir)
        status = latest["manifest"]["status"]
        if status in {"SUCCEEDED", "FAILED"}:
            assert status == "SUCCEEDED"
            return latest
        time.sleep(0.05)
    raise AssertionError(f"CLI run did not finish: {latest}")


def _wait_for_mcp_training(tools: UserTools, training_run_handle: str) -> dict[str, Any]:
    deadline = time.time() + 5.0
    latest: dict[str, Any] | None = None
    while time.time() < deadline:
        latest = tools.describe_training_run(training_run_handle=training_run_handle)
        assert latest["ok"] is True
        status = latest["data"]["training_run"]["status"]
        if status in {"SUCCEEDED", "FAILED"}:
            assert status == "SUCCEEDED"
            return latest
        time.sleep(0.05)
    raise AssertionError(f"MCP run did not finish: {latest}")


def _compiled_contract(compiled: Any) -> dict[str, Any]:
    return {
        "model_key": compiled.model.key,
        "reference_profile": compiled.profile.key,
        "profile_key": compiled.profile.key,
        "model_ref": compiled.model_ref,
        "train_dataset_kind": compiled.train_dataset_kind,
        "valid_dataset_kind": compiled.valid_dataset_kind,
        "effective_run_name": compiled.effective_run_name,
        "effective_config": compiled.effective_config,
        "trainer_kind": compiled.trainer_kind,
        "seed": compiled.request.seed,
        "device": compiled.request.device,
        "max_workers": compiled.request.max_workers,
        "warnings": [],
    }


def _mcp_compiled_contract(compiled: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_key": compiled["model_key"],
        "reference_profile": compiled["reference_profile"],
        "profile_key": compiled["reference_profile"],
        "model_ref": compiled["model_ref"],
        "train_dataset_kind": compiled["train_dataset_kind"],
        "valid_dataset_kind": compiled["valid_dataset_kind"],
        "effective_run_name": compiled["effective_run_name"],
        "effective_config": compiled["effective_config"],
        "trainer_kind": compiled["trainer_kind"],
        "seed": compiled["seed"],
        "device": compiled["device"],
        "max_workers": compiled["max_workers"],
        "warnings": compiled["warnings"],
    }


def test_cli_and_mcp_compile_equivalent_user_training_request(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "runs" / "shared_run"
    overrides = {"model": {"koopman_dimension": 6}}
    _write_regular_dataset(dataset_path, with_control=False)
    _write_cli_config(
        config_path,
        train_path=dataset_path,
        run_name=run_dir.name,
        overrides=overrides,
    )

    cli_result = CLIWorkflowService().validate_config(config_path, run_dir=run_dir)

    mcp_context = build_default_context(artifact_root=tmp_path / "mcp-artifacts")
    mcp_tools = UserTools(context=mcp_context)
    mcp_dataset_handle = mcp_context.facade.register_dataset_file(path=str(dataset_path)).handle
    mcp_result = mcp_tools.compile_training_request(
        train_dataset_handle=mcp_dataset_handle,
        model_key="kbf",
        overrides=overrides,
        run_name=run_dir.name,
        seed=123,
        device="cpu",
        max_workers=2,
    )

    assert mcp_result["ok"] is True
    assert _mcp_compiled_contract(mcp_result["data"]["compiled_request"]) == (
        _compiled_contract(cli_result["compiled_request"])
    )
    assert cli_result["dataset_handles"]["train"].startswith("ds_")
    assert mcp_result["data"]["summary"]["handle"].startswith("trainreq_")


def test_cli_and_mcp_training_share_compiled_run_semantics(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    cli_run_dir = tmp_path / "runs" / "shared_train"
    overrides = {"model": {"koopman_dimension": 6}}
    _write_regular_dataset(dataset_path, with_control=False)
    _write_cli_config(
        config_path,
        train_path=dataset_path,
        run_name=cli_run_dir.name,
        overrides=overrides,
    )

    cli_service = CLIWorkflowService()
    cli_started = cli_service.train(config_path=config_path, run_dir=cli_run_dir)
    cli_final = _wait_for_cli_training(cli_run_dir)
    cli_manifest_path = cli_run_dir / MANIFEST_FILENAME
    cli_manifest = json.loads(cli_manifest_path.read_text(encoding="utf-8"))

    mcp_context = build_default_context(artifact_root=tmp_path / "mcp-store")
    mcp_tools = UserTools(context=mcp_context)
    mcp_dataset_handle = mcp_context.facade.register_dataset_file(path=str(dataset_path)).handle
    mcp_compiled = mcp_tools.compile_training_request(
        train_dataset_handle=mcp_dataset_handle,
        model_key="kbf",
        overrides=overrides,
        run_name=cli_run_dir.name,
        seed=123,
        device="cpu",
        max_workers=2,
    )
    mcp_started = mcp_tools.start_training_run(
        compiled_request_handle=mcp_compiled["data"]["summary"]["handle"],
        artifact_root=str(tmp_path / "mcp-runs"),
    )
    mcp_final = _wait_for_mcp_training(
        mcp_tools,
        mcp_started["data"]["summary"]["handle"],
    )

    assert cli_started["manifest"]["compiled_request_handle"].startswith("trainreq_")
    assert cli_started["manifest"]["training_run_handle"].startswith("run_")
    assert cli_manifest["run_dir"] == str(cli_run_dir.resolve())
    assert cli_manifest["store_root"] == str((cli_run_dir / ".dymad-store").resolve())
    assert cli_manifest["normalized_config"]["run"]["name"] == cli_run_dir.name
    assert cli_manifest["normalized_config"]["run"]["seed"] == 123
    assert cli_manifest["status"] == "SUCCEEDED"
    assert cli_manifest["checkpoint_handle"].startswith("chk_")
    assert (cli_run_dir / ".dymad-store").is_dir()

    cli_run = cli_final["status"].training_run
    mcp_run = mcp_final["data"]["training_run"]
    assert cli_run.model_ref == mcp_run["model_ref"]
    assert cli_run.reference_profile == mcp_run["reference_profile"]
    assert cli_run.run_name == mcp_run["run_name"] == cli_run_dir.name
    assert cli_run.artifact_root == str(cli_run_dir.parent.resolve())
    assert mcp_run["artifact_root"] == str((tmp_path / "mcp-runs").resolve())
    assert cli_run.status.value == mcp_run["status"] == "SUCCEEDED"
