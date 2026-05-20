from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

import dymad.io
from dymad.agent.app import MANIFEST_FILENAME, CLIWorkflowError, CLIWorkflowService
from dymad.cli import main
from tests.test_agent_mcp_train_eval_tools import (
    _configure_worker_bootstrap,
    _write_regular_dataset,
)


def _write_cli_config(
    path: Path,
    *,
    train_path: Path,
    test_path: Path | None = None,
    model_key: str = "kbf",
    run_name: str | None = None,
    overrides: dict | None = None,
) -> None:
    payload: dict = {
        "version": 1,
        "model_key": model_key,
        "data": {
            "train": {
                "path": train_path.name,
            }
        },
    }
    if test_path is not None:
        payload["data"]["test"] = {"path": test_path.name}
    if run_name is not None:
        payload["run"] = {"name": run_name}
    if overrides is not None:
        payload["overrides"] = overrides
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _wait_for_manifest_status(run_dir: Path, expected: str = "SUCCEEDED") -> dict:
    service = CLIWorkflowService()
    deadline = time.time() + 5.0
    latest: dict | None = None
    while time.time() < deadline:
        latest = service.status(run_dir=run_dir)
        status = latest["manifest"]["status"]
        if status in {"SUCCEEDED", "FAILED"}:
            assert status == expected
            return latest
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {latest}")


def test_cli_config_schema_emits_json(capsys) -> None:
    code = main(["config", "schema"])
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["title"] == "DyMAD CLI config"
    assert output["properties"]["version"]["const"] == 1


def test_cli_validate_accepts_minimal_config_and_uses_user_compiler(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    _write_regular_dataset(dataset_path, with_control=False)
    _write_cli_config(
        config_path,
        train_path=dataset_path,
        overrides={"model": {"koopman_dimension": 6}},
    )

    result = CLIWorkflowService().validate_config(config_path, run_dir=tmp_path / "runs" / "foo")

    assert result["valid"] is True
    assert result["compiled_request"].model.key == "kbf"
    assert result["compiled_request"].effective_run_name == "foo"
    assert result["compiled_request"].effective_config["data"]["path"] == str(
        dataset_path.resolve()
    )
    assert result["compiled_request"].effective_config["model"]["koopman_dimension"] == 6


def test_cli_validate_rejects_bad_config_and_compiler_errors(tmp_path) -> None:
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    _write_regular_dataset(dataset_path, with_control=False)

    config_path.write_text("version: 1\nmodel_key: kbf\n", encoding="utf-8")
    try:
        CLIWorkflowService().validate_config(config_path)
    except CLIWorkflowError as exc:
        assert "data must be a mapping" in str(exc)
    else:
        raise AssertionError("missing data was accepted")

    _write_cli_config(config_path, train_path=dataset_path, run_name="bar")
    try:
        CLIWorkflowService().validate_config(config_path, run_dir=tmp_path / "runs" / "foo")
    except CLIWorkflowError as exc:
        assert "must match --out directory name" in str(exc)
    else:
        raise AssertionError("mismatched run name was accepted")

    _write_cli_config(
        config_path,
        train_path=dataset_path,
        model_key="lti",
        overrides={"model": {"encoder_layers": 0, "decoder_layers": 0}},
    )
    try:
        CLIWorkflowService().validate_config(config_path)
    except Exception as exc:
        assert "identity map" in str(exc)
    else:
        raise AssertionError("compiler-invalid overrides were accepted")


def test_cli_registry_list_json_covers_user_registries(capsys) -> None:
    for kind in ("models", "losses", "profiles", "training", "analyses", "evaluations"):
        code = main(["registry", "list", kind, "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert code == 0
        assert payload["kind"] == kind
        assert payload["items"]


def test_cli_train_detach_writes_manifest_and_run_local_store(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "runs" / "foo"
    _write_regular_dataset(dataset_path, with_control=False)
    _write_cli_config(config_path, train_path=dataset_path)

    code = main(["train", "--config", str(config_path), "--out", str(run_dir), "--detach"])

    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert code == 0
    assert manifest["artifact_root"] == str(run_dir.parent.resolve())
    assert manifest["run_dir"] == str(run_dir.resolve())
    assert manifest["store_root"] == str((run_dir / ".dymad-store").resolve())
    assert manifest["normalized_config"]["run"]["name"] == "foo"
    assert manifest["dataset_paths"]["train"] == str(dataset_path.resolve())
    assert manifest["dataset_handles"]["train"].startswith("ds_")
    assert manifest["compiled_request_handle"].startswith("trainreq_")
    assert manifest["training_run_handle"].startswith("run_")
    assert (run_dir / ".dymad-store").is_dir()
    _wait_for_manifest_status(run_dir)


def test_cli_status_log_and_report_recover_from_manifest(tmp_path, monkeypatch, capsys) -> None:
    _configure_worker_bootstrap(monkeypatch)
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "runs" / "recover"
    _write_regular_dataset(dataset_path, with_control=False)
    _write_cli_config(config_path, train_path=dataset_path)

    assert main(["train", "--config", str(config_path), "--out", str(run_dir), "--detach"]) == 0
    capsys.readouterr()
    _wait_for_manifest_status(run_dir)

    assert main(["status", "--run", str(run_dir), "--json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["manifest"]["status"] == "SUCCEEDED"

    assert main(["log", "--run", str(run_dir)]) == 0
    assert "fake trainer" in capsys.readouterr().out

    assert main(["report", "--run", str(run_dir), "--json"]) == 0
    report_payload = json.loads(capsys.readouterr().out)
    assert report_payload["status"] == "SUCCEEDED"
    assert report_payload["checkpoint_handle"].startswith("chk_")


def test_cli_blocking_train_returns_nonzero_on_failed_training(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch, mode="fail")
    dataset_path = tmp_path / "train.npz"
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "runs" / "failed"
    _write_regular_dataset(dataset_path, with_control=False)
    _write_cli_config(config_path, train_path=dataset_path)

    code = main(["train", "--config", str(config_path), "--out", str(run_dir)])

    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert code == 1
    assert manifest["status"] == "FAILED"
    assert manifest["training_run"]["error_type"] == "RuntimeError"


def test_cli_eval_uses_config_test_data_and_accepts_override(tmp_path, monkeypatch) -> None:
    _configure_worker_bootstrap(monkeypatch)
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    override_path = tmp_path / "override.npz"
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "runs" / "eval"
    _write_regular_dataset(train_path, with_control=False)
    _write_regular_dataset(test_path, with_control=False)
    _write_regular_dataset(override_path, with_control=False)
    _write_cli_config(config_path, train_path=train_path, test_path=test_path)

    def fake_load_model(model, checkpoint_path, *, context=None, **kwargs):
        del model, checkpoint_path, context, kwargs

        def predict_fn(x0, t, u=None, p=None, **inner_kwargs):
            del t, u, p, inner_kwargs
            return 0.5 * x0

        return object(), predict_fn

    def fake_plot_trajectory(traj, ts, model_name=None, prefix=".", **kwargs):
        del traj, ts, kwargs
        (Path(prefix) / f"{model_name}_prediction.png").write_bytes(b"plot")

    monkeypatch.setattr(dymad.io, "load_model", fake_load_model)
    monkeypatch.setattr("dymad.agent.exec.workflow.plot_trajectory", fake_plot_trajectory)

    assert main(["train", "--config", str(config_path), "--out", str(run_dir), "--detach"]) == 0
    _wait_for_manifest_status(run_dir)

    result = CLIWorkflowService().evaluate(run_dir=run_dir)
    assert result["evaluation"].evaluation_summary.handle.startswith("eval_")
    assert result["manifest"]["evaluation_handles"]

    override_result = CLIWorkflowService().evaluate(
        run_dir=run_dir,
        test_data=override_path,
    )
    assert override_result["manifest"]["dataset_paths"]["test_override"] == str(
        override_path.resolve()
    )
    assert len(override_result["manifest"]["evaluation_handles"]) == 2
