"""Path-first user-mode CLI workflow service."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

from dymad.agent.compiler import TrainingRequest, compile_training_request
from dymad.agent.exec.context import ExecutionContext, build_default_context
from dymad.agent.exec.workflow import CompatibilityExecutor
from dymad.agent.facade.operations import FacadeOperations
from dymad.agent.registry import SUPPORTED_EVALUATION_METRICS
from dymad.agent.store.object_store import ObjectStore, TrainingRunStatus

MANIFEST_FILENAME = "dymad-run.json"
MANIFEST_SCHEMA_VERSION = 1
CLI_CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DyMAD CLI config",
    "type": "object",
    "required": ["version", "model_key", "data"],
    "properties": {
        "version": {"const": 1},
        "model_key": {"type": "string", "minLength": 1},
        "reference_profile": {"type": "string", "minLength": 1},
        "data": {
            "type": "object",
            "required": ["train"],
            "properties": {
                "train": {"$ref": "#/$defs/dataset"},
                "valid": {"$ref": "#/$defs/dataset"},
                "test": {"$ref": "#/$defs/dataset"},
            },
            "additionalProperties": False,
        },
        "overrides": {"type": "object"},
        "run": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "seed": {"type": "integer"},
                "device": {"type": "string"},
                "max_workers": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "evaluation": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "plot_selection": {"type": "string"},
                "max_plots": {"type": "integer", "minimum": 0},
                "predict_kwargs": {"type": "object"},
            },
            "additionalProperties": False,
        },
    },
    "$defs": {
        "dataset": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "kind": {"enum": ["regular", "graph"]},
                "format": {"enum": ["npz"]},
            },
            "additionalProperties": False,
        }
    },
    "additionalProperties": False,
}


class CLIWorkflowError(RuntimeError):
    """Raised when a CLI workflow command cannot complete."""


@dataclass(frozen=True)
class LoadedCLIConfig:
    source_config_path: str
    normalized_config: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _build_in_memory_context() -> ExecutionContext:
    store = ObjectStore()
    facade = FacadeOperations(store)
    context_holder: dict[str, ExecutionContext] = {}
    executor = CompatibilityExecutor(facade, context_provider=lambda: context_holder["context"])
    context = ExecutionContext(
        artifact_store=None,
        store=store,
        facade=facade,
        executor=executor,
    )
    context_holder["context"] = context
    return context


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CLIWorkflowError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CLIWorkflowError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_data_path(raw_path: str, *, config_dir: Path) -> str:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    return str(candidate.resolve())


def _normalize_dataset(
    raw: object,
    *,
    field: str,
    config_dir: Path,
) -> dict[str, str]:
    dataset = _require_mapping(raw, field=field)
    raw_path = _require_string(dataset.get("path"), field=f"{field}.path")
    unknown = sorted(set(dataset) - {"path", "kind", "format"})
    if unknown:
        raise CLIWorkflowError(f"{field}.{unknown[0]} is not supported")
    kind = dataset.get("kind", "regular")
    format_name = dataset.get("format", "npz")
    if kind not in {"regular", "graph"}:
        raise CLIWorkflowError(f"{field}.kind must be 'regular' or 'graph'")
    if format_name != "npz":
        raise CLIWorkflowError(f"{field}.format must be 'npz'")
    return {
        "path": _resolve_data_path(raw_path, config_dir=config_dir),
        "kind": cast(str, kind),
        "format": cast(str, format_name),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CLIWorkflowError(f"config file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        raise CLIWorkflowError("config file is empty")
    return _require_mapping(loaded, field="config")


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CLIWorkflowError(f"run manifest does not exist: {manifest_path}")
    return _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        field="run manifest",
    )


def _save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _context_from_manifest(manifest: dict[str, Any]) -> ExecutionContext:
    store_root = _require_string(manifest.get("store_root"), field="manifest.store_root")
    return build_default_context(artifact_root=store_root)


class CLIWorkflowService:
    """Shared path-first user workflow service for the package CLI."""

    def load_config(self, config_path: str | Path) -> LoadedCLIConfig:
        source_path = Path(config_path).expanduser().resolve()
        raw = _load_yaml_mapping(source_path)
        unknown = sorted(
            set(raw)
            - {
                "version",
                "model_key",
                "reference_profile",
                "data",
                "overrides",
                "run",
                "evaluation",
            }
        )
        if unknown:
            raise CLIWorkflowError(f"{unknown[0]} is not supported")
        if raw.get("version") != 1:
            raise CLIWorkflowError("version must be 1")

        model_key = _require_string(raw.get("model_key"), field="model_key")
        normalized: dict[str, Any] = {
            "version": 1,
            "model_key": model_key,
        }
        if raw.get("reference_profile") is not None:
            normalized["reference_profile"] = _require_string(
                raw.get("reference_profile"),
                field="reference_profile",
            )

        data = _require_mapping(raw.get("data"), field="data")
        data_unknown = sorted(set(data) - {"train", "valid", "test"})
        if data_unknown:
            raise CLIWorkflowError(f"data.{data_unknown[0]} is not supported")
        if "train" not in data:
            raise CLIWorkflowError("data.train is required")
        normalized_data = {
            "train": _normalize_dataset(
                data["train"], field="data.train", config_dir=source_path.parent
            )
        }
        if data.get("valid") is not None:
            normalized_data["valid"] = _normalize_dataset(
                data["valid"],
                field="data.valid",
                config_dir=source_path.parent,
            )
        if data.get("test") is not None:
            normalized_data["test"] = _normalize_dataset(
                data["test"],
                field="data.test",
                config_dir=source_path.parent,
            )
        normalized["data"] = normalized_data

        overrides = raw.get("overrides")
        if overrides is not None:
            normalized["overrides"] = _require_mapping(overrides, field="overrides")

        run = raw.get("run", {})
        run_mapping = _require_mapping(run, field="run") if run is not None else {}
        run_unknown = sorted(set(run_mapping) - {"name", "seed", "device", "max_workers"})
        if run_unknown:
            raise CLIWorkflowError(f"run.{run_unknown[0]} is not supported")
        normalized_run: dict[str, Any] = {}
        if run_mapping.get("name") is not None:
            normalized_run["name"] = _require_string(run_mapping.get("name"), field="run.name")
        if run_mapping.get("seed") is not None:
            if not isinstance(run_mapping["seed"], int):
                raise CLIWorkflowError("run.seed must be an integer")
            normalized_run["seed"] = run_mapping["seed"]
        if run_mapping.get("device") is not None:
            normalized_run["device"] = _require_string(
                run_mapping.get("device"), field="run.device"
            )
        else:
            normalized_run["device"] = "auto"
        if run_mapping.get("max_workers") is not None:
            if not isinstance(run_mapping["max_workers"], int) or run_mapping["max_workers"] <= 0:
                raise CLIWorkflowError("run.max_workers must be a positive integer")
            normalized_run["max_workers"] = run_mapping["max_workers"]
        else:
            normalized_run["max_workers"] = 1
        normalized["run"] = normalized_run

        evaluation = raw.get("evaluation", {})
        evaluation_mapping = (
            _require_mapping(evaluation, field="evaluation") if evaluation is not None else {}
        )
        evaluation_unknown = sorted(
            set(evaluation_mapping) - {"metric", "plot_selection", "max_plots", "predict_kwargs"}
        )
        if evaluation_unknown:
            raise CLIWorkflowError(f"evaluation.{evaluation_unknown[0]} is not supported")
        normalized_evaluation: dict[str, Any] = {
            "metric": evaluation_mapping.get("metric", "rollout_rmse"),
            "plot_selection": evaluation_mapping.get("plot_selection", "median"),
            "max_plots": evaluation_mapping.get("max_plots", 1),
            "predict_kwargs": evaluation_mapping.get("predict_kwargs", {}),
        }
        if normalized_evaluation["metric"] not in SUPPORTED_EVALUATION_METRICS:
            raise CLIWorkflowError(
                f"evaluation.metric must be one of {sorted(SUPPORTED_EVALUATION_METRICS)}"
            )
        if not isinstance(normalized_evaluation["plot_selection"], str):
            raise CLIWorkflowError("evaluation.plot_selection must be a string")
        if (
            not isinstance(normalized_evaluation["max_plots"], int)
            or normalized_evaluation["max_plots"] < 0
        ):
            raise CLIWorkflowError("evaluation.max_plots must be a non-negative integer")
        if not isinstance(normalized_evaluation["predict_kwargs"], dict):
            raise CLIWorkflowError("evaluation.predict_kwargs must be a mapping")
        normalized["evaluation"] = normalized_evaluation

        return LoadedCLIConfig(
            source_config_path=str(source_path),
            normalized_config=normalized,
        )

    def validate_config(
        self, config_path: str | Path, *, run_dir: str | Path | None = None
    ) -> dict[str, Any]:
        loaded = self.load_config(config_path)
        context = _build_in_memory_context()
        dataset_handles = self._register_config_datasets(
            context=context,
            normalized_config=loaded.normalized_config,
        )
        run_name = None if run_dir is None else Path(run_dir).expanduser().resolve().name
        if run_name is not None:
            self._validate_run_name_match(loaded.normalized_config, run_name=run_name)
        compiled = compile_training_request(
            facade=context.facade,
            request=self._training_request(
                normalized_config=loaded.normalized_config,
                dataset_handles=dataset_handles,
                run_name=run_name or loaded.normalized_config["run"].get("name"),
            ),
        )
        return {
            "valid": True,
            "source_config_path": loaded.source_config_path,
            "normalized_config": loaded.normalized_config,
            "dataset_handles": dataset_handles,
            "compiled_request": compiled,
        }

    def train(
        self,
        *,
        config_path: str | Path,
        run_dir: str | Path,
    ) -> dict[str, Any]:
        loaded = self.load_config(config_path)
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        run_name = resolved_run_dir.name
        if not run_name:
            raise CLIWorkflowError("--out must name a run directory")
        self._validate_run_name_match(loaded.normalized_config, run_name=run_name)
        loaded.normalized_config["run"]["name"] = run_name

        store_root = resolved_run_dir / ".dymad-store"
        context = build_default_context(artifact_root=store_root)
        dataset_handles = self._register_config_datasets(
            context=context,
            normalized_config=loaded.normalized_config,
        )
        compiled = compile_training_request(
            facade=context.facade,
            request=self._training_request(
                normalized_config=loaded.normalized_config,
                dataset_handles=dataset_handles,
                run_name=run_name,
            ),
        )
        compiled_summary = context.facade.register_compiled_training_request(
            compiled_request=compiled
        )
        started = context.executor.start_training_run(
            compiled_request_handle=compiled_summary.handle,
            artifact_root=str(resolved_run_dir.parent),
        )
        manifest = self._build_manifest(
            loaded=loaded,
            run_dir=resolved_run_dir,
            store_root=store_root,
            artifact_root=resolved_run_dir.parent,
            dataset_handles=dataset_handles,
            compiled_request_handle=compiled_summary.handle,
            training_run_handle=started.summary.handle,
            training_run=started.training_run,
        )
        _save_manifest(resolved_run_dir, manifest)
        return {
            "manifest": manifest,
            "started": started,
            "manifest_path": str(resolved_run_dir / MANIFEST_FILENAME),
        }

    def status(self, *, run_dir: str | Path) -> dict[str, Any]:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        manifest = _load_manifest(resolved_run_dir)
        context = _context_from_manifest(manifest)
        handle = _require_string(
            manifest.get("training_run_handle"),
            field="manifest.training_run_handle",
        )
        described = context.executor.describe_training_run(training_run_handle=handle)
        self._update_manifest_from_training_run(manifest, training_run=described.training_run)
        _save_manifest(resolved_run_dir, manifest)
        return {
            "manifest": manifest,
            "status": described,
        }

    def read_log(
        self,
        *,
        run_dir: str | Path,
        offset: int = 0,
        max_bytes: int = 65536,
    ) -> dict[str, Any]:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        manifest = _load_manifest(resolved_run_dir)
        context = _context_from_manifest(manifest)
        handle = _require_string(
            manifest.get("training_run_handle"),
            field="manifest.training_run_handle",
        )
        chunk = context.executor.read_training_run_log(
            training_run_handle=handle,
            offset=offset,
            max_bytes=max_bytes,
        )
        described = context.executor.describe_training_run(training_run_handle=handle)
        self._update_manifest_from_training_run(manifest, training_run=described.training_run)
        _save_manifest(resolved_run_dir, manifest)
        return {
            "manifest": manifest,
            "log": chunk,
            "status": described,
        }

    def wait_for_training(
        self,
        *,
        run_dir: str | Path,
        stream=None,
        poll_interval: float = 0.2,
    ) -> dict[str, Any]:
        offset = 0
        latest: dict[str, Any] | None = None
        while True:
            latest = self.read_log(run_dir=run_dir, offset=offset)
            log_result = latest["log"]
            if log_result.text and stream is not None:
                stream.write(log_result.text)
                stream.flush()
            offset = log_result.next_offset
            status = latest["status"].training_run.status
            if status in {TrainingRunStatus.SUCCEEDED, TrainingRunStatus.FAILED} and log_result.eof:
                return latest
            time.sleep(poll_interval)

    def evaluate(
        self,
        *,
        run_dir: str | Path,
        test_data: str | Path | None = None,
    ) -> dict[str, Any]:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        manifest = _load_manifest(resolved_run_dir)
        context = _context_from_manifest(manifest)
        status = self.status(run_dir=resolved_run_dir)
        manifest = cast(dict[str, Any], status["manifest"])
        checkpoint_handle = manifest.get("checkpoint_handle")
        if not isinstance(checkpoint_handle, str) or not checkpoint_handle:
            raise CLIWorkflowError("training run does not have a checkpoint handle yet")

        normalized_config = _require_mapping(
            manifest.get("normalized_config"),
            field="manifest.normalized_config",
        )
        dataset_handles = _require_mapping(
            manifest.get("dataset_handles"),
            field="manifest.dataset_handles",
        )
        dataset_paths = _require_mapping(
            manifest.get("dataset_paths"),
            field="manifest.dataset_paths",
        )
        if test_data is None:
            test_dataset_handle = dataset_handles.get("test")
            if not isinstance(test_dataset_handle, str):
                raise CLIWorkflowError(
                    "no test dataset is recorded; pass --test-data or add data.test.path"
                )
        else:
            test_path = Path(test_data).expanduser()
            if not test_path.is_absolute():
                test_path = Path.cwd() / test_path
            test_path = test_path.resolve()
            data_config = _require_mapping(
                normalized_config.get("data"), field="normalized_config.data"
            )
            test_config = cast(dict[str, Any], data_config.get("test", {}))
            kind = cast(str, test_config.get("kind", "regular"))
            format_name = cast(str, test_config.get("format", "npz"))
            summary = context.facade.register_dataset_file(
                path=str(test_path),
                kind=kind,
                format=format_name,
            )
            test_dataset_handle = summary.handle
            dataset_handles["test_override"] = test_dataset_handle
            dataset_paths["test_override"] = str(test_path)

        evaluation_config = _require_mapping(
            normalized_config.get("evaluation"),
            field="normalized_config.evaluation",
        )
        result = context.executor.evaluate_model(
            checkpoint_handle=checkpoint_handle,
            test_dataset_handle=test_dataset_handle,
            metric=cast(str, evaluation_config.get("metric", "rollout_rmse")),
            artifact_root=str(resolved_run_dir),
            plot_selection=cast(str, evaluation_config.get("plot_selection", "median")),
            max_plots=int(evaluation_config.get("max_plots", 1)),
            predict_kwargs=cast(dict[str, Any], evaluation_config.get("predict_kwargs", {})),
        )
        evaluations = list(manifest.get("evaluation_handles", []))
        evaluations.append(result.evaluation_summary.handle)
        manifest["evaluation_handles"] = evaluations
        manifest["dataset_handles"] = dataset_handles
        manifest["dataset_paths"] = dataset_paths
        manifest["latest_evaluation"] = _json_safe(result)
        _save_manifest(resolved_run_dir, manifest)
        return {
            "manifest": manifest,
            "evaluation": result,
        }

    def report(self, *, run_dir: str | Path) -> dict[str, Any]:
        status = self.status(run_dir=run_dir)
        manifest = cast(dict[str, Any], status["manifest"])
        return {
            "run_dir": manifest.get("run_dir"),
            "status": manifest.get("status"),
            "model_key": manifest.get("normalized_config", {}).get("model_key"),
            "training_run_handle": manifest.get("training_run_handle"),
            "compiled_request_handle": manifest.get("compiled_request_handle"),
            "checkpoint_handle": manifest.get("checkpoint_handle"),
            "metrics": manifest.get("metrics", {}),
            "evaluation_handles": manifest.get("evaluation_handles", []),
            "artifacts": manifest.get("artifacts", {}),
            "manifest": manifest,
        }

    def _register_config_datasets(
        self,
        *,
        context: ExecutionContext,
        normalized_config: dict[str, Any],
    ) -> dict[str, str]:
        data = _require_mapping(normalized_config.get("data"), field="data")
        handles: dict[str, str] = {}
        for split in ("train", "valid", "test"):
            if split not in data:
                continue
            dataset = _require_mapping(data[split], field=f"data.{split}")
            summary = context.facade.register_dataset_file(
                path=cast(str, dataset["path"]),
                format=cast(str, dataset.get("format", "npz")),
                kind=cast(str, dataset.get("kind", "regular")),
            )
            handles[split] = summary.handle
        return handles

    def _training_request(
        self,
        *,
        normalized_config: dict[str, Any],
        dataset_handles: dict[str, str],
        run_name: str | None,
    ) -> TrainingRequest:
        run = _require_mapping(normalized_config.get("run"), field="run")
        return TrainingRequest(
            train_dataset_handle=dataset_handles["train"],
            valid_dataset_handle=dataset_handles.get("valid"),
            model_key=cast(str, normalized_config["model_key"]),
            reference_profile=cast(str | None, normalized_config.get("reference_profile")),
            overrides=cast(dict[str, Any] | None, normalized_config.get("overrides")),
            run_name=run_name,
            seed=cast(int | None, run.get("seed")),
            device=cast(str, run.get("device", "auto")),
            max_workers=int(run.get("max_workers", 1)),
        )

    def _validate_run_name_match(self, normalized_config: dict[str, Any], *, run_name: str) -> None:
        configured = normalized_config.get("run", {}).get("name")
        if configured is not None and configured != run_name:
            raise CLIWorkflowError(
                f"run.name '{configured}' must match --out directory name '{run_name}'"
            )

    def _build_manifest(
        self,
        *,
        loaded: LoadedCLIConfig,
        run_dir: Path,
        store_root: Path,
        artifact_root: Path,
        dataset_handles: dict[str, str],
        compiled_request_handle: str,
        training_run_handle: str,
        training_run,
    ) -> dict[str, Any]:
        data = _require_mapping(loaded.normalized_config.get("data"), field="data")
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_config_path": loaded.source_config_path,
            "normalized_config": loaded.normalized_config,
            "run_dir": str(run_dir),
            "store_root": str(store_root),
            "artifact_root": str(artifact_root),
            "dataset_paths": {
                split: cast(dict[str, Any], data[split])["path"] for split in dataset_handles
            },
            "dataset_handles": dataset_handles,
            "compiled_request_handle": compiled_request_handle,
            "training_run_handle": training_run_handle,
            "evaluation_handles": [],
        }
        self._update_manifest_from_training_run(manifest, training_run=training_run)
        return manifest

    def _update_manifest_from_training_run(self, manifest: dict[str, Any], *, training_run) -> None:
        manifest["status"] = training_run.status.value
        manifest["checkpoint_handle"] = training_run.checkpoint_handle
        manifest["metrics"] = dict(training_run.metrics)
        manifest["artifacts"] = dict(training_run.artifacts)
        manifest["training_run"] = _json_safe(training_run)
