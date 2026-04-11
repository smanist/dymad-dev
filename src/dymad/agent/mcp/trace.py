"""Tracing helpers for public MCP tool calls."""

from __future__ import annotations

import copy
import inspect
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, TypeAlias

TracePath: TypeAlias = list[str | int]

_TRACE_VERSION = 1
_HANDLE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("checkpoint", "chk_", "checkpoint_handle"),
    ("dataset", "ds_", "dataset_handle"),
    ("training_run", "run_", "training_run_handle"),
    ("evaluation", "eval_", "evaluation_handle"),
    ("prediction_request", "pred_", "prediction_request_handle"),
    ("prediction_result", "predres_", "prediction_result_handle"),
    ("spectral_snapshot", "specsnap_", "spectral_snapshot_handle"),
)
_HANDLE_RE = re.compile(r"^(chk|ds|run|eval|pred|predres|specsnap)_[a-z0-9]{6,}$")


@dataclass(frozen=True)
class ArgBinding:
    path: TracePath
    source_handle: str
    variable_name: str


@dataclass(frozen=True)
class HandleBinding:
    handle: str
    kind: str
    variable_name: str
    response_path: TracePath


@dataclass(frozen=True)
class ArtifactPath:
    path: TracePath
    value: str


@dataclass(frozen=True)
class ResultSummary:
    handles: list[HandleBinding]
    artifact_paths: list[ArtifactPath]
    raw_response: Any


@dataclass(frozen=True)
class TraceEvent:
    trace_version: int
    sequence_id: int
    tool_name: str
    args: dict[str, Any]
    arg_bindings: list[ArgBinding]
    result_summary: ResultSummary
    ok: bool
    error: dict[str, str] | None
    started_at: str
    finished_at: str


class TraceRecorder(Protocol):
    """Protocol for opt-in MCP tool tracing."""

    def wrap_tool(self, tool_name: str, fn: Any) -> Any:
        """Return a wrapped tool callable that records trace events."""


class JSONLTraceRecorder:
    """Append-only recorder for public MCP tool calls."""

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        replay_script_path: str | PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_script_path = (
            self.path.with_suffix(".py")
            if replay_script_path is None
            else Path(replay_script_path).expanduser().resolve()
        )
        self.replay_script_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._next_sequence_id = 1
        self._handle_variables: dict[str, str] = {}
        self._kind_counts: dict[str, int] = {}
        self._hydrate_existing_trace()
        self._write_replay_script()

    def wrap_tool(self, tool_name: str, fn: Any) -> Any:
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            call_args = dict(bound.arguments)
            started_at = _timestamp()
            sequence_id = self._reserve_sequence_id()
            arg_bindings = self._bind_arguments(call_args)
            response: Any = None
            error: dict[str, str] | None = None
            try:
                response = fn(*args, **kwargs)
                return response
            except Exception as exc:
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise
            finally:
                finished_at = _timestamp()
                ok = error is None
                result_error = error
                if isinstance(response, dict):
                    ok = bool(response.get("ok", error is None))
                    if not ok:
                        payload_error = response.get("error")
                        if isinstance(payload_error, dict):
                            result_error = {
                                "type": str(payload_error.get("type", "UnknownError")),
                                "message": str(payload_error.get("message", "")),
                            }
                event = TraceEvent(
                    trace_version=_TRACE_VERSION,
                    sequence_id=sequence_id,
                    tool_name=tool_name,
                    args=copy.deepcopy(call_args),
                    arg_bindings=arg_bindings,
                    result_summary=self._summarize_response(response),
                    ok=ok,
                    error=result_error,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                self._append_event(event)

        return wrapped

    def _hydrate_existing_trace(self) -> None:
        if not self.path.is_file():
            return
        for event in load_trace_events(self.path):
            self._next_sequence_id = max(self._next_sequence_id, event.sequence_id + 1)
            for handle_binding in event.result_summary.handles:
                self._handle_variables[handle_binding.handle] = handle_binding.variable_name
                self._kind_counts[handle_binding.kind] = max(
                    self._kind_counts.get(handle_binding.kind, 0),
                    _variable_index(handle_binding.variable_name),
                )

    def _reserve_sequence_id(self) -> int:
        with self._lock:
            sequence_id = self._next_sequence_id
            self._next_sequence_id += 1
            return sequence_id

    def _bind_arguments(self, call_args: dict[str, Any]) -> list[ArgBinding]:
        with self._lock:
            known_handles = dict(self._handle_variables)
        bindings: list[ArgBinding] = []
        for key, value in call_args.items():
            bindings.extend(_collect_arg_bindings(value, [key], known_handles))
        return bindings

    def _summarize_response(self, response: Any) -> ResultSummary:
        raw_response = copy.deepcopy(response)
        handles = self._discover_handles(raw_response)
        artifact_paths = self._discover_artifact_paths(raw_response)
        return ResultSummary(
            handles=handles,
            artifact_paths=artifact_paths,
            raw_response=raw_response,
        )

    def _discover_handles(self, response: Any) -> list[HandleBinding]:
        discovered: dict[str, TracePath] = {}
        _walk_response(response, [], lambda value, path: _collect_handle(discovered, value, path))
        bindings: list[HandleBinding] = []
        with self._lock:
            for handle, response_path in discovered.items():
                kind = _handle_kind(handle)
                variable_name = self._handle_variables.get(handle)
                if variable_name is None:
                    variable_name = _new_variable_name(kind, self._kind_counts)
                    self._handle_variables[handle] = variable_name
                bindings.append(
                    HandleBinding(
                        handle=handle,
                        kind=kind,
                        variable_name=variable_name,
                        response_path=response_path,
                    )
                )
        return bindings

    def _discover_artifact_paths(self, response: Any) -> list[ArtifactPath]:
        discovered: list[ArtifactPath] = []

        def visitor(value: Any, path: TracePath) -> None:
            if _looks_like_artifact_path(path, value):
                discovered.append(ArtifactPath(path=path, value=value))

        _walk_response(response, [], visitor)
        return discovered

    def _append_event(self, event: TraceEvent) -> None:
        payload = json.dumps(asdict(event), sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
        self._write_replay_script()

    def _write_replay_script(self) -> None:
        from dymad.agent.mcp.replay import generate_replay_script

        generate_replay_script(
            trace_path=self.path,
            output_path=self.replay_script_path,
        )


def load_trace_events(path: str | PathLike[str]) -> list[TraceEvent]:
    """Load one trace file and sort events by sequence id."""
    trace_path = Path(path).expanduser().resolve()
    if not trace_path.is_file():
        return []
    events: list[TraceEvent] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            events.append(
                TraceEvent(
                    trace_version=int(payload["trace_version"]),
                    sequence_id=int(payload["sequence_id"]),
                    tool_name=str(payload["tool_name"]),
                    args=dict(payload["args"]),
                    arg_bindings=[
                        ArgBinding(
                            path=list(item["path"]),
                            source_handle=str(item["source_handle"]),
                            variable_name=str(item["variable_name"]),
                        )
                        for item in payload.get("arg_bindings", [])
                    ],
                    result_summary=ResultSummary(
                        handles=[
                            HandleBinding(
                                handle=str(item["handle"]),
                                kind=str(item["kind"]),
                                variable_name=str(item["variable_name"]),
                                response_path=list(item["response_path"]),
                            )
                            for item in payload.get("result_summary", {}).get("handles", [])
                        ],
                        artifact_paths=[
                            ArtifactPath(
                                path=list(item["path"]),
                                value=str(item["value"]),
                            )
                            for item in payload.get("result_summary", {}).get("artifact_paths", [])
                        ],
                        raw_response=payload.get("result_summary", {}).get("raw_response"),
                    ),
                    ok=bool(payload["ok"]),
                    error=(
                        None
                        if payload.get("error") is None
                        else {
                            "type": str(payload["error"]["type"]),
                            "message": str(payload["error"]["message"]),
                        }
                    ),
                    started_at=str(payload["started_at"]),
                    finished_at=str(payload["finished_at"]),
                )
            )
    return sorted(events, key=lambda item: item.sequence_id)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _collect_arg_bindings(
    value: Any,
    path: TracePath,
    known_handles: dict[str, str],
) -> list[ArgBinding]:
    if isinstance(value, str):
        variable_name = known_handles.get(value)
        if variable_name is None:
            return []
        return [ArgBinding(path=path, source_handle=value, variable_name=variable_name)]
    if isinstance(value, dict):
        bindings: list[ArgBinding] = []
        for key, item in value.items():
            bindings.extend(_collect_arg_bindings(item, [*path, key], known_handles))
        return bindings
    if isinstance(value, list):
        bindings = []
        for index, item in enumerate(value):
            bindings.extend(_collect_arg_bindings(item, [*path, index], known_handles))
        return bindings
    return []


def _walk_response(value: Any, path: TracePath, visitor: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = [*path, key]
            visitor(item, next_path)
            _walk_response(item, next_path, visitor)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_path = [*path, index]
            visitor(item, next_path)
            _walk_response(item, next_path, visitor)


def _collect_handle(
    discovered: dict[str, TracePath],
    value: Any,
    path: TracePath,
) -> None:
    if isinstance(value, str) and _HANDLE_RE.match(value) and value not in discovered:
        discovered[value] = path


def _handle_kind(handle: str) -> str:
    for kind, prefix, _variable_base in _HANDLE_PATTERNS:
        if handle.startswith(prefix):
            return kind
    raise ValueError(f"unsupported handle kind: {handle}")


def _new_variable_name(kind: str, kind_counts: dict[str, int]) -> str:
    variable_base = next(
        base for active_kind, _prefix, base in _HANDLE_PATTERNS if active_kind == kind
    )
    next_index = kind_counts.get(kind, 0) + 1
    kind_counts[kind] = next_index
    return f"{variable_base}_{next_index}"


def _variable_index(variable_name: str) -> int:
    _, _, suffix = variable_name.rpartition("_")
    return int(suffix) if suffix.isdigit() else 0


def _looks_like_artifact_path(path: TracePath, value: Any) -> bool:
    if not isinstance(value, str) or _HANDLE_RE.match(value):
        return False
    key = _nearest_key(path)
    if key is None:
        return False
    return (
        key == "path"
        or key == "plot_paths"
        or key.endswith("_path")
        or key.endswith("_paths")
        or key.endswith("_dir")
        or key.endswith("_root")
    )


def _nearest_key(path: TracePath) -> str | None:
    for item in reversed(path):
        if isinstance(item, str):
            return item
    return None
