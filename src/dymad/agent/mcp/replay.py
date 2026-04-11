"""Generate replay scripts from MCP tool-call traces."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from dymad.agent.mcp.trace import ArgBinding, HandleBinding, TraceEvent, load_trace_events


@dataclass(frozen=True)
class _CodeRef:
    expression: str


def generate_replay_script(
    *,
    trace_path: str | PathLike[str],
    output_path: str | PathLike[str],
) -> Path:
    """Write one executable Python replay script from a JSONL trace."""
    trace_file = Path(trace_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        render_replay_script(load_trace_events(trace_file), trace_path=trace_file),
        encoding="utf-8",
    )
    output_file.chmod(output_file.stat().st_mode | 0o111)
    return output_file


def render_replay_script(
    events: Sequence[TraceEvent],
    *,
    trace_path: str | PathLike[str],
) -> str:
    """Render one standalone replay script."""
    trace_file = Path(trace_path).expanduser().resolve()
    lines = [
        "#!/usr/bin/env python3",
        '"""Replay one recorded sequence of MCP tool calls via DemoTools."""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "",
        "from dymad.agent.exec.context import build_default_context",
        "from dymad.agent.mcp import DemoTools",
        "",
        "",
        "def main() -> None:",
        '    parser = argparse.ArgumentParser(description="Replay traced MCP tool calls.")',
        '    parser.add_argument("--artifact-root", default=".dymad/replay_artifacts")',
        "    args = parser.parse_args()",
        "",
        f"    trace_path = {trace_file.as_posix()!r}",
        "    context = build_default_context(artifact_root=args.artifact_root)",
        "    tools = DemoTools(context=context)",
        "    del trace_path",
        "",
    ]
    for event in events:
        lines.extend(_render_event(event))
        lines.append("")
    lines.extend(
        [
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for trace-to-replay generation."""
    parser = argparse.ArgumentParser(description="Generate a Python replay script from a trace.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    generate_replay_script(trace_path=args.trace, output_path=args.output)


def _render_event(event: TraceEvent) -> list[str]:
    lines = [f"    # sequence {event.sequence_id}: {event.tool_name}"]
    if not event.ok and event.result_summary.raw_response is None:
        kwargs_expr = _render_literal(_args_with_bindings(event.args, event.arg_bindings))
        lines.append(f"    kwargs_{event.sequence_id} = {kwargs_expr}")
        lines.append("    try:")
        lines.append(f"        tools.{event.tool_name}(**kwargs_{event.sequence_id})")
        lines.append(f"        raise AssertionError('expected {event.error!r}')")
        assert event.error is not None
        lines.append("    except Exception as exc:")
        lines.append(f"        assert type(exc).__name__ == {event.error['type']!r}")
        return lines

    kwargs_expr = _render_literal(_args_with_bindings(event.args, event.arg_bindings))
    lines.append(f"    kwargs_{event.sequence_id} = {kwargs_expr}")
    lines.append(
        f"    response_{event.sequence_id} = tools.{event.tool_name}(**kwargs_{event.sequence_id})"
    )
    lines.append(f'    assert response_{event.sequence_id}["ok"] is {event.ok!r}')
    if event.ok:
        for handle_binding in event.result_summary.handles:
            lines.extend(_render_handle_binding(event.sequence_id, handle_binding))
        return lines

    assert event.error is not None
    lines.append(
        f'    assert response_{event.sequence_id}["error"]["type"] == {event.error["type"]!r}'
    )
    return lines


def _render_handle_binding(sequence_id: int, handle_binding: HandleBinding) -> list[str]:
    response_ref = _render_lookup(f"response_{sequence_id}", handle_binding.response_path)
    prefix = _expected_prefix(handle_binding.kind)
    return [
        f"    {handle_binding.variable_name} = {response_ref}",
        f"    assert isinstance({handle_binding.variable_name}, str)",
        f"    assert {handle_binding.variable_name}.startswith({prefix!r})",
    ]


def _expected_prefix(kind: str) -> str:
    return {
        "checkpoint": "chk_",
        "dataset": "ds_",
        "training_run": "run_",
        "evaluation": "eval_",
        "prediction_request": "pred_",
        "prediction_result": "predres_",
        "spectral_snapshot": "specsnap_",
    }[kind]


def _render_lookup(base: str, path: list[str | int]) -> str:
    rendered = base
    for item in path:
        if isinstance(item, int):
            rendered = f"{rendered}[{item}]"
        else:
            rendered = f"{rendered}[{item!r}]"
    return rendered


def _args_with_bindings(
    args: dict[str, Any],
    bindings: list[ArgBinding],
) -> dict[str, Any]:
    bound_args = copy.deepcopy(args)
    for binding in bindings:
        _set_path(bound_args, binding.path, _CodeRef(binding.variable_name))
    return bound_args


def _set_path(target: Any, path: list[str | int], value: Any) -> None:
    current = target
    for item in path[:-1]:
        current = current[item]
    current[path[-1]] = value


def _render_literal(value: Any) -> str:
    if isinstance(value, _CodeRef):
        return value.expression
    if isinstance(value, dict):
        items = ", ".join(
            f"{_render_literal(key)}: {_render_literal(item)}" for key, item in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, list):
        items = ", ".join(_render_literal(item) for item in value)
        return "[" + items + "]"
    return repr(value)


if __name__ == "__main__":
    main()
