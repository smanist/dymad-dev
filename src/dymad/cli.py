"""Package-level DyMAD CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from dymad.agent.app import CLI_CONFIG_SCHEMA, CLIWorkflowError, CLIWorkflowService
from dymad.agent.registry import (
    list_analysis_capabilities,
    list_evaluation_capabilities,
    list_loss_capabilities,
    list_model_capabilities,
    list_profile_capabilities,
    list_training_capabilities,
)


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


def _print_json(value: Any) -> None:
    print(json.dumps(_json_safe(value), indent=2, sort_keys=True))


def _print_error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _registry_payload(kind: str) -> dict[str, Any]:
    if kind == "models":
        return {"kind": kind, "items": list_model_capabilities()}
    if kind == "losses":
        return {"kind": kind, "items": list_loss_capabilities()}
    if kind == "profiles":
        return {"kind": kind, "items": list_profile_capabilities()}
    if kind == "training":
        return {"kind": kind, "items": list_training_capabilities()}
    if kind == "analyses":
        return {"kind": kind, "items": list_analysis_capabilities()}
    if kind == "evaluations":
        return {"kind": kind, "items": list_evaluation_capabilities()}
    raise CLIWorkflowError(f"unsupported registry kind: {kind}")


def _print_registry(payload: dict[str, Any]) -> None:
    for item in _json_safe(payload["items"]):
        print(f"{item['key']}\t{item.get('name', '')}")


def _print_status(result: dict[str, Any]) -> None:
    manifest = result["manifest"]
    print(f"run: {manifest['run_dir']}")
    print(f"status: {manifest.get('status')}")
    print(f"training_run: {manifest.get('training_run_handle')}")
    checkpoint = manifest.get("checkpoint_handle")
    if checkpoint:
        print(f"checkpoint: {checkpoint}")
    metrics = manifest.get("metrics") or {}
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")


def _print_report(report: dict[str, Any]) -> None:
    print(f"run: {report['run_dir']}")
    print(f"status: {report['status']}")
    print(f"model: {report['model_key']}")
    print(f"training_run: {report['training_run_handle']}")
    if report.get("checkpoint_handle"):
        print(f"checkpoint: {report['checkpoint_handle']}")
    metrics = report.get("metrics") or {}
    if metrics:
        print("metrics:")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value}")
    evaluations = report.get("evaluation_handles") or []
    if evaluations:
        print("evaluations:")
        for handle in evaluations:
            print(f"  {handle}")


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit structured JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dymad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="inspect or validate CLI configs")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("schema", help="emit the CLI config JSON Schema")
    validate_parser = config_subparsers.add_parser("validate", help="validate a CLI config")
    validate_parser.add_argument("config")
    validate_parser.add_argument("--out", help="optional run directory for run.name validation")
    _add_json_flag(validate_parser)

    registry_parser = subparsers.add_parser("registry", help="inspect agent registries")
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command", required=True)
    registry_list = registry_subparsers.add_parser("list", help="list one registry")
    registry_list.add_argument(
        "kind",
        choices=["models", "losses", "profiles", "training", "analyses", "evaluations"],
    )
    _add_json_flag(registry_list)

    train_parser = subparsers.add_parser("train", help="start a user-mode training run")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument(
        "--out",
        help="run directory; defaults to CONFIG's directory plus run.name when run.name is set",
    )
    train_parser.add_argument("--detach", action="store_true")
    _add_json_flag(train_parser)

    status_parser = subparsers.add_parser("status", help="describe a training run")
    status_parser.add_argument("--run", required=True)
    _add_json_flag(status_parser)

    log_parser = subparsers.add_parser("log", help="read a training run log")
    log_parser.add_argument("--run", required=True)
    log_parser.add_argument("--follow", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="evaluate a completed training run")
    eval_parser.add_argument("--run", required=True)
    eval_parser.add_argument("--test-data")
    _add_json_flag(eval_parser)

    report_parser = subparsers.add_parser("report", help="summarize a run")
    report_parser.add_argument("--run", required=True)
    _add_json_flag(report_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = CLIWorkflowService()

    try:
        if args.command == "config" and args.config_command == "schema":
            _print_json(CLI_CONFIG_SCHEMA)
            return 0

        if args.command == "config" and args.config_command == "validate":
            result = service.validate_config(args.config, run_dir=args.out)
            if args.json:
                _print_json(result)
            else:
                compiled = result["compiled_request"]
                print("valid")
                print(f"model: {compiled.model.key}")
                print(f"profile: {compiled.profile.key}")
                print(f"run_name: {compiled.effective_run_name}")
            return 0

        if args.command == "registry" and args.registry_command == "list":
            payload = _registry_payload(args.kind)
            if args.json:
                _print_json(payload)
            else:
                _print_registry(payload)
            return 0

        if args.command == "train":
            result = service.train(config_path=args.config, run_dir=args.out)
            run_dir = result["manifest"]["run_dir"]
            if args.detach:
                if args.json:
                    _print_json(result)
                else:
                    manifest = result["manifest"]
                    print(f"started: {manifest['training_run_handle']}")
                    print(f"run: {manifest['run_dir']}")
                    print(f"manifest: {result['manifest_path']}")
                return 0

            final = service.wait_for_training(
                run_dir=run_dir,
                stream=None if args.json else sys.stdout,
            )
            if args.json:
                _print_json(final)
            else:
                _print_status(final)
            return 0 if final["status"].training_run.status == "SUCCEEDED" else 1

        if args.command == "status":
            result = service.status(run_dir=args.run)
            if args.json:
                _print_json(result)
            else:
                _print_status(result)
            return 0

        if args.command == "log":
            if args.follow:
                final = service.wait_for_training(run_dir=args.run, stream=sys.stdout)
                return 0 if final["status"].training_run.status == "SUCCEEDED" else 1
            result = service.read_log(run_dir=args.run)
            sys.stdout.write(result["log"].text)
            return 0

        if args.command == "eval":
            result = service.evaluate(run_dir=args.run, test_data=args.test_data)
            if args.json:
                _print_json(result)
            else:
                evaluation = result["evaluation"]
                print(f"evaluation: {evaluation.evaluation_summary.handle}")
                for key, value in sorted(evaluation.metrics.items()):
                    print(f"{key}: {value}")
            return 0

        if args.command == "report":
            result = service.report(run_dir=args.run)
            if args.json:
                _print_json(result)
            else:
                _print_report(result)
            return 0

    except CLIWorkflowError as exc:
        _print_error(str(exc))
        return 1
    except Exception as exc:
        _print_error(f"{type(exc).__name__}: {exc}")
        return 1

    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
