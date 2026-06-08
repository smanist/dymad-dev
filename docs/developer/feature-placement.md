# Feature Placement

## Purpose

This is the shortest possible answer to "where should this feature go?".

Read [architecture.md](architecture.md) first if the
boundary between `agent/*` and the legacy runtime is unclear.
Use this file for fast triage before reading the longer
[algorithm-integration-guide.md](algorithm-integration-guide.md). After choosing the matching row
below, read only the guide sections relevant to that change type for interface, schema, test,
example, and benchmark expectations.

Maintenance rule:

- If a change adds a new workflow, capability family, persisted artifact type, or changes the
  recommended ownership for a feature, update this table in the same change.

## Decision Table

| You are adding | Primary files/packages | Notes |
| --- | --- | --- |
| A new MCP tool that exposes an existing workflow | `src/dymad/agent/mcp/user_tools.py` or `src/dymad/agent/mcp/demo_tools.py`, then `src/dymad/agent/mcp/server.py` | Keep business logic out of `server.py`; it should register tools, not implement them |
| A new package CLI command that exposes an existing user workflow | `src/dymad/cli.py` for argument parsing and `src/dymad/agent/app/*` for shared path/config workflow assembly | Keep `cli.py` thin; use registry/compiler/executor/facade/store boundaries instead of routing directly to trainers |
| A new standalone hyperparameter search primitive or tuning artifact behavior | `src/dymad/tuning/*` | Keep it evaluator-driven and independent of trainer, MCP, and CLI assumptions; trainer CV can adapt to it |
| A new standalone convergence-study primitive, aggregation, rate-fit, or diagnostic behavior | `src/dymad/studies/convergence/*` | Keep study orchestration independent of MCP/CLI exposure; user-facing exposure would be a later registry/compiler/executor change |
| A new user-mode training capability or model family choice | `src/dymad/agent/registry/models.py`, `src/dymad/agent/registry/workflows.py`, `src/dymad/agent/registry/training_schema.py` | Add stable keys, summaries, dataset compatibility, allowed overrides, CV support metadata, translation guidance, constraint notes, and examples |
| A new user-mode request field or override validation rule | `src/dymad/agent/compiler/training.py`, `src/dymad/agent/compiler/schemas.py`, maybe `src/dymad/agent/registry/training_schema.py` | Compiler owns validation and normalization of user input, including `overrides.cv` sweep/search/bounds/selection config and profile-default phase preservation |
| A new analysis workflow | `src/dymad/agent/registry/analyses.py`, `src/dymad/agent/compiler/analysis.py`, `src/dymad/agent/exec/workflow.py` | Registry declares the workflow, compiler validates inputs, executor runs it |
| A new persisted request/artifact handle | `src/dymad/agent/store/object_store.py`, `src/dymad/agent/store/filesystem_artifact_store.py`, `src/dymad/agent/facade/handles.py`, `src/dymad/agent/facade/operations.py` | Add both record persistence and typed facade accessors |
| A new execution workflow that stitches existing pieces together | `src/dymad/agent/exec/workflow.py` or another focused module under `src/dymad/agent/exec` | Executor owns orchestration and workflow results; durable subprocess entrypoints such as async training workers belong in focused modules under `agent/exec` |
| A new model family or variant implementation | `src/dymad/models/*` and then `src/dymad/agent/registry/models.py` | Implementation first, registry exposure second |
| A new training phase kind | `src/dymad/training/phases.py`, `src/dymad/training/phase_pipeline.py`, maybe compiler/registry schema files | Runtime phase semantics do not belong in MCP adapters |
| A new checkpoint compatibility behavior | `src/dymad/io/checkpoint.py`, maybe `src/dymad/agent/exec/workflow.py` and facade/store files | Keep the public `dymad.io` behavior accurate even if the executor only plans part of the flow |
| A new low-level runtime, transform, or batching behavior | `src/dymad/core/*`, `src/dymad/io/*`, `src/dymad/training/*` | Do not route this through `agent/*` unless the boundary contract changes |
| A new numerical primitive or solver | `src/dymad/numerics/*` | Keep numerics isolated from agent/MCP concerns |
| A new spectral runtime or adapter behavior | `src/dymad/sako/*`, then registry/executor files if you want it exposed as an analysis workflow | Library behavior first, user-facing exposure second |

## Follow-Up Guide Sections

After the decision table identifies the likely owner, use this map to read the focused parts of
[algorithm-integration-guide.md](algorithm-integration-guide.md):

| Triage result | Read these guide sections |
| --- | --- |
| MCP tool, user-facing capability, request validation, execution workflow, or persisted handle | `MCP Tools`, `Agent-Facing Schema Checklist`, `Data Structures To Consume And Return`, `Mandatory Tests`, `What Not To Touch` |
| Model family or variant | `Models and Prediction`, `Agent-Facing Schema Checklist` if user-mode exposed, `Mandatory Tests`, `Naming Conventions` |
| Training phase or optimizer behavior | `Training Algorithms and Phases`, `Agent-Facing Schema Checklist` if user-mode exposed, `Mandatory Tests`, `Examples` |
| Analysis workflow or spectral adapter | `Analysis Workflows`, `Agent-Facing Schema Checklist`, `Mandatory Tests`, `Examples` |
| Core runtime, transform, batching, or data structure | `Core Data Structures`, `Data Structures To Consume And Return`, `Mandatory Tests` |
| Numerical primitive or solver | `Numerical Algorithms`, `Mandatory Tests`, `Benchmarks` when the hot path changes |
| Runnable example or benchmark | `Examples`, `Benchmarks`, `Naming Conventions` |

## Rules of Thumb

- If the change introduces a new stable user-facing key, start in `agent/registry/*`.
- If the change turns file paths or CLI config into a workflow request, put the path/config
  adaptation in `agent/app/*`, then call the same compiler/executor path as MCP user mode.
- If the change teaches clients how to translate natural language into a user-mode request, put
  that guidance in `agent/registry/*`, not in MCP adapters or runtime code.
- If the change validates or normalizes user input, start in `agent/compiler/*`.
- If the change coordinates multiple steps, start in `agent/exec/*`.
- If the change stores something under a handle, start in `agent/store/*` and `agent/facade/*`.
- If the change is model/math/runtime behavior, it probably does not belong in `agent/*`.
- If the change is generic hyperparameter search or convergence-study machinery, it belongs in
  `tuning/*` or `studies/convergence/*`; only transport exposure belongs in `agent/*`.
- If you need to touch `server.py`, you usually also need to touch a tool adapter first.

## Common Mistakes

- Putting workflow logic directly in `src/dymad/agent/mcp/server.py`
- Routing package CLI configs directly into trainers instead of through `agent/app`, compiler, and
  executor boundaries
- Adding user-mode string parsing to `demo_tools.py` instead of `compiler/*`
- Adding registry-style metadata ad hoc inside executor methods
- Adding persistence behavior directly in executor methods instead of the store/facade layer
- Treating `agent/*` as the place for numerical or model implementation details

## Test Pointers

| Change type | Start with these tests |
| --- | --- |
| MCP mode/tool registration | `tests/test_mcp_server_modes.py`, `tests/test_demo_tools.py`, `tests/test_mcp_user_tools.py` |
| Package CLI commands and path-first user workflows | `tests/test_agent_cli.py` |
| Training capability/compiler changes | `tests/test_training_compiler.py`, `tests/test_compiled_training_requests.py`, `tests/test_agent_registry.py` |
| Analysis workflow changes | `tests/test_analysis_workflows.py` |
| Persisted handle/store/facade changes | `tests/test_boundary_persistence.py`, `tests/test_checkpoint_e2e_layering.py` |
| Checkpoint boundary changes | `tests/test_public_load_model_boundary.py`, `tests/test_checkpoint_e2e_layering.py` |
| Training runtime/phase changes | `tests/test_training_phase_runtime.py` plus the closest workflow tests |

## If Unsure

Start from the user-facing outcome:

1. Stable capability name or workflow choice: `registry/*`
2. Request validation and compile step: `compiler/*`
3. Execution path: `exec/*`
4. Durable artifact state: `facade/*` and `store/*`
5. Actual model/runtime/math behavior: implementation packages
