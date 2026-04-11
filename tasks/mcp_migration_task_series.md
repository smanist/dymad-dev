# MCP Migration Task Series

## Goal

Turn the migration plan in:

- `docs/mcp_current_state_audit.md`
- `docs/mcp_gap_analysis.md`
- `docs/mcp_target_architecture.md`
- `docs/mcp_migration_plan.md`

into an implementation-ready sequence of tasks with explicit scope, dependencies, deliverables, and acceptance criteria.

This task series is intentionally incremental. It preserves the current boundary stack and typed internals, and it stages new user-facing behavior only after the registry and compiler foundations exist.

## Ground Rules

- Preserve the current `agent.mcp -> agent.exec -> agent.facade -> store` boundary unless a task explicitly changes that boundary.
- Treat current atomic functions and script-era workflows as implementation building blocks, not as the final user-facing API.
- Keep developer-mode/raw tools working until user-mode replacements are implemented and tested.
- Prefer wrapping and extracting over rewriting.
- For each task, add or update tests before marking the task complete.

## Sequence Overview

| Task | Title | Purpose | Depends on |
| --- | --- | --- | --- |
| `T100` | Registry Foundation | consolidate model/workflow/profile metadata | none |
| `T110` | Discovery Tools | expose capability discovery without changing execution | `T100` |
| `T120` | Typed Training Compiler | add typed user-mode compile step for train requests | `T100` |
| `T130` | Compiled Request Persistence | persist compiled requests and integrate with execution | `T120` |
| `T140` | User-Mode MCP Façade | add stable high-level training/eval tools | `T110`, `T130` |
| `T150` | Analysis Capability Foundation | register non-training capabilities and shared execution nodes | `T100`, `T130` |
| `T160` | Spectral Workflow | move spectral analysis onto the registry/compiler/execution path | `T150` |
| `T170` | Transform/Mode Workflow | move script-backed transform/mode analysis behind supported workflows | `T150` |
| `T180` | Developer Mode Split | make raw and high-level MCP surfaces explicit | `T140`, `T160`, `T170` |
| `T190` | Cleanup and Deprecation | simplify skills/docs/tests and de-emphasize brittle raw paths | `T180` |

## `T100` Registry Foundation

## Goal

Create a unified registry layer for model, training, and workflow metadata using the current repo’s existing typed and alias structures.

## Why this task exists

The current repo already has registry fragments:

- `src/dymad/models/collections.py`
- `src/dymad/models/model_spec.py`
- `src/dymad/models/recipes.py`
- `src/dymad/agent/exec/training_profiles.py`

but there is no single agent-facing capability registry. This is the first dependency for everything else.

## Scope

- Add a new package:
  - `src/dymad/agent/registry/__init__.py`
  - `src/dymad/agent/registry/models.py`
  - `src/dymad/agent/registry/workflows.py`
  - `src/dymad/agent/registry/types.py`
- Define registry entry types for:
  - model capabilities
  - training workflow capabilities
  - profile capabilities
- Populate initial entries from:
  - predefined models in `src/dymad/models/collections.py`
  - profile aliases in `src/dymad/agent/exec/training_profiles.py`
- Add canonical user-facing keys such as:
  - `ldm`
  - `kbf`
  - `lti`
- Keep current `model_ref` strings in registry entries as implementation details.

## Deliverables

- typed registry entry dataclasses
- registry-loading functions
- registry accessors with deterministic output
- tests for:
  - all current model/profile mappings
  - dataset-kind compatibility metadata
  - profile inference through the new registry

## Acceptance criteria

- the repo has one authoritative place to ask:
  - which model families exist
  - which dataset kinds they support
  - which default profiles they map to
- registry tests cover at least the current `LDM`, `KBF`, `LTI`, graph variants, and profile aliases
- no existing training or MCP behavior changes yet

## Repo touch points

- `src/dymad/agent/registry/*`
- `src/dymad/agent/exec/training_profiles.py`
- `src/dymad/models/collections.py`
- new tests under `tests/`

## `T110` Discovery Tools

## Goal

Expose the new registry through code-level accessors and MCP discovery tools without changing execution contracts.

## Why this task exists

This is the smallest practical user-visible slice. It reduces repo/code inspection immediately and validates that the registry is usable before compiler work starts.

## Scope

- Add registry query helpers:
  - `list_model_capabilities()`
  - `list_training_capabilities(dataset_handle=None)`
  - `resolve_model_capability(key_or_alias)`
  - `list_profile_capabilities()`
- Add developer-facing MCP tools for discovery.
- Keep current `DemoTools`/`server.py` execution tools unchanged.

## Deliverables

- new registry discovery methods
- MCP tool registration for discovery
- tests for JSON-safe discovery responses

## Acceptance criteria

- the agent can discover supported model families and default profiles through MCP
- discovery tools do not require repo inspection to answer basic capability questions
- existing `train_model` and `evaluate_model` continue to work unchanged

## Repo touch points

- `src/dymad/agent/mcp/server.py`
- likely `src/dymad/agent/mcp/demo_tools.py` or new helper module
- `src/dymad/agent/registry/*`
- new tests under `tests/test_demo_tools.py` or a new discovery test module

## `T120` Typed Training Compiler

## Goal

Add a typed compile step for user-mode training requests that resolves capabilities and validates overrides before execution.

## Why this task exists

This is the main architectural gap. Today, the agent must translate free text into:

- exact `model_ref`
- optional `reference_profile`
- raw nested `config`

This task moves that resolution into code.

## Scope

- Add:
  - `src/dymad/agent/compiler/__init__.py`
  - `src/dymad/agent/compiler/schemas.py`
  - `src/dymad/agent/compiler/training.py`
- Define typed request objects:
  - `TrainingRequest`
  - `CompiledTrainingRequest`
  - compile diagnostics and warnings
- Reuse current logic from `src/dymad/agent/exec/workflow.py` for:
  - profile inference
  - effective-config construction
  - trainer-kind selection
  - reserved-field validation
- Add capability-aware override validation:
  - allow safe user-mode overrides
  - reject runtime-owned and unsupported paths with clear errors

## Deliverables

- typed training compiler
- validation errors with field-level context
- tests for:
  - valid regular/graph compile flows
  - illegal override rejection
  - inferred profiles
  - trainer-kind inference from compiled phases

## Acceptance criteria

- a caller can compile a training request using a stable model-family key instead of raw `model_ref`
- the compiler returns:
  - resolved model capability
  - resolved profile
  - effective config
  - trainer kind
  - warnings
- current executor code is reused rather than duplicated

## Repo touch points

- `src/dymad/agent/compiler/*`
- `src/dymad/agent/exec/workflow.py`
- `src/dymad/agent/registry/*`
- tests under `tests/`

## `T130` Compiled Request Persistence

## Goal

Persist compiled training requests as first-class handles so user-mode MCP tools can plan, inspect, and execute validated requests.

## Why this task exists

Without persisted compiled requests, the compiler remains an in-memory helper. The target architecture needs typed request objects that survive beyond one call boundary.

## Scope

- Extend the store/facade stack with a compiled-request record type.
- Add a new handle family, for example:
  - `trainreq_*`
- Update:
  - `src/dymad/agent/facade/handles.py`
  - `src/dymad/agent/store/object_store.py`
  - `src/dymad/agent/store/filesystem_artifact_store.py`
  - `src/dymad/agent/facade/operations.py`
- Add execution entrypoints that accept compiled request handles.

## Deliverables

- compiled-request record dataclass
- persistence and summary support
- façade methods to:
  - register compiled request
  - describe compiled request
  - fetch compiled request
- tests for persistence and rehydration

## Acceptance criteria

- compiled training requests can be created, persisted, listed, and reloaded by handle
- summaries are JSON-safe and useful for MCP inspection
- execution can consume a compiled request handle directly

## Repo touch points

- `src/dymad/agent/facade/handles.py`
- `src/dymad/agent/facade/operations.py`
- `src/dymad/agent/store/object_store.py`
- `src/dymad/agent/store/filesystem_artifact_store.py`
- `src/dymad/agent/exec/context.py`
- tests under `tests/`

## `T140` User-Mode MCP Façade

## Goal

Introduce a stable user-mode MCP surface for training and evaluation that no longer requires internal strings or raw nested config dicts.

## Scope

- Add:
  - `src/dymad/agent/mcp/user_tools.py`
- Implement high-level tools:
  - `list_training_capabilities`
  - `compile_training_request`
  - `train_compiled_request`
  - `evaluate_checkpoint`
- Route these tools through:
  - registry
  - compiler
  - persisted compiled requests
  - existing executor logic
- Keep current raw/developer tools working in parallel.

## Deliverables

- user-mode tool implementation
- server registration of user-mode tools
- updated tests covering:
  - compile -> train -> evaluate flow
  - graph/regular variants
  - validation failures

## Acceptance criteria

- a normal train/evaluate flow no longer requires:
  - `model_ref`
  - `reference_profile`
  - raw `config`
- the user-mode API is narrower and more stable than the current raw MCP tools
- existing developer/raw execution tools remain functional

## Repo touch points

- `src/dymad/agent/mcp/server.py`
- `src/dymad/agent/mcp/user_tools.py`
- possibly `src/dymad/agent/mcp/demo_tools.py`
- tests under `tests/test_mcp_train_eval_tools.py` or a new user-mode test module

## `T150` Analysis Capability Foundation

## Goal

Create the shared registry/compiler/execution foundation for non-training workflows, starting with analysis and post-processing capability metadata.

## Why this task exists

Today, analysis/postprocess support is mostly discoverable only through:

- `DataInterface`
- `SpectralAnalysis`
- scripts under `scripts/`

The migration needs a common way to represent these as supported capabilities before exposing them through MCP.

## Scope

- Extend the registry package with:
  - `src/dymad/agent/registry/analyses.py`
- Add analysis capability entry types and initial entries for:
  - spectral Koopman analysis
  - transform/mode analysis
- Add analysis compiler scaffolding:
  - `src/dymad/agent/compiler/analysis.py`
- Add analysis-plan record types and handles in store/facade if needed.
- Add execution-node wrappers for analysis workflows.

## Deliverables

- analysis capability registry
- typed analysis request/compiled-request scaffolding
- shared execution-node interfaces for analysis tasks
- tests for analysis capability discovery

## Acceptance criteria

- supported analysis workflows are discoverable via code and MCP
- the registry can represent whether a workflow is:
  - supported
  - experimental
  - script-backed
- later tasks can plug concrete analysis implementations into this foundation

## Repo touch points

- `src/dymad/agent/registry/*`
- `src/dymad/agent/compiler/analysis.py`
- façade/store additions as needed
- tests under `tests/`

## `T160` Spectral Workflow

## Goal

Move spectral analysis from partially layered library code into a supported registry/compiler/execution/MCP workflow.

## Current building blocks to preserve

- `CompatibilityExecutor.plan_spectral_analysis(...)`
- `FacadeOperations.register_spectral_snapshot(...)`
- `CompatibilityExecutor.materialize_spectral_adapter(...)`
- `src/dymad/sako/adapter.py`
- `src/dymad/sako/base.py`

## Scope

- Define a supported analysis capability for spectral Koopman analysis.
- Add a compiled spectral-analysis request type.
- Implement execution nodes that:
  - validate checkpoint/model compatibility
  - prepare snapshots/plans
  - run eigensystem/adapter-backed analysis
  - persist outputs and summaries
- Expose user-mode and developer-mode MCP entrypoints as appropriate.

## Deliverables

- spectral analysis compiler/executor integration
- persisted outputs or summaries for spectral runs
- MCP tools for spectral capability discovery and execution
- tests that cover:
  - plan creation
  - adapter materialization
  - end-to-end spectral analysis workflow

## Acceptance criteria

- spectral analysis is discoverable without reading scripts
- at least one end-to-end spectral workflow runs through the new architecture
- existing spectral library classes remain reusable as internals

## Repo touch points

- `src/dymad/agent/registry/analyses.py`
- `src/dymad/agent/compiler/analysis.py`
- `src/dymad/agent/exec/workflow.py`
- `src/dymad/sako/*`
- MCP tool registration and tests

## `T170` Transform/Mode Workflow

## Goal

Extract at least one concrete transform/mode analysis workflow from scripts into a supported library-backed capability.

## Current building blocks to preserve

- `DataInterface` in `src/dymad/io/checkpoint.py`
- script logic in:
  - `scripts/vortex/vor_proc_cli.py`
  - possibly related `scripts/vortex/vor_post.py`

## Scope

- Extract reusable library logic from the script path into a module under `src/dymad/agent/exec/` or another appropriate package.
- Register a transform/mode analysis capability.
- Add compiled request and execution support through the shared analysis foundation.
- Keep the script CLI as a thin wrapper over the extracted library path.

## Deliverables

- reusable analysis library function(s)
- capability registry entry
- MCP discovery and execution tool coverage
- tests that validate the supported analysis path without invoking ad hoc script logic

## Acceptance criteria

- at least one transform/mode workflow no longer requires the agent to inspect `scripts/vortex/vor_proc_cli.py`
- the script remains usable, but the supported path is library-backed and registered
- capability output includes enough metadata to understand artifacts and limitations

## Repo touch points

- extracted module under `src/dymad/...`
- `src/dymad/io/checkpoint.py`
- `scripts/vortex/vor_proc_cli.py`
- registry/compiler/MCP wiring
- tests under `tests/`

## `T180` Developer Mode Split

## Goal

Make the user-mode and developer-mode MCP surfaces explicit in code, docs, and tests.

## Scope

- Add a dedicated developer-tools module, for example:
  - `src/dymad/agent/mcp/developer_tools.py`
- Move or wrap current raw tools under the developer mode surface:
  - `register_dataset_file`
  - `register_checkpoint`
  - `prepare_prediction_request`
  - `plan_checkpoint_prediction`
  - raw `train_model`
  - raw `evaluate_model`
  - `describe_object`
  - `list_objects`
- Update `build_server()` to support:
  - `mode="user"`
  - `mode="developer"`
  - `mode="both"`
- Document which tools are stable user workflows vs low-level/raw tools.

## Deliverables

- explicit user/developer MCP modules
- server-mode registration logic
- tests for mode-specific tool registration

## Acceptance criteria

- tool registration clearly distinguishes user mode from developer mode
- compatibility-only planning tools are not part of the ordinary user surface
- raw tools remain available for debugging and custom workflows

## Repo touch points

- `src/dymad/agent/mcp/server.py`
- `src/dymad/agent/mcp/user_tools.py`
- `src/dymad/agent/mcp/developer_tools.py`
- tests under `tests/test_demo_tools.py` and related MCP test modules

## `T190` Cleanup, Skill Simplification, and Deprecation

## Goal

Finalize the migration by simplifying skill guidance, updating docs/tests, and de-emphasizing the brittle raw paths for user-facing workflows.

## Scope

- Update skill guidance so it relies on registry/compiler-backed tools rather than free-form mapping to raw internal strings.
- Update docs to describe:
  - user mode
  - developer mode
  - supported capability discovery
- Add migration/deprecation notes for:
  - raw user-facing `model_ref`
  - raw user-facing `reference_profile`
  - raw nested user-facing config dicts
- Expand regression tests to lock in:
  - discovery flow
  - compile/train/evaluate flow
  - supported analysis flows
  - developer-mode raw behavior

## Deliverables

- updated skill files and MCP docs
- deprecation notes in docs or release notes
- end-to-end tests covering both modes

## Acceptance criteria

- the primary user-facing guidance no longer tells the agent to infer raw internal strings on its own
- docs reflect the supported high-level workflow surface
- both user-mode and developer-mode flows are covered by tests

## Repo touch points

- `skills/dymad-train-eval-workflow/SKILL.md`
- any agent prompt/guidance files under `skills/`
- docs under `docs/`
- MCP and integration tests under `tests/`

## Suggested Execution Order in Practice

1. Do `T100` and `T110` together if the registry API remains small and read-only.
2. Land `T120` before any user-mode surface changes.
3. Land `T130` before introducing `train_compiled_request`.
4. Land `T140` only after compiler and persistence are stable.
5. Use `T150` as the architecture foundation before implementing concrete analysis flows.
6. Implement `T160` before `T170` because spectral analysis already has the best boundary scaffolding.
7. Land `T180` only after both user-mode and analysis-mode surfaces exist.
8. Use `T190` to simplify prompts/skills and finish deprecations once behavior is stable.

## Suggested Branching / PR Strategy

- PR 1: `T100` + `T110`
- PR 2: `T120`
- PR 3: `T130`
- PR 4: `T140`
- PR 5: `T150`
- PR 6: `T160`
- PR 7: `T170`
- PR 8: `T180`
- PR 9: `T190`

This keeps review scope manageable and allows validation at each architectural seam.

## Done Condition for the Full Migration

The migration is complete when all of the following are true:

- capability discovery is registry-backed rather than repo-inspection-backed
- user-mode training/evaluation no longer requires raw `model_ref` or raw config dicts
- at least one spectral analysis workflow is supported through the new architecture
- at least one transform/mode workflow is supported through the new architecture
- raw low-level tools remain available in explicit developer mode
- skills/docs/tests all point primarily to the new high-level workflow surface
