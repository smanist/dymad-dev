# Algorithm Integration Guide

## Purpose

Use this guide when adding a reusable algorithm, model family, solver, analysis workflow, training
phase, or agent-facing tool. It complements:

- [architecture.md](architecture.md), which defines package ownership and boundary layers
- [feature-placement.md](feature-placement.md), which gives the short placement table
- [example-script-pattern.md](example-script-pattern.md), which defines runnable script patterns

The goal is to help AI agents decide where an algorithm belongs, what interface it must expose, and
what supporting schema, tests, examples, and benchmarks are expected.

## Placement Decision Ladder

Start with the narrowest implementation package that can own the behavior. Add `agent/*` exposure
only when the algorithm becomes a stable user-facing workflow or capability.

| If the new functionality... | And it does not... | Put the primary implementation in... | Then expose it through... |
| --- | --- | --- | --- |
| Is a low-level solver, decomposition, derivative, quadrature, denoising primitive, or linear algebra utility | Own datasets, training phases, model construction, handles, or MCP schemas | `src/dymad/numerics/*` | Direct package exports and `test_assert_*` or `test_contract_*` |
| Is a reusable trajectory, graph, transform, runtime, batching, or model-context data structure | Train models, register capabilities, or parse user requests | `src/dymad/core/*` | `src/dymad/core/__init__.py` only when it is a public core surface |
| Is a model family, model variant, component recipe, rollout behavior, or prediction path | Persist handles or validate user-mode requests | `src/dymad/models/*` | `src/dymad/agent/registry/models.py` only if user-mode training should select it |
| Is an optimizer, training phase, training artifact, trainer state change, or phase pipeline change | Add a stable user-facing capability by itself | `src/dymad/training/*` | `src/dymad/agent/registry/training_schema.py` and `compiler/*` only for user-mode overrides |
| Is checkpoint loading, dataset registration, trajectory I/O, or legacy public runtime behavior | Merely orchestrate an existing workflow | `src/dymad/io/*` | `agent/exec/*`, `agent/store/*`, and `agent/facade/*` when handles must reflect it |
| Is Koopman/spectral analysis library behavior | Just call a one-off script | `src/dymad/sako/*` | `agent/registry/analyses.py`, `compiler/analysis.py`, and `exec/workflow.py` for user-mode analysis |
| Coordinates existing package behavior into a workflow | Implement new math or model internals | `src/dymad/agent/exec/*` | MCP tools after registry/compiler/store/facade contracts exist |
| Introduces a stable capability key, model key, profile key, workflow key, schema, examples, or natural-language translation guidance | Execute runtime logic | `src/dymad/agent/registry/*` | `compiler/*` and MCP user tools |
| Validates or normalizes user-mode request fields | Execute workflow logic or persist artifacts | `src/dymad/agent/compiler/*` | Store compiled requests through facade/store if calls must be durable |
| Persists a new request, artifact, run, snapshot, or handle | Implement the algorithm itself | `src/dymad/agent/store/*` and `src/dymad/agent/facade/*` | `agent/exec/*` and MCP tools |
| Adds an MCP tool for an existing capability or workflow | Define business logic in the server | `src/dymad/agent/mcp/user_tools.py` or `demo_tools.py` | `src/dymad/agent/mcp/server.py` registration only |
| Is a runnable demo, benchmark, or experiment | Need to be imported as library behavior | `scripts/*` or `examples/*` | Follow [example-script-pattern.md](example-script-pattern.md) |

If two rows seem to apply, split the change: implementation first, agent-facing exposure second.

## Interface Contracts

### Numerical Algorithms

Numerical code should be small, deterministic, and independent of agent/MCP concerns.

Expected interface:

- Pure functions or focused classes in `src/dymad/numerics/*`
- Tensor/array inputs with explicit shape expectations in the docstring or type name
- No object-store handles, MCP schemas, training-run records, or user-mode keys
- Stable return values that can be checked exactly or within a documented tolerance

Return structures:

- Prefer tensors, arrays, scalars, or small dataclasses when multiple named outputs are required.
- Do not return loosely shaped dicts unless an existing numerics module already uses that pattern
  for the same family.

### Core Data Structures

Core structures represent runtime data rather than algorithms tied to a single workflow.

Expected interface:

- Use typed dataclasses that follow `RegularSeries`, `GraphSeries`, `RegularTrainerBatch`, and
  `GraphTrainerBatch` conventions.
- Preserve device/dtype movement with `.to(...)` when the object owns tensors.
- Keep graph and regular semantics explicit instead of encoding them in ad hoc dicts.
- Add public exports in `src/dymad/core/__init__.py` only when the type is part of the package
  surface.

Primary consumed/returned data:

- Regular trajectories: `RegularSeries`, `RegularSeriesBatch`, `RegularTrainerBatch`
- Graph trajectories: `GraphSeries`, `GraphSeriesBatch`, `GraphTrainerBatch`
- Runtime views: `UniformRegularRuntime`, `RaggedRegularRuntime`, `UniformGraphRuntime`

### Models and Prediction

New model behavior belongs in `src/dymad/models/*` before any agent-facing registry work.

Expected interface:

- Add or reuse a `ModelSpec` from `src/dymad/models/model_spec.py`.
- Expose predefined variants through `PredefinedModel` in `src/dymad/models/collections.py` when
  the model should be selectable by configuration or registry keys.
- Make construction work through `build_model(model_spec, model_config, data_meta, dtype, device)`.
- Keep rollout behavior in `src/dymad/models/rollout_engine.py` or `prediction.py`, not in MCP
  adapters.

Configuration schema:

- Runtime construction still consumes the existing model config dict shape.
- User-mode selection is through `model_key`, `reference_profile`, and compiler-validated
  `overrides`.
- If users should select the model family, add a stable key in `agent/registry/models.py`.
- If users need different defaults, add or update a profile in `agent/registry/workflows.py` or the
  profile registry that owns the relevant config.

### Training Algorithms and Phases

Training behavior belongs in `src/dymad/training/*`.

Expected interface:

- Phase specs are dataclasses such as `OptimizerPhaseSpec`, `LinearSolvePhaseSpec`,
  `DataPhaseSpec`, `AnalysisPhaseSpec`, and `ExportPhaseSpec`.
- Executable phases subclass `BasePhase` and implement `execute(...) -> PhaseResult`.
- Shared state flows through `TrainerState`, `PhaseContext`, and `ArtifactRegistry`.
- New phase outputs should be explicit artifacts, preferably dataclasses in
  `training/phase_runtime.py`, rather than unstructured side effects.

Configuration schema:

- Add runtime phase semantics in `training/phases.py`.
- Add user-mode phase-entry schema in `agent/registry/training_schema.py` only when users can
  request it through `overrides.phases`.
- Add compiler validation/normalization in `agent/compiler/training.py` when new user request
  fields or override paths are accepted.
- Keep runtime-owned fields such as dataset paths and model implementation names protected from
  user overrides.

### Analysis Workflows

Reusable analysis code belongs in the implementation package first, commonly `src/dymad/sako/*`
for spectral analysis or a focused module near the runtime it analyzes.

Expected interface:

- Library analysis should expose functions/classes that consume package data structures, checkpoint
  paths, or model classes directly.
- Agent-facing analysis workflows compile through `AnalysisRequest` and return `AnalysisRunResult`.
- Persisted analysis artifacts should be written under the provided artifact root and returned as
  artifact paths plus a compact numeric/string summary.

Configuration schema:

- Add a stable workflow key and parameter schema in `agent/registry/analyses.py`.
- Add request requirements in `agent/compiler/analysis.py`.
- Add execution in `agent/exec/workflow.py` or a focused helper under `agent/exec/*`.
- Add store/facade records only if the analysis has durable state beyond the compiled request and
  returned artifacts.

### MCP Tools

MCP tools are boundary adapters. They should not own algorithm behavior.

Expected interface:

- User-mode tools call registry/compiler/executor/facade surfaces.
- Developer/demo tools may expose raw handles or compatibility behavior, but should still avoid
  embedding new algorithms.
- `server.py` registers tools and mode splits only.

Configuration schema:

- User-facing capability keys and schemas live in `agent/registry/*`.
- Request validation lives in `agent/compiler/*`.
- Persisted request and artifact handles live in `agent/store/*` and `agent/facade/*`.

## Agent-Facing Schema Checklist

If the algorithm is exposed to user-mode agents, document or implement all of these:

- Stable key: lower-case snake case, no raw import paths as user-facing identifiers.
- Capability summary: concise description of what the algorithm does and the dataset kind it
  supports.
- Dataset compatibility: `regular`, `graph`, or both.
- Parameter schema: explicit types, defaults, allowed enum values, and required fields.
- Override schema: allowed top-level keys and protected runtime-owned paths.
- Natural-language translation guidance: only in `agent/registry/*`, not in MCP adapters.
- Constraint notes: shape, dimension, time-domain, graph-mode, and runtime limitations that a
  client must know before compiling a request.
- Examples: at least one minimal request/override example for each new user-facing capability.
- Result contract: handles, artifact paths, metrics, summaries, and failure modes.

## Data Structures To Consume And Return

Use the typed structures that match the layer:

| Layer | Consume | Return |
| --- | --- | --- |
| Numerics | `torch.Tensor`, `numpy.ndarray`, scalars, typed dataclasses | Tensors, arrays, scalars, typed dataclasses |
| Core transforms/runtime | `RegularSeries*`, `GraphSeries*`, trainer batches, runtime views | Updated typed series/batches/runtime views |
| Models | `ModelSpec`, model config dict, data metadata, runtime payloads | `torch.nn.Module`, rollout tensors, prediction outputs |
| Training phases | `TrainerState`, `PhaseContext`, `ArtifactRegistry`, dataloaders | `PhaseResult` with explicit artifacts and metrics |
| Agent compiler | request dataclasses and registry capabilities | compiled request dataclasses plus diagnostics |
| Agent executor | compiled request records and facade handles | result dataclasses such as `StartModelTrainingResult` or `AnalysisRunResult` |
| MCP tools | JSON-compatible request fields and handles | JSON-compatible dicts built from typed records |

Avoid converting typed runtime objects into generic dicts at lower layers just because MCP eventually
returns JSON. The JSON conversion belongs at the boundary.

## Mandatory Tests

Every algorithm change needs targeted tests in the category that matches the risk:

| Change type | Required test category |
| --- | --- |
| Deterministic numerical primitive, solver, transform math, or exact reference value | `tests/test_assert_*` |
| Runtime interface, typed batch, model spec, transform builder, checkpoint boundary, or public API contract | `tests/test_contract_*` |
| Workflow path that should run without checking detailed numerical accuracy | `tests/test_workflow_*` |
| Agent-facing registry, compiler, executor, MCP tool, handle, or skill-staging behavior | `tests/test_agent_*` |
| Slow deterministic CLI, end-to-end regression, or baseline metric coverage | `tests/test_slow_*` plus `*_baselines.json` when needed |

Minimum expectations:

- New public model families need model-spec or runtime contract coverage.
- New user-mode capability keys need registry tests and compiler tests.
- New analysis workflows need compiler/executor tests and at least one artifact/summary assertion.
- New persisted handle types need store/facade persistence tests.
- New CLI examples that become regression surfaces need slow tests with deterministic seeds and
  bounded runtime.

New pytest files must use exactly one of the allowed prefixes enforced by `tests/conftest.py`:
`test_assert_`, `test_workflow_`, `test_slow_`, `test_contract_`, or `test_agent_`.

## Examples

Add examples only when they teach a workflow that tests and docs do not already make clear.

Expected placement:

- Reusable package API examples: notebooks or docs under `examples/*` when they are part of the
  public documentation.
- Runnable training, analysis, or benchmark demos: `scripts/*`.
- User-facing runnable scripts: CLI style with `parse_args()`, `main()`, and
  `if __name__ == "__main__": raise SystemExit(main())`.
- Developer-facing scripts that intentionally expose execution details may keep the existing
  if-block style.

Before adding or reshaping a script, read [example-script-pattern.md](example-script-pattern.md).

## Benchmarks

Benchmarks are expected when the algorithm changes hot-path runtime, memory behavior, solver
complexity, batch conversion, prediction rollout, or graph/regular data movement.

Expected placement and shape:

- Put focused benchmark scripts under `scripts/benchmarks/*`.
- Use CLI arguments for case, device, iteration count, and warmup count.
- Make CPU execution the default unless the benchmark is explicitly GPU-only.
- Report stable summary statistics such as mean, median, min, and max.
- Keep benchmark fixtures small enough for local iteration; use slow CI only for deterministic
  regression checks, not exploratory performance measurement.

Do not add benchmark results as correctness tests unless the threshold is deliberately stable across
CI machines.

## Naming Conventions

- User-facing capability keys: lower-case snake case, stable, and independent of implementation
  import paths.
- Model family keys: short lower-case names that match existing families such as `kbf`, `km`,
  `ldm`, `lti`, and `sdm`.
- Model variant names: established uppercase variant style in `models/collections.py`, with
  lower-case variant keys derived by the registry.
- Analysis workflow keys: lower-case snake case, such as `spectral_koopman`.
- Handles: keep the existing prefix style (`ds_*`, `chk_*`, `run_*`, `trainreq_*`,
  `analysisreq_*`, `eval_*`, `pred_*`, `specsnap_*`).
- Phase names: descriptive snake case for generated/default phases; preserve user labels when they
  improve readability.
- Test files: one of the five required prefixes.
- Scripts: use descriptive directory names and `*_cli.py` for user-facing CLI entrypoints.

## What Not To Touch

Avoid these changes unless the task explicitly requires them:

- Do not put algorithm, validation, or workflow logic in `src/dymad/agent/mcp/server.py`.
- Do not add user-facing string parsing to MCP adapters; use compiler modules.
- Do not add registry metadata inside executor methods.
- Do not persist new handle state directly in executor methods; use store and facade modules.
- Do not route numerical/model/runtime behavior through `agent/*` just because an agent will call
  it eventually.
- Do not expose raw import strings as the primary user-mode interface.
- Do not weaken protected runtime-owned config paths to make an override pass.
- Do not change checkpoint materialization through `dymad.io.checkpoint` without updating
  architecture docs and boundary tests.
- Do not rename existing capability keys, handle prefixes, or public model variants without a
  compatibility plan.
- Do not edit generated docs under `docs/_build`.

## Integration Checklist

Before opening a change:

1. Classify the algorithm with the placement decision ladder.
2. Implement reusable behavior in the implementation package first.
3. Add typed interfaces and return structures at the lowest layer that owns the behavior.
4. Add registry keys and schemas only if agents/users need a stable selection surface.
5. Add compiler validation only for new user-mode request fields or overrides.
6. Add executor orchestration only for multi-step workflows or persisted runs.
7. Add store/facade records only for durable objects that must be referenced across calls.
8. Add MCP tool exposure last.
9. Add mandatory targeted tests using the required test-file prefix.
10. Add examples and benchmarks when the algorithm introduces a new user workflow or hot path.
11. Update [architecture.md](architecture.md) and [feature-placement.md](feature-placement.md) if
    ownership, boundaries, handles, workflows, or placement guidance changed.
12. For Python edits, run `make check` before reporting completion.
