# MCP Target Architecture

## Design Goal

Evolve the current implementation toward a 4-layer, 2-mode architecture without discarding the useful work already in the repo.

The target should:

- keep current atomic functions as internal building blocks
- preserve the handle/persistence boundary
- preserve typed model-spec and training-phase work
- reduce user-mode dependence on raw strings and raw nested dicts
- make analysis/postprocess workflows discoverable without repo spelunking

## Proposed 4-Layer Stack

```text
Layer 1: MCP façade
  user mode tools
  developer mode tools

Layer 2: capability registries
  model families
  workflow recipes
  analysis/postprocess capabilities
  allowed overrides

Layer 3: typed compiler + validator
  user request -> compiled request -> effective config/plan

Layer 4: internal execution graph
  training phases
  evaluation nodes
  prediction nodes
  spectral/postprocess nodes
  legacy adapters
```

## Layer 1: Recommended MCP Tool Surface

## Mode split

Implement explicit user mode and developer mode, either by:

- separate tool groups in one server, or
- `build_server(mode="user" | "developer" | "both")`

Repo-specific recommendation:

- keep one server assembly
- register tools from two modules:
  - `src/dymad/agent/mcp/user_tools.py`
  - `src/dymad/agent/mcp/developer_tools.py`

This keeps deployment simple while making the mode split explicit in code.

## User mode tools

User mode should expose stable workflows and validated choices.

Recommended minimal v1 user tools:

1. `inspect_dataset(dataset_handle) -> DatasetInspection`
2. `list_training_capabilities(dataset_handle=None) -> capability summary`
3. `compile_training_request(dataset_handle, model_family, overrides=None, validation_dataset_handle=None, artifact_root=None, run_name=None, seed=None) -> compiled request preview`
4. `train_compiled_request(compiled_request_handle) -> training result`
5. `evaluate_checkpoint(checkpoint_handle, test_dataset_handle, metric="rollout_rmse", plot_selection="median", max_plots=1, artifact_root=...) -> evaluation result`
6. `list_analysis_capabilities(checkpoint_handle, dataset_handle=None) -> capability summary`
7. `run_analysis_request(...) -> analysis result`
8. `describe_object(handle)`

Optional user-mode convenience tool:

9. `train_and_evaluate(...)`

User mode should **not** require:

- raw `model_ref`
- raw `reference_profile`
- raw nested `config`
- direct plan/request registration primitives

## Developer mode tools

Developer mode should preserve low-level control.

Recommended developer tools:

- keep existing tools, possibly namespaced or clearly documented as developer-only:
  - `register_dataset_file`
  - `register_checkpoint`
  - `prepare_prediction_request`
  - `plan_checkpoint_prediction`
  - `train_model`
  - `evaluate_model`
  - `describe_object`
  - `list_objects`
- add explicit introspection helpers:
  - `list_registries`
  - `resolve_model_ref`
  - `resolve_reference_profile`
  - `compile_raw_training_config`
  - `list_script_workflows`

Developer mode should continue to allow:

- raw `model_ref`
- raw nested config
- direct handle work
- debugging of plan/materialization boundaries

## Candidate tool signatures

User mode candidate:

```python
compile_training_request(
    train_dataset_handle: str,
    model_family: str,
    validation_dataset_handle: str | None = None,
    schedule: str | None = None,
    overrides: dict[str, Any] | None = None,
    run_name: str | None = None,
    artifact_root: str | None = None,
    seed: int | None = None,
) -> CompiledTrainingSummary
```

Developer mode candidate:

```python
train_model(
    train_dataset_handle: str,
    artifact_root: str,
    model_ref: str,
    valid_dataset_handle: str | None = None,
    reference_profile: str | None = None,
    config: dict[str, Any] | None = None,
    ...
) -> TrainModelResult
```

This preserves the current developer path while creating a safer user path.

## Layer 2: Recommended Registries and Entry Structure

## Registry package

Add a new package:

```text
src/dymad/agent/registry/
  __init__.py
  models.py
  workflows.py
  analyses.py
  overrides.py
```

## Model capability entries

Seed these from:

- `src/dymad/models/collections.py`
- `src/dymad/models/model_spec.py`
- `src/dymad/agent/exec/training_profiles.py`

Candidate entry shape:

```python
@dataclass(frozen=True)
class ModelCapability:
    key: str                      # "kbf", "ldm", "lti", ...
    model_ref: str               # current developer path
    display_name: str
    typed_spec: ModelSpec
    dataset_kinds: tuple[str, ...]
    default_profiles: dict[str, str]   # {"regular": "...", "graph": "..."}
    default_metric: str
    allowed_schedule_keys: tuple[str, ...]
    supported_analyses: tuple[str, ...]
    allowed_override_paths: tuple[str, ...]
    tags: tuple[str, ...] = ()
```

Important repo-specific choice:

- `model_ref` stays in the registry as an implementation detail
- user mode references `key`, not `model_ref`

## Workflow recipe entries

Candidate training workflow entry:

```python
@dataclass(frozen=True)
class TrainingWorkflowCapability:
    key: str                     # "rollout_train_eval"
    dataset_kinds: tuple[str, ...]
    model_keys: tuple[str, ...]
    compiler: str                # dotted path or function ref
    executor: str                # dotted path or function ref
    default_metric: str
```

Candidate analysis workflow entry:

```python
@dataclass(frozen=True)
class AnalysisCapability:
    key: str                     # "spectral_koopman", "transform_modes"
    checkpoint_model_keys: tuple[str, ...]
    requires_dataset_handle: bool
    backend: str                 # adapter or workflow node
    maturity: str                # "supported", "experimental", "script-backed"
```

## Script-backed capabilities

Some current capabilities live primarily in scripts. Do not expose scripts directly as the stable user API. Instead:

- register them as capability entries
- back them with library-call adapters
- keep script names only as provenance/debug metadata

Examples:

- `scripts/vortex/vor_proc_cli.py`
  - likely capability key: `vortex_mode_analysis`
- `scripts/vortex/vor_post.py`
  - likely capability key: `vortex_spectral_postprocess`
- `scripts/sa_lti/lti_sa.py`
  - likely capability key: `spectral_lti_report`

## Layer 3: Recommended Typed Spec / Config Flow

## Compiler package

Add:

```text
src/dymad/agent/compiler/
  __init__.py
  training.py
  evaluation.py
  analysis.py
  schemas.py
```

## Request lifecycle

Recommended lifecycle:

```text
user request
  -> typed request object
  -> registry resolution
  -> override validation
  -> effective config/plan compilation
  -> persisted compiled-request handle
  -> execution
```

## Training request objects

Candidate shape:

```python
@dataclass(frozen=True)
class TrainingRequest:
    train_dataset_handle: str
    valid_dataset_handle: str | None
    model_family: str
    schedule: str | None
    overrides: dict[str, Any]
    run_name: str | None
    artifact_root: str | None
    seed: int | None
```

Candidate compiled output:

```python
@dataclass(frozen=True)
class CompiledTrainingRequest:
    request: TrainingRequest
    model_ref: str
    reference_profile: str
    effective_config: dict[str, Any]
    trainer_kind: str
    warnings: tuple[str, ...]
```

## Repo-specific compiler strategy

Do **not** replace the existing training implementation immediately.

Instead:

- reuse `resolve_profile_name(...)`
- reuse `profile_config(...)`
- reuse `_effective_config(...)` logic, but move it behind a typed compiler entrypoint
- reuse `normalize_phase_specs(...)` for validation
- reuse `_select_trainer(...)`

Recommended change:

- move current free functions in `src/dymad/agent/exec/workflow.py` into compiler/execution modules with narrower responsibilities

## Override policy

For user mode, validate overrides against an allowlist per capability.

Examples of safe user-mode override fields:

- `model.hidden_dimension`
- `model.encoder_layers`
- `model.decoder_layers`
- `model.koopman_dimension`
- `plotting.max_state_dims`
- selected phase hyperparameters through canonical aliases

Examples that should remain execution-owned:

- `data.path`
- `data_valid.path`
- `path.*`
- arbitrary checkpoint/result paths inside nested config

Examples that should remain developer-mode only:

- raw transform stacks with arbitrary transform `type`
- arbitrary kernel constructor payloads
- arbitrary custom phase graphs

## Layer 4: Recommended Internal Execution Graph Boundary

## Keep existing training graph

Preserve as the base execution graph:

- `PhaseSpec` types
- `normalize_phase_specs`
- `PhasePipeline`
- `TrainerRun`
- `ExecutionServices`
- trainer classes

This is already the strongest internal execution subsystem in the repo.

## Turn current atomic functions into graph nodes

Recommended execution nodes:

- `InspectDatasetNode`
- `ResolveModelCapabilityNode`
- `CompileTrainingConfigNode`
- `TrainModelNode`
- `RegisterCheckpointNode`
- `EvaluateRolloutNode`
- `PlanPredictionNode`
- `PlanSpectralAnalysisNode`
- `RunSpectralAnalysisNode`
- `RunTransformModeAnalysisNode`

These nodes should wrap existing implementation instead of replacing it.

## Repo-specific analysis boundary

### Spectral analysis

Use existing pieces:

- `CompatibilityExecutor.plan_spectral_analysis(...)`
- `SpectralAnalysisAdapter`
- `SAInterface`
- `SpectralAnalysis`

Recommended target:

- separate checkpoint/snapshot planning from analysis execution
- move the stable analysis execution into a workflow node
- leave plotting/report composition as a higher-level analysis recipe

### Transform/mode analysis

Use existing pieces:

- `DataInterface`
- current vortex analysis logic in `scripts/vortex/vor_proc_cli.py`

Recommended target:

- extract script logic into library functions
- register one supported analysis capability
- keep script CLI as a thin wrapper around the library call

## Mapping Existing Code to Target Layers

## Layer 1: façade

Keep / wrap:

- `src/dymad/agent/mcp/server.py`
- `src/dymad/agent/mcp/demo_tools.py`

Add:

- `user_tools.py`
- `developer_tools.py`

## Layer 2: registries

Keep / wrap:

- `src/dymad/models/collections.py`
- `src/dymad/models/model_spec.py`
- `src/dymad/models/recipes.py`
- `src/dymad/agent/exec/training_profiles.py`

Add:

- unified registry package under `src/dymad/agent/registry/`

## Layer 3: typed compiler

Keep / wrap:

- `_effective_config`
- `_validate_user_config`
- `_select_trainer`
- `normalize_phase_specs`

Add:

- typed request/compiled-request objects
- override validators
- capability-aware compile functions

## Layer 4: execution graph

Keep:

- `FacadeOperations`
- `ObjectStore`
- `FilesystemArtifactStore`
- `CompatibilityExecutor`
- training phase pipeline
- `DataInterface`
- spectral adapters

Wrap:

- script-era workflows as execution nodes/adapters

## Keep / Wrap / Replace / Deprecate

## Keep

- `ExecutionContext`
- `FacadeOperations`
- handle classes
- `ObjectStore`
- `FilesystemArtifactStore`
- typed model-spec system
- typed training phase system

## Wrap

- `DemoTools` as developer-mode façade
- current training/evaluation functions behind typed user-mode compilers
- `DataInterface` and `SpectralAnalysis` behind registered workflows

## Replace

- raw user-mode `model_ref` with registry keys
- raw user-mode `reference_profile` with automatic registry resolution
- user-mode raw nested `config` with typed request + validated overrides

## Deprecate

- any prompt-parser strategy based on keyword matching as the primary resolver
- relying on scripts/examples/tests as the primary capability map
- exposing compatibility planning tools as ordinary user-mode tools

## Why This Fits This Repo

This target architecture fits the existing code because:

- the repo already has real typed internals worth preserving
- training already has a graph-oriented execution subsystem
- predefined models already have typed metadata
- persistence and handles already exist

The missing work is not inventing a new core. It is building:

- a proper registry layer
- a proper compiler layer
- an explicit mode split at the MCP boundary
