# Migration Plan: Target Architecture for Core / Data / Training / Facade Split

## Goal
Define the target architecture for the major DyMAD refactor so later implementation tasks can be scoped against a stable design.

This document is the contract for the refactor. It should guide:
- data model redesign
- transform redesign
- model specification redesign
- training orchestration redesign
- public API cleanup
- future facade / object-store / MCP integration

The intent is to preserve current modeling breadth while making the system easier to extend, easier to analyze, and safer to expose through a handle-based external interface.

## Design Principles
- Keep the mathematical and numerical implementation pure. No handles, storage, MCP, agent, or JSON concerns in `core`.
- Separate semantic model types from storage/layout optimizations. We need both expressive data types and efficient fast paths.
- Prefer typed objects and explicit protocols over stringly-typed dispatch in new code.
- Keep concise predefined-model creation as a first-class requirement.
- Make transforms and post-processing usable inside differentiable model pipelines.
- Separate training state, artifact management, and execution control.
- Preserve backward compatibility with the current user-facing API where practical through compatibility shims and adapters.

## Target 4-Layer Stack

### 1) `core`
Pure implementation layer.

Responsibilities:
- data containers and dataset abstractions
- transform implementations
- model components and model composition
- prediction / rollout engines
- analysis adapters
- optimizers, smoothers, denoisers, solvers
- training phase primitives

Must not contain:
- handle registries
- artifact stores
- MCP request/response shaping
- planner / executor orchestration
- agent tracing concerns

### 2) `facade`
Stable typed boundary around `core`.

Responsibilities:
- convert raw values into typed operations over `core`
- expose handle-oriented APIs
- validate inputs and normalize outputs
- call `ObjectStore` for lookup / persistence

Representative concepts:
- `FacadeOperations`
- typed `DatasetHandle`, `PlanHandle`, `CodeHandle`
- conversion between user-level JSON payloads and `core` objects

### 3) `store`
Active object store and persistence layer.

Responsibilities:
- retain in-memory active objects
- persist datasets, plans, generated code, and possibly trained artifacts
- provide deterministic handle naming and lookup

Representative concepts:
- `ObjectStore`
- `FilesystemArtifactStore`
- active-memory objects keyed by `ds_*`, `plan_*`, `code_*`

### 4) `exec`
Developer workflow layer on top of `facade`.

Responsibilities:
- planning and execution workflows
- verification hooks
- generated-code execution through restricted runtime
- orchestration of multi-step solve flows

Representative concepts:
- `DeveloperPlanner`
- `DeveloperExecutor`
- `build_default_context()`

## MCP / Agent Exposure
The MCP-facing layer sits above `exec` and should not bypass the `facade`.

Planned assembly:
- `build_default_context()` wires `ObjectStore -> FacadeOperations -> DeveloperPlanner -> DeveloperExecutor`
- `DemoTools` adapts the context into JSON-shaped `ok/error` tool responses
- `server.py` publishes `DemoTools` through `FastMCP`
- `TraceRecorder` sits above `DemoTools` for replayable tool traces and workflow expansion

Guidance:
- If a workflow is directly supported by tools, stay on the direct tool path.
- If not, planner + executor may synthesize and run generated code through a restricted execution API.
- Keep MCP-specific schemas out of `core`.

## Data Layer Target

## Requirements
The new data layer must support:
- regular series and graph series
- even and uneven trajectory lengths
- uniform and non-uniform step sizes
- fixed-graph and time-varying-edge graph data
- efficient fast paths when a stronger storage/layout contract is known

## Key Design Rule
Do not model every case as one giant catch-all object like current `DynData`.
Also avoid immediately creating a full cross product of classes if the implementation cost is too high.

Use:
- a small number of semantic base types
- explicit layout/storage descriptors
- specialized concrete fast-path types only where they produce meaningful runtime benefit

## Proposed Semantic Types
- `RegularSeries`
- `GraphSeries`
- `LatentSeries`
- `DerivedSeries` for intermediate quantities such as denoised state increments or smoother outputs

Each series should expose:
- `time`
- `state`
- optional `control`
- optional `parameters`
- metadata / annotations
- batching and slicing operations
- conversion to device / dtype

## Proposed Layout / Storage Specializations
Examples of concrete optimized types:
- `UniformStepRegularSeries`
- `VariableStepRegularSeries`
- `UniformStepEvenLengthRegularBatch`
- `RaggedRegularBatch`
- `FixedGraphSeries`
- `VariableEdgeGraphSeries`
- `FixedGraphUniformBatch`

Important:
- start with the minimum set needed for correctness and a few high-value fast paths
- do not implement the full combinatorial family unless benchmarks justify it

## Data Bundle / Artifact Types
Training phases need to pass intermediate objects, not only models.

The state/artifact system should allow typed intermediate payloads such as:
- `SmoothedLatentSeries`
- `DenoisedDeltaSeries`
- `EncodedSeries`
- `LinearizationBundle`
- `ModalAnalysisBundle`

These should be first-class objects in `core`, not ad hoc dicts.

## Trajectory Manager Target
`TrajectoryManager` should become a focused loading and preprocessing component.

Responsibilities:
- load raw arrays / files / graph structures
- infer or build the appropriate series type
- fit preprocessing transforms if requested
- apply transforms
- construct train/valid/test datasets

Must not own:
- model instantiation
- optimizer setup
- checkpoint orchestration
- CV orchestration
- analysis logic

This likely means splitting current responsibilities into:
- data loading
- schema inference / validation
- transform fitting/application
- dataset/window construction

## Transform Layer Target

## Goals
- migrate transforms to PyTorch-first implementations
- allow transforms to participate naturally in models
- allow gradients through transform pipelines
- keep compatibility with external numerical methods when necessary

## Proposed Interface
Transforms should become `torch.nn.Module`-based objects with a fitted state.

Recommended base protocol:
- `fit(series_or_batch) -> self`
- `forward(x)`
- `inverse(x)`
- `jacobian(...)` or mode-related helpers when relevant
- `state_dict()` / `load_state_dict()`

Recommended categories:
- stateless differentiable transforms
- fitted differentiable transforms
- wrapped external transforms using custom autograd

## External Package Wrappers
For transforms relying on SciPy or other CPU-side libraries:
- keep a wrapper implementation for now
- run the underlying external routine on CPU
- wrap it in a customized `torch.autograd.Function`
- provide a clear gradient contract, even if approximate

The transform pipeline should be composable and embeddable into models.
It should replace the current numpy-list-only assumption.

## Model Layer Target

## Requirements
Models need concise predefined construction while supporting:
- encoder
- decoder
- dynamics
- transform pipeline
- Markovian and non-Markovian memory behavior
- prediction policies adapted to the memory type and time domain

## Proposed Specification Approach
Replace the new internal API with typed model specs instead of string maps.
String aliases may remain as a compatibility layer.

Recommended concepts:
- `ModelSpec`
- `EncoderSpec`
- `DecoderSpec`
- `DynamicsSpec`
- `TransformPipelineSpec`
- `MemorySpec`
- `PredictionSpec`

Key distinction:
- model structure describes what the model is
- rollout engine describes how it is simulated

Examples:
- `MarkovMemory(order=1)`
- `HistoryMemory(window=k, update="shift")`
- `ContinuousRollout(solver="dopri5")`
- `DiscreteRollout(mode="step")`

Predefined models should be short factory functions or registry entries building these specs, not large string combinations.

## Compatibility Strategy
Keep the current predefined names such as `LDM`, `KBF`, `DKBF`, etc. as adapters that build the new typed specs.

This allows:
- concise user-facing APIs
- deprecation of string-based internals
- minimal breakage in tests and examples during migration

## Prediction / Rollout Separation
Current prediction code mixes data normalization assumptions, memory assumptions, and integration logic.

Target split:
- model exposes state transition / rate functions and memory contract
- rollout engine handles:
  - continuous vs discrete stepping
  - control interpolation
  - batching
  - graph batching peculiarities
  - projection / no-projection behavior

This will make Markovian and non-Markovian prediction adapt cleanly.

## Post-Processing / Analysis Interface
Models should expose a stable post-processing boundary for downstream tools.

Target capability families:
- linearization
- modal analysis
- FTLE / CLV
- Koopman / operator inspection
- manifold-aware analysis

Recommended approach:
- define analysis-facing protocols or adapters such as `LinearizableModel`, `ModalAnalyzableModel`, etc.
- keep analysis logic outside the model core when possible
- expose standardized outputs such as Jacobians, local linear systems, tangent operators, and encoded trajectories

The model should provide the ingredients; analysis tools should consume a stable analysis view rather than rummaging through internal modules.

## Training Stack Target

## Required Hierarchy
The target training hierarchy is:
- `CVDriver`
- `TrainerRun` for one concrete training run and its artifacts
- `PhasePipeline` as a sequence of phases
- phase primitives, including optimizers and data-manipulation phases

This matches the desired structure:
- CV driver
- a trainer managing artifacts from one training
- phased optimizer as a sequence of optimizers / data manipulations

## Phase Types
At minimum support:
- `OptimizerPhase`
- `DataPhase`
- `AnalysisPhase`
- `ExportPhase`

Examples:
- smoother phase producing `SmoothedLatentSeries`
- denoising phase producing state-delta estimates
- linear solve phase as its own optimizer
- nonlinear parameter update phase

Important:
- linear solver must not remain hidden inside another optimizer
- intermediate products must be typed outputs persisted in run state / artifact store

## Run State vs Execution
Separate:
- immutable or checkpointable training artifacts/state
- execution-time services

For example:
- `TrainerState`
- `PhaseContext`
- `ArtifactRegistry`
- `ExecutionServices`

Avoid letting one object own data loaders, models, optimizer instances, schedulers, metrics, checkpoint payloads, and run control simultaneously.

## Public API Boundaries

## Goals
- reduce eager re-exports
- make internal imports explicit
- define stable user-facing import paths
- prepare for a future MCP boundary without making `core` depend on it

## Guidance
- package `__init__.py` files should be thin
- internal code should import concrete modules, not package re-exports
- re-export only intentional stable surface area
- keep convenience factories if they are lightweight and stable

The future MCP layer should consume the `facade`, not `core` modules directly.

## Suggested Package Direction
Long-term package direction can look like:
- `dymad.core.*`
- `dymad.facade.*`
- `dymad.store.*`
- `dymad.exec.*`
- `dymad.mcp.*`

Migration note:
- do not move everything at once
- first establish the boundaries inside the current package layout
- then relocate modules when the contracts are stable

## Migration Strategy

### Phase 0
- thin `__init__.py`
- stop internal imports through re-export modules
- define the target architecture contracts in docs/tasks

### Phase 1
- introduce new series/data abstractions
- create compatibility adapters from current `DynData`
- narrow `TrajectoryManager`

### Phase 2
- introduce torch-native transform base and pipeline
- provide wrappers for external CPU transforms
- migrate existing transforms incrementally

### Phase 3
- introduce typed model specs and rollout engines
- keep old predefined-model names as adapters
- add analysis-view interfaces

### Phase 4
- refactor training into `CVDriver -> TrainerRun -> PhasePipeline -> Phase`
- promote latent/smoother/denoising outputs to typed artifacts
- split linear solve into its own optimizer phase

### Phase 5
- introduce `facade`, `ObjectStore`, `FilesystemArtifactStore`, and `exec`
- add MCP-facing `DemoTools` adapter and server assembly

### Phase 6
- deprecate obsolete string-based internals and catch-all runtime objects
- shrink remaining compatibility shims

## Acceptance Criteria

### A) Data
- new data layer supports uneven lengths, non-uniform time grids, and both fixed and varying graph structure
- at least one optimized fast-path batch type exists for common uniform regular data

### B) Transforms
- transform pipeline can be used both as preprocessing and inside models
- gradients can flow through supported transforms
- wrapped external transforms have explicit gradient behavior

### C) Models
- predefined models remain concise to instantiate
- Markovian and non-Markovian prediction use explicit memory/rollout contracts
- analysis tools can consume a stable post-processing interface

### D) Training
- linear solve is a first-class phase
- intermediate artifacts are typed and checkpointable
- phased pipelines can pass intermediate outputs to later phases

### E) API / MCP
- `core` contains no MCP- or handle-related code
- `facade` is the only boundary exposed upward
- MCP tools operate through `DemoTools` and not through direct imports of `core`

## Notes
- The existing TODOs in `README.md` already align with this direction, especially splitting `DynData`, grouping transforms, and making transforms torch-native.
- The near-term implementation should favor compatibility adapters over flag-day rewrites.
- Performance-sensitive data specializations should be added selectively, driven by actual hot paths.
