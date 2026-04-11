# MCP Migration Plan

## Principles

- Do not start with a large rewrite.
- Preserve useful existing work.
- Keep current atomic functions as internal building blocks.
- Move capability resolution and validation into code before expanding the MCP surface.
- Add user mode without breaking developer mode.

## Planned Sequence

```text
Stage 0: audit and design docs
Stage 1: registry consolidation and discoverability
Stage 2: typed training compiler
Stage 3: user-mode MCP façade
Stage 4: analysis/postprocess workflow adapters
Stage 5: explicit developer mode and deprecations
```

## Stage 0: Audit and Design

Status:

- completed by the docs in this directory

Deliverables:

- `docs/mcp_current_state_audit.md`
- `docs/mcp_gap_analysis.md`
- `docs/mcp_target_architecture.md`
- `docs/mcp_migration_plan.md`

Acceptance criteria:

- repo-specific architecture description exists
- preserve/wrap/replace/deprecate guidance is explicit

## Stage 1: Smallest Practical First Slice

## Goal

Add a unified registry layer and MCP discovery tools without changing execution behavior.

This is the smallest high-value slice because it reduces repo inspection immediately and de-risks the later compiler work.

## Implementation

Add:

```text
src/dymad/agent/registry/
  __init__.py
  models.py
  workflows.py
```

Seed data from existing code:

- `src/dymad/models/collections.py`
- `src/dymad/models/model_spec.py`
- `src/dymad/agent/exec/training_profiles.py`

Add MCP developer-facing discovery tools first:

- `list_training_capabilities(dataset_handle=None)`
- `list_model_capabilities()`
- `resolve_model_capability(key_or_alias)`

Keep current execution path unchanged:

- `train_model` still takes current arguments
- `evaluate_model` still takes current arguments

## Why first

- no large refactor
- low-risk read-only addition
- immediately exposes supported model families/profiles without repo inspection

## Acceptance criteria

- one registry module exists and is covered by tests
- agent can list supported models and default profiles through MCP
- no regression in existing MCP training/evaluation tests

## Keep / Wrap / Replace / Deprecate

Keep:

- all existing tools

Wrap:

- current model/profile metadata into registry entries

Replace:

- nothing yet

Deprecate:

- nothing yet

## Stage 2: Typed Training Compiler

## Goal

Introduce a typed compile step for user-mode training requests, while still routing execution through the existing training implementation.

## Implementation

Add:

```text
src/dymad/agent/compiler/
  schemas.py
  training.py
```

New objects:

- `TrainingRequest`
- `CompiledTrainingRequest`

Compiler responsibilities:

- resolve model family key -> registry entry
- infer dataset compatibility from registered dataset
- select default profile from registry
- validate safe user overrides
- build effective config
- validate phases via `normalize_phase_specs`
- determine trainer kind via current trainer-selection logic

Important implementation rule:

- reuse current logic from `src/dymad/agent/exec/workflow.py`
- do not duplicate training semantics

## MCP additions

Add user-mode planning tool:

- `compile_training_request(...)`

Do not add training execution changes yet unless necessary.

## Acceptance criteria

- compiler can produce a validated effective config for at least current KBF/LDM/LTI families
- invalid user overrides fail with structured, capability-aware errors
- compiler output includes:
  - resolved model family
  - resolved profile
  - trainer kind
  - effective config preview

## Keep / Wrap / Replace / Deprecate

Keep:

- current training executor

Wrap:

- `_effective_config`
- `resolve_profile_name`
- `_select_trainer`

Replace:

- nothing on the developer path

Deprecate:

- direct user dependence on raw `model_ref`

## Stage 3: User-Mode MCP Façade

## Goal

Add a true user-mode tool surface that uses registry keys and compiled requests instead of raw import strings and raw nested config dicts.

## Implementation

Add:

```text
src/dymad/agent/mcp/user_tools.py
```

Recommended first user-mode tools:

- `inspect_dataset`
- `list_training_capabilities`
- `compile_training_request`
- `train_compiled_request`
- `evaluate_checkpoint`

Recommended server change:

- `build_server(mode="both")` by default
- clearly separate user and developer registrations internally

Execution behavior:

- `train_compiled_request` should route into the existing training executor after compilation

## Acceptance criteria

- a standard train/evaluate flow no longer requires:
  - `model_ref`
  - `reference_profile`
  - raw nested config
- existing developer tools still work
- current skill guidance can be simplified because capability resolution now lives in code

## Keep / Wrap / Replace / Deprecate

Keep:

- existing `DemoTools` behavior for developer mode

Wrap:

- current train/evaluate path behind user-mode compiler flow

Replace:

- user-mode reliance on `train_model(..., model_ref=..., config=...)`

Deprecate:

- skill-only mapping from natural language to raw internal strings

## Stage 4: Analysis and Post-Process Workflow Adapters

## Goal

Bring at least one real analysis workflow and one script-era postprocess workflow behind the same registry/compiler/execution pattern.

## Recommended order

1. spectral analysis
2. transform/mode analysis
3. any remaining script-backed reporting/postprocess workflows

This order is low-risk because spectral analysis already has partial boundary work.

## Implementation

### 4A. Spectral analysis

Add registry entry and execution node for:

- `spectral_koopman`

Build on:

- `CompatibilityExecutor.plan_spectral_analysis(...)`
- `SpectralAnalysisAdapter`
- `SpectralAnalysis`

Target API:

- compile analysis request
- run analysis request
- persist outputs and summaries as handles

### 4B. Transform/mode analysis

Extract reusable library logic from:

- `scripts/vortex/vor_proc_cli.py`

Build a library-backed analysis capability rather than shelling out to the script.

## Acceptance criteria

- at least one analysis workflow is discoverable via MCP
- at least one analysis workflow runs without requiring the agent to inspect scripts
- script CLIs remain usable as wrappers for reproducibility

## Keep / Wrap / Replace / Deprecate

Keep:

- script CLIs for reproducibility and manual workflows

Wrap:

- script logic into library adapters
- `DataInterface`
- spectral adapter/runtime hooks

Replace:

- script inspection as the primary agent path for supported analysis

Deprecate:

- exposing script behavior only through examples/tests

## Stage 5: Explicit Developer Mode and Cleanup

## Goal

Finish the mode split and clean up the MCP surface.

## Implementation

- keep low-level raw tools in developer mode
- document user-mode stability guarantees
- add registry/debug tools to developer mode
- mark compatibility-only tools as developer mode in docs and server registration

Potential cleanup:

- rename current `DemoTools` to reflect developer/raw role, or
- keep the class name but register its tools only in developer mode

Potential deprecations:

- user-facing access to:
  - `register_checkpoint`
  - `prepare_prediction_request`
  - `plan_checkpoint_prediction`
- direct user-facing raw config dicts

## Acceptance criteria

- tool surface clearly separates stable user workflows from low-level developer tools
- docs and tests reflect the distinction
- user mode covers the common train/evaluate/analyze path

## Milestones

## M1. Registry foundation

Outcome:

- discoverability without execution changes

Done when:

- model/workflow capability registries exist
- discovery tests pass

## M2. User-mode training compiler

Outcome:

- capability-safe training request compilation

Done when:

- compiler handles current main model families
- override validation is capability-aware

## M3. User-mode train/evaluate façade

Outcome:

- stable user workflow tools

Done when:

- common train/evaluate path avoids raw import strings

## M4. Analysis workflow support

Outcome:

- at least one supported analysis path through MCP

Done when:

- spectral or transform-mode analysis runs as a registered workflow

## M5. Mode split complete

Outcome:

- user mode and developer mode are explicit and documented

Done when:

- low-level tools are clearly developer-only
- common user flows no longer depend on developer semantics

## Risk Management

## Low-risk moves

- registry consolidation
- read-only discovery tools
- wrapping existing profile/model metadata
- typed compiler that reuses current executor behavior

## Medium-risk moves

- changing MCP tool contracts
- separating user and developer tool groups
- moving helper functions out of `exec/workflow.py`

## Higher-risk moves

- replacing training execution internals
- rewriting script workflows before extracting their library logic
- trying to solve capability resolution with more prompt parsing instead of registries

## Recommended Deprecation Policy

1. keep current developer/raw tools working while user mode is introduced
2. add registry-backed alternatives first
3. update skills/tests/docs
4. only then de-emphasize raw user-facing usage

## Final Recommendation

The smallest practical first implementation slice is:

1. build unified registries from existing model/profile metadata
2. add discovery MCP tools
3. add a typed training compiler that reuses current execution logic

That sequence gives immediate value, preserves current work, and creates the foundation for both:

- a safe user mode
- a powerful explicit developer mode
