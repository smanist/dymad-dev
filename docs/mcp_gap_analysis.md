# MCP Gap Analysis

## Framing

This gap analysis compares the current repo state to the target:

- 4 layers
  1. agent-facing MCP façade
  2. registries / capability metadata
  3. typed config compiler + validation
  4. internal execution graph / atomic functions
- 2 modes
  - user mode
  - developer mode

The key point is that the repo already has partial versions of all four layers, but they are unevenly developed and not yet aligned around a stable user-facing contract.

## Summary Table

| Target area | Current status | Main gap |
| --- | --- | --- |
| Layer 1: MCP façade | Partial | still exposes internal strings and raw dicts |
| Layer 2: registries | Fragmented | no unified, queryable capability registry |
| Layer 3: typed compiler | Weak | config is merged dicts, not typed requests |
| Layer 4: execution graph | Partial/strong for training | not unified across training, evaluation, analysis, postprocess |
| User mode | Missing | tools require developer knowledge |
| Developer mode | Implicit | exists as scripts/classes/low-level tools, not explicit as a mode |

## Layer 1: Agent-Facing MCP Façade

## What exists

- `src/dymad/agent/mcp/server.py` publishes a clean, small FastMCP tool set.
- `src/dymad/agent/mcp/demo_tools.py` provides JSON-safe envelopes.
- `train_model` and `evaluate_model` are already higher-level than raw internals.

## Gaps

The façade is still too close to internal implementation details:

- `train_model` requires `model_ref`
- `train_model` optionally requires `reference_profile`
- `train_model` accepts raw nested `config`
- `evaluate_model` requires metric strings and raw `predict_kwargs`
- prediction-planning tools are exposed alongside user workflow tools with no mode separation

Consequences:

- the agent must know import-path strings and profile keys
- the agent must infer which config fragments are legal
- tool selection is not mode-aware

Architectural mismatch:

- the façade is transport-safe, but not capability-safe

## Layer 2: Registries / Capability Metadata

## What exists

Fragmented registry-like structures already exist:

- predefined models and typed specs
  - `src/dymad/models/collections.py`
  - `src/dymad/models/model_spec.py`
- recipe registry
  - `RECIPE_REGISTRY` in `src/dymad/models/recipes.py`
- training profiles and aliases
  - `PROFILE_REGISTRY`
  - `PROFILE_ALIASES`
  - `src/dymad/agent/exec/training_profiles.py`
- typed phase categories
  - `src/dymad/training/phases.py`

## Gaps

These are not organized as one agent-facing capability registry.

Missing pieces:

- canonical user-facing model family keys
- dataset compatibility metadata
- allowed override metadata
- workflow metadata
  - train
  - evaluate
  - spectral analysis
  - transform/mode analysis
  - script-backed postprocess steps
- discoverability tools that query those registries

Consequences:

- the agent reads repo code to learn legal names
- the skill file carries workflow knowledge that should live in code
- script-specific knowledge is not first-class metadata

Architectural mismatch:

- the repo has registry ingredients, but not a registry layer

## Layer 3: Typed Config Compiler + Validation

## What exists

Useful pieces already in place:

- profile selection from model/dataset
- profile materialization
- reserved-path validation
- legacy config normalization
- trainer selection from normalized phases
- typed phase validation
- typed model resolution

## What is missing

There is no typed request object that sits between user intent and executable config.

Current config assembly is:

```text
reference profile dict
  + deep-merged user dict
  + injected runtime fields
  + legacy normalization
```

Missing capabilities:

- typed request schemas for user mode
- structured override validation by workflow/model family
- canonical field names for common user intent
- explanation of why an override is illegal
- registry-backed alias resolution
- compile output with provenance and warnings

Main brittleness points:

- wrong `model_ref`
- wrong `reference_profile`
- wrong nested config paths
- wrong `trainer` strings in phases
- wrong transform/kernel/processor names
- legal individually, illegal in combination

Architectural mismatch:

- there is config templating, but not config compilation

## Layer 4: Internal Execution Graph / Atomic Functions

## What exists

Training already has a strong internal graph:

- typed phase specs
- phase normalization
- phase pipeline
- checkpointable trainer state
- typed artifacts

Prediction and spectral analysis also have partial workflow nodes:

- `plan_checkpoint_prediction`
- `plan_spectral_analysis`
- `materialize_spectral_adapter`

## Gaps

The internal graph is inconsistent across domains.

Training:

- strong

Evaluation:

- fairly direct, embedded in `CompatibilityExecutor.evaluate_model`

Prediction:

- mostly planning only

Spectral analysis:

- partially layered, but not MCP-exposed

Post-process / script analysis:

- mostly outside the graph

Consequences:

- atomic functions exist, but are not consistently composed into reusable workflow nodes
- script logic remains the real execution graph for many analysis tasks

Architectural mismatch:

- layer 4 exists strongly for training, weakly elsewhere

## User Mode vs Developer Mode

## Current state

The repo does not currently define these as explicit modes.

Instead it has:

- a small MCP toolset that mixes user-ish and developer-ish tools
- skill instructions that try to steer the agent
- script and code inspection for everything not covered by the MCP path

## Why current behavior is brittle

User-mode tasks currently require developer-mode knowledge:

- import-path `model_ref`
- exact profile keys
- nested config structure
- hidden compatibility assumptions from scripts/tests

Developer-mode tasks currently lack explicit low-level affordances:

- there is no explicit namespace for raw planning/debug tools
- the distinction between stable surface and internal debugging surface is not encoded

## Recommended split

User mode should hide:

- raw import strings
- raw nested config dicts
- internal plan handles unless useful
- low-level prediction planning tools

Developer mode should keep or expose:

- raw handle registration and inspection
- raw model/profile/config access
- debug planning/materialization helpers
- registry introspection

## Main Brittleness Points

## 1. `model_ref` import strings

Current state:

- the user-facing training path requires exact Python import strings

Problem:

- this is fragile for agents and couples the tool surface to module layout

## 2. `reference_profile` keys

Current state:

- profile inference exists, but explicit profile names are still internal string ids

Problem:

- profile naming is registry-internal, not a user-level concept

## 3. Raw nested config dicts

Current state:

- `train_model.config` is the primary customization channel

Problem:

- powerful but unsafe for user mode
- legality depends on model family, trainer, transforms, and phase schedule

## 4. Analysis/postprocess flows are not discoverable

Current state:

- many capabilities exist only as scripts or library classes

Problem:

- the agent must inspect repo code to know what is supported

## 5. Capability inference happens outside MCP

Current state:

- the skill tells the agent to translate natural language itself

Problem:

- capability resolution is in prompt instructions instead of code
- this does not scale as combinations grow

## 6. Fragmented registries

Current state:

- multiple good registry fragments exist

Problem:

- none of them provide one authoritative capability map for the MCP façade

## Where Repo/Code/Example Inspection Is Compensating for Weak MCP Abstractions

The agent falls back to repo inspection because the MCP layer cannot answer:

- What model families are available for this dataset kind?
- Which schedule templates are supported?
- Which overrides are allowed for a given family?
- Which analysis workflows exist for a checkpoint?
- Which scripts represent supported workflows versus one-off experiments?

Where that knowledge currently lives instead:

- `src/dymad/models/collections.py`
- `src/dymad/agent/exec/training_profiles.py`
- `scripts/**/*`
- `tests/test_workflow_*`
- `tests/test_slow_*_cli.py`
- examples and YAML files

## Gap Conclusions

The biggest mismatch is not between “no architecture” and “desired architecture”.

It is between:

- good internal typed building blocks

and

- a user-facing MCP surface that still expects agents to discover and assemble those blocks manually

The main missing layers are:

- a unified capability registry
- a typed compile step between user intent and executable config
- an explicit mode split

The main migration principle should therefore be:

- preserve the current layered skeleton
- preserve training phase/runtime internals
- preserve typed model-spec work
- move string resolution and config legality into code, not skill prose
