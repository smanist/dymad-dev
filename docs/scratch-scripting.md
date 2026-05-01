# Scratch Scripting Contract

## Purpose

Use this contract when an agent needs to write temporary developer-facing scripts that import
DyMAD functions directly for exploration, benchmarking, or prototyping. Scratch scripting is
different from MCP tools, CLI examples, and committed runnable scripts:

- MCP and future CLI interfaces expose stable user-facing workflows.
- Committed scripts under `scripts/*` or examples under `examples/*` document reusable workflows.
- Scratch scripts are temporary, external work products for deeper investigation and should not
  modify repo-tracked files by default.

Scratch work is appropriate for tasks such as trying several data-processing variants, exploring a
custom training schedule, benchmarking a replacement phase against an existing phase, or inspecting
model rollout behavior with extra diagnostics.

## Default Workflow

1. Create or use a scratch folder outside repo-tracked paths.
2. Write scripts, generated data, checkpoints, plots, logs, and summary tables inside that scratch
   folder unless the user explicitly asks for another output location.
3. Start from the closest committed example or test, but copy/adapt the idea into the scratch
   folder instead of editing the repo.
4. Prefer public package imports first. Use implementation-module imports only when examples, tests,
   or this contract make that import path relevant to the experiment.
5. Read the closest tests before spelunking broadly through implementation code.
6. If no existing hook or user-facing registry supports the experiment, implement the one-off logic
   in the scratch script. Do not add repo support unless the user asks for a repo change.

Scratch scripts may import DyMAD internals for investigation, but those imports are not a stability
promise for user-facing APIs.

## Import Tiers

Preferred imports are public package barrels:

- `dymad.core`
- `dymad.models`
- `dymad.training`
- `dymad.numerics`
- `dymad.io`
- `dymad.sako`

Allowed with care:

- Concrete implementation modules listed by this contract, nearby examples, or nearby tests.
- Test helper patterns that clarify expected behavior, copied into scratch code only when they are
  not repo-specific fixtures.
- `scripts/*` examples as workflow references, with the script-style rules in
  [example-script-pattern.md](example-script-pattern.md) applying only when a script will be
  committed.

Avoid by default:

- `dymad.agent.mcp.*` adapters, unless the experiment is specifically about MCP behavior.
- `dymad.agent.store.*` and `dymad.agent.facade.*` persistence internals, unless testing handle
  boundaries.
- Private or underscored helpers, compatibility shims, and subprocess worker entrypoints unless the
  closest tests establish them as the behavior under investigation.
- Adding new registry keys, compiler fields, MCP tools, or committed examples for one-off scratch
  experiments.

## Registry Usage

Use existing `src/dymad/agent/registry/*` metadata to understand stable user-facing capabilities,
supported model/profile choices, accepted overrides, and analysis or evaluation workflows. Registry
metadata is a reading surface for scratch agents, not a reason to make every experiment
user-facing.

Do not add registry entries for scratch-only work. If an experiment proves generally useful, make a
separate repo change that adds runtime support in the owning implementation package first. Add
registry, compiler, executor, or MCP exposure only when the behavior should become a stable
user-facing capability.

## Curated Scratch Index

| Experiment area | Preferred imports | Closest examples/tests | Scratch artifacts | Fallback path |
| --- | --- | --- | --- | --- |
| Data transform classes and data interface class | `dymad.core`, `dymad.io`, `dymad.numerics` | `tests/test_assert_transform.py`, `tests/test_assert_denoising.py`, `tests/test_contract_transform_builder.py`, `tests/test_contract_public_load_model_boundary.py`, `scripts/denoise/*` | Processed arrays, transform state, metrics tables, diagnostic plots | Build the transform/data-interface experiment in scratch code and import concrete transform modules only after checking the closest tests. |
| Phased training and custom schedules | `dymad.training`, `dymad.models`, `dymad.core` | `tests/test_contract_training_phase_runtime.py`, `tests/test_contract_typed_trainer_batches.py`, `tests/test_workflow_lti.py`, `tests/test_workflow_ker_ctrl.py`, developer-facing scripts under `scripts/*/*train.py` | Run logs, checkpoint files, phase metrics, comparison tables, plots | Compose existing phase specs in scratch code. If a truly new phase kind is needed, prototype it locally first; add repo runtime support later only if requested. |
| Model rollout and checkpoint inspection | `dymad.models`, `dymad.io`, `dymad.core` | `tests/test_contract_public_load_model_boundary.py`, `tests/test_contract_checkpoint_e2e_layering.py`, `tests/test_contract_prediction_continuous_runtime.py` | Rollout arrays, error summaries, checkpoint traces, prediction plots | Use public checkpoint/model loading first. Import concrete model helpers only when the boundary tests show the lower-level behavior needed. |
| Spectral analysis | `dymad.sako`, `dymad.models`, `dymad.io` | `tests/test_contract_spectral_adapter.py`, `tests/test_contract_spectral_snapshot.py`, `tests/test_agent_analysis_workflows.py`, `scripts/vortex/*`, `scripts/sa_*/*` | Spectral snapshots, eigenvalue tables, spectra/pseudospectra plots, summary metrics | Use SAKO public exports first. If agent workflow wrappers are too coarse, inspect analysis tests before importing implementation modules. |
| Numerics primitives | `dymad.numerics` | `tests/test_assert_linalg.py`, `tests/test_assert_grad.py`, `tests/test_assert_dm.py`, `tests/test_assert_manifold.py`, `tests/test_assert_weak.py`, `tests/test_assert_spectrum.py` | Small deterministic arrays, reference residuals/errors, timing summaries, plots when useful | Write standalone numerical comparisons in scratch code. If no primitive exists, implement the experimental primitive locally and propose a repo change only after it is reusable. |

## When To Convert Scratch Work Into Repo Work

Keep scratch work external unless one of these is true:

- The user explicitly asks to add or change repo behavior.
- The experiment identifies a reusable runtime feature that belongs in `src/dymad/*`.
- A workflow should become stable enough for MCP, CLI, docs, or tests.
- A bug is found and the requested task is to fix it.

When converting, use [feature-placement.md](feature-placement.md) and
[algorithm-integration-guide.md](algorithm-integration-guide.md) to decide ownership, tests, and
documentation updates. Do not commit scratch artifacts as examples, fixtures, or baselines unless
they are deliberately promoted into repo-maintained assets.
