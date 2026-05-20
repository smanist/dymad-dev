# Codex Instructions

Read [docs/developer/architecture.md](docs/developer/architecture.md) and
[docs/developer/feature-placement.md](docs/developer/feature-placement.md)
before broad repo spelunking. They are the fastest path to the current boundaries and feature
ownership.

Repo fast map:

- `src/dymad/agent/*`: MCP surfaces, capability registries, typed request compilers, workflow
  orchestration, persisted handles, and boundary-facing stores/facades.
- `src/dymad/models`, `src/dymad/training`, `src/dymad/io`, `src/dymad/core`,
  `src/dymad/numerics`, `src/dymad/sako`: model/runtime/math implementation and legacy execution
  paths.
- `tests/*`: workflow and boundary truth. Prefer targeted tests that match the layer you changed.

Script example rule:

- For temporary developer-facing exploration that should not modify the repo, read
  [docs/developer/scratch-scripting.md](docs/developer/scratch-scripting.md). Scratch scripts
  should live outside repo-tracked paths and write artifacts under their scratch folder unless
  explicitly instructed otherwise.
- Before adding a new runnable example under `scripts/`, read
  [docs/developer/example-script-pattern.md](docs/developer/example-script-pattern.md). Both
  documented patterns are valid: use the old inline if-block style for developer-facing examples
  that intentionally expose the execution details, or the `*_cli.py` scaffold plus
  `scripts/cli_helpers.py` for user-facing examples that should be runnable without script edits.

DyMAD skill maintenance rule:

- When editing the DyMAD train/eval Codex skill, edit the repo source under
  `skills/dymad-train-eval-workflow/` first. Do not directly edit the installed copy under
  `$CODEX_HOME/skills` or `~/.codex/skills`.
- After changing the repo skill source, install it with `make install-dymad-skill`. Use
  `make check-dymad-skill-install` when you need to verify that the installed copy matches the
  repo source.

Test naming rule:

- New pytest files in `tests/` must use one of exactly five prefixes:
  `test_assert_`, `test_workflow_`, `test_slow_`, `test_contract_`, or `test_agent_`.
- Put numerics-heavy deterministic exact checks in `test_assert_*`.
- Put execution-path coverage that mainly checks a workflow runs without error in
  `test_workflow_*`.
- Put slow deterministic baseline/regression coverage, usually CLI or end-to-end, in
  `test_slow_*`.
- Put deterministic contract/interface coverage for runtimes, adapters, boundaries, persistence,
  typed batches, and public surfaces in `test_contract_*`.
- Put deterministic integration coverage for agent-facing surfaces such as registries, compilers,
  executors, MCP tools, demo tools, and skill staging in `test_agent_*`.
- Pytest collection enforces these prefixes in `tests/conftest.py`; do not introduce new
  `test_*.py` files outside these categories.

Targeted regression test rule:

- After any code edit, run targeted pytest tests that match the touched behavior before finishing.
  This is required in addition to static checks.
- Select tests by behavior and layer, not only by filename. Examples:
  - Agent, MCP, CLI, registry, compiler, executor, store, or facade edits: run the relevant
    `tests/test_agent_*` and `tests/test_contract_*` files, plus any focused cross-transport tests.
  - Models, training runtime, phase pipeline, checkpoint loading, core runtime, adapters, or public
    package boundary edits: run the closest `tests/test_contract_*`, `tests/test_workflow_*`, or
    `tests/test_agent_*` files that exercise that path.
  - Numerics, solvers, transforms, denoising, sampling, graph, spectrum, or low-level utilities:
    run the nearest deterministic `tests/test_assert_*` or `tests/test_contract_*` files.
  - Runnable scripts, CLI examples, or end-to-end workflows: run the matching
    `tests/test_workflow_*` or `tests/test_slow_*` tests when they are the maintained regression
    surface.
- If a relevant targeted pytest would be unusually expensive or requires unavailable external
  resources, run the nearest cheaper deterministic coverage first and clearly report the skipped
  test, why it was skipped, and the exact command the user or CI should run.
- Do not report code edits as complete after only `make check` when meaningful targeted pytest
  coverage exists for the behavior changed.

Placement rules:

- New MCP tool exposure belongs in `src/dymad/agent/mcp/*`; keep `server.py` focused on tool
  registration.
- New user-facing capability keys, metadata, and schemas belong in `src/dymad/agent/registry/*`.
- New user-mode request validation/compilation belongs in `src/dymad/agent/compiler/*`.
- New workflow orchestration belongs in `src/dymad/agent/exec/*`.
- New persisted artifact or handle types belong in `src/dymad/agent/store/*` and
  `src/dymad/agent/facade/*`.
- New numerical/model/runtime behavior belongs in the implementation packages, not in `agent/*`
  unless it changes the boundary contract.

Documentation maintenance:

- If your change affects package ownership, MCP tool surfaces, registry/compiler behavior,
  checkpoint layering, persisted handle types, or the recommended place to add features, review
  and update [docs/developer/architecture.md](docs/developer/architecture.md) and
  [docs/developer/feature-placement.md](docs/developer/feature-placement.md) in the
  same change.
- Do not leave architecture docs describing the pre-change state after moving boundaries or adding
  new workflows.

After any Python code edit, run the relevant static checks before finishing.

For Python changes, this repo requires:

- `make lint`
- `make typecheck`

Do not report the task as complete unless both commands pass. If a check fails, either fix the issue or clearly report the blocking failure.

Prefer `make check` when both checks should run together.
