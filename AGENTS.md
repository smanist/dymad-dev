# Codex Instructions

Read [docs/architecture.md](docs/architecture.md) and
[docs/feature-placement.md](docs/feature-placement.md)
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

- Before adding a new runnable example under `scripts/`, read
  [docs/example-script-pattern.md](docs/example-script-pattern.md). Both documented patterns are
  valid: use the old inline if-block style for developer-facing examples that intentionally expose
  the execution details, or the `*_cli.py` scaffold plus `scripts/cli_helpers.py` for user-facing
  examples that should be runnable without script edits.

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
  and update [docs/architecture.md](docs/architecture.md) and
  [docs/feature-placement.md](docs/feature-placement.md) in the
  same change.
- Do not leave architecture docs describing the pre-change state after moving boundaries or adding
  new workflows.

After any Python code edit, run the relevant static checks before finishing.

For Python changes, this repo requires:

- `make lint`
- `make typecheck`

Do not report the task as complete unless both commands pass. If a check fails, either fix the issue or clearly report the blocking failure.

Prefer `make check` when both checks should run together.
