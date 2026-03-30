# Checkpoint E2E Layering Path

## Purpose

Record one verified end-to-end checkpoint path in `dymad_migrate` that follows the same layer contract documented in `modules/mcp_test/ARCHITECTURE_SUMMARY.md`.

## Layer Mapping

Reference architecture (`mcp_test`) path:

`core -> facade -> exec -> mcp_server`

Current verified DyMAD migration path:

1. `core` (legacy compatibility target): `dymad.io.checkpoint.load_model(...)`
2. `facade`: `dymad.facade.operations.FacadeOperations`
   - `register_checkpoint(...)`
   - `prepare_prediction_request(...)`
   - `get_prediction_request(...)`
   - `get_checkpoint(...)`
3. `store`: `dymad.store.object_store.ObjectStore`
   - persists `chk_*` and `pred_*` records
4. `exec`: `dymad.exec.workflow.CompatibilityExecutor`
   - `plan_checkpoint_prediction(...)`
   - `materialize_checkpoint_prediction(...)`
5. `mcp_server`: not implemented yet in `dymad_migrate`; remains an upper layer target exactly as in `mcp_test`

## Verified Sequence

Public compatibility entrypoint:

- `dymad.io.checkpoint.load_model(...)`

Explicit test helper entrypoint:

- `dymad.io.load_model_compat.load_model_compat(...)`

Execution sequence:

1. `load_model(...)` calls the compatibility adapter path in `dymad.io.load_model_compat`.
2. the compatibility adapter calls `CompatibilityExecutor.plan_checkpoint_prediction(...)`.
3. `plan_checkpoint_prediction(...)` uses `FacadeOperations` to register a checkpoint handle and prediction-request handle in `ObjectStore`.
4. the compatibility adapter calls `CompatibilityExecutor.materialize_checkpoint_prediction(...)`.
5. `materialize_checkpoint_prediction(...)` resolves handles through `FacadeOperations`.
6. The resolved checkpoint path is materialized through legacy checkpoint internals.

## Verification

```bash
cd modules/dymad_migrate && PYTHONPATH=src pytest tests/test_checkpoint_e2e_layering.py -q
cd modules/dymad_migrate && PYTHONPATH=src pytest tests/test_public_load_model_boundary.py -q
```

The tests validate call order across `exec -> facade -> store` and confirm the default public entrypoint now routes through the compatibility boundary.
