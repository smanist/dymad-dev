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

Compatibility entrypoint:

- `dymad.io.load_model_compat.load_model_compat(...)`

Execution sequence:

1. `load_model_compat(...)` calls `CompatibilityExecutor.plan_checkpoint_prediction(...)`.
2. `plan_checkpoint_prediction(...)` uses `FacadeOperations` to register a checkpoint handle and prediction-request handle in `ObjectStore`.
3. `load_model_compat(...)` calls `CompatibilityExecutor.materialize_checkpoint_prediction(...)`.
4. `materialize_checkpoint_prediction(...)` resolves handles through `FacadeOperations`.
5. The resolved checkpoint path is materialized through legacy `dymad.io.checkpoint.load_model(...)`.

## Verification

```bash
cd modules/dymad_migrate && PYTHONPATH=src pytest tests/test_checkpoint_e2e_layering.py -q
```

The test validates call order across `exec -> facade -> store` and confirms the compatibility materialization entrypoint remains `dymad.io.checkpoint.load_model`.
