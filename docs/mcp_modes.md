# MCP Modes

DyMAD now exposes two MCP surfaces:

- User mode: stable high-level workflows such as `compile_training_request`, `train_compiled_request`, `evaluate_checkpoint`, `compile_analysis_request`, and `run_analysis_request`.
- Developer mode: raw and compatibility-oriented tools such as `train_model`, `evaluate_model`, `register_checkpoint`, and `plan_checkpoint_prediction`.

Use `build_server(mode="user")`, `build_server(mode="developer")`, or `build_server(mode="both")` to control which surface is registered.

# Deprecation Notes

The user-facing path should no longer require:

- raw `model_ref`
- raw `reference_profile`
- raw nested config dicts passed directly to `train_model`

Those remain available in developer mode for debugging and compatibility workflows.
