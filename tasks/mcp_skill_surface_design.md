# Migration Plan: MCP Surface and Skill Workflow Design

## Goal
Define the recommended DyMAD MCP surface and skill workflow layer for training, tuning, evaluation, spectral analysis, and data-interface workflows.

This document is the contract for external exposure planning. It should guide:
- MCP tool design
- skill design
- boundary rules between natural-language interpretation and deterministic execution
- handle and artifact strategy
- phased rollout of new external workflows

The intent is to expose stable, atomic capabilities through MCP while keeping the language-driven workflow logic in skills.

## Scope
This document covers:
- model training
- hyperparameter tuning
- model evaluation
- spectral analysis
- data-interface and transform workflows

This document does not redefine the internal `core` architecture. It assumes the layer split in [tasks/refactor_target_architecture.md](/Users/daninghuang/Repos/dymad-dev/tasks/refactor_target_architecture.md).

## Design Principles
- Keep MCP tools atomic, typed, and deterministic.
- Keep natural-language interpretation in skills, not MCP tools.
- Prefer handles over raw file paths once data or model state has been registered.
- Reuse existing typed boundaries in `facade`, `store`, and `exec` before adding new script-driven entrypoints.
- Expose catalogs and validators separately from execution tools.
- Split heavyweight workflows into composable steps even if a convenience wrapper also exists.
- Do not expose features as stable MCP tools until their underlying library seam is stable enough to support deterministic inputs and outputs.

## Boundary Rules

## Rule 1: Natural Language Stops at the Skill Layer
Skills may:
- read user free text
- infer `model_ref`
- infer recipe/profile choices
- translate user intent into structured JSON or YAML-like dicts
- select which MCP tools to call and in what order

MCP tools must not:
- accept free text prompts as their main payload
- infer hidden defaults from prose
- mix planning with execution
- silently reinterpret malformed or ambiguous user requests

## Rule 2: MCP Tools Accept Structured Inputs Only
Each MCP tool should accept one of:
- handles such as dataset, checkpoint, run, evaluation, spectral snapshot, or transform-interface handles
- explicit structured config dicts
- primitive scalars and enums

If a tool needs complex configuration, the skill should translate the user request into a deterministic structured config before calling the tool.

## Rule 3: Runtime-Owned Fields Stay Out of User Config
The skill layer may construct structured config, but runtime-owned fields must not be user-overridable through free-form config:
- `data.path`
- `data_valid.path`
- `path.*`
- persisted handle fields
- artifact output paths chosen by the MCP boundary

These fields must be derived from registered handles and tool arguments.

## Rule 4: Prefer Registered Objects Over Raw Paths
Allowed raw paths:
- initial dataset registration
- initial checkpoint registration
- optional export destinations when the tool is explicitly about writing artifacts

After registration, downstream tools should consume handles rather than paths.

## Rule 5: Separate Catalog, Validation, and Execution
For each major workflow family, expose distinct tools for:
- discovery
- validation
- execution
- inspection of results

This avoids one oversized MCP tool that both explains the API and executes it.

## Rule 6: Skills Compose Tools Into Human Workflows
Skills should combine atomic tools into typical user-facing flows such as:
- train then evaluate
- tune then retrain best model
- build spectral snapshot then compute pseudospectrum and plots
- encode data then inspect reconstruction and modes

Skills own the workflow narrative. Tools own typed execution.

## Rule 7: Unsupported Features Must Be Explicit
If an underlying library seam is incomplete, expose that limitation directly rather than pretending the capability is general.

Examples for current state:
- tuning is ready for single-split param-grid search, not full k-fold cross-validation
- spectral analysis is ready first for checkpoint-backed KBF and DKBF workflows
- graph evaluation plots are not yet ready as a first-class v1 output

## Rule 8: Atomic First, Convenience Second
When a workflow can be decomposed into stable atomic operations, expose the atomic operations first.

Convenience wrappers are allowed when they:
- reuse the atomic operations internally
- do not hide important intermediate artifacts
- return handles and artifact references for downstream reuse

## Recommended MCP Surface

## A. Dataset and Object Management
Purpose:
- register external assets
- inspect schemas
- provide stable handles for downstream workflows

Recommended tools:
- `register_dataset_file(path, format="npz", kind="regular"|"graph")`
- `inspect_dataset(dataset_handle)`
- `describe_object(handle)`
- `list_objects(kind=None)`
- `register_checkpoint(model_ref, checkpoint_path, device="cpu")`

Recommended additions:
- `summarize_dataset_samples(dataset_handle, max_samples=...)`
- `inspect_dataset_statistics(dataset_handle, fields=...)`
- `validate_dataset_compatibility(dataset_handle, model_ref)`

Notes:
- These are foundational tools and should remain simple and handle-oriented.
- Dataset compatibility checks should move out of train/evaluate-only flows and become reusable.

## B. Model Catalog and Recipe Discovery
Purpose:
- let skills discover valid model families and training profiles without hard-coding them
- expose the typed model catalog as data

Recommended tools:
- `list_model_families()`
- `describe_model_family(model_ref)`
- `list_reference_profiles(model_ref=None, dataset_kind=None)`
- `describe_reference_profile(profile_name)`
- `validate_training_config(model_ref, dataset_handle, config, reference_profile=None)`

Structured outputs should include:
- model name
- time domain
- graph mode
- allowed rollout modes
- expected dataset kind
- major model dimensions and tunable fields
- compatible default profiles

Notes:
- This is the main answer to the current gap between the small MCP profile set and the much larger internal predefined model catalog.
- Skills should use these tools before constructing train/tune payloads.

## C. Training
Purpose:
- train one model deterministically from registered datasets and structured config

Recommended tools:
- `train_model(train_dataset_handle, artifact_root, model_ref, valid_dataset_handle=None, reference_profile=None, config=None, run_name=None, seed=None, device="auto", max_workers=1)`

Recommended additions:
- `materialize_training_config(train_dataset_handle, model_ref, valid_dataset_handle=None, reference_profile=None, config=None, run_name=None)`
- `inspect_training_run(run_handle)`
- `list_training_artifacts(run_handle)`

Structured outputs should include:
- training run handle
- checkpoint handle
- resolved reference profile
- resolved trainer kind
- materialized training config path
- summary artifact paths
- aggregate training metrics

Notes:
- `materialize_training_config` gives the skill a deterministic checkpoint before execution and makes debugging easier.
- Training should continue to allow staged `phases`, because the phase system is already the right typed abstraction.

## D. Hyperparameter Tuning
Purpose:
- search structured model/training parameters on top of the training stack

Recommended v1 tools:
- `tune_model_single_split(train_dataset_handle, artifact_root, model_ref, valid_dataset_handle=None, reference_profile=None, base_config=None, param_grid=None, metric="total", run_name=None, seed=None, device="auto", max_workers=1)`
- `inspect_tuning_results(tuning_handle)`
- `select_best_tuning_result(tuning_handle)`

Recommended v2 tools after implementation maturity:
- `tune_model_kfold(...)`
- `resume_tuning_run(tuning_handle)`

Structured outputs should include:
- tuning handle
- best parameter combo
- best checkpoint handle
- best run handle
- aggregate result table
- persisted results path
- plot path

Current boundary:
- v1 should be described as single-split param-grid tuning
- do not expose general k-fold tuning until a real k-fold driver exists

Notes:
- This should wrap the existing driver machinery around `cv.param_grid`.
- Skills should translate natural-language search requests into `param_grid` over dotted keys.

## E. Prediction and Evaluation
Purpose:
- separate rollout generation from metric computation and visualization

Recommended atomic tools:
- `prepare_prediction_request(checkpoint_handle, horizon, has_control=False, has_graph=False)`
- `plan_checkpoint_prediction(model_ref, checkpoint_path, horizon, has_control=False, has_graph=False)`
- `predict_checkpoint(checkpoint_handle, dataset_handle=None, prediction_request_handle=None, predict_kwargs=None, selection=None)`
- `compute_rollout_metrics(prediction_handle, metric_specs)`
- `plot_rollouts(prediction_handle, selection="median", max_plots=1)`

Recommended convenience wrapper:
- `evaluate_model(checkpoint_handle, test_dataset_handle, metric, artifact_root, plot_selection="median", max_plots=1, predict_kwargs=None)`

Recommended metric support:
- `rollout_rmse`
- rollout MAE
- horizon-wise error summaries
- VPT
- task-specific custom metrics where the definition is fully structured

Structured outputs should include:
- prediction handle
- evaluation handle
- aggregate metrics
- per-trajectory metrics
- plot paths
- skip reasons when a visualization is unsupported

Notes:
- Prediction and evaluation should become independent tools so skills can mix and match metrics and plots.
- `evaluate_model` can remain as a convenience wrapper but should not be the only evaluation entrypoint.

## F. Spectral Analysis
Purpose:
- expose checkpoint-backed Koopman spectral workflows through stable atomic tools

Recommended tools:
- `build_spectral_snapshot(checkpoint_handle, dataset_handle=None, options=None)`
- `inspect_spectral_snapshot(spectral_snapshot_handle)`
- `compute_eigensystem(spectral_snapshot_handle, dt=1.0, filter=None)`
- `estimate_pseudospectrum(spectral_snapshot_handle, grid=None, mode="cont"|"disc", method="standard"|"sako", return_vec=False)`
- `estimate_spectral_measure(spectral_snapshot_handle, observable_spec, order, eps, thetas=101)`
- `evaluate_eigenfunctions(spectral_snapshot_handle, inputs, indices, rng=None)`
- `evaluate_eigenfunction_jacobian(spectral_snapshot_handle, ref=None, rng=None)`
- `evaluate_eigenmode_jacobian(spectral_snapshot_handle, ref=None, rng=None)`
- `plot_spectral_results(spectral_snapshot_handle, plot_spec)`

Recommended v1 scope:
- checkpoint-backed KBF and DKBF models
- full and low-rank Koopman weights
- pseudospectrum
- spectral measure
- eigenfunction and eigenmode Jacobians

Recommended v2 scope:
- broader model-family coverage
- richer modal filtering and residual reporting
- explicit spectral residual and spectral truncation tools

Structured outputs should include:
- spectral snapshot handle
- eigensystem summary
- residual summaries when available
- plot artifact paths
- numerical arrays or persisted result handles for downstream reuse

Notes:
- The object store already has a natural home for spectral snapshot handles.
- This is the cleanest near-term MCP expansion after training and basic evaluation.

## G. Data Interface and Transform Workflows
Purpose:
- expose learned and non-learned transform workflows for embedding, reconstruction, observables, and modes

Recommended tools:
- `build_data_interface(checkpoint_handle=None, config=None, dataset_handle=None, model_ref=None, device="cpu")`
- `inspect_data_interface(interface_handle)`
- `encode_data(interface_handle, payload, rng=None)`
- `decode_data(interface_handle, payload, rng=None)`
- `apply_observable(interface_handle, observable_spec)`
- `forward_modes(interface_handle, ref, rng=None)`
- `backward_modes(interface_handle, ref, rng=None)`
- `plot_embedding(interface_handle, dataset_handle=None, plot_spec=None)`
- `plot_reconstruction(interface_handle, dataset_handle=None, selection=None)`
- `plot_modes(interface_handle, ref, mode_type="forward"|"backward", plot_spec=None)`

Recommended structured observable spec fields:
- built-in observable name
- explicit callable alias from a controlled registry
- parameters for the observable

Recommended v1 scope:
- deterministic transform pipelines from config or checkpoint
- encode and decode
- forward and backward modes
- embedding and reconstruction plots for supported regular-data cases

Recommended v2 scope:
- graph-native interface handling
- richer observable registries
- batched derived artifact handles for repeated downstream use

Notes:
- This surface should lean on typed transform modules, not script-specific plotting code.
- A persistent interface handle may be useful if building the interface is expensive or reused across many calls.

## H. Visualization Utilities
Purpose:
- keep plotting separate from numerical computation where possible

Recommended tools:
- `visualize_model_graph(checkpoint_handle, ref_dataset_handle=None, depth=1)`
- `plot_training_history(run_handle)`
- `plot_tuning_results(tuning_handle)`
- `plot_rollouts(prediction_handle, ...)`
- `plot_spectral_results(spectral_snapshot_handle, plot_spec)`
- `plot_embedding(interface_handle, ...)`
- `plot_reconstruction(interface_handle, ...)`

Notes:
- Visualization tools should consume existing handles and persisted numerical results.
- They should not recompute expensive upstream analysis if a prior result handle already exists.

## Recommended Skill Surface

## 1. `dymad-train-model`
Purpose:
- translate a modeling and training request into one deterministic training run

Typical workflow:
1. Register and inspect datasets.
2. Discover compatible model families and profiles.
3. Translate free text into structured `model_ref`, `reference_profile`, and `config`.
4. Validate the config.
5. Materialize and run training.
6. Summarize checkpoint, run artifacts, and key metrics.

Skill responsibilities:
- choose model family
- choose reference profile
- map architecture requests into `model.*`
- map optimizer requests into `phases`
- reject incompatible dataset/model combinations

## 2. `dymad-tune-model`
Purpose:
- convert a tuning request into a structured param-grid search and summarize the outcome

Typical workflow:
1. Register and inspect datasets.
2. Discover compatible model families and profiles.
3. Translate the user request into `base_config` and `param_grid`.
4. Validate the tuning payload.
5. Run single-split tuning.
6. Summarize best config, metrics, checkpoint, and plots.
7. Optionally retrain or export the best configuration.

Skill responsibilities:
- translate phrases like "sweep hidden size and learning rate" into dotted-key grids
- choose a tuning metric
- explain that v1 is single-split tuning if the user asks for cross-validation

## 3. `dymad-evaluate-model`
Purpose:
- evaluate an existing checkpoint with richer metric and visualization workflows

Typical workflow:
1. Register or reuse checkpoint and test dataset handles.
2. Inspect compatibility.
3. Generate rollout predictions.
4. Compute requested metrics such as RMSE and VPT.
5. Select representative trajectories and plots.
6. Return an evaluation summary and artifact references.

Skill responsibilities:
- translate evaluation requests into structured metric specs
- choose representative plot selection rules
- explain unsupported graph-plot cases explicitly

## 4. `dymad-spectral-analysis`
Purpose:
- orchestrate the typical checkpoint-backed spectral workflow

Typical workflow:
1. Register or reuse checkpoint handle.
2. Build spectral snapshot.
3. Compute eigensystem summary.
4. Run requested spectral analyses:
   - pseudospectrum
   - spectral measure
   - eigenfunction evaluation
   - eigenfunction or eigenmode Jacobians
5. Generate requested plots.
6. Return a concise analysis report with handles and artifacts.

Skill responsibilities:
- translate free text like "show the pseudospectrum near the imaginary axis" into a grid spec
- translate observable requests into structured observable specs
- explain current model-family limitations

## 5. `dymad-data-interface`
Purpose:
- orchestrate embedding, reconstruction, observable, and mode workflows

Typical workflow:
1. Build a data interface from checkpoint or transform config.
2. Encode and decode representative data.
3. Compute forward or backward modes.
4. Produce requested plots such as embeddings, reconstructions, or mode visualizations.
5. Return summary metrics and artifact locations.

Skill responsibilities:
- translate requests like "embed into 2D diffusion map and show reconstruction" into structured transform config and plotting specs
- distinguish checkpoint-backed interfaces from config-only transform interfaces
- explain when the current implementation is regular-data-first

## Recommended Handle Types
Existing handles:
- dataset handle
- checkpoint handle
- training run handle
- evaluation handle
- prediction request handle
- spectral snapshot handle

Recommended additions:
- tuning handle
- prediction result handle
- data interface handle
- transform result handle for persisted encode/decode outputs when needed

Guidance:
- handles should represent persisted or reconstructible workflow state
- large numerical arrays should usually be written to artifacts and referenced by handle metadata rather than embedded directly in summaries

## Structured Config Translation Rules for Skills
Skills should translate natural-language requests into these structured buckets:
- model selection:
  - `model_ref`
  - `reference_profile`
- architecture:
  - `config.model`
- transforms:
  - `config.transform_x`
  - `config.transform_u`
  - related transform sections
- optimization:
  - `config.phases`
- tuning:
  - `param_grid`
  - tuning metric
- evaluation:
  - metric specs
  - plot specs
- spectral analysis:
  - grid specs
  - observable specs
  - filtering specs

Skills must not:
- pass raw free text to MCP tools
- write user prose into config fields that expect enums or numeric values
- invent unsupported metric names, model families, or transform types

## Recommended Rollout Order

## Phase 1: Tighten Current Surface
- keep existing dataset, checkpoint, training, evaluation, and object tools
- add model/profile discovery tools
- add config validation and config materialization tools
- expand the current train/eval skill into a recipe-driven training skill

## Phase 2: Add Tuning and Data Interface Tools
- add single-split tuning tools
- add data interface tools for encode, decode, and modes
- add visualization helpers for embeddings, reconstructions, and tuning plots

## Phase 3: Add Spectral Tooling
- add spectral snapshot construction
- add pseudospectrum and measure tools
- add eigenfunction and eigenmode Jacobian tools
- add spectral plotting tools

## Phase 4: Broaden Coverage
- broaden model-family support for spectral workflows
- add richer evaluation metrics and graph-native plotting
- add k-fold tuning once a real k-fold driver exists

## Out-of-Scope or Not Yet Stable for v1
- general free-form code generation from MCP requests
- unrestricted callable observables passed as arbitrary code
- claiming full cross-validation support while `KFoldDriver` is incomplete
- graph-native data-interface workflows without first tightening the underlying interface seam
- graph evaluation plotting as a stable first-class guarantee

## Immediate Implementation Priorities
1. Add model and profile catalog tools so skills can stop hard-coding recipe knowledge.
2. Add training config validation and config materialization tools.
3. Add single-split tuning tools built on existing driver support.
4. Split prediction, metrics, and plotting into separate evaluation tools.
5. Promote spectral snapshot and spectral adapter workflows into MCP tools.
6. Expose data-interface encode, decode, and mode operations through a dedicated handle-based tool family.

## Success Criteria
The MCP and skill design is successful when:
- the MCP layer is deterministic and typed
- skills can translate common user requests without custom per-case code
- workflows return reusable handles rather than only terminal text
- unsupported capabilities fail clearly and specifically
- the exposed surface mirrors stable internal seams instead of script-only workflows
