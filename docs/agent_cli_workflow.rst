Agent and CLI Workflow
======================

DyMAD has two user-facing ways to run the same maintained training and evaluation workflow:

- The DyMAD agent interface is conversational. It translates user requests into structured MCP
  calls, compiles the request against the registry, launches training, and reports artifacts.
- The ``dymad`` package CLI is file-based. It accepts a YAML config, writes a run manifest, and
  makes the run easy to audit, rerun, or share.

Both paths use the same registry, compiler, executor, artifact store, and checkpoint boundary. The
agent is usually the fastest way to explore; the CLI is the stable way to preserve and rerun the
result.

Start With A Dataset
--------------------

Download the sample linear time-invariant dataset:
:download:`lti.npz <extra_files/lti.npz>`.

Put the file in a working folder. For example:

.. code-block:: text

   lti-agent-demo/
   `-- lti.npz

The dataset is an ``.npz`` time-series file with arrays compatible with the regular DyMAD training
workflow. It is intentionally small enough for examples and smoke tests, but the same flow applies
to your own ``.npz`` datasets.

Typical Conversational Prompts
------------------------------

These prompts are written the way a user might ask an agent to work. They are not Python scripts;
the agent turns the request into structured MCP tool calls and persisted run records.

Baseline weak-form fit:

.. code-block:: text

   In this folder is a dataset lti.npz. As a baseline, use a 4-state LTI model
   with a concat-type autoencoder where applicable to fit the system. Use the
   weak-form trainer. Report prediction error metrics and sample plots of
   prediction vs truth.

Tune the baseline:

.. code-block:: text

   Tune the weak-form LTI baseline to improve prediction accuracy. Try a small
   sweep over Koopman dimension and weak-form window size. Pick the model with
   the lowest rollout RMSE and summarize what changed.

Compare trainer choices:

.. code-block:: text

   Consider the same problem again, but now train one model with the weak-form
   trainer, one with the one-step trainer, and one with the NODE trainer.
   Compare the accuracy of the three trained models.

Improve one-step performance:

.. code-block:: text

   Can you adjust the one-step trainer parameters so its prediction accuracy is
   closer to the weak-form run? Keep the dataset and model family fixed, and
   report the before/after metrics.

If a request mentions a detail that is not a valid user-mode override for the selected model or
dataset kind, the agent should inspect the capability description and either translate it to a
supported override or ask for clarification. For example, graph model requests can include
``autoencoder_type: cat``; a regular LTI run uses the regular ``lti`` capability and its supported
model fields.

What Happens Under The Hood
---------------------------

A typical agent run follows this boundary:

.. code-block:: text

   user prompt
     -> MCP user tools
     -> registry capability lookup
     -> training request compiler
     -> persisted compiled request handle
     -> asynchronous training run handle
     -> checkpoint and evaluation handles
     -> metrics, logs, plots, and artifacts

The important MCP calls are:

- ``list_training_capabilities`` and ``describe_training_capability`` to discover models,
  supported dataset kinds, allowed overrides, trainer names, sweep support, and examples.
- ``compile_training_request`` to validate ``model_key``, dataset handles, phase overrides, sweep
  settings, run name, seed, device, and worker count.
- ``start_training_run`` to launch the compiled request and return a training-run handle.
- ``describe_training_run`` and ``read_training_run_log`` to poll status and inspect logs.
- ``evaluate_checkpoint`` to compute prediction metrics and produce prediction plots.

The agent stores intermediate objects behind handles such as ``ds_...``, ``trainreq_...``,
``run_...``, ``chk_...``, and ``eval_...``. This keeps the conversation compact and makes each
step inspectable.

Typical Outputs And Artifacts
-----------------------------

A successful run usually produces:

- A materialized training YAML file with the effective config used by the trainer.
- A model checkpoint, typically a ``.pt`` file.
- A training summary, often including loss history and final/best validation metrics.
- Training logs that can be read incrementally while the worker runs.
- Prediction plots such as prediction-vs-truth trajectories.
- Evaluation records with metrics such as ``rollout_rmse``.
- Optional sweep outputs, including CV result arrays and plots, when ``overrides.cv`` is used.

The exact filenames depend on the run name and artifact root, but the handles and run manifest
record the paths.

.. _dymad-cli:

Package CLI Reference
---------------------

The CLI gives the same workflow a reproducible file interface. A minimal config for the sample
dataset is:

.. code-block:: yaml

   version: 1
   model_key: lti
   data:
     train:
       path: lti.npz
     test:
       path: lti.npz
   overrides:
     model:
       koopman_dimension: 4
     phases:
       - trainer: Weak
         name: weak_baseline
         n_epochs: 25
         learning_rate: 0.005
   run:
     name: lti_weak_baseline
     seed: 123
     device: cpu
     max_workers: 1
   evaluation:
     metric: rollout_rmse
     plot_selection: median
     max_plots: 1

Save this as ``lti_weak.cli.yaml`` next to ``lti.npz``. Then run:

.. code-block:: bash

   dymad config validate lti_weak.cli.yaml --out runs/lti_weak_baseline
   dymad train --config lti_weak.cli.yaml --out runs/lti_weak_baseline
   dymad status --run runs/lti_weak_baseline
   dymad log --run runs/lti_weak_baseline
   dymad eval --run runs/lti_weak_baseline
   dymad report --run runs/lti_weak_baseline

Common commands:

- ``dymad config schema`` prints the JSON Schema for CLI config files.
- ``dymad config validate CONFIG --out RUN_DIR`` validates paths, model keys, overrides, and the
  effective compiled training request without starting a run.
- ``dymad registry list models --json`` lists available user-facing model keys.
- ``dymad registry list training --json`` lists training capabilities.
- ``dymad train --config CONFIG --out RUN_DIR`` starts a run and waits for completion.
- ``dymad train --config CONFIG --out RUN_DIR --detach`` starts a run and returns immediately.
- ``dymad status --run RUN_DIR --json`` reads the run manifest and current training state.
- ``dymad log --run RUN_DIR`` prints the worker log.
- ``dymad log --run RUN_DIR --follow`` follows the log until the run reaches a terminal state.
- ``dymad eval --run RUN_DIR`` evaluates the latest checkpoint using the config's test data.
- ``dymad eval --run RUN_DIR --test-data OTHER.npz`` evaluates against another dataset.
- ``dymad report --run RUN_DIR --json`` summarizes the run, checkpoint, metrics, evaluations, and
  artifacts.

Trainer Variants In CLI Config
------------------------------

To compare trainers, keep the dataset and model fixed, then change only ``run.name`` and
``overrides.phases``.

Weak-form phase:

.. code-block:: yaml

   overrides:
     phases:
       - trainer: Weak
         name: weak
         n_epochs: 25
         learning_rate: 0.005
         weak_form_params:
           N: 13
           dN: 2
           ordpol: 2
           ordint: 2

One-step phase:

.. code-block:: yaml

   overrides:
     phases:
       - trainer: OneStep
         name: one_step
         n_epochs: 25
         learning_rate: 0.005

NODE phase:

.. code-block:: yaml

   overrides:
     phases:
       - trainer: NODE
         name: node
         n_epochs: 25
         learning_rate: 0.005

Small sweep for tuning:

.. code-block:: yaml

   overrides:
     cv:
       param_grid:
         model.koopman_dimension: [3, 4, 5]
         phases.0.weak_form_params.N: [9, 13, 17]
       metric: total
       selection:
         goal: minimize
         tie_breakers: [std_metric, combo_index]

Auditability And Reproducibility
--------------------------------

Every CLI run writes ``dymad-run.json`` under the run directory. That manifest records:

- The source config path.
- The normalized config after path resolution and defaults.
- The run directory, artifact root, and local ``.dymad-store`` path.
- Dataset paths and dataset handles.
- The compiled request handle and training-run handle.
- The latest status, checkpoint handle, metrics, artifacts, and evaluation handles.

This means a result can be audited without reconstructing the original conversation. The agent can
export or point to the same CLI config, and a reviewer can rerun the workflow with ``dymad train``
and inspect it with ``dymad report``.

Token Usage: Agent Versus Starting From Scratch
-----------------------------------------------

The exact token count depends on the model, dataset, number of tuning iterations, and how much log
or artifact content is summarized. As a practical estimate:

- Using the DyMAD agent for the prompt-driven workflow often takes a few thousand to low tens of
  thousands of tokens. Most of the work is registry inspection, request compilation, status
  polling, metric summaries, and concise artifact reporting.
- Writing a bespoke training script from scratch commonly takes several times more interaction
  budget: the agent must inspect APIs, write data loading, choose model classes, assemble trainer
  configs, debug shape/config errors, add evaluation code, and then explain how to reproduce the
  run.

For established DyMAD workflows, the agent/MCP/CLI path is usually more token-efficient because the
stable registry and compiler replace large amounts of exploratory code generation. For new
algorithms or unsupported workflows, the advantage narrows because the agent must design and test
new package behavior rather than assemble an existing capability.
