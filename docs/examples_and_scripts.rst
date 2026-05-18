:orphan:

Examples and Scripts
====================

DyMAD has two complementary learning surfaces:

- ``examples/`` is the starting point. The published :doc:`Examples <examples>` page is a notebook
  gallery for learning the main ideas step by step.
- ``scripts/`` is the broader runnable library. It contains more workflows, variants, utilities,
  and topic coverage than the notebook gallery.

Use them as a progression rather than expecting a strict mirror. The notebooks are introductory
tutorials, while the scripts are the place to look when you want more depth, more model variants,
or a runnable reference for a specific feature. Not every script has a matching notebook, and this
documentation does not assume a one-to-one mapping.

How to move from notebooks to scripts
-------------------------------------

Start with the notebook gallery when you want a guided walkthrough of the core DyMAD workflow.
Then move into ``scripts/`` when you want to:

- rerun a workflow from Python or the command line
- inspect the YAML configuration used for a model or dataset
- explore model families or analysis workflows that are not covered in the notebooks
- adapt an existing example to your own system

In many script folders, ``*_cli.py`` files are the most direct command-line entry points, nearby
``.py`` files show the underlying runnable example, and neighboring ``.yaml`` files provide the
configs those runs use.

Where to look in ``scripts/``
-----------------------------

The current script tree is broad, so it is easiest to navigate by topic:

- Introductory training workflows similar to the notebook material:
  ``scripts/linear_time_invariant/``, ``scripts/linear_graph/``, ``scripts/2d_koopman/``
- Kernel-based examples:
  ``scripts/ker_lti/``, ``scripts/ker_s1/``, ``scripts/ker_s1u/``, ``scripts/ker_lco/``
- Discrete-time, delayed, or variant system setups:
  ``scripts/lti_dt/``, ``scripts/ltg_dt/``, ``scripts/ltg_dt_tv/``, ``scripts/lti_delay/``,
  ``scripts/lti_1s/``, ``scripts/lti_vlen/``
- Reduced-order and PIROM workflows:
  ``scripts/pirom_dyn/``, ``scripts/pirom_res/``, ``scripts/pirom_res_dt/``
- Spectral-analysis-oriented examples:
  ``scripts/sa_lti/``, ``scripts/sa_lco/``, ``scripts/sa_2dk/``
- Data preparation, denoising, and post-processing utilities:
  ``scripts/denoise/``, ``scripts/kuramoto/``, ``scripts/vortex/``

If you began with a specific notebook topic, these are good next stops:

- After the linear and Koopman notebooks, compare the broader training and sweep examples under
  ``scripts/linear_time_invariant/``, ``scripts/linear_graph/``, and ``scripts/2d_koopman/``.
- After the spectral analysis notebooks, explore ``scripts/sa_lti/``, ``scripts/sa_lco/``, and
  ``scripts/sa_2dk/`` for additional analysis-oriented runs.
- After the vortex notebooks, look at ``scripts/vortex/vor_train_cli.py``,
  ``scripts/vortex/vor_proc_cli.py``, and ``scripts/vortex/vor_post.py`` for the surrounding
  training and processing workflow.

What this guide does not require
--------------------------------

You do not need a matching notebook for every runnable script. The intended structure is a curated
set of notebook tutorials in ``examples/`` plus a larger reference/demo surface in ``scripts/``.
