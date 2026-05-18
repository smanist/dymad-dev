From Notebooks To Scripts
=========================

DyMAD has two complementary example surfaces:

- ``examples/`` is the starting point. The notebook gallery gives guided, hands-on introductions to
  core workflows and modeling ideas.
- ``scripts/`` is the broader runnable demo and reference library. It covers more systems,
  training variants, preprocessing flows, and analysis workflows than the notebook gallery.

These surfaces are related, but they are not intended to be a strict one-to-one mirror. The
notebooks are curated tutorials for getting oriented. When you want wider coverage of the current
repository, ``scripts/`` is the next place to look.

Recommended Path
----------------

1. Start with the :doc:`notebook gallery <examples>` to learn the basic workflow in an interactive
   format.
2. Move to ``scripts/`` when you want to run additional model families, compare training setups,
   or explore topics that are not covered in the notebooks.
3. Prefer files named ``*_cli.py`` when they are available. In this repository, those are the
   user-facing script entry points that are meant to be runnable without editing the file first.

What You Will Find In ``scripts/``
----------------------------------

Most script folders combine one or more Python entry points with YAML configuration files. The
current top-level layout is a lightweight map of the available topics:

- Koopman training and parameter sweeps:
  ``scripts/2d_koopman/`` with entry points such as ``kp_train_cli.py``,
  ``kp_sweep_dt_cli.py``, and ``kp_sweep_ct_cli.py``.
- Linear time-invariant system workflows:
  ``scripts/linear_time_invariant/`` with ``lti_train_cli.py``, ``lti_multi_cli.py``, and
  ``lti_mp_cli.py``; related variants also live in ``scripts/lti_1s/``, ``scripts/lti_dt/``,
  ``scripts/lti_delay/``, and ``scripts/lti_vlen/``.
- Graph-structured dynamical systems:
  ``scripts/linear_graph/`` with ``ltg_train_cli.py``; related graph examples also live in
  ``scripts/linear_graph_auto/``, ``scripts/ltg_dt/``, and ``scripts/ltg_dt_tv/``.
- Kernel-based examples and related systems:
  ``scripts/ker_s1/``, ``scripts/ker_s1u/``, ``scripts/ker_lti/``, ``scripts/ker_lco/``, and
  ``scripts/lorenz63/`` with entry points such as ``ker_s1_cli.py``, ``ker_lti_cli.py``, and
  ``lor_train_cli.py``.
- Spectral analysis workflows:
  ``scripts/sa_2dk/``, ``scripts/sa_lco/``, and ``scripts/sa_lti/`` with analysis entry points
  such as ``kp_sa_cli.py`` alongside related configs.
- Data processing and application-oriented flows:
  ``scripts/vortex/`` for preprocessing, training, and postprocessing
  (``vor_proc_cli.py``, ``vor_train_cli.py``, ``vor_post.py``),
  ``scripts/denoise/`` for denoising workflows, and ``scripts/kuramoto/`` for data generation and
  training.
- PIROM and reduced-order modeling examples:
  ``scripts/pirom_dyn/``, ``scripts/pirom_res/``, and ``scripts/pirom_res_dt/`` with entry points
  such as ``dyn_train_cli.py`` and ``res_train_cli.py``.
- Benchmarking and focused experiments:
  ``scripts/benchmarks/runtime_hotpath.py`` and several developer-facing scripts that expose the
  workflow inline.

How To Use This Guide
---------------------

If a notebook introduces a topic and you want more depth, start by looking for the matching topic
area under ``scripts/`` rather than expecting a notebook for every variant. For example:

- after the ``2d_koopman`` notebooks, continue with ``scripts/2d_koopman/`` for training restarts
  and sweep runs
- after the ``linear_graph`` notebooks, continue with ``scripts/linear_graph/`` or ``scripts/ltg_dt/``
  for additional graph training variants
- after the ``vortex`` notebooks, continue with ``scripts/vortex/`` for preprocessing, training,
  and postprocessing steps

If you do not see a notebook for a topic, that does not mean DyMAD lacks an example. It often
means the runnable reference lives directly under ``scripts/``.
