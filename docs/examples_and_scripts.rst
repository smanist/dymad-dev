Examples and Scripts
====================

DyMAD has two complementary example surfaces:

- :doc:`examples` is the starting point. The notebooks are guided tutorials for learning the main
  workflow and vocabulary.
- ``scripts/`` is the broader runnable reference and demo library. It covers more topics than the
  notebook gallery and is the next place to look when you want deeper coverage or a variant that
  is not documented as a notebook.

The notebook gallery is intentionally not a one-to-one mirror of ``scripts/``. Some topics have
both notebooks and scripts, while others are currently available only as runnable scripts.

Suggested progression
---------------------

1. Start with :doc:`examples` for a guided introduction.
2. When you want to rerun an idea with different configs, look for the corresponding directory
   under ``scripts/``.
3. When you want a topic that is not covered in the notebooks, browse ``scripts/`` by system or
   workflow area.

Many script directories include ``*_cli.py`` entry points alongside the YAML configs and helper
code needed to run the example.

Where to look in ``scripts/``
-----------------------------

Use the current directory layout as a lightweight topic index:

- Linear time-invariant and Koopman-style training:
  ``scripts/linear_time_invariant/``, ``scripts/lti_1s/``, ``scripts/lti_dt/``,
  ``scripts/lti_delay/``, ``scripts/lti_vlen/``, and ``scripts/2d_koopman/``.
- Graph-structured dynamical systems:
  ``scripts/linear_graph/``, ``scripts/linear_graph_auto/``, ``scripts/ltg_dt/``, and
  ``scripts/ltg_dt_tv/``.
- Kernel model workflows:
  ``scripts/ker_s1/``, ``scripts/ker_s1u/``, ``scripts/ker_lti/``, and ``scripts/ker_lco/``.
- Spectral analysis examples:
  ``scripts/sa_2dk/``, ``scripts/sa_lco/``, and ``scripts/sa_lti/``.
- Denoising, preprocessing, and post-processing:
  ``scripts/denoise/`` and ``scripts/vortex/``.
- Additional system-specific demos:
  ``scripts/lorenz63/``, ``scripts/kuramoto/``, ``scripts/pirom_dyn/``,
  ``scripts/pirom_res/``, and ``scripts/pirom_res_dt/``.

If you started in a notebook topic such as 2D Koopman, linear graphs, denoising, SA-LCO, or
vortex, checking the similarly named directory under ``scripts/`` is usually the fastest way to
find more runnable variants. For example:

- the 2D Koopman notebooks lead naturally into ``scripts/2d_koopman/`` for training and sweep
  runs
- the linear graph notebooks lead naturally into ``scripts/linear_graph/`` and the related
  ``scripts/ltg_dt/`` families
- the denoising notebook points toward ``scripts/denoise/`` for the training and preprocessing
  steps

When you are not sure where to start, open the notebook gallery first and then use the closest
matching directory under ``scripts/`` as the broader reference surface.
