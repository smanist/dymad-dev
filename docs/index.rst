Dynamics Modeling and Analysis with Data (DyMAD)
================================================

DyMAD aims to provide a lightweight and user-friendly toolkit for modeling and analyzing dynamical systems using data-driven approaches.

Currently, we have implemented the following features:

- Data preprocessing pipeline specialized for time series data.
- Models for different types of dynamical systems, including
   * Latent Dynamics Models
   * Koopman Bilinear Forms
   * The graph version of the above
- Training utilities, including
   * Neural-ODE-based optimizer
   * Weak-form optimizer
- Miscellaneous utilities, including
   * Samplers for inputs and initial conditions
   * Visualization tools

It is still far from complete, see our :doc:`Roadmap <roadmap>` for more details.

The code is hosted on `GitHub <https://github.com/apus-lab/dymad>`_.


Developers
------------

The package is developed by the `APUS Lab at Penn State <https://apus-lab.github.io/>`_, directed by Dr. Daning Huang.

The lead developer is Dr. Yin Yu, whose PhD thesis is on the topic of data-driven modeling of dynamical systems on graphs.


Explore More
------------

.. toctree::
   :maxdepth: 2

   getting_started
   roadmap
   api_ref
