Roadmap
=======

The planned algorithms and features for DyMAD are mainly based on the research work from `APUS Lab at Penn State <https://apus-lab.github.io/>`_.

Highlights include:

1. `Learning Coarse-Grained Dynamics on Graphs <https://doi.org/10.1016/j.physd.2025.134801>`_ By Yu, Y., Harlim, J., Huang, D. and Li, Y.
2. `Learning Networked Dynamical System Models with Weak Form and Graph Neural Networks <https://arxiv.org/abs/2407.16779>`_ By Yu, Y., Huang, D., Park, S. and Pangborn, H.
3. `Physics-Infused Reduced-Order Modeling for Analysis of Multi-Layered Hypersonic Thermal Protection Systems <https://arxiv.org/abs/2505.22890>`_ By Vargas Venegas, C.A., Huang, D., Blonigan, P. and Tencer, J.
4. `Physics-Infused Reduced-Order Modeling of Aerothermal Loads for Hypersonic Aerothermoelastic Analysis <https://doi.org/10.2514/1.J062214>`_ By Vargas Venegas, C.A. and Huang, D.
5. `Global Description of Flutter Dynamics via Koopman Theory <https://arxiv.org/pdf/2505.14697>`_ By Song, J. and Huang, D.
6. `Modal Analysis of Spatiotemporal Data via Multivariate Gaussian Process Regression <https://doi.org/10.2514/1.J064185>`_ By Song, J. and Huang, D.
7. `Learning vector fields of differential equations on manifolds with geometrically constrained operator-valued kernels <https://openreview.net/pdf?id=OwpLQrpdwE>`_ By Huang, D., He, H., Harlim, J. and Li, Y.

What Problems are we solving?
-----------------------------

Overall, we are interested in dynamical systems of the following form,

.. math::
    \begin{align*}
    \dot{z} &= f(z, u, t) \\
    x &= g(z, u, t)
    \end{align*}

Comment:

    In control and dynamics community, the states are usually denoted as :math:`x`, and the observation as :math:`y`.

    But as later autoencoder-type structures will be used, we use the convention that the latent states as :math:`z` and the observation as :math:`x`.

Given a set of trajectory data, e.g., :math:`\mathcal{D}=\{(t_i, u_i, x_i)\}_{i=1}^N`, we aim to

- **Modeling**: Learn the model :math:`f` and :math:`g` from data, where the model may possess certain structures, such as
    - Bilinearity
    - Topological sparsity
    - Manifold geometry
    - etc.
- **Analysis**: When the model is learned, quantify its dynamical properties, such as
    - Stability, Controllability, Observability
    - Spectrum and Eigenfunctions
    - etc.

Comment:

    There are some easier cases:

    - If :math:`f` and :math:`g` are linear, then there are standard state space identification methods, such as ERA.
    - If :math:`x = z`, then one typical approach is to estimate :math:`\dot{z}` and fit :math:`f` directly, as is done, e.g., in SINDy-type methods.

    But we are more interested in the general cases, where :math:`f` and :math:`g` are nonlinear and :math:`x\neq z`,
    esp. when :math:`\mathrm{dim}(x)` and :math:`\mathrm{dim}(z)` are drastically different.


When are these problems relevant?
---------------------------------

Besides generic dynamical systems, that some purely data-driven methods are applicable,
we have several particular applications of interest:

Physics-Infused Reduced-Order Modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Part of :math:`f` and :math:`g` are known from physics, and we want to learn the rest from data.
- The physical part enhances the model interpretability and generalizability, esp. to new scenarios and conditions, while the data-driven part enhances the model accuracy and robustness.
- The challenging case is where the known physics only provides part of :math:`\dot{z}`, and there are additional hidden dynamics to be learned.
- Concrete examples
    - Physics model: Integral equation for boundary layer (3-states); Data: CFD simulations (millions of states). [4]
    - Physics model: Lumped model for heat condution (<5 states); Data: FEM simulations (thousands of states). [3]

Dynamics on graphs
^^^^^^^^^^^^^^^^^^

- The dynamics has a sparse structure: The states are naturally grouped, and a state group interacts with only a few other groups following a given topology.
- Such sparse structure can be represented as a graph, where the nodes are the state groups and the edges represent the interactions.
- Concrete examples
   - Power grids: The generators, transformers, and loads are connected via power lines, forming a natural graph. [1]
   - Electrified aircraft: the thermal, mechanical, and electrical devices are coupled, but each device only interacts with 1-3 other devices. [2]
   - Weather systems: Regions of Earth in general interact with their neighbors, but not with the whole globe within a short time.
   - Vortex dynamics: The vortices in some fluid flow interact with their neighbors, while global interactions are weak or negligible.

Dynamics on manifolds
^^^^^^^^^^^^^^^^^^^^^

- While the state dimension is high, the dynamics is constrained to a low-dimensional manifold.
- Leveraging the manifold geometry can help with sample efficiency and model size, e.g., a few states and several hundred samples to predict millions of observations.
- But geometry also poses challenges, e.g., in the identification of manifold and the preservation of geometry in prediction.
- Concrete examples
   - Canonical example in fluid dynamics: Vortex street. [7]
   - Mechanical systems: Rigid-body dynamics evolves on SE(3), and multi-body problems have additional algebraic constraints.
   - Structural dynamics: The dynamic responses of a structural often manifest in a limited combination of modes, forming a natural manifold. [5]

Comment:

    The scope of applications continues to expand as we explore more problems and methods at the `APUS Lab at Penn State <https://apus-lab.github.io/>`_.


How do we solve
---------------

Modeling
^^^^^^^^

We formulate the learning problem as a differentially constrained optimization problem,

.. math::
    \begin{align*}
    \min_{\theta} &\quad \int_{t_1}^{t_N} \lVert x-\hat{x} \rVert^2 dt + \mathcal{R}(\theta) \\
    \mathrm{s.t.} &\quad z = h(x, u, t; \theta) \\
    &\quad \dot{z} = f(z, u, t; \theta) \\
    &\quad x = g(z, u, t; \theta)
    \end{align*}

where 

- :math:`\theta` model parameters, e.g., neural network parameters or other model components.
- The objective is to minimize the discrepancy between the model prediction :math:`x` and the data :math:`\hat{x}`, with possibly a regularization term :math:`\mathcal{R}(\theta)`.
- The problem is subject to the differential constraints imposed by the dynamical system.
   - An additional :math:`h` is introduced as a state estimator to recover :math:`z` from :math:`x`.
   - In machine learning context, the pair :math:`h` and :math:`g` forms an encoder-decoder structure.

The optimization problem can be solved using various methods, some of which we (plan to) implement:

- **Neural-ODE-based optimizer**
   - Uses the adjoint method to compute gradients of the loss function w.r.t. :math:`\theta`.
   - Accurate if starting from a good initial guess; otherwise possibly slow convergence.
   - Furthermore, solving the adjoint equations can be computationally expensive, especially for long time horizons.
- **Weak-form optimizer** [2]
   - Uses weak form to convert the differential constraints to algebraic ones, and lump the latter as penalties in the objective.
   - Usually agnostic of initial guess, and can converge quickly.
   - However, it may not be as accurate as the Neural-ODE-based optimizer.
- **Kalman filter-based optimizer**
   - Uses Kalman filtering techniques to estimate the state and optimize :math:`\theta`.
- **Linear solver** [7]
   - When the model has certain linear structures that can be exploited, we can use linear solvers to directly solve the optimization problem.
- **Combinations of the above**
   - Sequentially apply the above methods to the whole problem, e.g., weak-form for fast convergence then Neural ODE for refinement, and/or
   - Apply the methods to different parts of the problem, e.g., linear solver for linearizable parts and weak-form for the rest.

Alternatively, we can discretize the system, and learn the discrete-time dynamics directly,

.. math::
    \begin{align*}
    z_k &= h(x_k, u_k, t_k; \theta) \\
    z_{k+1} &= f(z_k, u_k, t_k; \theta) \\
    x_k &= g(z_k, u_k, t_k; \theta)
    \end{align*}

where the model form can be, e.g., recurrent NNs, and the optimization problem can be solved using standard gradient descent methods.


Analysis
^^^^^^^^

Leveraging its structure, the learned model can be analyzed to extract dynamical properties.  Examples include:

- **Linear models**
   - All the linear system analysis methods can be applied - Modal analysis, controllability, observability, resolvent, etc.
- **Bilinear models** [5,6]
   - Using Koopman theory, the linear results can be extended.
- **Control-affine models**
   - Either linearize to linear/bilinear models and apply the above methods, or
   - Geometric Control methods: e.g., use Lie derivatives to perform non-local analysis.
- **Models on manifolds** [7]
   - Use the manifold geometry to perform analysis, e.g., using the Laplace-Beltrami operator to compute eigenfunctions and eigenvalues.
   - Identify symmetry and invariance properties of the system, from a Lie group perspective.


Table of Features
-----------------

Legends:

- |:white_check_mark:| Implemented in DyMAD
- |:building_construction:| Implemented in DyMAD but still need verification
- |:o:| Implemented in our other repos; to be integrated into DyMAD
- |:egg:| On-going development
- |:notepad_spiral:| Planned for future development
- |:heavy_multiplication_x:| Not applicable

Model and Optimizer
^^^^^^^^^^^^^^^^^^^

So far we consider a range of models and optimizers.  For models,

- Latent Dynamics Model (LDM) follows the generic formulations :math:`(f,g,h)` listed above.
- Koopman Bilinear Form (KBF) replaces :math:`g` with a bilinear model, :math:`A z + \sum_i B(u_i)z + B_0 z`.
- Recurrent Neural Network (RNN) includes standard architectures, e.g., LSTM and GRU.
- Kernel methods include standard kernels, and diffusion map, etc.
- Lastly, LDM/KBF/RNN can work with data on graphs.

For optimizers, a brief list is already provided above, and here we note that,

- Some models are continuous-time (CT), some are discrete-time (DT), and some can be either.
- Similarly, some optimizers are CT-based, some are DT-based, and some can be either.

The details are provided below.

.. list-table::
   :widths: 30 20 25 25 25 25 25

   * - 
     - CT
     - CT
     - CT/DT
     - CT/DT
     - DT
     - DT
   * - 
     - NODE
     - Weak Form
     - Kalman
     - Linear
     - Single-step
     - Multi-step
   * - (Graph) LDM
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:building_construction:|
     - |:building_construction:|
   * - (Graph) KBF
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:building_construction:|
     - |:building_construction:|
   * - (Graph) RNN
     - |:heavy_multiplication_x:|
     - |:heavy_multiplication_x:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:building_construction:|
     - |:building_construction:|
   * - Kernel
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:o:|
     - |:o:|
     - |:egg:|

Model Analysis
^^^^^^^^^^^^^^

Once we obtain a model, we can analyze it to extract dynamical properties.

The most straightforward way is certainly linearizing the model and applying the standard linear system analysis methods.

As explained above, we can also leverage the model structure to perform more refined nonlinear analysis.

The details are provided below.

.. list-table::
   :widths: 25 25 15 35 25

   * -
     - Spectrum/Stability
     - Modes
     - Controll-/Observ-ability
     - Symmetry
   * - Linear
     - |:o:|
     - |:o:|
     - |:o:|
     - |:notepad_spiral:|
   * - (Graph) LDM
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
   * - (Graph) KBF
     - |:o:|
     - |:o:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
   * - (Graph) RNN
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|
   * - Kernel
     - |:egg:|
     - |:egg:|
     - |:notepad_spiral:|
     - |:notepad_spiral:|

Supporting Functions
^^^^^^^^^^^^^^^^^^^^

Lastly, to ease the construction of modeling and analysis pipelines, we provide a set of tools.

.. list-table:: Data Pre-processing
   :widths: 25 25 25 25 25

   * - Normalization
     - Time delay
     - SVD/PCA
     - Polynomial
     - Fourier
   * - |:white_check_mark:|
     - |:white_check_mark:|
     - |:o:|
     - |:o:|
     - |:o:|

.. list-table:: Sampling
   :widths: 25 25 25 25 25

   * - Control
     - Chirp
     - Gaussian
     - Sine
     - Sphere
   * -
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - Init. Cond.
     - Gaussian
     - Grid
     - Random uniform
     - LDS
   * -
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:o:|

.. list-table:: Miscellaneous
   :widths: 25 25 25 25

   * - Control interpolator
     - Zeroth-order
     - Linear
     - Cubic
   * -
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - Infrastructure
     - YAML case parser
     - Logging
     - Checkpoint
   * -
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - Plotting
     - Loss history
     - Model prediction
     -
   * -
     - |:white_check_mark:|
     - |:white_check_mark:|
     -
