# keystone

Internal project for merging the methods in learning nonlinear dynamics via a weak-form formulation.

Depending on how one obtains the states from observations, we aim to implement:

- Reconstruction-based method: A non-local operator is applied to the observations, and the aggregated info is mapped to the states.
- Optimization-based method: An optimization is solved to infer the states from a trajectory of observation.
- Plus a conventional method for discrete-time models, as a baseline

## Modules

:o: for long-term development

- Model
  + [x] Dynamics (continuous-time (CT), discrete-time (DT))
  + [x] Transform (Encoders, Decoders)
  + [x] Collection of model components
  + [ ] :o: Non-local operator (NLP)
- Time discretization
  + [x] None (DT dynamics)
  + [x] Data-based, evenly sampled
    - [x] Jacobi
    - [x] Naive
    - [ ] :o: weakSINDy's scheme
  + [ ] :o: Interpolation-based, quadrature "samples"
- Losses
  + [x] Dynamics
    - [x] Multi-step for DT
    - [x] Weak form for CT
  + [x] Reconstruction
- Data manager
  + [x] Normalization
  + [x] Trajectory segmentation
- Optimizer
  + [x] Skeleton code for training
  + [x] Two-step optimization
  + [ ] :o: Scheduler for learning rate
  + [ ] :o: Scheduler for horizon
  + [ ] :o: Bi-level optimization (for PIROM-type)
  + [ ] :o: Gauss-Newton?
- Visualize
  + [x] Training history
  + [x] Trajectories

## Main applications
With special functionalities listed

- Discrete-time model
  + [x] Baseline
- Koopman bilinear form
  + [ ] :o: Post-processing of system for, e.g., eigenfunctions, timescales, stability
- Physics-infused reduced-order modeling
  + [ ] :o: Wrapper of dynamics for evaluation and jacobian
  + [ ] :o: Bi-level optimization with adjoint implementation
  + [ ] :o: Optimization with only first-order information
- Dynamics on graph
  + [ ] :o: Model definition
  + [ ] :o: Data manager
- Dynamics on manifolds
  + [ ] :o: GMLS as encoder - interface to another repo?

