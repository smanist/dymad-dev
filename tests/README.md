Organization of the test cases:

- YAML files: Contains the definitions of data and model.
- `*_baselines.json`: Reference metrics for slow deterministic regression tests.
- `conftest.py`: The pytest fixtures that serve as inputs to test cases and make clean-ups when necessary.
- `test_assert_*`: Exact tests that compare test outputs with reference outputs to numerical accuracy.
  - `di`: The DataInterface class
  - `dm`: Diffusion map
  - `grad`: Gradients
  - `graph`: Graph-related utilities
  - `krr`: Kernel ridge regression
  - `krr_tan`: Kernel ridge regression for the manifold case, also includes DMF besides the KRR classes.
  - `linalg`: Linear algebra
  - `loss`: Loss functions
  - `manifold`: Manifold-related calculations
  - `resolvent`: Resolvent analysis of linear systems
  - `spectrum`: Kernels used in spectral calculation
  - `trajmgr`: Trajectory manager
  - `trajmgr_graph`: Trajectory manager for graph data
  - `trans_lift`: Data transformations by the Lift class
  - `trans_mode`: Forward and backward modes in data transformations
  - `trans_ndr`: Data transformations by the NDR classes
  - `transform`: Data transformations
  - `weak`: Weak form parameters
  - `wrapper`: Wrapper for external code
- `test_workflow_*`: Tests that check the flow of execution, esp. the training process.  Does not check numerical accuracy.
  - `ker_auto`: Kernel-based dynamics, autonomous
  - `ker_ctrl`: Kernel-based dynamics, with inputs
  - `kp`: Autonomous dynamics, based on a classical 2D Koopman model
  - `ltg`: Dynamics with inputs on graph, based on a LTI model
  - `ltga`: Autonomous dynamics on graph, based on a LTI model
  - `lti`: Dynamics with inputs, based on a LTI model
  - `sa_lti`: Spectral analysis of a LTI model
  - `sample`: Sampling functionalities
- `test_contract_*`: Deterministic contract tests for typed runtimes, adapters, boundaries, persistence,
  model-spec resolution, and other public/runtime surfaces. These validate invariants and interface
  behavior rather than numerics-heavy reference values.
- `test_agent_*`: Deterministic integration tests for agent-facing surfaces, including the registry,
  compiler/executor flow, MCP/demo/user tools, and skill staging.
- `test_slow_*`: Slow deterministic regression cases, usually CLI or end-to-end flows checked against
  exact baseline metrics.
