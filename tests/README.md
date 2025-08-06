Organization of the test cases:

- YAML files: Contains the definitions of data and model.
- `conftest.py`: The pytest fixtures that serve as inputs to test cases and make clean-ups when necessary.
- `test_assert_*`: Exact tests that compare test outputs with reference outputs to numerical accuracy.
  - `trajmgr`: Trajectory manager
  - `trajmgr_graph`: Trajectory manager for graph data
  - `transform`: Data transformations
  - `weak`: Weak form parameters
- `test_workflow_*`: Tests that check the flow of execution, esp. the training process.  Does not check numerical accuracy.
  - `kp`: Autonomous dynamics, based on a classical 2D Koopman model
  - `ltg`: Dynamics with inputs on graph, based on a LTI model
  - `ltga`: Autonomous dynamics on graph, based on a LTI model
  - `lti`: Dynamics with inputs, based on a LTI model
  - `sample`: Sampling functionalities
