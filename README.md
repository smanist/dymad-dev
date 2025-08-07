# Dynamics Modeling and Analysis with Data (DyMAD)

This is the GitHub site for DyMAD, developed and maintained by the [APUS Lab at Penn State](https://apus-lab.github.io/).

This README is mainly for developer use.  For documentation, refer to [readthedocs.org](https://dymad.readthedocs.org).

## Commandline Collection

**Installation:** During development, test the install by
```
pip install -e .
```
so that edits in the source are applied directly.

**Tests:** To run the tests, stay in the root and
```
pytest
```
For specific test script
```
pytest [filename]
```
For specific case in specific test script
```
pytest [filename]::[casename]
```

**Documentation:** Generate the document by
```
sphinx-build -E -b html docs docs/_build/html
```
Remove `-E` for incremental build.

If there are API errors such as some files not found, try deleting the auto-generated `api` folder and try again.

## Code Structure

- `docs`: Documentation based on ReadTheDocs.
- `examples`: Example cases used in documentation - some of them are still in progress of conversion.
- `scripts`: Quick cases for checking model performance, mainly used during development.
- `src`: The source code.
- `tests`: Tests used by pytest and CI workflow.  See the README therein for more details.

# TODO notes

- Include edge_attr etc in DynGeoData, see [here](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html)
- Verify the RNN-type training
- Update the aircraft example
- Update the double pendulum example
