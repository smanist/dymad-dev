# Dynamics Modeling and Analysis with Data (DyMAD)

This is the GitHub site for DyMAD, developed and maintained by the [APUS Lab at Penn State](https://apus-lab.github.io/).

This README is mainly for developer use.  For documentation, refer to [readthedocs.org](https://dymad.readthedocs.org).

During development, test the install by
```
pip install -e .
```
so that edits in the source are applied directly.

Generate the document by
```
sphinx-build -E -b html docs docs/_build/html
```
Remove `-E` for incremental build.

If there are API errors such as some files not found, try deleting the auto-generated `api` folder and try again.

# TODO notes

- Extend DynGeoData with subgraph to parallelize batch processing
- Include edge_attr etc in DynGeoData, see [here](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html)
- Verify the RNN-type training
- Update the aircraft example
- Update the double pendulum example
