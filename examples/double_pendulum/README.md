# Double Pendulum

This directory contains the supported double-pendulum example surface:

- `double_pendulum_train.py`: script entrypoint for the full training configs.
- `dp_ldm_node.yaml`, `dp_ldm_wf.yaml`, `dp_kbf_node.yaml`, `dp_kbf_wf.yaml`: tracked training configs for LDM/KBF with NODE or weak-form training.
- `double_pendulum_data.yaml`: data-generation config used by the script and by any local notebook runs.
- `deprecated/`: historical material only; it is not the supported entrypoint.

The tracked configs now use explicit `phases` entries and pin `data.split_seed` so the train/validation split is reproducible.

## Quick Start

Script:

```bash
cd examples/double_pendulum
python double_pendulum_train.py
```

`double_pendulum_train.py` selects one of the tracked configs through the `case` index:

- `0`: `LDM` with `NODETrainer` using `dp_ldm_node.yaml`
- `1`: `LDM` with `WeakFormTrainer` using `dp_ldm_wf.yaml`
- `2`: `KBF` with `NODETrainer` using `dp_kbf_node.yaml`
- `3`: `KBF` with `WeakFormTrainer` using `dp_kbf_wf.yaml`

Training artifacts are written next to this example under the configured model name.
