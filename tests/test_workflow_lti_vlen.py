from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np


def _load_lti_vlen_script() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lti_vlen" / "lti_vlen.py"
    spec = spec_from_file_location("lti_vlen_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lti_vlen_example_writes_ragged_dataset(tmp_path) -> None:
    module = _load_lti_vlen_script()
    module.DATA_DIR = tmp_path

    module.generate_variable_length_data()

    data_path = tmp_path / "lti_vlen.npz"
    with np.load(data_path, allow_pickle=True) as payload:
        lengths = [len(item) for item in payload["t"]]
        state_widths = [item.shape[1] for item in payload["x"]]
        control_widths = [item.shape[1] for item in payload["u"]]

        assert payload["t"].dtype == object
        assert payload["x"].dtype == object
        assert payload["u"].dtype == object
        assert Counter(lengths) == {length: module.TRAJ_PER_LENGTH for length in module.LENGTHS}
        assert len(set(lengths)) == len(module.LENGTHS)
        assert state_widths == [2] * len(state_widths)
        assert control_widths == [1] * len(control_widths)
