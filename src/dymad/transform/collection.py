import logging
import numpy as np
from typing import Any, Dict, List, Union

from dymad.transform.base import AddOne, DelayEmbedder, Identity, Lift, Scaler, SVD, Transform
from dymad.transform.ndr import DiffMap, DiffMapVB, Isomap

Array = List[np.ndarray]

logger = logging.getLogger(__name__)

class Compose(Transform):
    """Apply transforms in order.  Inverse is applied in reverse."""
    def __init__(self, transforms: List[Transform] = None):
        super().__init__()

        if transforms is None:
            # Reload from state_dict is expected.
            return

        self.T = transforms
        self._T_names = [str(t) for t in transforms]
        self.NT = len(transforms)

        _n = self._T_names.count("delay")
        if _n > 1:
            raise ValueError(f"Compose: Multiple delay transforms ({_n}) are not allowed. "
                             "Please use only one delay transform in the composition.")
            # This is to reduce bookkeeping complexity in trajectory manager.
        elif _n == 1:
            _k = self._T_names.index("delay")
            self.delay = self.T[_k].delay
        else:
            self.delay = 0

    def __str__(self):
        return "compose"
    
    def _proc_rng(self, rng: Union[List, None]) -> List:
        if rng is not None:
            assert len(rng) == 2, "Range should be a list of two integers [start, end]."
            assert 0 <= rng[0] < rng[1] <= self.NT, f"Range should be within [0, {self.NT}]."
            return rng
        else:
            return [0, self.NT]

    def fit(self, data: Array) -> None:
        """"""
        _d = data
        for t in self.T:
            t.fit(_d)
            _d = t.transform(_d)

        self._inp_dim = self.T[0]._inp_dim
        self._out_dim = self.T[-1]._out_dim

        for _i in range(len(self.T)-1):
            assert self.T[_i]._out_dim == self.T[_i+1]._inp_dim, \
                f"Compose: Output dimension of transform {_i} ({self.T[_i]._out_dim}) " \
                f"does not match input dimension of transform {_i+1} ({self.T[_i+1]._inp_dim})."

    def append(self, t: Transform) -> None:
        """Append a transform to the composition."""
        if not isinstance(t, Transform):
            raise ValueError("Can only append Transform objects.")
        if self.NT > 0:
            if self.T[-1]._out_dim != t._inp_dim:
                raise ValueError(f"Compose: Output dimension of last transform ({self.T[-1]._out_dim}) "
                                 f"does not match input dimension of new transform ({t._inp_dim}).")
        self.T.append(t)
        self._T_names.append(str(t))
        self.NT += 1
        if str(t) == "delay":
            if self.delay > 0:
                raise ValueError("Compose: Multiple delay transforms are not allowed. "
                                 "Please use only one delay transform in the composition.")
            self.delay = t.delay
        self._out_dim = self.T[-1]._out_dim

    def pop(self) -> Transform:
        """Pop the last transform from the composition."""
        if self.NT <= 1:
            raise ValueError(f"Compose: {self.NT} transforms; no point to pop.")
        t = self.T.pop()
        self._T_names.pop()
        self.NT -= 1
        if str(t) == "delay":
            self.delay = 0
        self._out_dim = self.T[-1]._out_dim
        return t

    def transform(self, data: Array, rng: Union[List, None] = None) -> Array:
        """"""
        _rng = self._proc_rng(rng)
        for _i in range(_rng[0], _rng[1]):
            data = self.T[_i].transform(data)
        return data

    def inverse_transform(self, data: Array, rng: Union[List, None] = None) -> Array:
        """"""
        _rng = self._proc_rng(rng)
        for _i in range(_rng[1]-1, _rng[0]-1, -1):
            data = self.T[_i].inverse_transform(data)
        return data

    def get_forward_modes(self, ref=None, rng: Union[List, None] = None, **kwargs) -> np.ndarray:
        """"""
        _rng = self._proc_rng(rng)
        modes = self.T[_rng[0]].get_forward_modes(ref, **kwargs)
        for _i in range(_rng[0]+1, _rng[1]):
            if ref is not None:
                ref = self.T[_i-1].transform([ref])[0]
            tmp = self.T[_i].get_forward_modes(ref, **kwargs)
            modes = tmp.dot(modes)
        return modes

    def get_backward_modes(self, ref=None, rng: Union[List, None] = None, **kwargs) -> np.ndarray:
        """"""
        _rng = self._proc_rng(rng)
        modes = self.T[_rng[1]-1].get_backward_modes(ref, **kwargs)
        for _i in range(_rng[1]-2, _rng[0]-1, -1):
            if ref is not None:
                ref = self.T[_i+1].inverse_transform([ref])[0]
            tmp = self.T[_i].get_backward_modes(ref, **kwargs)
            modes = modes.dot(tmp)
        return modes

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "type": "Compose",
            "names": self._T_names,
            "delay": self.delay,
            "children": [t.state_dict() for t in self.T],
            "inp": self._inp_dim,
            "out": self._out_dim,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"Compose: Loading parameters from checkpoint")
        self._T_names = d["names"]
        self.T = []
        for name, sd in zip(self._T_names, d["children"]):
            if name not in TRN_MAP:
                raise ValueError(f"Unknown transform type in Compose: {name}")
            self.T.append(TRN_MAP[name]())
            self.T[-1].load_state_dict(sd)
        self.NT = len(self.T)
        self.delay = d["delay"]
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]

#: Mapping of transform names to classes.
TRN_MAP = {
    str(AddOne()):        AddOne,
    str(Compose()):       Compose,
    str(DelayEmbedder()): DelayEmbedder,
    str(DiffMap()):       DiffMap,
    str(DiffMapVB()):     DiffMapVB,
    str(Identity()):      Identity,
    str(Isomap()):        Isomap,
    str(Lift()):          Lift,
    str(Scaler()):        Scaler,
    str(SVD()):           SVD,
}

def make_transform(config: List[Dict[str, Any]]) -> Transform:
    """
    Create a transform object based on the provided configuration.

    Args:
        config (List[Dict[str, Any]]): List of dictionaries containing transform configurations.

    Returns:
        Transform: An instance of a Transform class.
    """
    if config is None or len(config) == 0:
        return Identity()

    if isinstance(config, dict):
        config = [config]

    transforms = []
    for t in config:
        trn_type = t.get("type", "").lower()
        if trn_type not in TRN_MAP:
            raise ValueError(f"Unknown transform type: {trn_type}")
        tmp = dict(t)
        tmp.pop("type", None)
        transforms.append(TRN_MAP[trn_type](**tmp))
    return Compose(transforms)
