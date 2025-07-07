from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Any, Dict, List, Optional

Array = List[np.ndarray]

logging = logging.getLogger(__name__)

class Transform(ABC):
    """
    Transform always assumes the input is list-like, where each element is a numpy array
    of shape (n_samples, n_features).
    """
    def __init__(self, **kwargs):  # Optional
        pass

    def fit(self, data: Array) -> None:  # Optional
        pass

    @abstractmethod
    def transform(self, data: Array) -> Array:
        raise NotImplementedError("Transform must implement the transform method.")

    @abstractmethod
    def inverse_transform(self, data: Array) -> Array:
        raise NotImplementedError("Transform must implement the transform method.")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        pass

class Compose(Transform):
    """Apply transforms **in order**.  Inverse is applied in reverse."""
    def __init__(self, transforms: List[Transform] = None):
        if transforms is None:
            # Reload from state_dict is expected.
            return

        self.T = transforms
        self._T_names = [str(t) for t in transforms]

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

    def fit(self, data: Array) -> None:
        _d = data
        for t in self.T:
            t.fit(_d)
            _d = t.transform(_d)

    def transform(self, data: Array) -> Array:
        for t in self.T:
            data = t.transform(data)
        return data

    def inverse_transform(self, data: Array) -> Array:
        for t in reversed(self.T):
            data = t.inverse_transform(data)
        return data

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "Compose",
            "names": self._T_names,
            "delay": self.delay,
            "children": [t.state_dict() for t in self.T]}

    def load_state_dict(self, d) -> None:
        logging.info(f"Compose: Loading parameters from checkpoint")
        self._T_names = d["names"]
        self.T = []
        for name, sd in zip(self._T_names, d["children"]):
            if name not in _TRN_MAP:
                raise ValueError(f"Unknown transform type in Compose: {name}")
            self.T.append(_TRN_MAP[name]())
            self.T[-1].load_state_dict(sd)
        self.delay = d["delay"]

class Identity(Transform):
    def __str__(self):
        return "identity"

    def transform(self, X: Array) -> Array:
        return X

    def inverse_transform(self, X: Array) -> Array:
        return X

class Scaler(Transform):
    """
    A class for handling data normalization and scaling.

    This class computes scaling parameters based on the provided dataset and applies
    scaling transformations to the data.
    """

    def __init__(self, mode: str = "01", scl: Optional[float] = None, off: Optional[float] = None):
        """
        Initialize the Scaler with a scaling mode.

        Args:
            mode (str): Scaling mode ('01', '-11', 'std', or 'none').
            scl (Optional[float]): Scaling factor for inference datasets (if provided).
            off (Optional[float]): Offset value for inference datasets (if provided).
        """
        self._mode = mode.lower()
        self._off = off
        self._scl = scl

    def __str__(self):
        return "scaler"

    def fit(self, X: Array) -> None:
        """
        Compute scaling parameters based on the provided data.

        Args:
            X: array-like or list of array-like, shape (n_samples, n_input_features)
               Training data. If training data contains multiple trajectories, X should be
               a list containing data for each trajectory. Individual trajectories may contain
               different numbers of samples.
        """
        # Combine all trajectories along the sample axis.
        X_combined = np.vstack(X)
        features = X_combined.shape[-1]

        if self._mode == "01":
            self._off = np.min(X_combined, axis=0)
            self._scl = np.max(X_combined, axis=0) - self._off
        elif self._mode == "-11":
            self._off = np.zeros(features)
            self._scl = np.max(np.abs(X_combined), axis=0)
        elif self._mode == "std":
            self._off = np.mean(X_combined, axis=0)
            self._scl = np.std(X_combined, axis=0)
        elif self._mode == "none":
            self._off = np.zeros(features)
            self._scl = np.ones(features)
        else:
            raise ValueError(f"Unknown scaling mode: {self._mode}")

        msk = self._scl < 1e-12
        self._scl[msk] = 1.0  # Avoid division by zero

    def transform(self, X: Array) -> Array:
        """
        Apply scaling to a list of trajectories.

        Args:
            X: List of array-like objects, each of shape (n_samples, n_input_features).

        Returns:
            List of numpy.ndarray: Scaled data with the same shapes as the input trajectories.
        """
        logging.info(f"Scaler: Applying scaling with offset={self._off}, scale={self._scl}.")
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return [(trajectory - self._off) / self._scl for trajectory in X]

    def inverse_transform(self, X: Array) -> Array:
        """
        Revert scaled data back to the original scale for a list of trajectories.

        Args:
            X: List of array-like objects representing scaled data.
               Each element should have the same shape as the corresponding original trajectory.

        Returns:
            List of numpy.ndarray: Data in the original scale with the same shapes as the input trajectories.
        """
        logging.info(f"Scaler: Applying un-scaling with offset={self._off}, scale={self._scl}.")
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return [trajectory * self._scl + self._off for trajectory in X]

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "off":  self._off,
            "scl":  self._scl,
            }

    def load_state_dict(self, d) -> None:
        logging.info(f"Scaler: Loading parameters from checkpoint :{d}")
        self._mode = d["mode"].lower()
        self._off  = d["off"]
        self._scl  = d["scl"]

class DelayEmbedder(Transform):
    """
    A class to perform delay embedding on sequences of data.

    For each individual sequence of shape (seq_length, features), this class creates
    delay-embedded sub-sequences by stacking the current time step with the next
    'delay' time steps.

    For example, if a sequence has shape (seq_length=100, features=5) and delay=2, then:
      - Each new row in the output will be::

            [ X[t], X[t+1], X[t+2] ]

      - The output will have shape: (seq_length - delay, features * (delay + 1)).

    When applied to a batch of sequences with shape (num_sequences, seq_length, features),
    the output shape will be::

        (num_sequences, seq_length - delay, features * (delay + 1)).
    """

    def __init__(self, delay: int = 1):
        """
        Initialize the DelayEmbedder.

        Args:
            delay (int): Number of subsequent time steps to include in the embedding.
        """
        self.delay = delay

    def __str__(self):
        return "delay"

    def fit(self, X: Array) -> None:
        self.dim = X[0].shape[-1]

    def _delay(self, sequence: np.ndarray) -> np.ndarray:
        """
        Perform delay embedding on a single sequence.

        Args:
            sequence (np.ndarray): A single sequence of shape (seq_length, features).

        Returns:
            np.ndarray: A delay-embedded sequence of shape
                           (seq_length - delay, features * (delay + 1)).
        """
        seq_length, _ = sequence.shape
        if seq_length <= self.delay:
            raise ValueError(
                f"Sequence length ({seq_length}) must be greater than delay ({self.delay})."
            )

        # Number of valid rows after applying delay embedding.
        M = seq_length - self.delay

        # Create concatenated sub-sequences for each shift in [0 .. delay].
        embedded = np.hstack([
            sequence[j: M + j]  # Each slice has shape (M, features)
            for j in range(self.delay + 1)
        ])
        # The resulting shape is (M, features * (delay + 1)).

        return embedded

    def _unroll(self, sequence: np.ndarray) -> np.ndarray:
        """
        Revert delay embedding on a single sequence.  Input is expected to be

        [x1, x2, ..., x_d]
        [x2, x3, ..., x_(d+1)]
        ...
        [x_(L-d+1), ..., x_L]

        We unroll this to [x1, x2, ..., x_L].

        Args:
            sequence (np.ndarray): A delay-embedded sequence of shape
                                   (seq_length - delay, features * (delay + 1)).

        Returns:
            np.ndarray: The original sequence of shape (seq_length, features).
        """
        arr = [
            sequence[:, :self.dim],
            sequence[-1, self.dim:].reshape(self.delay, self.dim)]
        return np.vstack(arr)

    def transform(self, X: Array) -> Array:
        """
        Apply delay embedding to the input data.

        Args:
            X (list[np.ndarray]): List of input arrays, each of shape (seq_length, features).

        Returns:
            list[np.ndarray]: List of delay-embedded arrays, each of shape
                              (seq_length - delay, features * (delay + 1)).
        """
        logging.info(f"DelayEmbedder: Applying delay embedding with delay={self.delay}.")
        delayed_sequences = []
        for sequence in X:
            delayed_sequences.append(self._delay(sequence))
        return delayed_sequences

    def inverse_transform(self, X: Array) -> Array:
        logging.info(f"DelayEmbedder: Unrolling the data.")
        unrolled_sequences = []
        for sequence in X:
            unrolled_sequences.append(self._unroll(sequence))
        return unrolled_sequences

    def state_dict(self) -> dict[str, Any]:
        return {
            "delay": self.delay,
            "dim":   self.dim,
            }

    def load_state_dict(self, d) -> None:
        logging.info(f"DelayEmbedder: Loading parameters from checkpoint :{d}")
        self.delay = d["delay"]
        self.dim   = d["dim"]

_TRN_MAP = {
    str(Compose()):       Compose,
    str(DelayEmbedder()): DelayEmbedder,
    str(Identity()):      Identity,
    str(Scaler()):        Scaler,
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
        if trn_type not in _TRN_MAP:
            raise ValueError(f"Unknown transform type: {trn_type}")
        tmp = dict(t)
        tmp.pop("type", None)
        transforms.append(_TRN_MAP[trn_type](**tmp))
    return Compose(transforms)
