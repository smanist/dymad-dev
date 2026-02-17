from abc import ABC, abstractmethod
import logging
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union

from dymad.numerics.gradients import complex_step, torch_jacobian
from dymad.numerics.linalg import truncated_svd
from dymad.transform.lift import poly_cross, poly_inverse, mixed_cross, mixed_inverse

Array = List[np.ndarray]

logger = logging.getLogger(__name__)

class Transform(ABC):
    """
    Transform always assumes the input is list-like, where each element is a numpy array
    of shape (n_samples, n_features).
    """
    def __init__(self, **kwargs):  # Optional
        self.delay = 0  # Default delay is 0 for non-delay transforms.
        self._inp_dim = None
        self._out_dim = None

    def fit(self, data: Array) -> None:  # Optional
        """
        Determine parameters of the transform based on the data.

        Args:
            data (List[np.ndarray]): array-like or list of array-like, shape (n_samples, n_input_features)
                Training data. If training data contains multiple trajectories, data should be
                a list containing data for each trajectory. Individual trajectories may contain
                different numbers of samples.
        """
        pass

    @abstractmethod
    def transform(self, data: Array) -> Array:
        """
        Apply the transform to the data.

        The shape of data is maintained as much as possible.

        Args:
            data (List[np.ndarray]): List of array-like objects, each of shape (n_samples, n_input_features).

        Returns:
            List[np.ndarray]: transformed data.
        """
        raise NotImplementedError("Transform must implement the transform method.")

    def inverse_transform(self, data: Array) -> Array:
        """
        Apply the inverse transform to the data.

        The shape of data is maintained as much as possible.

        Args:
            data (List[np.ndarray]): List of array-like objects, each of shape (n_samples, n_input_features).

        Returns:
            List[np.ndarray]: inversely transformed data.
        """
        raise NotImplementedError("Transform must implement the inverse_transform method.")

    def get_forward_modes(self, ref: Union[np.ndarray, None] = None, **kwargs) -> Array:
        """
        Get the forward modes of the transform.

        For x in the data space, and its transform z = f(x) in the feature space,
        this function computes the jacobian df/dx.

        For nested transforms, the modes are computed by chain rule.

        Args:
            refs (np.ndarray | None): Reference points to compute the modes.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Forward modes of the transform at references.
        """
        raise NotImplementedError("Transform must implement the get_forward_modes method.")

    def get_backward_modes(self, ref: Union[np.ndarray, None] = None, **kwargs) -> Array:
        """
        Get the backward modes of the transform.

        For z in the feature space, and its inverse transform x = g(z),
        this function computes the jacobian dg/dz (transposed for convenience).

        For nested transforms, the modes are computed by chain rule.

        Args:
            refs (np.ndarray | None): Reference points to compute the modes.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Backward modes of the transform at references, (dg/dz)^T.
        """
        raise NotImplementedError("Transform must implement the get_backward_modes method.")

    def state_dict(self) -> dict[str, Any]:
        """Return a dictionary containing the state of the transform.
        This is used for saving the transform parameters and reloading later.
        """
        return {}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Load the state of the transform from a dictionary."""
        pass

class AddOne(Transform):
    """A class that adds one to the data."""
    def __str__(self):
        return "add_one"

    def fit(self, X: Array) -> None:
        """"""
        self._inp_dim = X[0].shape[-1]
        self._out_dim = self._inp_dim + 1

    def transform(self, X: Array) -> Array:
        """"""
        res = []
        for x in X:
            shape = x.shape[:-1] + (1,)
            res.append(np.concatenate([x, np.ones(shape)], axis=-1))
        return res

    def inverse_transform(self, X: Array) -> Array:
        """"""
        return [x[..., :-1] for x in X]

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.eye(self._out_dim, self._inp_dim)

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.eye(self._out_dim, self._inp_dim)

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "inp": self._inp_dim,
            "out": self._out_dim,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"AddOne: Loading parameters from checkpoint :{d}")
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]

class Autoencoder(Transform):
    """
    A class for data reduction by autoencoder.

    Not meant for fitting; only used in post-processing to load a pre-trained model.

    Args:
        model (torch.nn.Module): The autoencoder model.
        order (int): Truncation order.
        device (str): Device to run the model on.
    """

    def __init__(self, model, encoder, decoder):
        self.encoder  = encoder
        self.decoder  = decoder
        self._inp_dim = model.n_total_state_features
        self._out_dim = model.latent_dimension
        self.dtype    = model.dtype
        self.device   = model.device

    def __str__(self):
        return "autoencoder"

    def fit(self, X: Array) -> None:
        """"""
        raise NotImplementedError("Autoencoder does not implement fit. Load a pre-trained model instead.")

    def transform(self, X: Array) -> Array:
        """"""
        with torch.no_grad():
            _X = torch.tensor(np.array(X), dtype=self.dtype).to(self.device)
            _Z = self.encoder(_X).cpu().detach().numpy()
        return _Z

    def inverse_transform(self, X: Array) -> Array:
        """"""
        with torch.no_grad():
            _X = torch.tensor(np.array(X), dtype=self.dtype).to(self.device)
            _Z = self.decoder(_X).cpu().detach().numpy()
        return _Z

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        assert ref is not None, "Autoencoder requires a reference point to compute modes."
        return torch_jacobian(lambda x: self.encoder(x), ref, dtype=self.dtype)

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        assert ref is not None, "Autoencoder requires a reference point to compute modes."
        return torch_jacobian(lambda x: self.decoder(x), ref, dtype=self.dtype).T

    def state_dict(self) -> dict[str, Any]:
        """"""
        raise NotImplementedError("Autoencoder does not implement state_dict.")

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """"""
        raise NotImplementedError("Autoencoder does not implement load_state_dict.")

class Identity(Transform):
    """A class that performs no transformation on the data."""
    def __str__(self):
        return "identity"

    def fit(self, X: Array) -> None:
        """"""
        self._inp_dim = X[0].shape[-1]
        self._out_dim = self._inp_dim

    def transform(self, X: Array) -> Array:
        """"""
        return X

    def inverse_transform(self, X: Array) -> Array:
        """"""
        return X

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.eye(self._out_dim, self._inp_dim)

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.eye(self._out_dim, self._inp_dim)

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "inp": self._inp_dim,
            "out": self._out_dim,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"Identity: Loading parameters from checkpoint :{d}")
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]

class Scaler(Transform):
    """
    A class for handling data normalization and scaling.

    This class computes scaling parameters based on the provided dataset and applies
    scaling transformations to the data.

    Args:
        mode (str): Scaling mode ('01', '-11', 'std', or 'none').
        scl (Optional[float]): Scaling factor for inference datasets (if provided).
        off (Optional[float]): Offset value for inference datasets (if provided).
    """

    def __init__(self, mode: str = "01", scl: Optional[float] = None, off: Optional[float] = None):
        self._mode = mode.lower()
        self._off = off
        self._scl = scl

    def __str__(self):
        return "scaler"

    def fit(self, X: Array) -> None:
        """"""
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

        self._inp_dim = len(self._scl)
        self._out_dim = self._inp_dim

    def transform(self, X: Array) -> Array:
        """"""
        logger.info(f"Scaler: Applying scaling with offset={self._off}, scale={self._scl}.")
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return [(trajectory - self._off) / self._scl for trajectory in X]

    def inverse_transform(self, X: Array) -> Array:
        """"""
        logger.info(f"Scaler: Applying un-scaling with offset={self._off}, scale={self._scl}.")
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return [trajectory * self._scl + self._off for trajectory in X]

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.diag(1.0 / self._scl)

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.diag(self._scl)

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "mode": self._mode,
            "off":  self._off,
            "scl":  self._scl,
            "inp": self._inp_dim,
            "out": self._out_dim,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"Scaler: Loading parameters from checkpoint :{d}")
        self._mode = d["mode"].lower()
        self._off  = d["off"]
        self._scl  = d["scl"]
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]

class Lift(Transform):
    """
    Lifting into higher-dimensional space by hand-crafted features
    """
    def __init__(self, fobs: Union[str, Callable] = None, finv: Union[str, Callable, None] = None, **kwargs):
        if fobs == 'poly':
            self._fobs = poly_cross
            self._finv = poly_inverse
        elif fobs == 'mixed':
            self._fobs = mixed_cross
            self._finv = mixed_inverse
        elif callable(fobs):
            self._fobs = fobs
            if callable(finv):
                self._finv = finv
            else:
                self._finv = self._pseudo_inv
        else:
            self._fobs = None
            self._finv = None
        self._fargs = kwargs

    def __str__(self):
        return "lift"

    def fit(self, X: Array) -> None:
        """"""
        self._inp_dim = X[0].shape[-1]

        _Z = self._fobs(X[0], **self._fargs)  # To check if fobs works
        self._out_dim = _Z.shape[-1]

        if self._finv == self._pseudo_inv:
            _Zs = [self._fobs(_X, **self._fargs) for _X in X]
            self._C = np.linalg.lstsq(np.vstack(_Zs), np.vstack(X), rcond=None)[0]
        else:
            self._C = None

    def transform(self, X: Array) -> Array:
        tmp = []
        for _X in X:
            _x_shape = _X.shape[:-1]
            _Z = self._fobs(_X.reshape(-1,self._inp_dim), **self._fargs)
            tmp.append(_Z.reshape(*_x_shape, -1))
        return tmp

    def inverse_transform(self, X: Array) -> Array:
        tmp = []
        for _X in X:
            _x_shape = _X.shape[:-1]
            _Z = self._finv(_X.reshape(-1,self._out_dim), **self._fargs)
            tmp.append(_Z.reshape(*_x_shape, -1))
        return tmp

    def _pseudo_inv(self, Z):
        return Z.dot(self._C)

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        assert ref is not None, "Lift requires a reference point to compute modes."
        func = lambda x: self._fobs(x.reshape(1,-1), **self._fargs).reshape(-1)
        return complex_step(func, ref)

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        if self._C is None:
            assert ref is not None, "Lift with finv requires a reference point to compute modes."
            func = lambda x: self._finv(x.reshape(1,-1), **self._fargs).reshape(-1)
            return complex_step(func, ref).T
        else:
            return self._C

    def state_dict(self) -> Dict[str, Any]:
        """"""
        return {
            "inp": self._inp_dim,
            "out": self._out_dim,
            "C": self._C,
            "fobs": self._fobs,
            "finv": self._finv,
            "fargs": self._fargs,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """"""
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]
        self._C = d["C"]
        self.__init__(d["fobs"], d["finv"], **d["fargs"])

class DelayEmbedder(Transform):
    """
    A class to perform delay embedding on sequences of data.

    For each individual sequence of shape (seq_length, features), this class creates
    delay-embedded sub-sequences by stacking the current time step with the next
    'delay' time steps.

    For example, if a sequence has shape (seq_length=100, features=5) and delay=2, then:
      - Each new row in the output will be:

            [ X[t], X[t+1], X[t+2] ]

      - The output will have shape:
      
            (seq_length - delay, features * (delay + 1)).

    When applied to a batch of sequences with shape (num_sequences, seq_length, features),
    the output shape will be:

        (num_sequences, seq_length - delay, features * (delay + 1)).

    Args:
        delay (int): Number of subsequent time steps to include in the embedding.
    """

    def __init__(self, delay: int = 1):
        self.delay = delay

    def __str__(self):
        return "delay"

    def fit(self, X: Array) -> None:
        """"""
        self._inp_dim = X[0].shape[-1]
        self._out_dim = self._inp_dim * (self.delay + 1)

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
            sequence[:, :self._inp_dim],
            sequence[-1, self._inp_dim:].reshape(self.delay, self._inp_dim)]
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
        logger.info(f"DelayEmbedder: Applying delay embedding with delay={self.delay}.")
        delayed_sequences = []
        for sequence in X:
            delayed_sequences.append(self._delay(sequence))
        return delayed_sequences

    def inverse_transform(self, X: Array) -> Array:
        """"""
        logger.info(f"DelayEmbedder: Unrolling the data.")
        unrolled_sequences = []
        for sequence in X:
            unrolled_sequences.append(self._unroll(sequence))
        return unrolled_sequences

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.eye(self._out_dim, self._inp_dim)

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        return np.eye(self._out_dim, self._inp_dim)

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "delay": self.delay,
            "inp":   self._inp_dim,
            "out":   self._out_dim,
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"DelayEmbedder: Loading parameters from checkpoint :{d}")
        self.delay    = d["delay"]
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]

class SVD(Transform):
    """
    A class for data reduction by SVD.

    Args:
        order (int | float): Truncation order.
        ifcen (bool): If center the data.
    """

    def __init__(self, order: Union[int, float] = 1.0, ifcen: bool = False):
        self._order = order
        self._ifcen = ifcen

    def __str__(self):
        return "svd"

    def fit(self, X: Array) -> None:
        """"""
        X_combined    = np.vstack(X)
        self._inp_dim = X_combined.shape[-1]

        if self._ifcen:
            self._off = np.mean(X_combined, axis=0)
            X_combined -= self._off
        else:
            self._off = np.zeros(self._inp_dim,)
        _, _, _V = truncated_svd(X_combined, self._order)
        self._P = _V.conj()

        self._out_dim = self._P.shape[1]
        logger.info(f"SVD: Fitted SVD with {self._out_dim} components.")

    def transform(self, X: Array) -> Array:
        """"""
        logger.info(f"SVD: Applying SVD with order={self._order}, center={self._ifcen}.")
        if self._P is None:
            raise ValueError("SVD parameters are not initialized. Call `fit` first.")

        return [(trajectory-self._off).dot(self._P) for trajectory in X]

    def inverse_transform(self, X: Array) -> Array:
        """"""
        logger.info(f"SVD: Applying projection with order={self._order}, center={self._ifcen}.")
        if self._P is None:
            raise ValueError("SVD parameters are not initialized. Call `fit` first.")

        return [trajectory.dot(self._P.conj().T) + self._off for trajectory in X]

    def get_forward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        if self._P is None:
            raise ValueError("SVD parameters are not initialized. Call `fit` first.")
        return self._P.T

    def get_backward_modes(self, ref=None, **kwargs) -> np.ndarray:
        """"""
        if self._P is None:
            raise ValueError("SVD parameters are not initialized. Call `fit` first.")
        return self._P.conj().T

    def state_dict(self) -> dict[str, Any]:
        """"""
        return {
            "order":  self._order,
            "ifcen":  self._ifcen,
            "inp":    self._inp_dim,
            "out":    self._out_dim,
            "P":      self._P,
            "off":    self._off
            }

    def load_state_dict(self, d) -> None:
        """"""
        logger.info(f"SVD: Loading parameters from checkpoint :{d}")
        self._order = d["order"]
        self._ifcen = d["ifcen"]
        self._inp_dim = d["inp"]
        self._out_dim = d["out"]
        self._P     = d["P"]
        self._off   = d["off"]
