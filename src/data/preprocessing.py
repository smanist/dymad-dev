import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass
class DatasetMetadata:
    """
    Holds metadata about the dataset.
    
    Attributes:
        name (Optional[str]): Optional name of the dataset.
        dt (Optional[float]): The time step (delta time) for the dataset.
        # TODO: add more fields as needed.
    """
    name: Optional[str] = None


class Scaler:
    """
    A class for handling data normalization and scaling.
    
    This class computes scaling parameters based on the provided dataset and applies
    scaling transformations to the data.
    """

    def __init__(self, mode: str = "01", metadata: dict = {}, prev: Optional[str] = None):
        """
        Initialize the Scaler with a specified scaling mode.

        Args:
            mode (str): Scaling mode ('01', '-11', 'std', or 'none').
            prev (str): Path prefix to load previously computed scaling parameters.
            metadata (DatasetMetadata): Optional dataset metadata.
        """
        self._mode = mode.lower()
        self.metadata = metadata 

        if prev:
            self._off, self._scl = np.load(f"{prev}_scl.npy", allow_pickle=True)
        else:
            self._off = None
            self._scl = None

    def fit(self, X: np.ndarray) -> None:
        """
        Compute scaling parameters based on the provided data.
        
        Args:
            X (numpy.ndarray): Input data with one of the following shapes:
                - (num_sequences, seq_length, features)
                - (seq_length, features)
                - (seq_length,)
        """
        features = X.shape[-1]
        if X.ndim < 1 or X.ndim > 3:
            raise ValueError("Input data must be 1D, 2D, or 3D.")
    
        if X.ndim == 1:     # Convert 1D array (seq_length,) to (seq_length, 1)
            X_reshaped = X.reshape(-1, 1)
        else:               # For 2D or 3D arrays, reshape to (-1, features)
            X_reshaped = X.reshape(-1, X.shape[-1])

        if self._mode == "01":
            self._off = np.min(X_reshaped, axis=0)
            self._scl = np.max(X_reshaped, axis=0) - self._off
        elif self._mode == "-11":
            self._off = np.zeros(features)
            self._scl = np.max(np.abs(X_reshaped), axis=0)
        elif self._mode == "std":
            self._off = np.mean(X_reshaped, axis=0)
            self._scl = np.std(X_reshaped, axis=0)
        elif self._mode == "none":
            self._off = np.zeros(features)
            self._scl = np.ones(features)
        else:
            raise ValueError(f"Unknown scaling mode: {self._mode}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scaling to the data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Scaled data with the same shape as input.
        """
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return (X - self._off) / self._scl

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Revert scaled data back to the original scale.

        Args:
            X (numpy.ndarray): Scaled data.

        Returns:
            numpy.ndarray: Data in the original scale with the same shape as input.
        """
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return X * self._scl + self._off

    def save(self, pref: str) -> None:
        """
        Save the scaling parameters to a file.

        Args:
            pref (str): File prefix for saving scaling parameters.
        """
        np.save(f"{pref}_scl.npy", [self._off, self._scl], allow_pickle=False)

    def load(self, pref: str) -> None:
        """
        Load scaling parameters from a file.

        Args:
            pref (str): File prefix for loading scaling parameters.
        """
        self._off, self._scl = np.load(f"{pref}_scl.npy", allow_pickle=True)


class DelayEmbedder:
    """
    A class to perform delay embedding on sequences of data.
    
    For each individual sequence of shape (seq_length, features), this class creates
    delay-embedded sub-sequences by stacking the current time step with the next
    'delay' time steps.
    
    For example, if a sequence has shape (seq_length=100, features=5) and delay=2, then:
      - Each new row in the output will be:
            [ X[t], X[t+1], X[t+2] ]
      - The output will have shape: (seq_length - delay, features * (delay + 1)).
      
    When applied to a batch of sequences with shape (num_sequences, seq_length, features),
    the output shape will be:
        (num_sequences, seq_length - delay, features * (delay + 1)).
    """

    def __init__(self, delay: int = 1):
        """
        Initialize the DelayEmbedder.

        Args:
            delay (int): Number of subsequent time steps to include in the embedding.
        """
        self.delay = delay

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply delay embedding to the input data.

        Args:
            X (np.ndarray): Input array of shape (num_sequences, seq_length, features).
            
        Returns:
            np.ndarray: Delay-embedded array of shape 
                           (num_sequences, seq_length - delay, features * (delay + 1)).
        """
        if X.ndim != 3:
            raise ValueError("Input must be a 3D array of shape (num_sequences, seq_length, features).")

        num_sequences, seq_length, _ = X.shape
        if seq_length <= self.delay:
            raise ValueError(
                f"Sequence length ({seq_length}) must be greater than delay ({self.delay})."
            )

        # Process each sequence individually.
        delayed_sequences = [self._delay(single_seq) for single_seq in X]
        
        return np.array(delayed_sequences)