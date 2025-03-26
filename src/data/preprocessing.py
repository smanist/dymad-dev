import numpy as np
from typing import Optional

class Scaler:
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
        self._off = scl
        self._scl = off

    def fit(self, X) -> None:
        """
        Compute scaling parameters based on the provided data.
        
        Args:
            X: array-like or list of array-like, shape (n_samples, n_input_features)
               Training data. If training data contains multiple trajectories, X should be 
               a list containing data for each trajectory. Individual trajectories may contain 
               different numbers of samples.
        """
        
        # Process each trajectory: ensure they are 2D arrays.
        processed = []
        for traj in X:
            traj = np.asarray(traj)
            if traj.ndim == 1:  # convert 1D array to 2D
                traj = traj.reshape(-1, 1)
            elif traj.ndim != 2:
                raise ValueError("Each trajectory must be 1D or 2D array-like.")
            processed.append(traj)

        # Combine all trajectories along the sample axis.
        X_combined = np.vstack(processed)
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

    def transform(self, X) -> list:
        """
        Apply scaling to a list of trajectories.

        Args:
            X: List of array-like objects, each of shape (n_samples, n_input_features).

        Returns:
            List of numpy.ndarray: Scaled data with the same shapes as the input trajectories.
        """
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return [(trajectory - self._off) / self._scl for trajectory in X]

    def inverse_transform(self, X) -> list:
        """
        Revert scaled data back to the original scale for a list of trajectories.

        Args:
            X: List of array-like objects representing scaled data.
               Each element should have the same shape as the corresponding original trajectory.

        Returns:
            List of numpy.ndarray: Data in the original scale with the same shapes as the input trajectories.
        """
        if self._off is None or self._scl is None:
            raise ValueError("Scaler parameters are not initialized. Call `fit` first.")

        return [trajectory * self._scl + self._off for trajectory in X]

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

    def transform(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply delay embedding to the input data.

        Args:
            X (list[np.ndarray]): List of input arrays, each of shape (seq_length, features).

        Returns:
            list[np.ndarray]: List of delay-embedded arrays, each of shape 
                              (seq_length - delay, features * (delay + 1)).
        """
        delayed_sequences = []
        for sequence in X:
            if sequence.ndim != 2:
                raise ValueError("Each sequence must be a 2D array of shape (seq_length, features).")
            delayed_sequences.append(self._delay(sequence))
        
        return delayed_sequences
