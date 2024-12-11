import jax
import numpy as np

class Scaler:
    """
    Non-dimensionalization.
    """
    def __init__(self, Xs=None, mode=None, prev=None):
        if prev is not None:
            # Recover from prev. data
            self._off, self._scl = np.load(open(f'{prev}_scl.npy', 'rb'))
        else:
            self._mode = mode.lower()
            X = np.vstack(Xs)
            if self._mode == '01':
                _min = np.min(X, axis=0)
                _max = np.max(X, axis=0)
                self._off = _min
                self._scl = _max-_min
            elif self._mode == '-11':
                self._off = np.zeros_like(X[0])
                self._scl = np.max(np.abs(X), axis=0)
            elif self._mode == 'std':
                self._off = np.mean(X, axis=0)
                self._scl = np.std(X, axis=0)
            elif self._mode == 'none':
                self._off = np.zeros_like(X[0])
                self._scl = np.ones_like(X[0])
            else:
                raise ValueError(f"Unknown mode {self._mode}")

        self.vND = jax.vmap(self.ND)
        self.vDM = jax.vmap(self.DM)

    def ND(self, X):
        return (X-self._off)/self._scl

    def DM(self, Z):
        return Z*self._scl + self._off

    def save(self, pref):
        np.save(open(f'{pref}_scl.npy', 'wb'), [self._off, self._scl], allow_pickle=False)

class DataManager:
    """
    Manager of trajectory data.
    A long trajectory is split into several short trajectories
    as samples for training.
    """
    def __init__(
        self,
        train: np.ndarray,
        valid: np.ndarray,
        horizon: int,
        shift: int,
        num_batch: int):
        """
        Args:
        train: Array of trajectories with states and inputs combined.
               For training.
               Do not need to be of same length.
        valid: Similar to train, but for validation.  Can be None.
        horizon: Length of sample trajectory.
        shift: Non-overlapping length between two consecutive samples.
        num_batch: Number of batches.
        """
        self.Nh = horizon
        self.Ns = shift
        self.Nb = num_batch

        self.data_train = self._gen_smpls(train)
        self.Nd = len(self.data_train)

        if valid is None:
            self.data_valid = None
        else:
            self.data_valid = self._gen_smpls(valid)

        self.shuffle()

    def __getitem__(self, idx):
        """
        Returns the idx'th batch of samples
        """
        return self.data_batch[idx]

    def shuffle(self):
        """
        Shuffle the training samples and organize in batches.
        """
        np.random.shuffle(self.data_train)
        sz = self.Nd // self.Nb
        tmp = self.data_train[:self.Nb*sz]
        self.data_batch = tmp.reshape(self.Nb, sz, self.Nh, -1)

        print(f'    Data shuffled: {self.Nb} batches of size {sz} for {self.Nd} samples')

    def _gen_smpls(self, data):
        """
        Returns samples of trajectory segments.
        """
        tmp = []
        for _i, _d in enumerate(data):
            N = (len(_d) - (self.Nh-self.Ns)) // self.Ns
            j = np.arange(N) * self.Ns
            for _j in j:
                tmp.append(_d[_j:_j+self.Nh])
        return np.array(tmp)