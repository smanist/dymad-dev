import matplotlib.pyplot as plt
import numpy as np

from dymad.sako.plotting import SpectralPlottingAdapter


class _DummyContext:
    _Nout = 2
    _Ninp = 3

    def encode(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            raise RuntimeError("bulk observable encoding is not supported")
        return np.array([x[0], 5.0])


class _DummyAnalysis:
    def __init__(self):
        self._ctx = _DummyContext()

    def predict(self, x0s, ts, return_obs=False):
        ts = np.asarray(ts, dtype=float)
        x0s = np.asarray(x0s)
        if x0s.ndim == 1:
            obs = np.stack([ts, np.full_like(ts, 5.0)], axis=1)
        else:
            obs = np.stack(
                [np.stack([ts + i, np.full_like(ts, 5.0)], axis=1) for i in range(len(x0s))],
                axis=0,
            )
        if return_obs:
            return obs, obs
        return obs


def test_plot_pred_ifobs_encodes_single_trajectory_stepwise():
    analysis = _DummyAnalysis()
    adapter = SpectralPlottingAdapter(analysis)
    ts = np.array([0.0, 1.0, 2.0])
    ref = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
        ]
    )

    fig, axes = adapter.plot_pred(
        x0s=ref[0],
        ts=ts,
        ref=ref,
        ifobs=True,
        idx="all",
        title="single",
    )

    assert len(axes) == 2
    assert axes[0].get_title() == "single, Error 0.00%"
    assert axes[1].get_title() == "single, Error 0.00%"
    plt.close(fig)


def test_plot_pred_ifobs_encodes_batched_trajectories_stepwise():
    analysis = _DummyAnalysis()
    adapter = SpectralPlottingAdapter(analysis)
    ts = np.array([0.0, 1.0, 2.0])
    x0s = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
        ]
    )
    ref = np.array(
        [
            [
                [0.0, 10.0, 20.0],
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
            ],
            [
                [1.0, 13.0, 23.0],
                [2.0, 14.0, 24.0],
                [3.0, 15.0, 25.0],
            ],
        ]
    )

    fig, axes = adapter.plot_pred(
        x0s=x0s,
        ts=ts,
        ref=ref,
        ifobs=True,
        idx="all",
        title="batch",
    )

    assert len(axes) == 2
    assert axes[0].get_title() == "batch, Error 0.00%"
    assert axes[1].get_title() == "batch, Error 0.00%"
    plt.close(fig)
