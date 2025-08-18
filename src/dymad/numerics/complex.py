import matplotlib.pyplot as plt
import numpy as np

def disc2cont(z, dt):
    return np.log(np.abs(z))/dt + 1j * np.angle(z)/dt

def complex_grid(grid):
    """
    Args:

    grid: If a real array of 2xN, a meshgrid of complex values is formed.
          If a list of two floats (a,b) and an int (N), a uniform grid is
          created over [-a, a]x[-b, b], each side N points.
          Otherwise use as is.
    """
    if isinstance(grid, list):
        _a, _b, _N = grid
        _as = np.linspace(-_a, _a, _N)
        _bs = np.linspace(-_b, _b, _N)
        _G = _as.reshape(-1,1) + 1j*_bs
    elif isinstance(grid, np.ndarray):
        if len(grid.shape) == 2:
            _G = grid[0].reshape(-1,1) + 1j*grid[1].reshape(-1)
        else:
            _G = np.copy(grid)
    else:
        _G = grid
    return _G.reshape(-1)

def complex_plot(grid, sv, levels, fig=None, mode='line', lwid=2, lsty=None):
    if fig is None:
        f, ax = plt.subplots()
    else:
        f, ax = fig
    _kw = dict(linewidths=lwid, linestyles=lsty)

    if mode == 'line':
        cs = ax.tricontour(grid.real, grid.imag, sv, levels=levels, **_kw)
        ax.clabel(cs, cs.levels, inline=True)
    else:
        cs = ax.tricontourf(grid.real, grid.imag, sv, levels=levels, **_kw)
        plt.colorbar(cs)

    return f, ax

complex_map = {
    'angle' : np.angle,
    'abs'   : np.abs,
    'real'  : np.real,
    'imag'  : np.imag,
    'iden'  : lambda x: x
}