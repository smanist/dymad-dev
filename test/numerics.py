import numpy as np

from dymad.src import weight_bdf, weight_nc

# Check BDF
x = np.linspace(0.5, 1, 5)
dx = x[1]-x[0]
y = x*np.sin(x)
dy = x*np.cos(x) + np.sin(x)

for _o in range(1,5):
    w = weight_bdf(_o)
    n = w.dot(y[-_o-1:]) / dx
    e = np.abs((dy[-1]-n)/dy[-1])
    print(_o, e)
