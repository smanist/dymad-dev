import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import jax
import jax.numpy as jn

# ----------------------
# 2D nonlinear system
# Continuous time with input
# ----------------------

def dyn_cnt_inp(t, x, u=None, ml=None):
    MU, LM = ml
    _u = u(t)
    _d1 = MU*x[0] + _u[0] + x[0]*_u[2]
    _d2 = LM*(x[1]-x[0]**2) + _u[1]
    return np.array([_d1, _d2])

def solIVP_ci(ml, ts, x0, u=None):
    if u is None:
        inp = 3*(np.random.rand(3) - 0.5)
    else:
        inp = u
    finp = lambda t: (np.ones_like(t).reshape(-1,1)*inp).squeeze()
    sol = spi.solve_ivp(dyn_cnt_inp, [0, ts[-1]], x0, t_eval=ts, args=(finp,ml), method='DOP853')
    tmp = np.hstack([sol.y.T, finp(ts)])
    return tmp


# ----------------------
# Double Pendulum
# ----------------------
G  = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
def dyn_dp(t, state, u=None):
    _u = u(t)
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    sin_ = np.sin(del_)
    cos_ = np.cos(del_)
    den1 = (M1 + M2)*L1 - M2*L1*cos_*cos_
    dydx[1] = (M2*L1*state[1]*state[1]*sin_*cos_ + M2*G*np.sin(state[2])*cos_ +
               M2*L2*state[3]*state[3]*sin_ - (M1 + M2)*G*np.sin(state[0]) + _u[0])/den1 - state[1]

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin_*cos_ +
               (M1 + M2)*G*np.sin(state[0])*cos_ -
               (M1 + M2)*L1*state[1]*state[1]*sin_ -
               (M1 + M2)*G*np.sin(state[2]) + _u[1])/den2 - state[3]

    return dydx

def solIVP_dp(ts, x0, u=None):
    if u is None:
        inp = 0.5*(np.random.rand(2) - 0.5)
    else:
        inp = u
    finp = lambda t: (np.ones_like(t).reshape(-1,1)*inp).squeeze()
    sol = spi.solve_ivp(dyn_dp, [0, ts[-1]], x0, t_eval=ts, args=(finp,), method='RK45')
    tmp = np.hstack([sol.y.T, finp(ts)])
    return tmp
