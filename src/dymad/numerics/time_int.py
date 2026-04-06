def fe_step(f, t, x, dt, **kwargs):
    r"""
    Perform a single Forward Euler integration step.

    Args:
        f: Function that computes the derivative dx/dt = f(t, x, \*\*kwargs).
        t: Current time.
        x: Current state vector.
        dt: Time step for the integration.
        **kwargs: Additional arguments to pass to the function f.
    """
    return x + dt * f(t, x, **kwargs)


def rk4_step(f, t, x, dt, **kwargs):
    r"""
    Perform a single Runge-Kutta 4th order (RK4) integration step.

    Args:
        f: Function that computes the derivative dx/dt = f(t, x, \*\*kwargs).
        t: Current time.
        x: Current state vector.
        dt: Time step for the integration.
        **kwargs: Additional arguments to pass to the function f.
    """
    k1 = f(t, x, **kwargs)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, **kwargs)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, **kwargs)
    k4 = f(t + dt, x + dt * k3, **kwargs)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
