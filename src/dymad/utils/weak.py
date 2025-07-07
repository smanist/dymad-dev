import numpy as np
from scipy import special as spc

# ------------------------------------------------------------------------------------
# Jacobi Polynomial Utilities for alpha=1, beta=1
# ------------------------------------------------------------------------------------

def jacobi_polynomial(order: int, coords: np.ndarray) -> np.ndarray:
    """
    Evaluate the Jacobi polynomials of order `order` (from 0 to order-1) with parameters
    alpha = 1, beta = 1 at the given coordinates.

    Args:
        order (int):
            The highest polynomial order to evaluate (exclusive).
            We compute polynomials 0, 1, ..., order-1.
        coords (np.ndarray): 1D array of points at which to evaluate.

    Returns:
        np.ndarray:
            A 2D array of shape (order, len(coords)) where each row is the
            evaluated polynomial of a specific order.
    """
    polynomials = np.zeros((order, len(coords)))
    for i in range(order):
        # eval_jacobi(n, alpha, beta, x)
        polynomials[i] = spc.eval_jacobi(i, 1, 1, coords)
    return polynomials

def jacobi_polynomial_derivative(order: int, coords: np.ndarray) -> np.ndarray:
    """
    Evaluate the first derivative of the Jacobi polynomials with alpha=1, beta=1
    for orders 1 to `order-1`, at the given coordinates.

    Note:
        The returned array has shape (order, len(coords)) for consistency,
        but the 0-th row remains all zeros (since derivative is not computed for n=0).

    Args:
        order (int): The highest polynomial order to evaluate (exclusive).
        coords (np.ndarray): 1D array of points at which to evaluate.

    Returns:
        np.ndarray:
            A 2D array of shape (order, len(coords)) where row i holds the
            derivative of Jacobi polynomial i at those coordinates.
            Row 0 is zeros by definition here.
    """
    derivatives = np.zeros((order, len(coords)))
    for i in range(1, order):
        # The derivative for Jacobi(n, 1, 1) can be related to Jacobi(n-1, 2, 2) up to a factor
        derivatives[i] = ((i + 3) / 2.0) * spc.eval_jacobi(i - 1, 2, 2, coords)
    return derivatives

# ------------------------------------------------------------------------------------
# Newton-Cotes Weights (for simple numeric integration)
# ------------------------------------------------------------------------------------

_NEWTON_COTES_COEFS = {
    '1': [0.5, 0.5],                   # Trapezoid rule
    '2': [1.0/3.0, 4.0/3.0, 1.0/3.0],   # Simpson's 1/3 rule
    '3': [3.0/8.0, 9.0/8.0, 9.0/8.0, 3.0/8.0],  # Simpson's 3/8 rule
    '4': [14.0/45.0, 64.0/45.0, 24.0/45.0, 64.0/45.0, 14.0/45.0],  # Boole's rule
}

def compute_newton_cotes_weights(num_points: int, dx: float, order: int) -> np.ndarray:
    """
    Compute Newton-Cotes integration weights for a set of `num_points` samples.

    This function assumes that (num_points - 1) is divisible by `order`.
    For example, for Simpson's 1/3 rule (order=2), we need (num_points - 1) to be even.

    Args:
        num_points (int): Total number of points (N).
        dx (float): The spacing between consecutive sample points.
        order (int): Newton-Cotes rule to use (1, 2, 3, or 4).

    Returns:
        np.ndarray: A 1D array of length `num_points` containing the integration weights.
    """
    if (num_points - 1) % order != 0:
        raise ValueError(
            f"(num_points - 1) = {(num_points - 1)} must be divisible by `order`={order}."
        )
    coefs = _NEWTON_COTES_COEFS[str(order)]
    weights = np.zeros(num_points)
    num_segments = (num_points - 1) // order

    # Add the rule coefficients in segments
    for i in range(num_segments):
        # i*order : (i+1)*order+1 -> covers exactly 'order + 1' points
        weights[i*order : (i+1)*order+1] += coefs

    return weights * dx

# ------------------------------------------------------------------------------------
# Weak Formulation Weight Generation
# ------------------------------------------------------------------------------------
'''
TODO this implementation needs to be updated to work with trajectories with different length (a list of n_steps)
'''
def generate_weak_weights(
    dt: float,
    n_steps: int,
    n_integration_points: int,
    integration_stride: int,
    poly_order: int = 4,
    int_rule_order: int = 4,
):
    r"""
    Generate weights for a weak formulation approach, returning arrays (C, D) and
    the number of "windows" K along the time dimension.

    Steps:
        1. Build a time array of size num_time_points given dt and n_steps
        2. Compute a length scale:

        .. math::
            L = \frac{t_{N-1} - t_0}{2}

        3. Generate Jacobi polynomial basis (P0) and its derivative (P1), each
           with alpha=1, beta=1, on a grid of size n_integration_points in [-1,1].
        4. Construct weighting arrays for the integrals:

        .. math::
            w_0 = 1 - h^2,\quad w_1 = -\frac{2h}{L}

        5. Compute Newton-Cotes integration weights on that grid.
        6. Combine everything to form:

        .. math::
            C = -(P_1 w_0 + P_0 w_1) w,\quad D = P_0 w_0 w

        7. K = number of segments in time = (len(ts) - (N - dN)) // dN

    Args:
        dt (float): Time step size.
        n_steps (int): Number of time points (Nt).
        n_integration_points (int): Number of integration points (N) in [-1, 1].
        integration_stride (int): Sub-sampling or stride in the time domain (dN).
        poly_order (int): Order of the Jacobi polynomial basis.
        int_rule_order (int): Order of Newton-Cotes integration rule (1..4).

    Returns:
        Tuple[np.ndarray, np.ndarray, int]:
            Weights C and D, and number of intervals K:

            - C: shape (poly_order, n_integration_points),
            - D: shape (poly_order, n_integration_points),
            - K: number of intervals in time after sub-sampling.
    """
    # Build time array
    time_array = np.arange(0, n_steps) * dt

    # L = half the (useful) interval length in the time domain
    L = (time_array[n_integration_points - 1] - time_array[0]) / 2.0

    # Grid in [-1, 1] for the polynomials
    h = np.linspace(-1.0, 1.0, n_integration_points)

    # Evaluate Jacobi polynomials
    P0 = jacobi_polynomial(poly_order, h)  # (poly_order, N)
    P1 = jacobi_polynomial_derivative(poly_order, h) / L

    # Build weighting factors
    w0 = 1.0 - h**2
    w1 = -2.0 * h / L

    # Newton-Cotes weights on that grid
    w = compute_newton_cotes_weights(n_integration_points, dt, int_rule_order)

    # Construct final C and D terms
    # We are broadcasting here:
    #  (P1 * w0 + P0 * w1) is shape (poly_order, N),
    #  multiplying by w (shape (N,)) must happen along axis=1.
    C = -(P1 * w0 + P0 * w1) * w  # shape: (poly_order, N)
    D = P0 * w0 * w               # shape: (poly_order, N)

    # Number of "windows" or segments
    K = (len(time_array) - (n_integration_points - integration_stride)) // integration_stride

    return C, D, K