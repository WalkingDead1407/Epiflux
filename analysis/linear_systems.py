"""Linear algebra: Gaussian elimination, least squares, SIR Jacobian."""
import numpy as np


def solve_linear_system(A, b):
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("A must be n x n and b must have length n")
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(A[col:, col])))
        if np.isclose(A[pivot, col], 0.0):
            raise ValueError("matrix is singular")
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            b[[col, pivot]] = b[[pivot, col]]
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row, col:] -= factor * A[col, col:]
            b[row] -= factor * b[col]
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (b[row] - A[row, row + 1:] @ x[row + 1:]) / A[row, row]
    return x


def least_squares(X, y):
    """Solve min ||Xw - y|| via the normal equations (X^T X) w = X^T y."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.shape[0] != y.shape[0] or X.shape[0] < X.shape[1]:
        raise ValueError("need at least as many rows as columns, matching y")
    return solve_linear_system(X.T @ X, X.T @ y)


def sir_jacobian(S, I, beta, gamma, N):
    """Jacobian of the SIR right-hand side at the state (S, I, R)."""
    a = beta / N
    return np.array([
        [-a * I, -a * S, 0.0],
        [a * I, a * S - gamma, 0.0],
        [0.0, gamma, 0.0],
    ])


def disease_free_eigenvalues(beta, gamma, N=1.0):
    """Eigenvalues at S=N, I=0. One equals beta - gamma: the epidemic grows iff > 0."""
    return np.sort(np.linalg.eigvals(sir_jacobian(N, 0.0, beta, gamma, N)).real)
