import numpy as np
from scipy.integrate import solve_ivp
from models.sir import sir_rhs

def peak_infection(t, I):
    t, I = np.asarray(t, float), np.asarray(I, float)
    k = int(np.argmax(I))
    if k == 0 or k == I.size - 1:
        return float(t[k]), float(I[k])
    h = t[1] - t[0]
    y0, y1, y2 = I[k - 1], I[k], I[k + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(t[k]), float(y1)
    shift = 0.5 * (y0 - y2) / denom
    return float(t[k] + shift * h), float(y1 - 0.25 * (y0 - y2) * shift)


def attack_rate(S, N):
    return float((N - np.asarray(S)[-1]) / N)


def doubling_time(growth):
    if growth <= 0:
        raise ValueError("growth rate must be > 0")
    return float(np.log(2) / growth)


def max_abs_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def conservation_error(y, N):
    return float(np.max(np.abs(np.asarray(y).sum(axis=1) - N)))


def reference_solution(params, y0, t):
    sol = solve_ivp(lambda tt, yy: sir_rhs(tt, yy, params), (t[0], t[-1]), y0,
                    method="DOP853", t_eval=t, rtol=1e-11, atol=1e-11)
    return sol.y.T
