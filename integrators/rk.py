import numpy as np
from integrators.base import prepare

def rk2(f, y0, t0, t1, h):
    t, y = prepare(y0, t0, t1, h)
    for k in range(t.size - 1):
        s = t[k + 1] - t[k]
        k1 = np.asarray(f(t[k], y[k]))
        k2 = np.asarray(f(t[k] + s / 2, y[k] + s / 2 * k1))
        y[k + 1] = y[k] + s * k2
    return t, y


def rk4(f, y0, t0, t1, h):
    t, y = prepare(y0, t0, t1, h)
    for k in range(t.size - 1):
        s = t[k + 1] - t[k]
        k1 = np.asarray(f(t[k], y[k]))
        k2 = np.asarray(f(t[k] + s / 2, y[k] + s / 2 * k1))
        k3 = np.asarray(f(t[k] + s / 2, y[k] + s / 2 * k2))
        k4 = np.asarray(f(t[k] + s, y[k] + s * k3))
        y[k + 1] = y[k] + s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y
