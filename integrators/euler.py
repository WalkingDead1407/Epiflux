import numpy as np
from integrators.base import prepare


def euler(f, y0, t0, t1, h):
    t, y = prepare(y0, t0, t1, h)
    for k in range(t.size - 1):
        step = t[k + 1] - t[k]
        y[k + 1] = y[k] + step * np.asarray(f(t[k], y[k]))
    return t, y
