import numpy as np

def make_time_grid(t0: float, t1: float, h: float) -> np.ndarray:
    if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(h)):
        raise ValueError("t0, t1 and h must be finite")
    if h <= 0:
        raise ValueError("h must be > 0")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    n_steps = max(1, int(round((t1 - t0) / h)))
    return np.linspace(t0, t1, n_steps + 1)


def prepare(y0, t0, t1, h):
    t = make_time_grid(t0, t1, h)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    y = np.empty((t.size, y0.size))
    y[0] = y0
    return t, y
