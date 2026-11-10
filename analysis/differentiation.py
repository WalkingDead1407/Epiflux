import numpy as np

def _check(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError("t and y must be 1-D arrays of the same length")
    if t.size < 3:
        raise ValueError("need at least 3 points")
    steps = np.diff(t)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0]):
        raise ValueError("t must be strictly increasing and uniformly spaced")
    return t, y, steps[0]

def forward_difference(t, y):
    t, y, h = _check(t, y)    #y[i+0]-y[i])/h for every point except the last
    return (y[1:] - y[:-1]) / h

def central_difference(t, y):
    t, y, h = _check(t, y)   #(y[i+1]-y[i-1])/(2h) for interior points (length n-2). Second order
    return (y[2:] - y[:-2]) / (2 * h)

def derivative(t, y):       #derivative at every point: central inside, one-sided at the two ends
    t, y, h = _check(t, y)
    d = np.empty_like(y)
    d[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    d[0] = (y[1] - y[0]) / h
    d[-1] = (y[-1] - y[-2]) / h
    return d


def growth_rate(t, I):
    I = np.asarray(I, float)
    return derivative(t, I) / I
