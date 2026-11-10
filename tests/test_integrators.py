import numpy as np
import pytest

from integrators.euler import euler
from integrators.rk import rk2, rk4

def decay(t, y):
    return -y  # exact solution: y0 * exp(-t)

def final_error(method, h):
    t, y = method(decay, [1.0], 0.0, 1.0, h)
    return abs(y[-1, 0] - np.exp(-1.0))

@pytest.mark.parametrize("method", [euler, rk2, rk4])
def test_hits_end_time_and_shape(method):
    t, y = method(decay, [1.0], 0.0, 1.0, 0.1)
    assert t[0] == 0.0 and t[-1] == pytest.approx(1.0)
    assert y.shape == (t.size, 1)

@pytest.mark.parametrize("method,order", [(euler, 1), (rk2, 2), (rk4, 4)])
def test_convergence_order(method, order):
    ratio = final_error(method, 0.1) / final_error(method, 0.05)
    assert ratio == pytest.approx(2 ** order, rel=0.15)

def test_rk4_more_accurate_than_euler():
    assert final_error(rk4, 0.1) < final_error(euler, 0.1) * 1e-3

@pytest.mark.parametrize("bad", [dict(h=0), dict(h=-1), dict(t1=0.0), dict(h=float("nan"))])
def test_invalid_inputs(bad):
    args = dict(y0=[1.0], t0=0.0, t1=1.0, h=0.1)
    args.update(bad)
    with pytest.raises(ValueError):
        euler(decay, **args)
