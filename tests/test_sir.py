import numpy as np
import pytest

from models.sir import SIRParams, make_initial_state, sir_rhs, simulate
from analysis.metrics import conservation_error

def test_params_validation():
    with pytest.raises(ValueError):
        SIRParams(-0.1, 0.2)
    with pytest.raises(ValueError):
        SIRParams(0.5, 0.0)
    with pytest.raises(ValueError):
        SIRParams(0.5, 0.2, N=0)
    assert SIRParams(0.5, 0.2).r0 == pytest.approx(2.5)

def test_initial_state():
    y0 = make_initial_state(1000, 5, 10)
    assert y0.tolist() == [985, 5, 10]
    with pytest.raises(ValueError):
        make_initial_state(10, 11)

def test_rhs_sums_to_zero():
    p = SIRParams(0.5, 0.2, 1000)
    assert sir_rhs(0, [900, 90, 10], p).sum() == pytest.approx(0.0)

def test_population_conserved_and_non_negative():
    p = SIRParams(0.5, 0.2, 1000)
    res = simulate(p, make_initial_state(1000, 1), 100, 0.1, "rk4")
    assert conservation_error(res.y, 1000) < 1e-8
    assert (res.y > -1e-9).all()

def test_no_epidemic_when_r0_below_one():
    p = SIRParams(0.1, 0.2, 1000)
    res = simulate(p, make_initial_state(1000, 5), 100, 0.1)
    assert res.I.max() == pytest.approx(5.0)  # infections only decline

def test_simulate_rejects_bad_input():
    p = SIRParams(0.5, 0.2, 1000)
    with pytest.raises(ValueError):
        simulate(p, [999, 1, 0], 10, method="magic")
    with pytest.raises(ValueError):
        simulate(p, [500, 1, 0], 10)  # does not sum to N
