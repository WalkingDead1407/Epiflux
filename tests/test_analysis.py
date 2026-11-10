import numpy as np
import pytest

from analysis.differentiation import central_difference, derivative, forward_difference
from analysis.fitting import add_noise, estimate_growth_rate, fit_sir_parameters
from analysis.linear_systems import disease_free_eigenvalues, least_squares, solve_linear_system
from analysis.metrics import attack_rate, doubling_time, peak_infection, reference_solution, max_abs_error
from analysis.root_finding import bisection, final_susceptible_fraction, newton
from models.sir import SIRParams, make_initial_state, simulate, sir_rhs

P = SIRParams(0.5, 0.2, 1000)
Y0 = make_initial_state(1000, 1)

def test_differentiation_of_sine():
    t = np.linspace(0, np.pi, 201)
    assert np.max(np.abs(central_difference(t, np.sin(t)) - np.cos(t[1:-1]))) < 1e-4
    assert np.max(np.abs(forward_difference(t, np.sin(t)) - np.cos(t[:-1]))) < 1e-1
    assert derivative(t, np.sin(t)).shape == t.shape

def test_differentiation_rejects_uneven_grid():
    with pytest.raises(ValueError):
        derivative([0, 1, 3, 4], [0, 1, 2, 3])

def test_numerical_dI_dt_matches_model():
    res = simulate(P, Y0, 60, 0.05)
    numeric = derivative(res.t, res.I)
    exact = np.array([sir_rhs(0, y, P)[1] for y in res.y])
    assert np.max(np.abs(numeric - exact)[1:-1]) < 0.05

def test_peak_is_where_derivative_changes_sign_and_S_equals_N_over_R0():
    res = simulate(P, Y0, 100, 0.05)
    tp, ip = peak_infection(res.t, res.I)
    s_at_peak = np.interp(tp, res.t, res.S)
    assert s_at_peak == pytest.approx(1000 / P.r0, rel=1e-2)

def test_root_finders():
    assert bisection(lambda x: x**2 - 2, 0, 2) == pytest.approx(2**0.5)
    assert newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.0) == pytest.approx(2**0.5)
    with pytest.raises(ValueError):
        bisection(lambda x: x**2 + 1, 0, 2)

def test_final_size_matches_simulation_and_methods_agree():
    res = simulate(P, Y0, 300, 0.1)
    s_inf = final_susceptible_fraction(P.r0, Y0[0] / 1000)
    assert 1 - s_inf == pytest.approx(attack_rate(res.S, 1000), abs=1e-3)
    assert s_inf == pytest.approx(
        final_susceptible_fraction(P.r0, Y0[0] / 1000, method="newton"))
    assert final_susceptible_fraction(0.8) == 1.0

def test_gaussian_elimination_and_least_squares():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    assert solve_linear_system(A, b) == pytest.approx([2, 3, -1])
    with pytest.raises(ValueError):
        solve_linear_system([[1, 2], [2, 4]], [1, 2])
    x = np.arange(10.0)
    X = np.column_stack([np.ones(10), x])
    assert least_squares(X, 3 + 2 * x) == pytest.approx([3, 2])

def test_disease_free_eigenvalue_is_beta_minus_gamma():
    assert disease_free_eigenvalues(0.5, 0.2).max() == pytest.approx(0.3)
    assert disease_free_eigenvalues(0.1, 0.2).max() == pytest.approx(0.0)

def test_growth_rate_estimate_close_to_beta_minus_gamma():
    res = simulate(P, Y0, 100, 0.1)
    r, _ = estimate_growth_rate(res.t, res.I)
    assert r == pytest.approx(P.beta - P.gamma, rel=0.1)
    assert doubling_time(r) == pytest.approx(np.log(2) / r)

def test_fit_recovers_parameters_from_noisy_data():
    res = simulate(P, Y0, 100, 0.5)
    noisy = add_noise(res.I, 0.03, seed=1)
    beta, gamma, _ = fit_sir_parameters(res.t, noisy, 1000, 1)
    assert beta == pytest.approx(0.5, rel=0.1)
    assert gamma == pytest.approx(0.2, rel=0.1)

def test_rk4_close_to_reference_euler_is_not():
    e = simulate(P, Y0, 100, 0.5, "euler")
    r = simulate(P, Y0, 100, 0.5, "rk4")
    ref = reference_solution(P, Y0, r.t)
    assert max_abs_error(r.y, ref) < 0.1 * max_abs_error(e.y, ref)
