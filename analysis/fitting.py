import numpy as np
from scipy.optimize import curve_fit
from analysis.linear_systems import least_squares
from integrators.rk import rk4
from models.sir import SIRParams, sir_rhs, make_initial_state


def add_noise(values, rel_sigma=0.05, seed=0):   #multiplicative Gaussian noise 
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    return np.clip(v * (1 + rel_sigma * rng.standard_normal(v.shape)), 0, None)


def estimate_growth_rate(t, I, early_fraction=0.1):
    t, I = np.asarray(t, float), np.asarray(I, float)
    peak = int(np.argmax(I))
    idx = np.where((np.arange(I.size) <= peak) & (I > 0)
                   & (I <= early_fraction * I[peak]))[0]
    if idx.size < 3:
        raise ValueError("not enough early-phase points to fit")
    X = np.column_stack([np.ones(idx.size), t[idx]])
    a, r = least_squares(X, np.log(I[idx]))
    return float(r), float(a)


def fit_sir_parameters(t, I_obs, N, I0, guess=(0.5, 0.2)):
    t, I_obs = np.asarray(t, float), np.asarray(I_obs, float)
    h = t[1] - t[0]
    y0 = make_initial_state(N, I0)
    def model(_, beta, gamma):
        p = SIRParams(beta, gamma, N)
        _, y = rk4(lambda tt, yy: sir_rhs(tt, yy, p), y0, t[0], t[-1], h)
        return y[:, 1]
    popt, pcov = curve_fit(model, t, I_obs, p0=guess,
                           bounds=([1e-6, 1e-6], [10.0, 10.0]))
    return float(popt[0]), float(popt[1]), pcov
