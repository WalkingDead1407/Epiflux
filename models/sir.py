"""
dS/dt = -beta * S * I / N
dI/dt =  beta * S * I / N - gamma * I
dR/dt =  gamma * I
"""
from dataclasses import dataclass

import numpy as np

from integrators.euler import euler
from integrators.rk import rk2, rk4

METHODS = {"euler": euler, "rk2": rk2, "rk4": rk4}


@dataclass(frozen=True)
class SIRParams:
    #beta = transmission rate, gamma = recovery rate, N = population
    beta: float
    gamma: float
    N: float = 1000.0

    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.N <= 0:
            raise ValueError("N must be > 0")

    @property
    def r0(self) -> float:
        return self.beta / self.gamma

def make_initial_state(N: float, infected: float = 1.0, recovered: float = 0.0):
    if infected <= 0:
        raise ValueError("infected must be > 0")
    if recovered < 0:
        raise ValueError("recovered must be >= 0")
    susceptible = N - infected - recovered
    if susceptible < 0:
        raise ValueError("infected + recovered cannot exceed N")
    return np.array([susceptible, infected, recovered], dtype=float)

def sir_rhs(t: float, y, params: SIRParams) -> np.ndarray:
    #Right-hand side f(t, y) of the SIR system(t is unused: autonomous ode)
    S, I, _ = y
    new_infections = params.beta * S * I / params.N
    recoveries = params.gamma * I
    return np.array([-new_infections, new_infections - recoveries, recoveries])


@dataclass
class SimulationResult:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    params: SIRParams
    method: str
    h: float

    @property
    def y(self) -> np.ndarray:
        return np.column_stack([self.S, self.I, self.R])


def simulate(params: SIRParams, y0, t_end: float, h: float = 0.1,
             method: str = "rk4", t0: float = 0.0) -> SimulationResult:
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', choose from {list(METHODS)}")
    y0 = np.asarray(y0, dtype=float)
    if y0.shape != (3,):
        raise ValueError("y0 must be [S0, I0, R0]")
    if not np.isclose(y0.sum(), params.N):
        raise ValueError("S0 + I0 + R0 must equal N")

    t, y = METHODS[method](lambda tt, yy: sir_rhs(tt, yy, params), y0, t0, t_end, h)
    return SimulationResult(t, y[:, 0], y[:, 1], y[:, 2], params, method, h)
