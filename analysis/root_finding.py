import math

def bisection(f, a, b, tol=1e-10, max_iter=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or (b - a) / 2 < tol:
            return m
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def newton(f, df, x0, tol=1e-12, max_iter=100):
    x = x0
    for _ in range(max_iter):
        d = df(x)
        if d == 0:
            raise ZeroDivisionError("derivative is zero")
        step = f(x) / d
        x -= step
        if abs(step) < tol:
            return x
    raise RuntimeError("Newton's method did not converge")


def _final_size_funcs(r0, s0, rec0):
    def f(s):
        return s - s0 * math.exp(-r0 * (1 - s - rec0))

    def df(s):
        return 1 - s0 * r0 * math.exp(-r0 * (1 - s - rec0))

    return f, df


def final_susceptible_fraction(r0, s0=1.0, recovered0=0.0, method="bisection"):
    if r0 <= 0 or not 0 < s0 <= 1:
        raise ValueError("need r0 > 0 and 0 < s0 <= 1")
    if r0 * s0 <= 1:
        return s0
    f, df = _final_size_funcs(r0, s0, recovered0)
    if method == "newton":
        return newton(f, df, 0.0)
    return bisection(f, 1e-12, 1.0 / r0)
