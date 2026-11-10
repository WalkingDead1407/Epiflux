import argparse
from pathlib import Path
import numpy as np
from analysis.metrics import (attack_rate, conservation_error, max_abs_error,
                              peak_infection, reference_solution)
from analysis.root_finding import final_susceptible_fraction
from models.sir import SIRParams, make_initial_state, simulate

def main(argv=None):
    p = argparse.ArgumentParser(description="Epiflux SIR simulation")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--N", type=float, default=1000)
    p.add_argument("--I0", type=float, default=1)
    p.add_argument("--days", type=float, default=120)
    p.add_argument("--h", type=float, default=0.5, help="time step in days")
    p.add_argument("--save", action="store_true", help="write CSVs to results/")
    args = p.parse_args(argv)

    params = SIRParams(args.beta, args.gamma, args.N)
    y0 = make_initial_state(args.N, args.I0)
    print(f"R0 = {params.r0:.3f}")

    results = {m: simulate(params, y0, args.days, args.h, m) for m in ("euler", "rk4")}
    ref = reference_solution(params, y0, results["rk4"].t)

    for name, res in results.items():
        tp, ip = peak_infection(res.t, res.I)
        print(f"[{name:5s}] peak I = {ip:8.2f} at t = {tp:6.2f} | "
              f"attack rate = {attack_rate(res.S, args.N):.4f} | "
              f"max error vs reference = {max_abs_error(res.y, ref):.3e} | "
              f"S+I+R drift = {conservation_error(res.y, args.N):.1e}")

    s_inf = final_susceptible_fraction(params.r0, y0[0] / args.N)
    print(f"Theory (final-size equation): attack rate = {1 - s_inf:.4f}")

    if args.save:
        out = Path("results")
        out.mkdir(exist_ok=True)
        for name, res in results.items():
            data = np.column_stack([res.t, res.y])
            np.savetxt(out / f"sir_{name}.csv", data, delimiter=",",
                       header="t,S,I,R", comments="")
        print(f"Saved CSVs to {out}/")


if __name__ == "__main__":
    main()

"""command line command:
    python -m scripts.run_simulation --beta 0.5 --gamma 0.2 --days 120 --save
I tend to forget this ;-;"""
