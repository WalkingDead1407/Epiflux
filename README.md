# Epiflux

**Epidemic Spread Simulation Using Numerical Methods**

Epiflux Numerics is a Python-based Computational Methods project that simulates the spread of infectious diseases using the **SIR (Susceptible–Infected–Recovered) model**. It applies numerical algorithms to analyze epidemic progression and visualize how transmission and recovery rates affect infection dynamics.

## Objectives

- Simulate epidemic progression using differential equations.
- Implement and compare Euler's method and Runge–Kutta 4 (RK4).
- Analyze infection growth using numerical differentiation.
- Apply linear algebra, curve fitting, and regression.
- Visualize epidemic dynamics through an interactive dashboard.

## Mathematical Model

The SIR model is defined by:

\[
\frac{dS}{dt}=-\beta\frac{SI}{N}
\]

\[
\frac{dI}{dt}=\beta\frac{SI}{N}-\gamma I
\]

\[
\frac{dR}{dt}=\gamma I
\]

Where \(S\), \(I\), and \(R\) represent susceptible, infected, and recovered populations; \(\beta\) is the transmission rate, \(\gamma\) is the recovery rate, and \(N\) is the total population.

## Features

- SIR epidemic simulation with adjustable parameters.
- Euler vs. RK4 accuracy comparison.
- Infection growth and peak analysis.
- Linear algebra and regression demonstrations.
- Interactive graphs and parameter controls using Streamlit.

## Tech Stack

- **Python** — Core implementation
- **NumPy** — Numerical computation and linear algebra
- **SciPy** — Numerical analysis and curve fitting
- **Matplotlib** — Visualization
- **Streamlit** — Interactive dashboard

## Project Structure

```text
epiflux/
├── analysis/
│   ├── differentiation.py
│   ├── fitting.py
│   ├── linear_systems.py
│   ├── metrics.py
│   └── root_finding.py
├── app/
│   └── streamlit_app.py
├── data/
├── docs/
├── integrators/
│   ├── euler.py
│   └── rk.py
├── models/
│   └── sir.py
├── notebooks/
├── results/
│   └── figures/
├── scripts/
│   └── run_simulation.py
├── tests/
│   └── test_integrators.py
├── visualization/
│   └── plots.py
├── requirements.txt
└── README.md
```

### Module Overview

- **`models/`** — Defines the SIR epidemic model.
- **`integrators/`** — Implements Euler's method and Runge–Kutta methods.
- **`analysis/`** — Contains numerical differentiation, root-finding, linear systems, curve fitting, and analysis metrics.
- **`visualization/`** — Generates epidemic and numerical comparison plots.
- **`app/`** — Provides the interactive Streamlit dashboard.
- **`scripts/`** — Runs simulations from the command line.
- **`tests/`** — Tests numerical integration algorithms.
- **`data/`** — Stores input datasets, if required.
- **`notebooks/`** — Contains exploratory analysis and experiments.
- **`results/figures/`** — Stores generated graphs and visualizations.
- **`docs/`** — Contains additional project documentation.
/
```

## Installation

```bash
git clone https://github.com/walkingdead1407/epiflu.git
cd epiflux

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python main.py
```

Launch the interactive dashboard from the project root:

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides adjustable SIR parameters, population and infection metrics, SIR trajectory plots, Euler/RK2/RK4 comparisons against a DOP853 reference, and numerical infection-growth analysis.

## Expected Results

- Susceptible, infected, and recovered population graphs.
- Peak infection estimates and growth-rate analysis.
- Euler and RK4 numerical comparisons.
- Regression curves and error measurements.
- Interactive epidemic parameter exploration.

## Academic Purpose

This project demonstrates the application of numerical integration, differentiation, linear algebra, and regression to a mathematical model of epidemic spread.

**Note:** Epiflux is intended for educational purposes, not clinical or public-health forecasting.

## License

License to be decided.

---

**Epiflux — Exploring epidemic dynamics through numerical computation.**
