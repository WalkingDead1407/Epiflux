"""
Epiflux — interactive SIR simulation dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app/streamlit_app.py` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis.differentiation import derivative, growth_rate
from analysis.metrics import (
    attack_rate,
    conservation_error,
    max_abs_error,
    peak_infection,
    reference_solution,
    rmse,
)
from analysis.root_finding import final_susceptible_fraction
from models.sir import SIRParams, make_initial_state, simulate


st.set_page_config(
    page_title="Epiflux | SIR Simulator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A compact visual system that keeps the plots and controls central.
st.markdown(
    """
    <style>
      :root { --ink:#202b33; --muted:#687780; --line:#dce3e5; --paper:#f5f7f6; --green:#236b59; --red:#b64b45; }
      html, body, [class*="css"] { font-family: "Segoe UI", "Helvetica Neue", sans-serif; }
      .stApp { background:var(--paper); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(245,247,246,.96); }
      .block-container { padding:2rem 2.5rem 3rem; max-width:1450px; }
      [data-testid="stSidebar"] { background:#edf1ef; border-right:1px solid var(--line); }
      [data-testid="stSidebar"] * { color:#26343a; }
      [data-testid="stSidebar"] h2 { font-size:1.05rem; font-weight:650; letter-spacing:0; }
      [data-testid="stSidebar"] [data-testid="stSlider"] { padding:.15rem 0 .5rem; }
      [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--green); border-color:var(--green); }
      [data-testid="stNumberInput"] input { background:#fff; color:#26343a; border:1px solid #cbd5d5; border-radius:4px; }
      .masthead { border-bottom:1px solid #cbd5d5; padding:0 0 1.15rem; margin-bottom:1.7rem; }
      .masthead-top { display:flex; justify-content:space-between; align-items:center; color:#687780; font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; }
      .brand { font-size:2.35rem; line-height:1.1; letter-spacing:-.065em; font-weight:750; color:#1e302f; margin:.55rem 0 .35rem; }
      .brand span { color:var(--green); }
      .masthead p { color:#5f6e74; max-width:760px; margin:0; font-size:.96rem; }
      .section-kicker { color:#63736f; font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:1.5rem 0 .35rem; }
      h2, h3, h4 { color:#26343a !important; letter-spacing:-.025em; }
      h2 { font-weight:650 !important; }
      [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:5px; padding:14px 16px; min-height:100px; box-shadow:none; }
      [data-testid="stMetricLabel"] { color:#66757b !important; font-size:.75rem; }
      [data-testid="stMetricValue"] { color:#20332f !important; font-weight:650; letter-spacing:-.035em; }
      [data-testid="stMetricDelta"] { font-size:.75rem; }
      [data-testid="stTabs"] { margin-top:1rem; }
      [data-testid="stTabs"] [role="tablist"] { gap:1.25rem; border-bottom:1px solid var(--line); }
      [data-testid="stTabs"] button { color:#647278; padding:.65rem .1rem; font-weight:600; }
      [data-testid="stTabs"] button[aria-selected="true"] { color:var(--green); border-bottom:2px solid var(--green); }
      [data-testid="stAlert"] { background:#edf3f0; border:1px solid #d2dfd9; border-radius:4px; color:#31443e; }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:4px; overflow:hidden; }
      .status-line { color:#63736f; font-size:.7rem; font-weight:650; letter-spacing:.08em; text-transform:uppercase; margin:.25rem 0 1rem; }
      .status-dot { color:var(--green); padding-right:.35rem; }
      .small-note { color:#718087; font-size:.78rem; line-height:1.65; }
      hr { border-color:var(--line) !important; }
      .stCaption, [data-testid="stCaptionContainer"] { color:#718087 !important; }
      .stMarkdown p, .stMarkdown li { color:#4e5d63; }
      @media(max-width:800px) { .block-container{padding:1rem 1rem 2rem}.brand{font-size:2rem}.masthead-top{gap:.5rem;flex-wrap:wrap} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="masthead">
      <div class="masthead-top"><span>Computational epidemiology / Project 01</span><span>Interactive SIR model</span></div>
      <div class="brand">Epiflux<span>.</span></div>
      <p>An interactive environment for exploring epidemic dynamics and comparing numerical integration methods.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("MODEL INPUTS")
    st.header("Simulation controls")
    st.caption("Tune the outbreak scenario. The model recalculates automatically.")

    beta = st.slider("Transmission rate (β)", 0.05, 1.50, 0.50, 0.01)
    gamma = st.slider("Recovery rate (γ)", 0.02, 1.00, 0.20, 0.01)
    population = st.number_input("Total population (N)", min_value=100, max_value=1_000_000,
                                 value=1000, step=100)
    infected0 = st.number_input("Initially infected (I₀)", min_value=1,
                                max_value=int(population) - 1, value=1, step=1)
    days = st.slider("Simulation duration (days)", 30, 365, 120, 5)
    step_size = st.select_slider("Time step (h, days)", options=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
                                 value=0.5)
    st.divider()
    st.markdown("**Solver engine**")
    st.caption("Euler · Runge–Kutta 2 · Runge–Kutta 4")
    st.caption("All values are illustrative model inputs, not live surveillance data.")

try:
    params = SIRParams(beta=float(beta), gamma=float(gamma), N=float(population))
    y0 = make_initial_state(float(population), float(infected0))
    methods = ("euler", "rk2", "rk4")
    results = {
        method: simulate(params, y0, float(days), float(step_size), method)
        for method in methods
    }
    reference = reference_solution(params, y0, results["rk4"].t)
except Exception as exc:
    st.error(f"Could not run this simulation: {exc}")
    st.stop()

rk4_result = results["rk4"]
peak_day, peak_count = peak_infection(rk4_result.t, rk4_result.I)
attack = attack_rate(rk4_result.S, float(population))
r0 = params.r0
max_error = max_abs_error(rk4_result.y, reference)
drift = conservation_error(rk4_result.y, float(population))

try:
    final_s = final_susceptible_fraction(r0, y0[0] / float(population))
    theoretical_attack = 1.0 - final_s
except (ValueError, RuntimeError, ZeroDivisionError):
    theoretical_attack = 0.0

st.markdown("<div class='section-kicker'>01 / Outbreak intelligence</div>", unsafe_allow_html=True)
st.subheader("Scenario snapshot")
st.markdown("<div class='status-line'><span class='status-dot'>●</span>Calculated using the RK4 trajectory</div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Basic reproduction number (R₀)", f"{r0:.2f}",
          help="R₀ = β / γ in this SIR model.")
m2.metric("Peak infected (RK4)", f"{peak_count:,.0f}",
          help=f"Estimated peak at day {peak_day:.1f}.")
m3.metric("Attack rate (RK4)", f"{attack * 100:.1f}%")
m4.metric("Max RK4 error vs reference", f"{max_error:.2e}")

if r0 > 1:
    st.info("Growth potential detected · R₀ is above 1, so infections initially tend to grow under the selected assumptions.")
else:
    st.info("Contained growth scenario · R₀ is at or below 1, so infections are not expected to sustain initial growth in this model.")

tab_overview, tab_methods, tab_growth = st.tabs(
    ["Epidemic overview", "Numerical methods", "Infection growth"]
)

with tab_overview:
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("<div class='section-kicker'>02 / Population dynamics</div>", unsafe_allow_html=True)
        st.markdown("#### Compartment trajectory")
        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(rk4_result.t, rk4_result.S, label="Susceptible (S)", linewidth=2.4, color="#527f9c")
        ax.plot(rk4_result.t, rk4_result.I, label="Infected (I)", linewidth=2.4, color="#b64b45")
        ax.plot(rk4_result.t, rk4_result.R, label="Recovered (R)", linewidth=2.4, color="#43836f")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("People")
        ax.legend(frameon=False, ncol=3, loc="upper center", labelcolor="#405057")
        ax.tick_params(colors="#66757b")
        ax.xaxis.label.set_color("#536269")
        ax.yaxis.label.set_color("#536269")
        for spine in ax.spines.values(): spine.set_color("#dce3e5")
        ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Simulation summary")
        st.metric("Estimated peak day", f"Day {peak_day:.1f}")
        st.metric("Final susceptible", f"{rk4_result.S[-1]:,.0f}")
        st.metric("Final recovered", f"{rk4_result.R[-1]:,.0f}")
        st.metric("Theoretical final-size attack rate", f"{theoretical_attack * 100:.1f}%")
        st.caption("The theoretical final-size value is obtained from the project's root-finding routine.")

    st.markdown("#### Current model assumptions")
    st.write(
        "A closed population with no births, deaths unrelated to the disease, migration, "
        "or changing contact behaviour. Individuals move from susceptible to infected to recovered."
    )

with tab_methods:
    st.markdown("<div class='section-kicker'>03 / Numerical validation</div>", unsafe_allow_html=True)
    st.markdown("#### Solver performance comparison")
    st.caption("All methods use the same parameters, initial state and time grid. DOP853 is used as a high-accuracy reference.")
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    palette = {"euler": "#b27b35", "rk2": "#75849a", "rk4": "#236b59"}
    for name, result in results.items():
        ax.plot(result.t, result.I, label=name.upper(), linewidth=2,
                color=palette[name], alpha=0.95)
    ax.plot(rk4_result.t, reference[:, 1], "--", label="DOP853 reference",
            color="#687780", linewidth=1.8)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected people")
    ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
    ax.legend(frameon=False, ncol=4, labelcolor="#405057")
    ax.tick_params(colors="#66757b")
    ax.xaxis.label.set_color("#536269")
    ax.yaxis.label.set_color("#536269")
    for spine in ax.spines.values(): spine.set_color("#dce3e5")
    ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    rows = []
    for name, result in results.items():
        peak_t, peak_i = peak_infection(result.t, result.I)
        rows.append({
            "Method": name.upper(),
            "Peak infected": round(peak_i, 2),
            "Peak day": round(peak_t, 2),
            "Attack rate": f"{attack_rate(result.S, float(population)):.4f}",
            "Max abs. error": f"{max_abs_error(result.y, reference):.3e}",
            "RMSE": f"{rmse(result.y, reference):.3e}",
            "Population drift": f"{conservation_error(result.y, float(population)):.3e}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if any(np.min(result.y) < -1e-8 for result in results.values()):
        st.warning(
            "At least one solver produced a negative compartment value. "
            "This can occur with a coarse time step; reduce h to improve numerical behaviour."
        )
    st.caption(f"Maximum population-conservation drift for RK4: {drift:.3e} people.")

with tab_growth:
    st.markdown("<div class='section-kicker'>04 / Transmission signals</div>", unsafe_allow_html=True)
    st.markdown("#### Infection growth rate")
    st.write(
        "The instantaneous relative growth rate is estimated numerically as "
        "g(t) = I′(t) / I(t), using finite differences on the RK4 trajectory."
    )
    try:
        g = growth_rate(rk4_result.t, rk4_result.I)
        fig, ax = plt.subplots(figsize=(10, 4.4), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(rk4_result.t, g, color="#527f9c", linewidth=2)
        ax.axhline(0, color="#aab5b8", linewidth=1, linestyle="--")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Relative growth rate (per day)")
        ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
        ax.tick_params(colors="#66757b")
        ax.xaxis.label.set_color("#536269")
        ax.yaxis.label.set_color("#536269")
        for spine in ax.spines.values(): spine.set_color("#dce3e5")
        ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        early = g[(rk4_result.t <= peak_day) & np.isfinite(g)]
        if early.size:
            st.metric("Maximum estimated growth rate before peak", f"{np.max(early):.4f} per day")
    except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
        st.warning(f"Growth-rate analysis is unavailable for this run: {exc}")

    st.markdown("#### Infected population and its rate of change")
    try:
        dI = derivative(rk4_result.t, rk4_result.I)
        fig, ax = plt.subplots(figsize=(10, 4.4), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(rk4_result.t, dI, color="#b27b35", linewidth=2)
        ax.axhline(0, color="#aab5b8", linewidth=1, linestyle="--")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Change in infected people (people/day)")
        ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
        ax.tick_params(colors="#66757b")
        ax.xaxis.label.set_color("#536269")
        ax.yaxis.label.set_color("#536269")
        for spine in ax.spines.values(): spine.set_color("#dce3e5")
        ax.grid(alpha=.55, color="#dce3e5", linewidth=.7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ValueError as exc:
        st.warning(f"Derivative plot is unavailable: {exc}")

st.divider()

