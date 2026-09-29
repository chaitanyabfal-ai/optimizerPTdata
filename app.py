"""
Oil Pipeline Pressure & Deposition Analysis - Streamlit Dashboard
Dual-view layout: LEFT PANEL 40% (inputs, KPI cards, raw data sub-table)
                  RIGHT PANEL 60% (Plotly charts, drill-down analytics tabs)

Run:
    streamlit run app.py

Inputs:
    - SCADA PT CSVs (upstream/downstream): columns `time` and `pressure_bar`
      (or a single file with `pt_upstream` / `pt_downstream` columns)
    - Elevation profile CSV: `distance_m,elevation_m` (chainage in m, m MSL)
    - Manual inputs: pipeline geometry, liquid properties, valve operations
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from pipeline_hydraulics import (
    PipelineGeometry,
    LiquidProperties,
    ValveOperations,
    run_simulation,
    optimize_deposition_diameters,
    deposition_profile,
    segment_grid,
)

st.set_page_config(
    page_title="Oil Pipeline Pressure & Deposition Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🛢️ Oil Pipeline Pressure & Deposition Analysis")
st.markdown(
    "Dual-panel dashboard for SCADA pressure-transmitter validation, upstream "
    "residual-pressure analysis, and wax-deposition diameter optimization over "
    "100 chainage segments."
)


# ────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def normalize_pt(df, side):
    """Accept time,pressure_bar (single PT) or pt_upstream/pt_downstream columns."""
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff").lower() for c in df.columns]
    if side in df.columns:
        return df[["time", side]].rename(columns={side: "pressure_bar"})
    if "pressure_bar" in df.columns:
        return df[["time", "pressure_bar"]]
    raise ValueError(
        f"PT CSV must contain a `pressure_bar` column or a `{side}` column."
    )


def normalize_elevation(df):
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff").lower() for c in df.columns]
    if "distance_m" in df.columns and "elevation_m" in df.columns:
        return df[["distance_m", "elevation_m"]]
    if len(df.columns) >= 2:
        return df.iloc[:, :2].rename(columns={df.columns[0]: "distance_m",
                                             df.columns[1]: "elevation_m"})
    raise ValueError("Elevation CSV must contain `distance_m` and `elevation_m`.")


def default_elevation(length_m):
    x = np.linspace(0.0, length_m, 100)
    y = 50.0 + 30.0 * np.sin(x / 1200.0) - 12.0 * np.cos(x / 450.0)
    return pd.DataFrame({"distance_m": x, "elevation_m": y})


# ────────────────────────────────────────────────────────────────────────────
# LEFT PANEL (40%) — Inputs, KPIs, raw data
# ────────────────────────────────────────────────────────────────────────────
left_panel, right_panel = st.columns([2, 3], gap="medium")

with left_panel:
    st.header("📋 Inputs & Controls")

    with st.container(border=True):
        st.subheader("1. Data Ingestion (SCADA & Elevation)")
        up_file = st.file_uploader("Upstream PT Data (CSV)", type="csv")
        down_file = st.file_uploader("Downstream PT Data (CSV)", type="csv")
        elev_file = st.file_uploader("Elevation Profile (CSV)", type="csv")

    geometry = PipelineGeometry()
    liquid = LiquidProperties()
    valve = ValveOperations()

    with st.expander("2. Pipeline Geometry", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            geometry.length_m = st.number_input(
                "Total Length (m)", value=8000.0, min_value=1.0, step=100.0)
            geometry.diameter_mm = st.number_input(
                "Nominal Diameter (mm)", value=193.7, min_value=1.0, step=1.0)
            geometry.wall_thickness_mm = st.number_input(
                "Wall Thickness (mm)", value=8.0, min_value=0.0, step=0.5)
        with c2:
            geometry.roughness_mm = st.number_input(
                "Roughness (mm)", value=0.045, min_value=0.001, step=0.005)
            geometry.n_segments = st.slider(
                "Chainage Segments", 10, 200, 100)
            geometry.valve_position_m = st.number_input(
                "Valve Location (m chainage)", value=8000.0, min_value=0.0,
                step=50.0)

    with st.expander("3. Liquid Properties", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            liquid.density_kgm3 = st.number_input(
                "Density (kg/m³)", value=860.0, min_value=1.0, step=5.0)
            liquid.viscosity_pas = st.number_input(
                "Dynamic Viscosity (Pa·s)", value=0.035, min_value=1e-6,
                format="%.6f")
        with c2:
            liquid.bulk_modulus_pa = st.number_input(
                "Bulk Modulus (Pa)", value=1.5e9, min_value=1e5, format="%.3e")
            flow_m3h = st.number_input(
                "Flow Rate (m³/h)", value=145.0, min_value=0.1, step=5.0)
    flow_m3s = flow_m3h / 3600.0

    with st.expander("4. Valve Operations", expanded=False):
        valve.opening_pct = st.slider("Valve Opening (%)", 0, 100, 100)
        valve.cv_max = st.number_input(
            "Valve Cv (max)", value=132.0, min_value=0.1, step=10.0)

    # ── Load PT data with sensible defaults ────────────────────────────────
    p_up_in, p_down_in = 55.2, 12.3
    up_df = down_df = None
    try:
        if up_file:
            up_df = normalize_pt(load_csv(up_file), "pt_upstream")
            p_up_in = float(up_df["pressure_bar"].iloc[-1])
        if down_file:
            down_df = normalize_pt(load_csv(down_file), "pt_downstream")
            p_down_in = float(down_df["pressure_bar"].iloc[-1])
        if up_file is None and down_file is None:
            # repo's B178 dataset: both PTs from a single-file convention
            st.info("No SCADA CSV loaded — using demo values (55.2 / 12.3 bar). "
                    "Upload PT CSVs with columns `time,pressure_bar`.")
    except Exception as e:
        st.error(f"PT CSV error: {e}")

    try:
        elev_df = normalize_elevation(load_csv(elev_file)) if elev_file \
            else default_elevation(geometry.length_m)
    except Exception as e:
        st.error(f"Elevation CSV error: {e}")
        elev_df = default_elevation(geometry.length_m)

    with st.expander("5. Deposition & Optimization", expanded=False):
        dep_intensity = st.slider("Deposition Intensity", 0.0, 1.0, 0.25, 0.05)
        run_opt = st.checkbox("Run Diameter Optimization (DE)", value=True)
        max_iter = st.number_input(
            "Optimizer Max Iterations", value=15, min_value=1, max_value=200)

    # ── Run simulation (cached) ─────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def cached_sim(p_up, p_down, flow, _geom_key, _liq, _valve, _elev,
                   length_m, dia_mm, wall, rough, n_seg, valve_pos,
                   dep_int, do_opt, iters):
        geometry = PipelineGeometry(length_m, dia_mm, wall, rough, n_seg, valve_pos)
        liquid = LiquidProperties(**_liq)
        valve = ValveOperations(**_valve)
        _, x_seg = segment_grid(length_m, n_seg)
        dep_frac = deposition_profile(x_seg, geometry, dep_int, 0.30)
        dep_d = dia_mm * (1.0 - dep_frac)
        dep_d_full = np.concatenate([dep_d, [dep_d[-1]]])

        base = run_simulation(p_up, p_down, flow, geometry, liquid, valve,
                              _elev, deposition_d_mm=dep_d_full)
        opt_d, opt_res, hist = None, None, []
        if do_opt:
            opt_d, opt_res, hist = optimize_deposition_diameters(
                p_up, p_down, flow, geometry, liquid, valve, _elev,
                max_iter=int(iters))
        return base, opt_d, opt_res, hist

    geom_key = (geometry.length_m, geometry.diameter_mm, geometry.n_segments)
    liq_args = dict(density_kgm3=liquid.density_kgm3,
                    viscosity_pas=liquid.viscosity_pas,
                    bulk_modulus_pa=liquid.bulk_modulus_pa)
    valve_args = dict(opening_pct=valve.opening_pct, cv_max=valve.cv_max,
                      downstream_backpressure_bar=valve.downstream_backpressure_bar)

    with st.spinner("Running hydraulic simulation" +
                    (" and DE optimization..." if run_opt else "...")):
        base_res, opt_d, opt_res, opt_hist = cached_sim(
            p_up_in, p_down_in, flow_m3s, geom_key, liq_args, valve_args,
            elev_df.astype(object), geometry.length_m, geometry.diameter_mm,
            geometry.wall_thickness_mm, geometry.roughness_mm,
            geometry.n_segments, geometry.valve_position_m, dep_intensity,
            run_opt, max_iter)

    sim = opt_res if opt_res is not None else base_res
    k = sim.kpis

    # ── KPI cards ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Key Performance Indicators")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Upstream PT (Actual)", f"{k['p_up_actual_bar']:.2f} bar")
            st.metric("Residual Pressure (upstream, mean)",
                      f"{k['residual_mean_bar']:.2f} bar",
                      delta=f"{k['residual_max_bar']:.2f} bar max")
        with r2:
            st.metric("Downstream PT (Actual)",
                      f"{k['p_down_actual_bar']:.2f} bar",
                      delta=f"Est: {k['p_down_estimated_bar']:.2f} bar")
            st.metric("Max Wax Deposition", f"{k['wax_max_mm']:.1f} mm",
                      delta=f"{k['area_loss_pct']:.1f}% area loss")

    # ── Filtered raw data sub-table ────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Raw Data Inspection")
        raw_tab, seg_tab = st.tabs(["SCADA PT", "Segment Table"])
        with raw_tab:
            if up_df is not None:
                st.caption("Upstream PT (latest 200 rows)")
                st.dataframe(up_df.tail(200), hide_index=True,
                             use_container_width=True, height=180)
            elif down_df is not None:
                st.caption("Downstream PT (latest 200 rows)")
                st.dataframe(down_df.tail(200), hide_index=True,
                             use_container_width=True, height=180)
            else:
                st.caption("No SCADA CSV uploaded.")
        with seg_tab:
            st.caption("Per-segment simulation results (first 50 rows)")
            st.dataframe(sim.df.head(50), hide_index=True,
                         use_container_width=True, height=180)
        csv_out = io.StringIO()
        sim.df.to_csv(csv_out, index=False)
        st.download_button("📥 Download Segment Results (CSV)",
                           csv_out.getvalue(),
                           "pipeline_segment_results.csv", "text/csv",
                           use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL (60%) — Charts & analytics
# ────────────────────────────────────────────────────────────────────────────
with right_panel:
    st.header("📊 Interactive Analytics & Profiles")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Pressure Profiles",
        "⭕ Diameter & Deposition Profile",
        "⛰️ Elevation Profile",
        "📉 Distribution & Trends",
    ])
    sdf = sim.df
    x_km = sdf["chainage_m"] / 1000.0

    with tab1:
        with st.container(border=True):
            st.markdown("### Upstream: Actual vs Ideal Pressure")
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=x_km, y=sdf["upstream_actual_bar"], mode="lines",
                name="Upstream Actual (PT)", line=dict(color="#1f77b4", width=3)))
            fig_p.add_trace(go.Scatter(
                x=x_km, y=sdf["upstream_ideal_bar"], mode="lines",
                name="Upstream Ideal (from downstream PT)",
                line=dict(color="#2ca02c", width=2, dash="dash")))
            fig_p.update_layout(
                xaxis_title="Chainage (km)", yaxis_title="Pressure (bar)",
                legend=dict(orientation="h", y=1.12), height=320,
                margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
            st.plotly_chart(fig_p, use_container_width=True)

        with st.container(border=True):
            st.markdown("### Downstream: Actual vs Estimated (Optimized)")
            fig_ds = go.Figure()
            fig_ds.add_trace(go.Scatter(
                x=x_km, y=np.full(len(sdf), k["p_down_actual_bar"]),
                mode="lines", name="Downstream Actual (PT)",
                line=dict(color="#d62728", width=2)))
            fig_ds.add_trace(go.Scatter(
                x=x_km, y=np.full(len(sdf), k["p_down_estimated_bar"]),
                mode="lines", name="Downstream Estimated (model)",
                line=dict(color="#ff7f0e", width=2, dash="dash")))
            fig_ds.update_layout(
                xaxis_title="Chainage (km)", yaxis_title="Pressure (bar)",
                legend=dict(orientation="h", y=1.12), height=260,
                margin=dict(l=20, r=20, t=30, b=20), template="plotly_white")
            st.plotly_chart(fig_ds, use_container_width=True)

        with st.container(border=True):
            st.markdown("### Residual Pressure (Upstream only)")
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=x_km, y=sdf["upstream_residual_bar"], mode="lines",
                name="Residual (Actual − Ideal)",
                line=dict(color="#d62728", width=2),
                fill="tozeroy", fillcolor="rgba(214,39,40,0.15)"))
            fig_res.update_layout(
                xaxis_title="Chainage (km)", yaxis_title="Residual (bar)",
                height=240, margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white")
            st.plotly_chart(fig_res, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown(
                "### Diameter Profile: Actual vs Deposition "
                f"({geometry.n_segments} segments, shaded deposit zone)")
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                x=x_km, y=sdf["nominal_d_mm"], mode="lines",
                name="Actual Diameter (clean)", line=dict(color="navy", width=2.5)))
            y_dep = opt_d if opt_d is not None else sdf["deposition_d_mm"].to_numpy()
            fig_d.add_trace(go.Scatter(
                x=x_km, y=y_dep, mode="lines",
                name="Deposition Diameter", fill="tonexty",
                fillcolor="rgba(255,127,14,0.4)",
                line=dict(color="darkorange", width=2, dash="dash")))
            fig_d.update_layout(
                xaxis_title="Chainage (km)", yaxis_title="Inner Diameter (mm)",
                legend=dict(orientation="h", y=1.12), height=360,
                margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
            st.plotly_chart(fig_d, use_container_width=True)

        with st.container(border=True):
            if opt_d is not None:
                st.markdown("### Optimized Diameter Profile (per segment)")
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(
                    x=x_km, y=sdf["deposition_d_mm"], mode="lines",
                    name="Initial Deposition Diameter",
                    line=dict(color="#ff7f0e", width=1.5, dash="dot")))
                fig_opt.add_trace(go.Scatter(
                    x=x_km, y=np.concatenate([opt_d, [opt_d[-1]]]), mode="lines",
                    name="Optimized Diameter",
                    line=dict(color="#9467bd", width=2.5)))
                fig_opt.update_layout(
                    xaxis_title="Chainage (km)", yaxis_title="Diameter (mm)",
                    legend=dict(orientation="h", y=1.12), height=300,
                    margin=dict(l=20, r=20, t=30, b=20), template="plotly_white")
                st.plotly_chart(fig_opt, use_container_width=True)
                st.caption(f"DE evaluations: {opt_hist[0]['evals']} — "
                           f"downstream model: "
                           f"{opt_hist[0]['p_down_model_bar']:.2f} bar vs "
                           f"actual {k['p_down_actual_bar']:.2f} bar")
            else:
                st.info("Enable diameter optimization to view the optimized "
                        "profile.")

        with st.container(border=True):
            st.markdown("### Deposit Thickness per Segment")
            fig_wax = px.bar(
                sdf, x="chainage_m", y="wax_thickness_mm",
                labels={"chainage_m": "Chainage (m)",
                        "wax_thickness_mm": "Deposit thickness (mm)"},
                color="wax_thickness_mm", color_continuous_scale="Oranges")
            fig_wax.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20),
                                  template="plotly_white")
            st.plotly_chart(fig_wax, use_container_width=True)

    with tab3:
        with st.container(border=True):
            st.markdown("### Elevation Profile from Sea Level")
            fig_e = px.area(sdf, x="chainage_m", y="elevation_m",
                            labels={"chainage_m": "Chainage (m)",
                                    "elevation_m": "Elevation (m MSL)"})
            fig_e.update_traces(line_color="#2ca02c",
                                fillcolor="rgba(44,160,44,0.3)")
            fig_e.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20),
                                template="plotly_white")
            st.plotly_chart(fig_e, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("### Residual Pressure Trendline")
                fig_t = px.scatter(sdf, x="chainage_m",
                                   y="upstream_residual_bar",
                                   labels={"chainage_m": "Chainage (m)",
                                           "upstream_residual_bar":
                                               "Residual (bar)"})
                trend = sdf["upstream_residual_bar"].rolling(
                    7, center=True, min_periods=1).mean()
                fig_t.add_trace(go.Scatter(
                    x=sdf["chainage_m"], y=trend, mode="lines",
                    name="Trendline (rolling mean)",
                    line=dict(color="#d62728", width=2)))
                fig_t.update_layout(height=340,
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    template="plotly_white")
                st.plotly_chart(fig_t, use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("### Residual Distribution")
                fig_h = px.histogram(sdf, x="upstream_residual_bar", nbins=25,
                                     labels={"upstream_residual_bar":
                                             "Residual (bar)"})
                fig_h.update_layout(height=340,
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    template="plotly_white")
                st.plotly_chart(fig_h, use_container_width=True)

st.caption("Oil Pipeline Pressure & Deposition Analysis — steady-state "
           "Darcy–Weisbach + hydrostatic engine with per-segment "
           "differential-evolution diameter optimization.")
