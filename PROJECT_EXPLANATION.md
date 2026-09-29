# Project Explanation — OptimizerPTdata / Gas Transient Simulation

This project analyses oil/gas pipeline pressure behaviour from SCADA
pressure-transmitter (PT) data, and optimises the pipeline's internal
diameter profile to explain the recorded pressures. It ships in two forms:

1. **`gas_moc_smart_optimizer_v5 (1).py`** — the original Tkinter desktop
   application: a Method-of-Characteristics (MOC) gas transient simulator with
   a Differential-Evolution diameter optimizer (Wylie & Streeter, Ch. 15).
2. **`app.py` + `pipeline_hydraulics.py`** — the new Streamlit web dashboard
   with a headless steady-state hydraulic engine, built for the oil-pipeline
   use case (upstream/downstream PTs, elevation profile, deposition).

---

## 1. The engineering problem

A pipeline is instrumented with two pressure transmitters:

- **Upstream PT** — records the actual pressure at the pipeline inlet
  (chainage 0).
- **Downstream PT** — records the actual pressure near the terminal
  (just upstream of the valve).

The pipeline also has a known **elevation profile** (height above sea level
along the entire chainage) and known operating conditions (geometry, liquid
properties, valve operations).

From these we want to answer:

- What pressure profile *should* exist along the pipeline (the **ideal**
  profile, back-calculated from the downstream PT on a clean pipe)?
- How does the **actual** profile (marched from the upstream PT through a
  deposition-laden pipe) differ from it? That difference is the
  **residual pressure** — a fingerprint of extra losses, typically wax or
  deposit build-up.
- What per-segment **effective (deposition) diameters** reproduce the
  recorded downstream pressure — i.e. where is the pipe restricted?

## 2. Data inputs

| Input | Format | Source |
|---|---|---|
| Upstream PT data | CSV: `time,pressure_bar` | SCADA (`Input/B178_set_1_filt500.csv`) |
| Downstream PT data | CSV: `time,pressure_bar` or `pt_upstream`/`pt_downstream` columns | SCADA |
| Elevation profile | CSV: `distance_m,elevation_m` | Survey (`Input/elevation.csv`) |
| Configuration | JSON (`Input/common_config_set1.json`) | Experiment setup |
| Manual inputs | UI fields | Geometry, liquid, valve parameters |

## 3. The physics engine (`pipeline_hydraulics.py`)

Steady-state simulation over N chainage segments (default 100):

- **Friction** — Darcy–Weisbach head loss with the Haaland explicit friction
  factor, evaluated per segment on the *effective* (deposition-reduced)
  diameter:

  ```
  Δp_friction = f · (dx/D) · (ρ·v²/2)
  ```

- **Hydrostatic head** — elevation change per segment from the interpolated
  profile: `Δp_hydro = ρ·g·Δz`.
- **Valve loss** — ISA-style Cv model at the valve chainage:
  `Δp = (Q/Cv_eff)²`.
- **Upstream actual profile** — marches downstream from the upstream PT
  through the deposition-laden pipe.
- **Upstream ideal profile** — back-calculated upstream from the downstream
  PT along a *clean* (nominal-diameter) pipe.
- **Residual pressure (upstream only)** — `actual − ideal` per node.
- **Downstream estimate** — the model's arrival pressure at the terminal,
  compared against the recorded downstream PT.

### Optimization

`optimize_deposition_diameters()` runs Differential Evolution (SciPy) over
per-segment diameter multipliers (default bounds 0.55–0.98 × nominal) so the
modelled downstream pressure matches the recorded downstream PT, with a
smoothness regularisation term — the same DE + regularisation philosophy as
the original Tkinter optimizer (`gas_moc_smart_optimizer_v5 (1).py`,
`GasMOCOptimizer`).

## 4. The original MOC transient optimizer

`gas_moc_smart_optimizer_v5 (1).py` (see `RUNBOOK.md`) solves the *transient*
problem for gas: valve open/hold/close cycles produce pressure waves
propagating at `B = √(Z·R·T)`; the MOC grid integrates
`C±` characteristic equations per time step, with:

- Courant number β ∈ [0.5, 0.9] setting `dt = β·dx/B`
- Wave-speed uncertainty factor B_factor ∈ [0.98, 1.02]
- Per-segment diameters, elevation profile, Colebrook-White/Haaland friction
- ISA/IEC N6 valve mass-flow model with K_multiplier
- Upstream BC modes: CONSTANT_P / CLOSED_END / FINITE_TANK

The optimizer fits per-segment diameters to the downstream PT record with a
weighted RMSE cost (transient windows weighted 10–25×, reflected-wave zone up
to 50×) plus smoothness/total-variation regularisation.

## 5. Streamlit dashboard (`app.py`)

**Dual-view layout (40% / 60%)** via `st.columns([2, 3])`:

- **LEFT PANEL — primary focus & controls**: CSV uploaders (upstream PT,
  downstream PT, elevation), manual input expanders (geometry, liquid,
  valve, deposition/optimizer), executive KPI metric cards (upstream PT,
  residual mean/max, downstream PT actual vs estimated, max wax thickness,
  area loss), filtered raw-data sub-tables (SCADA + segment table) with CSV
  export.
- **RIGHT PANEL — secondary context & visuals** in four drill-down tabs:
  1. **Pressure Profiles** — upstream actual vs ideal; downstream actual vs
     estimated; upstream-only residual (shaded to zero).
  2. **Diameter & Deposition Profile** — clean-bore vs deposition diameter
     with the deposit zone shaded (`fill='tonexty'`); optimized 100-segment
     diameter profile with DE diagnostics; per-segment deposit thickness.
  3. **Elevation Profile** — elevation from sea level along the chainage.
  4. **Distribution & Trends** — residual trendline (rolling mean) and
     residual histogram.

The simulation is wrapped in `@st.cache_data` so widget interactions do not
trigger re-runs.

## 6. How the pieces relate

```
Input/                           SCADA PT CSVs, elevation CSV, config JSON
   │
   ├── gas_moc_smart_optimizer_v5 (1).py   (Tkinter, gas transient MOC + DE)
   │        Output: optdias.csv, pressure.csv, result set1.pdf
   │
   └── app.py  ──►  pipeline_hydraulics.py  (steady-state, oil pipeline)
            Input: uploaded CSVs + manual UI inputs
            Output: KPIs, per-node DataFrame, Plotly charts, CSV export
```

## 7. Repository contents

| File | Role |
|---|---|
| `gas_moc_smart_optimizer_v5 (1).py` | Original Tkinter gas MOC optimizer (v4.1) |
| `app.py` | Streamlit dual-view dashboard |
| `pipeline_hydraulics.py` | Headless steady-state hydraulic + optimization engine |
| `RUNBOOK.md` | WSL run instructions for the Tkinter app |
| `DASHBOARD.md` | Streamlit dashboard usage and CSV schemas |
| `Input/` | SCADA PT CSV, elevation CSV, config JSON |
| `Output/` | Optimizer results (optdias.csv, pressure.csv, result set1.pdf) |

## 8. Running

```bash
python -m pip install -r requirements.txt

# Web dashboard (recommended)
streamlit run app.py

# Original desktop app (Tk GUI — see RUNBOOK.md)
python "gas_moc_smart_optimizer_v5 (1).py"
```
