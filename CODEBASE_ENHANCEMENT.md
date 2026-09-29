# Codebase Enhancement Document — Streamlit Dashboard Release

This document describes the enhancements made **on top of the previous
repository state** and records the rationale for each change.

## 1. Baseline (previous repository state)

The original repository contained a single-application codebase:

- `gas_moc_smart_optimizer_v5 (1).py` — a ~2,800-line Tkinter desktop
  application embedding the MOC transient engine, DE optimizer, plotting,
  PDF export, and GUI in one file. Gas use case; run from WSL per
  `RUNBOOK.md`.
- `Input/` — SCADA PT CSV, elevation CSV, config JSON.
- `Output/` — optimizer result CSVs and PDF.
- `requirements.txt` — numpy, pandas, matplotlib, scipy.
- `RUNBOOK.md` — desktop-app run instructions.

**Limitations addressed by this release:**

| Limitation | Consequence |
|---|---|
| Tkinter-only UI | Not usable as a shared/web tool; requires X display / WSLg |
| Engine coupled to GUI | Physics not reusable from scripts, tests, or other UIs |
| Gas-specific transient model | No support for the liquid/oil steady-state use case (dual PT + deposition) |
| No visualization of deposit zone or residual profile | Key diagnostics required manual PDF inspection |
| Results not exportable from UI | Analysts could not reuse outputs downstream |

## 2. Enhancements delivered

### 2.1 New: `pipeline_hydraulics.py` — headless hydraulic engine

A dependency-light, importable engine for the oil-pipeline use case:

- Dataclasses for configuration: `PipelineGeometry`, `LiquidProperties`,
  `ValveOperations`.
- `run_simulation()` — steady-state Darcy–Weisbach (Haaland friction factor)
  over N chainage segments with:
  - per-segment effective (deposition-reduced) diameters,
  - hydrostatic head from the interpolated elevation profile,
  - ISA-style valve pressure drop at the valve chainage,
  - **upstream actual profile** (from the upstream PT through the deposited
    pipe),
  - **upstream ideal profile** (back-calculated from the downstream PT on a
    clean pipe),
  - **residual pressure (upstream only)** = actual − ideal,
  - **downstream estimated pressure** for comparison with the downstream PT.
- `optimize_deposition_diameters()` — Differential-Evolution per-segment
  diameter optimization matching the modelled downstream pressure to the
  recorded downstream PT, with smoothness regularisation (same philosophy as
  `GasMOCOptimizer` in the original app).
- Returns a typed `SimulationResult` (per-node DataFrame + KPI dict) so any
  UI or script can consume it.

### 2.2 New: `app.py` — Streamlit dual-view dashboard

Web implementation of the analysis workflow with the requested 40/60 dual-view
layout:

**LEFT PANEL (40%) — primary focus & controls**
- File uploaders for upstream PT, downstream PT, and elevation CSVs; the
  loaders accept the repo's existing schemas (`time,pressure_bar` and
  `distance_m,elevation_m`) plus `pt_upstream`/`pt_downstream` column
  variants, with BOM handling and a default demo profile when no files are
  loaded.
- Manual inputs grouped in expanders: pipeline geometry (length, diameter,
  wall thickness, roughness, segment count 10–200, valve chainage), liquid
  properties (density, viscosity, bulk modulus, flow rate), valve operations
  (opening %, Cv), deposition intensity and optimizer iterations.
- Executive KPI `st.metric` cards: upstream PT actual, upstream residual
  (mean, max), downstream PT actual with estimated delta, max wax deposition
  with area-loss delta.
- Filtered raw-data sub-tables (SCADA PT tail, per-segment results) and a
  one-click CSV export of segment results.

**RIGHT PANEL (60%) — secondary context & visuals** (four tabs)
1. Pressure Profiles — upstream actual vs ideal; downstream actual vs
   estimated; upstream-only residual shaded to zero.
2. Diameter & Deposition Profile — actual vs deposition diameter with the
   deposit zone shaded via Plotly `fill='tonexty'`; optimized per-segment
   diameter profile with DE evaluation/fit diagnostics; deposit-thickness
   bars.
3. Elevation Profile — sea-level elevation along the chainage.
4. Distribution & Trends — residual trendline (rolling mean, dependency-free)
   and residual histogram.

Cross-cutting: `@st.cache_data`-wrapped simulation so widget interactions do
not re-run the engine or optimizer; `st.set_page_config(layout="wide")`;
consistent `plotly_white` template and card-based `st.container(border=True)`
sections.

### 2.3 New: `DASHBOARD.md` and `PROJECT_EXPLANATION.md`

- `DASHBOARD.md` — dashboard run instructions, layout map, CSV schemas, and
  the engine's Python API with examples.
- `PROJECT_EXPLANATION.md` — full project explanation: problem statement,
  data inputs, physics, both applications (transient MOC + steady-state
  oil), and how the components relate.

### 2.4 Updated: `requirements.txt`

Added the dashboard's two dependencies: `streamlit` and `plotly`. All other
heavy lifting (numpy, pandas, scipy) was already present. The LOWESS
trendline available in plotly express was replaced with a rolling mean to
avoid introducing a `statsmodels` dependency.

## 3. Compatibility and preservation

- **No existing files were modified or removed** except `requirements.txt`
  (additive change). The Tkinter app, `RUNBOOK.md`, `Input/`, and `Output/`
  are untouched and remain fully functional.
- The dashboard engine mirrors the original optimizer's conventions (per-
  segment diameters, elevation interpolation, DE with smoothness
  regularisation) so the two tools remain conceptually aligned.
- CSV loaders were verified against the repository's real input files
  (`B178_set_1_filt500.csv`, `elevation.csv`).

## 4. Verification performed

| Check | Result |
|---|---|
| Engine smoke test (repo elevation data, 100 segments) | KPIs computed; profiles finite |
| DE optimizer convergence | Downstream model 12.30 bar vs actual 12.30 bar (exact match) |
| Full app via `streamlit.testing.v1.AppTest` | No exceptions; all 4 KPI cards, 4 tabs render |
| CSV normalizers vs repo input files | Both schemas parse correctly |
| Import surface | `pipeline_hydraulics` importable without Tk/Streamlit |

## 5. Suggested next steps

- Publish the Streamlit app for team access (e.g. Streamlit Community Cloud).
- Extend the engine with transient (MOC) liquid capability for valve-closure
  surge analysis, reusing `run_gas_moc` patterns.
- Add PT time-series overlay in the dashboard (currently the latest sample
  drives the steady-state boundary condition).
- Add unit tests for `pipeline_hydraulics.py` (friction factor, hydrostatic
  sign conventions, optimizer convergence) as a CI gate.
