# Oil Pipeline Pressure & Deposition Dashboard (Streamlit)

A web implementation of the optimizer workflow in this repository: upstream/downstream
SCADA pressure-transmitter (PT) data, chainage elevation profile, and manual
pipeline/liquid/valve inputs are combined into a steady-state hydraulic
simulation with per-segment wax-deposition diameter optimization.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Layout — dual view (40% / 60%)

**LEFT PANEL — Inputs & Controls**
- CSV uploaders: upstream PT, downstream PT, elevation profile
- Manual inputs: pipeline geometry, liquid properties, valve operations,
  deposition intensity, optimizer settings
- Executive metric summary cards (upstream PT, residual pressure, downstream
  PT actual vs estimated, max wax deposition / area loss)
- Filtered raw data sub-tables (SCADA PT + per-segment results) with CSV export

**RIGHT PANEL — Interactive Analytics (tabs)**
1. **Pressure Profiles** — upstream actual vs ideal (ideal back-calculated from
   the downstream PT), downstream actual vs estimated, and residual pressure
   (upstream only, shaded to zero).
2. **Diameter & Deposition Profile** — actual clean-bore diameter vs
   deposition diameter with the area between them shaded (`fill='tonexty'`),
   optimized 100-segment diameter profile from differential evolution, and
   per-segment deposit thickness bars.
3. **Elevation Profile** — elevation from sea level along the chainage.
4. **Distribution & Trends** — residual trendline (rolling mean) and residual
   distribution histogram.

## CSV formats

Upstream/downstream PT (one file per PT, or a single file with both):

```csv
time,pressure_bar
0,36.26
0.01,36.26
```

A single file containing `pt_upstream` and `pt_downstream` columns is also
accepted. The repo's `Input/B178_set_1_filt500.csv` (`time,pressure_bar`) and
`Input/elevation.csv` (`distance_m,elevation_m`) load directly.

Elevation profile:

```csv
distance_m,elevation_m
0,72.99
2.71,72.99
```

## Engine

`pipeline_hydraulics.py` is a headless, importable engine:

- Steady-state Darcy–Weisbach friction (Haaland friction factor) on
  per-segment effective (deposition-reduced) diameters
- Hydrostatic head from the interpolated elevation profile
- ISA-style valve pressure drop at the valve chainage
- Upstream actual profile marched from the upstream PT; upstream ideal profile
  back-calculated from the downstream PT on the clean pipe; residual = actual − ideal
- Differential-evolution optimization of per-segment effective diameters so the
  modeled downstream pressure matches the downstream PT (mirrors the DE +
  smoothness-regularization cost in `gas_moc_smart_optimizer_v5 (1).py`)

```python
from pipeline_hydraulics import PipelineGeometry, LiquidProperties, ValveOperations, run_simulation
res = run_simulation(p_up_actual_bar=55.2, p_down_actual_bar=12.3,
                     flow_m3s=145/3600,
                     geometry=PipelineGeometry(length_m=8000, diameter_mm=193.7),
                     liquid=LiquidProperties(), valve=ValveOperations(),
                     elevation_df=elev_df)
print(res.kpis)   # dict of KPI values
print(res.df)     # per-node DataFrame: pressures, residual, diameters, wax
```

The Tkinter application (`gas_moc_smart_optimizer_v5 (1).py`) remains the
gas-MOC transient tool described in `RUNBOOK.md`; this dashboard targets the
oil-pipeline steady-state use case.
