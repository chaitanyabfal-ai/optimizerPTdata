"""
Oil Pipeline Hydraulics Engine
Steady-state Darcy-Weisbach pressure profile, hydrostatic head, valve loss,
wax deposition diameter profile, and per-segment diameter optimization.

Conventions:
- Pressures in bar (gauge), elevations in m MSL, chainage in m, diameters in mm.
- Upstream PT gives the actual recorded pressure at chainage 0.
- Downstream PT gives the recorded pressure at the terminal chainage and is
  back-propagated to build the "ideal" upstream profile.
- Residual pressure (upstream only) = actual profile - ideal profile.
- Deposition profile = wax-layer effective diameter per segment.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

G = 9.81
BAR_TO_PA = 1.0e5
MM_TO_M = 1.0e-3


@dataclass
class PipelineGeometry:
    length_m: float = 8000.0
    diameter_mm: float = 193.7
    wall_thickness_mm: float = 8.0
    roughness_mm: float = 0.045
    n_segments: int = 100
    valve_position_m: float = 8000.0
    pt_upstream_m: float = 0.0
    pt_downstream_m: float = 7950.0


@dataclass
class LiquidProperties:
    density_kgm3: float = 860.0
    viscosity_pas: float = 0.035
    bulk_modulus_pa: float = 1.5e9


@dataclass
class ValveOperations:
    opening_pct: float = 100.0
    cv_max: float = 132.0
    downstream_backpressure_bar: float = 1.0


def haaland_friction_factor(re, roughness_m, d_m):
    re = np.asarray(re, dtype=float)
    d_m = np.asarray(d_m, dtype=float)
    f = np.full(re.shape, 0.02)
    laminar = re < 2300.0
    f_lam = np.where(re > 0.0, 64.0 / np.maximum(re, 1.0e-9), 0.02)
    f[laminar] = f_lam[laminar]
    turb = ~laminar
    if np.any(turb):
        rel_rough = roughness_m / d_m[turb]
        inner = (rel_rough / 3.7) ** 1.11 + 6.9 / np.maximum(re[turb], 1.0e-9)
        inv_sqrt_f = -1.8 * np.log10(inner)
        f[turb] = 1.0 / inv_sqrt_f ** 2
    return f


def segment_grid(length_m, n_segments):
    x_nodes = np.linspace(0.0, length_m, n_segments + 1)
    x_segments = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    return x_nodes, x_segments


def interpolate_elevation(x_nodes, elevation_df):
    dist = elevation_df.iloc[:, 0].to_numpy(dtype=float)
    elev = elevation_df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(dist)
    return np.interp(x_nodes, dist[order], elev[order])


def deposition_profile(x_segments, geometry: PipelineGeometry, dep_factor, wax_amp):
    """Wax layer grows toward the cold downstream end and near low points."""
    frac = x_segments / max(geometry.length_m, 1.0e-9)
    shape = 0.15 + 0.10 * np.sin(2.0 * np.pi * frac) + 0.05 * np.cos(6.0 * np.pi * frac)
    thickness_frac = dep_factor * wax_amp * np.clip(shape, 0.0, None)
    return np.clip(thickness_frac, 0.0, 0.45)


def friction_gradient(flow_m3s, d_mm_arr, geometry, liquid, roughness_m):
    """dP/dx in Pa/m for each segment given effective diameters (mm array)."""
    d_m = np.asarray(d_mm_arr, dtype=float) * MM_TO_M
    d_m = np.maximum(d_m, 1.0e-3)
    area = np.pi * d_m ** 2 / 4.0
    velocity = flow_m3s / area
    re = liquid.density_kgm3 * velocity * d_m / max(liquid.viscosity_pas, 1.0e-9)
    f = haaland_friction_factor(re, roughness_m, d_m)
    return f * (1.0 / d_m) * (liquid.density_kgm3 * velocity ** 2 / 2.0)


def valve_pressure_drop(flow_m3s, valve: ValveOperations):
    """Pressure drop across the valve in bar (ISA-style Cv model, incompressible)."""
    if valve.opening_pct <= 0.0 or valve.cv_max <= 0.0:
        return np.inf
    cv_eff = valve.cv_max * (valve.opening_pct / 100.0)
    # Q_m3h = Cv * sqrt(dP_bar) for water-like scaling; correct for density
    q_m3h = flow_m3s * 3600.0
    sg = 1.0
    dp_bar = (q_m3h / (cv_eff * sg)) ** 2
    return dp_bar


@dataclass
class SimulationResult:
    df: pd.DataFrame = None
    kpis: dict = field(default_factory=dict)


def run_simulation(
    p_up_actual_bar,
    p_down_actual_bar,
    flow_m3s,
    geometry: PipelineGeometry,
    liquid: LiquidProperties,
    valve: ValveOperations,
    elevation_df=None,
    deposition_d_mm=None,
):
    """
    Steady-state simulation across n_segments.

    - Upstream actual profile: starts at upstream PT, marches downstream with
      friction (on deposition diameters) + hydrostatic head - valve loss.
    - Upstream ideal profile: back-calculated from downstream PT along the
      frictionless-ideal (clean pipe) path with the same elevation head.
    - Residual = actual - ideal, upstream only.
    """
    n = int(geometry.n_segments)
    x_nodes, x_segments = segment_grid(geometry.length_m, n)
    dx = geometry.length_m / n

    if elevation_df is not None and len(elevation_df) >= 2:
        elev_nodes = interpolate_elevation(x_nodes, elevation_df)
    else:
        elev_nodes = np.zeros(n + 1)
    elev_seg = 0.5 * (elev_nodes[:-1] + elev_nodes[1:])

    nominal_d = geometry.diameter_mm
    if deposition_d_mm is not None:
        dep_d = np.asarray(deposition_d_mm, dtype=float)
        if len(dep_d) == n:
            dep_d = np.concatenate([dep_d, [dep_d[-1]]])
        dep_d = np.clip(dep_d, 5.0, 2.0 * nominal_d)
    else:
        dep_d = np.full(n + 1, nominal_d)

    rho = liquid.density_kgm3
    roughness_m = geometry.roughness_mm * MM_TO_M

    grad = friction_gradient(flow_m3s, dep_d, geometry, liquid, roughness_m)
    dp_friction_seg_bar = grad * dx / BAR_TO_PA

    dz_seg = elev_nodes[1:] - elev_nodes[:-1]
    dp_hydro_seg_bar = rho * G * dz_seg / BAR_TO_PA

    # Upstream actual profile (nodes), deposition-laden pipe
    p_actual = np.zeros(n + 1)
    p_actual[0] = p_up_actual_bar
    for i in range(n):
        p_next = p_actual[i] - dp_friction_seg_bar[i] - dp_hydro_seg_bar[i]
        # valve loss applied at its chainage
        xm = x_nodes[i + 1]
        if geometry.valve_position_m - dx < xm <= geometry.valve_position_m:
            p_next -= valve_pressure_drop(flow_m3s, valve)
        p_actual[i + 1] = max(p_next, 0.0)

    # Ideal profile: clean (nominal) pipe, back-calculated from downstream PT
    grad_ideal = friction_gradient(flow_m3s, np.full(n + 1, nominal_d), geometry, liquid, roughness_m)
    dp_fric_ideal = grad_ideal * dx / BAR_TO_PA
    p_ideal = np.zeros(n + 1)
    p_ideal[-1] = p_down_actual_bar
    for i in range(n - 1, -1, -1):
        p_prev = p_ideal[i + 1] + dp_fric_ideal[i] + dp_hydro_seg_bar[i]
        xm = x_nodes[i + 1]
        if geometry.valve_position_m - dx < xm <= geometry.valve_position_m:
            p_prev += valve_pressure_drop(flow_m3s, valve)
        p_ideal[i] = p_prev

    residual = p_actual - p_ideal

    # Downstream estimated (from upstream actual through the deposition pipe)
    p_down_estimated = p_actual[-1]

    df = pd.DataFrame({
        "chainage_m": x_nodes,
        "elevation_m": elev_nodes,
        "nominal_d_mm": np.full(n + 1, nominal_d),
        "deposition_d_mm": dep_d,
        "wax_thickness_mm": np.maximum(nominal_d - dep_d, 0.0) / 2.0,
        "upstream_actual_bar": p_actual,
        "upstream_ideal_bar": p_ideal,
        "upstream_residual_bar": residual,
    })

    kpis = {
        "p_up_actual_bar": p_up_actual_bar,
        "p_down_actual_bar": p_down_actual_bar,
        "p_down_estimated_bar": p_down_estimated,
        "residual_mean_bar": float(np.mean(residual)),
        "residual_max_bar": float(np.max(residual)),
        "residual_min_bar": float(np.min(residual)),
        "wax_max_mm": float(np.max(df["wax_thickness_mm"])),
        "area_loss_pct": float(
            100.0 * (1.0 - np.mean((dep_d / nominal_d) ** 2))
        ),
        "dp_total_bar": float(p_up_actual_bar - p_down_actual_bar),
    }
    return SimulationResult(df=df, kpis=kpis)


def optimize_deposition_diameters(
    p_up_actual_bar,
    p_down_actual_bar,
    flow_m3s,
    geometry: PipelineGeometry,
    liquid: LiquidProperties,
    valve: ValveOperations,
    elevation_df=None,
    d_range=(0.55, 0.98),
    max_iter=30,
    popsize=12,
    seed=42,
    progress=False,
):
    """
    Optimize per-segment effective diameters so the actual upstream PT profile
    reproduces the downstream PT. Mirrors the repo's DE + smoothness cost.
    """
    n = int(geometry.n_segments)
    nominal_d = geometry.diameter_mm
    base = run_simulation(
        p_up_actual_bar, p_down_actual_bar, flow_m3s,
        geometry, liquid, valve, elevation_df, deposition_d_mm=None,
    )
    target_p_down = p_down_actual_bar
    lambda_smooth = 1.0e-6
    history = []
    eval_state = {"count": 0}

    def cost(mult):
        eval_state["count"] += 1
        d = np.asarray(mult) * nominal_d
        try:
            res = run_simulation(
                p_up_actual_bar, p_down_actual_bar, flow_m3s,
                geometry, liquid, valve, elevation_df, deposition_d_mm=d,
            )
        except Exception:
            return 1.0e10
        p_down_model = res.kpis["p_down_estimated_bar"]
        if not np.isfinite(p_down_model):
            return 1.0e10
        misfit = (p_down_model - target_p_down) ** 2
        smooth = float(np.mean(np.diff(np.asarray(mult)) ** 2))
        return misfit + lambda_smooth * smooth

    bounds = [(d_range[0], d_range[1])] * n
    result = differential_evolution(
        cost, bounds=bounds, maxiter=max_iter, popsize=popsize,
        seed=seed, tol=1.0e-4, polish=True, updating="deferred",
        disp=progress,
    )
    d_opt = np.asarray(result.x) * nominal_d
    res = run_simulation(
        p_up_actual_bar, p_down_actual_bar, flow_m3s,
        geometry, liquid, valve, elevation_df, deposition_d_mm=d_opt,
    )
    history.append({
        "evals": eval_state["count"],
        "cost": float(result.fun),
        "p_down_model_bar": res.kpis["p_down_estimated_bar"],
    })
    return d_opt, res, history
