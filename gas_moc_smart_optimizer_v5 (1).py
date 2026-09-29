#!/usr/bin/env python3
"""
Gas MOC Smart Optimizer — 2-BC-UP Pipeline  v4.0
=================================================

GUI-based smart optimizer — Gas MOC (Wylie & Streeter, Ch. 15).

Changes from v2.0 / v3.x
--------------------------
1. alpha REMOVED — replaced by Courant number beta in [0.5, 0.9]
      dt = beta * dx / B_eff
   beta=1 → exact CFL; beta<1 → sub-Courant (more diffusive, more stable).

2. B_factor in [0.98, 1.02] — wave-speed +/-2% uncertainty tuning
      B_eff = sqrt(Z*R*T) * B_factor

3. MOC area convention fixed per Wylie & Streeter §15-5/§15-7:
   - friction evaluated at FOOT-NODE D & A (i+/-1 interior; N-1 for DS BC)
   - boundary equations use BOUNDARY-NODE area (A[0], A[N])

4. Valve model: 5-phase exponential cycle from GAS_MOC.py
   open -> exp-close -> hold -> exp-open -> open
   ISA/IEC N6 mass-flow model retained; K_multiplier scales Cv_eff.

Features
--------
- Array-diameter MOC simulation (per-segment diameters)
- Elevation profile support (CSV: distance_m, elevation_m)
- Downstream PT data optimization (CSV: time, pressure_bar)
- Courant beta (0.5-0.9) + B_factor (+/-2%) numerical tuning in UI
- 4-phase linear valve cycle: CLOSED -> open ramp -> hold open -> close ramp -> CLOSED
  Params: t_valve_open_start / t_valve_opening / t_valve_hold_open / t_valve_closing
  ISA/IEC N6 flow model + K_multiplier; live timeline preview in UI
- Smart initial diameter profiles: Uniform / Probable
- Differential Evolution + L-BFGS-B optimization

Author: Bharat Flow Analytics
Date  : 2026-03
Version: 4.1 — Gas MOC Smart Optimizer (4-phase linear valve + Courant beta + B_factor)
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import multiprocessing
import os
import sys
import datetime
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.optimize import differential_evolution, minimize, brentq

# Detect logical CPU count for parallel workers
_N_CPU = os.cpu_count() or 1



# ============================================================
#  CORE GAS MOC SIMULATION ENGINE
#  (supports per-segment diameters AND elevation profile)
# ============================================================

def colebrook_white(Re, D_pipe, roughness):
    """Darcy friction factor — Colebrook-White iteration."""
    if Re < 1.0:
        return 0.02
    if Re < 2300.0:
        return 64.0 / Re
    f = 0.02
    for _ in range(60):
        rhs = -2.0 * math.log10(roughness / (3.7 * D_pipe) +
                                 2.51 / (Re * math.sqrt(f)))
        f_new = 1.0 / rhs ** 2
        if abs(f_new - f) < 1.0e-10:
            break
        f = f_new
    return f_new

def haaland(Re, D_pipe, roughness):
    """Darcy friction factor — Haaland explicit approximation.

    Valid range:  Re ≥ 2300,  1e-6 ≤ ε/D ≤ 0.05
    Max error vs Colebrook-White: ~2 %
    No iteration required — single closed-form evaluation.

    Haaland (1983):
        1/√f = -1.8 · log10[ (ε/D / 3.7)^1.11  +  6.9/Re ]
    """
    if Re < 1.0:
        return 0.02
    if Re < 2300.0:
        return 64.0 / Re          # Laminar — Hagen-Poiseuille, exact

    rel_rough = roughness / D_pipe
    inner = (rel_rough / 3.7) ** 1.11 + 6.9 / Re
    inv_sqrt_f = -1.8 * math.log10(inner)
    return 1.0 / inv_sqrt_f ** 2


def run_gas_moc(
    # Geometry
    L=7800.0, dx=50.0,
    D=None,               # scalar or 1-D array length N
    eps=45.0e-6,
    # Elevation profile: array of shape (2, M) = [distance_m, elevation_m]
    # or None for flat pipeline
    elevation_profile=None,
    # Gas properties
    Z_factor=0.996, gamma=1.26, R_gas=424.0, T_K=298.15, mu_dyn=1.1e-5,
    # Pressures (Pa absolute)
    P_upstream=36.5e5, P_atm=1.0e5,
    # Valve geometry / location
    valve_position=7800.0,
    # Valve ISA parameters (ISA/IEC model)
    xT=0.9, Fp=1.0, Cv_max=132.0,
    # Valve timing — 4-phase cycle:
    #   Phase 0: CLOSED           (0 … t_valve_open_start)
    #   Phase 1: opening ramp     (t_valve_open_start … +t_valve_opening)
    #   Phase 2: hold fully open  (+t_valve_hold_open)
    #   Phase 3: closing ramp     (+t_valve_closing)  → CLOSED again
    t_valve_open_start=30.0,   # time valve starts opening (s)
    t_valve_opening=5.0,       # opening ramp duration (s)
    t_valve_hold_open=60.0,    # hold-open duration (s)
    t_valve_closing=5.0,       # closing ramp duration (s)
    K_multiplier=5.0, T_total=200.0,
    # Courant number beta: dx_effective = B_eff * (dt / beta)
    # beta in [0.5, 0.9]; controls numerical stability (replaces alpha)
    beta=0.9,
    # Wave speed uncertainty factor: B_eff = B * B_factor, B_factor in [0.98, 1.02]
    B_factor=1.0,
    # Downstream PT location (for output recording)
    x_pt_m=7750.0,
    # Upstream BC mode
    upstream_bc='CLOSED_END', V_tank=10.0,
    verbose=False,
    g=9.81,
):
    """
    Run Gas-MOC simulation with optional elevation profile.

    elevation_profile: numpy array of shape (M, 2) with columns
                       [distance_m, elevation_m], or None for flat.

    Returns dict with keys:
        time, P_up, P_down_pt, P_valve_upstream, P_valve_downstream,
        P_flare, Q_valve, tau, x_grid, x_segments, D_arr, elevation_arr
    """
    N = int(round(L / dx))
    print(f"Running Gas MOC with {N} segments (dx={dx} m)")
    x_grid = np.linspace(0.0, L, N + 1)

    # ── Build per-node diameter array (length N+1) ───────────────────
    if D is None or np.isscalar(D):
        D_val = float(D) if D is not None else 0.2032
        D_arr = np.full(N + 1, D_val)
    else:
        D_np = np.asarray(D, dtype=float)
        if len(D_np) == N:
            D_arr = np.append(D_np, D_np[-1])
        elif len(D_np) == N + 1:
            D_arr = D_np.copy()
        else:
            D_arr = np.interp(np.arange(N + 1),
                              np.linspace(0, N, len(D_np)), D_np)

    A_arr = np.pi * D_arr ** 2 / 4.0

    # ── Build per-node elevation array (length N+1) ──────────────────
    if elevation_profile is not None:
        elev_np = np.asarray(elevation_profile, dtype=float)
        # elev_np shape: (M, 2)  columns: [dist, elev]
        elev_dist = elev_np[:, 0]
        elev_vals = elev_np[:, 1]
        # Interpolate onto grid nodes
        elevation_arr = np.interp(x_grid, elev_dist, elev_vals)
    else:
        elevation_arr = np.zeros(N + 1)

    # ── Wave speed (isothermal gas, Wylie & Streeter Eq. 15-2) ─────────
    # B_eff = B * B_factor  allows ±2% wave-speed uncertainty tuning
    B_iso = math.sqrt(Z_factor * R_gas * T_K)
    B = B_iso * B_factor        # effective wave speed used in all MOC equations
    print(f"Wave speed B = {B_iso:.2f} m/s  (B_factor={B_factor:.4f}  →  B_eff={B:.2f} m/s)")

    # ── Courant-number based time step ───────────────────────────────
    # Wylie & Streeter §15-5: characteristic slope dx/dt = ±B (Eq. 15-19)
    # Specified-time-interval method: dt = beta * dx / B_eff
    # beta (Courant number) ∈ [0.5, 0.9] — values < 1 add diffusion but
    # increase stability; beta = 1.0 is the exact CFL limit.
    beta = float(np.clip(beta, 0.01, 1.0))
    dt   = beta * dx / B
    print(f"Courant beta = {beta:.3f}  →  dt = {dt:.5f} s  (dx={dx} m)")
    Nt = int(T_total / dt)

    rho_ref = P_upstream / (Z_factor * R_gas * T_K)

    # ── PT recording node ────────────────────────────────────────────
    # CRITICAL: i_pt must NEVER equal N (the downstream valve BC node).
    # p[N] is set directly by valve_residual (ISA BC), not by MOC
    # propagation — it has no sensitivity to pipe diameter D.
    # If x_pt_m >= valve_position, record at the node just upstream (N-1).
    i_pt = int(round(x_pt_m / dx))
    i_pt = min(i_pt, N - 1)   # guard: always at least 1 node upstream of valve
    i_pt = max(i_pt, 0)

    # ISA N6 constant
    N6 = 2.73

    # ── friction helper ──────────────────────────────────────────────
    def local_friction(M_flow, p_node, i_node):
        """Haaland friction factor evaluated at foot node i_node."""
        D_i = D_arr[i_node]
        A_i = A_arr[i_node]
        if abs(M_flow) < 1.0e-12 or p_node < 1.0 or A_i < 1e-10 or D_i < 1e-6:
            return 0.02
        rho_loc = p_node / (Z_factor * R_gas * T_K)
        u_loc   = abs(M_flow) / (rho_loc * A_i)
        if not np.isfinite(u_loc) or u_loc > 1e6:
            return 0.02
        Re = rho_loc * u_loc * D_i / mu_dyn
        if not np.isfinite(Re) or Re < 1.0:
            return 0.02
        return haaland(Re, D_i, eps)

    def friction_force(f, M_flow, p_node, D_i, A_i):
        """
        Wylie §15-5 friction source term: fB²M|M| / (2DA²p)
        Clipped to prevent overflow when D or A is tiny.
        """
        denom = 2.0 * D_i * A_i**2 * max(p_node, 1.0)
        if denom < 1e-30:
            return 0.0
        val = (f * B**2 * M_flow * abs(M_flow)) / denom
        if not np.isfinite(val):
            return 0.0
        return float(np.clip(val, -1e12, 1e12))

    # ── elevation slope helper ───────────────────────────────────────
    def gravity_term(i_node, pres):
        """
        Gravity source term per Wylie & Streeter Eq. 15-9:
          ρg·sin(θ) / B²  →  integrated into C+/C- equations.
        """
        if i_node == 0:
            dz = elevation_arr[1] - elevation_arr[0]
        elif i_node == N:
            dz = elevation_arr[N] - elevation_arr[N - 1]
        else:
            dz = (elevation_arr[i_node + 1] - elevation_arr[i_node - 1]) / 2.0
        sin_theta = dz / dx
        rho_loc   = pres / (Z_factor * R_gas * T_K)
        # return (rho_loc * g * sin_theta) / (B ** 2)
        return (rho_loc * g * sin_theta)

    # ── valve opening fraction — 4-phase linear cycle ───────────────
    #
    #   tau = 0.0  →  fully CLOSED
    #   tau = 1.0  →  fully OPEN
    #
    #   Phase 0 : CLOSED            t < t_valve_open_start
    #   Phase 1 : opening ramp      t_valve_open_start  → _t1_end   (0 → 1)
    #   Phase 2 : hold fully open   _t1_end             → _t2_end   (1)
    #   Phase 3 : closing ramp      _t2_end             → _t3_end   (1 → 0)
    #   Phase 4 : CLOSED            t ≥ _t3_end
    #
    _t0     = float(t_valve_open_start)
    _t1_end = _t0     + float(t_valve_opening)
    _t2_end = _t1_end + float(t_valve_hold_open)
    _t3_end = _t2_end + float(t_valve_closing)

    def valve_tau(t):
        """
        4-phase linear valve cycle.
        0: closed → 1: linear open → 2: hold → 3: linear close → 0: closed
        """
        if t < _t0:
            return 0.0                                        # Phase 0: closed
        elif t < _t1_end:
            return (t - _t0) / (_t1_end - _t0)              # Phase 1: opening
        elif t < _t2_end:
            return 1.0                                        # Phase 2: full open
        elif t < _t3_end:
            return 1.0 - (t - _t2_end) / (_t3_end - _t2_end)  # Phase 3: closing
        else:
            return 0.0                                        # Phase 4: closed

    # ── ISA/IEC N6 mass-flow model ───────────────────────────────────
    def valve_mass_flow(Pu, Pd, tau_v):
        """ISA N6 mass-flow with choked-flow check and K_multiplier."""
        if tau_v <= 0.0 or Pu <= Pd:
            return 0.0
        Fgamma   = gamma / 1.40
        x_choked = Fgamma * xT
        x_actual = (Pu - Pd) / Pu
        if x_actual >= x_choked:
            x_sizing = x_choked
            Y = 2.0 / 3.0
        else:
            x_sizing = x_actual
            Y = 1.0 - x_sizing / (3.0 * Fgamma * xT)
        Cv_eff   = (Cv_max / K_multiplier) * tau_v
        rho_u    = Pu / (Z_factor * R_gas * T_K)
        Pu_kPa   = Pu / 1000.0
        mdot_kgh = N6 * Fp * Cv_eff * Y * math.sqrt(x_sizing * Pu_kPa * rho_u)
        return mdot_kgh / 3600.0

    # ── initial conditions ───────────────────────────────────────────
    p         = np.full(N + 1, P_upstream)
    M_flow    = np.zeros(N + 1)
    p_L_valve = P_upstream
    P_tank    = float(P_upstream)

    # ── time-history storage ─────────────────────────────────────────
    t_hist, P_up_h, P_down_pt_h = [], [], []
    P_Lv_h, P_Rv_h, P_flare_h, Q_v_h, tau_h = [], [], [], [], []

    # ── MOC MAIN TIME LOOP ───────────────────────────────────────────
    # Wylie & Streeter §15-5 (Eqs. 15-25, 15-26):
    #   C+: (1/A)*MA + (1/B)*pA - (1/A)*Mp - (1/B)*pp + dt*(FA+GA) = 0
    #   C-: (1/A)*MB - (1/B)*pB - (1/A)*Mp + (1/B)*pp + dt*(FB+GB) = 0
    # Solving:  Mp = A*(Cp+Cm)/2,  pp = B*(Cp-Cm)/2
    # where:    Cp = (1/A)*MA + (1/B)*pA - dt*(FA+GA)
    #           Cm = (1/A)*MB - (1/B)*pB - dt*(FB+GB)
    # Areas used for friction are the FOOT-NODE areas (D_arr[i±1]),
    # ensuring the friction term is consistent with §15-7 (boundary conditions).
    # Sub-Courant (beta<1): foot-points interpolated between grid nodes.

    frac = 1.0 - beta   # linear interpolation weight toward current node

    def _foot_plus(i):
        """Interpolated C+ foot between nodes i-1 and i."""
        w0, w1 = beta, frac          # w0 weight on i-1, w1 on i
        return (p[i-1]*w0 + p[i]*w1,
                M_flow[i-1]*w0 + M_flow[i]*w1,
                i - 1)

    def _foot_minus(i):
        """Interpolated C- foot between nodes i and i+1."""
        w0, w1 = frac, beta          # w0 weight on i, w1 on i+1
        return (p[i]*w0 + p[i+1]*w1,
                M_flow[i]*w0 + M_flow[i+1]*w1,
                i + 1)

    for n in range(Nt):
        t_now = n * dt
        tau_v = valve_tau(t_now)

        p_new  = p.copy()
        M_new  = M_flow.copy()

        # ── Interior nodes 1 .. N-1 ───────────────────────────────────
        for i in range(1, N):
            A_i = A_arr[i]

            # C+ foot (from left)
            pA, MA, iA = _foot_plus(i)
            D_A = D_arr[iA];  A_A = A_arr[iA]
            fA  = local_friction(MA, pA, iA)
            FA  = friction_force(fA, MA, pA, D_A, A_A)
            GA  = gravity_term(iA, pA)
            Cp  = MA / A_i + pA / B - dt * (FA + GA)

            # C- foot (from right)
            pB, MB, iB = _foot_minus(i)
            D_B = D_arr[iB];  A_B = A_arr[iB]
            fB  = local_friction(MB, pB, iB)
            FB  = friction_force(fB, MB, pB, D_B, A_B)
            GB  = gravity_term(iB, pB)
            Cm  = MB / A_i - pB / B - dt * (FB + GB)

            Mp = A_i * (Cp + Cm) / 2.0
            pp = B   * (Cp - Cm) / 2.0

            # Guard NaN/Inf produced by overflow in extreme D candidates
            M_new[i] = float(Mp) if np.isfinite(Mp) else 0.0
            p_new[i] = float(pp) if np.isfinite(pp) else P_upstream

        # ── Upstream boundary (node 0) — only C- arrives ─────────────
        # Foot interpolated between nodes 0 and 1
        A0  = A_arr[0]
        pB0 = p[0] * frac + p[1] * beta
        MB0 = M_flow[0] * frac + M_flow[1] * beta
        iB0 = 1
        D_B0 = D_arr[iB0];  A_B0 = A_arr[iB0]
        fB0  = local_friction(MB0, pB0, iB0)
        FB0  = friction_force(fB0, MB0, pB0, D_B0, A_B0)
        GB0  = gravity_term(iB0, pB0)
        Cm0  = MB0 / A0 - pB0 / B - dt * (FB0 + GB0)

        if upstream_bc == 'CONSTANT_P':
            # Known pressure → solve Cm0 for M_new[0]
            p_new[0] = P_upstream
            M_new[0] = A0 * (Cm0 + P_upstream / B)
        elif upstream_bc == 'CLOSED_END':
            # M = 0 at wall: 0/A0 - p_wall/B = Cm0 → p_wall = -B*Cm0
            M_new[0] = 0.0
            p_wall   = -B * Cm0
            p_new[0] = float(np.clip(p_wall, 0.3 * P_atm, 2.0 * P_upstream))
        else:  # FINITE_TANK
            M_new[0] = A0 * (Cm0 + P_tank / B)
            M_new[0] = max(0.0, M_new[0])
            P_tank  -= dt * (Z_factor * R_gas * T_K / V_tank) * M_new[0]
            P_tank   = max(P_tank, 0.3 * P_atm)
            p_new[0] = P_tank

        # ── Downstream boundary (node N) — valve — only C+ arrives ───
        # Foot interpolated between nodes N-1 and N.
        # Friction uses foot-node (N-1) D and A; BC equation uses node-N area.
        AN  = A_arr[N]
        pAN = p[N-1] * beta + p[N] * frac
        MAN = M_flow[N-1] * beta + M_flow[N] * frac
        iAN = N - 1
        D_AN = D_arr[iAN];  A_AN = A_arr[iAN]
        fAN  = local_friction(MAN, pAN, iAN)
        FAN  = friction_force(fAN, MAN, pAN, D_AN, A_AN)
        GAN  = gravity_term(iAN, pAN)
        # C+ at boundary node: Cp_N = MAN/A_N + pAN/B - dt*(FAN+GAN)
        Cp_N = MAN / AN + pAN / B - dt * (FAN + GAN)

        if tau_v > 0.0:
            # p_u from C+: p_u = B*(Cp_N - M_v/A_N)
            M_upper = min(rho_ref * B * AN, abs(Cp_N) * AN * 0.999)

            def valve_residual(M_v):
                p_u = B * (Cp_N - M_v / AN)
                if p_u <= P_atm * 0.05:
                    return M_v
                return M_v - valve_mass_flow(p_u, P_atm, tau_v)

            try:
                r_lo = valve_residual(0.0)
                r_hi = valve_residual(M_upper)
                if r_lo * r_hi < 0.0:
                    M_v_sol = brentq(valve_residual, 0.0, M_upper,
                                     xtol=1.0e-8, maxiter=200)
                else:
                    M_v_sol = 0.0 if abs(r_lo) <= abs(r_hi) else M_upper
            except Exception:
                M_v_sol = 0.0

            p_u_sol   = B * (Cp_N - M_v_sol / AN)
            p_u_sol   = max(p_u_sol, P_atm)
            p_L_valve = p_u_sol
            p_new[N]  = p_u_sol
            M_new[N]  = M_v_sol
        else:
            # Valve closed: M_N = 0 → p_N = B*Cp_N
            p_wall    = B * Cp_N
            p_L_valve = float(np.clip(p_wall, P_atm, 2.0 * P_upstream))
            p_new[N]  = p_L_valve
            M_new[N]  = 0.0

        p_new  = np.clip(p_new, 0.3 * P_atm, 2.0 * P_upstream)
        p[:]   = p_new
        M_flow[:] = M_new


        t_hist.append(t_now)
        P_up_h.append(p[0] / 1.0e5)
        P_down_pt_h.append(p[i_pt] / 1.0e5)
        P_Lv_h.append(p_L_valve / 1.0e5)
        P_Rv_h.append(P_atm / 1.0e5)
        P_flare_h.append(p[N] / 1.0e5)
        rho_v = p[N] / (Z_factor * R_gas * T_K)
        Q_v_h.append(M_flow[N] / rho_v if rho_v > 1e-3 else 0.0)
        tau_h.append(tau_v)

    x_seg = 0.5 * (x_grid[:-1] + x_grid[1:])
    return {
        'time':               np.array(t_hist),
        'P_up':               np.array(P_up_h),
        'P_down_pt':          np.array(P_down_pt_h),
        'P_valve_upstream':   np.array(P_Lv_h),
        'P_valve_downstream': np.array(P_Rv_h),
        'P_flare':            np.array(P_flare_h),
        'Q_valve':            np.array(Q_v_h),
        'tau':                np.array(tau_h),
        'x_grid':             x_grid,
        'x_segments':         x_seg,
        'D_arr':              D_arr,
        'elevation_arr':      elevation_arr,
    }



# ============================================================
#  PICKLABLE WORKER — module-level so multiprocessing can pickle it
# ============================================================

class _MOCWorkerParams:
    """
    Immutable container of all run_gas_moc keyword arguments.
    Stored at module level so multiprocessing workers can unpickle it
    (nested closures inside run_gas_moc cannot be pickled).
    """
    __slots__ = ('kwargs', 'D_nom', 'w', 't_pt', 'p_pt',
                 'lambda_smooth', 'lambda_tv', 'D_initial')

    def __init__(self, base_params, D_nom, t_pt, p_pt, w,
                 D_initial, lambda_smooth, lambda_tv):
        self.kwargs        = base_params      # dict – all run_gas_moc args except D
        self.D_nom         = float(D_nom)
        self.t_pt          = np.asarray(t_pt,  dtype=float)
        self.p_pt          = np.asarray(p_pt,  dtype=float)
        self.w             = np.asarray(w,     dtype=float)
        self.D_initial     = np.asarray(D_initial, dtype=float)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_tv     = float(lambda_tv)


# Module-level reference — set in the MAIN process before fork (Linux),
# OR injected via pool initializer (Windows/macOS spawn).
_WORKER_PARAMS: '_MOCWorkerParams | None' = None


def _worker_initializer(wp: '_MOCWorkerParams'):
    """Pool initializer: runs once in each child process on spawn.
    Injects _WORKER_PARAMS into the child's global namespace so that
    _moc_worker_cost can read it.  Required on Windows / macOS where
    multiprocessing uses 'spawn' (not 'fork') — the child starts with a
    fresh interpreter and the module-level assignment in the parent is NOT
    inherited.
    """
    global _WORKER_PARAMS
    _WORKER_PARAMS = wp


def _moc_worker_cost(D_multipliers):
    """
    Module-level cost function — picklable by multiprocessing.
    Called once per candidate solution by each worker process.
    """
    wp = _WORKER_PARAMS
    if wp is None:
        return 1e10

    D_array = np.asarray(D_multipliers, dtype=float) * wp.D_nom
    if np.any(D_array < 0.01) or np.any(D_array > 2.0 * wp.D_nom):
        return 1e10

    params        = dict(wp.kwargs)      # shallow copy is enough
    params['D']   = D_array
    params['verbose'] = False

    try:
        res = run_gas_moc(**params)
        if np.any(~np.isfinite(res['P_down_pt'])):
            return 1e10

        p_model   = np.interp(wp.t_pt, res['time'], res['P_down_pt'])
        residuals = p_model - wp.p_pt
        misfit    = float(np.sqrt(np.mean(wp.w * residuals ** 2)))

        smooth   = float(np.mean(np.diff(D_array) ** 2))
        tot_var  = float(np.mean((D_array - wp.D_initial) ** 2))

        lam_s = min(wp.lambda_smooth * max(misfit, 1e-9) / max(smooth,   1e-15),
                    1e-3 * misfit)
        lam_t = min(wp.lambda_tv     * max(misfit, 1e-9) / max(tot_var,  1e-15),
                    1e-3 * misfit)

        return misfit + lam_s * smooth + lam_t * tot_var

    except Exception:
        return 1e10


# ============================================================
#  DIAMETER OPTIMIZER ENGINE
# ============================================================

class GasMOCOptimizer:
    """
    Optimizer — fits per-segment pipe diameters to downstream PT data.

    Key improvements:
    - Cost function uses pure RMSE (bar) with adaptive transient weighting
    - Regularisation scaled relative to misfit magnitude (not absolute)
    - Bounds defined as absolute D multiplier range, symmetrically from initial
    - DE with best1bin + polish (L-BFGS-B) for global+local refinement
    - Parallel workers auto-detected
    - Warm-start: initial profile seeded into DE population
    - Full history including all evaluations, not just improvements
    """

    def __init__(self, t_pt, p_pt, base_params, D_nom, n_segments,
                 D_initial_profile=None):
        self.t_pt        = np.asarray(t_pt,  dtype=float)
        self.p_pt        = np.asarray(p_pt,  dtype=float)
        self.base_params = base_params
        self.D_nom       = float(D_nom)
        self.n_segments  = n_segments

        if D_initial_profile is not None and len(D_initial_profile) == n_segments:
            self.D_initial = np.asarray(D_initial_profile, dtype=float)
        else:
            self.D_initial = np.full(n_segments, D_nom)

        # Regularisation weights — kept very small; misfit dominates
        self.lambda_smooth = 1e-6
        self.lambda_tv     = 1e-7

        self.eval_count   = 0
        self.best_cost    = np.inf
        self.best_D_array = self.D_initial.copy()
        self.history      = []
        self.running      = False
        self.stop_requested = False
        self.progress_callback = None

        # Build adaptive transient weight mask once
        self._build_weight_mask()

    # ── Adaptive weight mask ────────────────────────────────────────── for b178 - set 1, set 3
    def _build_weight_mask(self):
        """
        Weight the transient window (valve open → 2× close time) 10× higher.
        Falls back gracefully if timing params missing.
        """
        t  = self.t_pt
        bp = self.base_params
        t0   = bp.get('t_valve_open_start', 0.0)
        dop  = bp.get('t_valve_opening',    5.0)
        dhld = bp.get('t_valve_hold_open',  60.0)
        dcl  = bp.get('t_valve_closing',    5.0)
        t_end_transient = t0 + dop + dhld + 2.0 * dcl  # cover closure + reflection

        self.w = np.ones(len(t), dtype=float)
        mask   = (t >= t0) & (t <= t_end_transient)
        if mask.sum() > 0:
            self.w[mask] = 20.0
        # Extra weight on the first 10 % of the transient (steepest gradient)
        t_peak = t0 + dop + 0.1 * dhld
        peak_mask = (t >= t0) & (t <= t_peak)
        if peak_mask.sum() > 0:
            self.w[peak_mask] = 25.0


#   mask for b179 - set 3
# # ── Adaptive weight mask ──────────────────────────────────────────
#     def _build_weight_mask(self):
#         """
#         Physics-driven weight mask for the cost function.

#         Eight zones are identified from the pressure record using only
#         the valve timing and the acoustic wave round-trip time (2L/B).
#         Each zone receives a weight proportional to its information
#         content for the pipe-diameter optimisation problem.

#         The single most important decision is the branch on t_wave_back:

#             t_wave_back = t3 + 2*L/B

#         If t_wave_back > T_total  (wave arrives AFTER the simulation ends)
#             → zones after valve closure CANNOT be modelled → w = 0
#             → this is the set_2 case (T=60s, wave at 79.1s)

#         If t_wave_back <= T_total (wave IS captured in the simulation)
#             → zone G (wave peak) gets the highest weight w=50
#             → this is the set_3 case (T=105s, wave at 79.1s)

#         Weight derivation basis (from set_3 data analysis):
#         ─────────────────────────────────────────────────────
#         Zone  Time          SNR    N       w     Why
#         A     0 – t0        1.0×   500     1     Pure sensor noise; no physics signal
#         B     t0 – t1       5.0×   300     30    Steepest dp/dt; encodes Cv_eff + D
#                                                 Few samples → needs high weight to compete
#         C     t1 – t2       2.6×   2500    8     Steady-state Darcy friction → encodes D, ε
#                                                 Many samples already well-represented
#         D     t2 – t3       0.04×  100     3     SNR below noise floor; 1s window; low signal
#         E     t3 – 42s      28×    800     20    Joukowski initial rise; encodes pipe inertia
#         F     42s – t_wb    15×    3710    15    Wave propagation; 35% of samples → moderate w
#         G     t_wb – +6s    1.5×*  600     50    *SNR is misleading: diagnostic info is in
#                                                 TIMING (t_arrival = t3 + 2L/B pins L/B)
#                                                 and AMPLITUDE → highest weight in record
#         H     +6s – end     8.3×   1991    12    Settling → pipe volume + compressibility
#         """
#         t  = self.t_pt
#         bp = self.base_params

#         # ── Read valve timing from config ────────────────────────────
#         t0   = bp.get('t_valve_open_start', 5.0)
#         t_op = bp.get('t_valve_opening',    3.0)
#         t_hd = bp.get('t_valve_hold_open',  25.0)
#         t_cl = bp.get('t_valve_closing',    1.0)
#         T    = bp.get('T_total',            105.0)
#         L    = bp.get('L',                  8000.0)
#         R    = bp.get('R_gas',              424.0)
#         Z    = bp.get('Z_factor',           0.996)
#         Tk   = bp.get('T_celsius',          25.0) + 273.15

#         # ── Key time boundaries ──────────────────────────────────────
#         t1 = t0 + t_op          # valve fully open
#         t2 = t1 + t_hd          # valve starts closing
#         t3 = t2 + t_cl          # valve fully closed

#         B           = math.sqrt(Z * R * Tk)          # isothermal wave speed [m/s]
#         t_wave_back = t3 + 2.0 * L / B              # reflected wave arrives at PT [s]
#         t_wave_end  = t_wave_back + 6.0             # wave peak window ends [s]

#         # ── Base weight = 1 everywhere (Zone A) ─────────────────────
#         self.w = np.ones(len(t), dtype=float)

#         # ── Zone B: valve opening ramp ───────────────────────────────
#         # dp/dt here is directly sensitive to Cv_eff and pipe diameter D.
#         # Only 300 samples (2.9% of record) so high weight is needed.
#         self.w[(t >= t0) & (t < t1)] = 30.0

#         # ── Zone C: open-valve plateau ───────────────────────────────
#         # Steady-state flow → Darcy-Weisbach friction → encodes D and ε.
#         # 2500 samples already give good statistical representation.
#         self.w[(t >= t1) & (t < t2)] = 8.0

#         # ── Zone D: closing ramp ─────────────────────────────────────
#         # Only 1 second, SNR ≈ 0.04× (below sensor noise floor).
#         # Negligible information content; keep it very low.
#         self.w[(t >= t2) & (t < t3)] = 3.0

#         if t_wave_back <= T:
#             # ════════════════════════════════════════════════════════
#             # WAVE IS INSIDE THE SIMULATION WINDOW
#             # (set_3 scenario: T_total=105s, wave arrives at 79.1s)
#             # All post-close zones are physically modelable → use them.
#             # ════════════════════════════════════════════════════════

#             # Zone E: Joukowski initial pressure rise (34–42s)
#             # Sudden valve closure converts kinetic energy to pressure.
#             # High SNR (28×) → strong signal, encodes pipe inertia.
#             self.w[(t >= t3)          & (t < 42.0)]         = 20.0

#             # Zone F: wave propagating back toward closed upstream end (42–79s)
#             # Pressure rises as the rarefaction wave travels 8000m and reflects.
#             # High sample count (3710) already gives good representation.
#             self.w[(t >= 42.0)        & (t < t_wave_back)]  = 15.0

#             # Zone G: reflected wave arrival at PT sensor (79–85s)
#             # *** HIGHEST WEIGHT IN THE ENTIRE RECORD ***
#             # The TIMING of arrival pins t_arrival = t3 + 2L/B
#             # → directly encodes L/B = pipe_length / wave_speed.
#             # The AMPLITUDE encodes wave attenuation → friction → D.
#             # Only 600 samples but extremely high diagnostic value.
#             self.w[(t >= t_wave_back) & (t < t_wave_end)]   = 50.0

#             # Zone H: slow settling toward new equilibrium (85–105s)
#             # Encodes total pipe volume and gas compressibility.
#             # Moderate information; 19% of samples.
#             self.w[t >= t_wave_end]                          = 12.0

#         else:
#             # ════════════════════════════════════════════════════════
#             # WAVE IS OUTSIDE THE SIMULATION WINDOW
#             # (set_2 scenario: T_total=60s, wave arrives at 79.1s)
#             # The model physically cannot reproduce the post-close rise.
#             # Including it in the cost drives the optimizer toward
#             # physically impossible solutions → mask it out entirely.
#             # ════════════════════════════════════════════════════════
#             self.w[t >= t3] = 0.0


    # ── Cost function ─────────────────────────────────────────────────
    def cost_function(self, D_multipliers):
        if self.stop_requested:
            raise StopIteration

        self.eval_count += 1
        D_array = np.asarray(D_multipliers, dtype=float) * self.D_nom

        # Hard physical bounds — reject silently
        if np.any(D_array < 0.01) or np.any(D_array > 2.0 * self.D_nom):
            return 1e10

        params = self.base_params.copy()
        params['D']       = D_array
        params['verbose'] = False

        try:
            res = run_gas_moc(**params)

            if np.any(~np.isfinite(res['P_down_pt'])):
                return 1e10

            # Interpolate model onto PT time base
            p_interp = np.interp(self.t_pt, res['time'], res['P_down_pt'])
            residuals = p_interp - self.p_pt          # bar

            # Weighted RMSE (bar) — primary objective
            w_res2  = self.w * residuals ** 2
            misfit  = float(np.sqrt(np.mean(w_res2)))   # weighted RMSE [bar]
            rmse_uw = float(np.sqrt(np.mean(residuals ** 2))) * 1000.0  # mbar

            # Weak regularisation — scale relative to misfit so it never dominates
            smoothness = float(np.mean(np.diff(D_array) ** 2))
            total_var  = float(np.mean((D_array - self.D_initial) ** 2))

            # Scale lambdas so regularisation ≤ 1% of misfit
            # lam_s = self.lambda_smooth * max(misfit, 1e-6) / max(smoothness, 1e-12)
            # lam_t = self.lambda_tv     * max(misfit, 1e-6) / max(total_var,  1e-12)
            # lam_s = min(lam_s, 1e-3 * misfit)
            # lam_t = min(lam_t, 1e-3 * misfit)

            # cost_total = misfit + lam_s * smoothness + lam_t * total_var

            cost_total = (misfit
                          + self.lambda_smooth * smoothness
                          + self.lambda_tv     * total_var)

            improved = False
            if misfit < self.best_cost:
                self.best_cost    = misfit
                self.best_D_array = D_array.copy()
                improved = True

            rec = {
                'eval':        self.eval_count,
                'D_mean':      float(D_array.mean()),
                'D_std':       float(D_array.std()),
                'D_min':       float(D_array.min()),
                'D_max':       float(D_array.max()),
                'cost':        misfit,
                'cost_total':  cost_total,
                'smoothness':  smoothness,
                'total_var':   total_var,
                'rmse_mbar':   rmse_uw,
                'improved':    improved,
            }
            self.history.append(rec)

            if self.progress_callback:
                self.progress_callback({
                    'eval':     self.eval_count,
                    'D_mean':   D_array.mean(),
                    'D_std':    D_array.std(),
                    'rmse':     rmse_uw,
                    'improved': improved,
                    'cost':     misfit,
                })

            return cost_total

        except StopIteration:
            raise
        except Exception:
            if self.stop_requested:
                raise StopIteration
            return 1e10

    # ── Optimization entry point ──────────────────────────────────────
    def optimize(self, method='differential_evolution',
                 D_range=(0.80, 1.05), max_iter=100, workers=1, **kwargs):
        self.running        = True
        self.stop_requested = False
        self.eval_count     = 0
        self.history        = []
        self.best_cost      = np.inf
        self.best_D_array   = self.D_initial.copy()

        # Pass workers into kwargs so _optimize_de can pick it up
        kwargs['workers'] = int(workers)

        try:
            if method == 'differential_evolution':
                return self._optimize_de(D_range, max_iter, **kwargs)
            elif method == 'lbfgsb':
                return self._optimize_lbfgsb(D_range, max_iter)
            elif method == 'de_then_lbfgsb':
                self._optimize_de(D_range, max_iter, **kwargs)
                return self._optimize_lbfgsb(D_range, max(5, max_iter // 5))
            else:
                raise ValueError(f"Unknown method: {method}")
        except StopIteration:
            return None
        finally:
            self.running = False

    def _optimize_de(self, D_range, max_iter, **kwargs):
        """
        Differential Evolution with optional parallel workers.

        workers = 1  → serial, progress_callback active every eval
        workers > 1  → multiprocessing pool; each generation parallelised;
                        progress_callback fires after each generation via
                        the DE 'callback' argument (xk = best vector so far).

        scipy DE requires the cost function to be picklable when workers > 1.
        We route through the module-level _moc_worker_cost() which reads a
        module-global _WORKER_PARAMS object set via pool initializer —
        this guarantees child processes on Windows/macOS (spawn) also receive
        the params (module-level assignment alone is NOT inherited on spawn).
        """
        global _WORKER_PARAMS

        lo = float(np.clip(D_range[0], 0.05, 0.999))
        hi = float(np.clip(D_range[1], lo + 0.01, 1.0))

        x0       = self.D_initial / self.D_nom
        bounds   = [(lo, hi)] * self.n_segments
        popsize  = max(kwargs.get('popsize', 15), 5)
        n_workers = int(kwargs.get('workers', 1))

        # Build seeded initial population (warm-start)
        rng        = np.random.default_rng(42)
        n_pop      = popsize * self.n_segments
        lhs        = rng.uniform(lo, hi, size=(max(n_pop - 1, 1), self.n_segments))
        x0_clipped = np.clip(x0, lo, hi)
        init_pop   = np.vstack([x0_clipped[np.newaxis, :], lhs])

        # Per-generation callback — fires in main process after each generation
        # regardless of worker count.  Used to push progress to GUI during
        # parallel runs (where cost_function is never called in main process).
        _gen_counter = [0]

        def _de_generation_callback(xk, convergence=None):
            """Called by scipy DE after every generation with the best vector."""
            if self.stop_requested:
                return True   # returning True signals DE to stop
            _gen_counter[0] += 1
            D_arr = np.asarray(xk, dtype=float) * self.D_nom
            # Compute cost quickly via instance (updates history + best_D_array)
            cost = self.cost_function(xk)
            return False   # False = keep going

        if n_workers > 1:
            # ── Parallel path ────────────────────────────────────────────
            # Build the shared params container.
            wp = _MOCWorkerParams(
                base_params    = self.base_params,
                D_nom          = self.D_nom,
                t_pt           = self.t_pt,
                p_pt           = self.p_pt,
                w              = self.w,
                D_initial      = self.D_initial,
                lambda_smooth  = self.lambda_smooth,
                lambda_tv      = self.lambda_tv,
            )
            # Also set the module-level slot (works on Linux fork).
            _WORKER_PARAMS = wp

            # scipy DE accepts workers as an int OR as a map-like callable.
            # By passing our own Pool (with initializer) as a map callable,
            # we guarantee every child process receives _WORKER_PARAMS via
            # _worker_initializer — this is the only reliable path on
            # Windows / macOS where 'spawn' is used (children start fresh).
            import multiprocessing as _mp
            _pool = _mp.Pool(
                processes   = n_workers,
                initializer = _worker_initializer,
                initargs    = (wp,),
            )
            cost_fn      = _moc_worker_cost
            workers_arg  = _pool.map   # pass pool.map as the 'workers' callable
        else:
            # ── Serial path — use instance cost_function (has progress CB) ─
            _WORKER_PARAMS = None
            cost_fn      = self.cost_function
            workers_arg  = 1
            _pool        = None

        try:
            result = differential_evolution(
                cost_fn,
                bounds        = bounds,
                strategy      = kwargs.get('strategy', 'best1bin'),
                maxiter       = max_iter,
                popsize       = popsize,
                tol           = kwargs.get('tol', 1e-3),
                mutation      = kwargs.get('mutation', (0.4, 1.2)),
                recombination = kwargs.get('recombination', 0.85),
                workers       = workers_arg,           # FIX: pool.map callable OR 1
                updating      = 'deferred',            # required when workers != 1
                disp          = False,
                polish        = False,
                init          = init_pop,
                atol          = 0,
                seed          = 42,
                callback      = _de_generation_callback if n_workers > 1 else None,
            )
        finally:
            if _pool is not None:
                _pool.close()
                _pool.join()

        # For the serial path the instance cost_function was called for every eval
        # and best_D_array is already up to date.
        # For the parallel path the generation callback already called
        # self.cost_function(result.x) once per generation, so best_D_array
        # reflects the final best.  One extra call here is harmless and ensures
        # the very last polish result is captured.
        if n_workers > 1 and result is not None:
            self.cost_function(result.x)

        _WORKER_PARAMS = None   # clear global slot after run
        return result

    def _optimize_lbfgsb(self, D_range, max_iter):
        """L-BFGS-B from best known solution (warm-start)."""
        lo = float(np.clip(D_range[0], 0.05, 0.999))
        hi = float(np.clip(D_range[1], lo + 0.01, 2.0))

        # Start from best-known solution so far (warm-start)
        x0     = self.best_D_array / self.D_nom
        x0     = np.clip(x0, lo, hi)
        bounds = [(lo, hi)] * self.n_segments

        return minimize(
            self.cost_function, x0=x0,
            method  = 'L-BFGS-B',
            bounds  = bounds,
            options = {
                'maxiter': max_iter,
                'ftol':    1e-3,
                'gtol':    1e-3,
                'eps':     1e-4,   # finite-difference step for gradient
                'disp':    False,
            },
        )

    def stop(self):
        self.stop_requested = True


# ============================================================
#  GUI APPLICATION
# ============================================================

class GasMOCOptimizerGUI:
    """Main GUI — Gas MOC Smart Optimizer for 2-BC-UP pipeline."""

    def __init__(self, root):
        self.root = root
        self.root.title("Gas MOC Optimizer")
        self.root.geometry("1600x1000")

        # State
        self.pt_data = None
        self.t_pt = None
        self.p_pt = None
        self.elevation_profile = None   # numpy array (M, 2) or None
        self.optimizer = None
        self.opt_thread = None
        self.result_data = None
        self.initial_simulation = None
        self.n_segments = 0
        self.D_initial_profile = None
        self.entries = {}

        self._create_menu()
        self._create_main_layout()
        self._update_segment_count()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------
    # MENU
    # ----------------------------------------------------------
    def _create_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        fm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Load PT Data",             command=self._load_pt_data)
        fm.add_command(label="Load Elevation Profile",   command=self._load_elevation)
        fm.add_command(label="Save Configuration",       command=self._save_config)
        fm.add_command(label="Load Configuration",       command=self._load_config)
        fm.add_separator()
        fm.add_command(label="Export Results",           command=self._export_results)
        fm.add_command(label="Export Segment Diameters", command=self._export_segment_diameters)
        fm.add_command(label="Generate PDF Report",      command=self._generate_pdf)
        fm.add_separator()
        fm.add_command(label="Exit",                     command=self._on_close)

        hm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Help", menu=hm)
        hm.add_command(label="User Guide", command=self._show_help)
        hm.add_command(label="About",      command=self._show_about)

    # ----------------------------------------------------------
    # MAIN LAYOUT
    # ----------------------------------------------------------
    def _create_main_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(paned, width=480)
        paned.add(left, weight=0)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._create_left_panel(left)
        self._create_right_panel(right)

    # ----------------------------------------------------------
    # LEFT PANEL
    # ----------------------------------------------------------
    def _create_left_panel(self, parent):
        canvas = tk.Canvas(parent)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)


        # ===== PT DATA =====
        pt_frame = ttk.LabelFrame(sf, text="Downstream PT Data", padding=8)
        pt_frame.pack(fill=tk.X, padx=5, pady=4)

        ttk.Button(pt_frame, text="Load PT CSV (time, pressure_bar)",
                   command=self._load_pt_data).pack(fill=tk.X, pady=2)
        self.pt_label = ttk.Label(pt_frame, text="No data loaded", foreground="red")
        self.pt_label.pack(pady=2)

        # ===== ELEVATION PROFILE =====
        elev_frame = ttk.LabelFrame(sf, text="Elevation Profile (optional)", padding=8)
        elev_frame.pack(fill=tk.X, padx=5, pady=4)

        ttk.Button(elev_frame, text="Load Elevation CSV (distance_m, elevation_m)",
                   command=self._load_elevation).pack(fill=tk.X, pady=2)
        self.elev_label = ttk.Label(elev_frame,
                                    text="No elevation loaded — flat pipeline assumed",
                                    foreground="gray")
        self.elev_label.pack(pady=2)

        ttk.Label(elev_frame,
                  text="CSV columns: distance_m  elevation_m",
                  foreground="gray", font=("Arial", 8)).pack()

        ttk.Label(elev_frame,
                  text="Gravity (m/s²):").pack(anchor=tk.W, padx=4)
        e_g = ttk.Entry(elev_frame, width=12)
        e_g.insert(0, "9.81")
        e_g.pack(anchor=tk.W, padx=4, pady=2)
        self.entries['g'] = e_g

        # ===== PIPELINE GEOMETRY =====
        geom = ttk.LabelFrame(sf, text="Pipeline Geometry", padding=8)
        geom.pack(fill=tk.X, padx=5, pady=4)

        row = 0
        geom_params = [
            ("L",            "Pipe Length (m)",                 "8000.0"),
            ("dx",           "Grid Spacing (m)",                "200.0"),
            ("D_nom",        "Pipe Diameter (m)",               "0.1937"),
            ("eps_mm",       "Roughness (mm)",                  "0.045"),
            ("x_pt_m",       "PT Location (m)",                 "7600.0"),
            ("valve_pos_m",  "Valve Location (m)  [= pipe end]",     "8000.0"),
        ]
        for key, lbl, default in geom_params:
            ttk.Label(geom, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2, padx=2)
            e = ttk.Entry(geom, width=14)
            e.grid(row=row, column=1, pady=2)
            e.insert(0, default)
            self.entries[key] = e
            row += 1

        # # Live PT position warning
        # self._pt_warn_lbl = ttk.Label(geom, text="", foreground="red",
        #                                font=("Arial", 7, "bold"), wraplength=280)
        # self._pt_warn_lbl.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=4)
        # row += 1

        # for key in ("x_pt_m", "valve_pos_m", "L", "dx"):
        #     self.entries[key].bind("<KeyRelease>", self._validate_pt_position)
        #     self.entries[key].bind("<FocusOut>",   self._validate_pt_position)

        ttk.Label(geom, text="Optimizable Segments:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.seg_count_lbl = ttk.Label(geom, text="0", foreground="blue",
                                        font=("Arial", 10, "bold"))
        self.seg_count_lbl.grid(row=row, column=1, pady=2)
        self.entries['L'].bind('<FocusOut>',  self._update_segment_count)
        self.entries['dx'].bind('<FocusOut>', self._update_segment_count)

        # ===== GAS PROPERTIES =====
        gas = ttk.LabelFrame(sf, text="Gas Properties", padding=8)
        gas.pack(fill=tk.X, padx=5, pady=4)

        row = 0
        gas_params = [
            ("T_celsius",   "Temperature (°C)",               "25.0"),
            ("mu_dyn",      "Dynamic Viscosity µ (Pa·s)",      "1.1e-5"),
            ("R_gas",       "Specific Gas Constant R (J/kg·K)","424.0"),
            ("Z_factor",    "Compressibility Z",               "0.996"),
            ("gamma",       "Heat Ratio γ",                    "1.26"),
        ]
        for key, lbl, default in gas_params:
            ttk.Label(gas, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2, padx=2)
            e = ttk.Entry(gas, width=14)
            e.grid(row=row, column=1, pady=2)
            e.insert(0, default)
            self.entries[key] = e
            row += 1

        # ===== PIPELINE INITIAL PRESSURE =====
        pres = ttk.LabelFrame(sf, text="Pipeline Initial Pressure", padding=8)
        pres.pack(fill=tk.X, padx=5, pady=4)

        row = 0
        pres_params = [
            ("P_upstream_bar",  "Upstream Pressure (bar)",   "8.098"),
            ("P_atm_bar",       "Downstream Pressure (bar)", "1.0"),
        ]
        for key, lbl, default in pres_params:
            ttk.Label(pres, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2, padx=2)
            e = ttk.Entry(pres, width=14)
            e.grid(row=row, column=1, pady=2)
            e.insert(0, default)
            self.entries[key] = e
            row += 1

        ttk.Label(pres, text="(P_upstream → upstream BC,  P_downstream → P_atm)",
                  foreground="gray", font=("Arial", 8)).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        row += 1
        ttk.Label(pres, text="Upstream BC Mode:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.upstream_bc_var = tk.StringVar(value="CLOSED_END")
        bc_combo = ttk.Combobox(pres, textvariable=self.upstream_bc_var, width=12,
                                values=["CONSTANT_P", "CLOSED_END", "FINITE_TANK"],
                                state="readonly")
        bc_combo.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(pres, text="V_tank (m³, FINITE_TANK only):").grid(
            row=row, column=0, sticky=tk.W, pady=2, padx=2)
        e = ttk.Entry(pres, width=14); e.insert(0, "10.0")
        e.grid(row=row, column=1, pady=2)
        self.entries["V_tank"] = e

        # ===== 2" BALL VALVE INPUTS =====
        valve = ttk.LabelFrame(sf, text='2" Ball Valve Parameters (ISA/IEC)', padding=8)
        valve.pack(fill=tk.X, padx=5, pady=4)

        row = 0
        # ISA/IEC flow model coefficients
        for key, lbl, default in [
            ("xT",           "Terminal Pressure-Drop Ratio xT",   "0.9"),
            ("Fp",           "Piping Geometry Factor Fp",          "1.0"),
            ("Cv_max",       "Cv_max  (rated full-open Cv)",       "500.0"),
            ("K_multiplier", "K Multiplier  (valve resistance ×)", "5.0"),
        ]:
            ttk.Label(valve, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2, padx=2)
            e = ttk.Entry(valve, width=14); e.insert(0, default)
            e.grid(row=row, column=1, pady=2)
            self.entries[key] = e
            row += 1

        ttk.Separator(valve, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        # 4-phase cycle header with small diagram
        ttk.Label(valve,
                  text="── Valve Cycle ──",
                  foreground="navy", font=("Arial", 9, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(2, 0))
        row += 1
        ttk.Label(valve,
                  text="CLOSED ─► opening ─► OPEN (hold) ─► closing ─► CLOSED",
                  foreground="gray", font=("Arial", 7, "italic")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=4, pady=(0, 4))
        row += 1

        for key, lbl, default, tip in [
            ("t_valve_open_start", "Valve open-start time  t₀  (s)",
             "30.0",  "Time at which valve begins to open  [Phase 0→1]"),
            ("t_valve_opening",    "Opening ramp duration  Δt_open  (s)",
             "5.0",   "Time to go from fully closed to fully open  [Phase 1]"),
            ("t_valve_hold_open",  "Hold-open duration  Δt_hold  (s)",
             "60.0",  "Time valve stays at 100% open  [Phase 2]"),
            ("t_valve_closing",    "Closing ramp duration  Δt_close  (s)",
             "5.0",   "Time to go from fully open back to fully closed  [Phase 3]"),
            ("T_total",            "Total simulation time  T_total  (s)",
             "200.0", "Must be > t₀ + Δt_open + Δt_hold + Δt_close"),
        ]:
            ttk.Label(valve, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2, padx=2)
            e = ttk.Entry(valve, width=14); e.insert(0, default)
            e.grid(row=row, column=1, pady=2)
            self.entries[key] = e
            # bind tooltip on hover
            e.bind("<Enter>", lambda ev, t=tip: self._show_tip(ev, t))
            e.bind("<Leave>", self._hide_tip)
            row += 1

        # Live timeline preview label
        # self._valve_timeline_lbl = ttk.Label(
        #     valve, text="", foreground="steelblue", font=("Consolas", 7),
        #     wraplength=340, justify=tk.LEFT)
        # self._valve_timeline_lbl.grid(
        #     row=row, column=0, columnspan=2, sticky=tk.W, padx=4, pady=(2, 0))
        # row += 1

        # Bind all timing entries to update the timeline preview
        # for key in ("t_valve_open_start", "t_valve_opening",
        #             "t_valve_hold_open", "t_valve_closing", "T_total"):
        #     self.entries[key].bind("<KeyRelease>", self._update_valve_timeline)
        #     self.entries[key].bind("<FocusOut>",   self._update_valve_timeline)

        # self._update_valve_timeline()  # initial render

        # ===== MOC NUMERICAL PARAMETERS =====
        moc_num = ttk.LabelFrame(sf, text="MOC Numerical Parameters", padding=8)
        moc_num.pack(fill=tk.X, padx=5, pady=4)

        row = 0
        ttk.Label(moc_num, text="Courant β  (0.5 – 0.9)").grid(
            row=row, column=0, sticky=tk.W, pady=2, padx=2)
        e_beta = ttk.Entry(moc_num, width=14); e_beta.insert(0, "0.9")
        e_beta.grid(row=row, column=1, pady=2)
        self.entries['beta'] = e_beta
        row += 1
        ttk.Label(moc_num,
                  text="dt = β·dx/B_eff   (β=1 → exact CFL; β<1 → diffusive/stable)",
                  foreground="gray", font=("Arial", 7)).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=14, pady=(0, 4))
        row += 1

        ttk.Label(moc_num, text="B_factor  (0.98 – 1.02)").grid(
            row=row, column=0, sticky=tk.W, pady=2, padx=2)
        e_bf = ttk.Entry(moc_num, width=14); e_bf.insert(0, "1.0")
        e_bf.grid(row=row, column=1, pady=2)
        self.entries['B_factor'] = e_bf
        row += 1
        ttk.Label(moc_num,
                  text="B_eff = sqrt(Z·R·T) × B_factor  (±2% wave-speed tuning)",
                  foreground="gray", font=("Arial", 7)).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=14)

        # ===== OPTIMIZATION SETTINGS =====
        opt = ttk.LabelFrame(sf, text="Optimization Settings", padding=8)
        opt.pack(fill=tk.X, padx=5, pady=4)

        # --- Initial diameter profile ---
        ttk.Label(opt, text="Initial Diameter Profile:",
                  font=("Arial", 9, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        self.profile_var = tk.StringVar(value="uniform")

        ttk.Radiobutton(opt, text="Uniform (default — all segments = D_pipe)",
                        variable=self.profile_var, value="uniform",
                        command=self._toggle_probable_inputs).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=20)

        ttk.Radiobutton(opt, text="Probable Profile (reduced diameter at location)",
                        variable=self.profile_var, value="probable",
                        command=self._toggle_probable_inputs).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=20)

        self.prob_frame = ttk.Frame(opt)
        self.prob_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=40, pady=2)

        ttk.Label(self.prob_frame, text="Baseline D multiplier D_init:").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        e = ttk.Entry(self.prob_frame, width=10); e.insert(0, "0.95")
        e.grid(row=0, column=1, pady=2)
        self.entries["prob_D_init"] = e

        ttk.Label(self.prob_frame, text="Reduced D multiplier (× D_pipe):").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        e = ttk.Entry(self.prob_frame, width=10); e.insert(0, "0.75")
        e.grid(row=1, column=1, pady=2)
        self.entries["prob_D_mult"] = e

        ttk.Label(self.prob_frame, text="Reduced segment location L_from_ds (m):").grid(
            row=2, column=0, sticky=tk.W, pady=2)
        e = ttk.Entry(self.prob_frame, width=10); e.insert(0, "200.0")
        e.grid(row=2, column=1, pady=2)
        self.entries["prob_L_ds"] = e

        ttk.Label(self.prob_frame, text="Reduced segment length (m):").grid(
            row=3, column=0, sticky=tk.W, pady=2)
        e = ttk.Entry(self.prob_frame, width=10); e.insert(0, "1000.0")
        e.grid(row=3, column=1, pady=2)
        self.entries["prob_seg_len"] = e

        self._toggle_probable_inputs()

        ttk.Separator(opt, orient='horizontal').grid(
            row=4, column=0, columnspan=2, sticky='ew', pady=8)

        # ── Optimizer settings ─────────────────────────────────────────
        row = 5
        opt_params = [
            ("D_range_min",   "D Multiplier Min  (e.g. 0.70)",   "0.70"),
            ("D_range_max",   "D Multiplier Max  (e.g. 1.05)",   "1.05"),
            ("max_iter",      "Max DE Iterations",                "150"),
            ("popsize",       "DE Population Size (×n_segs)",     "15"),
            ("lambda_smooth", "λ Smoothness  (e.g. 1e-6)",        "1e-6"),
            ("lambda_tv",     "λ Total Var   (e.g. 1e-7)",        "1e-7"),
        ]
        for key, lbl, default in opt_params:
            ttk.Label(opt, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2, padx=2)
            e = ttk.Entry(opt, width=14)
            e.grid(row=row, column=1, pady=2)
            e.insert(0, default)
            self.entries[key] = e
            row += 1

        # ── Parallel workers ───────────────────────────────────────────
        ttk.Separator(opt, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=4)
        row += 1

        n_cpu = _N_CPU
        ttk.Label(opt,
                  text=f"Parallel Workers  (CPU cores detected: {n_cpu})",
                  font=("Arial", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(2, 0), padx=2)
        row += 1

        ttk.Label(opt, text="DE Workers  (1 = serial)").grid(
            row=row, column=0, sticky=tk.W, pady=2, padx=2)
        e_workers = ttk.Entry(opt, width=14)
        e_workers.insert(0, str(max(1, n_cpu - 1)))   # default: leave 1 core for GUI
        e_workers.grid(row=row, column=1, pady=2)
        self.entries['workers'] = e_workers
        row += 1

        # Worker mode radio buttons
        self.worker_mode_var = tk.StringVar(value="custom")
        btn_frame = ttk.Frame(opt)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=14, pady=2)
        ttk.Radiobutton(btn_frame, text="Serial (1)",
                        variable=self.worker_mode_var, value="serial",
                        command=lambda: self._set_workers(1)).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(btn_frame, text=f"Half ({max(1, n_cpu//2)})",
                        variable=self.worker_mode_var, value="half",
                        command=lambda: self._set_workers(max(1, n_cpu//2))).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(btn_frame, text=f"All-1 ({max(1, n_cpu-1)})",
                        variable=self.worker_mode_var, value="all1",
                        command=lambda: self._set_workers(max(1, n_cpu-1))).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(btn_frame, text="Custom",
                        variable=self.worker_mode_var, value="custom").pack(side=tk.LEFT, padx=4)
        row += 1

        ttk.Label(opt,
                  text="workers>1 uses multiprocessing — progress shown per-generation",
                  foreground="gray", font=("Arial", 7)).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=14)
        row += 1

        ttk.Separator(opt, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=4)
        row += 1

        ttk.Label(opt, text="Method:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.method_var = tk.StringVar(value="differential_evolution")
        ttk.Combobox(opt, textvariable=self.method_var, width=22,
                     values=["differential_evolution",
                              "de_then_lbfgsb",
                              "lbfgsb"],
                     state="readonly").grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(opt,
                  text="de_then_lbfgsb = global DE + local polish (recommended)",
                  foreground="gray", font=("Arial", 7)).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=4)

        # ===== CONTROL BUTTONS =====
        ctrl = ttk.Frame(sf)
        ctrl.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(ctrl, text="▶  Run Simulation",
                   command=self._run_initial_simulation,
                   style="Accent.TButton").pack(fill=tk.X, pady=3)

        self.start_btn = ttk.Button(ctrl, text="🔍  Start Optimization",
                                    command=self._start_optimization,
                                    style="Accent.TButton")
        self.start_btn.pack(fill=tk.X, pady=3)

        self.stop_btn = ttk.Button(ctrl, text="⏹  Stop Optimization",
                                   command=self._stop_optimization,
                                   state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=3)

        self.progress_lbl = ttk.Label(ctrl, text="Ready", foreground="blue")
        self.progress_lbl.pack(pady=5)

    # ----------------------------------------------------------
    # RIGHT PANEL (Tabs)
    # ----------------------------------------------------------
    def _create_right_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 0: Log
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="📋 Log")
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD,
                                                   font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Convergence
        conv_frame = ttk.Frame(self.notebook)
        self.notebook.add(conv_frame, text="📈 Convergence")
        self.conv_canvas = None

        # Tab 2: Pressure Match
        press_frame = ttk.Frame(self.notebook)
        self.notebook.add(press_frame, text="🎯 Pressure Match")
        self.press_canvas = None

        # Tab 3: Diameter Profile
        diam_frame = ttk.Frame(self.notebook)
        self.notebook.add(diam_frame, text="📏 Diameter Profile")
        self.diam_canvas = None

        # Tab 4: Error Analysis
        err_frame = ttk.Frame(self.notebook)
        self.notebook.add(err_frame, text="📊 Error Analysis")
        self.error_canvas = None

        # Tab 5: Elevation Profile
        elev_tab = ttk.Frame(self.notebook)
        self.notebook.add(elev_tab, text="⛰ Elevation Profile")
        self.elev_canvas = None
        self._elev_tab_frame = elev_tab

        # Tab 6: Segment Data table
        seg_frame = ttk.Frame(self.notebook)
        self.notebook.add(seg_frame, text="🔢 Segment Data")
        self._build_segment_table(seg_frame)

        # Tab 7: Initial Conditions
        init_frame = ttk.Frame(self.notebook)
        self.notebook.add(init_frame, text="⚙ Initial Conditions")
        self.init_text = scrolledtext.ScrolledText(init_frame, wrap=tk.WORD,
                                                    font=("Consolas", 11))
        self.init_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 8: Transient Overview
        trans_frame = ttk.Frame(self.notebook)
        self.notebook.add(trans_frame, text="📈 Transient Overview")
        self.transient_fig = Figure(figsize=(11, 7))
        self.transient_canvas = FigureCanvasTkAgg(self.transient_fig, trans_frame)
        self.transient_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_segment_table(self, parent):
        tc = ttk.Frame(parent)
        tc.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        vsb = ttk.Scrollbar(tc, orient="vertical")
        hsb = ttk.Scrollbar(tc, orient="horizontal")
        self.segment_tree = ttk.Treeview(
            tc,
            columns=("Seg", "Pos", "Elev", "InitD", "OptD", "Chg"),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self.segment_tree.yview)
        hsb.config(command=self.segment_tree.xview)
        for col, hdr, w in [
            ("Seg",  "Segment #",        80),
            ("Pos",  "Position (m)",    110),
            ("Elev", "Elevation (m)",   110),
            ("InitD","Initial D (mm)",  120),
            ("OptD", "Optimized D (mm)",120),
            ("Chg",  "Change (%)",      100),
        ]:
            self.segment_tree.heading(col, text=hdr)
            self.segment_tree.column(col,  width=w, anchor="center")
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.segment_tree.pack(side="left", fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------
    # TOOLTIP HELPERS
    # ----------------------------------------------------------
    def _show_tip(self, event, text):
        """Show a small tooltip near the hovered widget."""
        try:
            if hasattr(self, '_tip_win') and self._tip_win:
                self._tip_win.destroy()
            x = event.widget.winfo_rootx() + 120
            y = event.widget.winfo_rooty() + 20
            self._tip_win = tw = tk.Toplevel(self.root)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=text, background="#ffffe0", relief="solid",
                     borderwidth=1, font=("Arial", 8), wraplength=260,
                     justify=tk.LEFT).pack()
        except Exception:
            pass

    def _hide_tip(self, event=None):
        if hasattr(self, '_tip_win') and self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None

    # ----------------------------------------------------------
    # VALVE TIMELINE PREVIEW
    # ----------------------------------------------------------
    # def _update_valve_timeline(self, event=None):
    #     """Recompute and display the valve timeline text in the UI."""
    #     try:
    #         t0   = float(self.entries['t_valve_open_start'].get())
    #         dop  = float(self.entries['t_valve_opening'].get())
    #         dhld = float(self.entries['t_valve_hold_open'].get())
    #         dcl  = float(self.entries['t_valve_closing'].get())
    #         ttot = float(self.entries['T_total'].get())

    #         t1 = t0 + dop
    #         t2 = t1 + dhld
    #         t3 = t2 + dcl

    #         ok = "✓" if t3 <= ttot else "⚠ T_total too short!"
    #         txt = (f"t=0…{t0:.1f}s  CLOSED  │  "
    #                f"t={t0:.1f}…{t1:.1f}s  OPENING  │  "
    #                f"t={t1:.1f}…{t2:.1f}s  OPEN  │  "
    #                f"t={t2:.1f}…{t3:.1f}s  CLOSING  │  "
    #                f"t>{t3:.1f}s  CLOSED   {ok}")
    #         color = "steelblue" if t3 <= ttot else "red"
    #     except Exception:
    #         txt   = "Enter valid numeric values above"
    #         color = "gray"

    #     if hasattr(self, '_valve_timeline_lbl'):
    #         self._valve_timeline_lbl.config(text=txt, foreground=color)

    # ----------------------------------------------------------
    # # PT POSITION VALIDATOR
    # # ----------------------------------------------------------
    # def _validate_pt_position(self, event=None):
    #     """
    #     Warn when x_pt_m is too close to or past the valve node.
    #     The valve BC node p[N] is directly set by the ISA solver — it has
    #     no sensitivity to pipe diameter D.  The PT must be at least 1 grid
    #     cell (dx) upstream of the valve for optimization to work.
    #     """
    #     try:
    #         x_pt    = float(self.entries['x_pt_m'].get())
    #         x_valve = float(self.entries['valve_pos_m'].get())
    #         dx      = float(self.entries['dx'].get())
    #         L       = float(self.entries['L'].get())
    #         N       = int(round(L / dx))
    #         i_pt    = int(round(x_pt / dx))
    #         i_valve = int(round(x_valve / dx))
    #         i_valve = min(i_valve, N)

    #         if i_pt >= i_valve:
    #             msg = (f"⚠ PT node ({i_pt}) ≥ valve node ({i_valve})! "
    #                    f"Optimizer will see ZERO diameter sensitivity. "
    #                    f"Set PT at least {dx:.0f} m upstream (x < {x_valve - dx:.0f} m).")
    #             self._pt_warn_lbl.config(text=msg, foreground="red")
    #         elif i_pt >= i_valve - 1:
    #             msg = (f"⚠ PT is only 1 node from valve — very weak sensitivity. "
    #                    f"Recommend x_pt ≤ {x_valve - 2*dx:.0f} m for good optimization.")
    #             self._pt_warn_lbl.config(text=msg, foreground="orange")
    #         else:
    #             gap = i_valve - i_pt
    #             self._pt_warn_lbl.config(
    #                 text=f"✓ PT node {i_pt}, valve node {i_valve}  ({gap} nodes apart — OK)",
    #                 foreground="green")
    #     except Exception:
    #         self._pt_warn_lbl.config(text="", foreground="gray")

    def _set_workers(self, n):
        """Set workers entry to n and switch radio to custom."""
        if 'workers' in self.entries:
            self.entries['workers'].delete(0, tk.END)
            self.entries['workers'].insert(0, str(n))

    # ----------------------------------------------------------
    # PROBABLE PROFILE TOGGLE
    # ----------------------------------------------------------
    def _toggle_probable_inputs(self):
        if self.profile_var.get() == "probable":
            for child in self.prob_frame.winfo_children():
                child.grid()
        else:
            for child in self.prob_frame.winfo_children():
                child.grid_remove()

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _update_segment_count(self, event=None):
        try:
            L  = float(self.entries['L'].get())
            dx = float(self.entries['dx'].get())
            self.n_segments = int(round(L / dx))
            self.seg_count_lbl.config(text=f"{self.n_segments} segments")
        except Exception:
            self.seg_count_lbl.config(text="Invalid L/dx")

    def _get_base_params(self):
        """Collect all simulation parameters from UI widgets."""
        T_celsius = float(self.entries['T_celsius'].get())
        T_K = T_celsius + 273.15
        g = float(self.entries.get('g', tk.Entry()).get() or 9.81)
        return dict(
            L               = float(self.entries['L'].get()),
            dx              = float(self.entries['dx'].get()),
            eps             = float(self.entries['eps_mm'].get()) / 1000.0,
            Z_factor        = float(self.entries['Z_factor'].get()),
            gamma           = float(self.entries['gamma'].get()),
            R_gas           = float(self.entries['R_gas'].get()),
            T_K             = T_K,
            mu_dyn          = float(self.entries['mu_dyn'].get()),
            P_upstream      = float(self.entries['P_upstream_bar'].get()) * 1e5,
            P_atm           = float(self.entries['P_atm_bar'].get())      * 1e5,
            valve_position  = float(self.entries['valve_pos_m'].get()),
            x_pt_m          = float(self.entries['x_pt_m'].get()),
            xT              = float(self.entries['xT'].get()),
            Fp              = float(self.entries['Fp'].get()),
            Cv_max          = float(self.entries['Cv_max'].get()),
            # 4-phase valve timing
            t_valve_open_start = float(self.entries['t_valve_open_start'].get()),
            t_valve_opening    = float(self.entries['t_valve_opening'].get()),
            t_valve_hold_open  = float(self.entries['t_valve_hold_open'].get()),
            t_valve_closing    = float(self.entries['t_valve_closing'].get()),
            K_multiplier    = float(self.entries['K_multiplier'].get()),
            T_total         = float(self.entries['T_total'].get()),
            # Courant number (replaces alpha)
            beta            = float(self.entries['beta'].get()),
            # Wave speed factor ±2%
            B_factor        = float(self.entries['B_factor'].get()),
            upstream_bc     = self.upstream_bc_var.get(),
            V_tank          = float(self.entries['V_tank'].get()),
            elevation_profile = self.elevation_profile,
            g               = g,
        )

    def _build_initial_profile(self, D_nom):
        ptype = self.profile_var.get()
        n     = self.n_segments

        if ptype == "uniform" or n == 0:
            return np.full(n, D_nom)

        L   = float(self.entries['L'].get())
        dx  = float(self.entries['dx'].get())
        try:
            D_init = float(self.entries.get('prob_D_init', tk.Entry()).get())
        except Exception:
            D_init = 0.95

        D_mult  = float(self.entries['prob_D_mult'].get())
        L_ds    = float(self.entries['prob_L_ds'].get())
        seg_len = float(self.entries['prob_seg_len'].get())

        D_prof = np.full(n, D_nom)
        pos_start = L - L_ds - seg_len
        pos_end   = L - L_ds

        for i in range(n):
            x_mid = (i + 0.5) * dx
            if pos_start <= x_mid <= pos_end:
                D_prof[i] = D_nom * D_mult
            else:
                D_prof[i] = D_nom * D_init

        self._log(f"  Probable profile: baseline×{D_init:.3f}, reduced×{D_mult:.3f} "
                  f"from x={pos_start:.0f} m to x={pos_end:.0f} m")
        return D_prof

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------
    def _load_pt_data(self):
        fn = filedialog.askopenfilename(
            title="Select Downstream PT Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not fn:
            return
        try:
            df = pd.read_csv(fn)
            if 'time' not in df.columns or 'pressure_bar' not in df.columns:
                raise ValueError("CSV must have 'time' and 'pressure_bar' columns")
            self.t_pt = df['time'].values
            self.p_pt = df['pressure_bar'].values
            self.pt_label.config(
                text=f"✓ {len(self.t_pt)} pts | "
                     f"{self.t_pt.min():.1f}–{self.t_pt.max():.1f} s | "
                     f"{self.p_pt.min():.3f}–{self.p_pt.max():.3f} bar",
                foreground="green")
            self._log(f"✓ Loaded PT data: {len(self.t_pt)} pts from {os.path.basename(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PT data:\n{e}")
            self._log(f"✗ PT load error: {e}")

    def _load_elevation(self):
        fn = filedialog.askopenfilename(
            title="Select Elevation Profile CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not fn:
            return
        try:
            df = pd.read_csv(fn)
            # Accept various column name conventions
            dist_col = None
            elev_col = None
            for c in df.columns:
                cl = c.lower().strip()
                if cl in ('distance_m', 'distance', 'dist', 'x', 'chainage'):
                    dist_col = c
                if cl in ('elevation_m', 'elevation', 'elev', 'height', 'z'):
                    elev_col = c

            if dist_col is None or elev_col is None:
                raise ValueError(
                    "CSV must have columns named 'distance_m' and 'elevation_m'.\n"
                    f"Found columns: {list(df.columns)}"
                )

            dist = df[dist_col].values.astype(float)
            elev = df[elev_col].values.astype(float)

            # Sort by distance
            order = np.argsort(dist)
            dist = dist[order]
            elev = elev[order]

            self.elevation_profile = np.column_stack([dist, elev])

            self.elev_label.config(
                text=f"✓ {len(dist)} pts | "
                     f"dist: {dist.min():.1f}–{dist.max():.1f} m | "
                     f"elev: {elev.min():.2f}–{elev.max():.2f} m",
                foreground="green")
            self._log(
                f"✓ Loaded elevation: {len(dist)} pts, "
                f"elev range {elev.min():.2f}–{elev.max():.2f} m "
                f"from {os.path.basename(fn)}"
            )
            self._plot_elevation_tab()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load elevation data:\n{e}")
            self._log(f"✗ Elevation load error: {e}")

    def _plot_elevation_tab(self):
        """Plot loaded elevation profile in the Elevation tab."""
        for w in self._elev_tab_frame.winfo_children():
            w.destroy()

        if self.elevation_profile is None:
            ttk.Label(self._elev_tab_frame,
                      text="No elevation profile loaded.").pack(pady=20)
            return

        fig = Figure(figsize=(11, 4))
        ax = fig.add_subplot(111)
        ax.plot(self.elevation_profile[:, 0], self.elevation_profile[:, 1],
                '-o', lw=2, ms=4, color='saddlebrown')
        ax.fill_between(self.elevation_profile[:, 0],
                        self.elevation_profile[:, 1].min(),
                        self.elevation_profile[:, 1],
                        alpha=0.25, color='saddlebrown')
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Pipeline Elevation Profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        c = FigureCanvasTkAgg(fig, master=self._elev_tab_frame)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.elev_canvas = c

    # ----------------------------------------------------------
    # RUN INITIAL SIMULATION
    # ----------------------------------------------------------
    def _run_initial_simulation(self):
        if self.t_pt is None:
            messagebox.showwarning("Warning", "Please load downstream PT data first")
            return
        self._update_segment_count()
        self._log("\n" + "="*50)
        self._log("Running initial simulation …")

        if self.elevation_profile is not None:
            self._log("  Elevation profile: ACTIVE (gravity terms included)")
        else:
            self._log("  Elevation profile: FLAT (gravity terms = 0)")

        # Warn if PT node is at or past valve — optimization will be blind
        try:
            x_pt    = float(self.entries['x_pt_m'].get())
            x_valve = float(self.entries['valve_pos_m'].get())
            dx_v    = float(self.entries['dx'].get())
            L_v     = float(self.entries['L'].get())
            N_v     = int(round(L_v / dx_v))
            i_pt_v  = min(int(round(x_pt / dx_v)), N_v - 1)
            i_valve_v = min(int(round(x_valve / dx_v)), N_v)
            if i_pt_v >= i_valve_v - 1:
                self._log(f"  ⚠ WARNING: PT node ({i_pt_v}) is at/adjacent to valve "
                          f"node ({i_valve_v}).")
                self._log(f"     Optimization will see no diameter sensitivity!")
                self._log(f"     Recommended: set x_pt_m ≤ {x_valve - 2*dx_v:.0f} m")
            else:
                self._log(f"  PT node {i_pt_v} / valve node {i_valve_v} — "
                          f"{i_valve_v - i_pt_v} nodes separation ✓")
        except Exception:
            pass

        self.progress_lbl.config(text="Building diameter profile …")

        try:
            params = self._get_base_params()
            D_nom  = float(self.entries['D_nom'].get())

            D_prof = self._build_initial_profile(D_nom)
            self.D_initial_profile = D_prof

            self._log(f"  Profile: {self.profile_var.get().upper()}")
            self._log(f"  D mean={D_prof.mean()*1e3:.3f} mm  "
                      f"min={D_prof.min()*1e3:.3f} mm  "
                      f"max={D_prof.max()*1e3:.3f} mm")

            params['D'] = D_prof
            self.progress_lbl.config(text="Running MOC …")

            res = run_gas_moc(**params)

            self._show_initial_conditions(params, res)
            self._plot_transient_overview(res)

            p_interp = np.interp(self.t_pt, res['time'], res['P_down_pt'])
            error    = p_interp - self.p_pt
            rmse     = np.sqrt(np.mean(error**2))
            mae      = np.mean(np.abs(error))

            self.initial_simulation = {
                't_model': res['time'], 'p_model': res['P_down_pt'],
                'p_interp': p_interp,
                'rmse': rmse, 'mae': mae,
                'D_array': D_prof, 'error': error,
                'profile_type': self.profile_var.get(),
                'res': res,
            }

            self._log(f"✓ Simulation done — RMSE: {rmse*1000:.3f} mbar  MAE: {mae*1000:.3f} mbar")
            self.progress_lbl.config(text=f"Initial RMSE: {rmse*1000:.3f} mbar")

            self._plot_initial_simulation()
            # Also refresh elevation tab in case profile was just loaded
            self._plot_elevation_tab()
            self.notebook.select(2)

            messagebox.showinfo("Success",
                f"Simulation completed\n"
                f"Profile : {self.profile_var.get()}\n"
                f"Elevation: {'Active' if self.elevation_profile is not None else 'Flat'}\n"
                f"RMSE    : {rmse*1000:.3f} mbar\n\n"
                f"Check the 'Pressure Match' tab.")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Simulation failed:\n{e}")
            self._log(f"✗ Error: {e}")
            self._log(traceback.format_exc())
            self.progress_lbl.config(text="Simulation error")

    def _show_initial_conditions(self, params, res):
        elev_info = "Not loaded (flat pipeline)"
        if self.elevation_profile is not None:
            ep = self.elevation_profile
            elev_info = (f"{len(ep)} points, "
                         f"range {ep[:,1].min():.2f}–{ep[:,1].max():.2f} m, "
                         f"max gradient {np.max(np.abs(np.diff(ep[:,1])/np.diff(ep[:,0]))):.4f} m/m")

        txt = f"""
PIPELINE INITIAL CONDITIONS
===========================

GEOMETRY
  Pipe length           : {params['L']:.0f} m
  Grid spacing          : {params['dx']:.1f} m  ({self.n_segments} segments)
  Pipe diameter         : {float(self.entries['D_nom'].get())*1000:.3f} mm
  Roughness             : {float(self.entries['eps_mm'].get()):.4f} mm

ELEVATION PROFILE
  {elev_info}

GAS PROPERTIES
  Temperature           : {float(self.entries['T_celsius'].get()):.1f} °C  ({params['T_K']:.2f} K)
  Z factor              : {params['Z_factor']}
  gamma                 : {params['gamma']}
  R_gas                 : {params['R_gas']} J/kg·K
  µ (dynamic viscosity) : {params['mu_dyn']:.3e} Pa·s

PRESSURES (initial)
  Upstream  P_upstream  : {params['P_upstream']/1e5:.3f} bar  → upstream BC
  Downstream P_atm      : {params['P_atm']/1e5:.3f} bar  → P_atm (valve downstream)

VALVE CONFIGURATION (ISA/IEC + 4-Phase Linear Cycle)
  xT                    : {params['xT']}
  Fp                    : {params['Fp']}
  Cv_max                : {params['Cv_max']}
  K_multiplier          : {params['K_multiplier']}
  t_valve_open_start    : {params['t_valve_open_start']} s   ← valve begins opening
  t_valve_opening       : {params['t_valve_opening']} s   ← opening ramp duration
  t_valve_hold_open     : {params['t_valve_hold_open']} s   ← hold fully open
  t_valve_closing       : {params['t_valve_closing']} s   ← closing ramp duration
  Valve closes at       : {params['t_valve_open_start']+params['t_valve_opening']+params['t_valve_hold_open']+params['t_valve_closing']:.1f} s
  T_total               : {params['T_total']} s

MOC NUMERICAL PARAMETERS
  Courant beta          : {params['beta']:.4f}  (β=1 → exact CFL, β<1 → sub-Courant)
  B_factor              : {params['B_factor']:.4f}  (B_eff = B_iso × B_factor)

SIMULATION OUTPUTS
  Peak valve flow       : {res['Q_valve'].max()*3600:.2f} m³/hr
  Max pipe pressure     : {res['P_up'].max():.3f} bar  (node 0)
  Min pipe pressure     : {res['P_up'].min():.3f} bar  (node 0)
  Downstream PT max     : {res['P_down_pt'].max():.3f} bar
  Downstream PT min     : {res['P_down_pt'].min():.3f} bar
"""
        self.init_text.delete("1.0", tk.END)
        self.init_text.insert(tk.END, txt)

    def _plot_transient_overview(self, res):
        self.transient_fig.clear()
        t = res['time']

        ax1 = self.transient_fig.add_subplot(311)
        ax1.plot(t, res['P_up'],             label='p[0] Upstream',         lw=2)
        ax1.plot(t, res['P_down_pt'],        label='Downstream PT location', lw=2)
        ax1.plot(t, res['P_valve_upstream'], label='Valve upstream face',    lw=1.5, ls='--')
        ax1.plot(t, res['P_flare'],          label='Flare / pipe end',       lw=1.5, ls=':')
        ax1.set_ylabel("Pressure (bar)")
        ax1.set_title("Pressures vs Time")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2 = self.transient_fig.add_subplot(312)
        ax2.plot(t, res['Q_valve'] * 3600, 'r-', lw=2, label='Valve flow')
        ax2.set_ylabel("Flow (m³/hr)")
        ax2.set_title("Valve Flow Rate vs Time")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        ax3 = self.transient_fig.add_subplot(313)
        ax3.plot(t, res['tau'] * 100, color='darkgreen', lw=2)
        ax3.fill_between(t, 0, res['tau'] * 100, alpha=0.3, color='green')
        ax3.set_ylabel("Valve opening (%)")
        ax3.set_xlabel("Time (s)")
        ax3.set_title("Valve Timing")
        ax3.grid(True, alpha=0.3)

        self.transient_fig.tight_layout()
        self.transient_canvas.draw()

    def _plot_initial_simulation(self):
        if not self.initial_simulation:
            return
        tab = self.notebook.winfo_children()[2]
        for w in tab.winfo_children():
            w.destroy()

        fig = Figure(figsize=(11, 7))

        ax1 = fig.add_subplot(211)
        ax1.plot(self.t_pt, self.p_pt,
                 'o', ms=2, alpha=0.6, label='Downstream PT Data', color='blue')
        ax1.plot(self.initial_simulation['t_model'],
                 self.initial_simulation['p_model'],
                 '-', lw=2,
                 label=f"Model ({self.initial_simulation['profile_type']})",
                 color='orange')
        ax1.set_ylabel("Pressure (bar)")
        ax1.set_title(
            f"Downstream Pressure Match — Initial Simulation\n"
            f"RMSE: {self.initial_simulation['rmse']*1000:.3f} mbar  "
            f"MAE: {self.initial_simulation['mae']*1000:.3f} mbar",
            fontweight='bold')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(212)
        err_mb = self.initial_simulation['error'] * 1000
        ax2.plot(self.t_pt, err_mb, '-', lw=1.5, color='orange', alpha=0.8)
        ax2.fill_between(self.t_pt, err_mb, alpha=0.3, color='orange')
        ax2.axhline(0, color='k', lw=0.8, ls='--')
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Error (mbar)")
        ax2.set_title("Model Error")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=tab)
        c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.press_canvas = c

    # ----------------------------------------------------------
    # OPTIMIZATION
    # ----------------------------------------------------------
    def _start_optimization(self):
        if self.t_pt is None:
            messagebox.showwarning("Warning", "Please load PT data first")
            return
        self._update_segment_count()
        if self.n_segments == 0:
            messagebox.showerror("Error", "Invalid grid parameters (L / dx)")
            return

        self._log("\n" + "="*50)
        self._log(f"Starting optimization — {self.n_segments} segments")
        if self.elevation_profile is not None:
            self._log("  Elevation profile: ACTIVE in optimizer")

        # Guard: PT node must not equal valve node
        # try:
        #     x_pt    = float(self.entries['x_pt_m'].get())
        #     x_valve = float(self.entries['valve_pos_m'].get())
        #     dx_val  = float(self.entries['dx'].get())
        #     L_val   = float(self.entries['L'].get())
        #     # N_val   = int(round(L_val / dx_val))
        #     # i_pt_chk = min(int(round(x_pt / dx_val)), N_val - 1)
        #     # i_v_chk  = min(int(round(x_valve / dx_val)), N_val)
        #     # if i_pt_chk >= i_v_chk - 1:
        #     #     msg = (f"PT location ({x_pt:.0f} m, node {i_pt_chk}) is at or adjacent "
        #     #            f"to valve node ({i_v_chk}).\n\n"
        #     #            f"The valve boundary condition directly sets p[{i_v_chk}] — "
        #     #            f"it has no sensitivity to pipe diameter.\n\n"
        #     #            f"Set x_pt_m at least {2*dx_val:.0f} m upstream of the valve "
        #     #            f"(e.g. {x_valve - 2*dx_val:.0f} m or less).")
        #     #     messagebox.showerror("PT Location Error", msg)
        #     #     self._log(f"✗ PT too close to valve — optimization aborted. "
        #     #               f"Set x_pt < {x_valve - dx_val:.0f} m")
        #     #     return
        # except Exception:
        #     pass

        try:
            params   = self._get_base_params()
            D_nom    = float(self.entries['D_nom'].get())
            D_range  = (float(self.entries['D_range_min'].get()),
                        float(self.entries['D_range_max'].get()))
            max_iter = int(self.entries['max_iter'].get())
            raw_pop  = int(self.entries['popsize'].get())

            if raw_pop <= 0:
                raw_pop = 15
                self._log(f"  ⚠ Population size set to {raw_pop}")

            # Read worker count
            try:
                n_workers = int(self.entries['workers'].get())
                n_workers = max(1, min(n_workers, _N_CPU))
            except Exception:
                n_workers = 1
            if n_workers > 1:
                self._log(f"  Parallel workers: {n_workers} / {_N_CPU} CPUs")
                self._log("  ⚠ Progress shown per-generation when workers > 1")
            else:
                self._log(f"  Serial mode (workers=1) — per-eval progress active")

            D_prof = self._build_initial_profile(D_nom)
            self.D_initial_profile = D_prof

            self._log(f"  Search space: D ∈ [{D_range[0]:.3f}×, {D_range[1]:.3f}×] D_nom")
            self._log(f"  D_nom = {D_nom*1e3:.3f} mm  →  "
                      f"[{D_nom*D_range[0]*1e3:.3f}, {D_nom*D_range[1]*1e3:.3f}] mm")
            self._log(f"  Method: {self.method_var.get()}  |  "
                      f"max_iter={max_iter}  |  popsize={raw_pop}×{self.n_segments}  |  "
                      f"workers={n_workers}")

            self.optimizer = GasMOCOptimizer(
                self.t_pt, self.p_pt, params, D_nom,
                self.n_segments, D_initial_profile=D_prof,
            )
            try:
                self.optimizer.lambda_smooth = float(self.entries['lambda_smooth'].get())
                self.optimizer.lambda_tv     = float(self.entries['lambda_tv'].get())
            except Exception:
                pass

            self.optimizer.progress_callback = self._on_progress

            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            self.opt_thread = threading.Thread(
                target=self._opt_thread,
                args=(D_range, max_iter, raw_pop, n_workers),
                daemon=True,
            )
            self.opt_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Could not start optimization:\n{e}")
            self._log(f"✗ {e}")

    def _opt_thread(self, D_range, max_iter, popsize, workers=1):
        try:
            method = self.method_var.get()
            result = self.optimizer.optimize(
                method=method, D_range=D_range, max_iter=max_iter,
                popsize=popsize, workers=workers,
            )
            if result is not None:
                self.root.after(0, self._opt_complete, result)
            else:
                self.root.after(0, self._opt_stopped)
        except Exception as e:
            self.root.after(0, lambda: self._opt_error(e))

    def _on_progress(self, data):
        self.root.after(0, lambda: self._update_progress(data))

    def _update_progress(self, data):
        rmse_str = f"{data['rmse']:.4f}"
        cost_str = f"{data['cost']:.6f}" if 'cost' in data else ""
        msg = (f"Eval {data['eval']:4d}: "
               f"D={data['D_mean']*1000:.3f}±{data['D_std']*1000:.3f} mm  "
               f"RMSE={rmse_str} mbar")
        if data.get('improved'):
            msg += "  ✓ BEST"
        # Always log every 10 evals, or whenever improved
        if data.get('improved') or data['eval'] % 10 == 0:
            self._log(msg)
        self.progress_lbl.config(
            text=f"Eval {data['eval']} | RMSE={rmse_str} mbar"
                 + (" ✓" if data.get('improved') else ""),
            foreground="green" if data.get('improved') else "blue")

    def _opt_complete(self, result):
        D_opt  = self.optimizer.best_D_array
        D_nom  = float(self.entries['D_nom'].get())
        params = self._get_base_params()
        params['D'] = D_opt

        res = run_gas_moc(**params)

        p_interp = np.interp(self.t_pt, res['time'], res['P_down_pt'])
        error    = p_interp - self.p_pt
        rmse     = np.sqrt(np.mean(error**2))
        mae      = np.mean(np.abs(error))

        self._log("\n" + "="*50)
        self._log("✓ Optimization completed!")
        self._log(f"  RMSE: {rmse*1000:.3f} mbar   MAE: {mae*1000:.3f} mbar")
        self._log(f"  D range: {D_opt.min()*1e3:.3f} – {D_opt.max()*1e3:.3f} mm")

        self.result_data = {
            'base_params': params, 'D_nom': D_nom,
            'D_opt_array': D_opt,
            't_model': res['time'], 'p_model': res['P_down_pt'],
            't_pt': self.t_pt,      'p_pt': self.p_pt,
            'p_interp': p_interp,   'error': error,
            'rmse': rmse, 'mae': mae,
            'x_segments': res['x_segments'],
            'elevation_arr': res['elevation_arr'],
            'history': self.optimizer.history,
            'n_evals': self.optimizer.eval_count,
        }

        self._plot_convergence()
        self._plot_pressure_match()
        self._plot_diameter_profile()
        self._plot_error_analysis()
        self._update_segment_table()

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_lbl.config(text=f"✓ Done: RMSE={rmse*1000:.3f} mbar")

        messagebox.showinfo("Done",
            f"Optimization complete!\n\n"
            f"RMSE: {rmse*1000:.3f} mbar\n"
            f"Segments: {self.n_segments}\n"
            f"D range: {D_opt.min()*1e3:.3f} – {D_opt.max()*1e3:.3f} mm")

    def _opt_stopped(self):
        self._log("⚠ Optimization stopped.")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_lbl.config(text="Stopped")

    def _opt_error(self, e):
        self._log(f"✗ Optimization error: {e}")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_lbl.config(text="Error")
        messagebox.showerror("Error", f"Optimization failed:\n{e}")

    def _stop_optimization(self):
        if self.optimizer:
            self.optimizer.stop()
            self._log("Stopping …")

    # ----------------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------------
    def _clear_tab(self, tab_idx):
        for w in self.notebook.winfo_children()[tab_idx].winfo_children():
            w.destroy()

    def _plot_convergence(self):
        if not self.result_data:
            return
        self._clear_tab(1)
        fig   = Figure(figsize=(11, 6))
        hist  = self.result_data['history']
        evals = [h['eval']      for h in hist]
        rmse  = [h['rmse_mbar'] for h in hist]
        cost  = [h['cost']      for h in hist]

        # Best-so-far envelope
        best_so_far = []
        best = np.inf
        for r in rmse:
            if r < best:
                best = r
            best_so_far.append(best)

        D_mean = [h['D_mean'] * 1e3 for h in hist]
        D_std  = [h['D_std']  * 1e3 for h in hist]

        ax1 = fig.add_subplot(211)
        ax1.semilogy(evals, rmse, '-', lw=1, color='lightsteelblue',
                     alpha=0.6, label='RMSE per eval')
        ax1.semilogy(evals, best_so_far, '-', lw=2.5, color='steelblue',
                     label='Best RMSE so far')
        ax1.set_ylabel("RMSE (mbar) — log scale")
        ax1.set_title("Convergence — RMSE vs Evaluation Number")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, which='both')

        ax2 = fig.add_subplot(212)
        ax2.plot(evals, D_mean, 'g-', lw=2, label='Mean D (mm)')
        ax2.fill_between(evals,
                         np.array(D_mean) - np.array(D_std),
                         np.array(D_mean) + np.array(D_std),
                         alpha=0.25, color='g', label='±1σ')
        ax2.axhline(self.result_data['D_nom'] * 1e3, color='gray',
                    ls='--', lw=1.2, label=f'Nominal {self.result_data["D_nom"]*1e3:.2f} mm')
        ax2.set_xlabel("Evaluation #")
        ax2.set_ylabel("Diameter (mm)")
        ax2.set_title("Diameter Mean ± Std vs Evaluation")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()

        tab = self.notebook.winfo_children()[1]
        c = FigureCanvasTkAgg(fig, master=tab)
        c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.conv_canvas = c

    def _plot_pressure_match(self):
        if not self.result_data:
            return
        self._clear_tab(2)
        fig = Figure(figsize=(11, 6))
        ax  = fig.add_subplot(111)
        ax.plot(self.result_data['t_pt'], self.result_data['p_pt'],
                'o', ms=2, alpha=0.6, label='PT Data (downstream)', color='blue')
        if self.initial_simulation:
            ax.plot(self.initial_simulation['t_model'],
                    self.initial_simulation['p_model'],
                    '--', lw=2, alpha=0.7, label='Initial (uniform D)', color='orange')
        ax.plot(self.result_data['t_model'], self.result_data['p_model'],
                '-', lw=2, label='Optimized', color='red')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Pressure (bar)")
        ax.set_title("Downstream Pressure Match — Segmented Diameter Optimization",
                     fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        tab = self.notebook.winfo_children()[2]
        c = FigureCanvasTkAgg(fig, master=tab)
        c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.press_canvas = c

    def _plot_diameter_profile(self):
        if not self.result_data:
            return
        self._clear_tab(3)

        has_elev = self.elevation_profile is not None
        fig = Figure(figsize=(11, 7 if has_elev else 5))

        n_subplots = 2 if has_elev else 1
        ax = fig.add_subplot(n_subplots, 1, 1)

        x   = self.result_data['x_segments']
        D_o = self.result_data['D_opt_array'] * 1e3
        D_n = self.result_data['D_nom'] * 1e3

        if self.D_initial_profile is not None:
            D_i = self.D_initial_profile * 1e3
            ax.plot(x, D_i, '--', lw=1.5, color='orange', alpha=0.7, label='Initial profile')

        ax.plot(x, D_o, '-o', lw=2, ms=3, color='red',
                label=f'Optimized ({D_o.min():.2f}–{D_o.max():.2f} mm)')
        ax.axhline(D_n, color='gray', ls=':', lw=1.5, label=f'Nominal {D_n:.2f} mm')
        ax.fill_between(x, D_n, D_o, alpha=0.2, color='green')

        vx = float(self.entries['valve_pos_m'].get())
        ax.axvline(vx, color='purple', ls=':', lw=1.5, label=f'Valve @ {vx:.0f} m')
        px = float(self.entries['x_pt_m'].get())
        ax.axvline(px, color='blue', ls=':', lw=1.5, label=f'PT @ {px:.0f} m')

        ax.set_xlabel("Position (m)"); ax.set_ylabel("Diameter (mm)")
        ax.set_title(f"Optimized Diameter Profile ({self.n_segments} segments)",
                     fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        if has_elev:
            ax2 = fig.add_subplot(2, 1, 2)
            ep = self.elevation_profile
            ax2.plot(ep[:, 0], ep[:, 1], '-', lw=2, color='saddlebrown',
                     label='Elevation profile')
            ax2.fill_between(ep[:, 0], ep[:, 1].min(), ep[:, 1],
                             alpha=0.2, color='saddlebrown')
            ax2.axvline(vx, color='purple', ls=':', lw=1.5)
            ax2.axvline(px, color='blue',   ls=':', lw=1.5)
            ax2.set_xlabel("Position (m)"); ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Elevation Profile")
            ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        tab = self.notebook.winfo_children()[3]
        c = FigureCanvasTkAgg(fig, master=tab)
        c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.diam_canvas = c

    def _plot_error_analysis(self):
        if not self.result_data:
            return
        self._clear_tab(4)
        fig = Figure(figsize=(11, 7))
        err_o = self.result_data['error'] * 1000

        ax1 = fig.add_subplot(211)
        if self.initial_simulation:
            err_i = (self.initial_simulation['p_interp'] - self.result_data['p_pt']) * 1000
            ax1.plot(self.result_data['t_pt'], err_i, '-', lw=1.5,
                     color='orange', alpha=0.7, label='Initial Error')
            ax1.fill_between(self.result_data['t_pt'], err_i, alpha=0.2, color='orange')
        ax1.plot(self.result_data['t_pt'], err_o, '-', lw=1.5,
                 color='red', label='Optimized Error')
        ax1.axhline(0, color='k', lw=0.8, ls='--')
        ax1.fill_between(self.result_data['t_pt'], err_o, alpha=0.3, color='red')
        ax1.set_ylabel("Error (mbar)"); ax1.set_title("Model Error Comparison")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(212)
        if self.initial_simulation:
            ax2.hist(err_i, bins=30, alpha=0.5, color='orange', label='Initial',
                     edgecolor='black')
        ax2.hist(err_o, bins=30, alpha=0.5, color='red', label='Optimized',
                 edgecolor='black')
        ax2.axvline(0, color='k', lw=1.5, ls='--')
        ax2.axvline(err_o.mean(), color='red', lw=1.5, ls='--',
                    label=f'Mean = {err_o.mean():.2f} mbar')
        ax2.set_xlabel("Error (mbar)"); ax2.set_ylabel("Frequency")
        ax2.set_title("Error Distribution"); ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        tab = self.notebook.winfo_children()[4]
        c = FigureCanvasTkAgg(fig, master=tab)
        c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.error_canvas = c

    def _update_segment_table(self):
        if not self.result_data:
            return
        for item in self.segment_tree.get_children():
            self.segment_tree.delete(item)

        x     = self.result_data['x_segments']
        D_n   = self.result_data['D_nom'] * 1e3
        D_i   = (self.D_initial_profile * 1e3
                 if self.D_initial_profile is not None
                 else np.full(len(x), D_n))
        D_o   = self.result_data['D_opt_array'] * 1e3

        # Elevation at segment midpoints
        elev_arr = self.result_data.get('elevation_arr', None)
        # elevation_arr is node-length (N+1); map to segment midpoints
        if elev_arr is not None and len(elev_arr) == len(x) + 1:
            elev_seg = 0.5 * (elev_arr[:-1] + elev_arr[1:])
        elif elev_arr is not None and len(elev_arr) == len(x):
            elev_seg = elev_arr
        else:
            elev_seg = np.zeros(len(x))

        for i in range(len(D_o)):
            chg = (D_o[i] - D_n) / D_n * 100
            self.segment_tree.insert("", "end", values=(
                f"{i+1}", f"{x[i]:.1f}", f"{elev_seg[i]:.2f}",
                f"{D_i[i]:.3f}", f"{D_o[i]:.3f}", f"{chg:+.2f}",
            ))

    # ----------------------------------------------------------
    # EXPORT
    # ----------------------------------------------------------
    def _export_results(self):
        if not self.result_data:
            messagebox.showwarning("Warning", "No results to export"); return
        fn = filedialog.asksaveasfilename(
            title="Export Results", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if not fn: return
        try:
            df = pd.DataFrame({
                'time_s': self.result_data['t_pt'],
                'pressure_pt_bar': self.result_data['p_pt'],
                'pressure_model_bar': self.result_data['p_interp'],
                'error_mbar': self.result_data['error'] * 1000,
            })
            df.to_csv(fn, index=False)
            self._log(f"✓ Exported results: {os.path.basename(fn)}")
            messagebox.showinfo("Success", f"Saved:\n{os.path.basename(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def _export_segment_diameters(self):
        if not self.result_data:
            messagebox.showwarning("Warning", "No results to export"); return
        fn = filedialog.asksaveasfilename(
            title="Export Segment Diameters", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if not fn: return
        try:
            x   = self.result_data['x_segments']
            D_n = self.result_data['D_nom'] * 1e3
            D_o = self.result_data['D_opt_array'] * 1e3

            elev_arr = self.result_data.get('elevation_arr', None)
            if elev_arr is not None and len(elev_arr) == len(x) + 1:
                elev_seg = 0.5 * (elev_arr[:-1] + elev_arr[1:])
            elif elev_arr is not None and len(elev_arr) == len(x):
                elev_seg = elev_arr
            else:
                elev_seg = np.zeros(len(x))

            df = pd.DataFrame({
                'segment':         np.arange(1, len(D_o) + 1),
                'position_m':      x,
                'elevation_m':     elev_seg,
                'nominal_D_mm':    D_n,
                'initial_D_mm':    (self.D_initial_profile * 1e3
                                    if self.D_initial_profile is not None
                                    else np.full(len(x), D_n)),
                'optimized_D_mm':  D_o,
                'change_pct':      (D_o - D_n) / D_n * 100,
            })
            df.to_csv(fn, index=False)
            self._log(f"✓ Exported diameters: {os.path.basename(fn)}")
            messagebox.showinfo("Success", f"Saved:\n{os.path.basename(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def _save_config(self):
        fn = filedialog.asksaveasfilename(
            title="Save Config", defaultextension=".json",
            filetypes=[("JSON files", "*.json")])
        if not fn: return
        try:
            cfg = {k: e.get() for k, e in self.entries.items()}
            cfg['method'] = self.method_var.get()
            cfg['profile'] = self.profile_var.get()
            cfg['upstream_bc'] = self.upstream_bc_var.get()
            with open(fn, 'w') as f:
                json.dump(cfg, f, indent=2)
            self._log(f"✓ Config saved: {os.path.basename(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

    def _load_config(self):
        fn = filedialog.askopenfilename(
            title="Load Config", filetypes=[("JSON files", "*.json")])
        if not fn: return
        try:
            with open(fn) as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if k in self.entries:
                    self.entries[k].delete(0, tk.END)
                    self.entries[k].insert(0, v)
                elif k == 'method':      self.method_var.set(v)
                elif k == 'profile':     self.profile_var.set(v)
                elif k == 'upstream_bc': self.upstream_bc_var.set(v)
            self._update_segment_count()
            self._toggle_probable_inputs()
            self._log(f"✓ Config loaded: {os.path.basename(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")

    def _generate_pdf(self):
        if not self.result_data:
            messagebox.showwarning("Warning", "No results to report"); return
        fn = filedialog.asksaveasfilename(
            title="Save PDF Report", defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")])
        if not fn: return
        try:
            with PdfPages(fn) as pdf:
                for canvas_attr in ['conv_canvas', 'press_canvas',
                                    'diam_canvas', 'error_canvas', 'elev_canvas']:
                    c = getattr(self, canvas_attr, None)
                    if c:
                        pdf.savefig(c.figure, bbox_inches='tight')

                fig = Figure(figsize=(10, 8))
                ax  = fig.add_subplot(111)
                ax.axis('off')
                rd  = self.result_data
                summary = (
                    f"Gas MOC Smart Optimizer — Report\n"
                    f"{'='*55}\n\n"
                    f"Segments     : {self.n_segments}\n"
                    f"Evaluations  : {rd['n_evals']}\n"
                    f"RMSE         : {rd['rmse']*1000:.3f} mbar\n"
                    f"MAE          : {rd['mae']*1000:.3f} mbar\n\n"
                    f"D nominal    : {rd['D_nom']*1e3:.3f} mm\n"
                    f"D optimized  : {rd['D_opt_array'].min()*1e3:.3f} – "
                    f"{rd['D_opt_array'].max()*1e3:.3f} mm\n"
                    f"D mean       : {rd['D_opt_array'].mean()*1e3:.3f} mm\n\n"
                    f"Elevation    : {'Active' if self.elevation_profile is not None else 'Flat'}\n"
                    f"Generated    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                        fontsize=10, va='top', family='monospace')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            self._log(f"✓ PDF report: {os.path.basename(fn)}")
            messagebox.showinfo("Success", f"Report saved:\n{os.path.basename(fn)}")
        except Exception as e:
            messagebox.showerror("Error", f"PDF failed:\n{e}")

    # ----------------------------------------------------------
    # ABOUT / HELP
    # ----------------------------------------------------------
    def _show_about(self):
        messagebox.showinfo("About",
            "Gas MOC Smart Optimizer — 2-BC-UP\n"
            "Version 4.1\n\n"
            "• Array-diameter MOC engine with elevation support\n"
            "• Downstream PT data optimization\n"
            "• 4-phase linear valve: CLOSED→open ramp→hold open→close ramp→CLOSED\n"
            "• ISA/IEC N6 ball-valve mass-flow model + K_multiplier\n"
            "• Courant β (0.5–0.9) replaces alpha\n"
            "• B_factor ±2% wave-speed uncertainty tuning\n"
            "• Foot-node D & A areas for friction (Wylie §15-5)\n"
            "• Live valve timeline preview in UI\n"
            "• Elevation profile CSV input (distance_m, elevation_m)\n"
            "• Uniform & Probable diameter profiles\n"
            "• Differential Evolution + L-BFGS-B\n\n"
            "Author: Bharat Flow Analytics — 2026-03")

    def _show_help(self):
        hw = tk.Toplevel(self.root)
        hw.title("User Guide — Gas MOC Smart Optimizer")
        hw.geometry("800x700")
        txt = scrolledtext.ScrolledText(hw, wrap=tk.WORD, font=("Consolas", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert("1.0", """
USER GUIDE — Gas MOC Smart Optimizer (2-BC-UP) v4.1
=====================================================

OVERVIEW
--------
Fits per-segment pipe diameters to measured downstream
pressure-transducer (PT) data using the Gas Method of
Characteristics (MOC) engine (Wylie & Streeter Ch. 15).

VALVE MODEL — 4-Phase Linear Cycle
------------------------------------
The valve starts CLOSED and follows this sequence:

  Phase 0 : CLOSED          t = 0 ... t_valve_open_start
  Phase 1 : Opening ramp    0->1 linearly over t_valve_opening (s)
  Phase 2 : Hold fully open tau = 1.0 for t_valve_hold_open (s)
  Phase 3 : Closing ramp    1->0 linearly over t_valve_closing (s)
  Phase 4 : CLOSED          tau = 0.0 for remainder

  A live timeline preview is shown in the valve panel.
  T_total must exceed the end of Phase 3.

VALVE FLOW MODEL — ISA/IEC N6
------------------------------
  M_dot = N6 * Fp * Cv_eff * Y * sqrt(x * p_u * rho_u)
  Cv_eff = (Cv_max / K_multiplier) * tau
  K_multiplier > 1 increases effective valve resistance.

MOC NUMERICAL PARAMETERS
--------------------------
  beta (0.5-0.9): Courant number, dt = beta*dx/B_eff
    beta=1 -> exact CFL; beta<1 -> sub-Courant (more stable)

  B_factor (0.98-1.02): wave-speed scale
    B_eff = sqrt(Z*R*T) * B_factor

WORKFLOW
--------
1. LOAD PT DATA (File -> Load PT Data)
   CSV columns: time  pressure_bar

2. LOAD ELEVATION PROFILE (optional)
   CSV columns: distance_m  elevation_m

3. PIPELINE GEOMETRY — L, dx, D_pipe, roughness, PT/valve location

4. GAS PROPERTIES — T, mu, R_gas, Z, gamma

5. PIPELINE INITIAL PRESSURE
   Upstream bar -> P_upstream (BC), Downstream bar -> P_atm

6. BALL VALVE — ISA coefficients + 4-phase timing
   Watch the live timeline preview for validation.

7. MOC NUMERICAL — set beta and B_factor

8. RUN SIMULATION — forward check vs PT data

9. START OPTIMIZATION — DE or L-BFGS-B diameter fitting

10. EXPORT — CSV / Segment diameters / PDF report

TIPS
----
• Set T_total = t_valve_open_start + t_valve_opening
              + t_valve_hold_open + t_valve_closing + buffer
• beta=0.9 is a good default. Lower to 0.5 if divergence occurs.
• B_factor shifts wave arrival time; +-1% is usually enough.
• K_multiplier > 1 dampens the peak flow pulse.
""")
        txt.config(state=tk.DISABLED)
        ttk.Button(hw, text="Close", command=hw.destroy).pack(pady=5)

    # ----------------------------------------------------------
    # CLOSE
    # ----------------------------------------------------------
    def _on_close(self):
        if self.opt_thread and self.opt_thread.is_alive():
            if messagebox.askyesno("Exit", "Optimization is running. Exit anyway?"):
                if self.optimizer:
                    self.optimizer.stop()
                self.root.quit()
        else:
            self.root.quit()


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    # Required on Windows when frozen (e.g. PyInstaller) so that
    # multiprocessing worker processes don't spawn infinite GUI windows.
    multiprocessing.freeze_support()

    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Accent.TButton',
                    foreground='white', background='#0078D4',
                    font=('Arial', 10, 'bold'))
    GasMOCOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
