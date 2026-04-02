#!/usr/bin/env python
r"""
Paper 1 Figure Generator
=========================

Generates all solver-based publication-quality figures for:
    "Viscoelastic Reynolds equation for thin-film lubrication via squeeze
     Deborah number expansion with the Giesekus constitutive model"

Target: Journal of Non-Newtonian Fluid Mechanics

Figures generated:
    Fig 2:  Material functions eta, Psi1, Psi2 for both PAM fluids
    Fig 3:  Parameter hierarchy (epsilon, De_sq, De, Wi) for both fluids
    Fig 4:  Grid convergence study (2nd-order verification)
    Fig 5:  UCM validation against Tichy (1996)
    Fig 6:  PTT (1985) Boger fluid validation (CRITICAL) with De_sq validity shading
    Fig 7:  Flux decomposition vs De_sq
    Fig 8:  Load ratio F_VE/F_GNF for both fluids vs De_sq
    Fig 9:  Pressure field comparison (Newtonian / GNF / Giesekus)
    Fig 10: Error budget visualisation for both fluids

Usage:
    cd giesekus-solver
    PYTHONPATH=. python scripts/generate_paper1_figures.py
    PYTHONPATH=. python scripts/generate_paper1_figures.py --only fig2 fig6

Author: N.C. Dhlamini (dhlnig001@myuct.ac.za), March 2026
"""
import numpy as np
import os
import sys
import warnings
import time as clk
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    GiesekusFluid,
    create_PAM_2pct,
    create_PAM_5pct,
    SlipperGeometry,
    PolarMesh,
    OperatingConditions,
    SolverConfig,
    solve_newtonian,
    solve_GNF,
    solve_viscoelastic,
)

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Output directory
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'figures', 'paper1')


# =====================================================================
# PUBLICATION STYLE
# =====================================================================
def setup_paper_style():
    """JNNFM-compatible matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })


def savefig(fig, name):
    """Save figure as PNG and PDF."""
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(OUTDIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


# =====================================================================
# FIG 2: MATERIAL FUNCTIONS
# =====================================================================
def fig2_material_functions():
    """
    Three-panel log-log plot of Giesekus material functions:
      (a) eta(gdot), (b) Psi1(gdot), (c) Psi2(gdot)
    for 2% and 5% PAM with low- and high-Wi asymptotes.
    """
    print("\n--- Fig 2: Material functions ---")
    fluid_2 = create_PAM_2pct()
    fluid_5 = create_PAM_5pct()

    gdot = np.logspace(-1, 7, 500)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Colours / styles
    c2, c5 = 'tab:blue', 'tab:red'
    ls2, ls5 = '-', '--'

    # (a) Viscosity
    ax = axes[0]
    for fluid, c, ls, label in [(fluid_2, c2, ls2, '2% PAM'),
                                  (fluid_5, c5, ls5, '5% PAM')]:
        eta = fluid.viscosity(gdot)
        ax.loglog(gdot, eta, color=c, ls=ls, label=label)
        # Zero-shear asymptote
        ax.axhline(fluid.eta_0, color=c, ls=':', lw=0.8, alpha=0.6)
        # Solvent asymptote
        ax.axhline(fluid.eta_s, color=c, ls='-.', lw=0.8, alpha=0.4)

    ax.set_xlabel(r'Shear rate $\dot{\gamma}$ [s$^{-1}$]')
    ax.set_ylabel(r'Viscosity $\eta$ [Pa$\cdot$s]')
    ax.set_title(r'(a) $\eta(\dot{\gamma})$')
    ax.legend(loc='best')
    ax.set_ylim(1e-3, 1)

    # (b) First normal stress coefficient
    ax = axes[1]
    for fluid, c, ls, label in [(fluid_2, c2, ls2, '2% PAM'),
                                  (fluid_5, c5, ls5, '5% PAM')]:
        psi1 = fluid.Psi1(gdot)
        ax.loglog(gdot, psi1, color=c, ls=ls, label=label)
        # Zero-shear asymptote Psi1_0 = 2 eta_p lambda
        Psi1_0 = 2 * fluid.eta_p * fluid.lambda_
        ax.axhline(Psi1_0, color=c, ls=':', lw=0.8, alpha=0.6)

    ax.set_xlabel(r'Shear rate $\dot{\gamma}$ [s$^{-1}$]')
    ax.set_ylabel(r'$\Psi_1$ [Pa$\cdot$s$^2$]')
    ax.set_title(r'(b) $\Psi_1(\dot{\gamma})$')
    ax.legend(loc='best')

    # (c) Second normal stress coefficient (magnitude)
    ax = axes[2]
    for fluid, c, ls, label in [(fluid_2, c2, ls2, '2% PAM'),
                                  (fluid_5, c5, ls5, '5% PAM')]:
        psi2 = fluid.Psi2(gdot)
        ax.loglog(gdot, np.abs(psi2), color=c, ls=ls, label=label)
        # Annotation: Psi2/Psi1 = -alpha/2
        Psi2_0 = fluid.alpha * fluid.eta_p * fluid.lambda_
        ax.axhline(Psi2_0, color=c, ls=':', lw=0.8, alpha=0.6)

    ax.set_xlabel(r'Shear rate $\dot{\gamma}$ [s$^{-1}$]')
    ax.set_ylabel(r'$|\Psi_2|$ [Pa$\cdot$s$^2$]')
    ax.set_title(r'(c) $|\Psi_2(\dot{\gamma})|$')
    ax.legend(loc='best')

    # Add ratio annotation
    ax.text(0.95, 0.05, r'$\Psi_2/\Psi_1 = -\alpha/2$',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    fig.tight_layout()
    savefig(fig, 'fig2_material_functions')


# =====================================================================
# FIG 3: PARAMETER HIERARCHY
# =====================================================================
def fig3_parameter_hierarchy():
    r"""
    Log-scale bar chart showing the parameter hierarchy:
        epsilon << De_sq << De < 1 << Wi
    for both 2% and 5% PAM at representative operating conditions.

    This is the central organising element of the paper's argument:
    De_sq is the *right* small parameter for squeeze-dominated lubrication.
    """
    print("\n--- Fig 3: Parameter hierarchy ---")

    # Representative operating conditions (from Table in Section 6)
    # Slipper bearing: R_out = 20mm, h_0 = 10 um, V_T = 5 m/s, h_dot ~ 0.5 mm/s
    h_0 = 10e-6        # m
    L = 10e-3           # R_out - R_in = land width
    V_T = 5.0           # m/s  (tangential sliding)
    h_dot = 0.5e-3      # m/s  (squeeze rate, typical)

    fluids = [
        ('2% PAM', create_PAM_2pct()),
        ('5% PAM', create_PAM_5pct()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for idx, (name, fluid) in enumerate(fluids):
        ax = axes[idx]
        lam = fluid.lambda_

        # Compute dimensionless parameters
        epsilon = h_0 / L                           # thin-film ratio
        De_sq = lam * abs(h_dot) / h_0              # squeeze Deborah number
        Wi = lam * V_T / h_0                        # Weissenberg number
        De = epsilon * Wi                            # (conventional) Deborah number

        params = {
            r'$\varepsilon = h_0/L$': epsilon,
            r'$\mathrm{De}_{sq} = \lambda|\dot{h}|/h_0$': De_sq,
            r'$\mathrm{De} = \varepsilon\,\mathrm{Wi}$': De,
            r'$\mathrm{Wi} = \lambda V_T/h_0$': Wi,
        }

        labels = list(params.keys())
        values = list(params.values())
        colors = ['#95a5a6', '#2ecc71', '#e67e22', '#e74c3c']

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black',
                       linewidth=0.6, height=0.55)

        # Value annotations
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(val * 1.3, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2e}' if val < 0.01 else f'{val:.3f}',
                        va='center', fontsize=9, fontweight='bold')

        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Magnitude')
        ax.set_title(f'({chr(97+idx)}) {name}')

        # Reference line at 1
        ax.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5)
        ax.text(1.0, len(labels) - 0.3, '$O(1)$', fontsize=8, ha='center',
                va='bottom', alpha=0.6)

        # Reference line at 0.1 (validity threshold)
        ax.axvline(0.1, color='green', ls=':', lw=1, alpha=0.5)
        ax.text(0.1, -0.5, r'$\mathrm{De}_{sq}$ validity', fontsize=7,
                ha='center', va='top', color='green', alpha=0.7)

        ax.set_xlim(1e-5, 1e3)
        ax.invert_yaxis()

        # Print values
        print(f"  {name}: eps={epsilon:.2e}, De_sq={De_sq:.4f}, "
              f"De={De:.4f}, Wi={Wi:.1f}")

    fig.suptitle('Parameter hierarchy at representative operating conditions',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    savefig(fig, 'fig3_parameter_hierarchy')


# =====================================================================
# FIG 4: GRID CONVERGENCE
# =====================================================================
def analytical_hydrostatic_pressure(r, R_in, R_out, p_supply):
    """Analytical log-pressure for annular hydrostatic bearing."""
    return p_supply * np.log(R_out / r) / np.log(R_out / R_in)


def analytical_hydrostatic_load(R_in, R_out, p_supply):
    """Analytical load from integration of log-pressure."""
    integrand = lambda r: analytical_hydrostatic_pressure(r, R_in, R_out, p_supply) * r
    integral, _ = quad(integrand, R_in, R_out)
    return 2 * np.pi * integral


def fig4_grid_convergence():
    """
    Grid convergence study for Newtonian hydrostatic case.
    Log-log plot of load error vs Nr with O(h^2) reference.
    """
    print("\n--- Fig 4: Grid convergence ---")

    R_in, R_out = 0.008, 0.020
    h_0 = 5e-6
    p_supply = 200e5
    eta = 0.050

    F_analytical = analytical_hydrostatic_load(R_in, R_out, p_supply)
    geometry = SlipperGeometry(R_in=R_in, R_out=R_out, h_0=h_0,
                               alpha_x=0.0, alpha_y=0.0)
    conditions = OperatingConditions(p_supply=p_supply, V_T=0.0,
                                     omega_s=0.0, h_dot=0.0)

    grid_sizes = [10, 20, 40, 80, 160]
    errors_load = []
    errors_pressure = []

    print(f"  F_analytical = {F_analytical:.4f} N")
    for Nr in grid_sizes:
        Ntheta = max(Nr, 20)
        mesh = PolarMesh(geometry, Nr=Nr, Ntheta=Ntheta)
        result = solve_newtonian(mesh, geometry, conditions, eta)

        p_exact = analytical_hydrostatic_pressure(mesh.R, R_in, R_out, p_supply)
        p_err = np.sqrt(np.mean((result.pressure - p_exact)**2)) / p_supply
        F_err = abs(result.F_total - F_analytical) / F_analytical

        errors_load.append(F_err)
        errors_pressure.append(p_err)
        print(f"    Nr={Nr:4d}: F_err={F_err:.2e}, p_err_rms={p_err:.2e}")

    # Convergence rate
    h = 1.0 / np.array(grid_sizes)
    log_h = np.log(h[:-1] / h[1:])
    log_e = np.log(np.array(errors_load[:-1]) / np.array(errors_load[1:]))
    rates = log_e / log_h
    print(f"  Convergence rates: {[f'{r:.2f}' for r in rates]}")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(grid_sizes, errors_load, 'bo-', ms=6, label='Load error')
    ax.loglog(grid_sizes, errors_pressure, 'rs-', ms=5, label='Pressure error (RMS)')

    # O(h^2) reference
    ref = errors_load[0] * (grid_sizes[0] / np.array(grid_sizes, dtype=float))**2
    ax.loglog(grid_sizes, ref, 'k--', lw=1, alpha=0.6, label=r'$O(h^2)$ reference')

    ax.set_xlabel(r'Number of radial nodes $N_r$')
    ax.set_ylabel('Relative error')
    ax.legend()
    ax.set_xlim(8, 200)

    savefig(fig, 'fig4_grid_convergence')


# =====================================================================
# FIG 5: UCM VALIDATION (TICHY 1996)
# =====================================================================
def fig5_ucm_validation():
    """
    UCM limit: F_VE/F_Newton vs De_sq.
    Solver (alpha -> 0) vs Tichy (1996) analytical: 1 - 2*De_sq.
    """
    print("\n--- Fig 5: UCM validation (Tichy 1996) ---")

    ETA_0 = 1.0
    ETA_S = 0.01
    ETA_P = ETA_0 - ETA_S
    LAMBDA = 0.01
    ALPHA_G = 1e-6

    R_IN = 0.25e-3
    R_OUT = 0.020
    H0 = 10e-6
    NR, NTHETA = 40, 8

    fluid = GiesekusFluid(eta_s=ETA_S, eta_p=ETA_P,
                          lambda_=LAMBDA, alpha=ALPHA_G)
    geometry = SlipperGeometry(R_in=R_IN, R_out=R_OUT, h_0=H0)
    mesh = PolarMesh(geometry, Nr=NR, Ntheta=NTHETA)
    config = SolverConfig(max_iter=50, tol=1e-6, omega=0.6, verbose=False,
                          include_memory=True, include_normal_stress=True)

    De_sq_values = np.array([0.005, 0.01, 0.02, 0.04, 0.06,
                              0.08, 0.10, 0.12, 0.15])
    ratios = []
    for De_sq in De_sq_values:
        h_dot = -De_sq * H0 / LAMBDA
        conditions = OperatingConditions(p_supply=0.0, V_T=0.0, h_dot=h_dot)
        r_ve = solve_viscoelastic(mesh, geometry, conditions, fluid, config)
        r_nw = solve_newtonian(mesh, geometry, conditions, ETA_0)
        ratio = r_ve.F_total / r_nw.F_total if abs(r_nw.F_total) > 1e-15 else 1.0
        ratios.append(ratio)
        print(f"    De_sq={De_sq:.3f}: F_VE/F_N={ratio:.4f}  "
              f"Tichy={1-2*De_sq:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 4))
    De_line = np.linspace(0, 0.18, 100)
    ax.plot(De_line, 1 - 2*De_line, 'k--', lw=2,
            label=r'Tichy (1996): $1 - 2\,\mathrm{De}_{sq}$')
    ax.plot(De_sq_values, ratios, 'ro-', ms=6,
            label=r'Solver ($\alpha \to 0$, UCM limit)')
    ax.set_xlabel(r'$\mathrm{De}_{sq} = \lambda|\dot{h}|/h_0$')
    ax.set_ylabel(r'$F_{\mathrm{VE}} / F_{\mathrm{Newton}}$')
    ax.legend()
    ax.set_xlim(0, 0.17)
    ax.set_ylim(0.6, 1.05)

    savefig(fig, 'fig5_ucm_validation')


# =====================================================================
# FIG 6: PTT (1985) BOGER FLUID VALIDATION  [CRITICAL]
# =====================================================================
def fig6_ptt1985_validation():
    """
    Boger fluid squeeze-flow validation: omega(tau) for 3 Wi conditions.
    Experimental data from Phan-Thien et al. (1985).
    """
    print("\n--- Fig 6: PTT (1985) Boger fluid validation ---")

    # Fluid S1 parameters
    ETA_S = 15.8
    ETA_P = 39.7
    ETA_0 = ETA_S + ETA_P
    LAMBDA = 0.93
    ALPHA_R = ETA_P / ETA_S
    ALPHA_G = 0.001
    R_DISK = 0.0254
    H0 = 0.001

    fluid = GiesekusFluid(eta_s=ETA_S, eta_p=ETA_P,
                          lambda_=LAMBDA, alpha=ALPHA_G)

    # Embedded experimental data
    datasets = {
        'Wi=0.005': dict(
            tau=np.array([0.101299588, 0.130424640, 0.150952149, 0.174822443,
                          0.200601256, 0.228289739, 0.250243457, 0.275058802,
                          0.299875298, 0.325157988, 0.349960671, 0.375229548,
                          0.400019569, 0.426710050, 0.450042782, 0.475269069,
                          0.500476938]),
            omega=np.array([28.39314560, 31.54906934, 34.45712356, 37.60882661,
                            41.24399032, 45.12165175, 49.47663465, 54.79777148,
                            59.87794537, 67.36813227, 75.33986133, 85.72160340,
                            96.34392470, 109.1364472, 124.8178389, 144.1152094,
                            167.2679868]),
            Wi=0.005, V=0.005 * H0 / LAMBDA),
        'Wi=0.025': dict(
            tau=np.array([0.100832243, 0.151445969, 0.200143120, 0.249306464,
                          0.298465204, 0.349504837, 0.399576396, 0.435787600,
                          0.480086409, 0.500079809]),
            omega=np.array([26.22409553, 31.08402623, 37.14723679, 45.62046036,
                            55.05753566, 70.76118195, 89.11465308, 108.9027745,
                            135.6853439, 150.4001980]),
            Wi=0.025, V=0.025 * H0 / LAMBDA),
        'Wi=0.125': dict(
            tau=np.array([0.101803769, 0.136657829, 0.175326624, 0.200154631,
                          0.234516021, 0.269355117, 0.326136420, 0.374308670,
                          0.430550109, 0.461497347, 0.485300878, 0.500053334]),
            omega=np.array([22.85138189, 26.73479881, 32.06706290, 34.73760748,
                            41.75315880, 48.76909382, 62.54964105, 78.49194808,
                            105.2841099, 126.9957141, 144.1232671, 155.9423454]),
            Wi=0.125, V=0.125 * H0 / LAMBDA),
    }

    # Analytical asymptotes
    def omega_upper(tau):
        return 6 * (1 + ALPHA_R) / (1 - tau)**3

    def omega_lower(tau):
        return 6 / (1 - tau)**3

    def omega_ptt_perturbation(tau, Wi):
        De_sq = Wi / (1 - tau)
        return omega_upper(tau) * (1 - 2 * De_sq)

    # Solve for each condition
    def solve_at_instant(V, h):
        R_in = 0.25e-3
        p_inner = 3.0 * fluid.eta_0 * V / h**3 * (R_DISK**2 - R_in**2)
        geometry = SlipperGeometry(R_in=R_in, R_out=R_DISK, h_0=h)
        mesh = PolarMesh(geometry, Nr=40, Ntheta=8)
        conditions = OperatingConditions(p_supply=p_inner, V_T=0.0, h_dot=-V)
        config = SolverConfig(max_iter=60, tol=1e-6, omega=0.5, verbose=False)
        result = solve_viscoelastic(mesh, geometry, conditions, fluid, config)
        result_N = solve_newtonian(mesh, geometry, conditions, fluid.eta_0)
        return result.F_total, result.F_GNF, result_N.F_total, result.converged

    # Run solver for each Wi
    N_tau = 15
    condition_results = {}
    for key, ds in datasets.items():
        Wi, V = ds['Wi'], ds['V']
        tau_exp, omega_exp = ds['tau'], ds['omega']
        # Slightly inset the tau range to avoid boundary divergence
        tau_mod = np.linspace(tau_exp.min(), tau_exp.max() * 0.995, N_tau)
        F_norm = np.pi * ETA_S * V * R_DISK**4 / (4 * H0**3)

        omega_vis = np.zeros(N_tau)
        omega_gnf = np.zeros(N_tau)
        converged = np.zeros(N_tau, dtype=bool)
        print(f"  {key}: V={V*1e3:.4f} mm/s, {N_tau} time steps...")

        for i, tau in enumerate(tau_mod):
            h = H0 * (1 - tau)
            if h < 1e-5:
                continue
            try:
                Ft, Fg, Fn, c = solve_at_instant(V, h)
                omega_vis[i] = Ft / F_norm
                omega_gnf[i] = Fg / F_norm
                converged[i] = c
            except Exception as e:
                print(f"    [WARN] tau={tau:.3f}: {e}")

        # Error metric: only use converged points
        mean_err, max_err = None, None
        valid = converged & (omega_vis > 0)
        if np.sum(valid) >= 2:
            f_interp = interp1d(tau_mod[valid], omega_vis[valid],
                                kind='linear', fill_value='extrapolate')
            # Only evaluate at experimental points within the valid range
            tau_clip = tau_exp[tau_exp <= tau_mod[valid].max()]
            omega_clip = omega_exp[tau_exp <= tau_mod[valid].max()]
            if len(tau_clip) >= 2:
                omega_mod_at_exp = f_interp(tau_clip)
                errors = np.abs(omega_mod_at_exp - omega_clip) / omega_clip * 100
                mean_err = np.mean(errors)
                max_err = np.max(errors)
            print(f"    Converged: {np.sum(valid)}/{N_tau}  "
                  f"Mean error: {mean_err:.1f}%, Max: {max_err:.1f}%"
                  if mean_err else f"    Converged: {np.sum(valid)}/{N_tau}")

        # Filter non-converged for plotting
        tau_plot = tau_mod[valid]
        omega_vis_plot = omega_vis[valid]
        omega_gnf_plot = omega_gnf[valid]

        condition_results[key] = dict(
            Wi=Wi, tau_exp=tau_exp, omega_exp=omega_exp,
            tau_mod=tau_plot, omega_vis=omega_vis_plot,
            omega_gnf=omega_gnf_plot,
            mean_err=mean_err, max_err=max_err)

    # Plot: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    tau_smooth = np.linspace(0.05, 0.55, 300)

    for idx, (key, r) in enumerate(condition_results.items()):
        ax = axes[idx]
        Wi = r['Wi']

        # Shade De_sq < 0.1 validity region
        tau_star = 1.0 - 10.0 * Wi  # tau where De_sq(tau) = 0.1
        if tau_star > tau_smooth[0]:
            ax.axvspan(tau_smooth[0], min(tau_star, tau_smooth[-1]),
                       alpha=0.10, color='green', zorder=0)
            if tau_star < tau_smooth[-1]:
                ax.axvline(tau_star, color='green', ls=':', lw=1.0, alpha=0.6)
            # Label: position inside the shaded region
            label_x = min(tau_star, tau_smooth[-1]) - 0.02
            ax.text(label_x, 12, r'$\mathrm{De}_{sq} < 0.1$',
                    fontsize=7.5, color='green', ha='right', va='bottom',
                    alpha=0.8)

        # Asymptotes
        ax.plot(tau_smooth, omega_upper(tau_smooth), 'k--', lw=1.2,
                label=r'$\omega_{\mathrm{upper}}$ (Stefan, $\eta_0$)')
        ax.plot(tau_smooth, omega_lower(tau_smooth), 'k:', lw=1.2,
                label=r'$\omega_{\mathrm{lower}}$ (Stefan, $\eta_s$)')

        # Experimental
        ax.plot(r['tau_exp'], r['omega_exp'], 'ko', ms=4.5, zorder=5,
                label='Experiment (S1)')

        # Solver: full viscoelastic
        ax.plot(r['tau_mod'], r['omega_vis'], 'r-o', lw=1.8, ms=3, zorder=4,
                label='Solver (viscoelastic)')

        # PTT perturbation
        ax.plot(tau_smooth, omega_ptt_perturbation(tau_smooth, Wi),
                'g-.', lw=1.2, alpha=0.8,
                label=r'PTT: $1 - 2\,\mathrm{De}_{sq}$')

        ax.set_xlabel(r'$\tau = tV/h_0$')
        if idx == 0:
            ax.set_ylabel(r'$\omega = F / (\pi\eta_s V R^4 / 4h_0^3)$')
        ax.set_title(f'Wi = {Wi}')
        ax.legend(fontsize=7, loc='upper left')

        # Error annotation
        if r['mean_err'] is not None:
            ax.text(0.97, 0.05,
                    f"Mean err: {r['mean_err']:.1f}%\n"
                    f"Max err: {r['max_err']:.1f}%",
                    transform=ax.transAxes, fontsize=7.5,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    fig.tight_layout()
    savefig(fig, 'fig6_ptt1985_validation')


# =====================================================================
# FIG 7: FLUX DECOMPOSITION vs De_sq
# =====================================================================
def fig7_flux_decomposition():
    """
    Stacked area chart showing flux component contributions vs De_sq.
    Uses 5% PAM fluid in slipper geometry.
    """
    print("\n--- Fig 7: Flux decomposition vs De_sq ---")

    fluid = create_PAM_5pct()
    geometry = SlipperGeometry(R_in=0.010, R_out=0.020, h_0=10e-6,
                               alpha_x=0.0, alpha_y=0.0)
    mesh = PolarMesh(geometry, Nr=50, Ntheta=50)
    config = SolverConfig(max_iter=80, tol=1e-6, omega=0.5, verbose=False)

    # Sweep De_sq via h_dot
    De_sq_targets = np.array([0.001, 0.005, 0.01, 0.02, 0.04,
                               0.06, 0.08, 0.10, 0.12, 0.15])
    De_sq_actual = []
    flux_data = {k: [] for k in ['GNF', 'memory', 'N1', 'hoop', 'alpha']}

    for De_sq in De_sq_targets:
        h_dot = -De_sq * geometry.h_0 / fluid.lambda_
        conditions = OperatingConditions(
            p_supply=100e5, V_T=5.0, omega_s=0.0, h_dot=h_dot)

        result = solve_viscoelastic(mesh, geometry, conditions, fluid, config)

        if result.fluxes is not None:
            ratios = result.fluxes.flux_ratios()
            De_sq_actual.append(De_sq)
            for k in flux_data:
                flux_data[k].append(ratios.get(k, ratios.get(k.upper(), 0)) * 100)
            print(f"    De_sq={De_sq:.3f}: GNF={ratios['GNF']*100:.1f}%, "
                  f"mem={ratios['memory']*100:.1f}%, "
                  f"N1={ratios['N1']*100:.1f}%")

    De_sq_actual = np.array(De_sq_actual)

    # Single-panel log-scale plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    labels = [r'$Q_r^{\mathrm{GNF}}$', r'$Q_r^{\mathrm{mem}}$',
              r'$Q_r^{N_1}$', r'$Q_r^{\mathrm{hoop}}$', r'$Q_r^{(\alpha)}$']
    colors = ['#2ca02c', '#1f77b4', '#9b59b6', '#e74c3c', '#f39c12']
    keys = ['GNF', 'memory', 'N1', 'hoop', 'alpha']

    for i, (k, label) in enumerate(zip(keys, labels)):
        vals = np.array(flux_data[k])
        if np.any(vals > 0.01):
            ax.semilogy(De_sq_actual, vals, 'o-', color=colors[i],
                        lw=2, ms=6, label=label)

    # Horizontal reference lines
    ax.axhline(1.0, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(0.1, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.text(0.152, 1.15, r'1\%', fontsize=8, color='grey', ha='right')
    ax.text(0.152, 0.115, r'0.1\%', fontsize=8, color='grey', ha='right')

    # Validity shading
    ax.axvspan(0, 0.1, alpha=0.06, color='green', zorder=0)
    ax.axvline(0.1, color='green', ls=':', lw=1.0, alpha=0.4)

    ax.set_xlabel(r'$\mathrm{De}_{sq}$', fontsize=13)
    ax.set_ylabel('Flux contribution [%]', fontsize=13)
    ax.set_ylim(5e-3, 200)
    ax.legend(fontsize=10, loc='center right')
    ax.grid(True, alpha=0.2, which='both')

    fig.tight_layout()
    savefig(fig, 'fig7_flux_decomposition')


# =====================================================================
# FIG 8: LOAD RATIO FOR BOTH FLUIDS
# =====================================================================
def fig8_load_ratio_both_fluids():
    """
    F_VE/F_GNF vs De_sq for 2% and 5% PAM.
    PTT reference line (1 - 2*De_sq) and valid regime shading.
    """
    print("\n--- Fig 8: Load ratio for both fluids ---")

    fluids = [('2% PAM', create_PAM_2pct(), 'tab:blue', 'o'),
              ('5% PAM', create_PAM_5pct(), 'tab:red', 's')]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # PTT reference
    De_line = np.linspace(0, 0.18, 100)
    ax.plot(De_line, 1 - 2*De_line, 'k--', lw=2,
            label=r'PTT: $1 - 2\,\mathrm{De}_{sq}$')

    # Valid regime shading
    ax.axvspan(0, 0.1, color='green', alpha=0.05)
    ax.axvline(0.1, color='green', ls=':', lw=1, alpha=0.5)

    for idx, (label, fluid, color, marker) in enumerate(fluids):
        geometry = SlipperGeometry(R_in=0.010, R_out=0.020, h_0=10e-6,
                                   alpha_x=0.0, alpha_y=0.0)
        mesh = PolarMesh(geometry, Nr=50, Ntheta=50)
        config = SolverConfig(max_iter=80, tol=1e-6, omega=0.5, verbose=False)

        De_sq_values = np.array([0.005, 0.01, 0.02, 0.04, 0.06,
                                  0.08, 0.10, 0.12, 0.15])
        ratios = []

        print(f"  {label}:")
        for De_sq in De_sq_values:
            h_dot = -De_sq * geometry.h_0 / fluid.lambda_
            conditions = OperatingConditions(
                p_supply=100e5, V_T=5.0, omega_s=0.0, h_dot=h_dot)

            r_ve = solve_viscoelastic(mesh, geometry, conditions, fluid, config)

            ratio = r_ve.F_total / r_ve.F_GNF if abs(r_ve.F_GNF) > 1e-15 else 1.0
            ratios.append(ratio)
            ptt = 1 - 2*De_sq
            print(f"    De_sq={De_sq:.3f}: F_VE/F_GNF={ratio:.4f}  "
                  f"PTT={ptt:.4f}")

        if idx == 0:
            # 2% PAM: filled circles
            ax.plot(De_sq_values, ratios, f'-{marker}', color=color, ms=6,
                    markerfacecolor=color, markeredgecolor=color, lw=1.5,
                    label=label)
        else:
            # 5% PAM: open squares
            ax.plot(De_sq_values, ratios, f'-{marker}', color=color, ms=6,
                    markerfacecolor='white', markeredgecolor=color,
                    markeredgewidth=1.5, lw=1.5, label=label)

    ax.set_xlabel(r'$\mathrm{De}_{sq} = \lambda|\dot{h}|/h_0$')
    ax.set_ylabel(r'$F_{\mathrm{VE}} / F_{\mathrm{GNF}}$')
    ax.legend()
    ax.set_xlim(0, 0.17)
    ax.set_ylim(0.55, 1.05)

    # Annotate valid regime
    ax.text(0.05, 0.97, r'Valid regime ($\mathrm{De}_{sq} < 0.1$)',
            transform=ax.transAxes, fontsize=8, va='top', color='green',
            alpha=0.8)

    savefig(fig, 'fig8_load_ratio_both_fluids')


# =====================================================================
# FIG 9: PRESSURE FIELD COMPARISON
# =====================================================================
def fig9_pressure_comparison():
    """
    Four-panel polar contour: Newtonian, GNF, full Giesekus pressure
    fields plus (Giesekus - Newtonian) difference.  Polar projection
    mirrors the annular slipper geometry for physical intuition.
    """
    print("\n--- Fig 9: Pressure field comparison (polar) ---")

    fluid = create_PAM_5pct()
    geometry = SlipperGeometry(R_in=0.010, R_out=0.020, h_0=10e-6,
                               alpha_x=0.0, alpha_y=0.0)
    mesh = PolarMesh(geometry, Nr=60, Ntheta=60)
    conditions = OperatingConditions(
        p_supply=100e5, V_T=5.0, omega_s=0.0, h_dot=-0.0005)
    config = SolverConfig(max_iter=80, tol=1e-6, omega=0.5, verbose=False)

    r_nwt = solve_newtonian(mesh, geometry, conditions, fluid.eta_0)
    r_gnf = solve_GNF(mesh, geometry, conditions, fluid, config)
    r_ve  = solve_viscoelastic(mesh, geometry, conditions, fluid, config)

    R, THETA = mesh.R, mesh.THETA
    F_nwt = r_nwt.F_total / 1e3
    F_gnf = r_gnf.F_total / 1e3
    F_ve  = r_ve.F_total / 1e3

    # Percentage changes
    pct_gnf = (F_gnf - F_nwt) / F_nwt * 100
    pct_ve  = (F_ve - F_nwt) / F_nwt * 100

    # --- Panels (a)-(c): pressure fields on common colour scale ---
    p_nwt = r_nwt.pressure / 1e6
    p_gnf = r_gnf.pressure / 1e6
    p_ve  = r_ve.pressure / 1e6

    p_all = np.concatenate([p_nwt.ravel(), p_gnf.ravel(), p_ve.ravel()])
    vmin, vmax = 0, np.max(p_all) * 1.02
    levels_p = np.linspace(vmin, vmax, 30)

    # --- Panel (d): difference field ---
    dp = (r_ve.pressure - r_nwt.pressure) / 1e6
    vabs = max(abs(dp.min()), abs(dp.max()))
    if vabs < 1e-12:
        vabs = 1.0
    levels_d = np.linspace(-vabs, vabs, 30)

    # --- Figure layout: 2x2 polar with gridspec for colourbar control ---
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 12))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  hspace=0.35, wspace=0.30)

    axes_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axes_all = [fig.add_subplot(gs[r, c], projection='polar')
                for r, c in axes_pos]
    ax_cbar_p = fig.add_subplot(gs[0, 2])  # pressure colourbar
    ax_cbar_d = fig.add_subplot(gs[1, 2])  # difference colourbar

    # Radial ticks (in mm)
    r_in_mm  = geometry.R_in * 1e3
    r_out_mm = geometry.R_out * 1e3
    r_mid_mm = (r_in_mm + r_out_mm) / 2
    rticks = [r_in_mm, r_mid_mm, r_out_mm]

    def style_polar_ax(ax):
        ax.set_rlabel_position(225)
        ax.set_rticks(rticks)
        ax.tick_params(labelsize=7)
        ax.set_rlim(0, r_out_mm * 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Panels (a)-(c): pressure fields
    panels_p = [
        (p_nwt, f'(a) Newtonian ($\\eta = \\eta_0$)\n$F$ = {F_nwt:.2f} kN'),
        (p_gnf, f'(b) GNF ($\\eta = \\eta(\\dot{{\\gamma}})$)\n'
                f'$F$ = {F_gnf:.2f} kN ({pct_gnf:+.0f}%)'),
        (p_ve,  f'(c) Full Giesekus\n'
                f'$F$ = {F_ve:.2f} kN ({pct_ve:+.0f}%)'),
    ]

    for ax, (p_field, title) in zip(axes_all[:3], panels_p):
        cf = ax.contourf(THETA, R * 1e3, p_field, levels=levels_p,
                         cmap='viridis', extend='both')
        ax.set_title(title, fontsize=9, pad=12)
        style_polar_ax(ax)

    # Pressure colourbar (right of top row)
    cbar_p = fig.colorbar(cf, cax=ax_cbar_p)
    cbar_p.set_label('$p$ [MPa]', fontsize=10)
    cbar_p.ax.tick_params(labelsize=8)

    # Panel (d): difference field
    ax_d = axes_all[3]
    cd = ax_d.contourf(THETA, R * 1e3, dp, levels=levels_d,
                       cmap='RdBu_r', extend='both')
    dF = (F_ve - F_nwt)
    ax_d.set_title(f'(d) $\\Delta p$ = Giesekus $-$ Newtonian\n'
                   f'$\\Delta F$ = {dF:.2f} kN ({pct_ve:+.1f}%)',
                   fontsize=9, pad=12)
    style_polar_ax(ax_d)

    # Difference colourbar (right of bottom row)
    cbar_d = fig.colorbar(cd, cax=ax_cbar_d)
    cbar_d.set_label('$\\Delta p$ [MPa]', fontsize=10)
    cbar_d.ax.tick_params(labelsize=8)

    savefig(fig, 'fig9_pressure_comparison')


# =====================================================================
# FIG 10: ERROR BUDGET VISUALISATION
# =====================================================================
def fig10_error_budget():
    r"""
    Stacked horizontal bar chart showing the error budget for each
    approximation in the model, evaluated at representative conditions
    for both 2% and 5% PAM.

    Error sources:
      1. Spatial discretisation (grid): from Richardson extrapolation
      2. De_sq truncation: O(De_sq^2) terms omitted
      3. Quasi-steady shear rate: Poiseuille-to-Couette ratio P
      4. Advective memory: O(De) = O(epsilon * Wi)
      5. Single-mode Giesekus: model fidelity

    Numbers adapted from thesis Ch. 4, Table 4.6 and Section 4.8.
    """
    print("\n--- Fig 10: Error budget ---")

    # Error estimates for each source (% of load)
    # Columns: [2% PAM, 5% PAM]
    error_sources = {
        r'Spatial discretisation ($N_r=50$)': [0.15, 0.15],
        r'$\mathrm{De}_{sq}^2$ truncation':   [0.5, 2.8],
        r'Quasi-steady $\dot{\gamma}$ ($\mathcal{P} \ll 1$)': [0.3, 0.8],
        r'Advective memory ($O(\mathrm{De})$)': [0.8, 10.7],
        r'Single-mode model fidelity':         [2.0, 5.0],
    }

    labels = list(error_sources.keys())
    err_2 = np.array([v[0] for v in error_sources.values()])
    err_5 = np.array([v[1] for v in error_sources.values()])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y = np.arange(len(labels))
    bar_h = 0.35
    bars_2 = ax.barh(y + bar_h/2, err_2, bar_h, label='2% PAM',
                     color='tab:blue', edgecolor='black', linewidth=0.5,
                     alpha=0.85)
    bars_5 = ax.barh(y - bar_h/2, err_5, bar_h, label='5% PAM',
                     color='tab:red', edgecolor='black', linewidth=0.5,
                     alpha=0.85)

    # Value annotations
    for bars in [bars_2, bars_5]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{w:.1f}%', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Estimated error contribution [%]')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(err_5.max(), err_2.max()) * 1.4)
    ax.invert_yaxis()

    # Highlight the dominant source
    ax.axvline(5.0, color='orange', ls='--', lw=1, alpha=0.5)
    ax.text(5.0, len(labels) - 0.2, '5% threshold', fontsize=7,
            ha='center', va='bottom', color='orange', alpha=0.8)

    # Total RSS error
    rss_2 = np.sqrt(np.sum(err_2**2))
    rss_5 = np.sqrt(np.sum(err_5**2))
    ax.text(0.97, 0.05,
            f'RSS total: 2% PAM = {rss_2:.1f}%, 5% PAM = {rss_5:.1f}%',
            transform=ax.transAxes, fontsize=8.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

    print(f"  2% PAM RSS: {rss_2:.1f}%")
    print(f"  5% PAM RSS: {rss_5:.1f}%")

    fig.tight_layout()
    savefig(fig, 'fig10_error_budget')


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate Paper 1 figures for JNNFM submission')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Generate only specific figures (e.g., fig2 fig6)')
    args = parser.parse_args()

    setup_paper_style()
    os.makedirs(OUTDIR, exist_ok=True)

    all_figs = {
        'fig2': ('Material functions', fig2_material_functions),
        'fig3': ('Parameter hierarchy', fig3_parameter_hierarchy),
        'fig4': ('Grid convergence', fig4_grid_convergence),
        'fig5': ('UCM validation', fig5_ucm_validation),
        'fig6': ('PTT 1985 validation', fig6_ptt1985_validation),
        'fig7': ('Flux decomposition', fig7_flux_decomposition),
        'fig8': ('Load ratio both fluids', fig8_load_ratio_both_fluids),
        'fig9': ('Pressure comparison', fig9_pressure_comparison),
        'fig10': ('Error budget', fig10_error_budget),
    }

    figs_to_run = args.only if args.only else list(all_figs.keys())

    print("=" * 60)
    print("Paper 1 Figure Generator")
    print("Target: J. Non-Newtonian Fluid Mech.")
    print(f"Output: {OUTDIR}")
    print("=" * 60)

    t0 = clk.time()
    for key in figs_to_run:
        if key not in all_figs:
            print(f"  Unknown figure: {key}")
            continue
        desc, func = all_figs[key]
        t1 = clk.time()
        try:
            func()
            print(f"  {key} ({desc}): {clk.time()-t1:.1f} s")
        except Exception as e:
            print(f"  {key} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTotal: {clk.time()-t0:.1f} s")
    print(f"Figures in: {OUTDIR}")


if __name__ == '__main__':
    main()
