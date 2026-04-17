#!/usr/bin/env python
r"""
Paper 1 Figure Generator
=========================

Generates the four validation figures for:
    "Viscoelastic Reynolds equation for thin-film lubrication via squeeze
     Deborah number expansion with the Giesekus constitutive model"

Target: Journal of Non-Newtonian Fluid Mechanics

Figures generated:
    Fig 2:  Material functions eta, Psi1, Psi2 for representative Giesekus fluids
    Fig 4:  Grid convergence study (Newtonian hydrostatic; 2nd-order verification)
    Fig 5:  UCM validation against Tichy (1996): 1 - 2 De_sq
    Fig 6:  PTT (1985) Boger fluid squeeze-flow validation

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
from scipy.integrate import quad
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    GiesekusFluid,
    SlipperGeometry,
    PolarMesh,
    OperatingConditions,
    SolverConfig,
    solve_newtonian,
    solve_viscoelastic,
)

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Output directory
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'figures', 'paper1')


# =====================================================================
# REPRESENTATIVE FLUID PARAMETERS (inline; no factory functions exported)
# =====================================================================
# Polyacrylamide (PAM) in water-glycol solutions used only as two concrete
# parameter sets for the material-function plot. These are not application-
# specific predictions; the theory paper uses them purely to illustrate the
# Giesekus constitutive response (Eqs. 7-10).
def _pam_2pct() -> GiesekusFluid:
    """2% PAM: eta_0 = 0.050 Pa.s, eta_s = 0.030 Pa.s, lambda = 1.0 ms, alpha = 0.25"""
    return GiesekusFluid(eta_s=0.030, eta_p=0.020, lambda_=1.0e-3, alpha=0.25)


def _pam_5pct() -> GiesekusFluid:
    """5% PAM: eta_0 = 0.150 Pa.s, eta_s = 0.030 Pa.s, lambda = 3.3 ms, alpha = 0.25"""
    return GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)


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
    fluid_2 = _pam_2pct()
    fluid_5 = _pam_5pct()

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

        mean_err, max_err = None, None
        valid = converged & (omega_vis > 0)
        if np.sum(valid) >= 2:
            f_interp = interp1d(tau_mod[valid], omega_vis[valid],
                                kind='linear', fill_value='extrapolate')
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

        tau_star = 1.0 - 10.0 * Wi
        if tau_star > tau_smooth[0]:
            ax.axvspan(tau_smooth[0], min(tau_star, tau_smooth[-1]),
                       alpha=0.10, color='green', zorder=0)
            if tau_star < tau_smooth[-1]:
                ax.axvline(tau_star, color='green', ls=':', lw=1.0, alpha=0.6)
            label_x = min(tau_star, tau_smooth[-1]) - 0.02
            ax.text(label_x, 12, r'$\mathrm{De}_{sq} < 0.1$',
                    fontsize=7.5, color='green', ha='right', va='bottom',
                    alpha=0.8)

        ax.plot(tau_smooth, omega_upper(tau_smooth), 'k--', lw=1.2,
                label=r'$\omega_{\mathrm{upper}}$ (Stefan, $\eta_0$)')
        ax.plot(tau_smooth, omega_lower(tau_smooth), 'k:', lw=1.2,
                label=r'$\omega_{\mathrm{lower}}$ (Stefan, $\eta_s$)')

        ax.plot(r['tau_exp'], r['omega_exp'], 'ko', ms=4.5, zorder=5,
                label='Experiment (S1)')

        ax.plot(r['tau_mod'], r['omega_vis'], 'r-o', lw=1.8, ms=3, zorder=4,
                label='Solver (viscoelastic)')

        ax.plot(tau_smooth, omega_ptt_perturbation(tau_smooth, Wi),
                'g-.', lw=1.2, alpha=0.8,
                label=r'PTT: $1 - 2\,\mathrm{De}_{sq}$')

        ax.set_xlabel(r'$\tau = tV/h_0$')
        if idx == 0:
            ax.set_ylabel(r'$\omega = F / (\pi\eta_s V R^4 / 4h_0^3)$')
        ax.set_title(f'Wi = {Wi}')
        ax.legend(fontsize=7, loc='upper left')

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
        'fig4': ('Grid convergence', fig4_grid_convergence),
        'fig5': ('UCM validation', fig5_ucm_validation),
        'fig6': ('PTT 1985 validation', fig6_ptt1985_validation),
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
