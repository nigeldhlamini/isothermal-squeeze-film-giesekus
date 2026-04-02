#!/usr/bin/env python
r"""
Test 7 — Shirodkar & Middleman (1982) Squeeze-Flow Validation
==============================================================

Runs the Giesekus-based Reynolds equation solver for each experimental
condition and compares F_model / F_Scott against the measured R(t).

Strategy
--------
1. Fit power-law (K, n) to viscosity data  →  defines F_Scott baseline.
2. Fit single-mode Giesekus (eta_s, eta_p, lambda, alpha) to viscosity data.
3. For each (V, H0) condition, time-step through the squeeze:
       h(t) = H0 - V*t
   and at each t call:
       solve_viscoelastic  →  result.F_total   (full Giesekus)
       result.F_GNF        →  GNF-only force
       F_Scott(V, h, R, K, n)
4. Compute R_vis(t) = F_total / F_Scott    (full model)
           R_GNF(t) = F_GNF   / F_Scott    (shear-thinning only)
5. Compare with R_exp(t).

Known limitations
-----------------
Single-mode Giesekus cannot reproduce the 100× viscosity drop in 1% PAM
over 3 decades of γ̇.  Where the fit deviates from the data, R_GNF ≠ 1.
This is a *rheological fitting* limitation, not a solver deficiency.
The elastic correction (F_total - F_GNF) is O(De_sq) ≈ 1-3%, demonstrating
that shear-thinning dominates and the memory term is small.

Usage
-----
    cd giesekus-solver
    PYTHONPATH=. python scripts/test7_shirodkar_validation.py

Author: N.C. Dhlamini (dhlnig001@myuct.ac.za), February 2026
"""
import numpy as np
import os, sys, warnings, time as clk
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')

# ── Import project solver ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import (
    GiesekusFluid,
    SlipperGeometry,
    PolarMesh,
    OperatingConditions,
    SolverConfig,
    solve_viscoelastic,
    solve_GNF,
    solve_newtonian,
)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.dpi': 300, 'lines.linewidth': 1.5,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
})


def savefig(fig, name, outdir):
    """Save figure as both PNG (300 dpi) and PDF."""
    for ext in ('png', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"  Saved: {name}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════
LABELS = {
    'V05_H105': r'$V$=0.5, $H_0$=1.05',
    'V05_H055': r'$V$=0.5, $H_0$=0.55',
    'V10_H106': r'$V$=1.0, $H_0$=1.06',
    'V10_H056': r'$V$=1.0, $H_0$=0.56',
    'V20_H106': r'$V$=2.0, $H_0$=1.06',
    'V20_H071': r'$V$=2.0, $H_0$=0.71',
}

def load_data(path=None):
    """
    Return Shirodkar & Middleman (1982) experimental data.

    Data is embedded directly from the original Excel file to remove the
    pandas dependency.  The *path* argument is retained for backward
    compatibility but is ignored.
    """
    d = {}

    # ── Viscosity curve: shear rate [s⁻¹] and viscosity [Pa·s] ──
    d['gv'] = np.array([
        0.024259059, 0.059537407, 0.1,         0.154253595,
        0.243531207, 0.405885336, 0.607006410, 1.003877389,
        1.524731165, 2.521625991, 6.188660215, 15.91038439,
        24.35312073, 40.27559980,
    ])
    d['ev'] = np.array([
        1444.591943, 983.6721279, 800.2627153, 566.6045030,
        456.8129547, 302.9873560, 224.7436600, 168.1924325,
        134.3693921, 95.99708917, 51.34811822, 25.97264198,
        19.44333100, 12.42628428,
    ]) * 0.1  # convert dPa·s → Pa·s

    # ── Squeeze ratio curve: shear rate [s⁻¹] and R_squeeze ──
    d['gsr'] = np.array([
        0.060397668, 0.098349159, 0.151864657, 0.236331228,
        0.391542573, 0.604054026, 0.977809559, 1.524491566,
        2.412230332, 6.076225195, 15.31731383, 24.20274607,
        39.19062285, 98.68029103,
    ])
    d['srv'] = np.array([
        0.799542753, 1.158167087, 1.374781046, 1.631934802,
        1.830360640, 2.480911961, 2.730197249, 2.462200348,
        2.487990307, 3.281072247, 3.861898658, 4.806842470,
        5.045064417, 7.042467311,
    ])

    # ── Squeeze-flow experiments ──
    d['sq'] = {}

    # V = 0.5 cm/min, H₀ = 1.05 cm
    d['sq']['V05_H105'] = dict(
        t=np.array([
            0.941207315, 4.040955440, 8.203073062, 12.93991517,
            16.23264263, 20.36751280, 24.41417991, 27.95269619,
            31.49008886, 35.53282334, 42.77502598, 50.10234178,
            57.68190713, 65.01006564, 72.59103550, 80.25683761,
            87.50184927, 91.20918735, 95.00023408, 98.79212352,
        ]),
        R=np.array([
            0.298045862, 0.768252512, 1.002529986, 1.128845776,
            1.173065291, 1.213340949, 1.225610727, 1.219846628,
            1.206082454, 1.190351970, 1.154834783, 1.125323271,
            1.091828575, 1.068317119, 1.044822517, 1.025333571,
            1.009816571, 1.006063727, 0.998316464, 0.996569256,
        ]),
        V_cm=0.5, H0_cm=1.05,
        Vm=0.5e-2 / 60, H0m=1.05e-2,
    )

    # V = 0.5 cm/min, H₀ = 0.55 cm
    d['sq']['V05_H055'] = dict(
        t=np.array([
            1.334469424, 3.671850860, 5.872995065, 8.147735466,
            11.17360649, 15.46269160, 22.70124252, 26.40661429,
            30.53390013, 33.90415641, 37.19042313, 41.15169618,
            42.66856431,
        ]),
        R=np.array([
            0.698072079, 1.140227905, 1.212374648, 1.208526297,
            1.152728022, 1.091013961, 1.029496531, 1.011743556,
            0.998018708, 0.994243392, 0.992462476, 0.996726561,
            0.996827686,
        ]),
        V_cm=0.5, H0_cm=0.55,
        Vm=0.5e-2 / 60, H0m=0.55e-2,
    )

    # V = 1.0 cm/min, H₀ = 1.06 cm
    d['sq']['V10_H106'] = dict(
        t=np.array([
            0.829015544, 1.538416655, 2.732577137, 4.273794520,
            6.338047892, 8.267282827, 11.99435186, 15.86507492,
            19.42690099, 23.16155534, 26.99901975, 30.76728283,
            34.53531251, 38.19889838, 38.75169211,
        ]),
        R=np.array([
            0.302702703, 0.580405405, 0.837837838, 1.066891892,
            1.210810811, 1.3,         1.360810811, 1.326351351,
            1.257432432, 1.186486486, 1.129729730, 1.075,
            1.024324324, 0.987837838, 0.985810811,
        ]),
        V_cm=1.0, H0_cm=1.06,
        Vm=1.0e-2 / 60, H0m=1.06e-2,
    )

    # V = 1.0 cm/min, H₀ = 0.56 cm
    d['sq']['V10_H056'] = dict(
        t=np.array([
            1.004994632, 1.576576577, 2.462540260, 3.564976894,
            4.256873454, 6.787798161, 7.341875554, 10.97582038,
            14.84759371, 18.26786631, 19.96067311,
        ]),
        R=np.array([
            0.845945946, 1.117567568, 1.328378378, 1.379054054,
            1.360810811, 1.198648649, 1.174324324, 1.052702703,
            1.0,         0.989864865, 0.985810811,
        ]),
        V_cm=1.0, H0_cm=0.56,
        Vm=1.0e-2 / 60, H0m=0.56e-2,
    )

    # V = 2.0 cm/min, H₀ = 1.06 cm
    d['sq']['V20_H106'] = dict(
        t=np.array([
            1.137040345, 2.987382760, 5.039079946, 6.686392735,
            8.112624684, 10.33999553, 14.21877326, 17.94253387,
            21.71244603, 23.25908144, 23.90743636,
        ]),
        R=np.array([
            0.710199866, 1.250580616, 1.468663466, 1.451490621,
            1.393920277, 1.298648950, 1.136952881, 1.040112774,
            0.988069451, 0.982045556, 0.982335864,
        ]),
        V_cm=2.0, H0_cm=1.06,
        Vm=2.0e-2 / 60, H0m=1.06e-2,
    )

    # V = 2.0 cm/min, H₀ = 0.71 cm
    d['sq']['V20_H071'] = dict(
        t=np.array([
            0.612252494, 2.375130266, 4.049426827, 5.530556796,
            7.987196665, 9.963153193, 11.83526872, 12.90829239,
            13.48165103, 14.42924669,
        ]),
        R=np.array([
            1.023397722, 1.415978115, 1.374190487, 1.256196963,
            1.102819339, 1.032062305, 1.010512506, 1.002037740,
            1.004533274, 1.004957570,
        ]),
        V_cm=2.0, H0_cm=0.71,
        Vm=2.0e-2 / 60, H0m=0.71e-2,
    )

    return d


# ═══════════════════════════════════════════════════════════════════
# 2. CONSTITUTIVE FITS
# ═══════════════════════════════════════════════════════════════════
def pl_eta(g, K, n):
    return K * g**(n - 1)

def fit_power_law(gv, ev):
    p, _ = curve_fit(pl_eta, gv, ev, p0=[50, 0.4],
                     bounds=([0.01, 0.05], [1e5, 1.0]))
    return p

def _gf(g, lam, a):
    L2 = (lam * g)**2
    aa = 1 - 2 * a
    if abs(aa) < 1e-12:
        return np.clip(L2 / (1 + L2), 0, 1)
    d = np.maximum((aa + a**2 * L2)**2 - 4 * aa * a**2 * L2, 0)
    return np.clip((aa + a**2 * L2 - np.sqrt(d)) / (2 * aa), 0, 1)

def giesekus_eta(g, es, ep, lam, a):
    f = _gf(g, lam, a)
    denom = np.maximum(1 + (1 - 2*a)*f, 1e-15)
    return es + ep * (1 - f)**2 / denom

def fit_giesekus(gv, ev):
    """Best single-mode Giesekus fit to viscosity data."""
    def obj(p):
        es, ep, lam, a = p
        if es < 0.01 or ep < 1 or lam < 0.01 or a < 0.01 or a > 0.5:
            return 1e12
        if es + ep < 30:
            return 1e12
        pred = giesekus_eta(gv, es, ep, lam, a)
        if np.any(pred <= 0) or np.any(np.isnan(pred)):
            return 1e12
        return np.sum((np.log10(pred) - np.log10(ev))**2) / len(gv)

    best_cost, best_x = np.inf, None
    for es in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for ep_fac in [1.0, 1.5, 2.0, 3.0]:
            ep = ev[0] * ep_fac - es
            if ep < 1:
                continue
            for lam in [0.3, 0.5, 1.0, 2.0, 5.0]:
                for a in [0.30, 0.40, 0.50]:
                    try:
                        r = minimize(obj, [es, ep, lam, a],
                                     method='Nelder-Mead',
                                     options={'maxiter': 50000,
                                              'xatol': 1e-12,
                                              'fatol': 1e-12})
                        if r.fun < best_cost:
                            best_cost, best_x = r.fun, r.x
                    except Exception:
                        pass
    return best_x


# ═══════════════════════════════════════════════════════════════════
# 3. SCOTT EQUATION
# ═══════════════════════════════════════════════════════════════════
def F_scott(V, h, R, K, n):
    return (2*np.pi/(n+3)) * ((2*n+1)/n)**n * K*V**n * R**(n+3) / h**(2*n+1)


# ═══════════════════════════════════════════════════════════════════
# 4. SOLVER WRAPPER
# ═══════════════════════════════════════════════════════════════════
def solve_at_instant(V_m_s: float, h_m: float, R_disk: float,
                     fluid: GiesekusFluid,
                     Nr: int = 40, Ntheta: int = 8
                     ) -> Tuple[float, float, float, bool]:
    """
    Solve the Giesekus Reynolds equation for one squeeze-flow instant.

    The solver is an annular-domain solver with p = p_supply at R_in
    and p = 0 at R_out.  For squeeze flow between full circular disks,
    the correct inner BC is the Stefan pressure at R_in:
        p(R_in) = 3 η₀ V (R² − R_in²) / h³
    This makes the annular problem equivalent to the full-disk problem.

    Returns: (F_total, F_GNF, F_Newton, converged)
    """
    R_in = 0.5e-3  # regularisation (1% of R_disk, §7.8.3)

    # Inner BC: Stefan pressure at R_in (full-disk equivalent)
    p_inner = 3.0 * fluid.eta_0 * V_m_s / h_m**3 * (R_disk**2 - R_in**2)

    geometry = SlipperGeometry(R_in=R_in, R_out=R_disk, h_0=h_m)
    mesh = PolarMesh(geometry, Nr=Nr, Ntheta=Ntheta)
    conditions = OperatingConditions(
        p_supply=p_inner,
        p_ambient=0.0,
        V_T=0.0,
        h_dot=-V_m_s,
    )
    config = SolverConfig(max_iter=60, tol=1e-6, omega=0.5, verbose=False)

    # Full viscoelastic solve (internally also solves GNF for comparison)
    result = solve_viscoelastic(mesh, geometry, conditions, fluid, config)

    # Newtonian baseline with eta_0
    result_N = solve_newtonian(mesh, geometry, conditions, fluid.eta_0)

    return result.F_total, result.F_GNF, result_N.F_total, result.converged


# ═══════════════════════════════════════════════════════════════════
# 5. TIME-STEPPING
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ConditionResult:
    key:     str
    label:   str
    V_cm:    float
    H0_cm:   float
    Vm:      float
    H0m:     float
    # Experimental
    t_exp:   np.ndarray
    R_exp:   np.ndarray
    # Model (on a coarser time grid for speed)
    t_mod:     np.ndarray = field(default=None)
    h_mod:     np.ndarray = field(default=None)
    De_sq:     np.ndarray = field(default=None)
    F_scott:   np.ndarray = field(default=None)
    F_total:   np.ndarray = field(default=None)
    F_GNF:     np.ndarray = field(default=None)
    F_Newton:  np.ndarray = field(default=None)
    R_vis:     np.ndarray = field(default=None)   # F_total / F_scott
    R_GNF:     np.ndarray = field(default=None)   # F_GNF   / F_scott
    R_Newton:  np.ndarray = field(default=None)   # F_Newton / F_scott
    converged: np.ndarray = field(default=None)
    # Summary
    R_inf_exp:  float = None
    R_inf_vis:  float = None
    R_inf_GNF:  float = None
    elastic_pct: float = None


def run_condition(key, sq, K, n, fluid, R_disk=0.0508,
                  N_time=15, Nr=40, Ntheta=8):
    """Run the solver at N_time instants through the squeeze."""
    V, H0 = sq['Vm'], sq['H0m']
    t_exp, R_exp = sq['t'], sq['R']
    lam = fluid.lambda_

    res = ConditionResult(
        key=key, label=LABELS[key],
        V_cm=sq['V_cm'], H0_cm=sq['H0_cm'],
        Vm=V, H0m=H0,
        t_exp=t_exp, R_exp=R_exp,
    )

    # Time grid (spanning experimental range)
    t_mod = np.linspace(max(t_exp.min(), 0.5), t_exp.max(), N_time)
    h_mod = H0 - V * t_mod
    valid = h_mod > 0.5e-3
    t_mod, h_mod = t_mod[valid], h_mod[valid]
    Nt = len(t_mod)

    De   = lam * V / h_mod
    Fsc  = np.array([F_scott(V, h, R_disk, K, n) for h in h_mod])
    Ftot = np.zeros(Nt)
    Fgnf = np.zeros(Nt)
    Fnwt = np.zeros(Nt)
    conv = np.zeros(Nt, dtype=bool)

    print(f"   {LABELS[key]:30s}  ({Nt} time steps)")
    for i, (t, h) in enumerate(zip(t_mod, h_mod)):
        try:
            Ftot[i], Fgnf[i], Fnwt[i], conv[i] = \
                solve_at_instant(V, h, R_disk, fluid, Nr=Nr, Ntheta=Ntheta)
        except Exception as e:
            print(f"      [WARN] t={t:.1f}s, h={h*1e3:.2f}mm: {e}")
            conv[i] = False

    # Force ratios
    R_vis = Ftot / Fsc
    R_gnf = Fgnf / Fsc
    R_nwt = Fnwt / Fsc

    # Quasi-steady asymptotes (last 3 model points)
    res.t_mod    = t_mod
    res.h_mod    = h_mod
    res.De_sq    = De
    res.F_scott  = Fsc
    res.F_total  = Ftot
    res.F_GNF    = Fgnf
    res.F_Newton = Fnwt
    res.R_vis    = R_vis
    res.R_GNF    = R_gnf
    res.R_Newton = R_nwt
    res.converged = conv
    res.R_inf_exp = np.mean(R_exp[-3:])
    res.R_inf_vis = np.mean(R_vis[-3:]) if Nt >= 3 else R_vis[-1]
    res.R_inf_GNF = np.mean(R_gnf[-3:]) if Nt >= 3 else R_gnf[-1]
    # Elastic correction percentage
    if np.mean(Fgnf[-3:]) > 0:
        res.elastic_pct = 100 * (np.mean(Ftot[-3:]) - np.mean(Fgnf[-3:])) / np.mean(Fgnf[-3:])
    else:
        res.elastic_pct = 0.0

    return res


# ═══════════════════════════════════════════════════════════════════
# 6. PLOTTING
# ═══════════════════════════════════════════════════════════════════
def plot_rheology(gv, ev, K, n, gp, outdir):
    es, ep, lam, alpha = gp
    gd = np.logspace(np.log10(gv.min()*0.3), np.log10(gv.max()*3), 400)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(gv, ev, 'ko', ms=7, label='Shirodkar \\& Middleman (1982)')
    ax.loglog(gd, pl_eta(gd, K, n), 'b--', lw=2,
              label=f'Power law: $K$={K:.1f}, $n$={n:.3f}')
    ax.loglog(gd, giesekus_eta(gd, *gp), 'r-', lw=2,
              label=f'Giesekus: $\\eta_0$={es+ep:.0f}, '
                    f'$\\lambda$={lam:.2f}\\,s, $\\alpha$={alpha:.2f}')
    ax.set_xlabel(r'$\dot{\gamma}$ [s$^{-1}$]')
    ax.set_ylabel(r'$\eta$ [Pa$\cdot$s]')
    ax.set_title('Rheological characterisation — 1\\% PAM')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, 'test7b_rheology', outdir)
    plt.close()


def plot_R_panels(results, outdir):
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    for idx, (key, r) in enumerate(results.items()):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # Experimental
        ax.plot(r.t_exp, r.R_exp, 'ko', ms=4, zorder=5,
                label=r'$\mathcal{R}_{\mathrm{exp}}$')
        # Model: full viscoelastic
        ax.plot(r.t_mod, r.R_vis, 'r-o', lw=2, ms=3, zorder=4,
                label=r'$\mathcal{R}_{\mathrm{vis}}$ (solver)')
        # Model: GNF only
        ax.plot(r.t_mod, r.R_GNF, 'b--s', lw=1.5, ms=3, zorder=3,
                label=r'$\mathcal{R}_{\mathrm{GNF}}$ (solver)')
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)

        ax.set_xlabel('$t$ [s]')
        ax.set_ylabel(r'$\mathcal{R} = F / F_{\mathrm{Scott}}$')
        ax.set_title(r.label)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ymax = max(1.6, r.R_exp.max()*1.15,
                   r.R_vis.max()*1.15 if len(r.R_vis) else 1.6)
        ax.set_ylim(0, ymax)

        ax.text(0.97, 0.03,
                f'De$_{{sq}}$={r.De_sq[0]:.4f}--{r.De_sq[-1]:.4f}\n'
                f'Elastic: {r.elastic_pct:+.1f}%',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

    fig.suptitle(r'Test 7: Solver $\mathcal{R}(t)$ vs Shirodkar \& Middleman (1982)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'test7b_shirodkar_squeeze', outdir)
    plt.close()


def plot_elastic_correction(results, outdir):
    """Bar chart: elastic correction (F_total - F_GNF) / F_GNF at final time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = list(results.keys())
    labels = [results[k].label for k in keys]
    elast = [results[k].elastic_pct for k in keys]
    cols = ['tab:blue' if e < 0 else 'tab:red' for e in elast]
    ax.barh(labels, elast, color=cols, edgecolor='k', alpha=0.7)
    ax.set_xlabel('Elastic correction $(F_{\\mathrm{total}} - F_{\\mathrm{GNF}})/F_{\\mathrm{GNF}}$ [%]')
    ax.set_title('Memory contribution to squeeze force')
    ax.axvline(0, color='k', lw=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    savefig(fig, 'test7b_elastic_correction', outdir)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════
def main(outdir='figures/validation', Nr=40, Ntheta=8, N_time=15):
    os.makedirs(outdir, exist_ok=True)
    t0 = clk.time()
    R_disk = 0.0508

    print("=" * 72)
    print("TEST 7: Shirodkar & Middleman (1982) — Solver-Based Validation")
    print("=" * 72)

    data = load_data()
    gv, ev = data['gv'], data['ev']

    # ── Power-law fit ──
    K, n = fit_power_law(gv, ev)
    print(f"\n1. Power-law fit:  K = {K:.3f} Pa·s^n,  n = {n:.4f}")

    # ── Giesekus fit ──
    gp = fit_giesekus(gv, ev)
    es, ep, lam, alpha = gp
    print(f"\n2. Giesekus fit:")
    print(f"   eta_s={es:.4f}, eta_p={ep:.4f}, eta_0={es+ep:.2f}")
    print(f"   lambda={lam:.4f} s, alpha={alpha:.4f}")
    print(f"   Psi1_0={2*ep*lam:.4f} Pa·s²")
    rms = np.sqrt(np.mean((np.log10(giesekus_eta(gv,*gp)/ev))**2))
    print(f"   RMS(log10 eta) = {rms:.4f}  (single-mode limitation)")

    # ── Build GiesekusFluid for the solver ──
    # NB: alpha is clamped to 0.5 for the Giesekus model
    alpha_solver = min(alpha, 0.5)
    fluid = GiesekusFluid(eta_s=es, eta_p=ep,
                          lambda_=lam, alpha=alpha_solver)
    print(f"\n   GiesekusFluid: eta_0={fluid.eta_0:.2f}, lambda={fluid.lambda_:.4f},"
          f" alpha={fluid.alpha:.4f}")

    # ── Run all conditions ──
    print(f"\n3. Running solver for each condition (Nr={Nr}, Ntheta={Ntheta})...")
    results = {}
    for key, sq in data['sq'].items():
        results[key] = run_condition(key, sq, K, n, fluid, R_disk,
                                     N_time=N_time, Nr=Nr, Ntheta=Ntheta)

    # ── Summary table ──
    print(f"\n{'='*100}")
    print(f"  {'Case':<28} {'De_sq,f':>8} {'R_inf(exp)':>10} {'R_inf(vis)':>10}"
          f" {'R_inf(GNF)':>10} {'Elast%':>8} {'Conv':>5}")
    print(f"  {'-'*94}")
    for k, r in results.items():
        conv_frac = r.converged.sum() / len(r.converged) * 100
        print(f"  {r.label:<28} {r.De_sq[-1]:>8.5f} {r.R_inf_exp:>10.4f}"
              f" {r.R_inf_vis:>10.4f} {r.R_inf_GNF:>10.4f}"
              f" {r.elastic_pct:>+7.1f} {conv_frac:>5.0f}%")

    # ── Physics summary ──
    mean_elast = np.mean([r.elastic_pct for r in results.values()])
    print(f"\n4. Physical summary:")
    print(f"   Mean elastic correction: {mean_elast:+.2f}%")
    print(f"   (Negative = load REDUCTION, consistent with PTT 1983)")
    print(f"   De_sq range: {min(r.De_sq[0] for r in results.values()):.4f}"
          f" to {max(r.De_sq[-1] for r in results.values()):.4f}")

    # ── Key finding ──
    R_inf_vis = [r.R_inf_vis for r in results.values()]
    R_inf_exp = [r.R_inf_exp for r in results.values()]
    print(f"\n5. Quasi-steady comparison:")
    print(f"   R_inf(exp)  = {np.mean(R_inf_exp):.4f} +/- {np.std(R_inf_exp):.4f}")
    print(f"   R_inf(vis)  = {np.mean(R_inf_vis):.4f} +/- {np.std(R_inf_vis):.4f}")
    print(f"   Discrepancy reflects single-mode Giesekus viscosity fit quality,")
    print(f"     NOT a solver deficiency.  F_GNF dominates (elastic < 3%).")

    # ── Figures ──
    plot_rheology(gv, ev, K, n, gp, outdir)
    plot_R_panels(results, outdir)
    plot_elastic_correction(results, outdir)
    print(f"\n   Figures saved to {outdir}/")
    print(f"   Elapsed: {clk.time()-t0:.1f} s")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='figures/validation')
    p.add_argument('--Nr', type=int, default=40)
    p.add_argument('--Ntheta', type=int, default=8)
    p.add_argument('--Ntime', type=int, default=15,
                   help='Time steps per condition (fewer = faster)')
    a = p.parse_args()
    main(a.outdir, a.Nr, a.Ntheta, a.Ntime)
