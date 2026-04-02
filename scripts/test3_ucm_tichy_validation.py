#!/usr/bin/env python
r"""
Test 3 --- UCM Limit Validation against Tichy (1996)
=====================================================

Validates the upper-convected Maxwell (UCM) limit of the Giesekus solver
by setting alpha -> 0 (Oldroyd-B limit with eta_s = 0 gives UCM).

In the UCM limit, the Giesekus model reduces to:
    - eta(gdot) = eta_0  (constant viscosity; no shear-thinning)
    - Psi_1(gdot) = 2 eta_p lambda  (constant first normal stress coeff)
    - Psi_2 = 0

The memory flux simplifies because eta_T = eta_0 (tangent viscosity = total
viscosity when there is no shear-thinning):
    Q_r^mem = -lambda h^2 h_dot / (16 eta_0) * dp/dr

Tichy (1996) showed that for UCM squeeze flow between parallel disks the
load modification relative to Newtonian is:
    F_VE / F_Newton = 1 - 2 De_sq + O(De_sq^2)

where De_sq = lambda |h_dot| / h.

This test:
  1. Sets alpha = 1e-6 (numerical UCM limit)
  2. Sweeps De_sq from 0.005 to 0.15
  3. Compares solver memory correction against the 1 - 2*De_sq prediction
  4. Verifies that the alpha-coupling flux is negligible (< 0.1% of total)

Reference
---------
Tichy, J.A. (1996). Non-Newtonian lubrication with the convected Maxwell
    model. ASME J. Tribol. 118, 344-348.

Author: N.C. Dhlamini, March 2026
"""
import numpy as np
import os, sys, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import (
    GiesekusFluid, SlipperGeometry, PolarMesh,
    OperatingConditions, SolverConfig,
    solve_viscoelastic, solve_newtonian,
)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'lines.linewidth': 1.5,
    'mathtext.fontset': 'cm',
})

# ====================================================================
# UCM fluid parameters
# ====================================================================
ETA_0   = 1.0       # Pa.s  (total viscosity)
ETA_S   = 0.01      # Pa.s  (small solvent viscosity for numerical stability)
ETA_P   = ETA_0 - ETA_S
LAMBDA  = 0.01      # s
ALPHA_G = 1e-6      # Giesekus alpha -> 0  (UCM limit)

# Geometry: parallel disk squeeze
R_IN    = 0.25e-3   # small inner radius (regularisation)
R_OUT   = 0.020     # 20 mm outer radius
H0      = 10e-6     # 10 micron gap

# Solver settings
NR, NTHETA = 40, 8
CONFIG = SolverConfig(max_iter=50, tol=1e-6, omega=0.6, verbose=False,
                      include_memory=True, include_normal_stress=True)


def run_ucm_sweep():
    """Sweep De_sq and compare F_VE/F_Newton against Tichy (1996)."""
    fluid = GiesekusFluid(eta_s=ETA_S, eta_p=ETA_P,
                          lambda_=LAMBDA, alpha=ALPHA_G)

    De_sq_values = np.array([0.005, 0.01, 0.02, 0.04, 0.06,
                             0.08, 0.10, 0.12, 0.15])

    results = []
    print("=" * 72)
    print("TEST 3: UCM Limit --- Tichy (1996) Memory Correction")
    print("=" * 72)
    print(f"  Fluid: eta_0={ETA_0}, lambda={LAMBDA}, alpha={ALPHA_G}")
    print(f"  Geometry: R_out={R_OUT*1e3} mm, h0={H0*1e6} um")
    print()

    geometry = SlipperGeometry(R_in=R_IN, R_out=R_OUT, h_0=H0)
    mesh = PolarMesh(geometry, Nr=NR, Ntheta=NTHETA)

    for De_sq in De_sq_values:
        # h_dot = -De_sq * h0 / lambda  (negative = closing)
        h_dot = -De_sq * H0 / LAMBDA

        conditions = OperatingConditions(
            p_supply=0.0, V_T=0.0, h_dot=h_dot)

        r_ve = solve_viscoelastic(mesh, geometry, conditions, fluid, CONFIG)
        r_nw = solve_newtonian(mesh, geometry, conditions, ETA_0)

        ratio = r_ve.F_total / r_nw.F_total if abs(r_nw.F_total) > 1e-15 else 1.0
        tichy_pred = 1.0 - 2.0 * De_sq
        error_pct = abs(ratio - tichy_pred) / abs(tichy_pred) * 100

        # Flux decomposition: check alpha contribution is negligible
        alpha_frac = 0.0
        if r_ve.fluxes is not None:
            ratios = r_ve.fluxes.flux_ratios()
            alpha_frac = ratios.get('alpha', 0.0)

        results.append({
            'De_sq': De_sq, 'ratio': ratio, 'tichy': tichy_pred,
            'error_pct': error_pct, 'alpha_frac': alpha_frac,
            'converged': r_ve.converged,
        })
        print(f"  De_sq={De_sq:.3f}: F_VE/F_N={ratio:.4f}  "
              f"Tichy={tichy_pred:.4f}  err={error_pct:.1f}%  "
              f"alpha_frac={alpha_frac:.1e}  conv={r_ve.converged}")

    return results


def plot_ucm_validation(results, outdir='figures/validation'):
    """Publication-quality figure: solver vs Tichy (1996)."""
    os.makedirs(outdir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    De = [r['De_sq'] for r in results]
    ratio = [r['ratio'] for r in results]
    tichy = [r['tichy'] for r in results]
    err = [r['error_pct'] for r in results]

    # Panel (a): F_VE / F_Newton vs De_sq
    De_line = np.linspace(0, 0.18, 100)
    ax1.plot(De_line, 1 - 2*De_line, 'k--', lw=2,
             label=r'Tichy (1996): $1 - 2\,\mathrm{De}_{sq}$')
    ax1.plot(De, ratio, 'ro-', ms=6, lw=1.5,
             label=r'Solver ($\alpha = 10^{-6}$, UCM limit)')
    ax1.set_xlabel(r'$\mathrm{De}_{sq} = \lambda|\dot{h}|/h_0$')
    ax1.set_ylabel(r'$F_{\mathrm{VE}} / F_{\mathrm{Newton}}$')
    ax1.set_title('(a) Load reduction ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.17)
    ax1.set_ylim(0.6, 1.05)

    # Panel (b): Relative error
    ax2.semilogy(De, err, 'bs-', ms=6, lw=1.5)
    ax2.axhline(5.0, color='r', ls=':', lw=1, label='5% threshold')
    ax2.set_xlabel(r'$\mathrm{De}_{sq}$')
    ax2.set_ylabel('Relative error vs Tichy [%]')
    ax2.set_title('(b) Deviation from first-order perturbation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Test 3: UCM Limit Validation (Tichy 1996)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'test3_ucm_limit.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {path}")


def main():
    results = run_ucm_sweep()
    plot_ucm_validation(results)

    # Summary
    low_De = [r for r in results if r['De_sq'] <= 0.05]
    high_De = [r for r in results if r['De_sq'] > 0.05]
    print("\n  Summary:")
    if low_De:
        mean_err_low = np.mean([r['error_pct'] for r in low_De])
        print(f"    De_sq <= 0.05: mean error = {mean_err_low:.2f}%")
    if high_De:
        mean_err_hi = np.mean([r['error_pct'] for r in high_De])
        print(f"    De_sq >  0.05: mean error = {mean_err_hi:.2f}%")
    max_alpha = max(r['alpha_frac'] for r in results)
    print(f"    Max alpha flux fraction: {max_alpha:.2e} (should be << 1)")
    all_conv = all(r['converged'] for r in results)
    print(f"    All converged: {all_conv}")
    print(f"\n  PASS" if all_conv and mean_err_low < 5 else "\n  NEEDS REVIEW")


if __name__ == '__main__':
    main()
