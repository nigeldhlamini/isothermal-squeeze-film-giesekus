#!/usr/bin/env python
r"""
Test 8 --- Viscometry Validation: Giesekus Fit vs Published Rheometry
======================================================================

Validates the Giesekus viscosity function against published rheometric data
for three categories of fluids relevant to slipper bearing lubrication:

  Category A: Boger fluids         (eta_s/eta_0 ~ 0.3, alpha ~ 0)
  Category B: Mild shear-thinning  (eta_s/eta_0 ~ 0.1--0.5, alpha ~ 0.1--0.3)
  Category C: Strong thinning      (eta_s/eta_0 < 0.05, alpha ~ 0.3--0.5)

For each category, the script:
  1. Defines reference rheometry data (published or dummy placeholder)
  2. Fits a single-mode Giesekus model via least-squares
  3. Computes relative error metrics (mean, max, R^2)
  4. Generates a publication-quality figure

References
----------
Category A:
    Phan-Thien, N. et al. (1985). Boger fluid S1 (PIB/PB/kerosene).

Category B:
    Marx, N., Fernández, L., Barceló, F. & Spikes, H.A. (2018).
    Tribol. Lett. 66:92 (Part I).  Oil #1: polymer-thickened mineral oil
    (HSDCP VM) at T = 60 C.  Digitised from Fig. 6.

Category C:
    Shirodkar, P. & Middleman, S. (1982). J. Rheol. 26, 1-17.
    (aqueous polyacrylamide, 1 wt%)

Author: N.C. Dhlamini, March 2026
"""
import numpy as np
import os, sys, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rheology import GiesekusFluid
from src.geometry import SlipperGeometry, OperatingConditions
from src.mesh import PolarMesh
from src.solver import solve, SolverConfig
from scripts.marx2018_oil1_properties import get_marx2018_oil1_all_properties

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


# ====================================================================
# REFERENCE RHEOMETRY DATA
# ====================================================================

def get_category_A_data():
    """
    Category A: Boger fluid (constant viscosity, measurable elasticity).
    PTT (1985) Fluid S1: PIB/PB/kerosene Boger fluid.
    """
    return {
        'label': 'Category A: Boger fluid (PTT 1985, S1)',
        'source': 'Phan-Thien et al. (1985) Table 1',
        'gdot': np.array([0.01, 0.1, 1.0, 10.0, 50.0, 100.0]),
        'eta':  np.array([55.5, 55.5, 55.4, 55.3, 55.1, 54.9]),
        'Psi1': np.array([73.8, 73.8, 73.5, 72.0, 60.0, 45.0]),
        # Known parameters (exact for this fluid)
        'eta_s': 15.8, 'eta_p': 39.7, 'lambda_': 0.93, 'alpha': 0.001,
    }


def get_category_B_data():
    """
    Category B: Mild shear-thinning (polymer-thickened mineral oil).
    Marx et al. (2018) Oil #1 at T = 60 C.  Digitised from Fig. 6.

    Initial estimates for Giesekus fitting are derived from the paper:
      - eta_s from base oil (second Newtonian plateau)
      - eta_p = eta_0 - eta_s
      - lambda estimated from Carreau-Yasuda parameter A (Table 7)
      - alpha left free (not measured; to be fitted)
    """
    marx = get_marx2018_oil1_all_properties()

    return {
        'label': 'Category B: Mild thinning (Marx et al. 2018, Oil #1)',
        'source': 'Marx et al. (2018) Tribol. Lett. 66:92, Fig. 6 (T = 60 C)',
        'gdot': marx['gdot'],
        'eta': marx['eta'],
        'Psi1': None,               # not measured
        # Initial estimates for fitting (not exact known parameters)
        'eta_s': marx['eta_s'],      # 15.0e-3 Pa.s (base oil plateau)
        'eta_p': marx['eta_p'],      # 25.3e-3 Pa.s (polymer contribution)
        'lambda_': marx['carreau_yasuda']['A_s'],  # 21.88e-3 s (CY estimate)
        'alpha': None,               # must be fitted
    }


def get_category_C_data():
    """
    Category C: Strong shear-thinning (aqueous PAM).
    Shirodkar & Middleman (1982), 1 wt% aqueous polyacrylamide.
    """
    # Viscosity data points (digitised from Fig. 1)
    gdot = np.array([0.1, 0.5, 1.0, 5.0, 10, 50, 100, 500, 1000])
    eta  = np.array([12.0, 7.5, 5.8, 2.5, 1.6, 0.55, 0.35, 0.12, 0.075])

    # Physical estimates from the data:
    # eta_0 ~ 12 Pa.s (first data point)
    # eta_s ~ 0.075 Pa.s (last data point — solvent plateau not fully reached)
    # eta_s/eta_0 ~ 0.006 → strong-thinning regime, single-mode poor
    return {
        'label': 'Category C: Strong thinning (Shirodkar 1982)',
        'source': 'Shirodkar & Middleman (1982) Fig. 1',
        'gdot': gdot,
        'eta': eta,
        'Psi1': None,  # not measured
        'eta_s': 0.075,      # approximate second Newtonian plateau
        'eta_p': 12.0 - 0.075,  # polymer contribution
        'lambda_': None,
        'alpha': None,        # triggers constrained fit (fix_plateaus)
    }


# ====================================================================
# FITTING
# ====================================================================

def fit_giesekus(gdot, eta_data, Psi1_data=None,
                 eta_s_init=None, eta_p_init=None,
                 lam_init=None, alpha_init=None,
                 fix_plateaus=False):
    """
    Fit a single-mode Giesekus model to viscosity (and optionally Psi1) data.

    Strategy
    --------
    1. Build physically informed bounds from the data.
    2. Transform eta_s, eta_p, lambda to log-space for uniform exploration.
    3. Global search with differential_evolution.
    4. Polish with Nelder-Mead from the global optimum.

    Parameters
    ----------
    fix_plateaus : bool
        If True, fix eta_s and eta_p to their initial values (eta_s_init,
        eta_p_init) and only fit lambda and alpha.  This is appropriate when
        the plateau viscosities are known from independent measurements
        (e.g. base oil and blended oil viscometry).

    Returns
    -------
    dict with fitted parameters and error metrics
    """
    eta_0_est = eta_data[0]
    eta_inf_est = eta_data[-1]

    if fix_plateaus and eta_s_init is not None and eta_p_init is not None:
        return _fit_giesekus_2param(
            gdot, eta_data, eta_s_init, eta_p_init,
            lam_init=lam_init)

    # --- Physical bounds (in real space) ---
    es_lo = max(eta_inf_est * 0.01, 1e-8)
    es_hi = eta_0_est * 1.5
    ep_lo = max(eta_0_est * 0.01, 1e-8)
    ep_hi = eta_0_est * 2.0
    la_lo = 1.0 / (1000.0 * gdot.max())
    la_hi = 1000.0 / max(gdot.min(), 1e-6)
    al_lo = 0.005
    al_hi = 0.5

    # Work in log-space for eta_s, eta_p, lambda (spans decades)
    log_bounds = [
        (np.log10(es_lo), np.log10(es_hi)),
        (np.log10(ep_lo), np.log10(ep_hi)),
        (np.log10(la_lo), np.log10(la_hi)),
        (al_lo, al_hi),  # alpha stays linear (0 to 0.5)
    ]

    def objective(x):
        es = 10**x[0]
        ep = 10**x[1]
        la = 10**x[2]
        al = x[3]
        try:
            fluid = GiesekusFluid(eta_s=es, eta_p=ep, lambda_=la, alpha=al)
            eta_pred = fluid.viscosity(gdot)
            # Log-space residual for viscosity
            res = np.sum((np.log10(eta_pred) - np.log10(eta_data))**2)
            if Psi1_data is not None:
                Psi1_pred = fluid.Psi1(gdot)
                valid = Psi1_data > 0
                if np.any(valid):
                    res += 0.5 * np.sum(
                        (np.log10(Psi1_pred[valid]) - np.log10(Psi1_data[valid]))**2
                    )
            return res
        except Exception:
            return 1e10

    # Global optimisation (log-space for first three parameters)
    de_result = differential_evolution(
        objective, log_bounds,
        seed=42, maxiter=3000, tol=1e-14,
        popsize=25, mutation=(0.5, 1.5), recombination=0.9,
        polish=False,
    )

    # Polish with Nelder-Mead
    result = minimize(objective, de_result.x, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})

    # Convert back to real space
    es = 10**result.x[0]
    ep = 10**result.x[1]
    la = 10**result.x[2]
    al = result.x[3]

    fluid_fit = GiesekusFluid(eta_s=es, eta_p=ep, lambda_=la, alpha=al)
    eta_fit = fluid_fit.viscosity(gdot)

    # Error metrics
    rel_err = np.abs(eta_fit - eta_data) / eta_data * 100
    ss_res = np.sum((eta_fit - eta_data)**2)
    ss_tot = np.sum((eta_data - np.mean(eta_data))**2)
    R2 = 1 - ss_res / (ss_tot + 1e-30)

    return {
        'fluid': fluid_fit,
        'eta_s': es, 'eta_p': ep, 'lambda_': la, 'alpha': al,
        'eta_fit': eta_fit,
        'mean_error_pct': np.mean(rel_err),
        'max_error_pct': np.max(rel_err),
        'R2': R2,
    }


def _fit_giesekus_2param(gdot, eta_data, eta_s, eta_p, lam_init=None):
    """
    Fit lambda and alpha only, with eta_s and eta_p fixed to known values.

    This ensures the low-shear (eta_0 = eta_s + eta_p) and high-shear
    (eta_s) plateaus are exactly correct, and only the transition shape
    is fitted.
    """
    # Physical bounds: the transition onset 1/lambda should be within
    # the data range.  Alpha must be > 0 for shear thinning.
    la_lo = 1.0 / (100.0 * gdot.max())
    la_hi = 100.0 / max(gdot.min(), 1e-6)
    al_lo, al_hi = 0.01, 0.5

    bounds = [
        (np.log10(la_lo), np.log10(la_hi)),
        (al_lo, al_hi),
    ]

    def objective(x):
        la = 10**x[0]
        al = np.clip(x[1], al_lo, al_hi)
        try:
            fluid = GiesekusFluid(eta_s=eta_s, eta_p=eta_p,
                                  lambda_=la, alpha=al)
            eta_pred = fluid.viscosity(gdot)
            return np.sum((np.log10(eta_pred) - np.log10(eta_data))**2)
        except Exception:
            return 1e10

    # Global search
    de_result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=5000, tol=1e-14,
        popsize=40, mutation=(0.5, 1.5), recombination=0.9,
        polish=False,
    )

    # Polish with bounded optimizer
    from scipy.optimize import minimize as _min
    result = _min(objective, de_result.x, method='L-BFGS-B',
                  bounds=bounds)

    la = 10**result.x[0]
    al = np.clip(result.x[1], al_lo, al_hi)

    fluid_fit = GiesekusFluid(eta_s=eta_s, eta_p=eta_p,
                              lambda_=la, alpha=al)
    eta_fit = fluid_fit.viscosity(gdot)

    rel_err = np.abs(eta_fit - eta_data) / eta_data * 100
    ss_res = np.sum((eta_fit - eta_data)**2)
    ss_tot = np.sum((eta_data - np.mean(eta_data))**2)
    R2 = 1 - ss_res / (ss_tot + 1e-30)

    return {
        'fluid': fluid_fit,
        'eta_s': eta_s, 'eta_p': eta_p, 'lambda_': la, 'alpha': al,
        'eta_fit': eta_fit,
        'mean_error_pct': np.mean(rel_err),
        'max_error_pct': np.max(rel_err),
        'R2': R2,
    }


# ====================================================================
# PLOTTING
# ====================================================================

def plot_viscometry(categories, fits, outdir='figures/validation'):
    """Three-panel figure: viscosity fits for each category."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (cat, fit) in enumerate(zip(categories, fits)):
        ax = axes[idx]
        gdot = cat['gdot']
        gdot_fine = np.logspace(np.log10(gdot.min()) - 0.5,
                                np.log10(gdot.max()) + 0.5, 300)

        # Asymptotes (draw first so data is on top)
        ax.axhline(fit['eta_s'] + fit['eta_p'], color='tab:blue', ls=':', lw=0.8,
                    label=r'$\eta_0$')
        ax.axhline(fit['eta_s'], color='tab:green', ls=':', lw=0.8,
                    label=r'$\eta_s$')

        # For Category B, add Carreau-Yasuda comparison line
        if idx == 1:
            try:
                from scripts.marx2018_oil1_properties import (
                    carreau_yasuda_viscosity, get_marx2018_oil1_all_properties)
                marx = get_marx2018_oil1_all_properties()
                cy = marx['carreau_yasuda']
                eta_cy = carreau_yasuda_viscosity(
                    gdot_fine, marx['eta_0'], marx['eta_s'],
                    cy['A_s'], cy['n'], cy['a'])
                ax.loglog(gdot_fine, eta_cy, 'b--', lw=1.2, alpha=0.7,
                          label='Carreau\u2013Yasuda')
            except Exception:
                pass

        # For Category C, add power-law fit for comparison
        if idx == 2:
            log_gdot = np.log10(gdot)
            log_eta = np.log10(cat['eta'])
            coeffs = np.polyfit(log_gdot, log_eta, 1)
            n_pl = coeffs[0] + 1  # power-law index
            K_pl = 10**coeffs[1]
            ax.loglog(gdot_fine, K_pl * gdot_fine**(n_pl - 1),
                      'b--', lw=1.2, alpha=0.7,
                      label=f'Power law ($n$={n_pl:.2f})')

        # Data
        ax.loglog(gdot, cat['eta'], 'ko', ms=5, zorder=5, label='Data')

        # Giesekus fit
        eta_fine = fit['fluid'].viscosity(gdot_fine)
        ax.loglog(gdot_fine, eta_fine, 'r-', lw=1.8, label='Giesekus fit')

        ax.set_xlabel(r'$\dot{\gamma}$ [s$^{-1}$]')
        if idx == 0:
            ax.set_ylabel(r'$\eta$ [Pa$\cdot$s]')

        # Short title from category label
        title = cat['label'].split('(')[0].strip()
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc='best')

        # Annotation
        ax.text(0.97, 0.05,
                f"$R^2$ = {fit['R2']:.4f}\n"
                f"Mean err: {fit['mean_error_pct']:.1f}%\n"
                f"$\\alpha$ = {fit['alpha']:.3f}\n"
                f"$\\lambda$ = {fit['lambda_']:.2e} s",
                transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    fig.tight_layout()
    savefig(fig, 'test8_rheology', outdir)
    plt.close()


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 72)
    print("TEST 8: Viscometry Validation --- Giesekus Fit Quality")
    print("=" * 72)

    categories = [get_category_A_data(), get_category_B_data(),
                  get_category_C_data()]
    fits = []

    for cat in categories:
        print(f"\n  {cat['label']}")
        print(f"    Source: {cat['source']}")

        # Use known parameters if available, else fit
        if cat['eta_s'] is not None and cat['alpha'] is not None:
            fluid = GiesekusFluid(eta_s=cat['eta_s'], eta_p=cat['eta_p'],
                                   lambda_=cat['lambda_'], alpha=cat['alpha'])
            eta_fit = fluid.viscosity(cat['gdot'])
            rel_err = np.abs(eta_fit - cat['eta']) / cat['eta'] * 100
            ss_res = np.sum((eta_fit - cat['eta'])**2)
            ss_tot = np.sum((cat['eta'] - np.mean(cat['eta']))**2)
            R2 = 1 - ss_res / (ss_tot + 1e-30)
            fit_result = {
                'fluid': fluid,
                'eta_s': cat['eta_s'], 'eta_p': cat['eta_p'],
                'lambda_': cat['lambda_'], 'alpha': cat['alpha'],
                'eta_fit': eta_fit,
                'mean_error_pct': np.mean(rel_err),
                'max_error_pct': np.max(rel_err),
                'R2': R2,
            }
            print(f"    Using known parameters (not fitted)")
        else:
            # For Category B, fix plateaus to known values from
            # independent viscometry (base oil and blended oil measurements)
            fix = (cat.get('eta_s') is not None and cat.get('eta_p') is not None
                   and cat.get('alpha') is None)
            fit_result = fit_giesekus(
                cat['gdot'], cat['eta'],
                Psi1_data=cat.get('Psi1'),
                eta_s_init=cat.get('eta_s'),
                eta_p_init=cat.get('eta_p'),
                lam_init=cat.get('lambda_'),
                alpha_init=cat.get('alpha'),
                fix_plateaus=fix,
            )
            if fix:
                print(f"    Fixed plateaus: eta_s={fit_result['eta_s']:.4e}, "
                      f"eta_p={fit_result['eta_p']:.4e}")
                print(f"    Fitted: lambda={fit_result['lambda_']:.2e}, "
                      f"alpha={fit_result['alpha']:.3f}")
            else:
                print(f"    Fitted: eta_s={fit_result['eta_s']:.4f}, "
                      f"eta_p={fit_result['eta_p']:.4f}, "
                      f"lambda={fit_result['lambda_']:.2e}, "
                      f"alpha={fit_result['alpha']:.3f}")

        print(f"    R^2 = {fit_result['R2']:.4f}, "
              f"mean error = {fit_result['mean_error_pct']:.1f}%, "
              f"max error = {fit_result['max_error_pct']:.1f}%")
        fits.append(fit_result)

    plot_viscometry(categories, fits)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'Category':30s} {'R^2':>8} {'Mean err':>10} {'Max err':>10}")
    print(f"  {'-'*64}")
    for cat, fit in zip(categories, fits):
        name = cat['label'].split('(')[0].strip()
        print(f"  {name:30s} {fit['R2']:>8.4f} "
              f"{fit['mean_error_pct']:>9.1f}% {fit['max_error_pct']:>9.1f}%")

    print(f"\n  Note: Category B uses digitised data from Marx et al. (2018)")
    print(f"  Tribol. Lett. 66:92, Oil #1 at T = 60 C (Fig. 6).")
    print(f"  No Psi1 data available; alpha fitted from viscosity alone.")

    # ==================================================================
    # SOLVER VALIDATION: Run the Reynolds equation with each fluid
    # ==================================================================
    print(f"\n{'='*72}")
    print("  SOLVER VALIDATION: Reynolds equation with fitted fluids")
    print(f"{'='*72}")

    # Representative slipper bearing geometry
    geom = SlipperGeometry(R_in=0.010, R_out=0.020, h_0=20e-6)
    mesh = PolarMesh(geom, Nr=30, Ntheta=30)
    config = SolverConfig(max_iter=200, tol=1e-6, omega=0.3)

    for cat, fit in zip(categories, fits):
        name = cat['label'].split('(')[0].strip()
        fluid = fit['fluid']
        print(f"\n  {name}")
        print(f"    eta_0 = {fluid.eta_0:.4e} Pa.s, "
              f"alpha = {fluid.alpha:.4f}, lambda = {fluid.lambda_:.4e} s")

        # Scale supply pressure and squeeze rate to the fluid viscosity
        # so that pressures stay within physically reasonable bounds
        p_supply = min(10e6, 1e6 / max(fluid.eta_0, 0.01))
        h_dot = -1e-4 / max(fluid.eta_0, 0.01)
        conds = OperatingConditions(p_supply=p_supply, h_dot=h_dot)

        # Newtonian solution (uses eta_0 only)
        try:
            res_N = solve(mesh, geom, conds, fluid, model='newtonian', config=config)
            print(f"    Newtonian:     F = {res_N.F_total:.4e} N, "
                  f"p_max = {res_N.pressure.max()/1e6:.2f} MPa, "
                  f"converged = {res_N.converged}")
        except Exception as e:
            print(f"    Newtonian:     FAILED -- {e}")
            res_N = None

        # GNF solution (uses viscosity(gdot))
        try:
            res_G = solve(mesh, geom, conds, fluid, model='GNF', config=config)
            W_ratio_gnf = res_G.F_total / res_N.F_total if res_N and res_N.converged else float('nan')
            print(f"    GNF:           F = {res_G.F_total:.4e} N, "
                  f"F/F_N = {W_ratio_gnf:.4f}, "
                  f"converged = {res_G.converged}")
        except Exception as e:
            print(f"    GNF:           FAILED -- {e}")

        # Viscoelastic solution (uses viscosity + Psi1 + memory)
        if not fluid.is_newtonian and fluid.alpha > 1e-4:
            try:
                res_V = solve(mesh, geom, conds, fluid, model='viscoelastic',
                              config=config)
                W_ratio_ve = res_V.F_total / res_N.F_total if res_N and res_N.converged else float('nan')
                print(f"    Viscoelastic:  F = {res_V.F_total:.4e} N, "
                      f"F/F_N = {W_ratio_ve:.4f}, "
                      f"converged = {res_V.converged}")
            except Exception as e:
                print(f"    Viscoelastic:  FAILED -- {e}")
        else:
            print(f"    Viscoelastic:  skipped (alpha too small for memory effects)")


if __name__ == '__main__':
    main()
