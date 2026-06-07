"""
Regression guard for the corrected Giesekus material functions.
======================================================================

These tests lock in the *correct* steady-shear physics and would FAIL against
the earlier buggy quadratic (fake sharp saturation at Lambda_crit, wrong Psi1,
hardcoded Psi2 = -alpha/2 Psi1):

  * Psi1(Lambda->0) = 2 eta_p lambda for all alpha (zero-shear plateau).
  * Psi2/Psi1 -> -alpha/2 as Lambda->0 but DRIFTS at finite shear (NOT a constant).
  * eta thins GRADUALLY: eta > eta_s strictly for every finite Lambda (no cutoff).
  * The library (eta, Psi1, Psi2) satisfies the exact Giesekus steady-shear
    stress equations to machine precision (solver-independent residual check).

Run: pytest tests/test_rheology_exact.py -v
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.rheology import GiesekusFluid

ES, EP, LAM = 0.030, 0.020, 1.0e-3
ALPHAS = [0.001, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45]


def fluid(a):
    return GiesekusFluid(eta_s=ES, eta_p=EP, lambda_=LAM, alpha=a)


@pytest.mark.parametrize("a", ALPHAS)
def test_psi1_zero_shear_plateau(a):
    fl = fluid(a)
    g0 = 1e-6 / LAM                       # Lambda = 1e-6
    assert np.isclose(fl.Psi1(g0), 2 * EP * LAM, rtol=1e-3)


@pytest.mark.parametrize("a", ALPHAS)
def test_psi2_psi1_ratio_zero_shear(a):
    fl = fluid(a)
    g0 = 1e-6 / LAM
    assert np.isclose(fl.Psi2(g0) / fl.Psi1(g0), -a / 2, rtol=1e-3)


@pytest.mark.parametrize("a", [0.1, 0.25, 0.4])
def test_psi2_psi1_ratio_drifts_at_finite_shear(a):
    """The ratio is NOT pinned to -alpha/2 at finite shear (would FAIL on the
    old hardcoded Psi2 = -alpha/2 Psi1)."""
    fl = fluid(a)
    ratio_10 = fl.Psi2(10.0 / LAM) / fl.Psi1(10.0 / LAM)
    assert abs(ratio_10 - (-a / 2)) > 0.05 * (a / 2)
    assert abs(ratio_10) < a / 2 + 1e-12          # magnitude bounded by alpha/2


@pytest.mark.parametrize("a", ALPHAS)
def test_eta_thins_gradually_no_cutoff(a):
    """eta > eta_s strictly for all finite Lambda (no spurious Lambda_crit cutoff)."""
    fl = fluid(a)
    gdot = np.logspace(-3, 6, 60) / LAM      # Lambda up to 1e9
    eta = fl.viscosity(gdot)
    assert np.all(eta > fl.eta_s)            # strictly above solvent everywhere
    assert np.all(np.diff(eta) <= 1e-12)     # monotone non-increasing


@pytest.mark.parametrize("a", ALPHAS)
def test_satisfies_giesekus_equations(a):
    """Solver-independent: reconstruct stress components from public eta,Psi1,Psi2
    and confirm they satisfy the steady Giesekus shear equations (relative
    residual at machine precision)."""
    fl = fluid(a)
    worst = 0.0
    for L in np.logspace(-3, 4, 30):
        g = L / LAM
        Sxy = (fl.viscosity(g) - ES) / EP * L
        Syy = fl.Psi2(g) * L ** 2 / (EP * LAM)
        Sxx = fl.Psi1(g) * L ** 2 / (EP * LAM) + Syy
        r1 = Sxx - 2 * L * Sxy + a * (Sxx ** 2 + Sxy ** 2)
        r2 = Syy + a * (Sxy ** 2 + Syy ** 2)
        r3 = Sxy - L * Syy + a * Sxy * (Sxx + Syy) - L
        s1 = max(abs(Sxx), 2 * L * abs(Sxy), a * (Sxx ** 2 + Sxy ** 2), 1.0)
        s2 = max(abs(Syy), a * (Sxy ** 2 + Syy ** 2), 1.0)
        s3 = max(abs(Sxy), L * abs(Syy), abs(a * Sxy * (Sxx + Syy)), L, 1.0)
        worst = max(worst, abs(r1) / s1, abs(r2) / s2, abs(r3) / s3)
    assert worst < 1e-11, f"alpha={a}: worst relative residual {worst:.2e}"


def test_alpha_zero_recovers_ucm():
    """alpha = 0 recovers UCM: constant eta = eta_0, Psi1 = 2 eta_p lambda, Psi2 = 0."""
    fl = GiesekusFluid(eta_s=ES, eta_p=EP, lambda_=LAM, alpha=0.0)
    gdot = np.logspace(-2, 6, 40) / LAM
    assert np.allclose(fl.viscosity(gdot), fl.eta_0, rtol=1e-9)
    assert np.allclose(fl.Psi1(gdot), 2 * EP * LAM, rtol=1e-9)
    assert np.allclose(fl.Psi2(gdot), 0.0, atol=1e-20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
