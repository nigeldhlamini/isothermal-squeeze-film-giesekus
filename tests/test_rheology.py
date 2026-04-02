"""
Unit Tests for Rheology Module
==============================

Test the Giesekus material functions against known analytical limits
and verify the key constraint Ψ₂/Ψ₁ = -α/2.

Run with: pytest tests/test_rheology.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.rheology import (
    GiesekusFluid, SOFFluid,
    create_newtonian, create_PAM_2pct, create_PAM_5pct,
    verify_psi_ratio, verify_sof_limit
)


class TestGiesekusFluidBasics:
    """Test basic GiesekusFluid functionality."""
    
    def test_creation(self):
        """Test fluid creation with valid parameters."""
        fluid = GiesekusFluid(eta_s=0.03, eta_p=0.12, lambda_=3.3e-3, alpha=0.25)
        
        assert fluid.eta_s == 0.03
        assert fluid.eta_p == 0.12
        assert fluid.lambda_ == 3.3e-3
        assert fluid.alpha == 0.25
    
    def test_derived_quantities(self):
        """Test computation of derived quantities."""
        fluid = GiesekusFluid(eta_s=0.03, eta_p=0.12, lambda_=3.3e-3, alpha=0.25)
        
        # η₀ = η_s + η_p
        assert np.isclose(fluid.eta_0, 0.15)
        
        # Ψ₁⁰ = 2·η_p·λ
        expected_Psi1_0 = 2 * 0.12 * 3.3e-3
        assert np.isclose(fluid.Psi1_0, expected_Psi1_0)
        
        # Ψ₂⁰ = -α·η_p·λ
        expected_Psi2_0 = -0.25 * 0.12 * 3.3e-3
        assert np.isclose(fluid.Psi2_0, expected_Psi2_0)
        
        # β = η_s/η₀
        assert np.isclose(fluid.beta, 0.03 / 0.15)
    
    def test_validation_negative_viscosity(self):
        """Test that negative viscosity raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            GiesekusFluid(eta_s=-0.03, eta_p=0.12, lambda_=3.3e-3, alpha=0.25)
    
    def test_validation_alpha_out_of_range(self):
        """Test that α outside [0, 0.5] raises error."""
        with pytest.raises(ValueError, match="0.5"):
            GiesekusFluid(eta_s=0.03, eta_p=0.12, lambda_=3.3e-3, alpha=0.6)


class TestGiesekusMaterialFunctions:
    """Test Giesekus material functions."""
    
    @pytest.fixture
    def fluid_5pct(self):
        """5% PAM fluid fixture."""
        return create_PAM_5pct()
    
    def test_viscosity_low_shear(self, fluid_5pct):
        """Test η → η₀ as γ̇ → 0."""
        gdot_low = 1e-3  # Very low shear rate
        eta = fluid_5pct.viscosity(gdot_low)
        
        assert np.isclose(eta, fluid_5pct.eta_0, rtol=0.01)
    
    def test_viscosity_high_shear(self, fluid_5pct):
        """Test η → η_s as γ̇ → ∞."""
        gdot_high = 1e6  # Very high shear rate
        eta = fluid_5pct.viscosity(gdot_high)
        
        # Should approach solvent viscosity (but not exactly equal)
        assert eta < fluid_5pct.eta_0
        assert eta > fluid_5pct.eta_s * 0.9  # Within 10% of solvent
    
    def test_viscosity_shear_thinning(self, fluid_5pct):
        """Test that viscosity decreases with shear rate (shear-thinning)."""
        gdot_values = np.logspace(0, 6, 20)
        eta_values = fluid_5pct.viscosity(gdot_values)
        
        # Check monotonic decrease (shear-thinning)
        for i in range(len(eta_values) - 1):
            assert eta_values[i] >= eta_values[i+1] * 0.999  # Allow tiny numerical noise
    
    def test_Psi1_low_shear(self, fluid_5pct):
        """Test Ψ₁ → Ψ₁⁰ as γ̇ → 0."""
        gdot_low = 1e-3
        Psi1 = fluid_5pct.Psi1(gdot_low)
        
        assert np.isclose(Psi1, fluid_5pct.Psi1_0, rtol=0.01)
    
    def test_Psi2_low_shear(self, fluid_5pct):
        """Test Ψ₂ → Ψ₂⁰ as γ̇ → 0."""
        gdot_low = 1e-3
        Psi2 = fluid_5pct.Psi2(gdot_low)
        
        assert np.isclose(Psi2, fluid_5pct.Psi2_0, rtol=0.01)
    
    def test_normal_stress_ratio_constraint(self, fluid_5pct):
        """Test that Ψ₂/Ψ₁ = -α/2 across all shear rates."""
        gdot_values = np.logspace(0, 6, 50)
        ratio = fluid_5pct.normal_stress_ratio(gdot_values)
        expected = -fluid_5pct.alpha / 2.0
        
        # Should be within 5% of expected at all shear rates
        assert np.allclose(ratio, expected, rtol=0.05)
    
    def test_vectorized_input(self, fluid_5pct):
        """Test that material functions handle array input."""
        gdot = np.array([10, 100, 1000, 10000])
        
        eta = fluid_5pct.viscosity(gdot)
        Psi1 = fluid_5pct.Psi1(gdot)
        Psi2 = fluid_5pct.Psi2(gdot)
        
        assert eta.shape == gdot.shape
        assert Psi1.shape == gdot.shape
        assert Psi2.shape == gdot.shape
    
    def test_scalar_input(self, fluid_5pct):
        """Test that material functions handle scalar input."""
        gdot = 1000.0
        
        eta = fluid_5pct.viscosity(gdot)
        Psi1 = fluid_5pct.Psi1(gdot)
        Psi2 = fluid_5pct.Psi2(gdot)
        
        assert isinstance(eta, float)
        assert isinstance(Psi1, float)
        assert isinstance(Psi2, float)


class TestNewtonianLimit:
    """Test Newtonian fluid limit (λ = 0)."""
    
    def test_newtonian_creation(self):
        """Test Newtonian fluid creation."""
        fluid = create_newtonian(eta=0.05)
        
        assert fluid.is_newtonian
        assert fluid.eta_0 == 0.05
        assert fluid.Psi1_0 == 0.0
        assert fluid.Psi2_0 == 0.0
    
    def test_newtonian_constant_viscosity(self):
        """Test that Newtonian fluid has constant viscosity."""
        fluid = create_newtonian(eta=0.05)
        
        gdot_values = np.logspace(-3, 6, 50)
        eta_values = fluid.viscosity(gdot_values)
        
        assert np.allclose(eta_values, 0.05)
    
    def test_newtonian_zero_normal_stresses(self):
        """Test that Newtonian fluid has zero normal stress coefficients."""
        fluid = create_newtonian(eta=0.05)
        
        gdot_values = np.logspace(-3, 6, 50)
        Psi1_values = fluid.Psi1(gdot_values)
        Psi2_values = fluid.Psi2(gdot_values)
        
        assert np.allclose(Psi1_values, 0.0)
        assert np.allclose(Psi2_values, 0.0)


class TestSOFFluid:
    """Test Second-Order Fluid class."""
    
    def test_sof_from_giesekus(self):
        """Test SOF creation from Giesekus fluid."""
        giesekus = create_PAM_5pct()
        sof = SOFFluid.from_giesekus(giesekus)
        
        assert sof.eta_0 == giesekus.eta_0
        assert sof.Psi1_0 == giesekus.Psi1_0
        assert sof.Psi2_0 == giesekus.Psi2_0
    
    def test_sof_constant_properties(self):
        """Test that SOF has constant material properties."""
        sof = SOFFluid(eta_0=0.15, Psi1_0=7.9e-4, Psi2_0=-9.9e-5)
        
        gdot_values = np.logspace(0, 6, 20)
        
        eta = sof.viscosity(gdot_values)
        Psi1 = sof.Psi1(gdot_values)
        Psi2 = sof.Psi2(gdot_values)
        
        assert np.allclose(eta, 0.15)
        assert np.allclose(Psi1, 7.9e-4)
        assert np.allclose(Psi2, -9.9e-5)


class TestVerificationFunctions:
    """Test built-in verification functions."""
    
    def test_verify_psi_ratio(self):
        """Test Ψ₂/Ψ₁ ratio verification."""
        fluid = create_PAM_5pct()
        result = verify_psi_ratio(fluid)
        
        assert result['all_pass'], f"Max error: {result['max_error']}"
        assert result['max_error'] < 0.01
    
    def test_verify_sof_limit(self):
        """Test SOF limit recovery at low Wi."""
        fluid = create_PAM_5pct()
        result = verify_sof_limit(fluid, gdot_test=1e-3)
        
        # At low Wi, Giesekus should match SOF
        assert result['Wi'] < 0.01, f"Wi = {result['Wi']} is not low enough"
        assert result['eta']['error'] < 0.01
        assert result['Psi1']['error'] < 0.01
        assert result['Psi2']['error'] < 0.01


class TestTableValues:
    """Test against Table 1 values from thesis."""
    
    def test_pam_2pct_parameters(self):
        """Test 2% PAM matches Table 1."""
        fluid = create_PAM_2pct()
        
        assert np.isclose(fluid.eta_0, 0.050, rtol=0.01)
        assert np.isclose(fluid.eta_s, 0.030, rtol=0.01)
        assert np.isclose(fluid.lambda_, 1.0e-3, rtol=0.01)
        assert np.isclose(fluid.alpha, 0.25, rtol=0.01)
        
        # Derived SOF parameters
        assert np.isclose(fluid.Psi1_0, 4.0e-5, rtol=0.01)
        assert np.isclose(fluid.Psi2_0, -5.0e-6, rtol=0.01)
    
    def test_pam_5pct_parameters(self):
        """Test 5% PAM matches Table 1."""
        fluid = create_PAM_5pct()
        
        assert np.isclose(fluid.eta_0, 0.150, rtol=0.01)
        assert np.isclose(fluid.eta_s, 0.030, rtol=0.01)
        assert np.isclose(fluid.lambda_, 3.3e-3, rtol=0.01)
        assert np.isclose(fluid.alpha, 0.25, rtol=0.01)
        
        # Derived SOF parameters
        assert np.isclose(fluid.Psi1_0, 7.9e-4, rtol=0.01)
        assert np.isclose(fluid.Psi2_0, -9.9e-5, rtol=0.01)
    
    def test_dimensionless_groups(self):
        """Test dimensionless group calculations from Table 1."""
        fluid = create_PAM_5pct()
        
        # Reference conditions: U = 10 m/s, L = 0.04 m, h₀ = 5 μm
        U = 10.0
        L = 0.04
        h0 = 5e-6
        
        De = fluid.De(U, L)
        Wi = fluid.Wi(U, h0)
        
        assert np.isclose(De, 0.83, rtol=0.05), f"De = {De}"
        assert np.isclose(Wi, 6600, rtol=0.05), f"Wi = {Wi}"


class TestComplexViscosity:
    """Test SAOS complex viscosity methods."""

    def test_zero_frequency_returns_eta0(self):
        """At omega=0, eta*(0) = eta_0 (purely real)."""
        fluid = create_PAM_5pct()
        eta_star = fluid.complex_viscosity(0.0)
        assert np.isclose(np.real(eta_star), fluid.eta_0, rtol=1e-10)
        assert np.isclose(np.imag(eta_star), 0.0, atol=1e-15)

    def test_high_frequency_approaches_eta_s(self):
        """At omega -> inf, Re[eta*] -> eta_s."""
        fluid = create_PAM_5pct()
        eta_star = fluid.complex_viscosity(1e12)
        assert np.isclose(np.real(eta_star), fluid.eta_s, rtol=1e-3)

    def test_newtonian_constant(self):
        """Newtonian fluid: eta*(omega) = eta_0 at all frequencies."""
        fluid = create_newtonian(eta=0.1)
        omegas = np.logspace(-2, 6, 50)
        eta_star = fluid.complex_viscosity(omegas)
        assert np.allclose(np.real(eta_star), 0.1, rtol=1e-10)
        assert np.allclose(np.imag(eta_star), 0.0, atol=1e-15)

    def test_eta_prime_double_prime_consistency(self):
        """eta' + i*eta'' should reconstruct eta* (with sign convention)."""
        fluid = create_PAM_5pct()
        omegas = np.logspace(0, 4, 20)
        eta_star = fluid.complex_viscosity(omegas)
        ep = fluid.eta_prime(omegas)
        epp = fluid.eta_double_prime(omegas)
        # eta* = eta' - i*eta''
        reconstructed = ep - 1j * epp
        assert np.allclose(eta_star, reconstructed, rtol=1e-12)

    def test_scalar_input(self):
        """Scalar input returns scalar output."""
        fluid = create_PAM_5pct()
        eta_star = fluid.complex_viscosity(100.0)
        assert isinstance(eta_star, complex)
        ep = fluid.eta_prime(100.0)
        assert isinstance(ep, float)
        epp = fluid.eta_double_prime(100.0)
        assert isinstance(epp, float)

    def test_array_input(self):
        """Array input returns array output."""
        fluid = create_PAM_5pct()
        omegas = np.array([10.0, 100.0, 1000.0])
        eta_star = fluid.complex_viscosity(omegas)
        assert isinstance(eta_star, np.ndarray)
        assert eta_star.shape == (3,)

    def test_eta_double_prime_positive(self):
        """eta'' should be positive (energy dissipation)."""
        fluid = create_PAM_5pct()
        omegas = np.logspace(-2, 6, 100)
        epp = fluid.eta_double_prime(omegas)
        assert np.all(epp >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
