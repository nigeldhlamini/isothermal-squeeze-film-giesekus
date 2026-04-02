"""
Unit Tests for Fluxes Module
============================

Test the flux components against known analytical limits and dimensional consistency.

Run with: pytest tests/test_fluxes.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.rheology import GiesekusFluid, create_newtonian, create_PAM_5pct
from src.geometry import SlipperGeometry, OperatingConditions, create_standard_slipper
from src.mesh import PolarMesh
from src.fluxes import (
    compute_gap_averaged_shear_rate,
    compute_Gamma_squared,
    compute_Q_GNF,
    compute_Q_memory,
    compute_Q_alpha,
    compute_Q_N1,
    compute_Q_hoop,
    compute_source_term,
    compute_newtonian_fluxes,
    FluxCalculator,
)


class TestGapAveragedShearRate:
    """Test gap-averaged shear rate computation."""
    
    def test_pure_couette(self):
        """Test shear rate for pure Couette flow (no pressure gradient)."""
        # Setup
        h = 5e-6 * np.ones((10, 10))  # 5 μm gap
        V_Tr = 5.0 * np.ones_like(h)  # 5 m/s radial velocity
        V_Ttheta = np.zeros_like(h)
        dp_dr = np.zeros_like(h)
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)  # 15 mm radius
        eta = 0.05 * np.ones_like(h)
        
        gdot_bar = compute_gap_averaged_shear_rate(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta
        )
        
        # Expected: γ̇ = V/h = 5/(5e-6) = 1e6 s⁻¹
        expected = 5.0 / 5e-6
        assert np.allclose(gdot_bar, expected, rtol=0.01)
    
    def test_pure_poiseuille(self):
        """Test shear rate for pure Poiseuille flow (no wall velocity)."""
        h = 5e-6 * np.ones((10, 10))
        V_Tr = np.zeros_like(h)
        V_Ttheta = np.zeros_like(h)
        dp_dr = -1e9 * np.ones_like(h)  # Strong pressure gradient
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        
        gdot_bar = compute_gap_averaged_shear_rate(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta
        )
        
        # Poiseuille contribution: h²|∇p|/(12η²) → h|∇p|/(√12 η)
        expected = (h[0,0] * abs(dp_dr[0,0])) / (np.sqrt(12) * eta[0,0])
        assert np.allclose(gdot_bar, expected, rtol=0.01)
    
    def test_combined_flow(self):
        """Test that combined Couette+Poiseuille gives correct magnitude."""
        h = 5e-6 * np.ones((10, 10))
        V_Tr = 5.0 * np.ones_like(h)
        V_Ttheta = np.zeros_like(h)
        dp_dr = -1e9 * np.ones_like(h)
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        
        gdot_bar = compute_gap_averaged_shear_rate(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta
        )
        
        # Should be larger than either component alone
        couette_only = 5.0 / 5e-6
        poiseuille_only = (h[0,0] * abs(dp_dr[0,0])) / (np.sqrt(12) * eta[0,0])
        
        assert gdot_bar[0,0] > couette_only * 0.99
        assert gdot_bar[0,0] > poiseuille_only * 0.99


class TestGNFFlux:
    """Test Generalised Newtonian Fluid flux."""
    
    def test_pure_couette_flux(self):
        """Test Couette flux: Q = (h/2)V."""
        h = 5e-6 * np.ones((10, 10))
        V_Tr = 5.0 * np.ones_like(h)
        V_Ttheta = np.zeros_like(h)
        dp_dr = np.zeros_like(h)
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        
        Q_r, Q_theta = compute_Q_GNF(h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta)
        
        # Expected: Q_r = h*V/2 = 5e-6 * 5 / 2 = 1.25e-5 m²/s
        expected = 5e-6 * 5.0 / 2
        assert np.allclose(Q_r, expected, rtol=0.01)
        assert np.allclose(Q_theta, 0.0)
    
    def test_pure_poiseuille_flux(self):
        """Test Poiseuille flux: Q = -h³/(12η) · ∂p/∂r."""
        h = 5e-6 * np.ones((10, 10))
        V_Tr = np.zeros_like(h)
        V_Ttheta = np.zeros_like(h)
        dp_dr = -1e8 * np.ones_like(h)  # Negative = outward pressure drop
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        
        Q_r, Q_theta = compute_Q_GNF(h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta)
        
        # Expected: Q_r = -h³/(12η) · dp/dr
        expected = -(h[0,0]**3 / (12 * eta[0,0])) * dp_dr[0,0]
        assert np.allclose(Q_r, expected, rtol=0.01)
    
    def test_dimensional_consistency(self):
        """Test that flux has correct dimensions [m²/s]."""
        h = 5e-6 * np.ones((10, 10))
        V_Tr = 5.0 * np.ones_like(h)
        V_Ttheta = 3.0 * np.ones_like(h)
        dp_dr = -1e8 * np.ones_like(h)
        dp_dtheta = -5e6 * np.ones_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        
        Q_r, Q_theta = compute_Q_GNF(h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta)
        
        # Q should be positive for outward flow with these conditions
        # Order of magnitude: h*V ~ 5e-6 * 5 ~ 2.5e-5 m²/s
        assert np.all(np.abs(Q_r) < 1e-3)  # Reasonable range
        assert np.all(np.abs(Q_r) > 1e-10)


class TestMemoryFlux:
    """Test squeeze-film memory flux."""
    
    def test_zero_squeeze_gives_zero_memory(self):
        """Test that Q_mem = 0 when ḣ = 0."""
        h = 5e-6 * np.ones((10, 10))
        h_dot = 0.0
        dp_dr = -1e8 * np.ones_like(h)
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        eta_T = 0.04 * np.ones_like(h)  # Tangent viscosity
        lambda_ = 3.3e-3
        
        Q_r_mem, Q_theta_mem = compute_Q_memory(
            h, h_dot, dp_dr, dp_dtheta, r, eta, eta_T, lambda_
        )
        
        assert np.allclose(Q_r_mem, 0.0)
        assert np.allclose(Q_theta_mem, 0.0)
    
    def test_squeeze_sign_convention(self):
        """Test that closing gap (ḣ < 0) gives positive outward flux correction."""
        h = 5e-6 * np.ones((10, 10))
        h_dot = -0.001  # Closing gap
        dp_dr = -1e8 * np.ones_like(h)  # Negative = pressure decreasing outward
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        eta_T = 0.04 * np.ones_like(h)
        lambda_ = 3.3e-3
        
        Q_r_mem, _ = compute_Q_memory(
            h, h_dot, dp_dr, dp_dtheta, r, eta, eta_T, lambda_
        )
        
        # Q_r_mem = -λh²ḣη_T/(16η²) × ∂p/∂r
        # With ḣ < 0 and ∂p/∂r < 0: 
        #   Q_r_mem = -(+)(-)(-)/(...) = -(+)/(...) < 0
        # Negative = opposes outward flow (reduces net outflow)
        # This leads to LOAD REDUCTION (consistent with Phan-Thien & Tanner 1983)
        assert np.all(Q_r_mem < 0)
    
    def test_memory_scaling_with_lambda(self):
        """Test that memory flux scales linearly with λ."""
        h = 5e-6 * np.ones((10, 10))
        h_dot = -0.001
        dp_dr = -1e8 * np.ones_like(h)
        dp_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05 * np.ones_like(h)
        eta_T = 0.04 * np.ones_like(h)
        
        Q_r_1, _ = compute_Q_memory(h, h_dot, dp_dr, dp_dtheta, r, eta, eta_T, 1e-3)
        Q_r_2, _ = compute_Q_memory(h, h_dot, dp_dr, dp_dtheta, r, eta, eta_T, 2e-3)
        
        # Should scale linearly
        assert np.allclose(Q_r_2 / Q_r_1, 2.0, rtol=0.01)


class TestNormalStressFluxes:
    """Test N₁ and hoop stress flux components."""
    
    def test_N1_zero_for_uniform_field(self):
        """Test that Q_N1 = 0 for uniform h·Ψ₁·Γ² field."""
        h = 5e-6 * np.ones((10, 10))
        Gamma_sq = 1e12 * np.ones_like(h)  # Uniform
        d_hPsiGamma_dr = np.zeros_like(h)  # Zero gradient
        eta_bar = 0.05 * np.ones_like(h)
        alpha = 0.25
        
        Q_r_N1 = compute_Q_N1(h, Gamma_sq, d_hPsiGamma_dr, eta_bar, alpha)
        
        assert np.allclose(Q_r_N1, 0.0)
    
    def test_hoop_always_positive(self):
        """Test that hoop stress flux is always positive (radially outward)."""
        h = 5e-6 * np.ones((10, 10))
        r = 0.015 * np.ones_like(h)
        Gamma_sq = 1e12 * np.ones_like(h)
        Psi1_bar = 5e-4 * np.ones_like(h)
        eta_bar = 0.05 * np.ones_like(h)
        alpha = 0.25
        
        Q_r_hoop = compute_Q_hoop(h, r, Gamma_sq, Psi1_bar, eta_bar, alpha)
        
        # Hoop stress creates outward flux
        assert np.all(Q_r_hoop > 0)
    
    def test_hoop_scaling_with_radius(self):
        """Test that hoop flux scales as 1/r."""
        h = 5e-6 * np.ones((10, 10))
        Gamma_sq = 1e12 * np.ones_like(h)
        Psi1_bar = 5e-4 * np.ones_like(h)
        eta_bar = 0.05 * np.ones_like(h)
        alpha = 0.25
        
        r1 = 0.010 * np.ones_like(h)
        r2 = 0.020 * np.ones_like(h)
        
        Q1 = compute_Q_hoop(h, r1, Gamma_sq, Psi1_bar, eta_bar, alpha)
        Q2 = compute_Q_hoop(h, r2, Gamma_sq, Psi1_bar, eta_bar, alpha)
        
        # Q ∝ 1/r → Q1/Q2 = r2/r1 = 2
        assert np.allclose(Q1 / Q2, 2.0, rtol=0.01)


class TestSourceTerm:
    """Test Reynolds equation source term."""
    
    def test_pure_squeeze(self):
        """Test source term for pure squeeze (no tangential motion)."""
        h = 5e-6 * np.ones((10, 10))
        h_dot = -0.001  # Closing
        V_Tr = np.zeros_like(h)
        V_Ttheta = np.zeros_like(h)
        dh_dr = np.zeros_like(h)
        dh_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        
        S = compute_source_term(h, h_dot, V_Tr, V_Ttheta, dh_dr, dh_dtheta, r)
        
        # Pure squeeze: S = ḣ
        assert np.allclose(S, h_dot)
    
    def test_wedge_effect(self):
        """Test that wedge (∂h/∂r ≠ 0) contributes to source."""
        h = 5e-6 * np.ones((10, 10))
        h_dot = 0.0
        V_Tr = 5.0 * np.ones_like(h)  # Radial velocity
        V_Ttheta = np.zeros_like(h)
        dh_dr = 1e-4 * np.ones_like(h)  # Positive wedge
        dh_dtheta = np.zeros_like(h)
        r = 0.015 * np.ones_like(h)
        
        S = compute_source_term(h, h_dot, V_Tr, V_Ttheta, dh_dr, dh_dtheta, r)
        
        # Wedge term: -(V_Tr/2)·∂h/∂r
        expected = -(V_Tr[0,0] / 2) * dh_dr[0,0]
        assert np.allclose(S, expected, rtol=0.01)


class TestFluxCalculator:
    """Test the complete FluxCalculator class."""
    
    @pytest.fixture
    def setup_calculator(self):
        """Setup standard test configuration."""
        fluid = create_PAM_5pct()
        geometry = create_standard_slipper()
        conditions = OperatingConditions(
            p_supply=200e5,  # 200 bar
            V_T=10.0,        # 10 m/s orbital
            omega_s=0.0,
            h_dot=-0.001     # Closing
        )
        mesh = PolarMesh(geometry, Nr=20, Ntheta=30)
        
        return FluxCalculator(fluid, geometry, conditions, mesh)
    
    def test_calculator_creation(self, setup_calculator):
        """Test that calculator can be created."""
        calc = setup_calculator
        assert calc.fluid is not None
        assert calc.geometry is not None
    
    def test_compute_all_fluxes(self, setup_calculator):
        """Test that all fluxes can be computed."""
        calc = setup_calculator
        
        # Create a simple pressure field
        R = calc.mesh.R
        p = calc.conditions.p_supply * (1 - (R - calc.geometry.R_in) / 
                                         (calc.geometry.R_out - calc.geometry.R_in))
        
        fluxes = calc.compute_all_fluxes(p)
        
        # Check all components present
        assert 'Q_r_total' in fluxes
        assert 'Q_r_GNF' in fluxes
        assert 'Q_r_mem' in fluxes
        assert 'Q_r_N1' in fluxes
        assert 'Q_r_hoop' in fluxes
        assert 'source' in fluxes
    
    def test_newtonian_has_no_viscoelastic(self, setup_calculator):
        """Test that Newtonian fluid gives zero viscoelastic fluxes."""
        calc = setup_calculator
        calc.fluid = create_newtonian(eta=0.05)
        
        R = calc.mesh.R
        p = 200e5 * (1 - (R - calc.geometry.R_in) / 
                     (calc.geometry.R_out - calc.geometry.R_in))
        
        fluxes = calc.compute_all_fluxes(p)
        
        # All viscoelastic terms should be zero
        assert np.allclose(fluxes['Q_r_mem'], 0.0)
        assert np.allclose(fluxes['Q_r_alpha'], 0.0)
        assert np.allclose(fluxes['Q_r_N1'], 0.0)
        assert np.allclose(fluxes['Q_r_hoop'], 0.0)
    
    def test_flux_ratios(self, setup_calculator):
        """Test flux ratio computation."""
        calc = setup_calculator
        
        R = calc.mesh.R
        p = 200e5 * (1 - (R - calc.geometry.R_in) / 
                     (calc.geometry.R_out - calc.geometry.R_in))
        
        fluxes = calc.compute_all_fluxes(p)
        ratios = calc.compute_flux_ratios(fluxes)
        
        # GNF should dominate
        assert ratios['GNF'] > 0.5
        
        # All ratios should be non-negative
        for name, value in ratios.items():
            assert value >= 0, f"{name} ratio is negative"


class TestNewtonianSpecialization:
    """Test Newtonian flux function."""
    
    def test_matches_gnf_with_constant_eta(self):
        """Test that Newtonian matches GNF with constant viscosity."""
        h = 5e-6 * np.ones((10, 10))
        V_Tr = 5.0 * np.ones_like(h)
        V_Ttheta = 3.0 * np.ones_like(h)
        dp_dr = -1e8 * np.ones_like(h)
        dp_dtheta = -5e6 * np.ones_like(h)
        r = 0.015 * np.ones_like(h)
        eta = 0.05
        
        Q_r_newton, Q_theta_newton = compute_newtonian_fluxes(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta
        )
        
        eta_array = eta * np.ones_like(h)
        Q_r_gnf, Q_theta_gnf = compute_Q_GNF(
            h, V_Tr, V_Ttheta, dp_dr, dp_dtheta, r, eta_array
        )
        
        assert np.allclose(Q_r_newton, Q_r_gnf)
        assert np.allclose(Q_theta_newton, Q_theta_gnf)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
