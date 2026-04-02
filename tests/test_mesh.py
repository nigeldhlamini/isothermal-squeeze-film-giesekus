"""
Unit Tests for Mesh Module
==========================

Test the polar mesh and derivative operators against known analytical solutions.

Run with: pytest tests/test_mesh.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.geometry import SlipperGeometry
from src.mesh import PolarMesh, verify_derivative_accuracy, convergence_study


class TestPolarMeshBasics:
    """Test basic PolarMesh functionality."""
    
    @pytest.fixture
    def standard_geometry(self):
        """Standard slipper geometry fixture."""
        return SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
    
    @pytest.fixture
    def standard_mesh(self, standard_geometry):
        """Standard mesh fixture."""
        return PolarMesh(standard_geometry, Nr=40, Ntheta=60)
    
    def test_mesh_creation(self, standard_geometry):
        """Test mesh creation with valid parameters."""
        mesh = PolarMesh(standard_geometry, Nr=40, Ntheta=60)
        
        assert mesh.Nr == 40
        assert mesh.Ntheta == 60
        assert mesh.N == 40 * 60
    
    def test_mesh_spacing(self, standard_mesh):
        """Test mesh spacing calculations."""
        mesh = standard_mesh
        geom = mesh.geometry
        
        expected_dr = (geom.R_out - geom.R_in) / mesh.Nr
        expected_dtheta = 2 * np.pi / mesh.Ntheta
        
        assert np.isclose(mesh.dr, expected_dr)
        assert np.isclose(mesh.dtheta, expected_dtheta)
    
    def test_radial_range(self, standard_mesh):
        """Test that radial nodes are within bounds."""
        mesh = standard_mesh
        geom = mesh.geometry
        
        # Cell centres should be inside the domain
        assert np.all(mesh.r > geom.R_in)
        assert np.all(mesh.r < geom.R_out)
    
    def test_azimuthal_range(self, standard_mesh):
        """Test azimuthal node positions."""
        mesh = standard_mesh
        
        # Should cover [0, 2π) with cell centres
        assert mesh.theta[0] > 0
        assert mesh.theta[-1] < 2 * np.pi
    
    def test_meshgrid_shape(self, standard_mesh):
        """Test meshgrid array shapes."""
        mesh = standard_mesh
        
        assert mesh.R.shape == (mesh.Nr, mesh.Ntheta)
        assert mesh.THETA.shape == (mesh.Nr, mesh.Ntheta)
    
    def test_to_1d_2d_roundtrip(self, standard_mesh):
        """Test 1D/2D conversion roundtrip."""
        mesh = standard_mesh
        
        field_2d = np.random.rand(mesh.Nr, mesh.Ntheta)
        field_1d = mesh.to_1d(field_2d)
        field_2d_back = mesh.to_2d(field_1d)
        
        assert np.allclose(field_2d, field_2d_back)


class TestDerivativeOperators:
    """Test derivative operators against analytical solutions."""
    
    @pytest.fixture
    def mesh(self):
        """Standard mesh fixture for derivative tests."""
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        return PolarMesh(geom, Nr=60, Ntheta=80)
    
    def test_D_r_polynomial(self, mesh):
        """Test radial derivative on polynomial: f = r²."""
        r, theta = mesh.R, mesh.THETA
        
        f = r**2
        df_dr_exact = 2 * r
        
        df_dr, _ = mesh.compute_field_gradient(f)
        
        # Check interior points (exclude boundary rows)
        error = np.abs(df_dr[1:-1, :] - df_dr_exact[1:-1, :])
        max_error = np.max(error)
        
        assert max_error < 1e-8, f"Max error: {max_error}"
    
    def test_D_theta_trigonometric(self, mesh):
        """Test azimuthal derivative on trig function: f = sin(θ)."""
        r, theta = mesh.R, mesh.THETA
        
        f = np.sin(theta)
        df_dtheta_exact = np.cos(theta)
        
        _, df_dtheta = mesh.compute_field_gradient(f)
        
        error = np.abs(df_dtheta - df_dtheta_exact)
        max_error = np.max(error)
        
        # Second-order FD with dtheta ~ 0.1 rad gives O(dtheta²) ~ 0.01 error
        assert max_error < 0.01, f"Max error: {max_error}"
    
    def test_D_theta_periodic(self, mesh):
        """Test that azimuthal derivative handles periodic BC."""
        r, theta = mesh.R, mesh.THETA
        
        # Function with period 2π
        f = np.sin(2 * theta)
        df_dtheta_exact = 2 * np.cos(2 * theta)
        
        _, df_dtheta = mesh.compute_field_gradient(f)
        
        error = np.abs(df_dtheta - df_dtheta_exact)
        max_error = np.max(error)
        
        # Higher frequency means higher error
        assert max_error < 0.05, f"Max error: {max_error}"
    
    def test_mixed_derivative(self, mesh):
        """Test gradient of mixed function: f = r·cos(θ)."""
        r, theta = mesh.R, mesh.THETA
        
        f = r * np.cos(theta)
        df_dr_exact = np.cos(theta)
        df_dtheta_exact = -r * np.sin(theta)
        
        df_dr, df_dtheta = mesh.compute_field_gradient(f)
        
        # Radial (interior only)
        error_r = np.max(np.abs(df_dr[1:-1, :] - df_dr_exact[1:-1, :]))
        # Azimuthal (all points)
        error_theta = np.max(np.abs(df_dtheta - df_dtheta_exact))
        
        assert error_r < 1e-4, f"Max radial error: {error_r}"
        assert error_theta < 0.001, f"Max azimuthal error: {error_theta}"


class TestDerivativeConvergence:
    """Test that derivative operators converge with grid refinement."""
    
    def test_radial_convergence_order(self):
        """Test that radial derivative converges at O(Δr²)."""
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        
        Nr_list = [20, 40, 80]
        errors = []
        drs = []
        
        for Nr in Nr_list:
            mesh = PolarMesh(geom, Nr=Nr, Ntheta=Nr)
            
            # Test function: f = r³
            f = mesh.R**3
            df_dr_exact = 3 * mesh.R**2
            
            df_dr, _ = mesh.compute_field_gradient(f)
            
            # Error in interior
            error = np.max(np.abs(df_dr[2:-2, :] - df_dr_exact[2:-2, :]))
            errors.append(error)
            drs.append(mesh.dr)
        
        # Check convergence order
        # If error ~ C·Δr^p, then log(error) ~ p·log(Δr) + const
        log_dr = np.log(drs)
        log_err = np.log(errors)
        
        # Fit line to get order
        order, _ = np.polyfit(log_dr, log_err, 1)
        
        assert order > 1.8, f"Convergence order {order:.2f} < 2 (expected ~2)"


class TestIntegration:
    """Test numerical integration."""
    
    @pytest.fixture
    def mesh(self):
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        return PolarMesh(geom, Nr=80, Ntheta=120)
    
    def test_integrate_constant(self, mesh):
        """Test integration of constant field."""
        # ∫∫ 1 · r dr dθ = π(R_out² - R_in²)
        field = np.ones((mesh.Nr, mesh.Ntheta))
        integral = mesh.integrate_field(field)
        
        R_in = mesh.geometry.R_in
        R_out = mesh.geometry.R_out
        expected = np.pi * (R_out**2 - R_in**2)
        
        assert np.isclose(integral, expected, rtol=0.01)
    
    def test_integrate_radial(self, mesh):
        """Test integration of f = r."""
        # ∫∫ r · r dr dθ = 2π · ∫ r² dr = 2π · (R³_out - R³_in)/3
        field = mesh.R.copy()
        integral = mesh.integrate_field(field)
        
        R_in = mesh.geometry.R_in
        R_out = mesh.geometry.R_out
        expected = 2 * np.pi * (R_out**3 - R_in**3) / 3
        
        assert np.isclose(integral, expected, rtol=0.01)


class TestBoundaryConditions:
    """Test boundary condition application."""
    
    @pytest.fixture
    def mesh(self):
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        return PolarMesh(geom, Nr=20, Ntheta=30)
    
    def test_dirichlet_bc_inner(self, mesh):
        """Test Dirichlet BC at inner radius."""
        from scipy.sparse import eye
        
        matrix = eye(mesh.N, format='csr')
        rhs = np.zeros(mesh.N)
        
        bc_inner = 100.0
        bc_outer = 0.0
        
        matrix_bc, rhs_bc = mesh.apply_dirichlet_bc(matrix, rhs, bc_inner, bc_outer)
        
        # Check that inner boundary rows have BC value
        for j in range(mesh.Ntheta):
            k_inner = mesh._index(0, j)
            assert rhs_bc[k_inner] == bc_inner
    
    def test_dirichlet_bc_outer(self, mesh):
        """Test Dirichlet BC at outer radius."""
        from scipy.sparse import eye
        
        matrix = eye(mesh.N, format='csr')
        rhs = np.zeros(mesh.N)
        
        bc_inner = 0.0
        bc_outer = 200.0
        
        matrix_bc, rhs_bc = mesh.apply_dirichlet_bc(matrix, rhs, bc_inner, bc_outer)
        
        # Check that outer boundary rows have BC value
        for j in range(mesh.Ntheta):
            k_outer = mesh._index(mesh.Nr - 1, j)
            assert rhs_bc[k_outer] == bc_outer


class TestVerificationFunctions:
    """Test built-in verification functions."""
    
    def test_verify_derivative_accuracy(self):
        """Test the derivative accuracy verification function."""
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        mesh = PolarMesh(geom, Nr=40, Ntheta=60)
        
        result = verify_derivative_accuracy(mesh, test_function='polynomial')
        
        assert result['passed'], f"Errors: r={result['max_error_r']}, θ={result['max_error_theta']}"
    
    def test_convergence_study(self):
        """Test the convergence study function."""
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        
        result = convergence_study(geom, Nr_list=[20, 40, 80])
        
        # Should have computed order
        assert 'order_r' in result
        # Order should be positive (errors decrease with refinement)
        # Note: exact order may vary due to boundary stencils
        # Just check that refinement improves accuracy
        assert result['error_r'][-1] < result['error_r'][0], \
            "Error should decrease with refinement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
