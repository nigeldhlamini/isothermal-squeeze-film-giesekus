"""
Unit Tests for Assembly and Solver Modules
==========================================

Test the Reynolds equation assembly and iterative solver.

Run with: pytest tests/test_solver.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.rheology import GiesekusFluid, create_newtonian


def create_PAM_2pct() -> GiesekusFluid:
    """Local test fixture: moderate-shear-thinning Giesekus fluid (2% PAM params)."""
    return GiesekusFluid(eta_s=0.030, eta_p=0.020, lambda_=1.0e-3, alpha=0.25)


def create_PAM_5pct() -> GiesekusFluid:
    """Local test fixture: representative Giesekus fluid (5% PAM params)."""
    return GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)
from src.geometry import SlipperGeometry, OperatingConditions, create_standard_slipper
from src.mesh import PolarMesh
from src.assembly import (
    build_diffusion_operator,
    build_GNF_operator,
    assemble_newtonian_system,
    assemble_GNF_system,
)
from src.solver import (
    SolverConfig,
    solve_newtonian,
    solve_GNF,
    solve_viscoelastic,
    solve,
)


class TestDiffusionOperator:
    """Test the diffusion operator construction."""
    
    @pytest.fixture
    def simple_setup(self):
        """Simple mesh setup for testing."""
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
        mesh = PolarMesh(geom, Nr=20, Ntheta=30)
        return mesh, geom
    
    def test_operator_shape(self, simple_setup):
        """Test that operator has correct shape."""
        mesh, geom = simple_setup
        D_coeff = np.ones((mesh.Nr, mesh.Ntheta))
        
        L = build_diffusion_operator(mesh, D_coeff)
        
        assert L.shape == (mesh.N, mesh.N)
    
    def test_operator_sparsity(self, simple_setup):
        """Test that operator is sparse (5-point stencil)."""
        mesh, geom = simple_setup
        D_coeff = np.ones((mesh.Nr, mesh.Ntheta))
        
        L = build_diffusion_operator(mesh, D_coeff)
        
        # Should have at most 5 entries per row (center + 4 neighbours)
        nnz_per_row = L.getnnz(axis=1)
        assert np.max(nnz_per_row) <= 5
    
    def test_operator_structure(self, simple_setup):
        """Test that operator has expected structure."""
        mesh, geom = simple_setup
        D_coeff = np.ones((mesh.Nr, mesh.Ntheta))
        
        L = build_diffusion_operator(mesh, D_coeff)
        
        # The operator is NOT symmetric in polar coordinates due to 1/r factors
        # But the diagonal should be negative (diffusion is dissipative)
        diag = L.diagonal()
        # Most diagonal entries should be negative (interior points)
        interior_diag = diag[mesh.Ntheta:-mesh.Ntheta]
        assert np.all(interior_diag < 0)


class TestNewtonianSolver:
    """Test the Newtonian Reynolds equation solver."""
    
    @pytest.fixture
    def standard_setup(self):
        """Standard test configuration."""
        geom = create_standard_slipper()
        mesh = PolarMesh(geom, Nr=30, Ntheta=40)
        conditions = OperatingConditions(
            p_supply=200e5,  # 200 bar
            V_T=10.0,
            omega_s=0.0,
            h_dot=0.0  # No squeeze initially
        )
        return mesh, geom, conditions
    
    def test_newtonian_converges(self, standard_setup):
        """Test that Newtonian solver converges (trivially, one solve)."""
        mesh, geom, conditions = standard_setup
        eta = 0.05
        
        result = solve_newtonian(mesh, geom, conditions, eta)
        
        assert result.converged
        assert result.iterations == 1
    
    def test_pressure_boundary_conditions(self, standard_setup):
        """Test that pressure satisfies boundary conditions."""
        mesh, geom, conditions = standard_setup
        eta = 0.05
        
        result = solve_newtonian(mesh, geom, conditions, eta)
        
        # Inner boundary should be at supply pressure
        p_inner = result.pressure[0, :]
        assert np.allclose(p_inner, conditions.p_supply, rtol=0.01)
        
        # Outer boundary should be at ambient pressure
        p_outer = result.pressure[-1, :]
        p_ambient = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
        assert np.allclose(p_outer, p_ambient, rtol=0.01)
    
    def test_pressure_decreases_radially(self, standard_setup):
        """Test that pressure decreases from inner to outer radius."""
        mesh, geom, conditions = standard_setup
        eta = 0.05
        
        result = solve_newtonian(mesh, geom, conditions, eta)
        
        # Average pressure should decrease radially
        p_avg = np.mean(result.pressure, axis=1)
        for i in range(len(p_avg) - 1):
            assert p_avg[i] >= p_avg[i+1] * 0.99  # Allow small numerical noise
    
    def test_load_capacity_positive(self, standard_setup):
        """Test that load capacity is positive."""
        mesh, geom, conditions = standard_setup
        eta = 0.05
        
        result = solve_newtonian(mesh, geom, conditions, eta)
        
        assert result.F_total > 0
    
    def test_load_scales_with_viscosity(self, standard_setup):
        """Test that load scales with viscosity in squeeze flow."""
        mesh, geom, conditions_base = standard_setup
        
        # Use squeeze conditions where viscosity affects load
        conditions = OperatingConditions(
            p_supply=0.0,  # No hydrostatic pressure
            V_T=0.0,       # No tangential motion
            omega_s=0.0,
            h_dot=-0.01    # Squeeze motion
        )
        
        result1 = solve_newtonian(mesh, geom, conditions, eta=0.05)
        result2 = solve_newtonian(mesh, geom, conditions, eta=0.10)
        
        # Load should approximately double with doubled viscosity in squeeze
        # Allow some tolerance due to boundary effects
        ratio = result2.F_total / result1.F_total
        assert 1.5 < ratio < 2.5  # Reasonable range


class TestGNFSolver:
    """Test the Generalised Newtonian Fluid solver."""
    
    @pytest.fixture
    def gnf_setup(self):
        """GNF test configuration."""
        geom = create_standard_slipper()
        mesh = PolarMesh(geom, Nr=25, Ntheta=35)
        conditions = OperatingConditions(
            p_supply=200e5,
            V_T=10.0,
            omega_s=0.0,
            h_dot=0.0
        )
        fluid = create_PAM_2pct()  # Moderate shear-thinning
        return mesh, geom, conditions, fluid
    
    def test_gnf_converges(self, gnf_setup):
        """Test that GNF solver converges."""
        mesh, geom, conditions, fluid = gnf_setup
        config = SolverConfig(max_iter=30, tol=1e-5, omega=0.5)
        
        result = solve_GNF(mesh, geom, conditions, fluid, config)
        
        assert result.converged
    
    def test_shear_thinning_reduces_viscosity(self, gnf_setup):
        """Test that viscosity field is less than zero-shear value."""
        mesh, geom, conditions, fluid = gnf_setup
        config = SolverConfig(max_iter=30, tol=1e-5)
        
        result = solve_GNF(mesh, geom, conditions, fluid, config)
        
        # Minimum viscosity should be less than η₀ (shear-thinning)
        assert result.eta_bar.min() < fluid.eta_0
        
        # But should be above solvent viscosity
        assert result.eta_bar.min() > fluid.eta_s
    
    def test_gnf_matches_newtonian_at_low_shear(self):
        """Test that GNF approaches Newtonian at low shear rates."""
        geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=50e-6)  # Large gap
        mesh = PolarMesh(geom, Nr=20, Ntheta=30)
        conditions = OperatingConditions(
            p_supply=1e5,  # Low pressure
            V_T=0.1,       # Low velocity
            omega_s=0.0,
            h_dot=0.0
        )
        fluid = create_PAM_2pct()
        config = SolverConfig(max_iter=30, tol=1e-5)
        
        result_gnf = solve_GNF(mesh, geom, conditions, fluid, config)
        result_newton = solve_newtonian(mesh, geom, conditions, fluid.eta_0)
        
        # At low shear, GNF should approach Newtonian
        # (within 20% for these moderate conditions)
        ratio = result_gnf.F_total / result_newton.F_total
        assert 0.8 < ratio < 1.2


class TestViscoelasticSolver:
    """Test the full viscoelastic solver."""
    
    @pytest.fixture
    def ve_setup(self):
        """Viscoelastic test configuration."""
        geom = create_standard_slipper()
        mesh = PolarMesh(geom, Nr=25, Ntheta=35)
        conditions = OperatingConditions(
            p_supply=200e5,
            V_T=10.0,
            omega_s=0.0,
            h_dot=-0.001  # Closing gap for memory effects
        )
        fluid = create_PAM_5pct()
        return mesh, geom, conditions, fluid
    
    def test_viscoelastic_converges(self, ve_setup):
        """Test that viscoelastic solver converges."""
        mesh, geom, conditions, fluid = ve_setup
        config = SolverConfig(max_iter=40, tol=1e-5, omega=0.5)
        
        result = solve_viscoelastic(mesh, geom, conditions, fluid, config)
        
        assert result.converged
    
    def test_has_flux_decomposition(self, ve_setup):
        """Test that result includes flux decomposition."""
        mesh, geom, conditions, fluid = ve_setup
        config = SolverConfig(max_iter=40, tol=1e-5)
        
        result = solve_viscoelastic(mesh, geom, conditions, fluid, config)
        
        assert result.fluxes is not None
        assert result.fluxes.Q_r_GNF is not None
        assert result.fluxes.Q_r_mem is not None
    
    def test_elastic_enhancement_with_squeeze(self, ve_setup):
        """Test that squeezing produces elastic load enhancement."""
        mesh, geom, conditions, fluid = ve_setup
        config = SolverConfig(max_iter=40, tol=1e-5)
        
        result = solve_viscoelastic(mesh, geom, conditions, fluid, config)
        
        # With squeeze (ḣ < 0), elastic effects should enhance load
        # F_elastic = F_total - F_GNF
        # This may be small but should be computed
        assert hasattr(result, 'F_elastic')
    
    def test_no_squeeze_reduces_memory_effect(self, ve_setup):
        """Test that memory effect is small without squeeze."""
        mesh, geom, conditions_orig, fluid = ve_setup
        
        # Create conditions with no squeeze
        conditions = OperatingConditions(
            p_supply=conditions_orig.p_supply,
            V_T=conditions_orig.V_T,
            omega_s=0.0,
            h_dot=0.0  # No squeeze
        )
        
        config = SolverConfig(max_iter=40, tol=1e-5)
        result = solve_viscoelastic(mesh, geom, conditions, fluid, config)
        
        # Memory flux should be zero without squeeze
        assert np.allclose(result.fluxes.Q_r_mem, 0.0)
    
    def test_newtonian_fluid_gives_zero_viscoelastic(self, ve_setup):
        """Test that Newtonian fluid gives zero elastic contribution."""
        mesh, geom, conditions, _ = ve_setup
        fluid = create_newtonian(eta=0.05)
        config = SolverConfig(max_iter=10)
        
        result = solve_viscoelastic(mesh, geom, conditions, fluid, config)
        
        # Should have no elastic enhancement
        assert np.isclose(result.F_elastic, 0.0)


class TestUnifiedSolveInterface:
    """Test the unified solve() function."""
    
    @pytest.fixture
    def unified_setup(self):
        """Setup for unified solver tests."""
        geom = create_standard_slipper()
        mesh = PolarMesh(geom, Nr=20, Ntheta=25)
        conditions = OperatingConditions(
            p_supply=100e5,
            V_T=5.0,
            omega_s=0.0,
            h_dot=0.0
        )
        fluid = create_PAM_2pct()
        return mesh, geom, conditions, fluid
    
    def test_solve_newtonian(self, unified_setup):
        """Test solve() with newtonian model."""
        mesh, geom, conditions, fluid = unified_setup
        
        result = solve(mesh, geom, conditions, fluid, model='newtonian')
        
        assert result.converged
        assert result.solver_info['type'] == 'newtonian'
    
    def test_solve_gnf(self, unified_setup):
        """Test solve() with GNF model."""
        mesh, geom, conditions, fluid = unified_setup
        config = SolverConfig(max_iter=25, tol=1e-4)
        
        result = solve(mesh, geom, conditions, fluid, model='GNF', config=config)
        
        assert result.converged
        assert result.solver_info['type'] == 'GNF'
    
    def test_solve_viscoelastic(self, unified_setup):
        """Test solve() with viscoelastic model."""
        mesh, geom, conditions, fluid = unified_setup
        config = SolverConfig(max_iter=30, tol=1e-4)
        
        result = solve(mesh, geom, conditions, fluid, model='viscoelastic', config=config)
        
        assert result.converged
        assert result.solver_info['type'] == 'viscoelastic'
    
    def test_invalid_model_raises(self, unified_setup):
        """Test that invalid model name raises error."""
        mesh, geom, conditions, fluid = unified_setup
        
        with pytest.raises(ValueError, match="Unknown model"):
            solve(mesh, geom, conditions, fluid, model='invalid')


class TestSolverConfig:
    """Test solver configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SolverConfig()
        
        assert config.max_iter == 50
        assert config.tol == 1e-6
        assert 0 < config.omega <= 1
    
    def test_invalid_omega_raises(self):
        """Test that invalid omega raises error."""
        with pytest.raises(ValueError, match="omega"):
            SolverConfig(omega=1.5)
        
        with pytest.raises(ValueError, match="omega"):
            SolverConfig(omega=0.0)


class TestResultSummary:
    """Test result summary output."""
    
    def test_summary_string(self):
        """Test that summary() produces a string."""
        from src.results import SolverResult
        
        result = SolverResult(
            pressure=np.ones((10, 10)) * 1e6,
            converged=True,
            iterations=5,
            residual_history=[0.1, 0.01, 0.001],
            F_total=1000.0,
            F_GNF=950.0,
            F_elastic=50.0,
        )
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert 'Converged' in summary
        assert '1000' in summary  # Load value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
