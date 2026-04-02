"""
Tests for EAF Fluid Models
===========================

Tests for the parametric EAF fluid model and regime classification.

Author: Nigel C. Dhlamini
"""

import pytest
import numpy as np

from src.eaf_fluids import (
    EAFFluidModel,
    FlowRegime,
    RegimeClassification,
    REPRESENTATIVE_FLUIDS,
    INDUSTRIAL_CONDITIONS,
    classify_regime,
    generate_regime_map,
    get_representative_fluid,
    create_fluid_range,
)


class TestEAFFluidModel:
    """Test the parametric EAF fluid model."""
    
    def test_newtonian_limit(self):
        """Polymer loading = 0 should give Newtonian fluid."""
        fluid = EAFFluidModel(base_viscosity=0.040, polymer_loading=0.0)
        
        assert fluid.eta_s == 0.040
        assert fluid.eta_p == 0.0
        assert fluid.lambda_ == 0.0
        assert fluid.eta_0 == 0.040
        assert fluid.Psi1_0 == 0.0
        assert fluid.Psi2_0 == 0.0
    
    def test_full_loading(self):
        """Polymer loading = 1 should give maximum VE properties."""
        fluid = EAFFluidModel(base_viscosity=0.030, polymer_loading=1.0)
        
        assert fluid.eta_s == 0.030
        assert fluid.eta_p == pytest.approx(0.120, rel=0.01)
        assert fluid.lambda_ == pytest.approx(3.3e-3, rel=0.01)
        assert fluid.eta_0 == pytest.approx(0.150, rel=0.01)
    
    def test_partial_loading(self):
        """Intermediate loading should interpolate linearly."""
        fluid = EAFFluidModel(polymer_loading=0.5)
        
        assert fluid.eta_p == pytest.approx(0.5 * 0.120, rel=0.01)
        assert fluid.lambda_ == pytest.approx(0.5 * 3.3e-3, rel=0.01)
    
    def test_psi_ratio_constraint(self):
        """Ψ₂/Ψ₁ = -α/2 should hold."""
        fluid = EAFFluidModel(polymer_loading=0.5, alpha=0.25)
        
        if fluid.Psi1_0 > 0:
            ratio = fluid.Psi2_0 / fluid.Psi1_0
            assert ratio == pytest.approx(-0.25 / 2, rel=0.01)
    
    def test_invalid_loading_raises(self):
        """Polymer loading outside [0, 1] should raise."""
        with pytest.raises(ValueError):
            EAFFluidModel(polymer_loading=-0.1)
        
        with pytest.raises(ValueError):
            EAFFluidModel(polymer_loading=1.5)
    
    def test_invalid_alpha_raises(self):
        """Alpha outside (0, 0.5] should raise."""
        with pytest.raises(ValueError):
            EAFFluidModel(alpha=0.0)
        
        with pytest.raises(ValueError):
            EAFFluidModel(alpha=0.6)
    
    def test_to_giesekus_conversion(self):
        """Conversion to GiesekusFluid should preserve parameters."""
        fluid = EAFFluidModel(polymer_loading=0.5, alpha=0.25)
        giesekus = fluid.to_giesekus()
        
        assert giesekus.eta_s == pytest.approx(fluid.eta_s, rel=0.01)
        assert giesekus.alpha == pytest.approx(fluid.alpha, rel=0.01)


class TestDimensionlessNumbers:
    """Test computation of dimensionless groups."""
    
    def test_newtonian_gives_zero_De_Wi(self):
        """Newtonian fluid should have De_sq = Wi = 0."""
        fluid = EAFFluidModel(polymer_loading=0.0)
        numbers = fluid.compute_dimensionless_numbers(h=5e-6, h_dot=-0.5e-3, V_T=8.0)
        
        assert numbers['De_sq'] == 0.0
        assert numbers['Wi'] == 0.0
    
    def test_De_sq_scales_with_lambda(self):
        """De_sq should scale linearly with λ."""
        fluid1 = EAFFluidModel(polymer_loading=0.5)
        fluid2 = EAFFluidModel(polymer_loading=1.0)
        
        nums1 = fluid1.compute_dimensionless_numbers(h=5e-6, h_dot=-0.5e-3, V_T=8.0)
        nums2 = fluid2.compute_dimensionless_numbers(h=5e-6, h_dot=-0.5e-3, V_T=8.0)
        
        # λ₂ = 2λ₁, so De_sq₂ = 2 De_sq₁
        assert nums2['De_sq'] == pytest.approx(2 * nums1['De_sq'], rel=0.01)
    
    def test_Wi_scales_with_velocity(self):
        """Wi should scale linearly with V_T."""
        fluid = EAFFluidModel(polymer_loading=0.5)
        
        nums1 = fluid.compute_dimensionless_numbers(h=5e-6, h_dot=-0.5e-3, V_T=4.0)
        nums2 = fluid.compute_dimensionless_numbers(h=5e-6, h_dot=-0.5e-3, V_T=8.0)
        
        assert nums2['Wi'] == pytest.approx(2 * nums1['Wi'], rel=0.01)
    
    def test_De_sq_scales_with_squeeze_rate(self):
        """De_sq should scale linearly with |ḣ|."""
        fluid = EAFFluidModel(polymer_loading=0.5)
        
        nums1 = fluid.compute_dimensionless_numbers(h=5e-6, h_dot=-0.25e-3, V_T=8.0)
        nums2 = fluid.compute_dimensionless_numbers(h=5e-6, h_dot=-0.5e-3, V_T=8.0)
        
        assert nums2['De_sq'] == pytest.approx(2 * nums1['De_sq'], rel=0.01)


class TestRegimeClassification:
    """Test flow regime classification."""
    
    def test_newtonian_regime(self):
        """Low De_sq and Wi should classify as Newtonian."""
        result = classify_regime(De_sq=1e-4, Wi=0.5)
        
        assert result.regime == FlowRegime.NEWTONIAN
        assert result.recommended_model == 'newtonian'
    
    def test_weak_ve_regime(self):
        """De_sq < 0.01 should classify as Weak VE."""
        result = classify_regime(De_sq=5e-3, Wi=50)
        
        assert result.regime == FlowRegime.WEAK_VE
        assert result.recommended_model == 'GNF'
    
    def test_moderate_ve_regime(self):
        """0.01 < De_sq < 0.1 should classify as Moderate VE."""
        result = classify_regime(De_sq=0.05, Wi=500)
        
        assert result.regime == FlowRegime.MODERATE_VE
        assert result.recommended_model == 'viscoelastic'
    
    def test_strong_ve_regime(self):
        """0.1 < De_sq < 0.5 should classify as Strong VE."""
        result = classify_regime(De_sq=0.3, Wi=2000)
        
        assert result.regime == FlowRegime.STRONG_VE
        assert result.recommended_model == 'viscoelastic'
        assert len(result.warnings) > 0  # Should have warnings
    
    def test_transient_regime(self):
        """De_sq > 0.5 should classify as Transient."""
        result = classify_regime(De_sq=0.7, Wi=5000)
        
        assert result.regime == FlowRegime.TRANSIENT
        assert result.recommended_model == 'CFD'
    
    def test_high_Wi_warning(self):
        """High Wi should generate a warning."""
        result = classify_regime(De_sq=0.05, Wi=500)
        
        warning_texts = ' '.join(result.warnings)
        assert 'Wi' in warning_texts or 'plateau' in warning_texts


class TestRepresentativeFluids:
    """Test pre-defined representative fluids."""
    
    def test_all_fluids_exist(self):
        """Check all expected fluids are defined."""
        expected = ['HFC_E46_Newtonian', 'HFC_enhanced_weak', 
                    'PEO_300k_0p3pct', 'PAM_2pct', 'PAM_5pct']
        
        for name in expected:
            assert name in REPRESENTATIVE_FLUIDS
    
    def test_hfc_newtonian_is_newtonian(self):
        """HFC-E46 should have zero polymer loading."""
        fluid = REPRESENTATIVE_FLUIDS['HFC_E46_Newtonian']
        
        assert fluid.polymer_loading == 0.0
        assert fluid.lambda_ == 0.0
    
    def test_pam_5pct_is_fully_loaded(self):
        """PAM 5% should have full polymer loading."""
        fluid = REPRESENTATIVE_FLUIDS['PAM_5pct']
        
        assert fluid.polymer_loading == 1.0
    
    def test_get_representative_fluid(self):
        """get_representative_fluid should return correct fluid."""
        fluid = get_representative_fluid('PAM_2pct')
        
        assert fluid.polymer_loading == pytest.approx(0.4, rel=0.1)
    
    def test_get_unknown_fluid_raises(self):
        """Unknown fluid name should raise ValueError."""
        with pytest.raises(ValueError):
            get_representative_fluid('unknown_fluid')


class TestFluidRange:
    """Test fluid range generation."""
    
    def test_create_fluid_range_count(self):
        """Should create requested number of fluids."""
        fluids = create_fluid_range(n_fluids=5)
        
        assert len(fluids) == 5
    
    def test_create_fluid_range_span(self):
        """Fluids should span min to max loading."""
        fluids = create_fluid_range(n_fluids=3, min_loading=0.0, max_loading=1.0)
        
        assert fluids[0].polymer_loading == pytest.approx(0.0, abs=0.01)
        assert fluids[-1].polymer_loading == pytest.approx(1.0, abs=0.01)
    
    def test_create_fluid_range_base_viscosity(self):
        """All fluids should have same base viscosity."""
        fluids = create_fluid_range(n_fluids=3, base_viscosity=0.050)
        
        for fluid in fluids:
            assert fluid.eta_s == 0.050


class TestIndustrialConditions:
    """Test industrial operating conditions."""
    
    def test_tangential_velocity_default(self):
        """Default tangential velocity should use typical rpm."""
        V_T = INDUSTRIAL_CONDITIONS.tangential_velocity()
        
        # V_T = ω * R_pcd = (2000 * 2π/60) * 0.040 ≈ 8.4 m/s
        assert V_T == pytest.approx(8.4, rel=0.1)
    
    def test_tangential_velocity_custom_rpm(self):
        """Custom rpm should scale velocity."""
        V_T_1000 = INDUSTRIAL_CONDITIONS.tangential_velocity(1000)
        V_T_2000 = INDUSTRIAL_CONDITIONS.tangential_velocity(2000)
        
        assert V_T_2000 == pytest.approx(2 * V_T_1000, rel=0.01)
    
    def test_shear_rate_range(self):
        """Shear rate range should be positive and ordered."""
        gdot_min, gdot_max = INDUSTRIAL_CONDITIONS.shear_rate_range()
        
        assert gdot_min > 0
        assert gdot_max > gdot_min
        assert gdot_max > 1e5  # Should be high for thin films


class TestRegimeMapGeneration:
    """Test regime map generation."""
    
    def test_generate_regime_map_shape(self):
        """Output arrays should have correct shape."""
        data = generate_regime_map(n_points=10)
        
        assert data['De_sq'].shape == (10, 10)
        assert data['Wi'].shape == (10, 10)
        assert data['regime'].shape == (10, 10)
        assert data['ve_correction'].shape == (10, 10)
    
    def test_generate_regime_map_range(self):
        """De_sq and Wi should span requested range."""
        data = generate_regime_map(
            De_sq_range=(1e-3, 1.0),
            Wi_range=(1, 1e4),
            n_points=10
        )
        
        assert data['De_sq'].min() >= 1e-3
        assert data['De_sq'].max() <= 1.0
        assert data['Wi'].min() >= 1
        assert data['Wi'].max() <= 1e4
    
    def test_regime_values_valid(self):
        """Regime codes should be in valid range."""
        data = generate_regime_map(n_points=10)
        
        assert data['regime'].min() >= 0
        assert data['regime'].max() <= len(FlowRegime) - 1


class TestFluidClassify:
    """Test fluid.classify() method."""
    
    def test_classify_newtonian_fluid(self):
        """Newtonian fluid should always be Regime I."""
        fluid = EAFFluidModel(polymer_loading=0.0)
        classification = fluid.classify(h=5e-6, h_dot=-0.5e-3, V_T=8.0)
        
        assert classification.regime == FlowRegime.NEWTONIAN
    
    def test_classify_varies_with_squeeze(self):
        """Higher squeeze rate should increase De_sq."""
        fluid = EAFFluidModel(polymer_loading=0.5)
        
        class1 = fluid.classify(h=5e-6, h_dot=-0.1e-3, V_T=8.0)
        class2 = fluid.classify(h=5e-6, h_dot=-1.0e-3, V_T=8.0)
        
        assert class2.De_sq > class1.De_sq
