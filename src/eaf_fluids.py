"""
Parametric Environmentally Acceptable Fluid (EAF) Model
========================================================

This module defines a parametric fluid model spanning the range of industrially
relevant EAF formulations, from Newtonian water-glycol bases to strongly
viscoelastic polymer-thickened systems.

The framework enables:
1. Classification of fluids into operating regimes based on De_sq and Wi
2. Generation of representative fluids across the parameter space
3. Industrial guidance on when viscoelastic modelling is necessary

Fluid Classes Covered:
- HFC: Water-glycol based (fire-resistant)
- HFDU: Polyol ester based (biodegradable)
- HETG: Triglyceride based (vegetable oils)
- Enhanced EAFs: Any of above with polymer VI improvers

Representative Commercial Fluids:
- Hydransafe HFC-E 46 (Totalenergies) - Low-MW PEG thickener
- Quintolubric 888-46 (Quaker Houghton) - Polyol ester
- Plantohydraulik 46 (Fuchs) - Rapeseed oil base

References:
    - Bair, S. (2019). High Pressure Rheology for Quantitative EHL.
    - Escudier, M.P. et al. (1999). J. Non-Newt. Fluid Mech. 81, 197-213.
    - Dontula, P. et al. (1998). J. Rheol. 42, 971-994.
    - Dhlamini, N.C. (2025). PhD Thesis, University of Cape Town.

Author: Nigel C. Dhlamini
Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import numpy as np

# Import from existing modules
try:
    from .rheology import GiesekusFluid
except ImportError:
    from rheology import GiesekusFluid


# =============================================================================
# Operating Regime Classification
# =============================================================================

class FlowRegime(Enum):
    """Classification of flow regimes based on dimensionless groups."""
    
    NEWTONIAN = "Newtonian"
    """De_sq < 0.001, Wi < 1: Newtonian approximation valid"""
    
    WEAK_VE = "Weak Viscoelastic"
    """De_sq < 0.01, Wi < 100: GNF model sufficient, VE corrections < 1%"""
    
    MODERATE_VE = "Moderate Viscoelastic"
    """0.01 < De_sq < 0.1, Wi < 1000: Giesekus RE valid, VE corrections 1-10%"""
    
    STRONG_VE = "Strong Viscoelastic"
    """0.1 < De_sq < 0.5: Giesekus RE marginal, VE corrections 10-30%"""
    
    TRANSIENT = "Transient Dominated"
    """De_sq > 0.5: Quasi-steady assumption invalid, full transient required"""


@dataclass
class RegimeClassification:
    """
    Classification result for a fluid/operating condition combination.
    
    Attributes
    ----------
    regime : FlowRegime
        Identified operating regime
    De_sq : float
        Squeeze Deborah number
    Wi : float
        Weissenberg number
    recommended_model : str
        Recommended model type ('newtonian', 'GNF', 'viscoelastic', 'CFD')
    expected_VE_correction : float
        Expected magnitude of viscoelastic correction (fractional)
    warnings : list of str
        Any warnings about model validity
    """
    regime: FlowRegime
    De_sq: float
    Wi: float
    recommended_model: str
    expected_VE_correction: float
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        s = f"Flow Regime: {self.regime.value}\n"
        s += f"  De_sq = {self.De_sq:.4f}\n"
        s += f"  Wi = {self.Wi:.0f}\n"
        s += f"  Recommended model: {self.recommended_model}\n"
        s += f"  Expected VE correction: {self.expected_VE_correction*100:.1f}%\n"
        if self.warnings:
            s += "  Warnings:\n"
            for w in self.warnings:
                s += f"    - {w}\n"
        return s


def classify_regime(
    De_sq: float, 
    Wi: float,
    eta_ratio: float = 0.2
) -> RegimeClassification:
    """
    Classify the operating regime based on dimensionless groups.
    
    Parameters
    ----------
    De_sq : float
        Squeeze Deborah number = λ|ḣ|/h
    Wi : float
        Weissenberg number = λγ̇
    eta_ratio : float
        Ratio η_s/η₀ (high-shear to zero-shear viscosity)
    
    Returns
    -------
    classification : RegimeClassification
        Complete regime classification with recommendations
    """
    warnings = []
    
    # Classify regime
    if De_sq < 0.001 and Wi < 1:
        regime = FlowRegime.NEWTONIAN
        model = 'newtonian'
        ve_correction = 0.0
        
    elif De_sq < 0.01:
        regime = FlowRegime.WEAK_VE
        model = 'GNF'
        ve_correction = De_sq * 0.5  # Rough estimate
        
    elif De_sq < 0.1:
        regime = FlowRegime.MODERATE_VE
        model = 'viscoelastic'
        ve_correction = De_sq * 2  # Memory effects scale with De_sq
        
    elif De_sq < 0.5:
        regime = FlowRegime.STRONG_VE
        model = 'viscoelastic'
        ve_correction = 0.1 + (De_sq - 0.1) * 0.5
        warnings.append("De_sq > 0.1: Perturbation expansion marginal")
        warnings.append("CFD validation recommended for quantitative predictions")
        
    else:
        regime = FlowRegime.TRANSIENT
        model = 'CFD'
        ve_correction = 0.3  # Unknown, need full transient
        warnings.append("De_sq > 0.5: Quasi-steady assumption invalid")
        warnings.append("Full transient CFD required")
    
    # Additional warnings
    if Wi > 100 and regime != FlowRegime.NEWTONIAN:
        warnings.append(f"High Wi = {Wi:.0f}: Viscosity in plateau region (η/η₀ → {eta_ratio:.2f})")
    
    if Wi > 10000:
        warnings.append("Extreme Wi: Consider finite extensibility effects (FENE model)")
    
    return RegimeClassification(
        regime=regime,
        De_sq=De_sq,
        Wi=Wi,
        recommended_model=model,
        expected_VE_correction=ve_correction,
        warnings=warnings
    )


# =============================================================================
# Parametric EAF Fluid Model
# =============================================================================

@dataclass
class EAFFluidModel:
    """
    Parametric model for Environmentally Acceptable Fluids.
    
    This model spans from Newtonian water-glycol bases to strongly viscoelastic
    polymer-thickened formulations through a single 'polymer_loading' parameter.
    
    Parameters
    ----------
    base_viscosity : float
        Base fluid (water-glycol) viscosity at reference temperature [Pa·s]
    polymer_loading : float
        Polymer contribution, 0 = none, 1 = "full" (5% PAM equivalent)
    alpha : float
        Giesekus mobility factor (typically 0.2-0.3)
    temperature : float
        Operating temperature [°C]
    fluid_class : str
        Fluid classification: 'HFC', 'HFDU', 'HETG', 'custom'
    
    Attributes
    ----------
    eta_s : float
        Solvent (base fluid) viscosity [Pa·s]
    eta_p : float
        Polymer viscosity contribution [Pa·s]
    lambda_ : float
        Relaxation time [s]
    eta_0 : float
        Zero-shear viscosity [Pa·s]
    
    Examples
    --------
    >>> # Newtonian HFC (like Hydransafe HFC-E 46)
    >>> fluid_N = EAFFluidModel(base_viscosity=0.040, polymer_loading=0.0)
    >>> fluid_N.eta_0
    0.040
    >>> fluid_N.lambda_
    0.0
    
    >>> # Moderately viscoelastic EAF
    >>> fluid_VE = EAFFluidModel(base_viscosity=0.030, polymer_loading=0.3)
    >>> fluid_VE.eta_0  # Increased by polymer
    0.066
    >>> fluid_VE.lambda_  # Non-zero relaxation time
    0.00099
    """
    
    base_viscosity: float = 0.030  # Pa·s (typical HFC at 40°C)
    polymer_loading: float = 0.0   # 0 = Newtonian, 1 = full (5% PAM equiv)
    alpha: float = 0.25            # Giesekus mobility
    temperature: float = 40.0      # °C
    fluid_class: str = 'HFC'       # Classification
    
    # Maximum polymer contribution (at loading = 1.0)
    _max_eta_p: float = field(default=0.120, repr=False)
    _max_lambda: float = field(default=3.3e-3, repr=False)
    
    def __post_init__(self):
        """Validate inputs."""
        if not 0 <= self.polymer_loading <= 1:
            raise ValueError(f"polymer_loading must be in [0, 1], got {self.polymer_loading}")
        if not 0 < self.alpha <= 0.5:
            raise ValueError(f"alpha must be in (0, 0.5], got {self.alpha}")
    
    @property
    def eta_s(self) -> float:
        """Solvent (base fluid) viscosity [Pa·s]."""
        return self.base_viscosity
    
    @property
    def eta_p(self) -> float:
        """Polymer viscosity contribution [Pa·s]."""
        return self.polymer_loading * self._max_eta_p
    
    @property
    def lambda_(self) -> float:
        """Relaxation time [s]."""
        return self.polymer_loading * self._max_lambda
    
    @property
    def eta_0(self) -> float:
        """Zero-shear viscosity [Pa·s]."""
        return self.eta_s + self.eta_p
    
    @property
    def Psi1_0(self) -> float:
        """Zero-shear first normal stress coefficient [Pa·s²]."""
        if self.lambda_ < 1e-15:
            return 0.0
        return 2 * self.eta_p * self.lambda_
    
    @property
    def Psi2_0(self) -> float:
        """Zero-shear second normal stress coefficient [Pa·s²]."""
        return -self.alpha / 2 * self.Psi1_0
    
    def to_giesekus(self) -> GiesekusFluid:
        """
        Convert to GiesekusFluid object for solver.
        
        Returns
        -------
        fluid : GiesekusFluid
            Giesekus fluid with equivalent parameters
        """
        # Handle Newtonian limit
        if self.polymer_loading < 1e-6:
            return GiesekusFluid(
                eta_s=self.eta_s,
                eta_p=1e-12,  # Tiny but non-zero for numerical stability
                lambda_=1e-15,
                alpha=self.alpha
            )
        
        return GiesekusFluid(
            eta_s=self.eta_s,
            eta_p=self.eta_p,
            lambda_=self.lambda_,
            alpha=self.alpha
        )
    
    def compute_dimensionless_numbers(
        self,
        h: float,
        h_dot: float,
        V_T: float
    ) -> Dict[str, float]:
        """
        Compute dimensionless groups for given operating conditions.
        
        Parameters
        ----------
        h : float
            Film thickness [m]
        h_dot : float
            Squeeze velocity [m/s] (negative for closing)
        V_T : float
            Tangential velocity [m/s]
        
        Returns
        -------
        numbers : dict
            Dictionary with Wi, De_sq, and typical shear rate
        """
        # Typical shear rate (Couette-dominated)
        gdot = abs(V_T) / h if h > 0 else 0
        
        Wi = self.lambda_ * gdot if self.lambda_ > 0 else 0
        De_sq = self.lambda_ * abs(h_dot) / h if (self.lambda_ > 0 and h > 0) else 0
        
        return {
            'Wi': Wi,
            'De_sq': De_sq,
            'gdot': gdot,
            'lambda': self.lambda_,
            'eta_0': self.eta_0,
            'eta_s_ratio': self.eta_s / self.eta_0 if self.eta_0 > 0 else 1.0
        }
    
    def classify(self, h: float, h_dot: float, V_T: float) -> RegimeClassification:
        """
        Classify the operating regime for given conditions.
        
        Parameters
        ----------
        h : float
            Film thickness [m]
        h_dot : float
            Squeeze velocity [m/s]
        V_T : float
            Tangential velocity [m/s]
        
        Returns
        -------
        classification : RegimeClassification
            Complete regime classification
        """
        numbers = self.compute_dimensionless_numbers(h, h_dot, V_T)
        eta_ratio = self.eta_s / self.eta_0 if self.eta_0 > 0 else 1.0
        return classify_regime(numbers['De_sq'], numbers['Wi'], eta_ratio)
    
    def __str__(self) -> str:
        s = f"EAF Fluid Model ({self.fluid_class})\n"
        s += f"  Base viscosity: {self.eta_s*1000:.1f} mPa·s\n"
        s += f"  Polymer loading: {self.polymer_loading*100:.0f}%\n"
        s += f"  Zero-shear viscosity: {self.eta_0*1000:.1f} mPa·s\n"
        s += f"  Relaxation time: {self.lambda_*1000:.3f} ms\n"
        s += f"  Mobility factor α: {self.alpha:.2f}\n"
        return s


# =============================================================================
# Pre-defined Representative Fluids
# =============================================================================

# Literature-based fluid definitions
REPRESENTATIVE_FLUIDS = {
    # Newtonian/near-Newtonian HFCs
    'HFC_E46_Newtonian': EAFFluidModel(
        base_viscosity=0.040,  # ISO VG 46 at 40°C
        polymer_loading=0.0,
        fluid_class='HFC'
    ),
    
    # Weak viscoelasticity (enhanced HFC)
    'HFC_enhanced_weak': EAFFluidModel(
        base_viscosity=0.035,
        polymer_loading=0.1,  # Light polymer addition
        alpha=0.25,
        fluid_class='HFC'
    ),
    
    # Moderate viscoelasticity (PEO-type)
    'PEO_300k_0p3pct': EAFFluidModel(
        base_viscosity=0.001,  # Water base
        polymer_loading=0.25,  # ~0.3% PEO 300k equivalent
        alpha=0.23,
        fluid_class='custom'
    ),
    
    # Moderate-strong viscoelasticity
    'PAM_2pct': EAFFluidModel(
        base_viscosity=0.030,
        polymer_loading=0.4,  # ~2% PAM
        alpha=0.25,
        fluid_class='custom'
    ),
    
    # Strong viscoelasticity (thesis reference)
    'PAM_5pct': EAFFluidModel(
        base_viscosity=0.030,
        polymer_loading=1.0,  # Full 5% PAM
        alpha=0.25,
        fluid_class='custom'
    ),
}


def get_representative_fluid(name: str) -> EAFFluidModel:
    """
    Get a pre-defined representative fluid by name.
    
    Parameters
    ----------
    name : str
        Fluid name (see REPRESENTATIVE_FLUIDS.keys())
    
    Returns
    -------
    fluid : EAFFluidModel
        The requested fluid model
    """
    if name not in REPRESENTATIVE_FLUIDS:
        available = list(REPRESENTATIVE_FLUIDS.keys())
        raise ValueError(f"Unknown fluid '{name}'. Available: {available}")
    return REPRESENTATIVE_FLUIDS[name]


def create_fluid_range(
    n_fluids: int = 5,
    base_viscosity: float = 0.030,
    min_loading: float = 0.0,
    max_loading: float = 1.0
) -> List[EAFFluidModel]:
    """
    Create a range of fluids spanning Newtonian to strongly viscoelastic.
    
    Parameters
    ----------
    n_fluids : int
        Number of fluids to create
    base_viscosity : float
        Common base viscosity [Pa·s]
    min_loading : float
        Minimum polymer loading
    max_loading : float
        Maximum polymer loading
    
    Returns
    -------
    fluids : list of EAFFluidModel
        List of fluid models
    """
    loadings = np.linspace(min_loading, max_loading, n_fluids)
    return [
        EAFFluidModel(
            base_viscosity=base_viscosity,
            polymer_loading=load,
            fluid_class='parametric'
        )
        for load in loadings
    ]


# =============================================================================
# Industrial Operating Conditions
# =============================================================================

@dataclass
class IndustrialConditions:
    """
    Representative industrial operating conditions for slipper bearings.
    
    Based on mobile hydraulic systems (excavators, wheel loaders, etc.)
    operating at 150-350 bar and 1000-3500 rpm.
    """
    
    # Pressure range
    p_min: float = 150e5   # Pa (150 bar)
    p_max: float = 350e5   # Pa (350 bar)
    p_typical: float = 250e5  # Pa (250 bar)
    
    # Speed range
    rpm_min: float = 1000
    rpm_max: float = 3500
    rpm_typical: float = 2000
    
    # Temperature range
    T_min: float = 30.0   # °C
    T_max: float = 80.0   # °C
    T_typical: float = 50.0  # °C
    
    # Geometry (typical 9-piston pump)
    R_pcd: float = 0.040   # Pitch circle diameter [m]
    slipper_R_in: float = 0.008   # Inner radius [m]
    slipper_R_out: float = 0.020  # Outer radius [m]
    h_0_typical: float = 5e-6    # Typical clearance [m]
    
    # Failure thresholds
    T_failure: float = 110.0   # °C (elastomer degradation)
    h_min_failure: float = 3e-6  # m (wear initiation)
    
    def tangential_velocity(self, rpm: Optional[float] = None) -> float:
        """
        Compute typical tangential velocity at slipper.
        
        Parameters
        ----------
        rpm : float, optional
            Shaft speed. Defaults to typical.
        
        Returns
        -------
        V_T : float
            Tangential velocity [m/s]
        """
        if rpm is None:
            rpm = self.rpm_typical
        omega = rpm * 2 * np.pi / 60  # rad/s
        return omega * self.R_pcd
    
    def squeeze_velocity_range(self) -> Tuple[float, float]:
        """
        Estimate typical squeeze velocity range.
        
        Returns
        -------
        h_dot_min, h_dot_max : float
            Squeeze velocity range [m/s]
        """
        # Squeeze is driven by pressure pulsation at kidney port transition
        # Typical: h_dot ~ 0.1-2 mm/s
        return (-2e-3, -0.1e-3)
    
    def shear_rate_range(self) -> Tuple[float, float]:
        """
        Compute typical shear rate range.
        
        Returns
        -------
        gdot_min, gdot_max : float
            Shear rate range [s⁻¹]
        """
        V_T_min = self.tangential_velocity(self.rpm_min)
        V_T_max = self.tangential_velocity(self.rpm_max)
        
        # Shear rate ~ V_T / h
        gdot_min = V_T_min / self.h_0_typical  # Lower bound
        gdot_max = V_T_max / self.h_min_failure  # Upper bound (thin film)
        
        return (gdot_min, gdot_max)


# Default industrial conditions
INDUSTRIAL_CONDITIONS = IndustrialConditions()


# =============================================================================
# Regime Map Generation
# =============================================================================

def generate_regime_map(
    De_sq_range: Tuple[float, float] = (1e-4, 1.0),
    Wi_range: Tuple[float, float] = (1, 1e5),
    n_points: int = 50
) -> Dict[str, np.ndarray]:
    """
    Generate a regime map over De_sq - Wi space.
    
    Parameters
    ----------
    De_sq_range : tuple
        (min, max) De_sq values
    Wi_range : tuple
        (min, max) Wi values
    n_points : int
        Number of points per dimension
    
    Returns
    -------
    map_data : dict
        Dictionary with:
        - 'De_sq': 2D array of De_sq values
        - 'Wi': 2D array of Wi values
        - 'regime': 2D array of regime codes (0-4)
        - 've_correction': 2D array of expected VE corrections
    """
    De_sq = np.logspace(np.log10(De_sq_range[0]), np.log10(De_sq_range[1]), n_points)
    Wi = np.logspace(np.log10(Wi_range[0]), np.log10(Wi_range[1]), n_points)
    
    DE_SQ, WI = np.meshgrid(De_sq, Wi)
    
    regime = np.zeros_like(DE_SQ, dtype=int)
    ve_correction = np.zeros_like(DE_SQ)
    
    for i in range(n_points):
        for j in range(n_points):
            result = classify_regime(DE_SQ[i, j], WI[i, j])
            regime[i, j] = list(FlowRegime).index(result.regime)
            ve_correction[i, j] = result.expected_VE_correction
    
    return {
        'De_sq': DE_SQ,
        'Wi': WI,
        'regime': regime,
        've_correction': ve_correction,
        'regime_names': [r.value for r in FlowRegime]
    }


def print_regime_summary():
    """Print a summary of flow regimes and model recommendations."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 FLOW REGIME CLASSIFICATION FOR EAF LUBRICATION               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  REGIME I: NEWTONIAN                                                         ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Conditions: De_sq < 0.001, Wi < 1                                           ║
║  Model: Constant viscosity η(T) or Newtonian Reynolds equation               ║
║  VE correction: ~0%                                                          ║
║  Representative: Hydransafe HFC-E 46, most commercial HFCs                   ║
║                                                                              ║
║  REGIME II: WEAK VISCOELASTIC                                                ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Conditions: De_sq < 0.01, Wi < 100                                          ║
║  Model: Generalised Newtonian (Carreau-Yasuda)                               ║
║  VE correction: < 1%                                                         ║
║  Representative: Enhanced HFCs with low-MW polymer VI improvers              ║
║                                                                              ║
║  REGIME III: MODERATE VISCOELASTIC                                           ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Conditions: 0.01 < De_sq < 0.1, Wi < 1000                                   ║
║  Model: Giesekus Reynolds equation (this thesis)                             ║
║  VE correction: 1-10%                                                        ║
║  Representative: PEO solutions, dilute PAM, specialty EAFs                   ║
║                                                                              ║
║  REGIME IV: STRONG VISCOELASTIC                                              ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Conditions: 0.1 < De_sq < 0.5                                               ║
║  Model: Giesekus RE (marginal), CFD validation recommended                   ║
║  VE correction: 10-30%                                                       ║
║  Representative: 2-5% PAM solutions, polymer-thickened drilling fluids       ║
║                                                                              ║
║  REGIME V: TRANSIENT DOMINATED                                               ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Conditions: De_sq > 0.5                                                     ║
║  Model: Full transient CFD required                                          ║
║  VE correction: > 30% (quasi-steady invalid)                                 ║
║  Representative: High-concentration polymer solutions                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Demonstrate the module
    print("=" * 70)
    print("PARAMETRIC EAF FLUID MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Print regime summary
    print_regime_summary()
    
    # Show representative fluids
    print("\n" + "=" * 70)
    print("REPRESENTATIVE FLUIDS")
    print("=" * 70)
    
    for name, fluid in REPRESENTATIVE_FLUIDS.items():
        print(f"\n{name}:")
        print(fluid)
    
    # Example classification
    print("\n" + "=" * 70)
    print("EXAMPLE: CLASSIFICATION AT OPERATING CONDITIONS")
    print("=" * 70)
    
    conditions = INDUSTRIAL_CONDITIONS
    h = 5e-6  # 5 μm
    h_dot = -0.5e-3  # -0.5 mm/s
    V_T = conditions.tangential_velocity(2000)  # 2000 rpm
    
    print(f"\nOperating conditions:")
    print(f"  Film thickness h = {h*1e6:.1f} μm")
    print(f"  Squeeze velocity ḣ = {h_dot*1000:.2f} mm/s")
    print(f"  Tangential velocity V_T = {V_T:.1f} m/s")
    
    for name, fluid in REPRESENTATIVE_FLUIDS.items():
        print(f"\n{name}:")
        classification = fluid.classify(h, h_dot, V_T)
        print(classification)
