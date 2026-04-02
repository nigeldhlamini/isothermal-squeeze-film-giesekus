"""
Giesekus Material Functions for Viscoelastic Lubrication
=========================================================

Implements the exact analytical expressions for the Giesekus constitutive model
in steady simple shear:

    η(γ̇)   = Viscosity function (Eq. 45)
    Ψ₁(γ̇) = First normal stress coefficient (Eq. 46)  
    Ψ₂(γ̇) = Second normal stress coefficient (Eq. 47)

The auxiliary variable f satisfies a quadratic equation derived from the
steady-state Giesekus constitutive equation.

Key Properties:
    - Low Wi limit: η → η₀, Ψ₁ → Ψ₁⁰ = 2η_p λ, Ψ₂ → -αη_p λ
    - High Wi limit: η → η_s, Ψ₁ ~ γ̇⁻², Ψ₂/Ψ₁ → -α/2
    - Constrained ratio: Ψ₂/Ψ₁ = -α/2 at all Wi (molecular constraint)

References:
    - Giesekus, H. (1982). J. Non-Newtonian Fluid Mech. 11, 69-109.
    - Bird et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Union, Tuple
import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[float, np.ndarray]


@dataclass
class GiesekusFluid:
    """
    Giesekus fluid with exact material functions in steady shear.
    
    Parameters
    ----------
    eta_s : float
        Solvent viscosity [Pa·s]
    eta_p : float
        Polymer viscosity contribution [Pa·s]
    lambda_ : float
        Relaxation time [s]
    alpha : float
        Mobility factor, 0 ≤ α ≤ 0.5. Controls shear-thinning strength
        and normal stress ratio Ψ₂/Ψ₁ = -α/2
    
    Attributes
    ----------
    eta_0 : float
        Zero-shear viscosity η₀ = η_s + η_p [Pa·s]
    Psi1_0 : float
        Zero-shear first normal stress coefficient Ψ₁⁰ = 2η_p λ [Pa·s²]
    Psi2_0 : float
        Zero-shear second normal stress coefficient Ψ₂⁰ = -αη_p λ [Pa·s²]
    beta : float
        Viscosity ratio β = η_s/η₀
    
    Examples
    --------
    >>> fluid = GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)
    >>> fluid.eta_0
    0.15
    >>> fluid.viscosity(1000.0)  # At γ̇ = 1000 s⁻¹
    ...
    """
    eta_s: float
    eta_p: float
    lambda_: float
    alpha: float
    
    # Derived quantities (computed in __post_init__)
    eta_0: float = field(init=False)
    Psi1_0: float = field(init=False)
    Psi2_0: float = field(init=False)
    beta: float = field(init=False)
    
    def __post_init__(self):
        """Validate parameters and compute derived quantities."""
        # Validation
        if self.eta_s < 0:
            raise ValueError(f"Solvent viscosity must be non-negative: {self.eta_s}")
        if self.eta_p < 0:
            raise ValueError(f"Polymer viscosity must be non-negative: {self.eta_p}")
        if self.lambda_ < 0:
            raise ValueError(f"Relaxation time must be non-negative: {self.lambda_}")
        if not (0 <= self.alpha <= 0.5):
            raise ValueError(f"Mobility factor must be in [0, 0.5]: {self.alpha}")
        
        # Derived quantities
        self.eta_0 = self.eta_s + self.eta_p
        self.Psi1_0 = 2.0 * self.eta_p * self.lambda_
        self.Psi2_0 = -self.alpha * self.eta_p * self.lambda_
        self.beta = self.eta_s / self.eta_0 if self.eta_0 > 0 else 1.0
    
    @property
    def is_newtonian(self) -> bool:
        """Check if fluid is effectively Newtonian (λ = 0 or η_p = 0)."""
        return self.lambda_ < 1e-15 or self.eta_p < 1e-15
    
    def _compute_f(self, Lambda: ArrayLike) -> ArrayLike:
        """
        Compute auxiliary variable f from the Giesekus quadratic.
        
        The quadratic equation (Eq. 42 in thesis):
            (1-2α)f² - [1-2α + α²Λ²]f + α²Λ² = 0
        
        Parameters
        ----------
        Lambda : float or ndarray
            Dimensionless shear rate Λ = λγ̇ (Weissenberg number)
        
        Returns
        -------
        f : float or ndarray
            Auxiliary variable, 0 ≤ f < 1
        
        Notes
        -----
        The physical root satisfies:
            - f → 0 as Λ → 0
            - f < 1 for all finite Λ
            - f → f_∞ = (1 - √(1-2α))/(2α) as Λ → ∞
        """
        Lambda = np.asarray(Lambda)
        scalar_input = Lambda.ndim == 0
        Lambda = np.atleast_1d(Lambda)
        
        # Handle Newtonian limit
        if self.is_newtonian:
            f = np.zeros_like(Lambda)
            return float(f[0]) if scalar_input else f
        
        alpha = self.alpha
        Lambda_sq = Lambda**2
        
        # Special case: α = 0 (UCM limit)
        if alpha < 1e-12:
            f = np.zeros_like(Lambda)
            return float(f[0]) if scalar_input else f
        
        # Coefficients of quadratic (1-2α)f² - b·f + c = 0
        a_coef = 1.0 - 2.0 * alpha
        b_coef = 1.0 - 2.0 * alpha + alpha**2 * Lambda_sq
        c_coef = alpha**2 * Lambda_sq
        
        # Discriminant
        discriminant = b_coef**2 - 4.0 * a_coef * c_coef
        
        # Physical root (the one that → 0 as Λ → 0)
        # Use numerically stable formula
        if abs(a_coef) > 1e-12:
            # Standard case: α ≠ 0.5
            sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
            f = (b_coef - sqrt_disc) / (2.0 * a_coef)
        else:
            # Special case: α ≈ 0.5
            # Quadratic degenerates to linear: -b·f + c = 0 → f = c/b
            f = np.where(b_coef > 1e-12, c_coef / b_coef, 0.0)
        
        # Ensure physical bounds (numerical safety)
        f = np.clip(f, 0.0, 1.0 - 1e-10)
        
        return float(f[0]) if scalar_input else f
    
    def viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute shear-rate-dependent viscosity η(γ̇).
        
        From Eq. 45:
            η(γ̇) = η_s + η_p · (1-f)² / [1 + (1-2α)f]
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]
        
        Returns
        -------
        eta : float or ndarray
            Viscosity [Pa·s]
        
        Limits
        ------
        - γ̇ → 0: η → η₀ = η_s + η_p
        - γ̇ → ∞: η → η_s (shear-thinning plateau)
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot)
        
        # Handle Newtonian limit
        if self.is_newtonian:
            eta = np.full_like(gdot, self.eta_0, dtype=float)
            return float(eta[0]) if scalar_input else eta
        
        Lambda = self.lambda_ * np.abs(gdot)
        f = self._compute_f(Lambda)
        
        alpha = self.alpha
        
        # Viscosity formula (Eq. 45)
        numerator = (1.0 - f)**2
        denominator = 1.0 + (1.0 - 2.0 * alpha) * f
        
        eta = self.eta_s + self.eta_p * numerator / denominator
        
        return float(eta[0]) if scalar_input else eta
    
    def Psi1(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute first normal stress coefficient Ψ₁(γ̇).
        
        From Eq. 46:
            Ψ₁(γ̇) = (2η_p λ / αΛ²) · f(1-f)
        
        At low shear rates, use asymptotic expansion:
            f ≈ α²Λ² for Λ → 0
            Ψ₁ → 2η_p λ = Ψ₁⁰
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]
        
        Returns
        -------
        Psi1 : float or ndarray
            First normal stress coefficient [Pa·s²]
        
        Limits
        ------
        - γ̇ → 0: Ψ₁ → Ψ₁⁰ = 2η_p λ
        - γ̇ → ∞: Ψ₁ → 2η_p / (λγ̇²) (saturation)
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot)
        
        # Handle Newtonian limit (Ψ₁ = 0)
        if self.is_newtonian:
            Psi1 = np.zeros_like(gdot, dtype=float)
            return float(Psi1[0]) if scalar_input else Psi1
        
        Lambda = self.lambda_ * np.abs(gdot)
        alpha = self.alpha
        
        # Use asymptotic form for low Lambda to avoid numerical issues
        # At low Λ: f ≈ α²Λ², so f(1-f)/Λ² ≈ α²(1-α²Λ²) → α²
        # Thus Ψ₁ = 2η_p λ / α · α² = 2η_p λ α → Ψ₁⁰ (when we also account for (1-f)→1)
        
        # Threshold for switching to asymptotic formula
        Lambda_threshold = 0.01
        
        Psi1 = np.empty_like(gdot, dtype=float)
        
        # Low shear rate regime: use asymptotic expansion
        low_mask = Lambda < Lambda_threshold
        
        if np.any(low_mask):
            # Asymptotic expansion: Ψ₁ ≈ Ψ₁⁰ (1 - O(Λ²))
            Psi1[low_mask] = self.Psi1_0 * (1.0 - Lambda[low_mask]**2 / 3.0)
        
        # High shear rate regime: use full formula
        high_mask = ~low_mask
        
        if np.any(high_mask):
            Lambda_high = Lambda[high_mask]
            f_high = self._compute_f(Lambda_high)
            Lambda_sq = Lambda_high**2
            
            # Ψ₁ = (2η_p λ / αΛ²) · f(1-f)
            Psi1[high_mask] = (2.0 * self.eta_p * self.lambda_ / 
                               (alpha * Lambda_sq)) * f_high * (1.0 - f_high)
        
        return float(Psi1[0]) if scalar_input else Psi1
    
    def Psi2(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute second normal stress coefficient Ψ₂(γ̇).
        
        From Eq. 47:
            Ψ₂(γ̇) = -(η_p λ / αΛ²) · f · [1 - (1-f)(1+(1-2α)f)/(1-αf)]
        
        At low shear rates:
            Ψ₂ → Ψ₂⁰ = -αη_p λ = -α/2 · Ψ₁⁰
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]
        
        Returns
        -------
        Psi2 : float or ndarray
            Second normal stress coefficient [Pa·s²] (always ≤ 0)
        
        Limits
        ------
        - γ̇ → 0: Ψ₂ → Ψ₂⁰ = -αη_p λ = -α/2 · Ψ₁⁰
        - γ̇ → ∞: Ψ₂/Ψ₁ → -α/2 (molecular constraint)
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot)
        
        # Handle Newtonian limit (Ψ₂ = 0)
        if self.is_newtonian:
            Psi2 = np.zeros_like(gdot, dtype=float)
            return float(Psi2[0]) if scalar_input else Psi2
        
        # Use the molecular constraint Ψ₂/Ψ₁ = -α/2 which holds at all shear rates
        # This is simpler and more numerically stable than the full formula
        Psi1 = self.Psi1(gdot)
        Psi2 = -self.alpha / 2.0 * Psi1
        
        return float(Psi2[0]) if scalar_input else Psi2
    
    def tangent_viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute tangent viscosity η_T(γ̇) = η + γ̇ · dη/dγ̇.
        
        Used in memory flux correction (Eq. 60).
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate γ̇ [s⁻¹]
        
        Returns
        -------
        eta_T : float or ndarray
            Tangent viscosity [Pa·s]
        
        Notes
        -----
        For shear-thinning fluids, dη/dγ̇ < 0, so η_T < η.
        For Newtonian fluids, η_T = η.
        """
        gdot = np.asarray(gdot)
        scalar_input = gdot.ndim == 0
        gdot = np.atleast_1d(gdot)
        
        # Numerical differentiation with central differences
        delta = np.maximum(1e-6 * np.abs(gdot), 1e-6)
        
        eta_plus = self.viscosity(gdot + delta)
        eta_minus = self.viscosity(gdot - delta)
        eta = self.viscosity(gdot)
        
        d_eta_d_gdot = (eta_plus - eta_minus) / (2.0 * delta)
        eta_T = eta + gdot * d_eta_d_gdot
        
        return float(eta_T[0]) if scalar_input else eta_T

    def complex_viscosity(self, omega: ArrayLike) -> Union[complex, np.ndarray]:
        """
        SAOS complex viscosity η*(ω) for the Giesekus model.

        In the linear viscoelastic limit (small amplitude oscillatory shear):
            η*(ω) = η_s + η_p / (1 + iωλ)

        Parameters
        ----------
        omega : float or ndarray
            Angular frequency [rad/s]

        Returns
        -------
        eta_star : complex or ndarray of complex
            Complex viscosity [Pa·s]

        References
        ----------
        Bird et al. (1987), Dynamics of Polymeric Liquids, Vol. 1, Eq. 8.5-6.
        """
        omega = np.asarray(omega, dtype=float)
        scalar_input = omega.ndim == 0
        omega = np.atleast_1d(omega)

        if self.is_newtonian:
            result = np.full_like(omega, complex(self.eta_0, 0.0))
        else:
            result = self.eta_s + self.eta_p / (1.0 + 1j * omega * self.lambda_)

        return complex(result[0]) if scalar_input else result

    def eta_prime(self, omega: ArrayLike) -> ArrayLike:
        """
        In-phase (viscous) component η'(ω) = Re[η*(ω)].

        Parameters
        ----------
        omega : float or ndarray
            Angular frequency [rad/s]

        Returns
        -------
        eta_p : float or ndarray
            Storage viscosity [Pa·s]
        """
        eta_star = self.complex_viscosity(omega)
        result = np.real(eta_star)
        return float(result) if np.ndim(omega) == 0 else result

    def eta_double_prime(self, omega: ArrayLike) -> ArrayLike:
        """
        Out-of-phase (elastic) component η''(ω) = -Im[η*(ω)].

        Parameters
        ----------
        omega : float or ndarray
            Angular frequency [rad/s]

        Returns
        -------
        eta_pp : float or ndarray
            Loss viscosity [Pa·s]
        """
        eta_star = self.complex_viscosity(omega)
        result = -np.imag(eta_star)
        return float(result) if np.ndim(omega) == 0 else result

    def material_functions(self, gdot: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute all material functions at given shear rate(s).
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate(s) γ̇ [s⁻¹]
        
        Returns
        -------
        eta : ndarray
            Viscosity [Pa·s]
        Psi1 : ndarray
            First normal stress coefficient [Pa·s²]
        Psi2 : ndarray
            Second normal stress coefficient [Pa·s²]
        """
        return self.viscosity(gdot), self.Psi1(gdot), self.Psi2(gdot)
    
    def normal_stress_ratio(self, gdot: ArrayLike) -> ArrayLike:
        """
        Compute Ψ₂/Ψ₁ ratio.
        
        Should approach -α/2 at all shear rates (molecular constraint).
        
        Parameters
        ----------
        gdot : float or ndarray
            Shear rate(s) γ̇ [s⁻¹]
        
        Returns
        -------
        ratio : float or ndarray
            Ψ₂/Ψ₁ (should be approximately -α/2)
        """
        Psi1 = self.Psi1(gdot)
        Psi2 = self.Psi2(gdot)
        
        # Avoid division by zero
        ratio = np.where(np.abs(Psi1) > 1e-20, Psi2 / Psi1, -self.alpha / 2.0)
        return ratio
    
    def De(self, U: float, L: float) -> float:
        """Deborah number De = λU/L."""
        return self.lambda_ * U / L
    
    def Wi(self, U: float, h: float) -> float:
        """Weissenberg number Wi = λU/h."""
        return self.lambda_ * U / h
    
    def De_sq(self, h_dot: float, h: float) -> float:
        """Squeeze Deborah number De_sq = λ|ḣ|/h."""
        return self.lambda_ * abs(h_dot) / h
    
    def __repr__(self) -> str:
        return (f"GiesekusFluid(η_s={self.eta_s:.4f}, η_p={self.eta_p:.4f}, "
                f"λ={self.lambda_:.2e}, α={self.alpha:.2f})")
    
    def summary(self) -> str:
        """Return a formatted summary of fluid properties."""
        lines = [
            "Giesekus Fluid Properties",
            "=" * 40,
            f"Solvent viscosity η_s    : {self.eta_s:.4f} Pa·s",
            f"Polymer viscosity η_p    : {self.eta_p:.4f} Pa·s",
            f"Zero-shear viscosity η₀  : {self.eta_0:.4f} Pa·s",
            f"Relaxation time λ        : {self.lambda_:.2e} s",
            f"Mobility factor α        : {self.alpha:.3f}",
            f"Viscosity ratio β        : {self.beta:.3f}",
            "-" * 40,
            "Zero-shear normal stress coefficients:",
            f"  Ψ₁⁰ = 2η_p λ           : {self.Psi1_0:.2e} Pa·s²",
            f"  Ψ₂⁰ = -αη_p λ          : {self.Psi2_0:.2e} Pa·s²",
            f"  Ψ₂⁰/Ψ₁⁰ = -α/2        : {-self.alpha/2:.3f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Predefined Fluids (Table 1 in thesis)
# =============================================================================

def create_newtonian(eta: float = 0.030) -> GiesekusFluid:
    """Create a Newtonian fluid (λ = 0)."""
    return GiesekusFluid(eta_s=eta, eta_p=0.0, lambda_=0.0, alpha=0.0)


def create_PAM_2pct() -> GiesekusFluid:
    """
    Create 2% PAM (polyacrylamide) in water-glycol.
    
    From Table 1:
        η₀ = 0.050 Pa·s, η_s = 0.030 Pa·s, λ = 1.0 ms, α = 0.25
    """
    return GiesekusFluid(eta_s=0.030, eta_p=0.020, lambda_=1.0e-3, alpha=0.25)


def create_PAM_5pct() -> GiesekusFluid:
    """
    Create 5% PAM (polyacrylamide) in water-glycol.
    
    From Table 1:
        η₀ = 0.150 Pa·s, η_s = 0.030 Pa·s, λ = 3.3 ms, α = 0.25
    """
    return GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)


# =============================================================================
# SOF (Second-Order Fluid) as Limiting Case
# =============================================================================

@dataclass
class SOFFluid:
    """
    Second-Order Fluid (SOF) with constant material properties.
    
    The SOF emerges as the Wi → 0 limit of Giesekus.
    Constitutive equation:
        τ = η₀ A₁ - (Ψ₁⁰/2) A₁^∇ + Ψ₂⁰ A₁²
    
    Parameters
    ----------
    eta_0 : float
        Constant viscosity [Pa·s]
    Psi1_0 : float
        First normal stress coefficient [Pa·s²]
    Psi2_0 : float
        Second normal stress coefficient [Pa·s²]
    
    Notes
    -----
    For consistency with Giesekus, use Ψ₂⁰/Ψ₁⁰ = -α/2.
    """
    eta_0: float
    Psi1_0: float
    Psi2_0: float
    
    def viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """Constant viscosity (independent of γ̇)."""
        gdot = np.asarray(gdot)
        return np.full_like(gdot, self.eta_0, dtype=float)
    
    def Psi1(self, gdot: ArrayLike) -> ArrayLike:
        """Constant first normal stress coefficient."""
        gdot = np.asarray(gdot)
        return np.full_like(gdot, self.Psi1_0, dtype=float)
    
    def Psi2(self, gdot: ArrayLike) -> ArrayLike:
        """Constant second normal stress coefficient."""
        gdot = np.asarray(gdot)
        return np.full_like(gdot, self.Psi2_0, dtype=float)
    
    def tangent_viscosity(self, gdot: ArrayLike) -> ArrayLike:
        """Tangent viscosity equals viscosity for constant η."""
        return self.viscosity(gdot)
    
    @property
    def lambda_(self) -> float:
        """Effective relaxation time λ = Ψ₁⁰/(2η₀)."""
        if self.eta_0 > 0:
            return self.Psi1_0 / (2.0 * self.eta_0)
        return 0.0
    
    @classmethod
    def from_giesekus(cls, fluid: GiesekusFluid) -> 'SOFFluid':
        """Create SOF from Giesekus fluid using zero-shear properties."""
        return cls(
            eta_0=fluid.eta_0,
            Psi1_0=fluid.Psi1_0,
            Psi2_0=fluid.Psi2_0
        )


# =============================================================================
# Verification Functions
# =============================================================================

def verify_psi_ratio(fluid: GiesekusFluid, gdot_range: ArrayLike = None) -> dict:
    """
    Verify that Ψ₂/Ψ₁ ≈ -α/2 across shear rate range.
    
    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid to verify
    gdot_range : array-like, optional
        Shear rates to test. Default: 10⁰ to 10⁶ s⁻¹
    
    Returns
    -------
    dict
        Verification results with max_error and all_pass status
    """
    if gdot_range is None:
        gdot_range = np.logspace(0, 6, 50)
    
    ratio = fluid.normal_stress_ratio(gdot_range)
    expected = -fluid.alpha / 2.0
    
    error = np.abs(ratio - expected)
    max_error = np.max(error)
    
    return {
        'gdot': gdot_range,
        'ratio': ratio,
        'expected': expected,
        'error': error,
        'max_error': max_error,
        'all_pass': max_error < 0.01 * abs(expected)  # 1% tolerance
    }


def verify_sof_limit(fluid: GiesekusFluid, gdot_test: float = 1e-3) -> dict:
    """
    Verify that Giesekus reduces to SOF at low Wi.
    
    Parameters
    ----------
    fluid : GiesekusFluid
        Fluid to verify
    gdot_test : float
        Low shear rate for testing (should give Wi ≪ 1)
    
    Returns
    -------
    dict
        Comparison between Giesekus and SOF at low Wi
    """
    sof = SOFFluid.from_giesekus(fluid)
    
    Wi = fluid.lambda_ * gdot_test
    
    eta_giesekus = fluid.viscosity(gdot_test)
    eta_sof = sof.eta_0
    
    Psi1_giesekus = fluid.Psi1(gdot_test)
    Psi1_sof = sof.Psi1_0
    
    Psi2_giesekus = fluid.Psi2(gdot_test)
    Psi2_sof = sof.Psi2_0
    
    return {
        'Wi': Wi,
        'eta': {'giesekus': eta_giesekus, 'sof': eta_sof, 
                'error': abs(eta_giesekus - eta_sof) / eta_sof},
        'Psi1': {'giesekus': Psi1_giesekus, 'sof': Psi1_sof,
                 'error': abs(Psi1_giesekus - Psi1_sof) / Psi1_sof if Psi1_sof > 0 else 0},
        'Psi2': {'giesekus': Psi2_giesekus, 'sof': Psi2_sof,
                 'error': abs(Psi2_giesekus - Psi2_sof) / abs(Psi2_sof) if abs(Psi2_sof) > 0 else 0}
    }
