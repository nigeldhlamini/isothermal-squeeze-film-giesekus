"""
Reynolds Equation Assembly
==========================

Builds the sparse matrix system for the viscoelastic Reynolds equation.

The Reynolds equation in operator form (Eq. 66-71):
    L_GN[p] + L_mem[p] + L_α + L_N1[p] + L_hoop = S

This module constructs the discretised form:
    A · p = b

where A is a sparse matrix and b is the right-hand side vector.

References:
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import spsolve

from .rheology import GiesekusFluid
from .geometry import SlipperGeometry, OperatingConditions
from .mesh import PolarMesh


# =============================================================================
# Operator Discretisation
# =============================================================================

def build_diffusion_operator(
    mesh: PolarMesh,
    D_coeff: np.ndarray
) -> csr_matrix:
    """
    Build the generalised diffusion operator in polar coordinates.
    
    Discretises:
        L[p] = (1/r)∂/∂r(r·D·∂p/∂r) + (1/r²)∂/∂θ(D·∂p/∂θ)
    
    This is the standard form for Reynolds equation with D = h³/(12η).
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    D_coeff : ndarray
        Diffusion coefficient field, shape (Nr, Ntheta)
    
    Returns
    -------
    L : csr_matrix
        Sparse diffusion operator, shape (N, N)
    
    Notes
    -----
    Uses second-order central differences with harmonic averaging
    of diffusion coefficients at cell faces.
    """
    Nr, Ntheta = mesh.Nr, mesh.Ntheta
    N = mesh.N
    dr, dtheta = mesh.dr, mesh.dtheta
    r = mesh.R  # 2D array of radial positions
    
    # Build matrix in LIL format for efficient construction
    L = lil_matrix((N, N), dtype=float)
    
    for i in range(Nr):
        for j in range(Ntheta):
            k = mesh._index(i, j)
            r_ij = r[i, j]
            D_ij = D_coeff[i, j]
            
            # Neighbour indices (with periodic BC in theta)
            j_plus = (j + 1) % Ntheta
            j_minus = (j - 1) % Ntheta
            
            # === Radial diffusion: (1/r)∂/∂r(r·D·∂p/∂r) ===
            if i > 0 and i < Nr - 1:
                # Interior point
                D_ip = D_coeff[i+1, j]
                D_im = D_coeff[i-1, j]
                r_ip_half = r_ij + dr/2  # r at i+1/2
                r_im_half = r_ij - dr/2  # r at i-1/2
                
                # Harmonic average at faces
                D_ip_half = 2 * D_ij * D_ip / (D_ij + D_ip + 1e-30)
                D_im_half = 2 * D_ij * D_im / (D_ij + D_im + 1e-30)
                
                # Coefficients
                coeff_ip = (r_ip_half * D_ip_half) / (r_ij * dr**2)
                coeff_im = (r_im_half * D_im_half) / (r_ij * dr**2)
                coeff_center_r = -(coeff_ip + coeff_im)
                
                L[k, mesh._index(i+1, j)] += coeff_ip
                L[k, mesh._index(i-1, j)] += coeff_im
                L[k, k] += coeff_center_r
            
            # === Azimuthal diffusion: (1/r²)∂/∂θ(D·∂p/∂θ) ===
            D_jp = D_coeff[i, j_plus]
            D_jm = D_coeff[i, j_minus]
            
            # Harmonic average at faces
            D_jp_half = 2 * D_ij * D_jp / (D_ij + D_jp + 1e-30)
            D_jm_half = 2 * D_ij * D_jm / (D_ij + D_jm + 1e-30)
            
            # Coefficients
            coeff_jp = D_jp_half / (r_ij**2 * dtheta**2)
            coeff_jm = D_jm_half / (r_ij**2 * dtheta**2)
            coeff_center_theta = -(coeff_jp + coeff_jm)
            
            L[k, mesh._index(i, j_plus)] += coeff_jp
            L[k, mesh._index(i, j_minus)] += coeff_jm
            L[k, k] += coeff_center_theta
    
    return L.tocsr()


def build_GNF_operator(
    mesh: PolarMesh,
    h: np.ndarray,
    eta_bar: np.ndarray
) -> csr_matrix:
    """
    Build the Generalised Newtonian Fluid operator.
    
    L_GN[p] = (1/r)∂/∂r(r·h³/(12η̄)·∂p/∂r) + (1/r²)∂/∂θ(h³/(12η̄)·∂p/∂θ)
    
    This is Eq. 66 from the thesis.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height field [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    
    Returns
    -------
    L_GN : csr_matrix
        GNF diffusion operator
    """
    D_coeff = h**3 / (12 * eta_bar)
    return build_diffusion_operator(mesh, D_coeff)


def build_memory_operator(
    mesh: PolarMesh,
    h: np.ndarray,
    eta_bar: np.ndarray,
    eta_T: np.ndarray,
    h_dot: float,
    lambda_: float
) -> csr_matrix:
    """
    Build the squeeze-film memory operator.
    
    L_mem[p] = -(λḣ/16)(1/r)∂/∂r(r·h²η_T/η̄²·∂p/∂r)
    
    This is Eq. 67 from the thesis.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    eta_T : ndarray
        Tangent viscosity [Pa·s]
    h_dot : float
        Squeeze velocity [m/s]
    lambda_ : float
        Relaxation time [s]
    
    Returns
    -------
    L_mem : csr_matrix
        Memory operator
    """
    # Prefactor from Eq. 67: L_mem = -(λḣ/16) × diffusion_operator
    # For closing gap (ḣ < 0): prefactor = -λ×(-)/16 = +λ|ḣ|/16
    # This ADDS to the diffusion, which is WRONG for load enhancement.
    # 
    # Actually, re-reading carefully: the operator form and flux form must be consistent.
    # The Reynolds eq is: ∇·Q = S, so L[p] = ∇·Q where Q = -D∇p + other terms
    # 
    # From the flux: Q_r^mem = -λh²ḣ/(16η̄²) × η_T × ∂p/∂r
    # The divergence of this flux gives L_mem.
    # 
    # The thesis Eq. 67 shows L_mem = -(λḣ/16) × ∇·(D_mem ∇p)
    # This is the divergence of Q_mem = +(λḣ/16) × D_mem × ∇p
    # 
    # Comparing: Q_r^mem = -λh²ḣη_T/(16η̄²) × ∂p/∂r
    #            Q_r from L_mem = -(λḣ/16) × (h²η_T/η̄²) × ∂p/∂r  (with D_mem = h²η_T/η̄²)
    # These match! So the sign in Eq. 67 is correct as written.
    #
    prefactor = -lambda_ * h_dot / 16
    
    # Diffusion coefficient for memory term
    D_mem = h**2 * eta_T / eta_bar**2
    
    # Build diffusion operator and scale
    L_base = build_diffusion_operator(mesh, D_mem)
    
    return prefactor * L_base


def compute_source_vector(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions
) -> np.ndarray:
    """
    Compute the source term (RHS) vector.
    
    S = ḣ - (V_Tr/2)∂h/∂r - (V_Tθ/(2r))∂h/∂θ
    
    This is Eq. 71 from the thesis.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    
    Returns
    -------
    S : ndarray
        Source vector, shape (N,)
    """
    R, THETA = mesh.R, mesh.THETA
    
    # Gap height and derivatives
    h = geometry.gap_height(R, THETA)
    _, dh_dr, dh_dtheta = geometry.gap_height_derivatives(R, THETA)
    
    # Velocity components
    V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
    omega_s = conditions.omega_s
    
    # Squeeze term
    S = conditions.h_dot * np.ones_like(h)
    
    # Wedge terms
    V_theta_total = V_Ttheta + omega_s * R
    S = S - (V_Tr / 2) * dh_dr - (V_theta_total / (2 * R)) * dh_dtheta
    
    return mesh.to_1d(S)


# =============================================================================
# Normal Stress Contributions (explicit, added to RHS)
# =============================================================================

def compute_N1_contribution(
    mesh: PolarMesh,
    h: np.ndarray,
    Gamma_sq: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the N₁ gradient contribution to the source term.
    
    From Eq. 69, this contributes as a pseudo-source:
        S_N1 = (1-α/2)/6 · (1/r)∂/∂r(r·h²/η̄ · ∂(hΨ̄₁Γ̄²)/∂r)
    
    Since this involves ∂(hΨ̄₁Γ̄²)/∂r which depends on pressure,
    we treat it explicitly (lagged) for simplicity.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    Gamma_sq : ndarray
        Γ̄² [s⁻²]
    Psi1_bar : ndarray
        Ψ̄₁ [Pa·s²]
    eta_bar : ndarray
        η̄ [Pa·s]
    alpha : float
        Mobility parameter
    
    Returns
    -------
    S_N1 : ndarray
        N₁ source contribution, shape (N,)
    """
    # Compute h·Ψ₁·Γ²
    h_Psi_Gamma = h * Psi1_bar * Gamma_sq

    # Compute gradient of h·Ψ₁·Γ² (both components)
    d_hPG_dr, d_hPG_dtheta = mesh.compute_field_gradient(h_Psi_Gamma)

    # Coefficient field
    coeff = (1 - alpha/2) / 6 * h**2 / eta_bar

    # Radial divergence: (1/r)∂/∂r(r·coeff·d_hPG_dr)
    flux_r = coeff * d_hPG_dr
    r_flux = mesh.R * flux_r
    d_rflux_dr, _ = mesh.compute_field_gradient(r_flux)
    div_N1_r = d_rflux_dr / mesh.R

    # Azimuthal divergence: (1/r²)∂/∂θ(coeff·d_hPG_dθ)
    # This term is zero for axisymmetric (no-tilt) cases but contributes
    # when tilt creates azimuthal variation in h, Ψ₁, and Γ².
    flux_theta = coeff * d_hPG_dtheta
    _, d_flux_theta_dtheta = mesh.compute_field_gradient(flux_theta)
    div_N1_theta = d_flux_theta_dtheta / mesh.R**2

    div_N1 = div_N1_r + div_N1_theta

    return mesh.to_1d(div_N1)


def compute_hoop_contribution(
    mesh: PolarMesh,
    h: np.ndarray,
    Gamma_sq: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the hoop stress contribution to the source term.

    From Eq. 70, the hoop stress (tau_rr - tau_thetatheta) creates an
    effective radial body force.  The full divergence includes both
    radial and azimuthal components:

        S_hoop = -(1+alpha/2)/12 * [ (1/r) d/dr(h^3 Psi1 Gamma^2 / eta)
                                    + (1/r^2) d/dtheta(h^3 Psi1 Gamma^2 / eta) ]

    The azimuthal term is non-zero for tilted geometries where h, Psi1, and
    Gamma^2 vary with theta.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    Gamma_sq : ndarray
        Gamma_bar^2 [s^-2]
    Psi1_bar : ndarray
        Psi_1_bar [Pa·s^2]
    eta_bar : ndarray
        eta_bar [Pa·s]
    alpha : float
        Mobility parameter

    Returns
    -------
    S_hoop : ndarray
        Hoop stress source contribution, shape (N,)
    """
    # The quantity inside the derivative
    inner = h**3 * Psi1_bar * Gamma_sq / eta_bar

    # Compute full gradient
    d_inner_dr, d_inner_dtheta = mesh.compute_field_gradient(inner)

    prefactor = -(1 + alpha/2) / 12

    # Radial: -(1+alpha/2)/12 · (1/r)d/dr(inner)
    S_hoop_r = prefactor * d_inner_dr / mesh.R

    # Azimuthal: -(1+alpha/2)/12 · (1/r^2)d/dtheta(inner)
    S_hoop_theta = prefactor * d_inner_dtheta / mesh.R**2

    S_hoop = S_hoop_r + S_hoop_theta

    return mesh.to_1d(S_hoop)


def compute_alpha_contribution(
    mesh: PolarMesh,
    h: np.ndarray,
    gdot_bar: np.ndarray,
    Psi1_bar: np.ndarray,
    eta_bar: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute the Giesekus coupling (L_alpha) contribution to the source term.

    From Eq. 68 (thesis Chapter 4):
        S_alpha = (alpha/4) · (1/r) ∂/∂r(r · h²/eta_bar · Psi1 · gdot² · ∂gdot/∂r)

    This term arises from the nonlinear tau·tau coupling in the Giesekus
    constitutive equation.  It is typically < 5% of the total viscoelastic
    correction and was previously computed only in post-processing flux
    decomposition.  It is now included as an explicit (lagged) RHS term
    for consistency with the full Eq. 4.31.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    h : ndarray
        Gap height [m]
    gdot_bar : ndarray
        Gap-averaged shear rate [s^-1]
    Psi1_bar : ndarray
        First normal stress coefficient [Pa·s^2]
    eta_bar : ndarray
        Gap-averaged viscosity [Pa·s]
    alpha : float
        Giesekus mobility parameter

    Returns
    -------
    S_alpha : ndarray
        Alpha coupling source contribution, shape (N,)
    """
    # Compute radial gradient of shear rate
    d_gdot_dr, d_gdot_dtheta = mesh.compute_field_gradient(gdot_bar)

    # Inner flux: (alpha/4) · (h^2 / eta_bar) · Psi1 · gdot^2 · d_gdot_dr
    coeff = (alpha / 4) * h**2 / eta_bar * Psi1_bar * gdot_bar**2

    # Radial component
    flux_r = coeff * d_gdot_dr
    r_flux = mesh.R * flux_r

    # Divergence: (1/r) d/dr (r · flux_r)
    d_rflux_dr, _ = mesh.compute_field_gradient(r_flux)
    div_alpha_r = d_rflux_dr / mesh.R

    # Azimuthal component: (alpha/4) · (h^2/eta_bar) · Psi1 · gdot^2 · (1/r^2) d_gdot_dtheta
    #   divergence: (1/r^2) d/dtheta(coeff * d_gdot_dtheta)
    flux_theta = coeff * d_gdot_dtheta
    _, d_flux_theta_dtheta = mesh.compute_field_gradient(flux_theta)
    div_alpha_theta = d_flux_theta_dtheta / mesh.R**2

    S_alpha = div_alpha_r + div_alpha_theta

    return mesh.to_1d(S_alpha)


# =============================================================================
# Complete System Assembly
# =============================================================================

@dataclass
class AssembledSystem:
    """
    Container for the assembled linear system.
    
    Attributes
    ----------
    A : csr_matrix
        System matrix
    b : ndarray
        Right-hand side vector
    A_GNF : csr_matrix
        GNF operator (for diagnostics)
    A_mem : csr_matrix, optional
        Memory operator
    """
    A: csr_matrix
    b: np.ndarray
    A_GNF: csr_matrix
    A_mem: Optional[csr_matrix] = None
    
    # Metadata
    info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}


def assemble_newtonian_system(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    eta: float
) -> AssembledSystem:
    """
    Assemble the Newtonian Reynolds equation system.
    
    This is the simplest case with constant viscosity and no
    viscoelastic effects. Used for validation.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    eta : float
        Constant viscosity [Pa·s]
    
    Returns
    -------
    system : AssembledSystem
        Assembled linear system
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    eta_bar = eta * np.ones_like(h)
    
    # Build GNF operator
    A_GNF = build_GNF_operator(mesh, h, eta_bar)
    
    # Compute source
    b = compute_source_vector(mesh, geometry, conditions)
    
    # Apply boundary conditions
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    
    A, b = mesh.apply_dirichlet_bc(A_GNF, b, p_inner, p_outer)
    
    return AssembledSystem(
        A=A, b=b, A_GNF=A_GNF,
        info={'type': 'newtonian', 'eta': eta}
    )


def assemble_GNF_system(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    eta_bar: np.ndarray
) -> AssembledSystem:
    """
    Assemble the Generalised Newtonian Fluid system.
    
    Uses shear-rate-dependent viscosity but no elastic effects.
    
    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Fluid model (for viscosity)
    eta_bar : ndarray
        Current viscosity field [Pa·s]
    
    Returns
    -------
    system : AssembledSystem
        Assembled linear system
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    
    # Build GNF operator with current viscosity
    A_GNF = build_GNF_operator(mesh, h, eta_bar)
    
    # Compute source
    b = compute_source_vector(mesh, geometry, conditions)
    
    # Apply boundary conditions
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    
    A, b = mesh.apply_dirichlet_bc(A_GNF, b, p_inner, p_outer)
    
    return AssembledSystem(
        A=A, b=b, A_GNF=A_GNF,
        info={'type': 'GNF'}
    )


def assemble_viscoelastic_system(
    mesh: PolarMesh,
    geometry: SlipperGeometry,
    conditions: OperatingConditions,
    fluid: GiesekusFluid,
    eta_bar: np.ndarray,
    gdot_bar: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    include_memory: bool = True,
    include_normal_stress: bool = True,
    ns_scale: float = 1.0
) -> AssembledSystem:
    """
    Assemble the full viscoelastic Reynolds equation system.

    Includes memory operator in LHS and normal stress terms in RHS.

    Parameters
    ----------
    mesh : PolarMesh
        Computational mesh
    geometry : SlipperGeometry
        Bearing geometry
    conditions : OperatingConditions
        Operating parameters
    fluid : GiesekusFluid
        Fluid model
    eta_bar : ndarray
        Current viscosity field [Pa·s]
    gdot_bar : ndarray
        Current shear rate field [s⁻¹]
    pressure : ndarray, optional
        Current pressure field (for explicit N₁, hoop terms)
    include_memory : bool
        Include memory operator in LHS
    include_normal_stress : bool
        Include N₁ and hoop stress in RHS
    ns_scale : float
        Scaling factor for normal stress RHS contributions, in [0, 1].
        Used for gradual ramp-up during Picard iteration to prevent
        divergence from the quadratic |∇p|² coupling.  Default: 1.0.

    Returns
    -------
    system : AssembledSystem
        Assembled linear system
    """
    R, THETA = mesh.R, mesh.THETA
    h = geometry.gap_height(R, THETA)
    
    # Build GNF operator
    A_GNF = build_GNF_operator(mesh, h, eta_bar)
    A = A_GNF.copy()
    A_mem = None
    
    # Add memory operator if requested
    if include_memory and not fluid.is_newtonian:
        eta_T = fluid.tangent_viscosity(gdot_bar)
        A_mem = build_memory_operator(
            mesh, h, eta_bar, eta_T, 
            conditions.h_dot, fluid.lambda_
        )
        A = A + A_mem
    
    # Compute base source term
    b = compute_source_vector(mesh, geometry, conditions)
    
    # Add normal stress and alpha contributions if requested and pressure available
    if (include_normal_stress and pressure is not None
            and not fluid.is_newtonian and ns_scale > 1e-12):
        # Compute Γ² from current pressure
        dp_dr, dp_dtheta = mesh.compute_field_gradient(pressure)
        V_Tr, V_Ttheta = conditions.radial_tangential_velocity(THETA)
        V_T = np.sqrt(V_Tr**2 + V_Ttheta**2)

        from .fluxes import compute_Gamma_squared
        Gamma_sq = compute_Gamma_squared(h, V_T, dp_dr, dp_dtheta, R, eta_bar)

        # Clamp Gamma_sq to prevent unbounded growth during early
        # iterations when the pressure field is not yet converged.
        # The physical Couette contribution V_T²/h² sets a sensible
        # scale; allow the Poiseuille part to be at most 100× that.
        Gamma_sq_couette = V_T**2 / h**2
        Gamma_sq_max = np.maximum(Gamma_sq_couette, 1e-10) * 100.0
        Gamma_sq = np.minimum(Gamma_sq, Gamma_sq_max)

        Psi1_bar = fluid.Psi1(gdot_bar)
        alpha = fluid.alpha

        # N₁ contribution (Eq. 69)
        S_N1 = compute_N1_contribution(mesh, h, Gamma_sq, Psi1_bar, eta_bar, alpha)
        b = b + ns_scale * S_N1

        # Hoop contribution (Eq. 70)
        S_hoop = compute_hoop_contribution(mesh, h, Gamma_sq, Psi1_bar, eta_bar, alpha)
        b = b + ns_scale * S_hoop

        # Giesekus coupling L_alpha contribution (Eq. 68)
        # Previously only computed in post-processing flux decomposition.
        # Now included as explicit (lagged) RHS for full Eq. 4.31 fidelity.
        if alpha > 1e-10:
            S_alpha = compute_alpha_contribution(
                mesh, h, gdot_bar, Psi1_bar, eta_bar, alpha
            )
            b = b + ns_scale * S_alpha
    
    # Apply boundary conditions
    p_inner = conditions.p_supply if hasattr(conditions, 'p_supply') else 0.0
    p_outer = conditions.p_ambient if hasattr(conditions, 'p_ambient') else 0.0
    
    A, b = mesh.apply_dirichlet_bc(A, b, p_inner, p_outer)
    
    return AssembledSystem(
        A=A, b=b, A_GNF=A_GNF, A_mem=A_mem,
        info={
            'type': 'viscoelastic',
            'include_memory': include_memory,
            'include_normal_stress': include_normal_stress
        }
    )
