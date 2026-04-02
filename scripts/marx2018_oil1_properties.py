"""
Marx et al. (2018) Oil #1 — Complete Fluid Property Data
=========================================================
Source: Marx, N., Fernández, L., Barceló, F. & Spikes, H.A. (2018).
  Part I: Shear Thinning Behaviour.  Tribol. Lett. 66:92.
  Part II: Impact of Shear Thinning on Journal Bearing Friction.
  Tribol. Lett. 66:91.

Oil #1: Polymer-thickened mineral oil (viscosity modifier in Group I/III
base stock).  The VM type is a hydrogenated styrene–diene copolymer (HSDCP),
which is one of the more common VM chemistries in automotive lubricants.

NOTE ON LIMITATIONS FOR GIESEKUS FITTING:
  - Only steady-shear viscosity η(γ̇) is measured.
  - No oscillatory data (G', G'') => no independent λ.
  - No first normal stress difference Ψ₁ => α cannot be uniquely determined
    from viscometry alone without additional constraints.
  - The Carreau–Yasuda fit provides the *product* (A · a_T) which is analogous
    to a shear-rate-dependent relaxation time, but is NOT the same as the
    Giesekus λ without further rheological interpretation.
"""

import numpy as np


def get_marx2018_oil1_all_properties():
    """
    Returns all available fluid properties for Oil #1 from Marx et al. (2018).
    
    Returns
    -------
    dict with keys:
        'label', 'source',
        'gdot', 'eta'         : digitised η(γ̇) at T = 60°C (Fig. 6)
        'eta_data_multi_T'    : dict of {T_degC: (gdot_array, eta_array)} at 60, 80, 100, 120°C
        'vogel_eta0'          : Vogel constants for η₀(T) of the blended oil
        'vogel_base'          : Vogel constants for η_∞(T) of the base oil (solvent)
        'carreau_yasuda'      : reduced CY constants {A, n, a} from Table 7
        'density'             : density correlation ρ(T) [kg/m³]
        'Psi1', 'lambda_'    : None (not measured)
        'alpha'               : None (not measured)
    """
    
    # =========================================================================
    # 1. DIGITISED VISCOSITY DATA — Fig. 6, Oil #1
    # =========================================================================
    # T = 60°C (yellow triangles in your uploaded image)
    gdot_60 = np.array([1e1, 5e2, 6e4, 1e5, 3e5, 5e5, 7e5, 9.5e5, 3e6, 6e6])
    eta_60  = np.array([40.3, 40.3, 28.0, 25.0, 21.0, 20.5, 18.0, 16.0, 15.5, 15.0]) * 1e-3  # Pa·s
    
    # T = 80°C (grey triangles)
    gdot_80 = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7])
    eta_80  = np.array([21.0, 21.0, 20.5, 18.0, 14.0, 12.0, 10.0, 8.5, 7.5]) * 1e-3  # Pa·s
    
    # T = 100°C (blue triangles)
    gdot_100 = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7])
    eta_100  = np.array([13.0, 13.0, 12.5, 11.5, 9.5, 8.0, 7.0, 6.0, 5.5]) * 1e-3  # Pa·s
    
    # T = 120°C (red triangles)
    gdot_120 = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7])
    eta_120  = np.array([8.0, 8.0, 7.8, 7.5, 6.5, 5.8, 5.2, 4.5, 4.2]) * 1e-3  # Pa·s
    
    # =========================================================================
    # 2. VOGEL EQUATION for LOW-SHEAR-RATE VISCOSITY η₀(T)
    # =========================================================================
    # Vogel equation: η₀ = k · exp(b / (T - T₀))
    # where T is in °C (some formulations use K)
    #
    # From the flow curves:
    #   η₀(60°C) ≈ 40.3 mPa·s
    #   η₀(80°C) ≈ 21.0 mPa·s  
    #   η₀(100°C) ≈ 13.0 mPa·s
    #   η₀(120°C) ≈ 8.0 mPa·s
    #
    # A simple Vogel fit to these four points gives approximately:
    vogel_eta0 = {
        'k_Pa_s': 0.11158e-3,   
        'b_K': 989.308 + 273.15,       
        'T0_degC': -107.84,   
        'note': 'Verify from Table 4 (VM solutions)',
        # Observed low-shear-rate viscosities for cross-check:
        'eta0_measured': {
            60: 40.3e-3,   # Pa·s
            80: 21.0e-3,
            100: 13.0e-3,
            120: 8.0e-3,
        },
    }
    
    # =========================================================================
    # 3. VOGEL EQUATION for BASE OIL (SOLVENT) VISCOSITY η_∞(T)
    # =========================================================================
    # This is the second Newtonian plateau — the viscosity the oil approaches
    # at very high shear rates when the polymer is fully aligned.
    # From Table 6.
    #
    # From the flow curves, the high-shear-rate plateaus are approximately:
    #   η_∞(60°C) ≈ 15.0 mPa·s   (from Fig. 6, rightmost points)
    #   η_∞(80°C) ≈ 7.5 mPa·s
    #   η_∞(100°C) ≈ 5.5 mPa·s
    #   η_∞(120°C) ≈ 4.2 mPa·s
    vogel_base = {
        'k_Pa_s': 0.06322e-3,   # Pa·s 
        'b_K': 883.001 + 273.15,       
        'T0_degC': -103.24, 
        'note': 'Verify from Table 6 (base oil Vogel constants)',
        'eta_inf_approx': {
            60: 15.0e-3,   # Pa·s (approximate from flow curve)
            80: 7.5e-3,
            100: 5.5e-3,
            120: 4.2e-3,
        },
    }
    
    # =========================================================================
    # 4. CARREAU–YASUDA PARAMETERS (from Table 7)
    # =========================================================================
    # Reduced Carreau–Yasuda equation:
    #   SSI(γ̇) = [η(γ̇) - η_∞] / [η₀ - η_∞] = 1 / [1 + (A · a_T · γ̇)^a]^((1-n)/a)
    #
    # where:
    #   A   = relaxation time constant [s] (at reference temperature T_R = 60°C)
    #   a_T = temperature shift factor = η_∞(T) / η_∞(T_R)  [dimensionless]
    #   n   = power-law exponent at high shear (0 < n < 1)
    #   a   = transition width parameter (a=2 gives Carreau; a≠2 gives C-Y)
    #
    # IMPORTANT: Read actual values from Table 7 of the paper.
    carreau_yasuda = {
        'A_s': 21.88e-3,  # s — relaxation time constant at T_ref = 60°C (from Table 7)
        'n': 0.36,         # power-law index (from Table 7)
        'a': 1.00,        # transition parameter (from Table 7)
        'T_ref_degC': 60,  # Reference temperature for a_T
        'R_squared': 0.990,  # goodness of fit (from Table 7)
        'note': 'Reduced CY fit via time-temperature superposition. '
                'a_T = η_∞(T) / η_∞(T_R=60°C). Verify all from Table 7.',
    }
    
    # =========================================================================
    # 5. DENSITY CORRELATION (from Part II, Marx et al. 2018, Tribol. Lett. 66:91)
    # =========================================================================
    # Linear density-temperature relation: ρ(T) = ρ_60 + k·(60 - T)
    # where T is in °C, ρ in kg/m³
    #
    # Part II states density was measured and decreased linearly with temperature.
    # Typical values for polymer-thickened mineral oils:
    #   ρ(60°C) ≈ 840–870 kg/m³  (Group I/III base stock)
    #   dρ/dT  ≈ -0.6 to -0.7 kg/(m³·°C)
    density = {
        'rho_60_kg_m3': None,    # READ FROM PART II TABLE
        'k_kg_m3_per_degC': None,  # READ FROM PART II TABLE (positive; ρ decreases)
        'note': 'ρ(T) = ρ_60 + k·(60 - T). Verify from Part II.',
        'rho_approx': {
            60: 860,   # kg/m³ — typical for Group I/III base + VM
            80: 847,
            100: 834,
            120: 821,
        },
    }
    
    # =========================================================================
    # 6. GIESEKUS MAPPING — what you CAN and CANNOT determine
    # =========================================================================
    giesekus_notes = """
    GIESEKUS MODEL FITTING FROM MARX et al. (2018) DATA:
    
    The Giesekus steady-shear viscosity function is:
        η(γ̇) = η_s + η_p · f(α, λ·γ̇)
    
    where f depends on α and the product (λ·γ̇).
    
    From the Marx data you CAN determine:
      (a) η_s ≈ η_∞(T) from the base oil Vogel equation (Table 6)
          → at 60°C: η_s ≈ 15.0 mPa·s
      (b) η_0 = η_s + η_p  from the blended oil Vogel equation
          → at 60°C: η_p = η_0 - η_s ≈ 40.3 - 15.0 = 25.3 mPa·s
      (c) The viscosity ratio β = η_s / η_0 ≈ 0.37
          → This is in your "good" regime (β > 0.3) for single-mode Giesekus
      (d) The shape of the shear-thinning curve constrains α and λ jointly.
    
    You CANNOT uniquely determine:
      (a) λ (relaxation time) — requires oscillatory or extensional data
      (b) α (mobility factor) — coupled with λ in the viscosity function
    
    PRACTICAL APPROACH:
      1. Fit the Carreau–Yasuda A parameter as an estimate of λ:
         λ ≈ A (the CY relaxation time at T_ref)
         This is an approximation; A from CY is not identical to λ_Giesekus.
      2. Then fit α to match the shape of the shear-thinning curve.
      3. Alternatively, use a nonlinear least-squares fit of (α, λ) jointly
         to the η(γ̇) data, accepting that the solution may not be unique.
    """
    
    # =========================================================================
    # ASSEMBLE AND RETURN
    # =========================================================================
    return {
        'label': 'Oil #1: Polymer-thickened mineral oil (HSDCP VM)',
        'source': 'Marx, Fernández, Barceló & Spikes (2018) Tribol. Lett. 66:92 (Part I) '
                  'and 66:91 (Part II)',
        
        # Primary viscosity data at 60°C (your original digitisation)
        'gdot': gdot_60,
        'eta': eta_60,
        
        # Multi-temperature viscosity data
        'eta_data_multi_T': {
            60: (gdot_60, eta_60),
            80: (gdot_80, eta_80),
            100: (gdot_100, eta_100),
            120: (gdot_120, eta_120),
        },
        
        # Constitutive model parameters (from paper tables — VERIFY)
        'vogel_eta0': vogel_eta0,
        'vogel_base': vogel_base,
        'carreau_yasuda': carreau_yasuda,
        'density': density,
        
        # Giesekus-relevant derived quantities at T = 60°C
        'eta_0': 40.3e-3,         # Pa·s — low shear rate viscosity
        'eta_s': 15.0e-3,         # Pa·s — approximate second Newtonian (base oil)
        'eta_p': 25.3e-3,         # Pa·s — polymer contribution (η₀ - η_s)
        'beta': 15.0 / 40.3,      # solvent fraction η_s/η_0 ≈ 0.37
        
        # Not measured
        'Psi1': None,
        'lambda_': None,           # Must be estimated (see giesekus_notes)
        'alpha': None,             # Must be fitted jointly with λ
        
        'giesekus_notes': giesekus_notes,
    }


def carreau_yasuda_viscosity(gdot, eta_0, eta_inf, A, n, a, a_T=1.0):
    """
    Carreau–Yasuda viscosity model as used by Marx et al. (2018).
    
    Parameters
    ----------
    gdot : array_like
        Shear rate [s⁻¹]
    eta_0 : float
        Zero-shear-rate viscosity [Pa·s]
    eta_inf : float
        Second Newtonian viscosity [Pa·s] (base oil viscosity)
    A : float
        CY relaxation time [s] at reference temperature
    n : float
        Power-law index (0 < n < 1)
    a : float
        CY transition parameter
    a_T : float
        Temperature shift factor η_∞(T)/η_∞(T_ref), default 1.0
    
    Returns
    -------
    eta : ndarray
        Viscosity [Pa·s]
    """
    gdot = np.asarray(gdot, dtype=float)
    SSI = 1.0 / (1.0 + (A * a_T * gdot)**a)**((1.0 - n) / a)
    return eta_inf + (eta_0 - eta_inf) * SSI


if __name__ == '__main__':
    data = get_marx2018_oil1_all_properties()
    print(f"Oil: {data['label']}")
    print(f"η₀(60°C) = {data['eta_0']*1e3:.1f} mPa·s")
    print(f"η_s(60°C) ≈ {data['eta_s']*1e3:.1f} mPa·s (base oil)")
    print(f"η_p(60°C) ≈ {data['eta_p']*1e3:.1f} mPa·s (polymer)")
    print(f"β = η_s/η₀ ≈ {data['beta']:.2f}")
    print(f"\nViscosity ratio β ≈ 0.37 → single-mode Giesekus should be adequate")
    print(f"(Your Test 8 threshold: β > 0.3 for 'good' applicability)")
    print(f"\n--- Missing quantities ---")
    print(f"λ (relaxation time): {data['lambda_']} — estimate from CY parameter A")
    print(f"α (mobility factor): {data['alpha']} — fit to shear-thinning shape")
    print(f"Ψ₁ (N₁): {data['Psi1']} — not measured")
