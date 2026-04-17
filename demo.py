#!/usr/bin/env python
"""
Giesekus Solver Demonstration
=============================

This script demonstrates the complete viscoelastic Reynolds equation solver
for a slipper bearing with squeeze motion.

Author: Nigel C. Dhlamini
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src import (
    GiesekusFluid,
    create_standard_slipper,
    PolarMesh,
    SolverConfig,
    solve,
)
from src.geometry import OperatingConditions


def main():
    print("=" * 60)
    print("Giesekus Viscoelastic Reynolds Equation Solver")
    print("=" * 60)

    # Representative Giesekus fluid (5% PAM in water-glycol parameters)
    # eta_0 = 0.150 Pa.s, eta_s = 0.030 Pa.s, lambda = 3.3 ms, alpha = 0.25
    fluid = GiesekusFluid(eta_s=0.030, eta_p=0.120,
                          lambda_=3.3e-3, alpha=0.25)
    print(f"\nFluid Properties:")
    print(f"  Zero-shear viscosity η₀ = {fluid.eta_0:.3f} Pa·s")
    print(f"  Solvent viscosity η_s   = {fluid.eta_s:.3f} Pa·s")
    print(f"  Relaxation time λ       = {fluid.lambda_*1e3:.2f} ms")
    print(f"  Mobility factor α       = {fluid.alpha}")
    print(f"  Ψ₂/Ψ₁ ratio             = {-fluid.alpha/2:.3f}")
    
    # Create geometry
    geometry = create_standard_slipper()
    print(f"\nGeometry:")
    print(f"  Inner radius R_in  = {geometry.R_in*1e3:.1f} mm")
    print(f"  Outer radius R_out = {geometry.R_out*1e3:.1f} mm")
    print(f"  Central gap h₀     = {geometry.h_0*1e6:.1f} μm")
    print(f"  Aspect ratio ε     = {geometry.aspect_ratio:.2e}")
    
    # Create mesh
    mesh = PolarMesh(geometry, Nr=40, Ntheta=60)
    print(f"\nMesh:")
    print(f"  Radial nodes       = {mesh.Nr}")
    print(f"  Azimuthal nodes    = {mesh.Ntheta}")
    print(f"  Total DOFs         = {mesh.N}")
    
    # Operating conditions with squeeze
    conditions = OperatingConditions(
        p_supply=200e5,   # 200 bar supply pressure
        V_T=10.0,         # 10 m/s orbital velocity
        omega_s=0.0,      # No spin
        h_dot=-0.001      # Closing gap at 1 mm/s
    )
    print(f"\nOperating Conditions:")
    print(f"  Supply pressure    = {conditions.p_supply/1e5:.0f} bar")
    print(f"  Orbital velocity   = {conditions.V_T:.1f} m/s")
    print(f"  Squeeze velocity   = {conditions.h_dot*1e3:.1f} mm/s (closing)")
    
    # Compute dimensionless numbers
    De = fluid.De(conditions.V_T, geometry.L)
    Wi = fluid.Wi(conditions.V_T, geometry.h_0)
    De_sq = fluid.lambda_ * abs(conditions.h_dot) / geometry.h_0  # Manual calculation
    
    print(f"\nDimensionless Numbers:")
    print(f"  Deborah number De     = {De:.2f}")
    print(f"  Weissenberg number Wi = {Wi:.0f}")
    print(f"  Squeeze Deborah De_sq = {De_sq:.3f}")
    
    # Solver configuration
    config = SolverConfig(
        max_iter=50,
        tol=1e-6,
        omega=0.6,
        verbose=True,
        include_memory=True,
        include_normal_stress=True
    )
    
    print("\n" + "=" * 60)
    print("Solving Viscoelastic Reynolds Equation...")
    print("=" * 60)
    
    # Solve
    result = solve(mesh, geometry, conditions, fluid, 
                   model='viscoelastic', config=config)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(result.summary())
    
    # Flux decomposition
    if result.fluxes is not None:
        ratios = result.fluxes.flux_ratios()
        print("\nFlux Decomposition (relative magnitudes):")
        print(f"  GNF (Couette+Poiseuille) : {ratios['GNF']*100:5.1f}%")
        print(f"  Memory (squeeze elastic) : {ratios['memory']*100:5.1f}%")
        print(f"  α-coupling               : {ratios['alpha']*100:5.1f}%")
        print(f"  N₁ gradient              : {ratios['N1']*100:5.1f}%")
        print(f"  Hoop stress              : {ratios['hoop']*100:5.1f}%")
    
    # Shear-thinning effect
    if result.eta_bar is not None:
        eta_min = result.eta_bar.min()
        eta_max = result.eta_bar.max()
        print(f"\nViscosity Field:")
        print(f"  Minimum η/η₀ = {eta_min/fluid.eta_0:.2f}")
        print(f"  Maximum η/η₀ = {eta_max/fluid.eta_0:.2f}")
    
    # Compare with GNF-only
    print("\n" + "-" * 40)
    print("Comparison: GNF vs Full Viscoelastic")
    print("-" * 40)
    print(f"  GNF load capacity:        {result.F_GNF:.2f} N")
    print(f"  Viscoelastic load:        {result.F_total:.2f} N")
    print(f"  Elastic enhancement:      {result.F_elastic:.2f} N")
    print(f"  Enhancement ratio:        {result.elastic_enhancement*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Solver demonstration complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
