"""
Giesekus-Based Viscoelastic Reynolds Equation Solver
=====================================================

A Python implementation of the isothermal Giesekus-based viscoelastic
Reynolds equation for slipper bearing lubrication.

Modules
-------
geometry : Slipper geometry and operating conditions
rheology : Giesekus material functions (η, Ψ₁, Ψ₂)
mesh : Polar grid and derivative operators
fluxes : Flux components (GNF, memory, N₁, hoop)
assembly : Reynolds equation matrix assembly
solver : Iteration loop and convergence control
results : Output dataclass and post-processing

Usage
-----
>>> from src import GiesekusFluid, SlipperGeometry, PolarMesh
>>> from src import create_PAM_5pct, create_standard_slipper
>>> 
>>> fluid = create_PAM_5pct()
>>> geometry = create_standard_slipper()
>>> mesh = PolarMesh(geometry, Nr=60, Ntheta=60)

Author: Nigel C. Dhlamini
"""

__version__ = "0.1.0"
__author__ = "Nigel C. Dhlamini"

# Core classes
from .geometry import SlipperGeometry, OperatingConditions
from .rheology import GiesekusFluid, SOFFluid
from .mesh import PolarMesh

# Factory functions
from .geometry import (
    create_standard_slipper,
    create_tilted_slipper,
    create_standard_operating_conditions
)
from .rheology import (
    create_newtonian,
)

# Verification utilities
from .rheology import verify_psi_ratio, verify_sof_limit
from .mesh import verify_derivative_accuracy, convergence_study
from .verification import (
    richardson_extrapolation,
    RichardsonResult,
    convergence_study as grid_convergence_study,
    format_convergence_table,
)
# Results
from .results import (
    SolverResult, 
    FluxDecomposition,
    compute_load_capacity,
    compute_moments,
    compute_leakage,
    post_process_result,
)

# Fluxes
from .fluxes import (
    FluxCalculator,
    compute_gap_averaged_shear_rate,
    compute_Gamma_squared,
    compute_Q_GNF,
    compute_Q_memory,
    compute_Q_alpha,
    compute_Q_N1,
    compute_Q_hoop,
    compute_source_term,
    compute_newtonian_fluxes,
)

# Assembly
from .assembly import (
    AssembledSystem,
    build_diffusion_operator,
    build_GNF_operator,
    build_memory_operator,
    compute_source_vector,
    assemble_newtonian_system,
    assemble_GNF_system,
    assemble_viscoelastic_system,
)

# Solver
from .solver import (
    SolverConfig,
    solve_newtonian,
    solve_GNF,
    solve_viscoelastic,
    solve,
)


__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Core classes
    'SlipperGeometry',
    'OperatingConditions', 
    'GiesekusFluid',
    'SOFFluid',
    'PolarMesh',
    
    # Factory functions
    'create_standard_slipper',
    'create_tilted_slipper',
    'create_standard_operating_conditions',
    'create_newtonian',
    
    # Verification
    'verify_psi_ratio',
    'verify_sof_limit',
    'verify_derivative_accuracy',
    'convergence_study',
    'richardson_extrapolation',
    'RichardsonResult',
    'grid_convergence_study',
    'format_convergence_table',

    # Results
    'SolverResult',
    'FluxDecomposition',
    'compute_load_capacity',
    'compute_moments',
    'compute_leakage',
    'post_process_result',
    
    # Fluxes
    'FluxCalculator',
    'compute_gap_averaged_shear_rate',
    'compute_Gamma_squared',
    'compute_Q_GNF',
    'compute_Q_memory',
    'compute_Q_alpha',
    'compute_Q_N1',
    'compute_Q_hoop',
    'compute_source_term',
    'compute_newtonian_fluxes',
    
    # Assembly
    'AssembledSystem',
    'build_diffusion_operator',
    'build_GNF_operator',
    'build_memory_operator',
    'compute_source_vector',
    'assemble_newtonian_system',
    'assemble_GNF_system',
    'assemble_viscoelastic_system',
    
    # Solver
    'SolverConfig',
    'solve_newtonian',
    'solve_GNF',
    'solve_viscoelastic',
    'solve',

]
