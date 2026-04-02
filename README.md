# giesekus-solver

[![Tests](https://img.shields.io/badge/tests-81%20passed-green)](tests/)
[![Validation](https://img.shields.io/badge/validation-6%2F6%20passed-green)](scripts/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

A Python implementation of the isothermal Giesekus-based viscoelastic Reynolds equation for slipper bearing lubrication.

**Author:** Nigel C. Dhlamini
**Affiliation:** University of Cape Town, Department of Mechanical Engineering
**Contact:** dhlnig001@myuct.ac.za | [GitHub](https://github.com/nigeldhlamini)

## Overview

This solver accompanies the paper:

> N.C. Dhlamini, *"Viscoelastic Reynolds equation for thin-film lubrication via squeeze Deborah number expansion with the Giesekus constitutive model"*, Journal of Non-Newtonian Fluid Mechanics (2026).

The key innovation is the use of the **squeeze Deborah number** (De_sq = lambda|h_dot|/h_0) as the perturbation parameter, which remains well-ordered (De_sq ~ 0.01-0.1) even at high Weissenberg numbers (Wi ~ 2000-6600) characteristic of polymer-thickened hydraulic fluids.

### Key Features

- **Complete flux decomposition**: GNF + memory + N1 + hoop contributions
- **Giesekus material functions** with correct asymptotic behaviour
- **Picard iteration** with under-relaxation for robust convergence
- **81 unit tests** + 6 validation benchmarks

## Quick Start

```python
from src import create_PAM_5pct, create_standard_slipper, PolarMesh, SolverConfig, solve_viscoelastic
from src.geometry import OperatingConditions

fluid = create_PAM_5pct()
geometry = create_standard_slipper()
mesh = PolarMesh(geometry, Nr=40, Ntheta=60)
conditions = OperatingConditions(p_supply=200e5, V_T=10.0, h_dot=-0.0005)
result = solve_viscoelastic(mesh, geometry, conditions, fluid)
print(result.summary())
```

## Validation

| Test | Status | Error |
|------|--------|-------|
| Hydrostatic analytical | Pass | 0.28% |
| Squeeze scaling (eta) | Pass | linear |
| SOF limit recovery | Pass | <0.01% |
| GNF convergence | Pass | stable |
| Viscoelastic solver | Pass | converges |
| De_sq parametric | Pass | correct |

## Reproducing Paper Figures

```bash
PYTHONPATH=. python scripts/generate_paper1_figures.py
```

Figures are saved to `figures/paper1/` as both PNG and PDF.

## References

1. Giesekus, H. (1982). J. Non-Newtonian Fluid Mech., 11, 69-109.
2. Tichy, J.A. (1996). J. Tribol., 118, 344-348.
3. Phan-Thien & Tanner (1983). J. Fluid Mech., 129, 265-281.

## License

MIT License
