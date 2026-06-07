# Isothermal Giesekus Squeeze-Film Solver

[![Tests](https://img.shields.io/badge/tests-124%20passed-green)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python implementation of the isothermal viscoelastic Reynolds equation derived in the accompanying paper from a perturbation expansion in the squeeze Deborah number using the Giesekus constitutive model.

**Author:** Nigel C. Dhlamini
**Affiliation:** University of Cape Town, Department of Mechanical Engineering
**Contact:** dhlnig001@myuct.ac.za | [GitHub](https://github.com/nigeldhlamini)

## Reference

This solver accompanies the paper:

> N.C. Dhlamini, *"Viscoelastic squeeze-film lubrication: a Giesekus-based perturbation expansion in λḣ/h"*, Physics of Fluids (under review, 2026).

The expansion parameter is the squeeze Deborah number $\mathrm{De}_\mathrm{sq} = \lambda|\dot h|/h$, which governs the temporal lag of polymer stress behind a changing gap. It is distinct from the conventional Deborah number $\mathrm{De} = \lambda U/L$ used in the lubrication literature, and remains well-ordered ($\mathrm{De}_\mathrm{sq} \sim 0.01$–$0.1$) even at large Weissenberg numbers where a $\mathrm{De}$-based expansion is invalid.

## Scope

This repository implements the theoretical framework described in the paper. It is intended as a reproducibility artifact for the analytical derivations, limiting-case checks, and Boger-fluid validation reported there. Application to specific bearing geometries, thermal effects, and quantitative engineering predictions are deliberately outside the paper's scope and are not implemented here.

## Key features

- Flux decomposition: steady-state viscometric, squeeze-film memory, first-normal-stress gradient, and hoop stress; the Giesekus α-coupling enters as a renormalisation of the squeeze-film memory flux rather than as a separate term.
- Closed-form Giesekus material functions ($\bar\eta$, $\eta_T$, $\bar\Psi_1$) at arbitrary Weissenberg number.
- Cell-centred polar finite-difference discretisation with ghost-node Dirichlet boundary conditions for second-order global accuracy.
- Picard iteration with under-relaxation and a normal-stress ramp for robust convergence.
- Recovery of the Newtonian, upper-convected Maxwell, and second-order fluid limits as built-in checks.

## Repository layout

```
src/                     core solver package
  geometry.py            slipper geometry and operating conditions
  rheology.py            Giesekus material functions
  mesh.py                polar mesh and derivative operators
  fluxes.py              flux decomposition
  assembly.py            sparse matrix assembly (with ghost-node BCs)
  solver.py              Picard iteration loop
  results.py             post-processing (load, moments, leakage)
  verification.py        Richardson extrapolation utilities
tests/                   124 unit tests
scripts/
  generate_paper1_figures.py    reproduces Figs 2, 4, 5, 6
  figstyle.py                   shared figure style
  convergence_check.py          grid-convergence sanity check
demo.py                  minimal end-to-end usage example
```

## Quick start

```python
from src import (
    GiesekusFluid,
    SlipperGeometry,
    OperatingConditions,
    PolarMesh,
    solve_viscoelastic,
)

fluid = GiesekusFluid(eta_s=0.030, eta_p=0.120, lambda_=3.3e-3, alpha=0.25)
geometry = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
mesh = PolarMesh(geometry, Nr=40, Ntheta=60)
conditions = OperatingConditions(p_supply=20e6, V_T=10.0, h_dot=-5e-4)

result = solve_viscoelastic(mesh, geometry, conditions, fluid)
print(f"F_total = {result.F_total:.2f} N, iterations = {result.iterations}")
```

A complete worked example with diagnostics is in `demo.py`.

## Reproducing the paper figures

```bash
PYTHONPATH=. python scripts/generate_paper1_figures.py
```

Outputs are written to `figures/paper1/` as both PNG and PDF:

| File                              | Paper figure | Description                                              |
|-----------------------------------|--------------|----------------------------------------------------------|
| `fig2_material_functions.{png,pdf}` | Fig. 2       | Giesekus material functions $\bar\eta$, $\eta_T$, $\bar\Psi_1$ |
| `fig4_grid_convergence.{png,pdf}`   | Fig. 4       | Newtonian hydrostatic grid-convergence study             |
| `fig5_ucm_validation.{png,pdf}`     | Fig. 5       | UCM limit ($\alpha\to 0$) vs Tichy (1996) analytical     |
| `fig6_ptt1985_validation.{png,pdf}` | Fig. 6       | Boger fluid squeeze flow vs Phan-Thien et al. (1985)     |

## Tests

```bash
pytest tests/
```

All 88 tests should pass on Python 3.10+ with NumPy, SciPy, and Matplotlib.

## References

1. H. Giesekus, *J. Non-Newtonian Fluid Mech.* **11**, 69–109 (1982).
2. J.A. Tichy, *J. Tribol.* **118**, 344–348 (1996).
3. N. Phan-Thien and R.I. Tanner, *J. Fluid Mech.* **129**, 265–281 (1983).
4. N. Phan-Thien, J. Dudek, D.V. Boger, and V. Tirtaatmadja, *J. Non-Newtonian Fluid Mech.* **18**, 227–254 (1985).

## License

Released under the MIT License — see [LICENSE](LICENSE).
