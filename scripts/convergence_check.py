"""Quick convergence check for ghost-node BC fix.

Computes load capacity at several radial grid resolutions against a
fine-grid reference and reports the observed order of accuracy.

Run with:
    python -u scripts/convergence_check.py
"""
import sys
import numpy as np

from src import (
    SlipperGeometry,
    OperatingConditions,
    PolarMesh,
    GiesekusFluid,
    solve_viscoelastic,
    SolverConfig,
)


def main() -> None:
    geometry = SlipperGeometry(
        R_in=0.008, R_out=0.020, h_0=5e-6,
        alpha_x=0.0, alpha_y=0.0,
    )
    conditions = OperatingConditions(
        p_supply=20e6, p_ambient=0.0,
        V_T=10.0, h_dot=-0.01,
    )
    fluid = GiesekusFluid(
        eta_s=0.030, eta_p=0.120,
        lambda_=3.3e-3, alpha=0.25,
    )
    config = SolverConfig(max_iter=60, tol=1e-8, omega=0.5, verbose=False)

    print("Running fine-grid reference Nr=320...", flush=True)
    mesh_ref = PolarMesh(geometry, Nr=320, Ntheta=640)
    res_ref = solve_viscoelastic(mesh_ref, geometry, conditions, fluid, config)
    F_ref = res_ref.F_total
    print(f"F_ref (Nr=320) = {F_ref:.6e}", flush=True)

    Nr_list = [10, 20, 40, 80, 160]
    err_list = []
    for Nr in Nr_list:
        mesh = PolarMesh(geometry, Nr=Nr, Ntheta=2 * Nr)
        res = solve_viscoelastic(mesh, geometry, conditions, fluid, config)
        rel_err = abs(res.F_total - F_ref) / abs(F_ref)
        err_list.append(rel_err)
        print(
            f"Nr={Nr:4d}: F={res.F_total:.6e}  rel_err={rel_err:.3e}",
            flush=True,
        )

    rates = [
        np.log(err_list[i] / err_list[i + 1]) / np.log(2)
        for i in range(len(err_list) - 1)
    ]
    print("Convergence rates:", [f"{r:.2f}" for r in rates], flush=True)


if __name__ == "__main__":
    sys.exit(main())
