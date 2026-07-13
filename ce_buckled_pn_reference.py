#!/usr/bin/env python3
"""Buckled continuous-energy/fine-group PN reference.

This script solves the homogeneous Fourier-mode PN equations

    (Sigma_t - S_l) phi_l
      + i B [ l/(2l+1) phi_{l-1} + (l+1)/(2l+1) phi_{l+1} ]
      = delta_{l0} chi

on the same fine lethargy grid and with the same material setup used by
``ce_buckled_transport_reference.py``.  For a fixed B2, a unit fission source is
applied and the resulting fission production is k(B2).  A scalar bisection
search can also find the critical B2 for which k(B2) = 1.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import tracemalloc
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, lgmres

import ce_buckled_transport_reference as snref


def multiplication_matrix(order: int) -> np.ndarray:
    """Return the PN moment matrix for multiplication by mu."""
    L = order + 1
    mat = np.zeros((L, L), dtype=float)
    for ell in range(L):
        if ell > 0:
            mat[ell, ell - 1] = ell / (2 * ell + 1)
        if ell + 1 < L:
            mat[ell, ell + 1] = (ell + 1) / (2 * ell + 1)
    return mat


class BuckledPNReference:
    def __init__(self, data: snref.MaterialData, pn_order: int):
        if pn_order < 0:
            raise ValueError("pn_order must be nonnegative.")
        self.data = data
        self.pn_order = pn_order
        self.moment_count = pn_order + 1
        self.mu_matrix = multiplication_matrix(pn_order)
        self.last_moments: np.ndarray | None = None
        self.last_b2: float | None = None

    def negative_b2_floor(self, safety: float) -> float:
        eigs = np.linalg.eigvals(self.mu_matrix)
        max_positive_mu = float(np.max(np.real(eigs)))
        if max_positive_mu <= 0.0:
            return -math.inf
        beta_limit = float(np.min(self.data.sigma_t)) / max_positive_mu
        return -(safety * beta_limit) ** 2

    def _streaming_factor(self, b2: float) -> complex:
        return 1j * np.sqrt(np.complex128(b2))

    def _scatter_apply(self, moments: np.ndarray) -> np.ndarray:
        out = np.zeros_like(moments)
        scatter_count = min(self.data.scatter.shape[0], self.moment_count)
        for ell in range(scatter_count):
            out[ell] = self.data.scatter[ell] @ moments[ell]
        return out

    def matvec_for_b2(self, b2: float):
        streaming = self._streaming_factor(b2)

        def matvec(vec: np.ndarray) -> np.ndarray:
            moments = vec.reshape(self.moment_count, self.data.sigma_t.size)
            out = self.data.sigma_t[None, :] * moments - self._scatter_apply(moments)
            out += streaming * (self.mu_matrix @ moments)
            return out.ravel()

        return matvec

    def local_preconditioner(self, b2: float) -> LinearOperator:
        G = self.data.sigma_t.size
        L = self.moment_count
        streaming = self._streaming_factor(b2)
        blocks = np.zeros((G, L, L), dtype=np.complex128)
        for g in range(G):
            blocks[g] = streaming * self.mu_matrix
            for ell in range(L):
                blocks[g, ell, ell] += self.data.sigma_t[g]
                if ell < self.data.scatter.shape[0]:
                    blocks[g, ell, ell] -= self.data.scatter[ell, g, g]

        def apply(vec: np.ndarray) -> np.ndarray:
            rhs = vec.reshape(L, G).T
            solved = np.linalg.solve(blocks, rhs)
            return solved.T.ravel()

        return LinearOperator((L * G, L * G), matvec=apply, dtype=np.complex128)

    def solve_fixed_source(
        self,
        b2: float,
        tol: float,
        max_iters: int,
        gmres_restart: int,
        linear_solver: str,
        verbose: bool,
    ) -> tuple[float, np.ndarray, int, float]:
        snref.report_memory(f"before PN fixed-source solve B2={b2:.8e}")
        shape = (self.moment_count, self.data.sigma_t.size)
        rhs_moments = np.zeros(shape, dtype=np.complex128)
        rhs_moments[0] = self.data.chi
        rhs = rhs_moments.ravel()
        x0 = self.last_moments.ravel() if self.last_moments is not None and self.last_moments.shape == shape else rhs.copy()

        for name, arr in (
            ("sigma_t", self.data.sigma_t),
            ("scatter", self.data.scatter),
            ("nu_sigma_f", self.data.nu_sigma_f),
            ("chi", self.data.chi),
            ("rhs", rhs_moments),
        ):
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"{name} contains NaN or inf before the B2={b2:g} PN solve.")

        matvec = self.matvec_for_b2(b2)
        op = LinearOperator((rhs.size, rhs.size), matvec=matvec, dtype=np.complex128)
        preconditioner = self.local_preconditioner(b2)
        snref.report_memory(f"after PN preconditioner B2={b2:.8e}")
        residuals: list[float] = []

        def callback(residual_norm):
            if np.isscalar(residual_norm):
                residuals.append(float(residual_norm))
            else:
                residuals.append(float("nan"))
            if verbose and (len(residuals) == 1 or len(residuals) % 25 == 0):
                print(f"  {linear_solver} iter {len(residuals):5d}: callback={residuals[-1]:.4e}")

        def add_tolerance_kwargs(kwargs: dict, solver_fn) -> dict:
            params = inspect.signature(solver_fn).parameters
            if "rtol" in params:
                kwargs["rtol"] = tol
                if "atol" in params:
                    kwargs["atol"] = 0.0
            else:
                kwargs["tol"] = tol
                if "atol" in params:
                    kwargs["atol"] = tol
            return kwargs

        if linear_solver == "gmres":
            solve_kwargs = {
                "x0": x0,
                "M": preconditioner,
                "restart": gmres_restart,
                "maxiter": max_iters,
                "callback": callback,
            }
            add_tolerance_kwargs(solve_kwargs, gmres)
            if "callback_type" in inspect.signature(gmres).parameters:
                solve_kwargs["callback_type"] = "pr_norm"
            sol, info = gmres(op, rhs, **solve_kwargs)
        elif linear_solver == "lgmres":
            solve_kwargs = {
                "x0": x0,
                "M": preconditioner,
                "maxiter": max_iters,
                "callback": callback,
                "inner_m": gmres_restart,
            }
            add_tolerance_kwargs(solve_kwargs, lgmres)
            sol, info = lgmres(op, rhs, **solve_kwargs)
        elif linear_solver == "bicgstab":
            solve_kwargs = {
                "x0": x0,
                "M": preconditioner,
                "maxiter": max_iters,
                "callback": callback,
            }
            add_tolerance_kwargs(solve_kwargs, bicgstab)
            sol, info = bicgstab(op, rhs, **solve_kwargs)
        else:
            raise ValueError(f"Unknown linear solver: {linear_solver}")
        snref.report_memory(f"after PN {linear_solver} B2={b2:.8e}")

        moments = sol.reshape(shape)
        residual_vec = matvec(sol) - rhs
        err = float(np.linalg.norm(residual_vec) / max(np.linalg.norm(rhs), 1.0e-300))
        iters = len(residuals) if residuals else (max_iters if info > 0 else 1)
        if info != 0:
            raise RuntimeError(f"{linear_solver} did not converge for PN B2={b2:g}; info={info}, residual={err:.3e}")

        self.last_moments = moments
        self.last_b2 = b2
        k = np.dot(self.data.nu_sigma_f, moments[0]).real
        return float(k), moments, iters, err

    def k_of_b2(self, b2: float, args: argparse.Namespace) -> tuple[float, np.ndarray]:
        k, moments, iters, err = self.solve_fixed_source(
            b2=b2,
            tol=args.inner_tol,
            max_iters=args.inner_max_iters,
            gmres_restart=args.gmres_restart,
            linear_solver=args.linear_solver,
            verbose=args.verbose_inner,
        )
        print(f"PN{self.pn_order} B2={b2:.8e}, k={k:.10e}, fixed-source iters={iters}, residual={err:.3e}")
        return k, moments

    def find_critical_b2(self, args: argparse.Namespace) -> tuple[float, float, np.ndarray]:
        negative_floor = self.negative_b2_floor(args.negative_b2_safety)
        print(
            f"Negative-B2 PN floor = {negative_floor:.8e} "
            f"(safety={args.negative_b2_safety:g}); values below this can cross a local PN streaming pole."
        )
        if args.initial_b2 <= negative_floor:
            raise ValueError(
                f"Initial B2={args.initial_b2:g} is below the PN negative-buckling floor "
                f"{negative_floor:g}. Choose a less negative --initial-b2."
            )

        b_start = args.initial_b2
        k_start, m_start = self.k_of_b2(b_start, args)
        f_start = k_start - 1.0
        if abs(f_start) < args.b2_k_tol:
            return b_start, k_start, m_start

        step = max(args.initial_b2_step, 1.0e-12)
        if f_start > 0.0:
            lo, flo = b_start, f_start
            hi = b_start + step
            while abs(hi) <= args.max_abs_b2:
                k_hi, _ = self.k_of_b2(hi, args)
                fhi = k_hi - 1.0
                if fhi <= 0.0:
                    break
                step *= 2.0
                hi = b_start + step
            else:
                raise RuntimeError(f"Could not bracket critical B2 above initial guess {b_start:g}.")
        else:
            hi, fhi = b_start, f_start
            lo = b_start - step
            while abs(lo) <= args.max_abs_b2 and lo > negative_floor:
                k_lo, _ = self.k_of_b2(lo, args)
                flo = k_lo - 1.0
                if flo >= 0.0:
                    break
                step *= 2.0
                lo = b_start - step
            else:
                raise RuntimeError(
                    f"Could not bracket critical B2 below initial guess {b_start:g} before reaching "
                    f"the PN negative-buckling floor {negative_floor:g}."
                )

        best_b2 = b_start
        best_k = k_start
        best_moments = m_start
        for it in range(1, args.b2_max_iters + 1):
            mid = 0.5 * (lo + hi)
            k_mid, m_mid = self.k_of_b2(mid, args)
            fmid = k_mid - 1.0
            best_b2, best_k, best_moments = mid, k_mid, m_mid
            width = abs(hi - lo)
            print(f"PN B2 search {it:3d}: lo={lo:.6e}, hi={hi:.6e}, mid={mid:.6e}, k-1={fmid:.4e}")
            if abs(fmid) < args.b2_k_tol or width < args.b2_abs_tol:
                return best_b2, best_k, best_moments
            if fmid * flo > 0.0:
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid

        return best_b2, best_k, best_moments


def write_outputs(
    outdir: Path,
    b2: float,
    k: float,
    data: snref.MaterialData,
    moments: np.ndarray,
    collapsed: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "ce_buckled_pn_reference.npz"
    np.savez_compressed(
        npz_path,
        reference_type="PN",
        run_mode=("critical_search" if args.fixed_b2 is None else "fixed_b2"),
        B2=b2,
        is_critical_search=(args.fixed_b2 is None),
        critical_B2=b2,
        k_at_B2=k,
        pn_order=args.pn_order,
        moment_count=args.pn_order + 1,
        scatter_legendre_order=args.legendre_order,
        e_bounds=data.e_bounds,
        e_mid=data.e_mid,
        u_bounds=data.u_bounds,
        sigma_t=data.sigma_t,
        nu_sigma_f=data.nu_sigma_f,
        chi=data.chi,
        moments=moments,
        scatter=data.scatter,
        **collapsed,
    )

    metadata = {
        "reference_type": "PN",
        "critical_B2": b2,
        "B2": b2,
        "k_at_B2": k,
        "run_mode": "critical_search" if args.fixed_b2 is None else "fixed_b2",
        "is_critical_search": args.fixed_b2 is None,
        "fine_groups": int(args.fine_groups),
        "pn_order": int(args.pn_order),
        "moment_count": int(args.pn_order + 1),
        "scatter_legendre_order": int(args.legendre_order),
        "NH": args.NH,
        "NU": args.NU,
        "nu": args.nu,
        "group_structure": args.group_structure,
        "upscatter": not args.no_upscatter,
        "thermal_upscatter_nuclides": args.thermal_upscatter_nuclides,
        "p0_only_upscatter": args.p0_only_upscatter,
        "thermal_upscatter_cutoff_ev": args.thermal_upscatter_cutoff_ev,
        "linear_solver": args.linear_solver,
        "initial_B2": args.initial_b2,
        "fixed_B2": args.fixed_b2,
        "negative_b2_safety": args.negative_b2_safety,
        "memory_profile": snref.MEMORY_PROFILE,
        "note": "Scattering moments above scatter_legendre_order are treated as zero in the PN solve.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with (outdir / "few_group_spectrum.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "E_hi_eV", "E_lo_eV", "phi0", "phi0_imag", "phi0_per_lethargy", "chi"])
        edges = collapsed["group_edges_e"]
        for i in range(edges.size - 1):
            phi = collapsed["coarse_phi0"][i]
            dens = collapsed["coarse_phi0_density"][i]
            writer.writerow([i + 1, edges[i], edges[i + 1], phi.real, phi.imag, dens.real, collapsed["chi_few"][i]])

    with (outdir / "few_group_vectors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "Sigma_t", "Sigma_t_imag", "nuSigma_f", "nuSigma_f_imag", "chi"])
        for i in range(collapsed["Sigma_t"].size):
            st = collapsed["Sigma_t"][i]
            nf = collapsed["nuSigma_f"][i]
            writer.writerow([i + 1, st.real, st.imag, nf.real, nf.imag, collapsed["chi_few"][i]])

    with (outdir / "few_group_scatter_matrices.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ell", "exit_group", "incident_group", "Sigma_s", "Sigma_s_imag"])
        S = collapsed["Sigma_s_moment_weighted"]
        for ell in range(S.shape[0]):
            for i in range(S.shape[1]):
                for j in range(S.shape[2]):
                    val = S[ell, i, j]
                    writer.writerow([ell, i + 1, j + 1, val.real, val.imag])

    edges = collapsed["group_edges_e"][::-1]
    phi = collapsed["coarse_phi0"].real
    density = (phi / max(np.sum(phi), 1.0e-300) / collapsed["coarse_du"])[::-1]
    y = np.r_[density, density[-1]]
    plt.figure(figsize=(9, 5.5))
    plt.step(edges, y, where="post", label=fr"$P_{{{args.pn_order}}}$, $B^2={b2:.6e}$")
    plt.xscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized few-group scalar flux per lethargy")
    plt.title(f"Buckled P{args.pn_order} reference, k={k:.8f}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "few_group_spectrum.png", dpi=200)
    plt.close()
    print(f"Wrote PN reference npz to {npz_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--group-structure", default=os.getenv("CE_GROUP_STRUCTURE", "wims69.txt"))
    p.add_argument("--outdir", default=os.getenv("PN_OUTDIR", "results/ce_pn_reference"))
    p.add_argument("--fine-groups", type=int, default=int(os.getenv("CE_FINE_GROUPS", "12000")))
    p.add_argument("--pn-order", type=int, default=int(os.getenv("PN_ORDER", "3")), help="Highest PN order N. Moment count is N+1.")
    p.add_argument(
        "--scatter-legendre-order",
        type=int,
        default=int(os.getenv("PN_SCATTER_LEGENDRE_ORDER", "4")),
        help="Number of scattering Legendre moments to generate. Current thermal kernel supports 1..4. Default: 4.",
    )
    p.add_argument("--NH", type=float, default=float(os.getenv("SP3_NH", "0.10")))
    p.add_argument("--NU", type=float, default=float(os.getenv("SP3_NU", "0.02")))
    p.add_argument("--nu", type=float, default=float(os.getenv("SP3_NU_BAR", "2.43")))
    p.add_argument("--e0", type=float, default=None)
    p.add_argument("--emin", type=float, default=None)
    p.add_argument("--kT", type=float, default=8.617e-5 * 293.15)
    p.add_argument("--kernel-quad", type=int, default=64)
    p.add_argument("--no-upscatter", action="store_true")
    p.add_argument(
        "--thermal-upscatter-nuclides",
        default=os.getenv("CE_THERMAL_UPSCATTER_NUCLIDES", "H,U"),
        help="Comma-separated nuclides that receive thermal upscatter replacement. Supported: H, U. Default: H,U.",
    )
    p.add_argument(
        "--thermal-upscatter-cutoff-ev",
        type=float,
        default=float(os.getenv("CE_THERMAL_UPSCATTER_CUTOFF_EV", "4.0")),
        help="Incident and exit energy cutoff for thermal upscatter replacement. Default: 4 eV.",
    )
    p.add_argument(
        "--p0-only-upscatter",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("CE_P0_ONLY_UPSCATTER", "true").lower() == "true",
        help="Apply thermal upscatter replacement only to P0, leaving higher-L downscatter moments. Default: true.",
    )
    p.add_argument("--diagnostics", action="store_true", help="Print input and scattering-matrix diagnostics.")
    p.add_argument("--no-scatter-validation", action="store_true", help="Allow nonphysical scattering matrices for debugging.")
    p.add_argument(
        "--max-legendre-ratio",
        type=float,
        default=float(os.getenv("CE_MAX_LEGENDRE_RATIO", "inf")),
        help="Optional failure threshold for |Sigma_sl|/|Sigma_s0|. Default: disabled.",
    )
    p.add_argument("--inner-tol", type=float, default=float(os.getenv("PN_INNER_TOL", os.getenv("CE_INNER_TOL", "1e-8"))))
    p.add_argument("--inner-max-iters", type=int, default=int(os.getenv("PN_INNER_MAX_ITERS", os.getenv("CE_INNER_MAX_ITERS", "5000"))))
    p.add_argument("--gmres-restart", type=int, default=int(os.getenv("PN_GMRES_RESTART", os.getenv("CE_GMRES_RESTART", "80"))))
    p.add_argument(
        "--linear-solver",
        choices=("lgmres", "bicgstab", "gmres"),
        default=os.getenv("PN_LINEAR_SOLVER", os.getenv("CE_LINEAR_SOLVER", "lgmres")),
        help="Krylov method for the PN fixed-source solve. Default: lgmres.",
    )
    p.add_argument("--initial-b2", type=float, default=float(os.getenv("PN_INITIAL_B2", os.getenv("CE_INITIAL_B2", "-0.01"))))
    p.add_argument("--fixed-b2", type=float, default=None, help="Skip B2 search and solve PN at this B2.")
    p.add_argument("--initial-b2-step", type=float, default=float(os.getenv("PN_INITIAL_B2_STEP", os.getenv("CE_INITIAL_B2_STEP", "0.01"))))
    p.add_argument("--max-abs-b2", type=float, default=float(os.getenv("PN_MAX_ABS_B2", os.getenv("CE_MAX_ABS_B2", "1.0"))))
    p.add_argument(
        "--negative-b2-safety",
        type=float,
        default=float(os.getenv("PN_NEGATIVE_B2_SAFETY", os.getenv("CE_NEGATIVE_B2_SAFETY", "0.98"))),
        help="Fraction of the negative-buckling local PN streaming-pole limit allowed during B2 search. Default: 0.98.",
    )
    p.add_argument("--b2-k-tol", type=float, default=float(os.getenv("PN_B2_K_TOL", os.getenv("CE_B2_K_TOL", "1e-6"))))
    p.add_argument("--b2-abs-tol", type=float, default=float(os.getenv("PN_B2_ABS_TOL", os.getenv("CE_B2_ABS_TOL", "1e-8"))))
    p.add_argument("--b2-max-iters", type=int, default=int(os.getenv("PN_B2_MAX_ITERS", os.getenv("CE_B2_MAX_ITERS", "80"))))
    p.add_argument("--verbose-inner", action="store_true")
    p.add_argument("--no-memory-profile", action="store_true", help="Disable memory usage reports.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    snref.MEMORY_PROFILE = not args.no_memory_profile
    if snref.MEMORY_PROFILE and not tracemalloc.is_tracing():
        tracemalloc.start()
    snref.report_memory("startup")

    if args.pn_order < 0:
        raise ValueError("--pn-order must be nonnegative.")
    if args.scatter_legendre_order < 1 or args.scatter_legendre_order > 4:
        raise ValueError("--scatter-legendre-order must be between 1 and 4 for the current scattering kernel.")

    args.legendre_order = args.scatter_legendre_order
    data = snref.load_problem(args)
    snref.report_memory("after load_problem")
    solver = BuckledPNReference(data, args.pn_order)
    if args.fixed_b2 is None:
        b2, k, moments = solver.find_critical_b2(args)
    else:
        floor = solver.negative_b2_floor(args.negative_b2_safety)
        if args.fixed_b2 <= floor:
            raise ValueError(f"--fixed-b2={args.fixed_b2:g} is below the nonsingular PN negative-B2 floor {floor:g}.")
        b2 = args.fixed_b2
        k, moments = solver.k_of_b2(b2, args)
    snref.report_memory("after PN solve")
    collapsed = snref.collapse_results(data, moments)
    snref.report_memory("after few-group collapse")
    write_outputs(Path(args.outdir), b2, k, data, moments, collapsed, args)
    snref.report_memory("after writing outputs")
    print(f"PN order     = {args.pn_order}")
    print(f"Critical B2  = {b2:.12e}")
    print(f"k(B2)       = {k:.12e}")
    print(f"Wrote outputs to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
