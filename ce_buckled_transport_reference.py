#!/usr/bin/env python3
"""Buckled continuous-energy/fine-group transport reference.

This script solves the homogeneous Fourier-mode transport problem

    (Sigma_t(E) + i B mu) psi(mu, E)
      = scatter[psi] + 0.5 chi(E) F

on the same fine lethargy grid and with the same H/U number densities used by
``hf_sd_sp3.py.py``.  For a fixed B2, a unit fission source is applied and the
resulting fission production is k(B2).  A scalar bisection search then finds the
critical B2 for which k(B2) = 1.

The output is intended as a transport reference for comparing the two SP3
condensation orderings, not as another SP3 solve.
"""

from __future__ import annotations

from input import *

class BuckledSNReference:
    def __init__(self, data, sn_order, legendre_order):
        self.data = data
        self.sn_order = sn_order
        self.legendre_order = legendre_order
        self.mu, self.w = np.polynomial.legendre.leggauss(sn_order)
        self.P = np.vstack([np.polynomial.legendre.Legendre.basis(ell)(self.mu) for ell in range(legendre_order)])
        self.last_moments = None
        self.last_b2 = None

    def negative_b2_floor(self, safety):
        max_positive_mu = float(np.max(self.mu))
        if max_positive_mu <= 0.0:
            return -math.inf
        beta_limit = float(np.min(self.data.sigma_t)) / max_positive_mu
        return -(safety * beta_limit) ** 2

    def _inv_denom(self, b2):
        B = np.sqrt(np.complex128(b2))
        denom = self.data.sigma_t[None, :] + 1j * B * self.mu[:, None]
        if np.any(np.abs(denom) < 1.0e-13):
            raise FloatingPointError(f"Streaming denominator nearly singular for B2={b2:g}.")
        return 1.0 / denom

    def response_coeffs(self, inv_denom):
        coeffs = np.empty((self.legendre_order, self.legendre_order, self.data.sigma_t.size), dtype=np.complex128)
        for ell in range(self.legendre_order):
            for m in range(self.legendre_order):
                coeffs[ell, m] = (
                    0.5
                    * (2 * m + 1)
                    * np.einsum("a,a,a,ag->g", self.w, self.P[ell], self.P[m], inv_denom, optimize=True)
                )
        return coeffs

    def response(self, angle_source, inv_denom):
        psi = angle_source * inv_denom
        return np.einsum("m,lm,mg->lg", self.w, self.P, psi, optimize=True)

    def scatter_response(self, moments, coeffs):
        out = np.zeros_like(moments)
        for m in range(self.legendre_order):
            q_m = self.data.scatter[m] @ moments[m]
            out += coeffs[:, m, :] * q_m[None, :]
        return out

    def fission_response(self, inv_denom):
        source = 0.5 * self.data.chi[None, :]
        return self.response(source, inv_denom)

    def local_preconditioner(self, coeffs):
        G = self.data.sigma_t.size
        L = self.legendre_order
        blocks = np.zeros((G, L, L), dtype=np.complex128)
        diag_scatter = np.vstack([np.diag(self.data.scatter[m]) for m in range(L)])
        for ell in range(L):
            blocks[:, ell, ell] = 1.0
            for m in range(L):
                blocks[:, ell, m] -= coeffs[ell, m] * diag_scatter[m]

        def apply(vec):
            rhs = vec.reshape(L, G).T
            solved = np.linalg.solve(blocks, rhs)
            return solved.T.ravel()

        return LinearOperator((L * G, L * G), matvec=apply, dtype=np.complex128)

    def solve_fixed_source(self, b2, tol, max_iters, gmres_restart, linear_solver, verbose):
        report_memory(f"before fixed-source solve B2={b2:.8e}")
        inv_denom = self._inv_denom(b2)
        coeffs = self.response_coeffs(inv_denom)
        fixed = self.fission_response(inv_denom)
        if self.last_moments is not None and self.last_moments.shape == fixed.shape:
            moments = self.last_moments.copy()
        else:
            moments = fixed.copy()

        shape = fixed.shape
        size = fixed.size

        def matvec(vec):
            trial = vec.reshape(shape)
            out = trial - self.scatter_response(trial, coeffs)
            return out.ravel()

        op = LinearOperator((size, size), matvec=matvec, dtype=np.complex128)
        preconditioner = self.local_preconditioner(coeffs)
        report_memory(f"after preconditioner B2={b2:.8e}")
        residuals = []

        def callback(residual_norm):
            if np.isscalar(residual_norm):
                residuals.append(float(residual_norm))
            else:
                residuals.append(float("nan"))
            if verbose and (len(residuals) == 1 or len(residuals) % 25 == 0):
                print(f"  {linear_solver} iter {len(residuals):5d}: callback={residuals[-1]:.4e}")

        def add_tolerance_kwargs(kwargs, solver_fn):
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

        rhs = fixed.ravel()
        x0 = moments.ravel()
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
        report_memory(f"after {linear_solver} B2={b2:.8e}")

        moments = sol.reshape(shape)
        residual_vec = matvec(sol) - fixed.ravel()
        err = float(np.linalg.norm(residual_vec) / max(np.linalg.norm(fixed.ravel()), 1.0e-300))
        it = len(residuals) if residuals else (max_iters if info > 0 else 1)
        if info != 0:
            raise RuntimeError(f"{linear_solver} did not converge for B2={b2:g}; info={info}, residual={err:.3e}")

        self.last_moments = moments
        self.last_b2 = b2
        k = np.dot(self.data.nu_sigma_f, moments[0]).real
        return float(k), moments, it, err

    def k_of_b2(self, b2, args):
        k, moments, iters, err = self.solve_fixed_source(
            b2=b2,
            tol=args.inner_tol,
            max_iters=args.inner_max_iters,
            gmres_restart=args.gmres_restart,
            linear_solver=args.linear_solver,
            verbose=args.verbose_inner,
        )
        print(f"B2={b2:.8e}, k={k:.10e}, fixed-source iters={iters}, residual={err:.3e}")
        return k, moments

    def find_critical_b2(self, args):
        negative_floor = self.negative_b2_floor(args.negative_b2_safety)
        print(
            f"Negative-B2 transport floor = {negative_floor:.8e} "
            f"(safety={args.negative_b2_safety:g}); values below this cross a streaming pole."
        )
        if args.initial_b2 <= negative_floor:
            raise ValueError(
                f"Initial B2={args.initial_b2:g} is below the physical negative-buckling floor "
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
                    f"Could not bracket critical B2 below initial guess {b_start:g} before reaching the "
                    f"negative-buckling transport floor {negative_floor:g}. No valid negative transport root "
                    f"was found in the nonsingular interval."
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
            print(f"B2 search {it:3d}: lo={lo:.6e}, hi={hi:.6e}, mid={mid:.6e}, k-1={fmid:.4e}")
            if abs(fmid) < args.b2_k_tol or width < args.b2_abs_tol:
                return best_b2, best_k, best_moments
            if fmid * flo > 0.0:
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid

        return best_b2, best_k, best_moments


def write_outputs(
    outdir,
    b2,
    k,
    data,
    moments,
    collapsed,
    args,
):
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "ce_buckled_transport_reference.npz",
        run_mode=("critical_search" if args.fixed_b2 is None else "fixed_b2"),
        B2=b2,
        is_critical_search=(args.fixed_b2 is None),
        critical_B2=b2,
        k_at_B2=k,
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
        "critical_B2": b2,
        "B2": b2,
        "k_at_B2": k,
        "run_mode": "critical_search" if args.fixed_b2 is None else "fixed_b2",
        "is_critical_search": args.fixed_b2 is None,
        "fine_groups": int(args.fine_groups),
        "sn_order": int(args.sn_order),
        "legendre_order": int(args.legendre_order),
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
        "memory_profile": args.memory_profile and not args.no_memory_profile,
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
    density = collapsed["coarse_phi0_density"].real[::-1]
    density_sum = np.sum(density)
    if abs(density_sum) > 1.0e-300:
        density = density / density_sum
    y = np.r_[density, density[-1]]
    plt.figure(figsize=(9, 5.5))
    plt.step(edges, y, where="post", label=fr"$\phi_0$, $B^2={b2:.6e}$")
    plt.xscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Few-group scalar flux per lethargy")
    plt.title(f"Buckled S{args.sn_order} transport reference, k={k:.8f}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "few_group_spectrum.png", dpi=200)
    plt.close()


def parse_args(argv=None):
    return ce_args()


def main(argv=None):
    args = parse_args(argv)
    set_memory_profile(args.memory_profile and not args.no_memory_profile)
    report_memory("startup")
    if args.legendre_order < 1 or args.legendre_order > 4:
        raise ValueError("--legendre-order must be between 1 and 4 for the current scattering kernel.")
    if args.sn_order < 2 or args.sn_order % 2:
        raise ValueError("--sn-order must be an even integer >= 2.")

    data = load_problem(args)
    report_memory("after load_problem")
    solver = BuckledSNReference(data, args.sn_order, args.legendre_order)
    if args.fixed_b2 is None:
        b2, k, moments = solver.find_critical_b2(args)
    else:
        floor = solver.negative_b2_floor(args.negative_b2_safety)
        if args.fixed_b2 <= floor:
            raise ValueError(f"--fixed-b2={args.fixed_b2:g} is below the nonsingular negative-B2 floor {floor:g}.")
        b2 = args.fixed_b2
        k, moments = solver.k_of_b2(b2, args)
    report_memory("after critical B2 search")
    collapsed = collapse_results(data, moments)
    report_memory("after few-group collapse")
    write_outputs(Path(args.outdir), b2, k, data, moments, collapsed, args)
    report_memory("after writing outputs")
    print(f"Critical B2 = {b2:.12e}")
    print(f"k(B2)      = {k:.12e}")
    print(f"Wrote outputs to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
