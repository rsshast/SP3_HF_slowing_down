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

import argparse
import csv
import inspect
import json
import math
import os
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, lgmres


MEMORY_PROFILE = True


def _linux_status_memory_gb() -> tuple[float | None, float | None]:
    status = Path("/proc/self/status")
    if not status.exists():
        return None, None

    rss_gb = None
    hwm_gb = None
    for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            rss_gb = float(line.split()[1]) / (1024.0 * 1024.0)
        elif line.startswith("VmHWM:"):
            hwm_gb = float(line.split()[1]) / (1024.0 * 1024.0)
    return rss_gb, hwm_gb


def report_memory(label: str) -> None:
    if not MEMORY_PROFILE:
        return
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    current, peak = tracemalloc.get_traced_memory()
    rss_gb, hwm_gb = _linux_status_memory_gb()
    rss = f"{rss_gb:.3f} GB" if rss_gb is not None else "n/a"
    hwm = f"{hwm_gb:.3f} GB" if hwm_gb is not None else "n/a"
    print(
        f"[mem] {label}: rss={rss}, hwm={hwm}, "
        f"py_current={current / 1e9:.3f} GB, py_peak={peak / 1e9:.3f} GB"
    )


@dataclass
class MaterialData:
    u_bounds: np.ndarray
    e_bounds: np.ndarray
    e_mid: np.ndarray
    du: np.ndarray
    sigma_t: np.ndarray
    nu_sigma_f: np.ndarray
    chi: np.ndarray
    scatter: np.ndarray
    group_edges_e: np.ndarray
    group_names: list[str]


def read_table(path: Path, delimiter: str | None = None, skip_header: int = 0) -> np.ndarray:
    return np.genfromtxt(path, delimiter=delimiter, names=True, skip_header=skip_header)


def read_plain_table(path: Path, delimiter: str | None = None, skiprows: int = 0) -> np.ndarray:
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)


def alpha_fn(A: float) -> float:
    return ((A - 1.0) / (A + 1.0)) ** 2


def lga_fn(alpha: float) -> float:
    return -math.log(alpha) if alpha != 0.0 else math.inf


def group_bound(u_bounds: np.ndarray, A: float, g: int, lga: float) -> int:
    return u_bounds.size if A == 1.0 else 1 + int(np.searchsorted(u_bounds, u_bounds[g] + lga))


def gmax_vec_fn(u_bounds: np.ndarray, A: float, lga: float) -> np.ndarray:
    out = np.zeros_like(u_bounds, dtype=np.int64)
    for g in range(u_bounds.size):
        out[g] = group_bound(u_bounds, A, g, lga)
    return out


def interpolate_on_u(data: np.ndarray, e0: float, emin: float, u_grid: np.ndarray) -> np.ndarray:
    """Interpolate columns from tabulated energy data onto a lethargy grid."""
    data = np.asarray(data, dtype=float)
    data = data[(data[:, 0] <= e0) & (data[:, 0] >= emin)].copy()
    if data.size == 0:
        raise ValueError(f"No tabulated data between {emin:g} and {e0:g} eV.")
    data[:, 0] = np.log(e0 / data[:, 0])
    data = data[np.argsort(data[:, 0])]

    out = np.empty((u_grid.size, data.shape[1]), dtype=float)
    out[:, 0] = u_grid
    for j in range(1, data.shape[1]):
        out[:, j] = np.interp(u_grid, data[:, 0], data[:, j])
    return out


def group_average_from_bounds(values_at_bounds: np.ndarray, u_bounds: np.ndarray) -> np.ndarray:
    return 0.5 * (values_at_bounds[:-1] + values_at_bounds[1:])


def normalize_group_source(values_at_bounds: np.ndarray, u_bounds: np.ndarray) -> np.ndarray:
    avg = group_average_from_bounds(values_at_bounds, u_bounds)
    group_integral = avg * np.diff(u_bounds)
    total = float(np.sum(group_integral))
    if abs(total) < 1.0e-300:
        raise ValueError("Fission spectrum collapsed to zero on the requested grid.")
    return group_integral / total


def diagnose_vector(name: str, values: np.ndarray, e_mid: np.ndarray | None = None, max_items: int = 8) -> None:
    arr = np.asarray(values)
    finite = np.isfinite(arr)
    print(
        f"[diag] {name}: shape={arr.shape}, finite={int(np.count_nonzero(finite))}/{arr.size}, "
        f"nan={int(np.count_nonzero(np.isnan(arr)))}, inf={int(np.count_nonzero(np.isinf(arr)))}"
    )
    if np.any(finite):
        vals = arr[finite]
        print(f"[diag] {name}: min={np.min(vals):.8e}, max={np.max(vals):.8e}, sum={np.sum(vals):.8e}")
    bad = np.argwhere(~finite)
    for row in bad[:max_items]:
        idx = int(row[0])
        loc = f" idx={idx}"
        if e_mid is not None and idx < e_mid.size:
            loc += f", E_mid={e_mid[idx]:.8e} eV"
        print(f"[diag] {name}: bad{loc}, value={arr[idx]}")


def diagnose_scatter_matrix(
    name: str,
    scatter: np.ndarray,
    e_mid: np.ndarray,
    sig_s0: np.ndarray | None = None,
    max_items: int = 12,
) -> None:
    finite = np.isfinite(scatter)
    print(
        f"[diag] {name} scatter: shape={scatter.shape}, finite={int(np.count_nonzero(finite))}/{scatter.size}, "
        f"nan={int(np.count_nonzero(np.isnan(scatter)))}, inf={int(np.count_nonzero(np.isinf(scatter)))}"
    )
    if np.any(finite):
        vals = scatter[finite]
        print(f"[diag] {name} scatter: min={np.min(vals):.8e}, max={np.max(vals):.8e}, sum={np.sum(vals):.8e}")
    for ell in range(scatter.shape[0]):
        mat = scatter[ell]
        bad = np.argwhere(~np.isfinite(mat))
        finite_mat = np.where(np.isfinite(mat), mat, np.nan)
        if np.any(np.isfinite(finite_mat)):
            min_flat = int(np.nanargmin(finite_mat))
            max_flat = int(np.nanargmax(finite_mat))
            min_out, min_inc = np.unravel_index(min_flat, mat.shape)
            max_out, max_inc = np.unravel_index(max_flat, mat.shape)
            print(
                f"[diag] {name} ell={ell}: min entry={mat[min_out, min_inc]:.8e} "
                f"at exit={min_out}, incident={min_inc}, "
                f"E_exit={e_mid[min_out]:.8e} eV, E_inc={e_mid[min_inc]:.8e} eV"
            )
            print(
                f"[diag] {name} ell={ell}: max entry={mat[max_out, max_inc]:.8e} "
                f"at exit={max_out}, incident={max_inc}, "
                f"E_exit={e_mid[max_out]:.8e} eV, E_inc={e_mid[max_inc]:.8e} eV"
            )
        row_sums = np.sum(mat, axis=0)
        finite_rows = np.isfinite(row_sums)
        if np.any(finite_rows):
            print(
                f"[diag] {name} ell={ell}: transfer-column-sum min={np.min(row_sums[finite_rows]):.8e}, "
                f"max={np.max(row_sums[finite_rows]):.8e}"
            )
        if sig_s0 is not None and ell == 0:
            mismatch = row_sums - sig_s0
            finite_mismatch = np.isfinite(mismatch)
            if np.any(finite_mismatch):
                print(
                    f"[diag] {name} ell=0: row-sum-minus-sigma_s min={np.min(mismatch[finite_mismatch]):.8e}, "
                    f"max={np.max(mismatch[finite_mismatch]):.8e}"
                )
        for out_idx, inc_idx in bad[:max_items]:
            print(
                f"[diag] {name} bad scatter ell={ell}, exit={int(out_idx)}, incident={int(inc_idx)}, "
                f"E_exit={e_mid[out_idx]:.8e} eV, E_inc={e_mid[inc_idx]:.8e} eV, "
                f"value={mat[out_idx, inc_idx]}"
            )


def validate_scatter_matrix(name: str, scatter: np.ndarray, e_mid: np.ndarray, max_legendre_ratio: float) -> None:
    if not np.all(np.isfinite(scatter)):
        diagnose_scatter_matrix(name, scatter, e_mid)
        raise FloatingPointError(f"{name} scatter contains NaN or inf.")

    p0 = scatter[0]
    min_p0 = float(np.min(p0))
    if min_p0 < -1.0e-10:
        out_idx, inc_idx = np.unravel_index(int(np.argmin(p0)), p0.shape)
        diagnose_scatter_matrix(name, scatter, e_mid)
        raise ValueError(
            f"{name} P0 scatter has a negative entry {min_p0:.8e} at "
            f"exit={out_idx}, incident={inc_idx}, E_exit={e_mid[out_idx]:.8e} eV, "
            f"E_inc={e_mid[inc_idx]:.8e} eV."
        )

    if not np.isfinite(max_legendre_ratio) or scatter.shape[0] <= 1:
        return

    p0_scale = np.maximum(np.abs(p0), 1.0e-300)
    for ell in range(1, scatter.shape[0]):
        ratio = np.abs(scatter[ell]) / p0_scale
        max_ratio = float(np.max(ratio))
        if max_ratio > max_legendre_ratio:
            out_idx, inc_idx = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
            diagnose_scatter_matrix(name, scatter, e_mid)
            raise ValueError(
                f"{name} ell={ell} scatter is {max_ratio:.8e} times P0 at "
                f"exit={out_idx}, incident={inc_idx}, E_exit={e_mid[out_idx]:.8e} eV, "
                f"E_inc={e_mid[inc_idx]:.8e} eV. This usually indicates a bad higher-L scattering kernel."
            )


@njit(parallel=True, fastmath=True)
def gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, du, n_sub=8):
    """Generate downscatter Legendre transfer moments.

    Returns sigma_gtg[ell, incident, exit].  This is transposed later to the
    matrix convention scatter[ell, exit, incident].
    """
    G = boundaries.size
    den = (1.0 - alpha) * du
    lga = -math.log(alpha) if A != 1.0 else math.inf
    sigma_gtg = np.zeros((order, G - 1, G - 1))
    Am1 = A - 1.0
    Ap1 = A + 1.0
    K = 5.0 * A * A - 1.0

    for ell in range(order):
        for gp in prange(G - 1):
            x1 = boundaries[gp]
            x2 = boundaries[gp + 1]
            for g in range(gp, min(gmax_vec[gp], G - 1)):
                y1 = boundaries[g]
                y2 = boundaries[g + 1]
                c = max(x1, y1 - lga)
                length = x2 - c
                if length <= 0.0:
                    continue
                n_steps_base = int(math.ceil(length / du[gp]))
                if n_steps_base < 1:
                    n_steps_base = 1
                n_steps = n_steps_base * n_sub
                dx = length / n_steps
                acc = 0.0

                for s in range(n_steps):
                    xm = c + (s + 0.5) * dx
                    a = y1 if xm < y1 else xm
                    bx = xm + lga
                    b = y2 if bx > y2 else bx

                    if ell == 0:
                        val = math.exp(xm - a) - math.exp(xm - b)
                    elif ell == 1:
                        if A == 1.0:
                            val = (1.0 / 3.0) * Ap1 * (
                                math.exp(1.5 * (xm - a)) - math.exp(1.5 * (xm - b))
                            )
                        else:
                            val = (
                                Am1 * (math.exp(0.5 * (xm - b)) - math.exp(0.5 * (xm - a)))
                                + (1.0 / 3.0)
                                * Ap1
                                * (math.exp(1.5 * (xm - a)) - math.exp(1.5 * (xm - b)))
                            )
                    elif ell == 2:
                        if A == 1.0:
                            val = (
                                0.25
                                * (1.0 - 3.0 * A * A)
                                * (math.exp(xm - a) - math.exp(xm - b))
                                + 0.1875
                                * Ap1
                                * Ap1
                                * (math.exp(2.0 * (xm - a)) - math.exp(2.0 * (xm - b)))
                            )
                        else:
                            val = (
                                0.375 * Am1 * Am1 * (b - a)
                                + 0.25
                                * (1.0 - 3.0 * A * A)
                                * (math.exp(xm - a) - math.exp(xm - b))
                                + 0.1875
                                * Ap1
                                * Ap1
                                * (math.exp(2.0 * (xm - a)) - math.exp(2.0 * (xm - b)))
                            )
                    else:
                        if A == 1.0:
                            val = 0.0625 * (
                                2.0
                                * (Ap1**3)
                                * (math.exp(2.5 * (xm - a)) - math.exp(2.5 * (xm - b)))
                                - 2.0
                                * Ap1
                                * K
                                * (math.exp(1.5 * (xm - a)) - math.exp(1.5 * (xm - b)))
                            )
                        else:
                            val = 0.0625 * (
                                10.0
                                * (Am1**3)
                                * (math.exp(-0.5 * (xm - a)) - math.exp(-0.5 * (xm - b)))
                                + 2.0
                                * (Ap1**3)
                                * (math.exp(2.5 * (xm - a)) - math.exp(2.5 * (xm - b)))
                                + 6.0
                                * Am1
                                * K
                                * (math.exp(0.5 * (xm - a)) - math.exp(0.5 * (xm - b)))
                                - 2.0
                                * Ap1
                                * K
                                * (math.exp(1.5 * (xm - a)) - math.exp(1.5 * (xm - b)))
                            )
                    acc += val

                sigma_gtg[ell, gp, g] = sig_s0[gp] * acc * dx / den[gp]

    return sigma_gtg


@njit
def eval_p0_kernel(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(math.sqrt(E / kT))
        return (Sigma_fr / E_prime) * math.exp((E_prime - E) / kT) * math.erf(math.sqrt(E_prime / kT))

    kappa_min = math.sqrt(2.0 * m) * abs(math.sqrt(E_prime) - math.sqrt(E))
    kappa_max = math.sqrt(2.0 * m) * (math.sqrt(E_prime) + math.sqrt(E))
    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0
    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = -(A * m) / (2.0 * kT * kappa2) * (exp_inner**2)
        integral += w[i] * math.exp(exp_arg)
    integral *= half_width
    coeff = (
        Sigma_fr
        / (8.0 * m * E_prime * E)
        * (1.0 + 1.0 / A) ** 2
        * math.sqrt(E / E_prime)
        * math.sqrt((A * m) / (2.0 * math.pi * kT))
    )
    return coeff * integral


@njit
def eval_p1_kernel(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        mu_bar = math.sqrt(E / E_prime)
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(math.sqrt(E / kT)) * mu_bar
        return (
            (Sigma_fr / E_prime)
            * math.exp((E_prime - E) / kT)
            * math.erf(math.sqrt(E_prime / kT))
            * mu_bar
        )

    kappa_min = math.sqrt(2.0 * m) * abs(math.sqrt(E_prime) - math.sqrt(E))
    kappa_max = math.sqrt(2.0 * m) * (math.sqrt(E_prime) + math.sqrt(E))
    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0
    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa
        poly_part = E_prime + E - (kappa2 / (2.0 * m))
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = -(A * m) / (2.0 * kT * kappa2) * (exp_inner**2)
        integral += w[i] * poly_part * math.exp(exp_arg)
    integral *= half_width
    coeff = (
        Sigma_fr
        / (8.0 * m * E_prime * E)
        * (1.0 + 1.0 / A) ** 2
        * math.sqrt(E / E_prime)
        * math.sqrt((A * m) / (2.0 * math.pi * kT))
    )
    return coeff * integral


@njit
def eval_p2_kernel(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        mu_bar = math.sqrt(E / E_prime)
        p2_val = 0.5 * (3.0 * mu_bar**2 - 1.0)
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(math.sqrt(E / kT)) * p2_val
        return (
            (Sigma_fr / E_prime)
            * math.exp((E_prime - E) / kT)
            * math.erf(math.sqrt(E_prime / kT))
            * p2_val
        )

    kappa_min = math.sqrt(2.0 * m) * abs(math.sqrt(E_prime) - math.sqrt(E))
    kappa_max = math.sqrt(2.0 * m) * (math.sqrt(E_prime) + math.sqrt(E))
    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0
    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa
        poly_inner = (E_prime + E - kappa2 / (2.0 * m)) / (2.0 * math.sqrt(E_prime * E))
        poly_part = 3.0 * poly_inner**2 - 1.0
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = -(A * m) / (2.0 * kT * kappa2) * (exp_inner**2)
        integral += w[i] * poly_part * math.exp(exp_arg)
    integral *= half_width
    coeff = (
        Sigma_fr
        / (8.0 * m * math.sqrt(E_prime * E))
        * (1.0 + 1.0 / A) ** 2
        * math.sqrt(E / E_prime)
        * math.sqrt((A * m) / (2.0 * math.pi * kT))
    )
    return coeff * integral


@njit
def eval_p3_kernel(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        mu_bar = math.sqrt(E / E_prime)
        p3_val = 0.5 * (5.0 * mu_bar**3 - 3.0 * mu_bar)
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(math.sqrt(E / kT)) * p3_val
        return (
            (Sigma_fr / E_prime)
            * math.exp((E_prime - E) / kT)
            * math.erf(math.sqrt(E_prime / kT))
            * p3_val
        )

    kappa_min = math.sqrt(2.0 * m) * abs(math.sqrt(E_prime) - math.sqrt(E))
    kappa_max = math.sqrt(2.0 * m) * (math.sqrt(E_prime) + math.sqrt(E))
    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0
    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa
        poly_inner = (E_prime + E - kappa2 / (2.0 * m)) / (2.0 * math.sqrt(E_prime * E))
        poly_part = 5.0 * poly_inner**3 - 3.0 * poly_inner
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = -(A * m) / (2.0 * kT * kappa2) * (exp_inner**2)
        integral += w[i] * poly_part * math.exp(exp_arg)
    integral *= half_width
    coeff = (
        Sigma_fr
        / (8.0 * m * math.sqrt(E_prime * E))
        * (1.0 + 1.0 / A) ** 2
        * math.sqrt(E / E_prime)
        * math.sqrt((A * m) / (2.0 * math.pi * kT))
    )
    return coeff * integral


@njit(parallel=True)
def build_thermal_to_all(ell, E_mid, sigma_fr, gmax_vec, A, kT, m, x, w, dE, insert_idx, exit_start_idx):
    G = len(E_mid)
    N_th = G - insert_idx
    sigma_th_all = np.zeros((N_th, G))
    for i in prange(N_th):
        gp = insert_idx + i
        sig_val = sigma_fr[gp]
        E_prime = E_mid[gp]
        gmax = min(gmax_vec[gp], G)
        for g in range(exit_start_idx, gmax):
            E_exit = E_mid[g]
            if ell == 0:
                val = eval_p0_kernel(E_prime, E_exit, A, sig_val, kT, m, x, w)
            elif ell == 1:
                val = eval_p1_kernel(E_prime, E_exit, A, sig_val, kT, m, x, w)
            elif ell == 2:
                val = eval_p2_kernel(E_prime, E_exit, A, sig_val, kT, m, x, w)
            else:
                val = eval_p3_kernel(E_prime, E_exit, A, sig_val, kT, m, x, w)
            sigma_th_all[i, g] = val * dE[g]
    return sigma_th_all


def build_scatter_matrix(
    A: float,
    sig_s0: np.ndarray,
    sigma_fr: np.ndarray,
    order: int,
    u_bounds: np.ndarray,
    e_bounds: np.ndarray,
    kT: float,
    nquad: int,
    upscatter: bool,
    p0_only_upscatter: bool,
    thermal_cutoff_ev: float,
) -> np.ndarray:
    du = np.diff(u_bounds)
    alpha = alpha_fn(A)
    gmax_vec = gmax_vec_fn(u_bounds, A, lga_fn(alpha))
    sigma_gtg = gen_sig_sn_gtg(A, sig_s0, order, u_bounds, gmax_vec, alpha, du)

    if upscatter:
        e_mid = 0.5 * (e_bounds[:-1] + e_bounds[1:])
        thermal = np.where(e_mid <= thermal_cutoff_ev)[0]
        if thermal.size:
            insert_idx = int(thermal[0])
            exit_start_idx = insert_idx
            x, w = np.polynomial.legendre.leggauss(nquad)
            dE = np.abs(np.diff(e_bounds))
            for ell in range(order):
                if p0_only_upscatter and ell > 0:
                    continue
                sigma_gtg[ell, insert_idx:, :] = build_thermal_to_all(
                    ell, e_mid, sigma_fr, gmax_vec, A, kT, 1.0, x, w, dE, insert_idx, exit_start_idx
                )

            scaled_p0_rows = 0
            min_p0_scale = 1.0
            for gp in range(e_mid.size):
                sigma_gtg[0, gp, gp] = 0.0
                offdiag_sum = np.sum(sigma_gtg[0, gp, :])
                if offdiag_sum <= sig_s0[gp]:
                    sigma_gtg[0, gp, gp] = sig_s0[gp] - offdiag_sum
                elif offdiag_sum > 0.0:
                    scale = sig_s0[gp] / offdiag_sum
                    sigma_gtg[0, gp, :] *= scale
                    sigma_gtg[0, gp, gp] = 0.0
                    scaled_p0_rows += 1
                    min_p0_scale = min(min_p0_scale, scale)
                else:
                    sigma_gtg[0, gp, gp] = sig_s0[gp]
            if scaled_p0_rows:
                print(
                    f"[diag] P0 thermal balance scaled {scaled_p0_rows} incident rows; "
                    f"minimum scale={min_p0_scale:.8e}"
                )

    return np.transpose(sigma_gtg, (0, 2, 1))


def thermal_upscatter_enabled(args: argparse.Namespace, name: str) -> bool:
    if args.no_upscatter:
        return False
    enabled = {item.strip().lower() for item in args.thermal_upscatter_nuclides.split(",") if item.strip()}
    aliases = {
        "H-1": {"h", "h1", "h-1", "hydrogen"},
        "U-238": {"u", "u238", "u-238", "uranium"},
    }
    return bool(enabled & aliases[name])


def load_problem(args: argparse.Namespace) -> MaterialData:
    data_dir = Path(args.data_dir)
    group_table = read_plain_table(data_dir / args.group_structure)
    group_edges_e = np.asarray(group_table[:, 1], dtype=float)
    group_edges_e = group_edges_e[np.argsort(group_edges_e)[::-1]]
    e0 = float(args.e0 or group_edges_e[0])
    emin = float(args.emin or group_edges_e[-1])

    u_bounds = np.linspace(0.0, math.log(e0 / emin), args.fine_groups + 1)
    e_bounds = e0 * np.exp(-u_bounds)
    e_mid = 0.5 * (e_bounds[:-1] + e_bounds[1:])

    h1 = read_table(data_dir / "xs_h1_T293k.txt", delimiter="\t")
    u238 = read_table(data_dir / "xs_u238_T293k.txt", delimiter="\t")
    chi35 = read_table(data_dir / "chi_u235.txt", delimiter="\t")
    sig_f = read_plain_table(data_dir / "xs_u238_fission.csv", delimiter=",", skiprows=1)
    sig_fr_u = read_plain_table(data_dir / "sigma_fr_U.csv", delimiter=";", skiprows=1)
    sig_fr_h = read_plain_table(data_dir / "sigma_fr_H.csv", delimiter=";", skiprows=1)
    report_memory("after reading raw input tables")

    h = np.column_stack((h1["E"], h1["sigma_t"], h1["sigma_s"]))
    u = np.column_stack((u238["E"], u238["sigma_t"], u238["sigma_s"]))
    chi = np.column_stack((chi35["E"], chi35["chi"]))

    h_i = interpolate_on_u(h, e0, emin, u_bounds)
    u_i = interpolate_on_u(u, e0, emin, u_bounds)
    chi_i = interpolate_on_u(chi, e0, emin, u_bounds)
    sig_f_i = interpolate_on_u(sig_f, e0, emin, u_bounds)
    sig_fr_u_i = interpolate_on_u(sig_fr_u, e0, emin, u_bounds)
    sig_fr_h_i = interpolate_on_u(sig_fr_h, e0, emin, u_bounds)

    sig_t_h = args.NH * group_average_from_bounds(h_i[:, 1], u_bounds)
    sig_s_h = args.NH * group_average_from_bounds(h_i[:, 2], u_bounds)
    sig_t_u = args.NU * group_average_from_bounds(u_i[:, 1], u_bounds)
    sig_s_u = args.NU * group_average_from_bounds(u_i[:, 2], u_bounds)
    sigma_t = sig_t_h + sig_t_u
    nu_sigma_f = args.nu * args.NU * group_average_from_bounds(sig_f_i[:, 1], u_bounds)
    chi_g = normalize_group_source(chi_i[:, 1], u_bounds)
    sigma_fr_u = args.NU * group_average_from_bounds(sig_fr_u_i[:, 1], u_bounds)
    sigma_fr_h = args.NH * group_average_from_bounds(sig_fr_h_i[:, 1], u_bounds)
    report_memory("after interpolating fine-grid XS")

    if args.diagnostics:
        print("[diag] Fine-grid/input summaries")
        for name, values in (
            ("sigma_t_h", sig_t_h),
            ("sigma_s_h", sig_s_h),
            ("sigma_fr_h", sigma_fr_h),
            ("sigma_t_u", sig_t_u),
            ("sigma_s_u", sig_s_u),
            ("sigma_fr_u", sigma_fr_u),
            ("sigma_t_total", sigma_t),
            ("nu_sigma_f", nu_sigma_f),
            ("chi", chi_g),
        ):
            diagnose_vector(name, values, e_mid)

    print(f"Building U-238 scattering matrix, A=238, G={args.fine_groups}")
    scatter_u = build_scatter_matrix(
        238.0,
        sig_s_u,
        sigma_fr_u,
        args.legendre_order,
        u_bounds,
        e_bounds,
        args.kT,
        args.kernel_quad,
        thermal_upscatter_enabled(args, "U-238"),
        args.p0_only_upscatter,
        args.thermal_upscatter_cutoff_ev,
    )
    report_memory("after U-238 scattering matrix")
    if args.diagnostics or not np.all(np.isfinite(scatter_u)):
        diagnose_scatter_matrix("U-238", scatter_u, e_mid, sig_s_u)
    if not args.no_scatter_validation:
        validate_scatter_matrix("U-238", scatter_u, e_mid, args.max_legendre_ratio)
    print("Building H-1 scattering matrix, A=1")
    scatter_h = build_scatter_matrix(
        1.0,
        sig_s_h,
        sigma_fr_h,
        args.legendre_order,
        u_bounds,
        e_bounds,
        args.kT,
        args.kernel_quad,
        thermal_upscatter_enabled(args, "H-1"),
        args.p0_only_upscatter,
        args.thermal_upscatter_cutoff_ev,
    )
    report_memory("after H-1 scattering matrix")
    if args.diagnostics or not np.all(np.isfinite(scatter_h)):
        diagnose_scatter_matrix("H-1", scatter_h, e_mid, sig_s_h)
    if not args.no_scatter_validation:
        validate_scatter_matrix("H-1", scatter_h, e_mid, args.max_legendre_ratio)

    scatter = scatter_u + scatter_h
    report_memory("after total scattering matrix")
    if args.diagnostics or not np.all(np.isfinite(scatter)):
        diagnose_scatter_matrix("total", scatter, e_mid, sig_s_u + sig_s_h)
    if not args.no_scatter_validation:
        validate_scatter_matrix("total", scatter, e_mid, args.max_legendre_ratio)

    return MaterialData(
        u_bounds=u_bounds,
        e_bounds=e_bounds,
        e_mid=e_mid,
        du=np.diff(u_bounds),
        sigma_t=sigma_t,
        nu_sigma_f=nu_sigma_f,
        chi=chi_g,
        scatter=scatter,
        group_edges_e=group_edges_e,
        group_names=[f"g{i + 1}" for i in range(group_edges_e.size - 1)],
    )


class BuckledSNReference:
    def __init__(self, data: MaterialData, sn_order: int, legendre_order: int):
        self.data = data
        self.sn_order = sn_order
        self.legendre_order = legendre_order
        self.mu, self.w = np.polynomial.legendre.leggauss(sn_order)
        self.P = np.vstack([np.polynomial.legendre.Legendre.basis(ell)(self.mu) for ell in range(legendre_order)])
        self.last_moments: np.ndarray | None = None
        self.last_b2: float | None = None

    def negative_b2_floor(self, safety: float) -> float:
        max_positive_mu = float(np.max(self.mu))
        if max_positive_mu <= 0.0:
            return -math.inf
        beta_limit = float(np.min(self.data.sigma_t)) / max_positive_mu
        return -(safety * beta_limit) ** 2

    def _inv_denom(self, b2: float) -> np.ndarray:
        B = np.sqrt(np.complex128(b2))
        denom = self.data.sigma_t[None, :] + 1j * B * self.mu[:, None]
        if np.any(np.abs(denom) < 1.0e-13):
            raise FloatingPointError(f"Streaming denominator nearly singular for B2={b2:g}.")
        return 1.0 / denom

    def response_coeffs(self, inv_denom: np.ndarray) -> np.ndarray:
        coeffs = np.empty((self.legendre_order, self.legendre_order, self.data.sigma_t.size), dtype=np.complex128)
        for ell in range(self.legendre_order):
            for m in range(self.legendre_order):
                coeffs[ell, m] = (
                    0.5
                    * (2 * m + 1)
                    * np.einsum("a,a,a,ag->g", self.w, self.P[ell], self.P[m], inv_denom, optimize=True)
                )
        return coeffs

    def response(self, angle_source: np.ndarray, inv_denom: np.ndarray) -> np.ndarray:
        psi = angle_source * inv_denom
        return np.einsum("m,lm,mg->lg", self.w, self.P, psi, optimize=True)

    def scatter_response(self, moments: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(moments)
        for m in range(self.legendre_order):
            q_m = self.data.scatter[m] @ moments[m]
            out += coeffs[:, m, :] * q_m[None, :]
        return out

    def fission_response(self, inv_denom: np.ndarray) -> np.ndarray:
        source = 0.5 * self.data.chi[None, :]
        return self.response(source, inv_denom)

    def local_preconditioner(self, coeffs: np.ndarray) -> LinearOperator:
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

    def solve_fixed_source(
        self,
        b2: float,
        tol: float,
        max_iters: int,
        gmres_restart: int,
        linear_solver: str,
        verbose: bool,
    ) -> tuple[float, np.ndarray, int, float]:
        report_memory(f"before fixed-source solve B2={b2:.8e}")
        inv_denom = self._inv_denom(b2)
        coeffs = self.response_coeffs(inv_denom)
        fixed = self.fission_response(inv_denom)
        if self.last_moments is not None and self.last_moments.shape == fixed.shape:
            moments = self.last_moments.copy()
        else:
            moments = fixed.copy()

        for name, arr in (
            ("sigma_t", self.data.sigma_t),
            ("scatter", self.data.scatter),
            ("nu_sigma_f", self.data.nu_sigma_f),
            ("chi", self.data.chi),
            ("fixed source response", fixed),
            ("response coefficients", coeffs),
        ):
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"{name} contains NaN or inf before the B2={b2:g} solve.")

        shape = fixed.shape
        size = fixed.size

        def matvec(vec):
            trial = vec.reshape(shape)
            out = trial - self.scatter_response(trial, coeffs)
            return out.ravel()

        op = LinearOperator((size, size), matvec=matvec, dtype=np.complex128)
        preconditioner = self.local_preconditioner(coeffs)
        report_memory(f"after preconditioner B2={b2:.8e}")
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

    def k_of_b2(self, b2: float, args: argparse.Namespace) -> tuple[float, np.ndarray]:
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

    def find_critical_b2(self, args: argparse.Namespace) -> tuple[float, float, np.ndarray]:
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


def fine_to_coarse_indices(e_mid: np.ndarray, group_edges_e: np.ndarray) -> list[np.ndarray]:
    edges = group_edges_e
    groups = []
    for i in range(edges.size - 1):
        hi = edges[i]
        lo = edges[i + 1]
        mask = (e_mid <= hi) & (e_mid >= lo)
        groups.append(np.where(mask)[0])
    return groups


def collapse_vector(xs: np.ndarray, weight: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(groups), dtype=np.complex128)
    for i, idx in enumerate(groups):
        den = np.sum(weight[idx])
        out[i] = np.sum(xs[idx] * weight[idx]) / den if abs(den) > 1.0e-250 else 0.0
    return out


def collapse_scatter(scatter: np.ndarray, moments: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    L = scatter.shape[0]
    Gc = len(groups)
    out = np.zeros((L, Gc, Gc), dtype=np.complex128)
    for ell in range(L):
        w = moments[ell]
        for j, inc in enumerate(groups):
            den = np.sum(w[inc])
            if abs(den) <= 1.0e-250:
                continue
            weighted_inc = w[inc]
            for i, outg in enumerate(groups):
                block = scatter[ell][np.ix_(outg, inc)]
                out[ell, i, j] = np.sum(block * weighted_inc[None, :]) / den
    return out


def collapse_results(data: MaterialData, moments: np.ndarray) -> dict[str, np.ndarray]:
    groups = fine_to_coarse_indices(data.e_mid, data.group_edges_e)
    phi0 = moments[0]
    phi0_real = np.real_if_close(phi0).real
    coarse_phi0 = np.array([np.sum(phi0[idx]) for idx in groups], dtype=np.complex128)
    coarse_du = np.array([math.log(data.group_edges_e[i] / data.group_edges_e[i + 1]) for i in range(len(groups))])
    chi = np.array([np.sum(data.chi[idx]) for idx in groups], dtype=float)

    return {
        "group_edges_e": data.group_edges_e,
        "coarse_phi0": coarse_phi0,
        "coarse_phi0_density": coarse_phi0 / coarse_du,
        "coarse_du": coarse_du,
        "chi_few": chi / np.sum(chi),
        "Sigma_t": collapse_vector(data.sigma_t, phi0_real, groups),
        "nuSigma_f": collapse_vector(data.nu_sigma_f, phi0_real, groups),
        "Sigma_s_moment_weighted": collapse_scatter(data.scatter, moments, groups),
    }


def write_outputs(
    outdir: Path,
    b2: float,
    k: float,
    data: MaterialData,
    moments: np.ndarray,
    collapsed: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
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
        "memory_profile": MEMORY_PROFILE,
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--group-structure", default=os.getenv("CE_GROUP_STRUCTURE", "wims69.txt"))
    p.add_argument("--outdir", default=os.getenv("CE_OUTDIR", "results/ce_transport_reference"))
    p.add_argument("--fine-groups", type=int, default=int(os.getenv("CE_FINE_GROUPS", "12000")))
    p.add_argument("--sn-order", type=int, default=int(os.getenv("CE_SN_ORDER", "32")))
    p.add_argument("--legendre-order", type=int, default=int(os.getenv("CE_LEGENDRE_ORDER", "4")))
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
    p.add_argument("--inner-tol", type=float, default=float(os.getenv("CE_INNER_TOL", "1e-8")))
    p.add_argument("--inner-max-iters", type=int, default=int(os.getenv("CE_INNER_MAX_ITERS", "5000")))
    p.add_argument("--gmres-restart", type=int, default=int(os.getenv("CE_GMRES_RESTART", "80")))
    p.add_argument(
        "--linear-solver",
        choices=("lgmres", "bicgstab", "gmres"),
        default=os.getenv("CE_LINEAR_SOLVER", "lgmres"),
        help="Krylov method for the fixed-source solve. Default: lgmres.",
    )
    p.add_argument("--initial-b2", type=float, default=float(os.getenv("CE_INITIAL_B2", "-0.01")))
    p.add_argument("--fixed-b2", type=float, default=None, help="Skip B2 search and solve CE transport at this B2.")
    p.add_argument("--initial-b2-step", type=float, default=float(os.getenv("CE_INITIAL_B2_STEP", "0.01")))
    p.add_argument("--max-abs-b2", type=float, default=float(os.getenv("CE_MAX_ABS_B2", "1.0")))
    p.add_argument(
        "--negative-b2-safety",
        type=float,
        default=float(os.getenv("CE_NEGATIVE_B2_SAFETY", "0.98")),
        help="Fraction of the negative-buckling streaming-pole limit allowed during B2 search. Default: 0.98.",
    )
    p.add_argument("--b2-k-tol", type=float, default=float(os.getenv("CE_B2_K_TOL", "1e-6")))
    p.add_argument("--b2-abs-tol", type=float, default=float(os.getenv("CE_B2_ABS_TOL", "1e-8")))
    p.add_argument("--b2-max-iters", type=int, default=int(os.getenv("CE_B2_MAX_ITERS", "80")))
    p.add_argument("--verbose-inner", action="store_true")
    p.add_argument("--no-memory-profile", action="store_true", help="Disable memory usage reports.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    global MEMORY_PROFILE
    args = parse_args(argv)
    MEMORY_PROFILE = not args.no_memory_profile
    if MEMORY_PROFILE and not tracemalloc.is_tracing():
        tracemalloc.start()
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
