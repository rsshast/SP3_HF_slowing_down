"""Input deck and shared data helpers for the CE, PN/SN comparison, and HF SP3 scripts."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import copy
import csv
import gc
import inspect
import json
import math
import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import psutil
import torch
from numba import njit, prange
from scipy.linalg import null_space
from scipy.integrate import simpson, quad
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, lgmres


data_dir = Path(os.getenv("DATA_DIR", os.getenv("SP3_DATA_DIR", os.getenv("CE_DATA_DIR", "data"))))
group_structure = os.getenv("GROUP_STRUCTURE", os.getenv("SP3_GROUP_STRUCTURE", os.getenv("CE_GROUP_STRUCTURE", "wims69.txt")))
nbins = int(os.getenv("NBINS", os.getenv("SP3_NBINS", os.getenv("CE_FINE_GROUPS", "12000"))))

NH = float(os.getenv("NH", os.getenv("SP3_NH", "0.10")))
NU = float(os.getenv("NU", os.getenv("SP3_NU", "0.02")))
NO = float(os.getenv("NO", os.getenv("SP3_NO", "0.10")))
nu_bar = float(os.getenv("NU_BAR", os.getenv("SP3_NU_BAR", "2.43")))

AH = 1.0
AU = 238.0
AO = 16.0
neutron_mass_amu = 1.0

temperature_k = float(os.getenv("TEMPERATURE_K", os.getenv("SP3_TEMPERATURE_K", "293.15")))
boltzmann_ev_per_k = 8.617e-5
kT = boltzmann_ev_per_k * temperature_k
kernel_quad = int(os.getenv("KERNEL_QUAD", os.getenv("SP3_KERNEL_QUAD", os.getenv("CE_KERNEL_QUAD", "64"))))

sn_order = int(os.getenv("SN_ORDER", os.getenv("CE_SN_ORDER", "32")))
legendre_order = int(os.getenv("LEGENDRE_ORDER", os.getenv("CE_LEGENDRE_ORDER", "4")))
pn_order = int(os.getenv("PN_ORDER", "3"))

B2 = float(os.getenv("B2", os.getenv("SP3_B2", "0.0")))
initial_b2 = float(os.getenv("INITIAL_B2", os.getenv("CE_INITIAL_B2", "-0.01")))
fixed_b2 = os.getenv("FIXED_B2", os.getenv("SP3_FIXED_B2", os.getenv("CE_FIXED_B2", None)))
fixed_b2 = None if fixed_b2 in (None, "") else float(fixed_b2)
fixed_b2_mode = fixed_b2 is not None
xs_tol = float(os.getenv("XS_TOL", os.getenv("SP3_XS_TOL", "5")))

outdir = os.getenv("OUTDIR", "results/sp3_b2=0.0")
reference_npz = os.getenv("REFERENCE_NPZ", os.getenv("SP3_REFERENCE_NPZ", os.getenv("SP3_CE_NPZ", None)))
from_h5 = os.getenv("FROM_H5", os.getenv("SP3_FROM_H5", "false"))
adaptive = os.getenv("ADAPTIVE", os.getenv("SP3_ADAPTIVE", "false"))
wims = os.getenv("WIMS", os.getenv("SP3_WIMS", "true"))
verbose = os.getenv("VERBOSE", os.getenv("SP3_VERBOSE", "false"))
fixed_iters = int(os.getenv("FIXED_ITERS", os.getenv("SP3_FIXED_ITERS", "10000")))
fixed_tol = float(os.getenv("FIXED_TOL", os.getenv("SP3_FIXED_TOL", "1e-10")))
plot_b2_zero = os.getenv("PLOT_B2_ZERO", os.getenv("SP3_PLOT_B2_ZERO", "true"))
no_upscatter = os.getenv("NO_UPSCATTER", os.getenv("SP3_NO_UPSCATTER", os.getenv("CE_NO_UPSCATTER", "false")))
p0_only_upscatter = os.getenv("P0_ONLY_UPSCATTER", os.getenv("SP3_P0_ONLY_UPSCATTER", os.getenv("CE_P0_ONLY_UPSCATTER", "false")))
#p0_only_upscatter = os.getenv("P0_ONLY_UPSCATTER", os.getenv("SP3_P0_ONLY_UPSCATTER", os.getenv("CE_P0_ONLY_UPSCATTER", "true")))
thermal_upscatter_cutoff_ev = float(os.getenv("THERMAL_UPSCATTER_CUTOFF_EV", os.getenv("SP3_THERMAL_UPSCATTER_CUTOFF_EV", os.getenv("CE_THERMAL_UPSCATTER_CUTOFF_EV", "4.0"))))
thermal_upscatter_nuclides = os.getenv("THERMAL_UPSCATTER_NUCLIDES", os.getenv("CE_THERMAL_UPSCATTER_NUCLIDES", "H,U"))

ce_outdir = os.getenv("CE_OUTDIR", outdir)
pn_outdir = os.getenv("PN_OUTDIR", outdir)
compare_outdir = os.getenv("COMPARE_OUTDIR", outdir)
compare_reference = os.getenv("COMPARE_REFERENCE", "sn")
compare_sn_npz = os.getenv("COMPARE_SN_NPZ", None)
compare_pn_npz = os.getenv("COMPARE_PN_NPZ", "")
compare_sp3_npz = os.getenv("COMPARE_SP3_NPZ", None)

max_legendre_ratio = float(os.getenv("MAX_LEGENDRE_RATIO", os.getenv("CE_MAX_LEGENDRE_RATIO", "inf")))
inner_tol = float(os.getenv("INNER_TOL", os.getenv("CE_INNER_TOL", "1e-8")))
inner_max_iters = int(os.getenv("INNER_MAX_ITERS", os.getenv("CE_INNER_MAX_ITERS", "5000")))
gmres_restart = int(os.getenv("GMRES_RESTART", os.getenv("CE_GMRES_RESTART", "80")))
linear_solver = os.getenv("LINEAR_SOLVER", os.getenv("CE_LINEAR_SOLVER", "lgmres"))
initial_b2_step = float(os.getenv("INITIAL_B2_STEP", os.getenv("CE_INITIAL_B2_STEP", "0.01")))
max_abs_b2 = float(os.getenv("MAX_ABS_B2", os.getenv("CE_MAX_ABS_B2", "1.0")))
negative_b2_safety = float(os.getenv("NEGATIVE_B2_SAFETY", os.getenv("CE_NEGATIVE_B2_SAFETY", "0.98")))
b2_k_tol = float(os.getenv("B2_K_TOL", os.getenv("CE_B2_K_TOL", "1e-6")))
b2_abs_tol = float(os.getenv("B2_ABS_TOL", os.getenv("CE_B2_ABS_TOL", "1e-8")))
b2_max_iters = int(os.getenv("B2_MAX_ITERS", os.getenv("CE_B2_MAX_ITERS", "80")))
scatter_validation = os.getenv("SCATTER_VALIDATION", os.getenv("CE_SCATTER_VALIDATION", "false"))
diagnostics = os.getenv("DIAGNOSTICS", os.getenv("CE_DIAGNOSTICS", "false"))
memory_profile = os.getenv("MEMORY_PROFILE", os.getenv("CE_MEMORY_PROFILE", "false"))

pn_scatter_legendre_order = int(os.getenv("PN_SCATTER_LEGENDRE_ORDER", str(legendre_order)))
pn_inner_tol = float(os.getenv("PN_INNER_TOL", str(inner_tol)))
pn_inner_max_iters = int(os.getenv("PN_INNER_MAX_ITERS", str(inner_max_iters)))
pn_gmres_restart = int(os.getenv("PN_GMRES_RESTART", str(gmres_restart)))
pn_linear_solver = os.getenv("PN_LINEAR_SOLVER", linear_solver)

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
    scatter_u: object = None
    scatter_h: object = None


def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_float_or_none(name):
    raw = os.getenv(name)
    return None if raw in (None, "") else float(raw)


def ce_args():
    return SimpleNamespace(
        data_dir=str(data_dir),
        group_structure=group_structure,
        outdir=ce_outdir,
        fine_groups=nbins,
        sn_order=sn_order,
        legendre_order=legendre_order,
        NH=NH,
        NU=NU,
        nu=nu_bar,
        e0=env_float_or_none("CE_E0"),
        emin=env_float_or_none("CE_EMIN"),
        kT=kT,
        kernel_quad=kernel_quad,
        no_upscatter=as_bool(no_upscatter),
        thermal_upscatter_nuclides=thermal_upscatter_nuclides,
        thermal_upscatter_cutoff_ev=thermal_upscatter_cutoff_ev,
        p0_only_upscatter=as_bool(p0_only_upscatter),
        diagnostics=as_bool(diagnostics),
        no_scatter_validation=not as_bool(scatter_validation),
        max_legendre_ratio=max_legendre_ratio,
        inner_tol=inner_tol,
        inner_max_iters=inner_max_iters,
        gmres_restart=gmres_restart,
        linear_solver=linear_solver,
        initial_b2=initial_b2,
        fixed_b2=fixed_b2,
        initial_b2_step=initial_b2_step,
        max_abs_b2=max_abs_b2,
        negative_b2_safety=negative_b2_safety,
        b2_k_tol=b2_k_tol,
        b2_abs_tol=b2_abs_tol,
        b2_max_iters=b2_max_iters,
        verbose_inner=env_bool("CE_VERBOSE_INNER"),
        memory_profile=as_bool(memory_profile),
        no_memory_profile=False,
    )


def pn_args():
    args = ce_args()
    args.outdir = pn_outdir
    args.pn_order = pn_order
    args.scatter_legendre_order = pn_scatter_legendre_order
    args.inner_tol = pn_inner_tol
    args.inner_max_iters = pn_inner_max_iters
    args.gmres_restart = pn_gmres_restart
    args.linear_solver = pn_linear_solver
    return args


def sp3_config():
    return SimpleNamespace(
        fixed_b2=fixed_b2,
        outdir=outdir,
        ce_npz=reference_npz,
        nbins=nbins,
        NH=NH,
        xs_tol=xs_tol,
        group_structure=group_structure,
        kernel_quad=kernel_quad,
        no_upscatter=as_bool(no_upscatter),
        thermal_upscatter_cutoff_ev=thermal_upscatter_cutoff_ev,
        p0_only_upscatter=as_bool(p0_only_upscatter),
        from_h5=as_bool(from_h5),
        adaptive=as_bool(adaptive),
        wims=as_bool(wims),
        verbose=as_bool(verbose),
        fixed_iters=fixed_iters,
        fixed_tol=fixed_tol,
        plot_b2_zero=as_bool(plot_b2_zero),
    )


def compare_args():
    pn_npz = [item.strip() for item in compare_pn_npz.split(";") if item.strip()]
    return SimpleNamespace(
        sn_npz=compare_sn_npz,
        pn_npz=pn_npz,
        sp3_npz=compare_sp3_npz,
        outdir=compare_outdir,
        reference=compare_reference,
    )


def read_table(path, delimiter=None, skip_header=0):
    return np.genfromtxt(path, delimiter=delimiter, names=True, skip_header=skip_header)


def read_plain_table(path, delimiter=None, skiprows=0):
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)


def group_edges(data_dir=data_dir, group_structure=group_structure):
    table = read_plain_table(Path(data_dir) / group_structure)
    edges = np.asarray(table[:, 1], dtype=float)
    return edges[np.argsort(edges)[::-1]]


def lethargy_grid(fine_groups=nbins, e0=None, emin=None, edges=None):
    edges = group_edges() if edges is None else np.asarray(edges, dtype=float)
    e0 = float(e0 or edges[0])
    emin = float(emin or edges[-1])
    u_bounds = np.linspace(0.0, math.log(e0 / emin), fine_groups + 1)
    e_bounds = e0 * np.exp(-u_bounds)
    return u_bounds, e_bounds, 0.5 * (e_bounds[:-1] + e_bounds[1:])


def alpha_fn(A):
    return ((A - 1.0) / (A + 1.0)) ** 2


def lga_fn(alpha):
    return -math.log(alpha) if alpha != 0.0 else math.inf


def group_bound(u_bounds, A, g, lga):
    return u_bounds.size if A == 1.0 else 1 + int(np.searchsorted(u_bounds, u_bounds[g] + lga))


def gmax_vec_fn(u_bounds, A, lga):
    out = np.zeros_like(u_bounds, dtype=np.int64)
    for g in range(u_bounds.size):
        out[g] = group_bound(u_bounds, A, g, lga)
    return out


def interpolate_on_u(data, e0, emin, u_grid):
    data = np.asarray(data, dtype=float)
    data = data[(data[:, 0] <= e0) & (data[:, 0] >= emin)].copy()
    data[:, 0] = np.log(e0 / data[:, 0])
    data = data[np.argsort(data[:, 0])]

    out = np.empty((u_grid.size, data.shape[1]), dtype=float)
    out[:, 0] = u_grid
    for j in range(1, data.shape[1]):
        out[:, j] = np.interp(u_grid, data[:, 0], data[:, j])
    return out


def uniform_lethargy_table(data, gridpoints, e0, emin):
    u_grid = np.linspace(0.0, math.log(e0 / emin), gridpoints)
    return interpolate_on_u(data, e0, emin, u_grid)


def adaptive_u_grid(data, e0, emin, tol):
    data = np.asarray(data, dtype=float)
    data = data[(data[:, 0] <= e0) & (data[:, 0] >= emin)].copy()
    data[:, 0] = np.log(e0 / data[:, 0])
    data = data[np.argsort(data[:, 0])]
    grid = [data[0]]
    i = 0
    while i < data.shape[0] - 1:
        for j in range(i + 1, data.shape[0]):
            diff = 100.0 * abs(data[i, 1] - data[j, 1]) / data[i, 1]
            if diff >= tol:
                grid.append(data[j])
                i = j
                break
        else:
            break
    return np.array(grid)


def interp_to_u_grid(data, u_grid, e0, emin):
    return interpolate_on_u(data, e0, emin, np.asarray(u_grid, dtype=float))


def group_average_from_bounds(values_at_bounds, u_bounds):
    return 0.5 * (values_at_bounds[:-1] + values_at_bounds[1:])


def normalize_group_source(values_at_bounds, u_bounds):
    avg = group_average_from_bounds(values_at_bounds, u_bounds)
    group_integral = avg * np.diff(u_bounds)
    return group_integral / np.sum(group_integral)


def load_hf_tables(data_dir=data_dir):
    data_dir = Path(data_dir)
    o16_path = data_dir / "xs_o16_T293k.csv"
    if not o16_path.exists():
        o16_path = data_dir / "xs_o16_T293k (1).csv"

    h1 = read_table(data_dir / "xs_h1_T293k.txt", delimiter="\t")
    u238 = read_table(data_dir / "xs_u238_T293k.txt", delimiter="\t")
    chi35 = read_table(data_dir / "chi_u235.txt", delimiter="\t")

    return {
        "group_edges_e": group_edges(data_dir),
        "H": np.column_stack((h1["E"], h1["sigma_t"], h1["sigma_s"])),
        "U238": np.column_stack((u238["E"], u238["sigma_t"], u238["sigma_s"])),
        "O16": read_plain_table(o16_path, delimiter=";", skiprows=3),
        "chi": np.column_stack((chi35["E"], chi35["chi"])),
        "sigma_f": read_plain_table(data_dir / "xs_u238_fission.csv", delimiter=",", skiprows=1),
        "sigma_fr_U": read_plain_table(data_dir / "sigma_fr_U.csv", delimiter=";", skiprows=1),
        "sigma_fr_H": read_plain_table(data_dir / "sigma_fr_H.csv", delimiter=";", skiprows=1),
    }



MEMORY_PROFILE = False


def _linux_status_memory_gb():
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


def report_memory(label):
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


def diagnose_vector(name, values, e_mid=None, max_items=8):
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
    name,
    scatter,
    e_mid,
    sig_s0=None,
    max_items=12,
):
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


def validate_scatter_matrix(name, scatter, e_mid, max_legendre_ratio):
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
    A,
    sig_s0,
    sigma_fr,
    order,
    u_bounds,
    e_bounds,
    kT,
    nquad,
    upscatter,
    p0_only_upscatter,
    thermal_cutoff_ev,
):
    stt = time.time()
    du = np.diff(u_bounds)
    alpha = alpha_fn(A)
    gmax_vec = gmax_vec_fn(u_bounds, A, lga_fn(alpha))
    sigma_gtg = gen_sig_sn_gtg(A, sig_s0, order, u_bounds, gmax_vec, alpha, du)

    if upscatter:
        e_mid = 0.5 * (e_bounds[:-1] + e_bounds[1:])
        thermal = np.where(e_mid <= thermal_cutoff_ev)[0]
        if thermal.size:
            insert_idx = int(thermal[0])
            exit_start_idx = 0
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

    scatter = np.transpose(sigma_gtg, (0, 2, 1))
    print(
        f"XS generation A={A:g}, moments={order}, groups={sig_s0.size}, "
        f"upscatter={upscatter}, wall={time.time() - stt:.5f} s"
    )
    return scatter
    #return np.transpose(sigma_gtg, (0, 2, 1))


def thermal_upscatter_enabled(args, name):
    if args.no_upscatter:
        return False
    enabled = {item.strip().lower() for item in args.thermal_upscatter_nuclides.split(",") if item.strip()}
    aliases = {
        "H-1": {"h", "h1", "h-1", "hydrogen"},
        "U-238": {"u", "u238", "u-238", "uranium"},
    }
    return bool(enabled & aliases[name])


def load_problem(args, reuse_sn=False):
    if reuse_sn:
        sn_npz = sn_reference_npz_path(args.outdir)
        if sn_npz.exists():
            print(f'Loading hyperfine XS from {sn_npz}')
            return material_from_npz(sn_npz)

    data_dir = Path(args.data_dir)
    group_edges_e = group_edges(data_dir, args.group_structure)
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
        scatter_u=scatter_u,
        scatter_h=scatter_h,
    )



def set_memory_profile(enabled):
    global MEMORY_PROFILE
    MEMORY_PROFILE = bool(enabled)
    if MEMORY_PROFILE and not tracemalloc.is_tracing():
        tracemalloc.start()


def sn_reference_npz_path(outdir=None):
    return Path(outdir or ce_outdir) / "ce_buckled_transport_reference.npz"


def material_from_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return MaterialData(
            u_bounds=archive["u_bounds"],
            e_bounds=archive["e_bounds"],
            e_mid=archive["e_mid"],
            du=np.diff(archive["u_bounds"]),
            sigma_t=archive["sigma_t"],
            nu_sigma_f=archive["nu_sigma_f"],
            chi=archive["chi"],
            scatter=archive["scatter"],
            group_edges_e=archive["group_edges_e"],
            group_names=[f"g{i + 1}" for i in range(archive["group_edges_e"].size - 1)],
            scatter_u=archive["scatter_u"] if "scatter_u" in archive.files else None,
            scatter_h=archive["scatter_h"] if "scatter_h" in archive.files else None,
        )


def fine_to_coarse_indices(e_mid, group_edges_e):
    groups = []
    for i in range(group_edges_e.size - 1):
        hi = group_edges_e[i]
        lo = group_edges_e[i + 1]
        groups.append(np.where((e_mid <= hi) & (e_mid >= lo))[0])
    return groups


def collapse_vector(xs, weight, groups):
    out = np.zeros(len(groups), dtype=np.complex128)
    for i, idx in enumerate(groups):
        den = np.sum(weight[idx])
        out[i] = np.sum(xs[idx] * weight[idx]) / den if abs(den) > 1.0e-250 else 0.0
    return out


def collapse_scatter(scatter, moments, groups):
    L = scatter.shape[0]
    Gc = len(groups)
    out = np.zeros((L, Gc, Gc), dtype=np.complex128)
    for ell in range(L):
        w = moments[ell]
        for j, inc in enumerate(groups):
            den = np.sum(w[inc])
            if abs(den) <= 1.0e-250:
                continue
            for i, outg in enumerate(groups):
                block = scatter[ell][np.ix_(outg, inc)]
                out[ell, i, j] = np.sum(block * w[inc][None, :]) / den
    return out


def collapse_results(data, moments):
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
