#!/usr/bin/env python3
"""Compare new SP3, conventional SP3, and SN scalar-flux spectra over a B2 sweep."""

from __future__ import annotations

from input import *


CASE_NAME = os.getenv("CASE_NAME", "sp3_wims69")
B2_LIST = ("-0.04", "-0.02", "-0.01", "-0.005", "-0.001", "0.0", "0.001", "0.005")

SCRATCH_RESULTS = Path("/scratch/bckiedro_root/bckiedro0/rsshast/Sp3/results")
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", SCRATCH_RESULTS if SCRATCH_RESULTS.exists() else "results"))
OUTDIR = Path(os.getenv("COMPARE_OUTDIR", RESULTS_ROOT / f"{CASE_NAME}_scalar_flux_comparison"))


def b2_values():
    return tuple(os.getenv("B2_VALUES", "").split()) or B2_LIST


def real(data, key):
    return np.real_if_close(data[key]).real


def normalized(phi):
    phi = np.asarray(np.real_if_close(phi).real, dtype=float).ravel()
    total = np.sum(phi)
    if abs(total) <= 1.0e-300:
        raise ValueError("Cannot normalize a spectrum with zero total flux.")
    return phi / total


def relative_l2(candidate, reference):
    candidate = normalized(candidate)
    reference = normalized(reference)
    denom = np.linalg.norm(reference)
    return np.nan if denom <= 1.0e-300 else float(np.linalg.norm(candidate - reference) / denom)


def read_case(b2_text):
    case_dir = RESULTS_ROOT / f"{CASE_NAME}_b2={b2_text}_15k"
    sp3_path = case_dir / "sp3_results.npz"
    sn_path = case_dir / "ce_buckled_transport_reference.npz"
    missing = [str(path) for path in (sp3_path, sn_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result file(s):\n  " + "\n  ".join(missing))

    with np.load(sp3_path, allow_pickle=False) as sp3, np.load(sn_path, allow_pickle=False) as sn:
        new = real(sp3, "phi0_new")
        conv = real(sp3, "phi0_trad")
        sn_phi = real(sn, "coarse_phi0")
        if not (new.shape == conv.shape == sn_phi.shape):
            raise ValueError(f"{case_dir}: shape mismatch new={new.shape}, conv={conv.shape}, sn={sn_phi.shape}")

        return {
            "B2": float(real(sp3, "B2")) if "B2" in sp3.files else float(b2_text),
            "new_vs_conventional": relative_l2(new, conv),
            "new_vs_sn": relative_l2(new, sn_phi),
            "conventional_vs_sn": relative_l2(conv, sn_phi),
            "case_dir": str(case_dir),
            "fine_energy_e": real(sp3, "fine_energy_e"),
            "fine_phi0": real(sp3, "fine_phi0"),
            "fine_phi2": real(sp3, "fine_phi2"),
        }


def metric_rows(cases):
    return [
        {
            "B2": case["B2"],
            "new_vs_conventional": case["new_vs_conventional"],
            "new_vs_sn": case["new_vs_sn"],
            "conventional_vs_sn": case["conventional_vs_sn"],
            "case_dir": case["case_dir"],
        }
        for case in cases
    ]


def write_csv(cases):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "scalar_flux_relative_l2.csv"
    rows = metric_rows(cases)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_l2_norms(cases):
    path = OUTDIR / "scalar_flux_relative_l2_vs_b2.png"
    b2 = [case["B2"] for case in cases]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(b2, [case["new_vs_sn"] for case in cases], marker="o", label="New vs. SN")
    plt.plot(b2, [case["conventional_vs_sn"] for case in cases], marker="s", label="Conventional vs. SN")
    plt.xlabel(r"$B^2$")
    plt.yscale("log")
    plt.ylabel("Relative L2 norm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_fine_spectra(cases, key, ylabel, filename):
    path = OUTDIR / filename
    plt.figure(figsize=(8.2, 5.2))
    for case in cases:
        energy = case["fine_energy_e"]
        phi = case[key]
        if key == "fine_phi2": plt.plot(energy[::-1], phi[::-1], label=fr"$B^2$={case['B2']:.3g}")
        else: plt.plot(energy[::-1], phi[::-1] / np.sum(phi[::-1]), label=fr"$B^2$={case['B2']:.3g}")
        #plt.plot(energy[::-1], phi[::-1] / np.sum(phi[::-1]), label=fr"$B^2$={case['B2']:.3g}")
    plt.xscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_phi0_difference_from_zero(cases):
    path = OUTDIR / "fine_phi0_difference_from_b2_zero.png"
    zero_cases = [case for case in cases if abs(case["B2"]) <= 1.0e-14]
    if not zero_cases:
        raise ValueError("Cannot plot phi0 differences because no B2=0 case was found.")

    ref = zero_cases[0]
    ref_energy = ref["fine_energy_e"]
    ref_phi0 = ref["fine_phi0"]
    tiny = np.finfo(float).tiny

    plt.figure(figsize=(8.2, 5.2))
    for case in cases:
        if abs(case["B2"]) <= 1.0e-14:
            continue
        if case["fine_phi0"].shape != ref_phi0.shape:
            raise ValueError(f"B2={case['B2']}: phi0 shape does not match B2=0.")
        diff = np.maximum(np.abs(case["fine_phi0"] - ref_phi0), tiny)
        plt.plot(ref_energy[::-1], diff[::-1], label=fr"$B^2$={case['B2']:.5g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel(r"$|\phi_0(B^2)-\phi_0(0)|$")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    cases = sorted((read_case(b2) for b2 in b2_values()), key=lambda row: row["B2"])
    csv_path = write_csv(cases)
    plot_paths = [
        plot_l2_norms(cases),
        plot_fine_spectra(cases, "fine_phi0", r"Fine-group $\phi_0$", "fine_phi0_all_b2.png"),
        plot_fine_spectra(cases, "fine_phi2", r"Fine-group $\phi_2$", "fine_phi2_all_b2.png"),
        plot_phi0_difference_from_zero(cases),
    ]

    print(f"Read {len(cases)} B2 cases from {RESULTS_ROOT.resolve()}")
    print(f"Wrote {csv_path.resolve()}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path.resolve()}")
    print("\nB2,new_vs_conventional,new_vs_sn,conventional_vs_sn")
    for row in metric_rows(cases):
        print(
            f"{row['B2']:.8g},"
            f"{row['new_vs_conventional']:.8e},"
            f"{row['new_vs_sn']:.8e},"
            f"{row['conventional_vs_sn']:.8e}"
        )


if __name__ == "__main__":
    main()
