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
    case_dir = RESULTS_ROOT / f"{CASE_NAME}_b2={b2_text}_5k"
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
        }


def write_csv(rows):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "scalar_flux_relative_l2.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows):
    path = OUTDIR / "scalar_flux_relative_l2_vs_b2.png"
    b2 = [row["B2"] for row in rows]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(b2, [row["new_vs_sn"] for row in rows], marker="o", label="new vs. SN")
    plt.plot(b2, [row["conventional_vs_sn"] for row in rows], marker="s", label="conv vs. SN")
    plt.title(CASE_NAME)
    plt.xlabel(r"$B^2$")
    plt.ylabel("Relative L2 norm")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    rows = sorted((read_case(b2) for b2 in b2_values()), key=lambda row: row["B2"])
    csv_path = write_csv(rows)
    plot_path = plot(rows)

    print(f"Read {len(rows)} B2 cases from {RESULTS_ROOT.resolve()}")
    print(f"Wrote {csv_path.resolve()}")
    print(f"Wrote {plot_path.resolve()}")
    print("\nB2,new_vs_conventional,new_vs_sn,conventional_vs_sn")
    for row in rows:
        print(
            f"{row['B2']:.8g},"
            f"{row['new_vs_conventional']:.8e},"
            f"{row['new_vs_sn']:.8e},"
            f"{row['conventional_vs_sn']:.8e}"
        )


if __name__ == "__main__":
    main()
