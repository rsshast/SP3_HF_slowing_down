#!/usr/bin/env python3
"""Compare buckled PN, SN, and SP3 result archives.

This is a post-processing script only.  It reads existing ``.npz`` files from
the PN reference, SN reference, and SP3 fixed-B2 runs, then writes comparison
plots and CSV/JSON metric summaries.
"""

from __future__ import annotations

from input import *


def real_array(data, key):
    return np.real_if_close(data[key]).real


def scalar_float(data, key, default=np.nan):
    if key not in data.files:
        return float(default)
    return float(np.real_if_close(data[key]).real)


def reference_label(data, fallback):
    if "reference_type" in data.files:
        raw = data["reference_type"]
        ref_type = str(raw.item() if getattr(raw, "shape", ()) == () else raw).upper()
        if ref_type == "PN":
            order = int(data["pn_order"]) if "pn_order" in data.files else int(data["moments"].shape[0] - 1)
            return f"CE P{order}"
    if "sn_order" in data.files:
        return f"CE S{int(data['sn_order'])}"
    return fallback


def group_integral_phi(data, key="coarse_phi0"):
    if key in data.files:
        return real_array(data, key)
    raise KeyError(f"{key} is not present in {data.files}")


def normalized_density_from_integral(phi, widths):
    phi = np.real_if_close(phi).real
    total = np.sum(phi)
    if abs(total) > 1e-300:
        phi = phi / total
    return phi / widths


def normalized_l2(candidate, reference):
    candidate = np.real_if_close(candidate).real.ravel()
    reference = np.real_if_close(reference).real.ravel()
    csum = np.sum(candidate)
    rsum = np.sum(reference)
    if abs(csum) <= 1e-300 or abs(rsum) <= 1e-300:
        return float("nan")
    return float(np.linalg.norm(candidate / csum - reference / rsum))


def relative_l2(candidate, reference):
    candidate = np.real_if_close(candidate).real.ravel()
    reference = np.real_if_close(reference).real.ravel()
    denom = np.linalg.norm(reference)
    if denom <= 1e-300:
        return float("nan")
    return float(np.linalg.norm(candidate - reference) / denom)


def max_abs_rel(candidate, reference):
    candidate = np.real_if_close(candidate).real.ravel()
    reference = np.real_if_close(reference).real.ravel()
    denom = np.maximum(np.abs(reference), 1e-300)
    return float(np.max(np.abs(candidate - reference) / denom))


def add_metric(rows, quantity, candidate_name, reference_name, candidate, reference, metric_name, metric_fn):
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        rows.append(
            {
                "quantity": quantity,
                "candidate": candidate_name,
                "reference": reference_name,
                "metric": metric_name,
                "value": np.nan,
                "note": f"shape mismatch candidate={candidate.shape}, reference={reference.shape}",
            }
        )
        return
    if not np.all(np.isfinite(np.real_if_close(candidate).real)):
        note = "candidate has non-finite values"
        value = np.nan
    elif not np.all(np.isfinite(np.real_if_close(reference).real)):
        note = "reference has non-finite values"
        value = np.nan
    else:
        note = ""
        value = metric_fn(candidate, reference)
    rows.append(
        {
            "quantity": quantity,
            "candidate": candidate_name,
            "reference": reference_name,
            "metric": metric_name,
            "value": value,
            "note": note,
        }
    )


def coarse_reference_moment(data, ell):
    coarse_key = f"coarse_phi{ell}"
    if coarse_key in data.files:
        return real_array(data, coarse_key), coarse_key
    if ell == 0 and "coarse_phi0" in data.files:
        return real_array(data, "coarse_phi0"), "coarse_phi0"
    if "moments" in data.files and "e_mid" in data.files and "group_edges_e" in data.files:
        moments = np.real_if_close(data["moments"]).real
        if ell < moments.shape[0]:
            groups = fine_to_coarse_indices(real_array(data, "e_mid"), real_array(data, "group_edges_e"))
            return np.array([np.sum(moments[ell, idx]) for idx in groups], dtype=float), f"collapsed_moment_{ell}"
    if "coarse_phi0" in data.files:
        return real_array(data, "coarse_phi0"), "coarse_phi0_fallback"
    raise KeyError("No usable coarse reference moment was found.")


def add_scatter_action_metric(rows, quantity, candidate_name, reference_name, candidate_s, reference_s, action_vector, vector_note):
    candidate_s = np.asarray(candidate_s)
    reference_s = np.asarray(reference_s)
    action_vector = np.asarray(action_vector)
    metric_name = "action_relative_l2"
    if candidate_s.shape != reference_s.shape:
        note = f"shape mismatch candidate={candidate_s.shape}, reference={reference_s.shape}"
        value = np.nan
    elif candidate_s.ndim != 2:
        note = f"expected 2-D scattering matrix, got shape={candidate_s.shape}"
        value = np.nan
    elif action_vector.shape != (reference_s.shape[1],):
        note = f"action vector shape mismatch vector={action_vector.shape}, matrix_incident={reference_s.shape[1]}"
        value = np.nan
    elif not np.all(np.isfinite(np.real_if_close(candidate_s).real)):
        note = "candidate has non-finite values"
        value = np.nan
    elif not np.all(np.isfinite(np.real_if_close(reference_s).real)):
        note = "reference has non-finite values"
        value = np.nan
    elif not np.all(np.isfinite(np.real_if_close(action_vector).real)):
        note = "action vector has non-finite values"
        value = np.nan
    else:
        cand_action = np.real_if_close(candidate_s).real @ np.real_if_close(action_vector).real
        ref_action = np.real_if_close(reference_s).real @ np.real_if_close(action_vector).real
        value = relative_l2(cand_action, ref_action)
        note = f"vector={vector_note}"
    rows.append(
        {
            "quantity": quantity,
            "candidate": candidate_name,
            "reference": reference_name,
            "metric": metric_name,
            "value": value,
            "note": note,
        }
    )


def load_npz(path: str | None):
    if path is None:
        return None, None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p, np.load(p)


def plot_spectra(outdir, refs, sp3):
    plt.figure(figsize=(9.5, 5.8))
    for ref in refs:
        data = ref["data"]
        edges = real_array(data, "group_edges_e")
        du = real_array(data, "coarse_du")
        phi = group_integral_phi(data)
        dens = normalized_density_from_integral(phi, du)
        x = edges[::-1]
        y = dens[::-1]
        plt.step(x, np.r_[y, y[-1]], where="post", label=f"{ref['label']}", linestyle='dotted')
        #plt.step(x, np.r_[y, y[-1]], where="post", label=f"{ref['label']}, k={scalar_float(data, 'k_at_B2'):.6g}", linestyle='dotted')

    if sp3 is not None:
        data = sp3["data"]
        edges = real_array(data, "group_edges_e")
        du = real_array(data, "coarse_du")
        for key, label, color, linestyle in (
            ("phi0_new", "new SP3", None, 'solid'),
            ("phi0_trad", "traditional SP3", None, 'dashed'),
        ):
            if key not in data.files:
                continue
            dens = normalized_density_from_integral(real_array(data, key), du)
            x = edges[::-1]
            y = dens[::-1]
            k_key = "k_new" if key == "phi0_new" else "k_trad"
            plt.step(x, np.r_[y, y[-1]], where="post", label=f"{label}", color=color,linestyle=linestyle)
            #plt.step(x, np.r_[y, y[-1]], where="post", label=f"{label}, k={scalar_float(data, k_key):.6g}", color=color,linestyle=linestyle)

    plt.xscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized few-group scalar flux per lethargy")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pn_sn_sp3_few_group_spectra.png", dpi=200)
    plt.close()


def build_metrics(refs, sp3):
    rows = []

    if len(refs) >= 2:
        base = refs[0]
        for other in refs[1:]:
            add_metric(rows, "phi0", other["label"], base["label"], group_integral_phi(other["data"]), group_integral_phi(base["data"]), "normalized_l2", normalized_l2)
            for key in ("Sigma_t", "nuSigma_f", "chi_few"):
                if key in other["data"].files and key in base["data"].files:
                    metric = "normalized_l2" if key == "chi_few" else "relative_l2"
                    fn = normalized_l2 if key == "chi_few" else relative_l2
                    add_metric(rows, key, other["label"], base["label"], other["data"][key], base["data"][key], metric, fn)
            if "Sigma_s_moment_weighted" in other["data"].files and "Sigma_s_moment_weighted" in base["data"].files:
                n_ell = min(other["data"]["Sigma_s_moment_weighted"].shape[0], base["data"]["Sigma_s_moment_weighted"].shape[0])
                for ell in range(n_ell):
                    add_metric(
                        rows,
                        f"Sigma_s_l{ell}",
                        other["label"],
                        base["label"],
                        other["data"]["Sigma_s_moment_weighted"][ell],
                        base["data"]["Sigma_s_moment_weighted"][ell],
                        "relative_l2",
                        relative_l2,
                    )
                    try:
                        action_vector, vector_note = coarse_reference_moment(base["data"], ell)
                        add_scatter_action_metric(
                            rows,
                            f"Sigma_s_l{ell}",
                            other["label"],
                            base["label"],
                            other["data"]["Sigma_s_moment_weighted"][ell],
                            base["data"]["Sigma_s_moment_weighted"][ell],
                            action_vector,
                            vector_note,
                        )
                    except KeyError as exc:
                        rows.append(
                            {
                                "quantity": f"Sigma_s_l{ell}",
                                "candidate": other["label"],
                                "reference": base["label"],
                                "metric": "action_relative_l2",
                                "value": np.nan,
                                "note": str(exc),
                            }
                        )

    if sp3 is not None and refs:
        for ref in refs:
            ref_data = ref["data"]
            sp3_data = sp3["data"]
            for key, label in (("phi0_new", "new SP3"), ("phi0_trad", "traditional SP3")):
                if key in sp3_data.files:
                    add_metric(rows, "phi0", label, ref["label"], sp3_data[key], group_integral_phi(ref_data), "normalized_l2", normalized_l2)
            pairs = (
                ("Sigma_t", "new_Sigma_t_phi0", "new SP3 phi0-weighted"),
                ("Sigma_t", "trad_Sigma_t", "traditional SP3"),
                ("nuSigma_f", "new_nuSigma_f_phi0", "new SP3 phi0-weighted"),
                ("nuSigma_f", "trad_nuSigma_f", "traditional SP3"),
                ("chi_few", "chi", "SP3"),
            )
            for ref_key, sp3_key, label in pairs:
                if ref_key in ref_data.files and sp3_key in sp3_data.files:
                    metric = "normalized_l2" if ref_key == "chi_few" else "relative_l2"
                    fn = normalized_l2 if ref_key == "chi_few" else relative_l2
                    add_metric(rows, ref_key, label, ref["label"], sp3_data[sp3_key], ref_data[ref_key], metric, fn)
            if "Sigma_s_moment_weighted" in ref_data.files and "trad_Sigma_s" in sp3_data.files:
                ref_s = ref_data["Sigma_s_moment_weighted"]
                for sp3_key, label in (
                    ("trad_Sigma_s", "traditional SP3"),
                    ("new_Sigma_s_phi0", "new SP3 phi0-weighted"),
                    ("new_Sigma_s_phi2", "new SP3 phi2-weighted"),
                ):
                    if sp3_key not in sp3_data.files:
                        continue
                    n_ell = min(ref_s.shape[0], sp3_data[sp3_key].shape[0])
                    for ell in range(n_ell):
                        add_metric(rows, f"Sigma_s_l{ell}", label, ref["label"], sp3_data[sp3_key][ell], ref_s[ell], "relative_l2", relative_l2)
                        try:
                            action_vector, vector_note = coarse_reference_moment(ref_data, ell)
                            add_scatter_action_metric(
                                rows,
                                f"Sigma_s_l{ell}",
                                label,
                                ref["label"],
                                sp3_data[sp3_key][ell],
                                ref_s[ell],
                                action_vector,
                                vector_note,
                            )
                        except KeyError as exc:
                            rows.append(
                                {
                                    "quantity": f"Sigma_s_l{ell}",
                                    "candidate": label,
                                    "reference": ref["label"],
                                    "metric": "action_relative_l2",
                                    "value": np.nan,
                                    "note": str(exc),
                                }
                            )
    return rows


def write_k_summary(outdir, refs, sp3):
    rows = []
    for ref in refs:
        rows.append({"case": ref["label"], "B2": scalar_float(ref["data"], "B2"), "k": scalar_float(ref["data"], "k_at_B2"), "file": str(ref["path"])})
    if sp3 is not None:
        rows.append({"case": "new SP3", "B2": scalar_float(sp3["data"], "B2"), "k": scalar_float(sp3["data"], "k_new"), "file": str(sp3["path"])})
        rows.append({"case": "traditional SP3", "B2": scalar_float(sp3["data"], "B2"), "k": scalar_float(sp3["data"], "k_trad"), "file": str(sp3["path"])})

    with (outdir / "k_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "B2", "k", "file"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args():
    return compare_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    refs = []
    sn_path, sn = load_npz(args.sn_npz)
    if sn is not None:
        refs.append({"kind": "sn", "path": sn_path, "data": sn, "label": reference_label(sn, "CE SN")})
    for path in args.pn_npz:
        pn_path, pn = load_npz(path)
        refs.append({"kind": "pn", "path": pn_path, "data": pn, "label": reference_label(pn, "CE PN")})

    if args.reference == "pn":
        refs.sort(key=lambda item: 0 if item["kind"] == "pn" else 1)
    else:
        refs.sort(key=lambda item: 0 if item["kind"] == "sn" else 1)

    sp3_path, sp3_data = load_npz(args.sp3_npz)
    sp3 = {"path": sp3_path, "data": sp3_data} if sp3_data is not None else None

    if not refs and sp3 is None:
        raise ValueError("Provide at least one of --sn-npz, --pn-npz, or --sp3-npz.")

    plot_spectra(outdir, refs, sp3)
    metric_rows = build_metrics(refs, sp3)
    with (outdir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["quantity", "candidate", "reference", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(metric_rows)
    k_rows = write_k_summary(outdir, refs, sp3)

    summary = {
        "references": [{"kind": r["kind"], "label": r["label"], "file": str(r["path"])} for r in refs],
        "sp3_file": str(sp3["path"]) if sp3 is not None else None,
        "k_summary": k_rows,
        "metric_count": len(metric_rows),
    }
    (outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote comparison outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
