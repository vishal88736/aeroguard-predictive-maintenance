"""Regime-conditioned, engine-level drift screening with explicit limits.

One median per engine/regime from a fixed observed prefix avoids pretending
overlapping rolling windows are independent samples. Holm correction is applied
across all available sensor/regime comparisons. Independence between distinct
engine cohorts remains an assumption, not an operational guarantee.
"""

import numpy as np
from scipy.stats import ks_2samp

from fleetpm.features import load_engines


def quality_report(manifest, settings):
    reasons = []
    if manifest["missing_sensor_fraction"] > settings.maximum_missing_fraction:
        reasons.append("Sensor missingness exceeds the configured quality limit")
    if manifest["label_status"] == "complete_run_to_failure" and manifest["missing_cycles"]:
        reasons.append("Complete training trajectories contain missing cycles")
    return {"passed": not reasons, "reasons": reasons,
            "missing_sensor_fraction": manifest["missing_sensor_fraction"],
            "missing_cycles": manifest["missing_cycles"], "engines": manifest["engines"],
            "labels_available": manifest["label_status"] == "complete_run_to_failure",
            "label_provenance": manifest["source_description"]}


def _engine_statistics(engines, processor, prefix_cycles, exclude=()):
    groups = {}
    unsupported, observations = 0, 0
    for unit, engine in engines.items():
        if unit in exclude:
            continue
        length = min(prefix_cycles, len(engine["raw"]))
        prefix = dict(engine, raw=engine["raw"][:length], cycles=engine["cycles"][:length])
        item = processor.transform(prefix)
        unsupported += int((~item["domain"]).sum())
        observations += length
        if length < prefix_cycles:
            continue
        for regime in np.unique(item["regime"]):
            take = (item["regime"] == regime) & item["domain"]
            if take.sum() < 3:
                continue
            sensors = np.where(item["mask"][take], item["sensors"][take], np.nan)
            row = np.full(21, np.nan)
            for column in range(21):
                finite = sensors[:, column][np.isfinite(sensors[:, column])]
                if len(finite):
                    row[column] = float(np.median(finite))
            groups.setdefault(int(regime), []).append(row)
    return {k: np.asarray(v) for k, v in groups.items()}, unsupported / max(1, observations)


def drift_report(reference_path, incoming_path, bundle, settings, label_status):
    reference = load_engines(reference_path)
    incoming = load_engines(incoming_path)
    fit_ids = set(bundle.processor.fit_ids)
    baseline = {unit: value for unit, value in reference.items() if unit in fit_ids}
    # Full labelled snapshots retain old engines. Their repeated rows must not
    # masquerade as new, independent samples in the incoming population.
    exclude = set(reference) if label_status == "complete_run_to_failure" else set()
    left, _ = _engine_statistics(baseline, bundle.processor, settings.reference_cycles)
    right, out_of_domain = _engine_statistics(incoming, bundle.processor, settings.reference_cycles, exclude)
    comparisons = []
    for regime in sorted(set(left) & set(right)):
        for column in range(21):
            a, b = left[regime][:, column], right[regime][:, column]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if min(len(a), len(b)) < settings.min_engines_per_regime:
                continue
            statistic, pvalue = ks_2samp(a, b, method="auto")
            comparisons.append({"regime": regime, "sensor": f"sensor_{column + 1}",
                                "reference_engines": len(a), "incoming_engines": len(b),
                                "ks_statistic": float(statistic), "p_value": float(pvalue)})
    ordered = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p_value"])
    previous = 0.0
    for rank, index in enumerate(ordered):
        item = comparisons[index]
        adjusted = min(1.0, max(previous, item["p_value"] * (len(comparisons) - rank)))
        previous = adjusted
        item["adjusted_p_value"] = adjusted
        item["drifted"] = adjusted < settings.ks_alpha and item["ks_statistic"] >= settings.ks_min_effect
    sensors = sorted({item["sensor"] for item in comparisons if item["drifted"]})
    status = "insufficient_data" if not comparisons else "drift" if len(sensors) >= settings.drift_sensor_count else "stable"
    return {"status": status, "drifted_sensors": sensors, "comparisons": comparisons,
            "out_of_domain_fraction": out_of_domain,
            "method": "Two-sample KS on one prefix median per engine/regime; Holm family-wise correction and a minimum KS effect",
            "alpha": settings.ks_alpha, "minimum_effect": settings.ks_min_effect,
            "minimum_engines_per_regime": settings.min_engines_per_regime,
            "reference_cycles": settings.reference_cycles,
            "scope": "Distribution screening, not measured RUL degradation. Distinct incoming engine cohorts are assumed independent. Early-prefix readings are a healthy proxy, not fault-onset labels.",
            "retraining_note": "A drift alert alone supplies no RUL labels; supervised training requires declared complete eligible trajectories."}
