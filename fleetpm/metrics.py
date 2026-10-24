"""Evaluation utilities independent of training and of PyTorch."""

import math


def regression_metrics(truth, predictions, cap=125.0):
    y, p = list(map(float, truth)), list(map(float, predictions))
    if not y or len(y) != len(p) or not all(math.isfinite(x) for x in y + p):
        raise ValueError("Metrics need equal, finite, non-empty prediction and truth vectors")
    errors = [b - a for a, b in zip(y, p)]
    mean = sum(y) / len(y)
    squared = sum(d * d for d in errors)
    total = sum((a - mean) ** 2 for a in y)
    exponents = [(-d / 13 if d < 0 else d / 10) for d in errors]
    overflow = max(exponents) > 700
    near = [d for a, d in zip(y, errors) if a <= 30]
    return {"n_engines": len(y), "rmse": math.sqrt(squared / len(y)),
            "mae": sum(abs(d) for d in errors) / len(y),
            "r2": 1 - squared / total if total else None,
            "nasa_score": None if overflow else sum(math.expm1(v) for v in exponents),
            "nasa_score_overflow": overflow,
            "capped_rmse": math.sqrt(sum((b - min(cap, a)) ** 2 for a, b in zip(y, p)) / len(y)),
            "near_failure_rmse": math.sqrt(sum(d*d for d in near) / len(near)) if near else None,
            "late_over_10_rate": sum(d > 10 for d in errors) / len(y)}


def conformal_radius(residuals, alpha=0.1):
    values = sorted(float(x) for x in residuals)
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Calibration residuals must be finite and nonnegative")
    k = math.ceil((len(values) + 1) * (1-alpha))
    return {"radius": values[k-1] if k <= len(values) else None,
            "alpha": alpha, "n_engines": len(values), "order_statistic": k,
            "target": "capped_RUL", "scope": "one declared prefix per independent held-out engine"}


def prediction_interval(point, calibration, cap=125.0):
    q = calibration.get("radius")
    if q is None:
        return None, None
    return max(0.0, point-q), min(cap, point+q)
