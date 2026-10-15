"""Prediction intervals for one-step-ahead forecasts.

A prediction interval is a range the next observed value should fall in with a
given probability (e.g. 95%). Two methods are used:

- **Model-based (normal)**: ARIMA/SARIMA estimate the standard error of each
  forecast, so the interval is prediction ± z * standard error.
- **Conformal**: for models without a built-in uncertainty estimate (tree models,
  neural networks, the naive baseline), the interval half-width is a quantile of
  the model's absolute errors on data it was not trained on. During the test
  period each step's error is added to that pool once its actual value is known
  (adaptive/online conformal), so the intervals adjust if errors grow. Only
  errors from earlier steps are used, so there is no look-ahead.

Each model stores an "interval spec" when it is trained; intervals are computed
from it at display time so the confidence level can change without retraining:

    {'method': 'normal', 'std_errors': array, 'source': str}
    {'method': 'conformal', 'calibration_residuals': array, 'source': str}
"""

import numpy as np
from scipy import stats


def normal_intervals(predictions, std_errors, level=0.95):
    """Intervals of prediction ± z * standard error."""
    predictions = np.asarray(predictions, dtype=float)
    z = stats.norm.ppf(0.5 + level / 2)
    half_width = z * np.asarray(std_errors, dtype=float)
    return predictions - half_width, predictions + half_width


def conformal_quantile(scores, level=0.95):
    """
    Conformal quantile of absolute errors with the finite-sample correction.

    With n scores, the ceil((n + 1) * level) / n empirical quantile gives at least
    `level` coverage for exchangeable errors. When n is too small for the requested
    level the largest score is used, which may slightly under-cover.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.nan
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return np.quantile(scores, q, method='higher')


def conformal_intervals(predictions, actual, calibration_residuals, level=0.95):
    """
    Adaptive conformal intervals for one-step-ahead predictions.

    The half-width at step t uses the calibration errors plus the test errors of
    steps 0..t-1 (already observed when forecasting step t).
    """
    predictions = np.asarray(predictions, dtype=float)
    actual = np.asarray(actual, dtype=float)
    calibration = np.abs(np.asarray(calibration_residuals, dtype=float))
    scores = list(calibration[~np.isnan(calibration)])

    half_width = np.empty(len(predictions))
    for t in range(len(predictions)):
        half_width[t] = conformal_quantile(scores, level)
        scores.append(abs(actual[t] - predictions[t]))

    return predictions - half_width, predictions + half_width


def prediction_intervals(predictions, actual, spec, level=0.95):
    """Compute (lower, upper) from a model's interval spec, or (None, None) if it has none."""
    if not spec:
        return None, None
    if spec['method'] == 'normal':
        return normal_intervals(predictions, spec['std_errors'], level)
    if spec['method'] == 'conformal':
        return conformal_intervals(predictions, actual, spec['calibration_residuals'], level)
    raise ValueError(f"Unknown interval method: {spec['method']}")


def interval_metrics(actual, lower, upper):
    """
    Evaluate intervals on the test period.

    Returns:
        dict: Coverage (% of actual values inside the interval; should be close to
        the confidence level) and average interval width (narrower is better at
        the same coverage).
    """
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    return {
        'Coverage (%)': np.mean(inside) * 100,
        'Avg Width': np.mean(upper - lower),
    }
