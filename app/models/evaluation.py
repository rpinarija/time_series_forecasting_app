"""Model evaluation utilities."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(actual, predicted):
    """Mean absolute percentage error, ignoring points where the actual value is zero."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return np.nan
    return np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100


class ModelEvaluator:
    """Class for model evaluation functions."""

    @staticmethod
    def calculate_metrics(actual, predicted):
        """Calculate a comprehensive set of evaluation metrics."""
        actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
        residuals = actual - predicted

        return {
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAPE': mape(actual, predicted),
            'R2': r2_score(actual, predicted),
            'Mean Error': np.mean(residuals),
            'Std Error': np.std(residuals),
            'Skewness': stats.skew(residuals),
            'Kurtosis': stats.kurtosis(residuals),
        }

    @staticmethod
    def test_residuals(residuals):
        """Perform statistical tests on residuals."""
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

        residuals = np.asarray(residuals, dtype=float)
        tests = {}

        stat, p_value = stats.normaltest(residuals)
        tests['Normality'] = {
            'statistic': stat,
            'p_value': p_value,
            'null_hypothesis': 'Residuals are normally distributed'
        }

        lb = acorr_ljungbox(residuals, lags=[10])
        tests['Autocorrelation'] = {
            'statistic': lb['lb_stat'].iloc[0],
            'p_value': lb['lb_pvalue'].iloc[0],
            'null_hypothesis': 'No autocorrelation present (Ljung-Box, 10 lags)'
        }

        # Regress squared residuals on a trend to detect changing variance over time
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])
        bp_test = het_breuschpagan(residuals, exog)
        tests['Heteroscedasticity'] = {
            'statistic': bp_test[0],
            'p_value': bp_test[1],
            'null_hypothesis': 'Homoscedasticity present (Breusch-Pagan vs. time)'
        }

        return tests

    @staticmethod
    def diebold_mariano(errors1, errors2):
        """
        Diebold-Mariano test for equal one-step-ahead forecast accuracy (squared-error loss).

        Returns:
            tuple: (statistic, p_value). A negative statistic means model 1 is more accurate.
        """
        d = np.asarray(errors1, dtype=float) ** 2 - np.asarray(errors2, dtype=float) ** 2
        var = np.var(d, ddof=1)
        if var == 0:
            return 0.0, 1.0
        dm_stat = np.mean(d) / np.sqrt(var / len(d))
        p_value = 2 * stats.norm.sf(abs(dm_stat))
        return dm_stat, p_value

    @classmethod
    def compare_models(cls, models_results):
        """
        Compare models pairwise with the Diebold-Mariano test.

        Args:
            models_results (dict): {model_name: {'residuals': array}} on the same test period
        """
        names = list(models_results.keys())
        if len(names) < 2:
            return None

        comparisons = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                model1, model2 = names[i], names[j]
                dm_stat, dm_pvalue = cls.diebold_mariano(
                    models_results[model1]['residuals'],
                    models_results[model2]['residuals']
                )
                if dm_pvalue >= 0.05:
                    better = 'No significant difference'
                else:
                    better = model1 if dm_stat < 0 else model2

                comparisons.append({
                    'Model 1': model1,
                    'Model 2': model2,
                    'DM Statistic': dm_stat,
                    'p-value': dm_pvalue,
                    'More Accurate (5%)': better
                })

        return pd.DataFrame(comparisons)

    def display_evaluation_results(self, metrics, tests=None):
        """Display evaluation results in Streamlit."""
        cols = st.columns(4)
        for i, metric in enumerate(['MAE', 'RMSE', 'MAPE', 'R2']):
            value = metrics.get(metric, np.nan)
            label = f"{metric} (%)" if metric == 'MAPE' else metric
            cols[i].metric(label, f"{value:.4f}")

        if tests:
            rows = [
                {
                    'Test': name,
                    'Null Hypothesis': result['null_hypothesis'],
                    'Statistic': result['statistic'],
                    'p-value': result['p_value'],
                }
                for name, result in tests.items()
            ]
            st.caption("Residual diagnostics")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
