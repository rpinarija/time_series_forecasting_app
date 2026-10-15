"""Traditional time series model implementations."""

import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from .base import TimeSeriesModel


def _combine(train_data, val_data):
    """Concatenate train and validation series into one numpy array."""
    parts = [np.asarray(train_data, dtype=float)]
    if val_data is not None and len(val_data) > 0:
        parts.append(np.asarray(val_data, dtype=float))
    return np.concatenate(parts)


def _one_step_predictions(fitted_model, n_fit, test_data):
    """
    One-step-ahead predictions and their standard errors over the test period.

    The fitted parameters are kept fixed; the model's state is updated with each
    actual test observation before predicting the next one, matching how the ML
    and deep learning models are evaluated.
    """
    test_values = np.asarray(test_data, dtype=float)
    extended = fitted_model.append(test_values, refit=False)
    prediction = extended.get_prediction(start=n_fit, end=n_fit + len(test_values) - 1)
    return np.asarray(prediction.predicted_mean), np.asarray(prediction.se_mean)


class ARIMAModel(TimeSeriesModel):
    """ARIMA model implementation."""

    def __init__(self, name="ARIMA"):
        super().__init__(name)
        self.order = None
        self.fitted_model = None
        self.n_fit = 0
        self.test_std_errors = None

    def build(self, order=None):
        """
        Build ARIMA model with specified or auto-determined order.

        Args:
            order (tuple): ARIMA order (p,d,q); None selects it automatically by AIC
        """
        self.order = order
        return self

    def auto_order(self, train_data):
        """Automatically determine ARIMA order using pmdarima."""
        auto_arima = pm.auto_arima(
            train_data,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        return auto_arima.order

    def train(self, train_data, val_data=None, **kwargs):
        """Fit ARIMA on train + validation data (there is no early-stopping step)."""
        data = _combine(train_data, val_data)
        if self.order is None:
            self.order = self.auto_order(data)

        self.fitted_model = ARIMA(data, order=self.order).fit()
        self.n_fit = len(data)
        return self

    def predict(self, steps=1):
        """Forecast `steps` periods past the end of the fitted data."""
        return self.fitted_model.forecast(steps=steps)

    def evaluate(self, test_data):
        """Return one-step-ahead predictions for the test period (standard errors are kept
        in ``test_std_errors`` for prediction intervals)."""
        predictions, self.test_std_errors = _one_step_predictions(self.fitted_model, self.n_fit, test_data)
        return predictions

    def save(self, path):
        """Save model to file."""
        self.fitted_model.save(path)

    def load(self, path):
        """Load model from file."""
        self.fitted_model = ARIMAResults.load(path)


class SARIMAModel(TimeSeriesModel):
    """SARIMA model implementation."""

    def __init__(self, name="SARIMA", seasonal_period=12):
        super().__init__(name)
        self.seasonal_period = seasonal_period
        self.order = None
        self.seasonal_order = None
        self.fitted_model = None
        self.n_fit = 0
        self.test_std_errors = None

    def build(self, order=None, seasonal_order=None):
        """
        Build SARIMA model with specified or auto-determined orders.

        Args:
            order (tuple): ARIMA order (p,d,q)
            seasonal_order (tuple): Seasonal order (P,D,Q,s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        return self

    def auto_order(self, train_data):
        """Automatically determine SARIMA orders using pmdarima."""
        auto_arima = pm.auto_arima(
            train_data,
            seasonal=True,
            m=self.seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        return auto_arima.order, auto_arima.seasonal_order

    def train(self, train_data, val_data=None, **kwargs):
        """Fit SARIMA on train + validation data (there is no early-stopping step)."""
        data = _combine(train_data, val_data)
        if self.order is None or self.seasonal_order is None:
            self.order, self.seasonal_order = self.auto_order(data)

        self.fitted_model = SARIMAX(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order
        ).fit(disp=False)
        self.n_fit = len(data)
        return self

    def predict(self, steps=1):
        """Forecast `steps` periods past the end of the fitted data."""
        return self.fitted_model.forecast(steps=steps)

    def evaluate(self, test_data):
        """Return one-step-ahead predictions for the test period (standard errors are kept
        in ``test_std_errors`` for prediction intervals)."""
        predictions, self.test_std_errors = _one_step_predictions(self.fitted_model, self.n_fit, test_data)
        return predictions

    def save(self, path):
        """Save model to file."""
        self.fitted_model.save(path)

    def load(self, path):
        """Load model from file."""
        self.fitted_model = SARIMAXResults.load(path)


def create_traditional_model(model_type, **kwargs):
    """
    Factory function to create traditional models.

    Args:
        model_type (str): Type of model ('ARIMA' or 'SARIMA')
        **kwargs: Additional arguments for model construction

    Returns:
        TimeSeriesModel: Instance of requested model
    """
    if model_type.upper() == 'ARIMA':
        return ARIMAModel(**kwargs)
    elif model_type.upper() == 'SARIMA':
        return SARIMAModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
