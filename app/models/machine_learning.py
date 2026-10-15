"""Machine learning model implementations for time series forecasting."""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from .base import TimeSeriesModel


class RandomForestModel(TimeSeriesModel):
    """Random Forest model implementation."""

    def __init__(self, name="Random Forest"):
        super().__init__(name)
        self.n_estimators = 100
        self.max_depth = None
        self.random_state = 42

    def build(self, n_estimators=100, max_depth=None, random_state=42):
        """Build Random Forest model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            oob_score=True,
            n_jobs=-1
        )
        return self

    def train(self, train_data, val_data=None, **kwargs):
        """Train on train + validation data (Random Forest has no early-stopping step)."""
        X_train, y_train = train_data
        if val_data is not None and len(val_data[0]) > 0:
            X_train = pd.concat([X_train, val_data[0]])
            y_train = pd.concat([y_train, val_data[1]])
        self.model.fit(X_train, y_train)
        self._oob_residuals = y_train.to_numpy() - self.model.oob_prediction_
        return self

    def calibration_residuals(self, val_data=None):
        """
        Out-of-sample errors for prediction intervals.

        The forest trains on the validation data, so it uses out-of-bag errors instead:
        each training row is predicted only by trees that didn't see it.
        """
        return self._oob_residuals

    def evaluate(self, test_data):
        """Return one-step-ahead predictions for the test rows."""
        X_test, _ = test_data
        return self.predict(X_test)

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def feature_importance(self, columns):
        """Return feature importances as a Series."""
        return pd.Series(self.model.feature_importances_, index=columns)

    def save(self, path):
        """Save model to file."""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        """Load model from file."""
        import joblib
        self.model = joblib.load(path)


class XGBoostModel(TimeSeriesModel):
    """XGBoost model implementation."""

    def __init__(self, name="XGBoost"):
        super().__init__(name)
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.random_state = 42

    def build(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """Build XGBoost model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        return self

    def train(self, train_data, val_data=None, **kwargs):
        """Train the model, using the validation data for early stopping when available."""
        X_train, y_train = train_data

        if val_data is not None and len(val_data[0]) > 0:
            self.model.set_params(early_stopping_rounds=10)
            self.model.fit(X_train, y_train, eval_set=[val_data], verbose=False)
        else:
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X_train, y_train, verbose=False)
        return self

    def calibration_residuals(self, val_data):
        """Errors on the validation data (not used for fitting) for prediction intervals."""
        X_val, y_val = val_data
        return y_val.to_numpy() - self.predict(X_val)

    def evaluate(self, test_data):
        """Return one-step-ahead predictions for the test rows."""
        X_test, _ = test_data
        return self.predict(X_test)

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def feature_importance(self, columns):
        """Return feature importances as a Series."""
        return pd.Series(self.model.feature_importances_, index=columns)

    def save(self, path):
        """Save model to file."""
        self.model.save_model(path)

    def load(self, path):
        """Load model from file."""
        self.model.load_model(path)


def create_ml_model(model_type, **kwargs):
    """Factory function to create ML models."""
    if model_type.upper() == 'RANDOM FOREST':
        return RandomForestModel(**kwargs)
    elif model_type.upper() == 'XGBOOST':
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
