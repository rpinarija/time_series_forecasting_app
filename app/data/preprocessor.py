"""Data preprocessing and dataset creation for all model families.

Every model is evaluated on the same chronological split of the target series:

    | train | validation | test |

The test window is always the last ``test_split`` fraction of the series, so
metrics from traditional, ML and deep learning models are directly comparable.
The validation window is used by models that tune themselves while training
(deep learning and XGBoost early stopping); models without such a step train
on train + validation instead.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class TimeSeriesPreprocessor:
    """Class for preprocessing time series data."""

    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def split_points(n, val_split=0.2, test_split=0.2):
        """
        Return (train_end, val_end) positions for a series of length n.

        Train is [0, train_end), validation is [train_end, val_end) and test is [val_end, n).
        """
        test_size = max(1, int(n * test_split))
        val_size = max(1, int(n * val_split))
        train_end = n - test_size - val_size
        if train_end < 1:
            raise ValueError("Validation and test splits leave no training data.")
        return train_end, n - test_size

    def create_features(self, series, lags=None, add_date_features=True):
        """
        Create lag, rolling and calendar features for ML models.

        All features at time t only use values up to t-1, so they are safe to use
        for one-step-ahead forecasting.
        """
        if lags is None:
            lags = [1, 7, 14, 30]

        features = pd.DataFrame(index=series.index)

        for lag in lags:
            features[f'lag_{lag}'] = series.shift(lag)

        # Rolling statistics are computed on the lagged series to avoid leaking y_t
        lagged = series.shift(1)
        for window in [7, 14, 30]:
            features[f'rolling_mean_{window}'] = lagged.rolling(window=window).mean()
            features[f'rolling_std_{window}'] = lagged.rolling(window=window).std()

        if add_date_features and isinstance(series.index, pd.DatetimeIndex):
            features['year'] = series.index.year
            features['month'] = series.index.month
            features['day'] = series.index.day
            features['day_of_week'] = series.index.dayofweek
            features['quarter'] = series.index.quarter

        return features

    def prepare_ml_data(self, series, val_split=0.2, test_split=0.2, lags=None, predict_changes=True):
        """
        Build feature matrices for ML models split into train, validation and test.

        Tree models can't predict values outside the range seen in training, so they
        fail on trending series. With ``predict_changes`` the target becomes the
        change from the previous value (y_t - y_{t-1}) and the features are built
        from past changes; a level forecast is then previous value + predicted change.

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test), test_offset)
            where test_offset is added to model predictions to get forecasts in the
            series' units (the previous value when predicting changes, otherwise 0).
        """
        train_end, val_end = self.split_points(len(series), val_split, test_split)

        if predict_changes:
            target = series.diff()
            offset = series.shift(1)
        else:
            target = series
            offset = pd.Series(0.0, index=series.index)

        features = self.create_features(target, lags=lags)
        position = pd.Series(np.arange(len(series)), index=series.index)
        valid = (features.notna().all(axis=1) & target.notna()).to_numpy()

        X, y, pos = features[valid], target[valid], position[valid].to_numpy()

        train_mask = pos < train_end
        val_mask = (pos >= train_end) & (pos < val_end)
        test_mask = pos >= val_end

        if train_mask.sum() < 10:
            raise ValueError("Not enough training rows after creating lag features. "
                             "Use a longer series or smaller splits.")

        return (
            (X[train_mask], y[train_mask]),
            (X[val_mask], y[val_mask]),
            (X[test_mask], y[test_mask]),
            offset[valid][test_mask].to_numpy(),
        )

    def create_cross_validation_splits(self, X, y, n_splits=5):
        """Create time series cross validation splits."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            splits.append(((X_train, y_train), (X_val, y_val)))

        return splits

    def prepare_dl_datasets(self, series, sequence_length, batch_size, val_split=0.2, test_split=0.2):
        """
        Create windowed TensorFlow datasets for deep learning models.

        Each sample is the previous ``sequence_length`` values and the value that
        follows them. Samples are assigned to train/val/test by the position of
        their target, so the test targets are exactly the shared test window.
        The scaler is fit on the training portion only to avoid look-ahead bias.

        Returns:
            tuple: (train_dataset, val_dataset, (X_val, y_val), (X_test, y_test), scaler)
            where the X/y pairs are scaled numpy arrays.
        """
        if series.isna().any():
            raise ValueError("Data contains missing values. Apply a missing-value strategy first.")

        n = len(series)
        train_end, val_end = self.split_points(n, val_split, test_split)

        if train_end <= sequence_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) must be shorter than the training "
                f"portion ({train_end} points). Reduce it or the validation/test splits."
            )

        data = series.to_numpy(dtype='float32').reshape(-1, 1)
        self.scaler = StandardScaler().fit(data[:train_end])
        scaled = self.scaler.transform(data).astype('float32')

        targets_pos = np.arange(sequence_length, n)
        X = np.stack([scaled[t - sequence_length:t] for t in targets_pos])
        y = scaled[targets_pos]

        train_mask = targets_pos < train_end
        val_mask = (targets_pos >= train_end) & (targets_pos < val_end)
        test_mask = targets_pos >= val_end

        train_dataset = (tf.data.Dataset.from_tensor_slices((X[train_mask], y[train_mask]))
                         .shuffle(1024).batch(batch_size))
        val_dataset = tf.data.Dataset.from_tensor_slices((X[val_mask], y[val_mask])).batch(batch_size)

        return (train_dataset, val_dataset, (X[val_mask], y[val_mask]),
                (X[test_mask], y[test_mask]), self.scaler)
