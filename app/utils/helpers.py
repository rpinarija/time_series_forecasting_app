"""Helper utilities for the time series forecasting app."""

import numpy as np
import pandas as pd
import streamlit as st


def validate_data(df, min_points=100):
    """
    Validate input data.

    Args:
        df: Input dataframe or series
        min_points: Minimum required data points

    Returns:
        bool: Whether data is valid
    """
    if df is None:
        return False

    if len(df) < min_points:
        st.error(f"Dataset too small for reliable training (minimum {min_points} points required)")
        return False

    return True


def get_time_index(df):
    """Get appropriate time index for the dataset."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    try:
        first_col = df.iloc[:, 0]
        converted = pd.to_datetime(first_col, errors='coerce')
        if converted.notnull().all():
            return converted
    except (TypeError, ValueError):
        pass
    return np.arange(len(df))


def infer_seasonal_period(index, default=12):
    """Guess a seasonal period from a datetime index's frequency."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default
    freq = pd.infer_freq(index)
    if freq is None:
        # Irregular spacing (e.g. trading days with holidays): use the median gap
        days = np.median(np.diff(index.values).astype('timedelta64[D]').astype(int))
        return 5 if days <= 1 else 52 if days <= 7 else 12 if days <= 31 else 4 if days <= 92 else default
    freq = freq.upper()
    for prefix, period in [('B', 5), ('D', 7), ('W', 52), ('M', 12), ('Q', 4), ('BQ', 4)]:
        if freq.startswith(prefix):
            return period
    return default


def create_download_button(df, filename="results.csv"):
    """Create a download button for dataframe."""
    st.download_button(
        label="Download results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
    )


def setup_environment():
    """Configure logging and environment variables. Call before importing TensorFlow."""
    import os
    import warnings

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
