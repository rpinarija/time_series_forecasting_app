"""Main Streamlit application for time series forecasting."""

from utils.helpers import setup_environment

setup_environment()  # Must run before TensorFlow is imported

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow import keras

from config import (
    APP_DESCRIPTION, APP_TITLE, DATA_SOURCE_OPTIONS, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT_RATE,
    DEFAULT_INTERVAL_LEVEL,
    DEFAULT_EPOCHS, DEFAULT_GITHUB_REPO, DEFAULT_LAGS, DEFAULT_LEARNING_RATE, DEFAULT_N_ESTIMATORS,
    DEFAULT_RNN_UNITS, DEFAULT_SEQUENCE_LENGTH, DEFAULT_TEST_SPLIT, DEFAULT_VAL_SPLIT, DL_MODELS,
    EXAMPLE_DATA_URL, MIN_DATA_POINTS, ML_MODELS, MODEL_CATEGORIES, TRADITIONAL_MODELS,
)
from data.loader import get_dtypes_info, get_github_files, load_data, load_example_data
from data.preprocessor import TimeSeriesPreprocessor
from models.deep_learning import LSTMModel, SimpleRNNModel, StackedModel
from models.evaluation import ModelEvaluator
from models.machine_learning import create_ml_model
from models.trainer import ModelTrainer
from models.traditional import create_traditional_model
from utils.confidence_intervals import interval_metrics, prediction_intervals
from utils.helpers import create_download_button, infer_seasonal_period, validate_data
from utils.visualization import DataVisualizer

NAIVE_BASELINE = 'Naive (last value)'


def main():
    """Main application function."""
    st.set_page_config(layout="wide", page_title="Time Series Forecasting App")
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)

    raw_df, source_key = select_data_source()
    if raw_df is None:
        st.info("Select a data source in the sidebar to get started.")
        return

    df = get_working_data(raw_df, source_key)

    st.header("Data Exploration & Preprocessing")
    explore_data(df)
    time_index_settings(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("The dataset has no numeric columns to forecast.")
        return

    target_column = st.sidebar.selectbox("Select target variable", numeric_cols)
    preprocessing_options(df, target_column)

    if not validate_data(df, MIN_DATA_POINTS):
        return

    split_params = get_split_parameters()
    selected_models = get_model_selection()
    params = {
        **split_params,
        'traditional': get_traditional_parameters(selected_models, df.index),
        'ml': get_ml_parameters(selected_models),
        'dl': get_dl_parameters(selected_models),
    }
    interval_level = get_interval_parameters()

    st.header("Model Training")
    if not selected_models:
        st.info("Select at least one model in the sidebar.")
    elif st.button("Train Models", type="primary"):
        series = df[target_column].astype(float)
        if series.isna().any():
            st.error("The target has missing values. Apply a missing-value strategy under "
                     "Data Preprocessing first.")
        else:
            st.session_state.results = train_models(series, selected_models, params)
            st.session_state.results['key'] = (source_key, target_column)

    results = st.session_state.get('results')
    if results and results.get('key') == (source_key, target_column):
        display_results(results, interval_level)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def select_data_source():
    """Sidebar data source selection. Returns (dataframe, key identifying the source)."""
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Select data source", DATA_SOURCE_OPTIONS)

    if data_source == "GitHub Repository":
        github_url = st.sidebar.text_input("GitHub repository URL", DEFAULT_GITHUB_REPO)
        csv_files = get_github_files(github_url) if github_url else []
        if not csv_files:
            return None, None
        files = dict(csv_files)
        selected_file = st.sidebar.selectbox("Select a CSV file", list(files))
        return load_data("github", file_url=files[selected_file]), files[selected_file]

    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is None:
            return None, None
        return load_data("upload", uploaded_file=uploaded_file), f"upload:{uploaded_file.file_id}"

    df = load_example_data(EXAMPLE_DATA_URL)
    if df is not None:
        st.sidebar.info("Using example stock closing prices (yfinance)")
    return df, "example"


def get_working_data(raw_df, source_key):
    """
    Keep an editable copy of the data in session state.

    Streamlit reruns the script on every interaction, so changes like setting the
    time index or preprocessing must be stored or they are lost on the next click.
    """
    if st.session_state.get('source_key') != source_key:
        st.session_state.source_key = source_key
        st.session_state.raw_df = raw_df
        st.session_state.df = raw_df.copy()
        st.session_state.preprocessing_log = []
        st.session_state.pop('results', None)
    return st.session_state.df


def update_working_data(df, log_entry):
    """Store modified data, discard stale results and rerun so the page reflects it."""
    st.session_state.df = df
    st.session_state.preprocessing_log.append(log_entry)
    st.session_state.pop('results', None)
    st.rerun()


# ---------------------------------------------------------------------------
# Exploration and preprocessing
# ---------------------------------------------------------------------------

def explore_data(df):
    """Show dataset overview, statistics and plots."""
    visualizer = DataVisualizer()

    with st.expander("📊 Data Overview", expanded=True):
        st.dataframe(df.head())
        visualizer.display_data_info(df)
        st.subheader("Column Details")
        st.dataframe(get_dtypes_info(df))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return

    with st.expander("📈 Time Series Visualization", expanded=True):
        selected_cols = st.multiselect("Select columns to visualize", numeric_cols,
                                       default=numeric_cols[:1])
        if selected_cols:
            st.plotly_chart(px.line(df, y=selected_cols, title="Time Series Plot"),
                            width='stretch')
        if len(selected_cols) > 1:
            corr = df[selected_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                                      zmin=-1, zmax=1, title="Correlation Heatmap"),
                            width='stretch')


def time_index_settings(df):
    """Let the user choose which column is the time index."""
    with st.expander("⏰ Time Index Settings"):
        if isinstance(df.index, pd.DatetimeIndex):
            st.write(f"Current time index: **{df.index.name or 'index'}** "
                     f"({df.index.min().date()} to {df.index.max().date()})")
        else:
            st.warning("No datetime index detected. Choose the column that holds the dates.")

        if len(df.columns) == 0:
            return
        time_col = st.selectbox("Select time index column", df.columns)
        if st.button("Set Time Index"):
            try:
                new_df = df.copy()
                new_df.index = pd.to_datetime(new_df.pop(time_col))
                update_working_data(new_df.sort_index(), f"Set '{time_col}' as time index")
            except (ValueError, TypeError) as e:
                st.error(f"Error setting time index: {e}")


def preprocessing_options(df, target_col):
    """Preprocessing controls; changes are applied to the working data in session state."""
    with st.expander("🔧 Data Preprocessing"):
        col1, col2 = st.columns(2)
        with col1:
            missing_method = st.selectbox(
                "Missing values strategy",
                ["None", "Drop", "Forward Fill", "Backward Fill", "Linear Interpolation", "Mean"]
            )
            outlier_method = st.selectbox(
                "Outlier strategy (target only)",
                ["None", "IQR Method", "Z-Score Method", "Winsorization"]
            )
        with col2:
            transform_method = st.selectbox(
                "Transformation (target only)",
                ["None", "Log", "Square Root", "Box-Cox", "Standardization", "Min-Max Scaling"]
            )
            decomposition = st.selectbox(
                "Replace target with decomposition component",
                ["None", "Trend", "Seasonal", "Residual"]
            )
            period = st.number_input("Decomposition period", min_value=2, max_value=365,
                                     value=infer_seasonal_period(df.index))

        st.caption("Metrics are reported in the units of the (transformed) target.")

        btn1, btn2 = st.columns(2)
        if btn1.button("Apply Preprocessing"):
            try:
                new_df = apply_preprocessing(df.copy(), target_col, missing_method,
                                             outlier_method, transform_method, decomposition, period)
                steps = [f"{name}: {value}" for name, value in [
                    ("Missing", missing_method), ("Outliers", outlier_method),
                    ("Transform", transform_method), ("Decomposition", decomposition)
                ] if value != "None"]
                update_working_data(new_df, f"{target_col} → " + (", ".join(steps) or "no changes"))
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")

        if btn2.button("Reset to Original Data"):
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.preprocessing_log = []
            st.session_state.pop('results', None)
            st.rerun()

        if st.session_state.preprocessing_log:
            st.write("**Applied steps:**")
            for step in st.session_state.preprocessing_log:
                st.write(f"- {step}")


def apply_preprocessing(df, target_col, missing_method, outlier_method, transform_method,
                        decomposition, period):
    """Apply the selected preprocessing steps and return the new dataframe."""
    if missing_method == "Drop":
        df = df.dropna()
    elif missing_method == "Forward Fill":
        df = df.ffill()
    elif missing_method == "Backward Fill":
        df = df.bfill()
    elif missing_method == "Linear Interpolation":
        df = df.interpolate(method='linear', limit_direction='both')
    elif missing_method == "Mean":
        df = df.fillna(df.mean(numeric_only=True))

    target = df[target_col]
    if outlier_method != "None" and target.isna().any():
        raise ValueError("Handle missing values in the target before removing outliers.")

    if outlier_method == "IQR Method":
        q1, q3 = target.quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[target.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    elif outlier_method == "Z-Score Method":
        df = df[np.abs(stats.zscore(target)) < 3]
    elif outlier_method == "Winsorization":
        df[target_col] = np.asarray(stats.mstats.winsorize(target.to_numpy(), limits=[0.05, 0.05]))

    target = df[target_col]
    if transform_method == "Log":
        if (target <= -1).any():
            raise ValueError("Log transform (log1p) requires values greater than -1.")
        df[target_col] = np.log1p(target)
    elif transform_method == "Square Root":
        if (target < 0).any():
            raise ValueError("Square root requires non-negative values.")
        df[target_col] = np.sqrt(target)
    elif transform_method == "Box-Cox":
        if (target <= 0).any():
            raise ValueError("Box-Cox requires strictly positive values.")
        df[target_col] = stats.boxcox(target.to_numpy())[0]
    elif transform_method == "Standardization":
        df[target_col] = (target - target.mean()) / target.std()
    elif transform_method == "Min-Max Scaling":
        df[target_col] = (target - target.min()) / (target.max() - target.min())

    if decomposition != "None":
        if df[target_col].isna().any():
            raise ValueError("Handle missing values in the target before decomposing.")
        result = seasonal_decompose(df[target_col], period=int(period))
        component = {"Trend": result.trend, "Seasonal": result.seasonal,
                     "Residual": result.resid}[decomposition]
        df[target_col] = component
        df = df.dropna(subset=[target_col])  # Trend/residual are undefined at the edges

    return df


# ---------------------------------------------------------------------------
# Sidebar model configuration
# ---------------------------------------------------------------------------

def get_split_parameters():
    """Train/validation/test split shared by every model."""
    st.sidebar.header("Dataset Splits")
    val_split = st.sidebar.slider("Validation Split (%)", 5, 30, int(DEFAULT_VAL_SPLIT * 100)) / 100
    test_split = st.sidebar.slider("Test Split (%)", 5, 30, int(DEFAULT_TEST_SPLIT * 100)) / 100
    st.sidebar.caption("Every model is scored on the same final test window. Validation data is "
                       "used for early stopping (deep learning, XGBoost); ARIMA, SARIMA and "
                       "Random Forest train on train + validation.")
    return {'val_split': val_split, 'test_split': test_split}


def get_interval_parameters():
    """Prediction interval settings. Returns the confidence level (e.g. 0.95) or None."""
    st.sidebar.header("Prediction Intervals")
    show = st.sidebar.checkbox("Show prediction intervals", value=True,
                               help="A range the actual value should fall in with the chosen "
                                    "probability. Changing these settings doesn't require retraining.")
    if not show:
        return None
    level = st.sidebar.select_slider("Confidence level (%)", [80, 90, 95, 99], DEFAULT_INTERVAL_LEVEL)
    return level / 100


def get_model_selection():
    """Return the list of selected model names."""
    st.sidebar.header("Model Selection")
    categories = st.sidebar.multiselect("Select model categories", MODEL_CATEGORIES,
                                        default=['Deep Learning'])
    selected = []
    if 'Traditional' in categories:
        selected += st.sidebar.multiselect("Traditional models", TRADITIONAL_MODELS, default=['ARIMA'])
    if 'Machine Learning' in categories:
        selected += st.sidebar.multiselect("ML models", ML_MODELS, default=['Random Forest'])
    if 'Deep Learning' in categories:
        selected += st.sidebar.multiselect("Deep learning models", DL_MODELS,
                                           default=['Simple RNN', 'LSTM'])
    return selected


def get_traditional_parameters(selected_models, index):
    """ARIMA/SARIMA order settings."""
    if not any(m in TRADITIONAL_MODELS for m in selected_models):
        return {}

    st.sidebar.header("Traditional Model Parameters")
    auto = st.sidebar.checkbox("Select orders automatically (AIC)", value=True)
    params = {'auto': auto, 'order': None, 'seasonal_order': None, 'seasonal_period': 12}

    if not auto:
        c1, c2, c3 = st.sidebar.columns(3)
        params['order'] = (c1.number_input("p", 0, 5, 1), c2.number_input("d", 0, 2, 1),
                           c3.number_input("q", 0, 5, 1))

    if 'SARIMA' in selected_models:
        period = st.sidebar.number_input("Seasonal period (m)", 2, 52, infer_seasonal_period(index))
        params['seasonal_period'] = int(period)
        if not auto:
            c1, c2, c3 = st.sidebar.columns(3)
            params['seasonal_order'] = (c1.number_input("P", 0, 2, 1), c2.number_input("D", 0, 1, 1),
                                        c3.number_input("Q", 0, 2, 1), int(period))
        if period > 24:
            st.sidebar.warning("Large seasonal periods make SARIMA very slow to fit.")
    return params


def get_ml_parameters(selected_models):
    """Random Forest / XGBoost settings."""
    if not any(m in ML_MODELS for m in selected_models):
        return {}

    st.sidebar.header("ML Parameters")
    max_depth = st.sidebar.slider("Max tree depth (0 = unlimited for Random Forest)", 0, 20, 6)
    params = {
        'n_estimators': st.sidebar.slider("Number of trees", 50, 1000, DEFAULT_N_ESTIMATORS, step=50),
        'max_depth': max_depth,
        'lags': DEFAULT_LAGS,
        'predict_changes': st.sidebar.checkbox(
            "Predict changes instead of levels (recommended)", value=True,
            help="Tree models can only predict values within the range they saw in training, so "
                 "they flatten out on trending data. With this on they predict the change from "
                 "the previous value, which is added back to give the forecast. Turn it off for "
                 "series without a trend that move around a stable level."),
    }
    if 'XGBoost' in selected_models:
        params['xgb_learning_rate'] = st.sidebar.number_input(
            "XGBoost learning rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    basis = "past changes" if params['predict_changes'] else "past values"
    st.sidebar.caption(f"Features from {basis}: lags {DEFAULT_LAGS}, rolling mean/std (7, 14, 30), "
                       "and calendar features when the index is a date.")
    return params


def get_dl_parameters(selected_models):
    """Deep learning settings."""
    if not any(m in DL_MODELS for m in selected_models):
        return {}

    st.sidebar.header("Deep Learning Parameters")
    params = {
        'sequence_length': st.sidebar.slider("Sequence length (time steps)", 5, 120,
                                             DEFAULT_SEQUENCE_LENGTH),
        'epochs': st.sidebar.slider("Number of epochs", 5, 100, DEFAULT_EPOCHS),
        'batch_size': st.sidebar.select_slider("Batch size", [8, 16, 32, 64, 128],
                                               DEFAULT_BATCH_SIZE),
        'learning_rate': st.sidebar.number_input("Learning rate", min_value=0.0001, max_value=0.01,
                                                 value=DEFAULT_LEARNING_RATE, step=0.0001,
                                                 format="%.4f"),
        'rnn_units': st.sidebar.slider("RNN/LSTM units", 16, 256, DEFAULT_RNN_UNITS),
        'dropout_rate': st.sidebar.slider("Dropout rate", 0.0, 0.5, DEFAULT_DROPOUT_RATE),
    }
    if 'Stacked LSTM+RNN' in selected_models:
        params['stacked_units'] = st.sidebar.slider("Stacked LSTM layer units", 32, 256,
                                                    params['rnn_units'] * 2)
    return params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(series, selected_models, params):
    """
    Train every selected model and collect one-step-ahead test predictions.

    Returns:
        dict: actual test values, history, and per-model predictions and details
    """
    val_split, test_split = params['val_split'], params['test_split']
    preprocessor = TimeSeriesPreprocessor()
    train_end, val_end = preprocessor.split_points(len(series), val_split, test_split)

    actual = series.iloc[val_end:]
    train, val = series.iloc[:train_end], series.iloc[train_end:val_end]

    predictions = {NAIVE_BASELINE: series.shift(1).iloc[val_end:]}
    details = {NAIVE_BASELINE: {
        'Info': 'Predicts each value as the previous actual value.',
        'interval': {'method': 'conformal',
                     'calibration_residuals': series.diff().iloc[:val_end].dropna().to_numpy(),
                     'source': 'Conformal, from its errors before the test period'},
    }}

    test_range = f"{actual.index[0]} to {actual.index[-1]}"
    if isinstance(actual.index, pd.DatetimeIndex):
        test_range = f"{actual.index[0].date()} to {actual.index[-1].date()}"
    st.write(f"Train: {len(train)} points · Validation: {len(val)} points · "
             f"Test: {len(actual)} points ({test_range})")

    def run(model_name, fn):
        with st.status(f"Training {model_name}...", expanded=True) as status:
            try:
                preds, info = fn()
                predictions[model_name] = pd.Series(np.asarray(preds, dtype=float),
                                                    index=actual.index)
                details[model_name] = info
                status.update(label=f"{model_name} trained", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                status.update(label=f"{model_name} failed", state="error")

    for name in [m for m in selected_models if m in TRADITIONAL_MODELS]:
        run(name, lambda name=name: train_traditional(name, train, val, actual, params['traditional']))

    ml_selected = [m for m in selected_models if m in ML_MODELS]
    if ml_selected:
        ml_data = preprocessor.prepare_ml_data(series, val_split, test_split, params['ml']['lags'],
                                               params['ml']['predict_changes'])
        for name in ml_selected:
            run(name, lambda name=name: train_ml(name, ml_data, params['ml']))

    dl_selected = [m for m in selected_models if m in DL_MODELS]
    if dl_selected:
        dl_params = params['dl']
        try:
            dl_data = preprocessor.prepare_dl_datasets(
                series, dl_params['sequence_length'], dl_params['batch_size'], val_split, test_split)
        except ValueError as e:
            st.error(f"Could not prepare deep learning data: {e}")
            dl_selected = []
        for name in dl_selected:
            run(name, lambda name=name: train_dl(name, dl_data, dl_params))

    return {
        'actual': actual,
        'history': series.iloc[:val_end],
        'predictions': predictions,
        'details': details,
    }


def train_traditional(model_name, train, val, actual, params):
    """Fit ARIMA/SARIMA and return (test predictions, details)."""
    if model_name == 'SARIMA':
        model = create_traditional_model(model_name, seasonal_period=params['seasonal_period'])
        model.build(order=params['order'], seasonal_order=params['seasonal_order'])
    else:
        model = create_traditional_model(model_name).build(order=params['order'])

    st.write("Selecting orders and fitting..." if params['auto'] else "Fitting...")
    model.train(train, val)
    preds = model.evaluate(actual)

    info = {'Order (p,d,q)': str(tuple(int(v) for v in model.order))}
    if model_name == 'SARIMA':
        info['Seasonal order (P,D,Q,m)'] = str(tuple(int(v) for v in model.seasonal_order))
    info['AIC'] = round(float(model.fitted_model.aic), 2)
    info['interval'] = {'method': 'normal', 'std_errors': model.test_std_errors,
                        'source': "Model-based, from the model's forecast standard errors"}
    return preds, info


def train_ml(model_name, ml_data, params):
    """Fit Random Forest/XGBoost and return (test predictions, details)."""
    train_data, val_data, test_data, test_offset = ml_data
    model = create_ml_model(model_name)
    if model_name == 'XGBoost':
        model.build(n_estimators=params['n_estimators'], max_depth=params['max_depth'] or 6,
                    learning_rate=params['xgb_learning_rate'])
    else:
        model.build(n_estimators=params['n_estimators'], max_depth=params['max_depth'] or None)

    model.train(train_data, val_data)
    info = {'Target': 'change from previous value' if params['predict_changes'] else 'level',
            'importance': model.feature_importance(train_data[0].columns)}
    if model_name == 'XGBoost' and getattr(model.model, 'best_iteration', None) is not None:
        info['Trees used (early stopping)'] = model.model.best_iteration + 1
    info['interval'] = {
        'method': 'conformal',
        'calibration_residuals': model.calibration_residuals(val_data),
        'source': ('Conformal, from out-of-bag errors' if model_name == 'Random Forest'
                   else 'Conformal, from validation errors'),
    }
    return model.evaluate(test_data) + test_offset, info


def train_dl(model_name, dl_data, params):
    """Build, train and evaluate a deep learning model; return (test predictions, details)."""
    train_dataset, val_dataset, val_data, test_data, scaler = dl_data
    seq_len = params['sequence_length']

    if model_name == 'Simple RNN':
        model = SimpleRNNModel(model_name, seq_len)
        model.build(units=params['rnn_units'], dropout_rate=params['dropout_rate'])
    elif model_name == 'LSTM':
        model = LSTMModel(model_name, seq_len)
        model.build(units=params['rnn_units'], dropout_rate=params['dropout_rate'])
    else:
        model = StackedModel(model_name, seq_len)
        model.build(lstm_units=params['stacked_units'], rnn_units=params['rnn_units'],
                    dropout_rate=params['dropout_rate'])

    model.model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                        loss='mse', metrics=['mae'])

    preds, history = ModelTrainer(model).train_and_evaluate(
        train_dataset, val_dataset, test_data, params['epochs'], scaler)

    val_actual = scaler.inverse_transform(val_data[1]).ravel()
    val_residuals = val_actual - model.evaluate(val_data, scaler=scaler)
    return preds, {
        'Epochs run': len(history['loss']),
        'training_history': history,
        'interval': {'method': 'conformal', 'calibration_residuals': val_residuals,
                     'source': 'Conformal, from validation errors'},
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def display_results(results, interval_level=None):
    """Show the model comparison, prediction plot and per-model details."""
    visualizer = DataVisualizer()
    evaluator = ModelEvaluator()
    actual, predictions = results['actual'], results['predictions']

    metrics = {name: evaluator.calculate_metrics(actual, preds) for name, preds in predictions.items()}
    rows = [{'Model': name, **{k: m[k] for k in ['MAE', 'RMSE', 'MAPE', 'R2']}}
            for name, m in metrics.items()]

    intervals = {}
    if interval_level:
        for row in rows:
            spec = results['details'].get(row['Model'], {}).get('interval')
            lower, upper = prediction_intervals(predictions[row['Model']], actual, spec, interval_level)
            if lower is not None:
                intervals[row['Model']] = (pd.Series(lower, index=actual.index),
                                           pd.Series(upper, index=actual.index))
                row.update(interval_metrics(actual, lower, upper))

    st.header("Model Comparison")
    results_df = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    styled = (results_df.style
              .highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#c8e6c9')
              .highlight_max(subset=['R2'], color='#c8e6c9')
              .format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'MAPE': '{:.2f}%', 'R2': '{:.4f}',
                       'Coverage (%)': '{:.1f}%', 'Avg Width': '{:.4f}'}, na_rep='-'))
    st.dataframe(styled, hide_index=True)
    st.caption(f"'{NAIVE_BASELINE}' is a baseline: a model that can't beat it isn't adding value.")
    if intervals:
        st.caption(f"Prediction intervals at {interval_level:.0%}: **Coverage** is the share of actual "
                   f"test values inside the interval and should be close to {interval_level:.0%}. "
                   "**Avg Width** is how wide the intervals are; narrower is better at the same coverage. "
                   "Coverage well below target means test errors were larger than the model expected. "
                   "If the series' swings grow as its level rises, try a Log transform under "
                   "Data Preprocessing. With a short test period, each point moves coverage a lot.")
    create_download_button(results_df, "forecast_results.csv")

    visualizer.plot_model_comparison(rows)

    band = None
    if intervals:
        names = list(intervals)
        best = results_df['Model'].iloc[0]
        band_model = st.selectbox("Show prediction interval for", names,
                                  index=names.index(best) if best in names else 0)
        band = (band_model, *intervals[band_model], interval_level)
    visualizer.plot_test_predictions(actual, predictions, history=results['history'], interval=band)

    residuals = {name: {'residuals': actual.to_numpy() - preds.to_numpy()}
                 for name, preds in predictions.items()}
    dm_table = evaluator.compare_models(residuals)
    if dm_table is not None:
        with st.expander("Statistical comparison (Diebold-Mariano test)"):
            st.dataframe(dm_table.style.format({'DM Statistic': '{:.3f}', 'p-value': '{:.4f}'}),
                         hide_index=True)

    st.header("Model Details")
    for name, preds in predictions.items():
        info = results['details'].get(name, {})
        with st.expander(f"**{name}**"):
            evaluator.display_evaluation_results(metrics[name],
                                                 evaluator.test_residuals(residuals[name]['residuals']))
            for key, value in info.items():
                if key not in ('importance', 'training_history', 'interval'):
                    st.write(f"**{key}:** {value}")
            if 'interval' in info:
                st.write(f"**Prediction interval method:** {info['interval']['source']}")
            if 'training_history' in info:
                ModelTrainer.plot_training_history(name, info['training_history'])
            if 'importance' in info:
                visualizer.plot_feature_importance(info['importance'])
            visualizer.plot_model_diagnostics(residuals[name]['residuals'], preds.to_numpy(),
                                              actual.to_numpy())


if __name__ == "__main__":
    main()
