# Time Series Forecasting Application

A Streamlit application that guides you through univariate time series forecasting: load a
dataset, explore and preprocess it, then train and compare traditional, machine learning and
deep learning models on the same held-out test period, complete with prediction intervals.

Every model comes with sensible defaults, so you can get results without tuning anything, and
every parameter can be adjusted from the sidebar.

## Features

- **Three data sources:** upload a CSV, pick a CSV from any public GitHub folder, or use the
  built-in example stock price data
- **Automatic data handling:** date columns are detected and used as the time index, and text
  columns are converted to dates or numbers where possible
- **Exploration:** summary statistics, missing-value counts, interactive time series plots and a
  correlation heatmap
- **Preprocessing:** missing values, outliers, transformations and seasonal decomposition
- **Seven models plus a naive baseline**, all evaluated the same way so their metrics are directly
  comparable
- **Prediction intervals** at an adjustable confidence level, with coverage and width metrics
- **Diagnostics:** residual plots and statistical tests, feature importance, training history and
  a Diebold-Mariano test for comparing models

## Installation

Tested with Python 3.12. Use a Python version that TensorFlow publishes builds for; the newest
Python releases often aren't supported yet.

1. Clone the repository:
   ```bash
   git clone https://github.com/rpinarija/time_series_forecasting_app.git
   cd time_series_forecasting_app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
streamlit run app/main.py
```

The application opens at `http://localhost:8501`.

### Workflow

1. **Choose a data source** in the sidebar. For GitHub, paste a folder URL such as
   `https://github.com/<owner>/<repo>/tree/<branch>/<folder>` and pick a CSV from the list.
2. **Explore the data** in the overview and plots.
3. **Check the time index.** It's set automatically when a date column is found; otherwise choose
   one under *Time Index Settings*.
4. **Select the target variable** (the column to forecast) in the sidebar.
5. **Preprocess if needed** under *Data Preprocessing*. Applied steps are listed and can be undone
   with *Reset to Original Data*.
6. **Choose the dataset splits, models and parameters** in the sidebar.
7. **Click Train Models** and compare the results.

### Preprocessing Options

| Step | Options |
|---|---|
| Missing values | Drop, forward fill, backward fill, linear interpolation, mean |
| Outliers (target only) | IQR method, Z-score method, winsorization |
| Transformation (target only) | Log, square root, Box-Cox, standardization, min-max scaling |
| Decomposition (target only) | Replace the target with its trend, seasonal or residual component |

Metrics are reported in the units of the transformed target. If a series' swings grow as its level
rises (e.g. airline passengers, sales), a **log transform** usually improves both accuracy and
prediction interval coverage.

### Models

| Model | When to use it | Key defaults |
|---|---|---|
| **ARIMA** | Data without a seasonal pattern; a strong, fast statistical baseline | Orders chosen automatically by AIC |
| **SARIMA** | Data with a repeating seasonal pattern (monthly, quarterly, weekly) | Seasonal period guessed from the date frequency |
| **Random Forest** | Nonlinear patterns, robust with little tuning | 100 trees, predicts the change from the previous value |
| **XGBoost** | Nonlinear patterns, often the most accurate ML model with tuning | 100 trees, early stopping, predicts the change from the previous value |
| **Simple RNN** | Learning sequence patterns from longer histories | 60-step input window, 64 units |
| **LSTM** | Long-range dependencies; needs more data than the others | 60-step input window, 64 units |
| **Stacked LSTM+RNN** | Complex patterns in large datasets | LSTM layer with 128 units feeding an RNN layer |
| **Naive (last value)** | Always included as a baseline: a model that can't beat it isn't adding value | Predicts the previous actual value |

Random Forest and XGBoost predict the change from the previous value by default, because tree
models can't predict values outside the range they saw in training and would flatten out on
trending data. Turn this off in the sidebar for series that move around a stable level.

### How Models Are Evaluated

The series is split chronologically into train, validation and test periods (60/20/20 by default).
Every model makes **one-step-ahead forecasts** over the same test period, using the actual values up
to each step, so the metrics are directly comparable:

- **ARIMA, SARIMA and Random Forest** train on train + validation.
- **XGBoost and the deep learning models** train on the training period and use the validation
  period for early stopping.

Results include MAE, RMSE, MAPE and R², a chart of each model's test predictions, and per-model
diagnostics. They stay on screen while you change other settings.

### Prediction Intervals

A prediction interval is a range the actual value should fall in with a chosen probability (80, 90,
95 or 99%). The level can be changed without retraining.

- **ARIMA and SARIMA** use the model's own forecast standard errors.
- **All other models** use conformal intervals, sized from the model's errors on data it wasn't
  trained on (validation errors, or out-of-bag errors for Random Forest). During the test period,
  each step's error is added once its actual value is known, so intervals adapt if errors grow.

The comparison table reports **coverage** (the share of actual values inside the interval, which
should be close to the chosen level) and **average width** (narrower is better at the same
coverage).

## Project Structure

```
time_series_forecasting_app/
├── app/
│   ├── main.py                    # Streamlit application (entry point)
│   ├── config.py                  # Default parameters and constants
│   ├── data/
│   │   ├── loader.py              # CSV, GitHub and example data loading
│   │   └── preprocessor.py        # Splits, ML features and deep learning windows
│   ├── models/
│   │   ├── base.py                # Common model interface
│   │   ├── traditional.py         # ARIMA, SARIMA
│   │   ├── machine_learning.py    # Random Forest, XGBoost
│   │   ├── deep_learning.py       # Simple RNN, LSTM, Stacked LSTM+RNN
│   │   ├── trainer.py             # Deep learning training loop and progress display
│   │   └── evaluation.py          # Metrics, residual tests, Diebold-Mariano test
│   └── utils/
│       ├── confidence_intervals.py  # Prediction intervals
│       ├── visualization.py       # Charts
│       └── helpers.py             # Validation and environment setup
├── requirements.txt
├── setup.py
└── README.md
```