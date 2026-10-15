"""Visualization utilities for the time series forecasting app."""

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st


class DataVisualizer:
    """Enhanced class for data visualization functions."""

    def plot_time_series(self, df, selected_cols, time_index, title="Time Series Plot"):
        """Plot time series data with enhanced styling."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for col in selected_cols:
            ax.plot(time_index, df[col], label=col, alpha=0.7)

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_predictions_with_intervals(self, time_index, actual, predictions,
                                        lower=None, upper=None, title="Model Predictions"):
        """Plot predictions with confidence intervals."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual values
        ax.plot(time_index, actual, label='Actual', color='black', alpha=0.7)

        # Plot predictions
        ax.plot(time_index, predictions, label='Predictions', color='blue', linestyle='--')

        # Plot confidence intervals if provided
        if lower is not None and upper is not None:
            ax.fill_between(time_index, lower, upper, color='blue', alpha=0.2,
                            label='Prediction Interval')

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_forecast(self, history, forecast, intervals=None, title="Forecast"):
        """Plot historical data with forecast."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax.plot(history.index, history, label='Historical', color='black', alpha=0.7)

        # Plot forecast
        ax.plot(forecast.index, forecast, label='Forecast', color='blue',
                linestyle='--', linewidth=2)

        # Plot prediction intervals if provided
        if intervals is not None:
            ax.fill_between(forecast.index,
                            intervals['lower'], intervals['upper'],
                            color='blue', alpha=0.2, label='Prediction Interval')

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_model_diagnostics(self, residuals, predictions, actual):
        """Plot model diagnostic charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals plot
        axes[0, 0].plot(residuals, marker='o', linestyle='None', alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Residuals Distribution')

        # Q-Q plot
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Normal Q-Q Plot')

        # Predicted vs Actual
        axes[1, 1].scatter(predictions, actual, alpha=0.5)
        axes[1, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()],
                        'r--', lw=2)
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Predicted vs Actual')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_feature_importance(self, importance_df):
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 6))

        importance_df = importance_df.sort_values(ascending=True)
        importance_df.plot(kind='barh', ax=ax)

        ax.set_title('Feature Importance', fontsize=14, pad=20)
        ax.set_xlabel('Importance Score', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_model_comparison(self, results: List[Dict], metrics=('MAE', 'RMSE', 'MAPE')):
        """Plot model comparison charts and return the results table."""
        if not results:
            st.warning("No results to display")
            return None

        results_df = pd.DataFrame(results)
        metrics = [m for m in metrics if m in results_df.columns]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            ax.bar(results_df['Model'], results_df[metric], color='steelblue')
            ax.set_title(f"{metric} (%)" if metric == 'MAPE' else metric)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        return results_df

    def plot_test_predictions(self, actual: pd.Series, predictions: Dict[str, pd.Series], history=None,
                              interval=None):
        """
        Interactive plot of each model's one-step-ahead test predictions against the actual values.

        Args:
            interval: Optional (model_name, lower, upper, level) to draw as a shaded band
        """
        fig = go.Figure()
        if history is not None:
            fig.add_trace(go.Scatter(x=history.index, y=history.values, name='History',
                                     line=dict(color='lightgray')))
        if interval is not None:
            name, lower, upper, level = interval
            fig.add_trace(go.Scatter(x=upper.index, y=upper.values, line=dict(width=0),
                                     showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=lower.index, y=lower.values, line=dict(width=0),
                                     fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)',
                                     name=f'{name} {level:.0%} interval', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name='Actual',
                                 line=dict(color='black', width=2)))
        for name, preds in predictions.items():
            fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name=name,
                                     line=dict(dash='dot')))
        fig.update_layout(title='Test Period: Actual vs. One-Step-Ahead Predictions',
                          xaxis_title='Time', yaxis_title=actual.name, hovermode='x unified')
        st.plotly_chart(fig, width='stretch')

    def plot_training_history(self, history):
        """Plot training history for deep learning models."""
        if not history:
            st.warning("No training history to display")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot training & validation loss
        ax1.plot(history.history['loss'], label='Train')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot training & validation metrics
        for metric in history.history.keys():
            if metric not in ['loss', 'val_loss']:
                ax2.plot(history.history[metric],
                         label=metric.replace('_', ' ').title())
        ax2.set_title('Model Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def display_data_info(self, df):
        """Display enhanced dataset information."""
        st.subheader("Dataset Overview")

        # Basic info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024 ** 2:.2f} MB")

        # Data types and missing values
        st.subheader("Data Types and Missing Values")
        dtypes_df = pd.DataFrame({
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(dtypes_df)

        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        # Additional info for time series
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            st.subheader("Time Series Information")
            freq = pd.infer_freq(df.index)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Start Date", str(df.index[0].date()))
            with col2:
                st.metric("End Date", str(df.index[-1].date()))
            with col3:
                st.metric("Frequency", str(freq) if freq else "Irregular")