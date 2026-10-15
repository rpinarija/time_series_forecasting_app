"""Deep learning training utilities."""

import matplotlib.pyplot as plt
import streamlit as st
from tensorflow import keras


class StreamlitProgressCallback(keras.callbacks.Callback):
    """Show epoch progress and metrics in Streamlit while a Keras model trains."""

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0.0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.epochs)
        metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in (logs or {}).items())
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs}  {metrics_str}")

    def on_train_end(self, logs=None):
        self.progress_bar.empty()
        self.status_text.empty()


class ModelTrainer:
    """Class for handling deep learning model training and evaluation."""

    def __init__(self, model):
        self.model = model

    def create_callbacks(self, epochs):
        """Create training callbacks."""
        return [
            StreamlitProgressCallback(epochs),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

    def train_and_evaluate(self, train_dataset, val_dataset, test_data, epochs, scaler):
        """
        Train the model and return (test predictions in original units, history dict).

        Early stopping watches the validation loss and restores the best weights.
        """
        history = self.model.train(
            train_dataset,
            val_dataset,
            epochs=epochs,
            callbacks=self.create_callbacks(epochs)
        )
        return self.model.evaluate(test_data, scaler=scaler), history.history

    @staticmethod
    def plot_training_history(model_name, history):
        """Plot a Keras history dict (loss and validation loss per epoch)."""
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title(f'{model_name} Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE, scaled)')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
