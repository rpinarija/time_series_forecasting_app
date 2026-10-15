"""Base model class for time series forecasting."""

from abc import ABC, abstractmethod


class TimeSeriesModel(ABC):
    """Abstract base class for all time series models."""

    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def build(self, **kwargs):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, train_data, val_data=None, **kwargs):
        """Train the model. Each model decides how to use the validation data."""
        pass

    @abstractmethod
    def evaluate(self, test_data):
        """Return one-step-ahead predictions for the test period (in the target's units)."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the model."""
        pass
