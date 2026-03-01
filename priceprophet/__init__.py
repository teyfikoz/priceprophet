"""
PriceProphet - Time-Series Forecasting & Analysis
"""

__version__ = "0.3.0"

from .forecaster import Forecaster


class PriceProphet:
    """
    Central facade for PriceProphet library.
    Provides unified access to price forecasting and anomaly detection.
    """
    def __init__(self):
        self._forecaster = Forecaster()

    def forecast(self, df, periods: int = 30):
        """Generates a price forecast."""
        return self._forecaster.predict(df, periods=periods)

    def detect_anomalies(self, df):
        """Identifies price anomalies in historical data."""
        return self._forecaster.detect_anomalies(df)

    def simulate_impact(self, current_price: float, shock_magnitude: float, shock_type: str = "competitor"):
        """Simulates market impact of price shocks."""
        return self._forecaster.simulate_impact(current_price, shock_magnitude, shock_type)

__all__ = ["PriceProphet", "Forecaster"]
