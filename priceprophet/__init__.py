"""
PriceProphet v1.2.0 — Multi-model price forecasting, anomaly detection,
price elasticity, seasonality analysis, and market shock simulation.
"""

__version__ = "1.2.0"

from .elasticity import ElasticityResult, PriceElasticity
from .forecaster import EnsembleForecaster, Forecaster
from .seasonality import SeasonalityDetector, SeasonalityResult


class PriceProphet:
    """
    Central facade for PriceProphet — multi-model price intelligence.

    Supports: linear, ridge, polynomial, EMA forecasting + model comparison,
    price elasticity, anomaly detection, seasonality analysis, and shock simulation.

    Example::

        from priceprophet import PriceProphet
        import pandas as pd
        import numpy as np

        pp = PriceProphet()
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=180),
            'price': [100 + i*0.3 + (i % 7)*2 for i in range(180)]
        })

        # Forecast with best model
        forecast = pp.forecast(df, periods=30, model="auto")

        # Compare all models
        comparison = pp.compare_models(df)
        print(comparison)

        # Price elasticity
        prices  = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0])
        demands = pd.Series([100,   90,   80,   88,   70])
        result = pp.elasticity(prices, demands)
        print(result)
    """

    def __init__(self):
        self._forecaster = Forecaster()
        self._ensemble = EnsembleForecaster()
        self._elasticity = PriceElasticity()
        self._seasonality = SeasonalityDetector()

    # ── Forecasting ────────────────────────────────────────────────────────────

    def forecast(self, df, periods: int = 30, model: str = "linear", **kwargs):
        """
        Forecast future prices using the selected model.

        Args:
            df: Historical DataFrame with date and price columns
            periods: Number of periods to forecast
            model: 'linear' | 'ridge' | 'polynomial' | 'ema' | 'auto'
            **kwargs: alpha (ridge), degree (polynomial), span (ema)

        Returns:
            DataFrame with Date, Predicted_Value, Lower_Bound, Upper_Bound
        """
        if model == "auto":
            model = self._auto_select(df)

        dc, vc = self._detect_cols(df)

        if model == "ridge":
            return self._forecaster.fit_predict_ridge(df, dc, vc, periods, alpha=kwargs.get("alpha", 1.0))
        if model == "polynomial":
            return self._forecaster.fit_predict_polynomial(df, dc, vc, periods, degree=kwargs.get("degree", 2))
        if model == "ema":
            return self._forecaster.fit_predict_ema(df, dc, vc, periods, span=kwargs.get("span", 12))
        return self._forecaster.predict(df, periods=periods)

    def forecast_ensemble(self, df, periods: int = 30, cv_split: float = 0.8):
        """
        Forecast using inverse-MAE weighted ensemble of all base models.

        More robust than any single model — automatically weights better
        performers higher based on cross-validation performance.

        Args:
            df: Historical DataFrame with date and price columns.
            periods: Forecast horizon (days).
            cv_split: Train/validation split ratio for weighting.

        Returns:
            DataFrame with Date, Predicted_Value, Lower_Bound, Upper_Bound.
        """
        dc, vc = self._detect_cols(df)
        return self._ensemble.fit_predict(df, dc, vc, periods=periods, cv_split=cv_split)

    def detect_anomalies(self, df, threshold: float = 2.0):
        """Detect price anomalies using Z-score method."""
        return self._forecaster.detect_anomalies(df, threshold=threshold)

    def simulate_impact(self, current_price: float, shock_magnitude: float, shock_type: str = "competitor"):
        """Simulate market shock impact on price and volume."""
        return self._forecaster.simulate_impact(current_price, shock_magnitude, shock_type)

    def compare_models(self, df, cv_split: float = 0.8):
        """
        Compare linear, ridge, polynomial, and EMA models using train/test split.

        Returns:
            DataFrame sorted by MAE with MAE, RMSE, R2 per model
        """
        return self._forecaster.compare_models(df, cv_split=cv_split)

    # ── Elasticity & Seasonality ───────────────────────────────────────────────

    def elasticity(self, price_series, demand_series, method: str = "arc") -> "ElasticityResult":
        """Calculate price elasticity of demand."""
        return self._elasticity.calculate(price_series, demand_series, method=method)

    def seasonality(self, df) -> "SeasonalityResult":
        """Detect weekly, monthly, and custom seasonality in price data."""
        return self._seasonality.detect(df)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _detect_cols(self, df):
        """Auto-detect date and value columns."""
        import numpy as np
        import pandas as pd
        date_col = value_col = None
        for col in df.columns:
            if date_col is None and (
                pd.api.types.is_datetime64_any_dtype(df[col])
                or df[col].dtype == object
            ):
                date_col = col
            elif value_col is None and np.issubdtype(df[col].dtype, np.number):
                value_col = col
        return date_col, value_col

    def _auto_select(self, df) -> str:
        """Auto-select best model based on train/test comparison."""
        try:
            comparison = self._forecaster.compare_models(df)
            best_row = comparison.dropna(subset=["MAE"]).iloc[0]
            return best_row["model"].split("_")[0]
        except Exception:
            return "linear"


__all__ = [
    "PriceProphet",
    "Forecaster",
    "EnsembleForecaster",
    "PriceElasticity",
    "ElasticityResult",
    "SeasonalityDetector",
    "SeasonalityResult",
]
