from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Forecaster:
    """Simple time-series forecasting with anomaly detection and impact simulation."""

    def __init__(self):
        self.model = LinearRegression()
        self._residual_std = None

    def fit_predict(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30) -> pd.DataFrame:
        """
        Train on data and predict future prices.

        Args:
            df: Historical data
            date_col: Name of date column
            value_col: Name of price/value column
            periods: Days to predict

        Returns:
            DataFrame with 'Date', 'Predicted_Value', 'Lower_Bound', 'Upper_Bound'
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Prepare features (convert dates to ordinal)
        df['ordinal'] = df[date_col].map(pd.Timestamp.toordinal)
        X = df[['ordinal']]
        y = df[value_col]

        # Fit
        self.model.fit(X, y)

        # Calculate residual std for dynamic confidence intervals
        train_predictions = self.model.predict(X)
        residuals = y.values - train_predictions
        self._residual_std = np.std(residuals)

        # Future dates
        last_date = df[date_col].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

        # Predict
        predictions = self.model.predict(future_ordinal)

        # Dynamic confidence interval based on residual std (1.96 * std for 95% CI)
        margin = 1.96 * self._residual_std

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Value': predictions,
            'Lower_Bound': predictions - margin,
            'Upper_Bound': predictions + margin
        })

        return future_df

    def predict(self, df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        """
        Convenience wrapper for fit_predict.
        Auto-detects date and value columns.

        Args:
            df: Historical data with a date column and a numeric column
            periods: Days to predict

        Returns:
            DataFrame with predictions
        """
        date_col = None
        value_col = None

        for col in df.columns:
            if df[col].dtype in ('datetime64[ns]', 'object'):
                try:
                    pd.to_datetime(df[col])
                    date_col = col
                except (ValueError, TypeError):
                    pass
            elif np.issubdtype(df[col].dtype, np.number) and value_col is None:
                value_col = col

        if date_col is None or value_col is None:
            raise ValueError("Could not auto-detect date and value columns. Use fit_predict() with explicit column names.")

        return self.fit_predict(df, date_col, value_col, periods)

    def detect_anomalies(self, df: pd.DataFrame, date_col: str = None, value_col: str = None, threshold: float = 2.0) -> pd.DataFrame:
        """
        Identifies price anomalies using Z-score method.

        Args:
            df: Historical price data
            date_col: Date column name (auto-detected if None)
            value_col: Value column name (auto-detected if None)
            threshold: Z-score threshold for anomaly (default 2.0)

        Returns:
            DataFrame with anomaly flags and Z-scores
        """
        df = df.copy()

        # Auto-detect columns if not provided
        if date_col is None or value_col is None:
            for col in df.columns:
                if date_col is None:
                    try:
                        pd.to_datetime(df[col])
                        date_col = col
                    except (ValueError, TypeError):
                        pass
                elif value_col is None and np.issubdtype(df[col].dtype, np.number):
                    value_col = col

        if value_col is None:
            raise ValueError("Could not detect value column.")

        values = df[value_col].values
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            df['z_score'] = 0.0
            df['is_anomaly'] = False
            return df

        df['z_score'] = (values - mean) / std
        df['is_anomaly'] = np.abs(df['z_score']) > threshold

        return df

    def simulate_impact(self, current_price: float, shock_magnitude: float, shock_type: str = "competitor") -> Dict:
        """
        Simulates market impact of price shocks.

        Args:
            current_price: Current price point
            shock_magnitude: Magnitude of shock (e.g., 0.1 for 10%)
            shock_type: Type of shock - "competitor", "supply", "demand", "regulation"

        Returns:
            Dict with simulated impact metrics
        """
        # Impact multipliers by shock type
        multipliers = {
            "competitor": {"price_impact": 0.7, "volume_impact": 0.5, "recovery_days": 14},
            "supply": {"price_impact": 1.2, "volume_impact": 0.8, "recovery_days": 30},
            "demand": {"price_impact": 0.9, "volume_impact": 1.0, "recovery_days": 21},
            "regulation": {"price_impact": 1.5, "volume_impact": 0.3, "recovery_days": 60},
        }

        params = multipliers.get(shock_type, multipliers["competitor"])

        price_change = current_price * shock_magnitude * params["price_impact"]
        new_price = current_price - price_change

        return {
            "current_price": current_price,
            "shock_type": shock_type,
            "shock_magnitude": shock_magnitude,
            "price_impact": round(price_change, 2),
            "new_price": round(new_price, 2),
            "price_change_pct": round(-shock_magnitude * params["price_impact"] * 100, 2),
            "estimated_volume_change_pct": round(-shock_magnitude * params["volume_impact"] * 100, 2),
            "estimated_recovery_days": params["recovery_days"],
        }
