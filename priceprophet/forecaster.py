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
        X = df[['ordinal']].values  # numpy array — avoids sklearn feature-name warnings
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
            # Use pd.api.types for dtype detection — robust across pandas 1.x/2.x
            # (pandas 2.x may use datetime64[us] instead of datetime64[ns])
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
            elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                try:
                    pd.to_datetime(df[col])
                    date_col = col
                except (ValueError, TypeError):
                    pass
            elif pd.api.types.is_numeric_dtype(df[col]) and value_col is None:
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

        # Auto-detect columns if not provided (pandas 2.x compatible)
        if date_col is None or value_col is None:
            for col in df.columns:
                if date_col is None and pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                elif date_col is None and df[col].dtype == object:
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

    def fit_predict_ridge(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, alpha: float = 1.0) -> pd.DataFrame:
        """Ridge regression forecasting (L2 regularization, robust to outliers)."""
        from sklearn.linear_model import Ridge
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df['ordinal'] = df[date_col].map(pd.Timestamp.toordinal)
        X = df[['ordinal']].values  # numpy array — avoids sklearn feature-name warnings
        y = df[value_col]
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        residual_std = np.std(y.values - model.predict(X))
        last_date = df[date_col].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        future_X = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        predictions = model.predict(future_X)
        margin = 1.96 * residual_std
        return pd.DataFrame({
            'Date': future_dates, 'Predicted_Value': predictions,
            'Lower_Bound': predictions - margin, 'Upper_Bound': predictions + margin,
            'model': 'ridge',
        })

    def fit_predict_polynomial(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, degree: int = 2) -> pd.DataFrame:
        """Polynomial regression forecasting (captures non-linear trends)."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df['ordinal'] = df[date_col].map(pd.Timestamp.toordinal)
        X = df[['ordinal']].values  # numpy array — avoids sklearn feature-name warnings
        y = df[value_col]
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        model.fit(X, y)
        residual_std = np.std(y.values - model.predict(X))
        last_date = df[date_col].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        future_X = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        predictions = model.predict(future_X)
        margin = 1.96 * residual_std
        return pd.DataFrame({
            'Date': future_dates, 'Predicted_Value': predictions,
            'Lower_Bound': predictions - margin, 'Upper_Bound': predictions + margin,
            'model': f'polynomial_deg{degree}',
        })

    def fit_predict_ema(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30, span: int = 12) -> pd.DataFrame:
        """Exponential Moving Average forecasting (follows recent trends)."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        ema_values = df[value_col].ewm(span=span, adjust=False).mean()
        last_ema = float(ema_values.iloc[-1])
        # Extrapolate: EMA tends toward the last value
        alpha = 2.0 / (span + 1)
        last_actual = float(df[value_col].iloc[-1])
        decay = alpha * (last_actual - last_ema)
        last_date = df[date_col].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        residual_std = float(np.std(df[value_col].values - ema_values.values))
        predictions = np.array([last_ema + decay * np.exp(-0.1 * i) for i in range(periods)])
        margin = 1.96 * residual_std
        return pd.DataFrame({
            'Date': future_dates, 'Predicted_Value': predictions,
            'Lower_Bound': predictions - margin, 'Upper_Bound': predictions + margin,
            'model': 'ema',
        })

    def compare_models(self, df: pd.DataFrame, date_col: str = None, value_col: str = None,
                       periods: int = 30, cv_split: float = 0.8) -> pd.DataFrame:
        """
        Compare multiple forecasting models using train/test split.

        Args:
            df: Historical price data
            date_col: Date column (auto-detected if None)
            value_col: Value column (auto-detected if None)
            periods: Periods to forecast
            cv_split: Fraction of data for training (rest for validation)

        Returns:
            DataFrame with MAE, RMSE, R2 for each model
        """
        df = df.copy()
        # Auto-detect columns
        if date_col is None or value_col is None:
            for col in df.columns:
                if date_col is None and pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                elif date_col is None and df[col].dtype == object:
                    try:
                        pd.to_datetime(df[col])
                        date_col = col
                    except (ValueError, TypeError):
                        pass
                elif value_col is None and np.issubdtype(df[col].dtype, np.number):
                    value_col = col

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        split = int(n * cv_split)
        train = df.iloc[:split]
        test = df.iloc[split:]

        if len(test) == 0:
            raise ValueError("Not enough data for validation. Need at least 10 rows.")

        test_periods = len(test)
        results = []

        for model_name, method in [
            ("linear", self.fit_predict),
            ("ridge", self.fit_predict_ridge),
            ("polynomial_deg2", lambda d, dc, vc, p: self.fit_predict_polynomial(d, dc, vc, p, degree=2)),
            ("ema", self.fit_predict_ema),
        ]:
            try:
                forecast = method(train, date_col, value_col, test_periods)
                actual = test[value_col].values
                predicted = forecast['Predicted_Value'].values[:len(actual)]
                mae = float(np.mean(np.abs(actual - predicted)))
                rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
                results.append({'model': model_name, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)})
            except Exception as e:
                results.append({'model': model_name, 'MAE': None, 'RMSE': None, 'R2': None, 'error': str(e)})

        result_df = pd.DataFrame(results)
        if 'error' not in result_df.columns:
            result_df = result_df.sort_values('MAE')
        return result_df


class EnsembleForecaster:
    """
    Ensemble forecaster — combines linear, ridge, polynomial, and EMA models
    using inverse-MAE weighted averaging for more robust predictions.

    Models with lower validation MAE receive higher weight. Falls back to
    equal weighting when all models fail cross-validation.

    Example::

        from priceprophet import EnsembleForecaster
        import pandas as pd

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=90),
            "price": [100 + i * 0.5 for i in range(90)],
        })
        ens = EnsembleForecaster()
        forecast = ens.fit_predict(df, "date", "price", periods=30)
        print(forecast[["Date", "Predicted_Value", "Lower_Bound", "Upper_Bound"]].head())
    """

    def __init__(self):
        self._base = Forecaster()

    def fit_predict(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        periods: int = 30,
        cv_split: float = 0.8,
    ) -> pd.DataFrame:
        """
        Train all base models, weight by inverse validation MAE, and return
        the weighted-average forecast with combined confidence intervals.

        Args:
            df: Historical data.
            date_col: Date column name.
            value_col: Numeric value column name.
            periods: Forecast horizon (days).
            cv_split: Train/validation split ratio for weighting (default 0.8).

        Returns:
            DataFrame with ``Date``, ``Predicted_Value``, ``Lower_Bound``,
            ``Upper_Bound``, and ``model`` = ``"ensemble"``.
        """
        comparison = self._base.compare_models(df, date_col, value_col, periods, cv_split)
        valid = comparison[comparison["MAE"].notna()].copy()

        # Inverse-MAE weights — better models (lower MAE) get higher weight
        valid["weight"] = 1.0 / (valid["MAE"] + 1e-9)
        total_w = valid["weight"].sum()
        valid["weight"] /= total_w

        model_weights: Dict[str, float] = dict(zip(valid["model"], valid["weight"]))

        methods: Dict[str, object] = {
            "linear": self._base.fit_predict,
            "ridge": self._base.fit_predict_ridge,
            "polynomial_deg2": lambda d, dc, vc, p: self._base.fit_predict_polynomial(d, dc, vc, p, degree=2),
            "ema": self._base.fit_predict_ema,
        }

        forecasts: Dict[str, pd.DataFrame] = {}
        for name, method in methods.items():
            try:
                forecasts[name] = method(df, date_col, value_col, periods)  # type: ignore[call-arg]
            except Exception:
                pass

        if not forecasts:
            return self._base.fit_predict(df, date_col, value_col, periods)

        # Fall back to equal weights when cross-validation data is missing
        active = {k: model_weights.get(k, 0.0) for k in forecasts}
        if sum(active.values()) == 0:
            active = {k: 1.0 / len(forecasts) for k in forecasts}
        norm = sum(active.values())

        first = next(iter(forecasts.values()))
        n = len(first)
        preds = np.zeros(n)
        lower = np.zeros(n)
        upper = np.zeros(n)

        for name, df_f in forecasts.items():
            w = active[name] / norm
            preds += w * df_f["Predicted_Value"].values
            lower += w * df_f["Lower_Bound"].values
            upper += w * df_f["Upper_Bound"].values

        return pd.DataFrame({
            "Date": first["Date"],
            "Predicted_Value": preds,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "model": "ensemble",
        })
