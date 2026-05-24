"""PriceProphet — Seasonality detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""
    has_weekly_seasonality: bool
    has_monthly_seasonality: bool
    peak_day: Optional[str]
    peak_week: Optional[int]
    peak_month: Optional[str]
    autocorr_7d: float
    autocorr_30d: float
    dominant_period: Optional[int]

    def __str__(self) -> str:
        lines = [
            f"Weekly seasonality: {'Yes' if self.has_weekly_seasonality else 'No'}",
            f"Monthly seasonality: {'Yes' if self.has_monthly_seasonality else 'No'}",
        ]
        if self.peak_day:
            lines.append(f"Peak day of week: {self.peak_day}")
        if self.peak_month:
            lines.append(f"Peak month: {self.peak_month}")
        if self.dominant_period:
            lines.append(f"Dominant cycle: ~{self.dominant_period} days")
        return "\n".join(lines)


class SeasonalityDetector:
    """
    Detect weekly, monthly, and custom seasonality in price data.

    Example::

        from priceprophet import SeasonalityDetector
        import pandas as pd

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=365),
            'price': [100 + 20*np.sin(2*np.pi*i/7) for i in range(365)]
        })

        sd = SeasonalityDetector()
        result = sd.detect(df)
        print(result)
    """

    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MONTHS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    def detect(
        self,
        df: pd.DataFrame,
        date_col: str = None,
        value_col: str = None,
        min_autocorr: float = 0.2,
    ) -> SeasonalityResult:
        """
        Detect seasonal patterns in time series data.

        Args:
            df: Historical price data
            date_col: Date column name (auto-detected if None)
            value_col: Value column name (auto-detected if None)
            min_autocorr: Minimum autocorrelation to flag seasonality

        Returns:
            SeasonalityResult with detected patterns
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
        df = df.sort_values(date_col)
        values = df[value_col].values

        # Autocorrelations
        autocorr_7d = self._autocorr(values, lag=7) if len(values) > 14 else 0.0
        autocorr_30d = self._autocorr(values, lag=30) if len(values) > 60 else 0.0

        has_weekly = abs(autocorr_7d) > min_autocorr
        has_monthly = abs(autocorr_30d) > min_autocorr

        # Peak day detection
        peak_day = None
        if "dayofweek" not in df.columns:
            df["_dow"] = df[date_col].dt.dayofweek
            day_means = df.groupby("_dow")[value_col].mean()
            if not day_means.empty:
                peak_day = self.WEEKDAYS[int(day_means.idxmax())]

        # Peak month detection
        peak_month = None
        if len(df) >= 60:
            df["_month"] = df[date_col].dt.month
            month_means = df.groupby("_month")[value_col].mean()
            if not month_means.empty:
                peak_month = self.MONTHS[int(month_means.idxmax()) - 1]

        # Peak week of month
        peak_week = None
        if len(df) >= 28:
            df["_week"] = df[date_col].dt.isocalendar().week.astype(int) % 4 + 1
            week_means = df.groupby("_week")[value_col].mean()
            if not week_means.empty:
                peak_week = int(week_means.idxmax())

        # Dominant period (simple FFT-based)
        dominant_period = self._dominant_period(values)

        return SeasonalityResult(
            has_weekly_seasonality=has_weekly,
            has_monthly_seasonality=has_monthly,
            peak_day=peak_day,
            peak_week=peak_week,
            peak_month=peak_month,
            autocorr_7d=round(autocorr_7d, 3),
            autocorr_30d=round(autocorr_30d, 3),
            dominant_period=dominant_period,
        )

    def _autocorr(self, values: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        n = len(values)
        if n <= lag:
            return 0.0
        v1 = values[: n - lag]
        v2 = values[lag:]
        if np.std(v1) == 0 or np.std(v2) == 0:
            return 0.0
        return float(np.corrcoef(v1, v2)[0, 1])

    def _dominant_period(self, values: np.ndarray) -> Optional[int]:
        """Find dominant cycle via FFT."""
        if len(values) < 10:
            return None
        detrended = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
        fft = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended))
        if len(fft) < 2:
            return None
        dominant_idx = np.argmax(fft[1:]) + 1
        if freqs[dominant_idx] == 0:
            return None
        period = int(round(1 / freqs[dominant_idx]))
        return period if 2 <= period <= len(values) // 2 else None
