# PriceProphet

**Multi-model price forecasting, anomaly detection, price elasticity, and seasonality analysis** for Python.

[![PyPI version](https://badge.fury.io/py/priceprophet.svg)](https://pypi.org/project/priceprophet/)
[![Build](https://github.com/teyfikoz/priceprophet/actions/workflows/publish.yml/badge.svg)](https://github.com/teyfikoz/priceprophet/actions/workflows/publish.yml)
[![CI](https://github.com/teyfikoz/priceprophet/actions/workflows/ci.yml/badge.svg)](https://github.com/teyfikoz/priceprophet/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install priceprophet
```

## Quick Start

```python
from priceprophet import PriceProphet
import pandas as pd

pp = PriceProphet()

df = pd.DataFrame({
    'date':  pd.date_range('2025-01-01', periods=180),
    'price': [100 + i*0.3 + (i % 7)*2 for i in range(180)]
})

# Auto-select best model
forecast = pp.forecast(df, periods=30, model="auto")
print(forecast.head())

# Detect price anomalies
anomalies = pp.detect_anomalies(df)
print(anomalies[anomalies['is_anomaly']])

# Market shock simulation
impact = pp.simulate_impact(current_price=150.0, shock_magnitude=0.15, shock_type="competitor")
print(impact)
```

---

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **4 Forecasting Models** | Linear, Ridge, Polynomial (deg 1–5), EMA |
| **Auto Model Selection** | Picks the best model via train/test split |
| **Model Comparison** | Compare all models — MAE, RMSE, R² table |
| **Price Elasticity** | Arc and point elasticity with revenue simulation |
| **Seasonality Detection** | Weekly, monthly, FFT-based dominant cycle |
| **Anomaly Detection** | Z-score with configurable threshold |
| **Shock Simulation** | competitor / supply / demand / regulation shocks |

---

## Forecasting Models

```python
from priceprophet import PriceProphet
import pandas as pd, numpy as np

pp = PriceProphet()
df = pd.DataFrame({
    'date':  pd.date_range('2024-01-01', periods=365),
    'price': 100 + np.cumsum(np.random.randn(365) * 0.5)
})

# Linear regression (fastest, good for stable trends)
fc_linear = pp.forecast(df, periods=30, model="linear")

# Ridge (L2-regularized, robust to outliers)
fc_ridge  = pp.forecast(df, periods=30, model="ridge", alpha=0.5)

# Polynomial (captures curve, degree=3 for S-curves)
fc_poly   = pp.forecast(df, periods=30, model="polynomial", degree=3)

# EMA (exponential moving average, follows recent trend)
fc_ema    = pp.forecast(df, periods=30, model="ema", span=14)

# Auto-select: runs comparison, picks lowest MAE model
fc_auto   = pp.forecast(df, periods=30, model="auto")

print(fc_auto[['Date', 'Predicted_Value', 'Lower_Bound', 'Upper_Bound']].tail())
```

---

## Model Comparison

```python
comparison = pp.compare_models(df, cv_split=0.8)
print(comparison)
#              model      MAE     RMSE      R2
# 0            ridge   1.832    2.341   0.972
# 1           linear   1.944    2.501   0.969
# 2  polynomial_deg2   2.103    2.788   0.961
# 3              ema   2.891    3.672   0.943
```

---

## Price Elasticity

```python
from priceprophet import PriceElasticity
import pandas as pd

prices  = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0, 12.5])
demands = pd.Series([100,   90,   80,   88,   70,   83])

pe = PriceElasticity()
result = pe.calculate(prices, demands)
print(result)
# Price Elasticity: -1.247
# Interpretation: Elastic — demand changes more than proportionally to price
# A 10% price increase → 12.5% demand drop
# Revenue Impact: Raising price DECREASES revenue

# Simulate a specific price change
sim = pe.simulate_price_change(
    current_price=12.0,
    current_demand=80,
    price_change_pct=0.10,    # +10%
    elasticity=result.elasticity,
)
print(sim)
# {'new_price': 13.2, 'new_demand': 69.8, 'revenue_change_pct': -2.7, ...}
```

---

## Seasonality Detection

```python
from priceprophet import SeasonalityDetector
import pandas as pd, numpy as np

df = pd.DataFrame({
    'date':  pd.date_range('2024-01-01', periods=365),
    'price': [100 + 20 * np.sin(2 * np.pi * i / 7) for i in range(365)]
})

sd = SeasonalityDetector()
result = sd.detect(df)
print(result)
# Weekly seasonality  : Yes
# Monthly seasonality : No
# Peak day of week    : Wednesday
# Dominant cycle      : ~7 days
```

---

## Anomaly Detection

```python
df_with_spike = df.copy()
df_with_spike.loc[45, 'price'] = 999  # inject spike

anomalies = pp.detect_anomalies(df_with_spike, threshold=2.5)
spikes = anomalies[anomalies['is_anomaly']]
print(f"Found {len(spikes)} anomalies")
print(spikes[['date', 'price', 'z_score']])
```

---

## Market Shock Simulation

```python
for shock_type in ["competitor", "supply", "demand", "regulation"]:
    result = pp.simulate_impact(
        current_price=100.0,
        shock_magnitude=0.20,
        shock_type=shock_type
    )
    print(f"{shock_type:12s}: price {result['new_price']:.1f}  |  "
          f"recovery {result['estimated_recovery_days']} days")
# competitor  : price 86.0  |  recovery 14 days
# supply      : price 76.0  |  recovery 30 days
# demand      : price 82.0  |  recovery 21 days
# regulation  : price 70.0  |  recovery 60 days
```

---

## Full Analysis Pipeline

```python
from priceprophet import PriceProphet
import pandas as pd, numpy as np

pp = PriceProphet()

# Generate realistic price series
np.random.seed(42)
n = 365
trend   = np.linspace(100, 140, n)
weekly  = 8 * np.sin(2 * np.pi * np.arange(n) / 7)
noise   = np.random.randn(n) * 2
prices  = trend + weekly + noise

df = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=n), 'price': prices})

# 1. Seasonality
season = pp.seasonality(df)
print("Peak day:", season.peak_day)

# 2. Best model forecast
fc = pp.forecast(df, periods=90, model="auto")
print(f"90-day forecast: {fc['Predicted_Value'].mean():.2f} avg")

# 3. Anomalies
anom = pp.detect_anomalies(df, threshold=2.5)
print(f"Anomalies: {anom['is_anomaly'].sum()}")

# 4. Model comparison
print(pp.compare_models(df).to_string(index=False))
```

---

## License

MIT — [Teyfik Öz](https://github.com/teyfikoz)
