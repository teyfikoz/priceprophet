# PriceProphet

Time-Series Forecasting for Prices with anomaly detection and impact simulation.

[![PyPI version](https://badge.fury.io/py/priceprophet.svg)](https://pypi.org/project/priceprophet/)
[![CI](https://github.com/teyfikoz/priceprophet/actions/workflows/ci.yml/badge.svg)](https://github.com/teyfikoz/priceprophet/actions/workflows/ci.yml)

## Installation

```bash
pip install priceprophet
```

## Quick Start

```python
from priceprophet import PriceProphet
import pandas as pd

pp = PriceProphet()

# Create sample data
df = pd.DataFrame({
    'date': pd.date_range('2025-01-01', periods=90),
    'price': [100 + i * 0.5 + (i % 7) * 2 for i in range(90)]
})

# Forecast future prices
forecast = pp.forecast(df, periods=30)
print(forecast.head())

# Detect anomalies
anomalies = pp.detect_anomalies(df)
print(anomalies[anomalies['is_anomaly']])

# Simulate market impact
impact = pp.simulate_impact(current_price=150.0, shock_magnitude=0.15, shock_type="competitor")
print(impact)
```

## Features

- **Price Forecasting** - Linear regression with dynamic confidence intervals
- **Anomaly Detection** - Z-score based anomaly identification
- **Impact Simulation** - Market shock simulation (competitor, supply, demand, regulation)

## License

MIT
