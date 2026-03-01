"""Basic tests for PriceProphet package."""

import pytest
import pandas as pd
import numpy as np


def _sample_data():
    """Create sample price data."""
    dates = pd.date_range('2025-01-01', periods=90)
    prices = [100 + i * 0.5 + (i % 7) * 2 for i in range(90)]
    return pd.DataFrame({'date': dates, 'price': prices})


def test_import():
    """Test that priceprophet can be imported."""
    import priceprophet
    assert hasattr(priceprophet, "__version__")
    assert priceprophet.__version__ == "0.3.0"


def test_forecaster_fit_predict():
    """Test Forecaster.fit_predict."""
    from priceprophet import Forecaster

    f = Forecaster()
    df = _sample_data()
    result = f.fit_predict(df, 'date', 'price', periods=10)
    assert len(result) == 10
    assert 'Predicted_Value' in result.columns
    assert 'Lower_Bound' in result.columns
    assert 'Upper_Bound' in result.columns
    # Dynamic CI should not be fixed 5%
    assert not np.allclose(result['Lower_Bound'], result['Predicted_Value'] * 0.95)


def test_forecaster_predict():
    """Test Forecaster.predict auto-detection."""
    from priceprophet import Forecaster

    f = Forecaster()
    df = _sample_data()
    result = f.predict(df, periods=5)
    assert len(result) == 5


def test_detect_anomalies():
    """Test anomaly detection."""
    from priceprophet import Forecaster

    f = Forecaster()
    df = _sample_data()
    # Add an obvious anomaly
    df.loc[45, 'price'] = 500.0
    result = f.detect_anomalies(df, 'date', 'price')
    assert 'is_anomaly' in result.columns
    assert 'z_score' in result.columns
    assert result.loc[45, 'is_anomaly'] is True or result.loc[45, 'is_anomaly'] == True


def test_simulate_impact():
    """Test impact simulation."""
    from priceprophet import Forecaster

    f = Forecaster()
    result = f.simulate_impact(100.0, 0.1, "competitor")
    assert isinstance(result, dict)
    assert 'new_price' in result
    assert 'price_impact' in result
    assert result['new_price'] < 100.0


def test_priceprophet_facade():
    """Test PriceProphet facade class."""
    from priceprophet import PriceProphet

    pp = PriceProphet()
    df = _sample_data()
    forecast = pp.forecast(df, periods=5)
    assert len(forecast) == 5

    anomalies = pp.detect_anomalies(df)
    assert 'is_anomaly' in anomalies.columns

    impact = pp.simulate_impact(100.0, 0.2, "supply")
    assert isinstance(impact, dict)
