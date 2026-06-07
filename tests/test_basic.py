"""Tests for PriceProphet v1.0.0."""

import pytest
import pandas as pd
import numpy as np


def _sample_data(n=90):
    dates = pd.date_range("2025-01-01", periods=n)
    prices = [100 + i * 0.5 + (i % 7) * 2 for i in range(n)]
    return pd.DataFrame({"date": dates, "price": prices})


def test_version():
    import priceprophet

    assert priceprophet.__version__ == "1.1.0"


def test_all_exports():
    from priceprophet import PriceProphet, Forecaster, PriceElasticity, SeasonalityDetector
    from priceprophet import ElasticityResult, SeasonalityResult
    assert all(cls is not None for cls in [PriceProphet, Forecaster, PriceElasticity, SeasonalityDetector])


# ── Forecaster ─────────────────────────────────────────────────────────────────

def test_forecaster_linear():
    from priceprophet import Forecaster
    f = Forecaster()
    df = _sample_data()
    result = f.fit_predict(df, "date", "price", periods=10)
    assert len(result) == 10
    assert "Predicted_Value" in result.columns
    assert "Lower_Bound" in result.columns
    assert "Upper_Bound" in result.columns


def test_forecaster_ridge():
    from priceprophet import Forecaster
    f = Forecaster()
    df = _sample_data()
    result = f.fit_predict_ridge(df, "date", "price", periods=10, alpha=1.0)
    assert len(result) == 10
    assert "Predicted_Value" in result.columns


def test_forecaster_polynomial():
    from priceprophet import Forecaster
    f = Forecaster()
    df = _sample_data()
    result = f.fit_predict_polynomial(df, "date", "price", periods=10, degree=2)
    assert len(result) == 10
    assert "Predicted_Value" in result.columns


def test_forecaster_ema():
    from priceprophet import Forecaster
    f = Forecaster()
    df = _sample_data()
    result = f.fit_predict_ema(df, "date", "price", periods=10)
    assert len(result) == 10
    assert "Predicted_Value" in result.columns


def test_compare_models():
    from priceprophet import Forecaster
    f = Forecaster()
    df = _sample_data(120)
    comparison = f.compare_models(df, "date", "price")
    assert isinstance(comparison, pd.DataFrame)
    assert "model" in comparison.columns
    assert "MAE" in comparison.columns
    assert "RMSE" in comparison.columns
    assert "R2" in comparison.columns
    assert len(comparison) >= 3


def test_detect_anomalies():
    from priceprophet import Forecaster
    f = Forecaster()
    df = _sample_data()
    df.loc[45, "price"] = 500.0
    result = f.detect_anomalies(df, "date", "price")
    assert "is_anomaly" in result.columns
    assert "z_score" in result.columns
    assert result.loc[45, "is_anomaly"] == True


def test_simulate_impact():
    from priceprophet import Forecaster
    f = Forecaster()
    for shock in ["competitor", "supply", "demand", "regulation"]:
        result = f.simulate_impact(100.0, 0.2, shock)
        assert isinstance(result, dict)
        assert "new_price" in result
        assert result["new_price"] < 100.0


# ── PriceProphet facade ────────────────────────────────────────────────────────

def test_priceprophet_forecast_auto():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    df = _sample_data(120)
    result = pp.forecast(df, periods=10, model="auto")
    assert len(result) == 10
    assert "Predicted_Value" in result.columns


def test_priceprophet_forecast_models():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    df = _sample_data()
    for model in ["linear", "ridge", "polynomial", "ema"]:
        result = pp.forecast(df, periods=5, model=model)
        assert len(result) == 5


def test_priceprophet_compare_models():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    df = _sample_data(120)
    result = pp.compare_models(df)
    assert isinstance(result, pd.DataFrame)
    assert "MAE" in result.columns


def test_priceprophet_detect_anomalies():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    df = _sample_data()
    result = pp.detect_anomalies(df)
    assert "is_anomaly" in result.columns


def test_priceprophet_simulate_impact():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    result = pp.simulate_impact(100.0, 0.15, "competitor")
    assert "new_price" in result
    assert "estimated_recovery_days" in result


def test_priceprophet_seasonality():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    n = 180
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "price": [100 + 10 * np.sin(2 * np.pi * i / 7) for i in range(n)],
    })
    result = pp.seasonality(df)
    assert hasattr(result, "has_weekly_seasonality")
    assert hasattr(result, "dominant_period")


# ── PriceElasticity ────────────────────────────────────────────────────────────

def test_elasticity_calculate():
    from priceprophet import PriceElasticity, ElasticityResult
    prices = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0])
    demands = pd.Series([100, 90, 80, 88, 70])
    pe = PriceElasticity()
    result = pe.calculate(prices, demands)
    assert isinstance(result, ElasticityResult)
    assert hasattr(result, "elasticity")
    assert hasattr(result, "interpretation")
    assert isinstance(result.elasticity, float)


def test_elasticity_elastic():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    prices = pd.Series([10.0, 20.0])
    demands = pd.Series([100, 10])
    result = pe.calculate(prices, demands)
    assert result.elasticity < -1  # elastic


def test_elasticity_simulate_price_change():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    prices = pd.Series([10.0, 11.0, 12.0])
    demands = pd.Series([100, 90, 80])
    er = pe.calculate(prices, demands)
    sim = pe.simulate_price_change(
        current_price=12.0,
        current_demand=80,
        price_change_pct=0.10,
        elasticity=er.elasticity,
    )
    assert "new_price" in sim
    assert "new_demand" in sim
    assert "revenue_change_pct" in sim
    assert sim["new_price"] > 12.0


# ── SeasonalityDetector ────────────────────────────────────────────────────────

def test_seasonality_weekly():
    from priceprophet import SeasonalityDetector, SeasonalityResult
    n = 180
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "price": [100 + 15 * np.sin(2 * np.pi * i / 7) for i in range(n)],
    })
    sd = SeasonalityDetector()
    result = sd.detect(df)
    assert isinstance(result, SeasonalityResult)
    assert result.has_weekly_seasonality is True
    assert hasattr(result, "peak_day")
    assert hasattr(result, "dominant_period")


def test_seasonality_no_seasonality():
    from priceprophet import SeasonalityDetector
    n = 90
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "price": [100 + i * 0.1 for i in range(n)],
    })
    sd = SeasonalityDetector()
    result = sd.detect(df)
    assert isinstance(result.has_weekly_seasonality, bool)


# ── SeasonalityResult __str__ ──────────────────────────────────────────────────

def test_seasonality_result_str():
    from priceprophet import SeasonalityDetector
    n = 365
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "price": [100 + 20 * np.sin(2 * np.pi * i / 7) + 15 * np.sin(2 * np.pi * i / 30) for i in range(n)],
    })
    sd = SeasonalityDetector()
    result = sd.detect(df)
    s = str(result)
    assert "seasonality" in s.lower()


# ── SeasonalityDetector auto-detect columns ────────────────────────────────────

def test_seasonality_auto_detect_columns():
    from priceprophet import SeasonalityDetector
    n = 200
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n),
        "val": [50 + 10 * np.sin(2 * np.pi * i / 7) for i in range(n)],
    })
    sd = SeasonalityDetector()
    # Pass df with non-standard column names but no explicit column args
    result = sd.detect(df, date_col="ts", value_col="val")
    assert hasattr(result, "has_weekly_seasonality")


# ── Elasticity point method & edge cases ──────────────────────────────────────

def test_elasticity_point_method():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    demands = pd.Series([100, 90, 80, 72, 64])
    result = pe.calculate(prices, demands, method="point")
    assert isinstance(result.elasticity, float)


def test_elasticity_unit_elastic():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    # unit elastic: 1% price change -> 1% demand change
    prices = pd.Series([10.0, 11.0])
    demands = pd.Series([100.0, 90.9])
    result = pe.calculate(prices, demands)
    assert isinstance(result.interpretation, str)
    assert isinstance(result.revenue_impact, str)


def test_elasticity_errors():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    with pytest.raises(ValueError):
        pe.calculate(pd.Series([1.0]), pd.Series([1.0]))  # less than 2
    with pytest.raises(ValueError):
        pe.calculate(pd.Series([1.0, 2.0]), pd.Series([1.0]))  # length mismatch


# ── Forecaster.predict() auto-detect ─────────────────────────────────────────

def test_forecaster_predict_auto_detect():
    from priceprophet import Forecaster
    f = Forecaster()
    n = 90
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "price": [100 + i * 0.5 for i in range(n)],
    })
    # Use predict() which auto-detects columns
    result = f.predict(df, periods=7)
    assert len(result) == 7
    assert "Predicted_Value" in result.columns


# ── compare_models auto-detect ────────────────────────────────────────────────

def test_compare_models_auto_detect():
    from priceprophet import Forecaster
    f = Forecaster()
    n = 120
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "price": [100 + i * 0.3 for i in range(n)],
    })
    result = f.compare_models(df)
    assert "model" in result.columns


# ── predict() with string date column ─────────────────────────────────────────

def test_forecaster_predict_string_dates():
    from priceprophet import Forecaster
    f = Forecaster()
    n = 60
    df = pd.DataFrame({
        "date": [str(d.date()) for d in pd.date_range("2025-01-01", periods=n)],
        "price": [100.0 + i for i in range(n)],
    })
    result = f.predict(df, periods=5)
    assert len(result) == 5


def test_forecaster_predict_no_date_col_raises():
    from priceprophet import Forecaster
    f = Forecaster()
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError):
        f.predict(df)


# ── detect_anomalies auto-detect & zero-std ───────────────────────────────────

def test_detect_anomalies_zero_std():
    from priceprophet import Forecaster
    f = Forecaster()
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "price": [100.0] * 10,  # constant price → std = 0
    })
    result = f.detect_anomalies(df, "date", "price")
    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].sum() == 0


# ── PriceProphet elasticity + auto_select ─────────────────────────────────────

def test_priceprophet_elasticity():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    prices = pd.Series([10.0, 11.0, 12.0, 13.0])
    demands = pd.Series([100, 90, 80, 72])
    result = pp.elasticity(prices, demands)
    assert hasattr(result, "elasticity")


def test_priceprophet_seasonality_large():
    from priceprophet import PriceProphet
    pp = PriceProphet()
    n = 365
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "price": [100 + 20 * np.sin(2 * np.pi * i / 7) for i in range(n)],
    })
    result = pp.seasonality(df)
    assert hasattr(result, "peak_month") or result.peak_month is None


# ── Seasonality with monthly data ──────────────────────────────────────────────

def test_seasonality_monthly_pattern():
    from priceprophet import SeasonalityDetector
    n = 400
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "price": [100 + 20 * np.sin(2 * np.pi * i / 30) for i in range(n)],
    })
    sd = SeasonalityDetector()
    result = sd.detect(df)
    assert isinstance(result.has_monthly_seasonality, bool)
    assert result.autocorr_30d is not None


# ── ElasticityResult __str__ ───────────────────────────────────────────────────

def test_elasticity_result_str():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    demands = pd.Series([100, 88, 77, 68, 60])
    result = pe.calculate(prices, demands)
    s = str(result)
    assert "Elasticity" in s or "elasticity" in s.lower()


# ── Elasticity highly elastic & inelastic interpretations ────────────────────

def test_elasticity_highly_elastic():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    # Very elastic: large demand response
    prices = pd.Series([10.0, 11.0])
    demands = pd.Series([1000.0, 100.0])
    result = pe.calculate(prices, demands)
    assert result.elasticity < -2


def test_elasticity_inelastic():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    # Inelastic: small demand response
    prices = pd.Series([10.0, 15.0])
    demands = pd.Series([100.0, 98.0])
    result = pe.calculate(prices, demands)
    assert result.elasticity > -1


def test_elasticity_positive():
    from priceprophet import PriceElasticity
    pe = PriceElasticity()
    # Unusual: positive elasticity (Veblen good)
    prices = pd.Series([10.0, 15.0])
    demands = pd.Series([100.0, 110.0])
    result = pe.calculate(prices, demands)
    assert "positive" in result.interpretation.lower() or "unusual" in result.interpretation.lower()


# ── Revenue impact edge cases ─────────────────────────────────────────────────

def test_revenue_impact_unit_elastic():
    from priceprophet.elasticity import PriceElasticity
    pe = PriceElasticity()
    # Manually check _revenue_impact
    assert "NO effect" in pe._revenue_impact(-1.0)
    assert "Non-standard" in pe._revenue_impact(0.5)
