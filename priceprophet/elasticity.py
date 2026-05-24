"""PriceProphet — Price elasticity analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ElasticityResult:
    """Price elasticity analysis result."""
    elasticity: float
    interpretation: str
    demand_change_pct: float
    revenue_impact: str

    def __str__(self) -> str:
        return (
            f"Price Elasticity: {self.elasticity:.3f}\n"
            f"Interpretation: {self.interpretation}\n"
            f"A 10% price increase → {abs(self.demand_change_pct):.1f}% demand {'drop' if self.demand_change_pct < 0 else 'increase'}\n"
            f"Revenue Impact: {self.revenue_impact}"
        )


class PriceElasticity:
    """
    Price elasticity of demand calculator.

    Measures how sensitive demand is to price changes.
    Elasticity < -1: elastic (price-sensitive)
    Elasticity -1 to 0: inelastic (price-insensitive)
    Elasticity > 0: Giffen good (rare)

    Example::

        from priceprophet import PriceElasticity
        import pandas as pd

        prices  = pd.Series([10, 11, 12, 11.5, 13, 12])
        demands = pd.Series([100, 90, 80, 88, 70, 82])

        pe = PriceElasticity()
        result = pe.calculate(prices, demands)
        print(result)
    """

    def calculate(
        self,
        price_series: pd.Series,
        demand_series: pd.Series,
        method: str = "arc",
    ) -> ElasticityResult:
        """
        Calculate price elasticity of demand.

        Args:
            price_series: Historical price data
            demand_series: Historical demand/quantity data (same length)
            method: 'arc' (midpoint method, default) or 'point' (regression-based)

        Returns:
            ElasticityResult with elasticity coefficient and interpretation
        """
        prices = np.array(price_series, dtype=float)
        demands = np.array(demand_series, dtype=float)

        if len(prices) != len(demands):
            raise ValueError("price_series and demand_series must have the same length.")
        if len(prices) < 2:
            raise ValueError("At least 2 observations required.")

        if method == "arc":
            elasticity = self._arc_elasticity(prices, demands)
        else:
            elasticity = self._point_elasticity(prices, demands)

        interpretation = self._interpret(elasticity)
        demand_change_pct = elasticity * 10  # for a 10% price change
        revenue_impact = self._revenue_impact(elasticity)

        return ElasticityResult(
            elasticity=round(elasticity, 3),
            interpretation=interpretation,
            demand_change_pct=round(demand_change_pct, 2),
            revenue_impact=revenue_impact,
        )

    def _arc_elasticity(self, prices: np.ndarray, demands: np.ndarray) -> float:
        """Midpoint (arc) elasticity averaged across all consecutive pairs."""
        elasticities = []
        for i in range(len(prices) - 1):
            dp = prices[i + 1] - prices[i]
            dq = demands[i + 1] - demands[i]
            avg_p = (prices[i] + prices[i + 1]) / 2
            avg_q = (demands[i] + demands[i + 1]) / 2
            if avg_p != 0 and avg_q != 0 and dp != 0:
                e = (dq / avg_q) / (dp / avg_p)
                elasticities.append(e)
        return float(np.median(elasticities)) if elasticities else 0.0

    def _point_elasticity(self, prices: np.ndarray, demands: np.ndarray) -> float:
        """Regression-based point elasticity (log-log model)."""
        log_p = np.log(np.where(prices > 0, prices, 1e-9))
        log_q = np.log(np.where(demands > 0, demands, 1e-9))
        # OLS: log_q = a + b * log_p; b is the elasticity
        cov = np.cov(log_p, log_q)
        if cov[0, 0] == 0:
            return 0.0
        return float(cov[0, 1] / cov[0, 0])

    def _interpret(self, e: float) -> str:
        if e < -2:
            return "Highly elastic — buyers are very sensitive to price changes"
        if e < -1:
            return "Elastic — demand changes more than proportionally to price"
        if e == -1:
            return "Unit elastic — demand changes proportionally to price"
        if -1 < e < 0:
            return "Inelastic — demand changes less than proportionally to price"
        if e >= 0:
            return "Unusual (positive elasticity) — may indicate Giffen or Veblen good"
        return "Undefined"

    def _revenue_impact(self, e: float) -> str:
        if e < -1:
            return "Raising price DECREASES revenue (demand falls more than price rises)"
        if -1 < e < 0:
            return "Raising price INCREASES revenue (demand falls less than price rises)"
        if e == -1:
            return "Price change has NO effect on total revenue"
        return "Non-standard — consult demand data carefully"

    def simulate_price_change(
        self,
        current_price: float,
        current_demand: float,
        price_change_pct: float,
        elasticity: float,
    ) -> dict:
        """
        Simulate revenue impact of a price change.

        Args:
            current_price: Current price point
            current_demand: Current demand/units sold
            price_change_pct: Proposed price change (e.g. 0.10 for +10%)
            elasticity: Elasticity coefficient from calculate()

        Returns:
            Dict with new_price, new_demand, old_revenue, new_revenue, revenue_change_pct
        """
        new_price = current_price * (1 + price_change_pct)
        demand_change = elasticity * price_change_pct
        new_demand = current_demand * (1 + demand_change)
        old_revenue = current_price * current_demand
        new_revenue = new_price * new_demand
        return {
            "current_price": current_price,
            "new_price": round(new_price, 2),
            "price_change_pct": round(price_change_pct * 100, 2),
            "current_demand": current_demand,
            "new_demand": round(new_demand, 2),
            "demand_change_pct": round(demand_change * 100, 2),
            "old_revenue": round(old_revenue, 2),
            "new_revenue": round(new_revenue, 2),
            "revenue_change_pct": round((new_revenue - old_revenue) / old_revenue * 100, 2),
            "recommendation": "Raise price" if new_revenue > old_revenue else "Lower price",
        }
