"""Financial data analysis tools for the Financial Analyst agent."""

from __future__ import annotations

import logging
import statistics
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _detect_anomalies(
    values: list[float], window: int = 5, z_threshold: float = 2.5
) -> list[dict[str, Any]]:
    """Detect anomalies using a rolling z-score method."""
    anomalies = []
    for i in range(window, len(values)):
        window_data = values[i - window : i]
        mean = statistics.mean(window_data)
        std = statistics.stdev(window_data) if len(window_data) > 1 else 0.0
        if std > 0:
            z_score = (values[i] - mean) / std
            if abs(z_score) > z_threshold:
                anomalies.append(
                    {
                        "index": i,
                        "value": values[i],
                        "z_score": round(z_score, 3),
                        "window_mean": round(mean, 3),
                        "direction": "spike" if z_score > 0 else "drop",
                    }
                )
    return anomalies


@tool
def analyze_financial_data(data_description: str, values: list[float]) -> str:
    """Analyze a financial time series for anomalies and basic statistics.

    Use this tool to detect unusual patterns in financial data such as
    price movements, volume spikes, or transaction anomalies.

    Args:
        data_description: Description of what the data represents.
        values: List of numerical values to analyze.
    """
    if not values:
        return "No data provided for analysis."

    stats = {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }

    anomalies = _detect_anomalies(values)

    report_lines = [
        f"## Financial Analysis: {data_description}",
        "",
        "### Statistics",
        f"- Count: {stats['count']}",
        f"- Mean: {stats['mean']}",
        f"- Median: {stats['median']}",
        f"- Std Dev: {stats['stdev']}",
        f"- Range: [{stats['min']}, {stats['max']}]",
        "",
    ]

    if anomalies:
        report_lines.append(f"### Anomalies Detected ({len(anomalies)})")
        for a in anomalies:
            report_lines.append(
                f"- Index {a['index']}: value={a['value']}, z-score={a['z_score']} "
                f"({a['direction']}, window mean={a['window_mean']})"
            )
    else:
        report_lines.append("### No anomalies detected")

    return "\n".join(report_lines)


@tool
def compute_risk_metrics(
    returns: list[float], risk_free_rate: float = 0.02
) -> str:
    """Compute risk metrics for a series of financial returns.

    Use this tool to assess the risk profile of a financial entity or portfolio.

    Args:
        returns: List of period returns (as decimals, e.g., 0.05 for 5%).
        risk_free_rate: Annual risk-free rate (default 2%).
    """
    if len(returns) < 2:
        return "Insufficient data for risk analysis (need at least 2 periods)."

    mean_return = statistics.mean(returns)
    volatility = statistics.stdev(returns)
    excess_return = mean_return - risk_free_rate / 252  # daily adjustment

    sharpe = excess_return / volatility if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))
    peak = cumulative[0]
    max_drawdown = 0.0
    for val in cumulative:
        peak = max(peak, val)
        drawdown = (peak - val) / peak
        max_drawdown = max(max_drawdown, drawdown)

    # Value at Risk (historical, 95%)
    sorted_returns = sorted(returns)
    var_index = max(0, int(len(sorted_returns) * 0.05))
    var_95 = sorted_returns[var_index]

    return (
        f"## Risk Metrics\n\n"
        f"- Mean Return: {mean_return:.6f}\n"
        f"- Volatility: {volatility:.6f}\n"
        f"- Sharpe Ratio: {sharpe:.4f}\n"
        f"- Max Drawdown: {max_drawdown:.4%}\n"
        f"- VaR (95%): {var_95:.6f}\n"
        f"- Total Periods: {len(returns)}"
    )
