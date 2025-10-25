"""
Baseline Comparison Module for Index-Coin
Implements Section 7.5 from tech spec.

Validates hypothesis: "My ML risk methodology achieves better returns
with lower risk than a simple equal-weight baseline"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class EqualWeightBaseline:
    """
    Equal-weight baseline strategy.

    Rules (Section 7.5.1):
    - Each cluster gets 33.33% allocation
    - Within cluster, equal weight to all assets
    - BTC enters at 0% weight before 2020, equal weight after
    - Rebalances at same frequency as ML strategy
    - Same transaction costs as ML strategy (ETF: 5bps, BTC: 15bps)
    """

    def __init__(self, btc_integration_year=2020):
        self.btc_year = btc_integration_year

        # Define clusters (Section 1.1)
        self.clusters = {
            "C1_crypto_tech": ["BTC-USD", "QQQ", "SMH"],
            "C2_bonds_reits": ["ANGL", "IYR"],
            "C3_commodities_beta": ["GLD", "SPY"],
        }

    def get_weights(self, date):
        """
        Get equal weights for given date.

        Pre-BTC (before 2020): 6 assets
        - C1 (QQQ, SMH): 16.665% each
        - C2 (ANGL, IYR): 16.665% each
        - C3 (GLD, SPY): 16.665% each

        Post-BTC (2020+): 7 assets
        - C1 (BTC, QQQ, SMH): 11.11% each
        - C2 (ANGL, IYR): 16.665% each
        - C3 (GLD, SPY): 16.665% each
        """
        year = date.year

        # Pre-BTC allocation (6 assets)
        if year < self.btc_year:
            return {
                "BTC-USD": 0.0,
                "QQQ": 0.16665,
                "SMH": 0.16665,
                "ANGL": 0.16665,
                "IYR": 0.16665,
                "GLD": 0.16665,
                "SPY": 0.16665,
            }

        # Post-BTC allocation (7 assets)
        else:
            return {
                "BTC-USD": 0.1111,
                "QQQ": 0.1111,
                "SMH": 0.1111,
                "ANGL": 0.16665,
                "IYR": 0.16665,
                "GLD": 0.16665,
                "SPY": 0.16665,
            }

    def run_backtest(
        self, price_data, initial_nav=10000, rebalance_freq="M", transaction_costs=None
    ):
        """
        Run baseline backtest with same mechanics as ML strategy.

        Args:
            price_data: DataFrame with asset prices (columns may have _adj suffix)
            initial_nav: Starting capital ($10,000 default)
            rebalance_freq: 'M' (monthly), 'Q' (quarterly), '2Q' (semiannual)
            transaction_costs: dict of {asset: bps_one_way}

        Returns:
            pd.Series: Daily NAV values
        """
        if transaction_costs is None:
            transaction_costs = {
                "GLD": 5,
                "QQQ": 5,
                "SPY": 5,
                "SMH": 5,
                "IYR": 5,
                "ANGL": 5,
                "BTC-USD": 15,
            }

        # Normalize column names (handle _adj suffix)
        price_data = self._normalize_column_names(price_data)

        # Generate rebalance schedule
        rebalance_dates = price_data.resample(rebalance_freq).last().index

        LOGGER.info(
            f"Running baseline backtest: {len(rebalance_dates)} rebalances "
            f"over {len(price_data)} days"
        )

        # Initialize
        nav = pd.Series(index=price_data.index, dtype=float)
        nav.iloc[0] = initial_nav
        current_weights = self.get_weights(price_data.index[0])

        # Track holdings (shares)
        holdings = {}
        for asset in current_weights.keys():
            if asset in price_data.columns and current_weights[asset] > 0:
                holdings[asset] = (
                    initial_nav * current_weights[asset]
                ) / price_data.loc[price_data.index[0], asset]

        # Simulate day by day
        rebalance_count = 0
        total_cost = 0.0

        for i in range(1, len(price_data)):
            date = price_data.index[i]

            # Calculate portfolio value from holdings
            portfolio_value = sum(
                holdings[asset] * price_data.loc[date, asset]
                for asset in holdings.keys()
            )

            # Check if rebalance date
            if date in rebalance_dates:
                rebalance_count += 1

                # Get target weights
                target_weights = self.get_weights(date)

                # Calculate current portfolio weights
                current_portfolio_weights = {
                    asset: (holdings[asset] * price_data.loc[date, asset])
                    / portfolio_value
                    for asset in holdings.keys()
                }

                # Calculate turnover (one-way, Section 5.9)
                turnover = (
                    sum(
                        abs(
                            target_weights.get(asset, 0)
                            - current_portfolio_weights.get(asset, 0)
                        )
                        for asset in set(target_weights.keys())
                        | set(current_portfolio_weights.keys())
                    )
                    / 2
                )
                # Log turnover for analysis
                LOGGER.debug(f"Rebalance {rebalance_count}: turnover={turnover:.4f}")

                # Apply transaction costs (one-way)
                cost_bps = (
                    sum(
                        abs(
                            target_weights.get(asset, 0)
                            - current_portfolio_weights.get(asset, 0)
                        )
                        * transaction_costs.get(asset, 5)
                        for asset in set(target_weights.keys())
                        | set(current_portfolio_weights.keys())
                    )
                    / 2
                )

                cost_dollars = portfolio_value * (cost_bps / 10000)
                portfolio_value -= cost_dollars
                total_cost += cost_dollars

                # Update holdings to target weights
                holdings = {}
                for asset in target_weights.keys():
                    if asset in price_data.columns and target_weights[asset] > 0:
                        holdings[asset] = (
                            portfolio_value * target_weights[asset]
                        ) / price_data.loc[date, asset]

            nav.iloc[i] = portfolio_value

        LOGGER.info(
            f"Baseline backtest complete: {rebalance_count} rebalances, "
            f"${total_cost:.2f} total costs"
        )

        return nav

    def _normalize_column_names(self, price_data):
        """Handle different column naming conventions"""
        rename_map = {}
        for col in price_data.columns:
            # Remove _adj suffix if present
            clean_name = col.replace("_adj", "")
            # Handle BTC naming variations
            if clean_name in ["BTC", "BTCUSD", "BTC_USD"]:
                rename_map[col] = "BTC-USD"
            else:
                rename_map[col] = clean_name

        return price_data.rename(columns=rename_map)


def calculate_baseline_metrics(baseline_nav, initial_nav=10000):
    """
    Calculate performance metrics for baseline strategy.

    Implements Section 8.1 metric definitions from tech spec.
    """
    # Total return
    total_return_pct = (baseline_nav.iloc[-1] / initial_nav - 1) * 100

    # Time period
    n_years = (baseline_nav.index[-1] - baseline_nav.index[0]).days / 365.25

    # CAGR
    cagr_pct = ((baseline_nav.iloc[-1] / initial_nav) ** (1 / n_years) - 1) * 100

    # Drawdown analysis
    rolling_max = baseline_nav.expanding().max()
    drawdowns = (baseline_nav - rolling_max) / rolling_max
    max_drawdown_pct = drawdowns.min() * 100

    # Daily returns
    daily_returns = baseline_nav.pct_change().dropna()

    # Volatility (annualized)
    volatility_pct = daily_returns.std() * np.sqrt(252) * 100

    # Sharpe ratio (2% risk-free rate, Section 8.1)
    risk_free = 0.02
    sharpe_ratio = (cagr_pct / 100 - risk_free) / (volatility_pct / 100)

    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (
        (cagr_pct / 100 - risk_free) / downside_std if downside_std > 0 else np.nan
    )

    # MAR ratio (Calmar)
    mar_ratio = cagr_pct / abs(max_drawdown_pct) if max_drawdown_pct != 0 else np.nan

    # CVaR (95%)
    cvar_95_pct = (
        daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * 100
    )

    # Ulcer Index
    ulcer_index_pct = np.sqrt((drawdowns**2).mean()) * 100

    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility_pct": volatility_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "mar_ratio": mar_ratio,
        "cvar_95_pct": cvar_95_pct,
        "ulcer_index_pct": ulcer_index_pct,
        "final_nav": baseline_nav.iloc[-1],
        "n_years": n_years,
    }


def compare_to_baseline(
    ml_nav_series, ml_metrics, price_data, initial_nav=10000, rebalance_freq="M"
):
    """
    Compare ML strategy to equal-weight baseline.

    Implements Section 7.5.2 from tech spec.

    Args:
        ml_nav_series: pd.Series of ML strategy NAV (from your backtest)
        ml_metrics: dict of ML strategy metrics
        price_data: DataFrame with asset prices
        initial_nav: Starting capital
        rebalance_freq: Rebalancing frequency

    Returns:
        dict: Comprehensive comparison results with verdict
    """
    LOGGER.info("=" * 80)
    LOGGER.info("BASELINE COMPARISON")
    LOGGER.info("=" * 80)

    # Run baseline strategy
    baseline = EqualWeightBaseline(btc_integration_year=2020)
    baseline_nav = baseline.run_backtest(
        price_data=price_data, initial_nav=initial_nav, rebalance_freq=rebalance_freq
    )

    # Calculate baseline metrics
    baseline_metrics = calculate_baseline_metrics(baseline_nav, initial_nav)

    # Calculate improvements
    improvements = {
        "cagr_diff": ml_metrics["cagr_pct"] - baseline_metrics["cagr_pct"],
        "drawdown_diff": ml_metrics["max_drawdown_pct"]
        - baseline_metrics["max_drawdown_pct"],
        "sharpe_diff": ml_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"],
        "sortino_diff": ml_metrics["sortino_ratio"] - baseline_metrics["sortino_ratio"],
        "mar_diff": ml_metrics["mar_ratio"] - baseline_metrics["mar_ratio"],
        "volatility_diff": ml_metrics["volatility_pct"]
        - baseline_metrics["volatility_pct"],
        "value_added": ml_metrics["final_nav"] - baseline_metrics["final_nav"],
    }

    # Generate verdict
    verdict = generate_verdict(improvements)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("VERDICT")
    LOGGER.info("=" * 80)
    LOGGER.info(verdict)
    LOGGER.info("=" * 80)

    return {
        "baseline_nav": baseline_nav,
        "baseline_metrics": baseline_metrics,
        "ml_metrics": ml_metrics,
        "improvements": improvements,
        "verdict": verdict,
        "status": (
            "PASS"
            if improvements["cagr_diff"] > 0 and improvements["sharpe_diff"] > 0
            else "REVIEW"
        ),
    }


def generate_verdict(improvements):
    """Generate human-readable verdict (Section 7.5.2)"""
    cagr_better = improvements["cagr_diff"] > 0
    risk_better = improvements["drawdown_diff"] > 0  # Less negative is better
    sharpe_better = improvements["sharpe_diff"] > 0

    if cagr_better and risk_better and sharpe_better:
        return (
            f"✓ HYPOTHESIS CONFIRMED: ML strategy outperforms baseline on ALL metrics. "
            f"CAGR +{improvements['cagr_diff']:.2f}%, "
            f"Drawdown +{improvements['drawdown_diff']:.2f}%, "
            f"Sharpe +{improvements['sharpe_diff']:.3f}. "
            f"Value added: ${improvements['value_added']:,.0f}."
        )

    elif cagr_better and sharpe_better:
        return (
            f"✓ HYPOTHESIS MOSTLY CONFIRMED: ML strategy achieves higher returns "
            f"({improvements['cagr_diff']:+.2f}%) "
            f"and better risk-adjusted returns (Sharpe {improvements['sharpe_diff']:+.3f}), "
            f"though drawdown similar ({improvements['drawdown_diff']:+.2f}%)."
        )

    elif risk_better and sharpe_better:
        return (
            f"⚠ PARTIAL CONFIRMATION: ML strategy has better risk metrics "
            f"(Drawdown {improvements['drawdown_diff']:+.2f}%, "
            f"Sharpe {improvements['sharpe_diff']:+.3f}), "
            f"but lower returns ({improvements['cagr_diff']:+.2f}%). "
            f"Consider risk-adjusted perspective."
        )

    else:
        return (
            f"✗ HYPOTHESIS NOT CONFIRMED: ML strategy underperforms baseline. "
            f"CAGR {improvements['cagr_diff']:+.2f}%, "
            f"Sharpe {improvements['sharpe_diff']:+.3f}. "
            f"Requires strategy revision."
        )


def export_comparison_results(results, output_dir="web/public/data"):
    """
    Export baseline comparison results (Section 12.7-12.9 from tech spec).

    Exports:
        1. nav_baseline.csv - Baseline NAV series
        2. comparison_metrics.json - Side-by-side metrics
        3. comparison_table.csv - Formatted comparison table
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Exporting comparison results to {output_dir}/")

    # 1. Export baseline NAV (Section 12.7)
    baseline_nav_df = pd.DataFrame(
        {
            "date": results["baseline_nav"].index,
            "nav_baseline": results["baseline_nav"].values,
        }
    )
    baseline_nav_df.to_csv(output_path / "nav_baseline.csv", index=False)
    LOGGER.info("  ✓ nav_baseline.csv")

    # 2. Export comparison metrics JSON (Section 12.8)
    comparison_json = {
        "verdict": results["verdict"],
        "status": results["status"],
        "ml_strategy": {
            "cagr_pct": round(results["ml_metrics"]["cagr_pct"], 2),
            "max_drawdown_pct": round(results["ml_metrics"]["max_drawdown_pct"], 2),
            "sharpe_ratio": round(results["ml_metrics"]["sharpe_ratio"], 3),
            "sortino_ratio": round(results["ml_metrics"]["sortino_ratio"], 3),
            "mar_ratio": round(results["ml_metrics"]["mar_ratio"], 3),
            "volatility_pct": round(results["ml_metrics"]["volatility_pct"], 2),
            "final_nav": round(results["ml_metrics"]["final_nav"], 2),
        },
        "baseline_strategy": {
            "cagr_pct": round(results["baseline_metrics"]["cagr_pct"], 2),
            "max_drawdown_pct": round(
                results["baseline_metrics"]["max_drawdown_pct"], 2
            ),
            "sharpe_ratio": round(results["baseline_metrics"]["sharpe_ratio"], 3),
            "sortino_ratio": round(results["baseline_metrics"]["sortino_ratio"], 3),
            "mar_ratio": round(results["baseline_metrics"]["mar_ratio"], 3),
            "volatility_pct": round(results["baseline_metrics"]["volatility_pct"], 2),
            "final_nav": round(results["baseline_metrics"]["final_nav"], 2),
        },
        "improvements": {
            "cagr_diff": round(results["improvements"]["cagr_diff"], 2),
            "drawdown_diff": round(results["improvements"]["drawdown_diff"], 2),
            "sharpe_diff": round(results["improvements"]["sharpe_diff"], 3),
            "sortino_diff": round(results["improvements"]["sortino_diff"], 3),
            "mar_diff": round(results["improvements"]["mar_diff"], 3),
            "volatility_diff": round(results["improvements"]["volatility_diff"], 2),
            "value_added": round(results["improvements"]["value_added"], 2),
        },
    }

    with open(output_path / "comparison_metrics.json", "w") as f:
        json.dump(comparison_json, f, indent=2)
    LOGGER.info("  ✓ comparison_metrics.json")

    # 3. Export comparison table CSV (Section 12.9)
    comparison_table = pd.DataFrame(
        {
            "Metric": [
                "CAGR (%)",
                "Max Drawdown (%)",
                "Sharpe Ratio",
                "Sortino Ratio",
                "MAR Ratio",
                "Volatility (%)",
                "Final NAV ($)",
            ],
            "ML_Strategy": [
                results["ml_metrics"]["cagr_pct"],
                results["ml_metrics"]["max_drawdown_pct"],
                results["ml_metrics"]["sharpe_ratio"],
                results["ml_metrics"]["sortino_ratio"],
                results["ml_metrics"]["mar_ratio"],
                results["ml_metrics"]["volatility_pct"],
                results["ml_metrics"]["final_nav"],
            ],
            "Baseline": [
                results["baseline_metrics"]["cagr_pct"],
                results["baseline_metrics"]["max_drawdown_pct"],
                results["baseline_metrics"]["sharpe_ratio"],
                results["baseline_metrics"]["sortino_ratio"],
                results["baseline_metrics"]["mar_ratio"],
                results["baseline_metrics"]["volatility_pct"],
                results["baseline_metrics"]["final_nav"],
            ],
            "Improvement": [
                results["improvements"]["cagr_diff"],
                results["improvements"]["drawdown_diff"],
                results["improvements"]["sharpe_diff"],
                results["improvements"]["sortino_diff"],
                results["improvements"]["mar_diff"],
                results["improvements"]["volatility_diff"],
                results["improvements"]["value_added"],
            ],
        }
    )

    comparison_table.to_csv(output_path / "comparison_table.csv", index=False)
    LOGGER.info("  ✓ comparison_table.csv")

    LOGGER.info(f"\n✓ All comparison results exported to {output_dir}/")

    return str(output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s"
    )

    print("Baseline comparison module loaded successfully.")
    print("Use: from src.eval.baseline_comparison import compare_to_baseline")
