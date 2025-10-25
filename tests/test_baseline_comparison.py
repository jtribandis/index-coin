"""Tests for baseline comparison functionality"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.baseline_comparison import (
    EqualWeightBaseline,
    calculate_baseline_metrics,
    compare_to_baseline
)


def test_baseline_weights_sum_to_one():
    """Verify baseline weights always sum to 1.0"""
    baseline = EqualWeightBaseline(btc_integration_year=2020)
    
    # Test pre-BTC
    weights_pre = baseline.get_weights(pd.Timestamp('2019-06-15'))
    assert abs(sum(weights_pre.values()) - 1.0) < 1e-6
    assert weights_pre['BTC-USD'] == 0.0
    
    # Test post-BTC
    weights_post = baseline.get_weights(pd.Timestamp('2021-06-15'))
    assert abs(sum(weights_post.values()) - 1.0) < 1e-6
    assert weights_post['BTC-USD'] > 0


def test_baseline_cluster_allocation():
    """Verify each cluster gets exactly 33.33%"""
    baseline = EqualWeightBaseline(btc_integration_year=2020)
    
    weights = baseline.get_weights(pd.Timestamp('2021-06-15'))
    
    # C1: BTC + QQQ + SMH
    c1_weight = weights['BTC-USD'] + weights['QQQ'] + weights['SMH']
    assert abs(c1_weight - 0.3333) < 0.001
    
    # C2: ANGL + IYR
    c2_weight = weights['ANGL'] + weights['IYR']
    assert abs(c2_weight - 0.3333) < 0.001
    
    # C3: GLD + SPY
    c3_weight = weights['GLD'] + weights['SPY']
    assert abs(c3_weight - 0.3333) < 0.001


def test_baseline_backtest_runs():
    """Verify baseline backtest completes without errors"""
    # Create synthetic price data
    dates = pd.date_range('2015-01-01', '2020-12-31', freq='D')
    assets = ['GLD', 'QQQ', 'SPY', 'SMH', 'IYR', 'ANGL', 'BTC-USD']
    
    # Random walk prices
    np.random.seed(42)
    price_data = pd.DataFrame(
        100 * np.exp(np.random.randn(len(dates), len(assets)).cumsum(axis=0) * 0.01),
        index=dates,
        columns=assets
    )
    
    # Run baseline backtest
    baseline = EqualWeightBaseline(btc_integration_year=2020)
    nav = baseline.run_backtest(price_data, initial_nav=10000, rebalance_freq='M')
    
    # Verify output
    assert len(nav) == len(price_data)
    assert nav.iloc[0] == 10000
    assert nav.iloc[-1] > 0  # NAV should be positive
    assert not nav.isnull().any()  # No missing values


def test_baseline_metrics_calculation():
    """Verify metric calculation is correct"""
    # Create synthetic NAV series
    dates = pd.date_range('2015-01-01', '2020-12-31', freq='D')
    nav = pd.Series(
        10000 * np.exp(np.random.randn(len(dates)).cumsum() * 0.01),
        index=dates
    )
    nav = nav.clip(lower=5000)  # Prevent negative NAV
    
    # Calculate metrics
    metrics = calculate_baseline_metrics(nav, initial_nav=10000)
    
    # Verify metric ranges
    assert -100 < metrics['cagr_pct'] < 100  # Reasonable CAGR range
    assert -100 < metrics['max_drawdown_pct'] <= 0  # Drawdown should be negative
    assert -5 < metrics['sharpe_ratio'] < 10  # Reasonable Sharpe range
    assert 0 < metrics['volatility_pct'] < 200  # Reasonable vol range
    
    print(f"✓ Metrics calculated: CAGR={metrics['cagr_pct']:.2f}%, "
          f"DD={metrics['max_drawdown_pct']:.2f}%, Sharpe={metrics['sharpe_ratio']:.3f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
