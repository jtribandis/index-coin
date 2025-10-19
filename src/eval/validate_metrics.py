"""
Financial metrics validation gates for CI/CD.
Ensures computed metrics are within sensible bounds (Critical Issue #4).
Tech Spec Section 8.1 - Metric definitions with valid ranges.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class MetricsValidator:
    """Validate financial metrics are within acceptable bounds."""
    
    # Define valid ranges for each metric
    BOUNDS = {
        'Sharpe': (-5.0, 10.0),
        'Sortino': (-5.0, 15.0),
        'MAR': (0.0, 20.0),
        'Calmar': (0.0, 20.0),
        'MaxDD': (-1.0, 0.0),
        'CVaR95': (-1.0, 0.0),
        'Ulcer': (0.0, 1.0),
        'CAGR': (-0.5, 2.0),  # -50% to +200% annualized
        'Vol': (0.0, 1.5),     # 0% to 150% annualized
        'RiskOff_pct': (0.0, 1.0),
        'Turnover': (0.0, 5.0),  # 0 to 500% annual
        'CostDrag_bps': (0.0, 200.0),  # 0 to 200 bps
        'UpCapture': (0.0, 3.0),
        'DownCapture': (0.0, 3.0),
    }
    
    # Relationships between metrics
    RELATIONSHIPS = [
        ('MAR', 'MaxDD', 'inverse'),  # MAR = CAGR / |MaxDD|
        ('Sharpe', 'Vol', 'inverse'),  # Sharpe = Return / Vol
    ]
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_metrics_file(self, filepath: str) -> bool:
        """
        Validate metrics from JSON file.
        
        Args:
            filepath: Path to perf_metrics.json
            
        Returns:
            True if all metrics valid, False otherwise
        """
        path = Path(filepath)
        if not path.exists():
            self.errors.append(f"Metrics file not found: {filepath}")
            return False
        
        with open(path) as f:
            metrics = json.load(f)
        
        return self.validate_metrics(metrics)
    
    def validate_metrics(self, metrics: Dict[str, float]) -> bool:
        """
        Validate metrics dictionary.
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            True if all metrics valid, False otherwise
        """
        print("🔍 Validating financial metrics...")
        
        # Check bounds
        bounds_valid = self._check_bounds(metrics)
        
        # Check for NaN/Inf
        finiteness_valid = self._check_finiteness(metrics)
        
        # Check relationships
        relationships_valid = self._check_relationships(metrics)
        
        # Check special cases
        special_valid = self._check_special_cases(metrics)
        
        all_valid = all([
            bounds_valid,
            finiteness_valid,
            relationships_valid,
            special_valid
        ])
        
        self._print_summary()
        
        return all_valid
    
    def _check_bounds(self, metrics: Dict[str, float]) -> bool:
        """Check all metrics are within valid ranges."""
        all_valid = True
        
        for metric, value in metrics.items():
            if metric not in self.BOUNDS:
                continue  # Skip unknown metrics
            
            min_val, max_val = self.BOUNDS[metric]
            
            if not (min_val <= value <= max_val):
                msg = f"{metric} out of bounds: {value:.4f} not in [{min_val}, {max_val}]"
                self.errors.append(msg)
                all_valid = False
            else:
                # Warn if near boundaries
                range_size = max_val - min_val
                if abs(value - min_val) < 0.1 * range_size:
                    self.warnings.append(f"{metric} near lower bound: {value:.4f}")
                elif abs(value - max_val) < 0.1 * range_size:
                    self.warnings.append(f"{metric} near upper bound: {value:.4f}")
        
        return all_valid
    
    def _check_finiteness(self, metrics: Dict[str, float]) -> bool:
        """Check no metrics are NaN or Inf."""
        all_valid = True
        
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            
            if np.isnan(value):
                self.errors.append(f"{metric} is NaN")
                all_valid = False
            elif np.isinf(value):
                # Inf might be acceptable for MAR in some cases
                if metric in ['MAR', 'Calmar']:
                    self.warnings.append(f"{metric} is infinite (zero drawdown?)")
                else:
                    self.errors.append(f"{metric} is infinite")
                    all_valid = False
        
        return all_valid
    
    def _check_relationships(self, metrics: Dict[str, float]) -> bool:
        """Check logical relationships between metrics."""
        all_valid = True
        
        # MAR = CAGR / |MaxDD|
        if all(k in metrics for k in ['MAR', 'CAGR', 'MaxDD']):
            if metrics['MaxDD'] != 0:
                computed_mar = metrics['CAGR'] / abs(metrics['MaxDD'])
                if not np.isclose(metrics['MAR'], computed_mar, rtol=0.01):
                    msg = f"MAR inconsistent: {metrics['MAR']:.3f} vs computed {computed_mar:.3f}"
                    if self.strict:
                        self.errors.append(msg)
                        all_valid = False
                    else:
                        self.warnings.append(msg)
        
        # Sharpe = (CAGR - RF) / Vol
        if all(k in metrics for k in ['Sharpe', 'CAGR', 'Vol']):
            rf = 0.0  # Assuming 0 risk-free rate
            computed_sharpe = (metrics['CAGR'] - rf) / metrics['Vol'] if metrics['Vol'] > 0 else 0
            if not np.isclose(metrics['Sharpe'], computed_sharpe, rtol=0.05):
                msg = f"Sharpe inconsistent: {metrics['Sharpe']:.3f} vs computed {computed_sharpe:.3f}"
                if self.strict:
                    self.errors.append(msg)
                    all_valid = False
                else:
                    self.warnings.append(msg)
        
        return all_valid
    
    def _check_special_cases(self, metrics: Dict[str, float]) -> bool:
        """Check special cases and common issues."""
        all_valid = True
        
        # MaxDD should be negative
        if 'MaxDD' in metrics and metrics['MaxDD'] > 0:
            self.errors.append(f"MaxDD must be negative: {metrics['MaxDD']}")
            all_valid = False
        
        # CVaR should be negative (losses)
        if 'CVaR95' in metrics and metrics['CVaR95'] > 0:
            self.errors.append(f"CVaR95 must be non-positive: {metrics['CVaR95']}")
            all_valid = False
        
        # Turnover warning
        if 'Turnover' in metrics and metrics['Turnover'] > 2.0:
            self.warnings.append(f"High turnover detected: {metrics['Turnover']:.2f} (>200%/yr)")
        
        # Cost drag warning
        if 'CostDrag_bps' in metrics and metrics['CostDrag_bps'] > 50:
            self.warnings.append(f"High cost drag: {metrics['CostDrag_bps']:.1f} bps")
        
        # Risk-Off percentage reasonableness
        if 'RiskOff_pct' in metrics:
            if metrics['RiskOff_pct'] > 0.5:
                self.warnings.append(f"High Risk-Off time: {metrics['RiskOff_pct']:.1%}")
            elif metrics['RiskOff_pct'] == 0:
                self.warnings.append("Risk-Off never triggered (normal or concerning?)")
        
        # Ulcer should be less than or comparable to |MaxDD|
        if all(k in metrics for k in ['Ulcer', 'MaxDD']):
            if metrics['Ulcer'] > abs(metrics['MaxDD']) * 1.5:
                self.warnings.append(
                    f"Ulcer ({metrics['Ulcer']:.3f}) >> |MaxDD| ({abs(metrics['MaxDD']):.3f})"
                )
        
        return all_valid
    
    def _print_summary(self):
        """Print validation summary."""
        if not self.errors and not self.warnings:
            print("✅ All metrics within valid ranges")
            return
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")


def validate_inline_metrics(**metrics) -> bool:
    """
    Validate metrics passed as keyword arguments.
    Useful for inline checks in code.
    
    Example:
        validate_inline_metrics(
            Sharpe=1.05,
            MAR=0.68,
            MaxDD=-0.21,
            CVaR95=-0.028
        )
    """
    validator = MetricsValidator(strict=True)
    return validator.validate_metrics(metrics)


def main():
    """CLI entry point for metrics validation."""
    if len(sys.argv) < 2:
        print("Usage: python validate_metrics.py <path_to_metrics.json> [--strict]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    strict = '--strict' in sys.argv
    
    validator = MetricsValidator(strict=strict)
    valid = validator.validate_metrics_file(filepath)
    
    if valid:
        print("\n✅ Metrics validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Metrics validation FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
