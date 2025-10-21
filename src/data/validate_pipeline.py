"""
Data validation pipeline for Index-Coin quantitative system.
Validates data integrity, alignment, and quality per Tech Spec Section 2 & 14.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class DataValidator:
    """Validates financial data integrity for quantitative trading system."""

    SYMBOLS = ["GLD", "QQQ", "SPY", "SMH", "IYR", "ANGL", "BTC-USD"]
    BUSINESS_DAY_GAP_LIMIT = 3
    BTC_WEEKLY_MIN_DAYS = 5
    EXTREME_RETURN_THRESHOLD = 0.50  # 50% daily move signals data error

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._panel: Optional[pd.DataFrame] = None

    def _load_panel(self) -> Optional[pd.DataFrame]:
        """Load and cache panel data, return None if fails."""
        if self._panel is not None:
            return self._panel

        panel_path = Path("data/staging/panel.parquet")

        if not panel_path.exists():
            self.errors.append("Missing panel.parquet")
            return None

        try:
            self._panel = pd.read_parquet(panel_path)
            return self._panel
        except Exception as e:
            self.errors.append(f"Failed reading panel.parquet: {e}")
            return None

    def validate_all(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🔍 Starting data validation pipeline...")

        results: Dict[str, Any] = {
            "manifest_check": self.validate_manifest(),
            "panel_integrity": self.validate_panel(),
            "gap_analysis": self.validate_gaps(),
            "return_sanity": self.validate_returns(),
            "btc_alignment": self.validate_btc_alignment(),
            "dividend_verification": self.validate_dividends(),
            "status": "PASS",
        }

        if self.errors:
            results["status"] = "FAIL"
            results["errors"] = self.errors
        if self.warnings:
            results["warnings"] = self.warnings

        self._save_report(results)
        return results

    def validate_manifest(self) -> Dict[str, Any]:
        """Validate ingest manifest checksums (Tech Spec 2.4)."""
        manifest_path = Path("data/raw/_ingest_manifest.json")

        if not manifest_path.exists():
            self.errors.append("Missing _ingest_manifest.json")
            return {"valid": False}

        # Load and parse manifest with exception handling
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except PermissionError as e:
            self.errors.append(
                f"Permission denied reading manifest {manifest_path}: {e}"
            )
            return {"valid": False}
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in manifest {manifest_path}: {e}")
            return {"valid": False}
        except Exception as e:
            self.errors.append(f"Failed to read manifest {manifest_path}: {e}")
            return {"valid": False}

        mismatches: List[str] = []
        for symbol in self.SYMBOLS:
            file_path = Path(f"data/raw/{symbol}.csv")
            if not file_path.exists():
                self.errors.append(f"Missing raw data file: {symbol}.csv")
                continue

            # Read file and compute hash with exception handling
            try:
                with open(file_path, "rb") as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
            except (OSError, IOError) as e:
                error_msg = f"Failed to read {file_path}: {e}"
                self.errors.append(error_msg)
                mismatches.append(f"{symbol}: read error")
                continue
            except PermissionError as e:
                error_msg = f"Permission denied reading {file_path}: {e}"
                self.errors.append(error_msg)
                mismatches.append(f"{symbol}: permission error")
                continue
            except Exception as e:
                error_msg = f"Unexpected error reading {file_path}: {e}"
                self.errors.append(error_msg)
                mismatches.append(f"{symbol}: unexpected error")
                continue

            expected = manifest.get("symbols", {}).get(symbol, {}).get("sha256")
            if expected and actual_hash != expected:
                mismatches.append(f"{symbol}: hash mismatch")

        if mismatches:
            self.errors.extend(mismatches)
            return {"valid": False, "mismatches": mismatches}

        return {"valid": True, "symbols_checked": len(self.SYMBOLS)}

    def validate_panel(self) -> Dict[str, Any]:
        """Validate staging panel structure (Tech Spec 2.3)."""
        panel = self._load_panel()

        if panel is None:
            return {"valid": False}

        # Check for empty panel
        if panel.empty:
            self.errors.append("Panel is empty (no rows)")
            return {"valid": False}

        # Check for expected columns
        expected_cols = [f"{sym}_adj" for sym in self.SYMBOLS]
        missing_cols = [c for c in expected_cols if c not in panel.columns]

        if missing_cols:
            self.errors.append(f"Missing columns: {missing_cols}")
            return {"valid": False}

        # Verify date index is business days
        if not isinstance(panel.index, pd.DatetimeIndex):
            self.errors.append("Panel index is not DatetimeIndex")
            return {"valid": False}

        # Use min/max to handle unsorted indices and avoid IndexError
        date_min = panel.index.min()
        date_max = panel.index.max()

        return {
            "valid": True,
            "rows": len(panel),
            "date_range": f"{date_min} to {date_max}",
            "columns": len(panel.columns),
        }

    def validate_gaps(self) -> Dict[str, Any]:
        """Check for excessive data gaps (Tech Spec 2.3)."""
        panel = self._load_panel()

        if panel is None:
            return {"valid": False}

        gap_violations: List[str] = []

        for symbol in self.SYMBOLS:
            col = f"{symbol}_adj"
            if col not in panel.columns:
                continue

            # Find consecutive NaN groups
            is_na = panel[col].isna()
            na_groups = (is_na != is_na.shift()).cumsum()
            gap_sizes = is_na.groupby(na_groups).sum()
            max_gap = gap_sizes.max() if len(gap_sizes) > 0 else 0

            total_missing = panel[col].isna().sum()

            # BTC has stricter limit (1 day)
            limit = 1 if "BTC" in symbol else self.BUSINESS_DAY_GAP_LIMIT

            if max_gap > limit:
                msg = f"{symbol}: max gap {int(max_gap)} days exceeds limit {limit}"
                gap_violations.append(msg)
                if self.strict:
                    self.errors.append(msg)
                else:
                    self.warnings.append(msg)

            if total_missing > 5:
                self.warnings.append(f"{symbol}: {total_missing} total missing days")

        return {
            "valid": len(gap_violations) == 0 or not self.strict,
            "violations": gap_violations,
        }

    def validate_returns(self) -> Dict[str, Any]:
        """Detect extreme returns indicating data errors (Tech Spec 13.2)."""
        panel = self._load_panel()

        if panel is None:
            return {"valid": False}

        returns = panel.pct_change()

        extreme_events: List[Dict[str, Any]] = []
        for symbol in self.SYMBOLS:
            col = f"{symbol}_adj"
            if col not in returns.columns:
                continue

            extreme = returns[col].abs() > self.EXTREME_RETURN_THRESHOLD
            if extreme.any():
                dates = returns[extreme].index.tolist()
                vals = returns[col][extreme].values
                msg = f"{symbol}: {len(dates)} extreme returns detected"
                extreme_events.append(
                    {
                        "symbol": symbol,
                        "dates": [str(d) for d in dates[:5]],  # First 5
                        "values": [float(v) for v in vals[:5]],
                    }
                )
                self.errors.append(msg)

        return {"valid": len(extreme_events) == 0, "extreme_events": extreme_events}

    def validate_btc_alignment(self) -> Dict[str, Any]:
        """Verify BTC 7-day data aligned to business calendar (Tech Spec 2.3)."""
        panel = self._load_panel()

        if panel is None:
            return {"valid": False}

        btc_col = "BTC-USD_adj"

        if btc_col not in panel.columns:
            self.errors.append("BTC column missing from panel")
            return {"valid": False}

        # Check weekly availability
        btc_weekly = panel[btc_col].resample("W").apply(lambda x: x.notna().sum())
        weeks_below_threshold = (btc_weekly < self.BTC_WEEKLY_MIN_DAYS).sum()

        if weeks_below_threshold > 0:
            msg = f"BTC has {weeks_below_threshold} weeks with <{self.BTC_WEEKLY_MIN_DAYS} observations"
            self.errors.append(msg)
            return {"valid": False}

        return {
            "valid": True,
            "weeks_checked": len(btc_weekly),
            "min_weekly_obs": int(btc_weekly.min()),
        }

    def validate_dividends(self) -> Dict[str, Any]:
        """Verify dividend reinvestment calculation (Tech Spec 2.2)."""
        # This is a simplified check - in production you'd compute explicit
        # reinvestment and compare to Adjusted Close
        panel = self._load_panel()

        if panel is None:
            return {"valid": False}

        etf_symbols = ["GLD", "QQQ", "SPY", "SMH", "IYR", "ANGL"]
        results: Dict[str, Dict[str, float]] = {}

        for symbol in etf_symbols:
            col = f"{symbol}_adj"
            if col not in panel.columns:
                continue

            # Check for reasonable growth consistency
            prices = panel[col].dropna()
            if len(prices) < 252:
                continue

            # Annual returns should be within reasonable bounds
            # Compute actual time span for proper annualization
            first_date = prices.index[0]
            last_date = prices.index[-1]
            years = (last_date - first_date).days / 365.25

            if years <= 0:
                annual_return = np.nan
            else:
                annual_return = (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1

            if not -0.50 < annual_return < 1.00:  # -50% to +100% annual
                self.warnings.append(
                    f"{symbol}: unusual annual return {annual_return:.2%}"
                )

            results[symbol] = {
                "total_return": float((prices.iloc[-1] / prices.iloc[0]) - 1),
                "annual_return": float(annual_return),
            }

        return {"valid": True, "assets_checked": results}

    def _save_report(self, results: Dict[str, Any]) -> None:
        """Save validation report to staging directory."""
        output_path = Path("data/staging/_validation_report.json")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\n📄 Validation report saved to {output_path}")
        except Exception as e:
            error_msg = f"Failed to save validation report: {e}"
            print(f"\n❌ {error_msg}")
            self.errors.append(f"Report save failed: {str(e)}")
            results["status"] = "ERROR"

        # Print summary
        status_emoji = "✅" if results["status"] == "PASS" else "❌"
        print(f"\n{status_emoji} Validation Status: {results['status']}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Index-Coin data pipeline")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on warnings (for CI environment)"
    )
    args = parser.parse_args()

    validator = DataValidator(strict=args.strict)
    results = validator.validate_all()

    # Exit with error code if validation failed or errored
    if results["status"] != "PASS":
        sys.exit(1)

    print("\n✅ All validation checks passed!")


if __name__ == "__main__":
    main()
