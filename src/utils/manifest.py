"""Manifest generator for Index-Coin data pipeline."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


# Default symbols for manifest operations
DEFAULT_SYMBOLS = ["GLD", "QQQ", "SPY", "SMH", "IYR", "ANGL", "BTC-USD"]


class ManifestGenerator:
    """Generates and validates data ingestion manifests."""

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "_ingest_manifest.json"

    def generate_manifest(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Generate manifest with file checksums for given symbols.

        Args:
            symbols: List of symbol names to include in manifest

        Returns:
            Dictionary containing manifest data
        """
        manifest: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbols": {},
        }

        for symbol in symbols:
            file_path = self.data_dir / f"{symbol}.csv"
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        data = f.read()
                    file_hash = hashlib.sha256(data).hexdigest()
                    size_bytes = file_path.stat().st_size
                    manifest["symbols"][symbol] = {
                        "file": f"{symbol}.csv",
                        "sha256": file_hash,
                        "size_bytes": size_bytes,
                    }
                except (OSError, PermissionError) as e:
                    manifest["symbols"][symbol] = {
                        "file": f"{symbol}.csv",
                        "sha256": None,
                        "size_bytes": 0,
                        "error": f"Failed to read file: {e}",
                    }
            else:
                manifest["symbols"][symbol] = {
                    "file": f"{symbol}.csv",
                    "sha256": None,
                    "size_bytes": 0,
                    "error": "File not found",
                }

        return manifest

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save manifest to JSON file."""
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)

            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except (PermissionError, OSError, IOError) as e:
            error_msg = f"Failed to save manifest to {self.manifest_path}: {e}"
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg) from e

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        """Load existing manifest from JSON file."""
        if not self.manifest_path.exists():
            return None

        try:
            with open(self.manifest_path, "r") as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except json.JSONDecodeError as e:
            # Log the error if you have a logger configured
            # logger.error(f"Failed to parse manifest file: {e}")
            print(f"Warning: Failed to parse manifest file {self.manifest_path}: {e}")
            return None
        except (OSError, IOError) as e:
            print(f"Warning: Failed to read manifest file {self.manifest_path}: {e}")
            return None

    def validate_manifest(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Validate manifest against current files.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dictionary with validation results
        """
        manifest = self.load_manifest()
        if not manifest:
            return {"valid": False, "error": "Manifest not found"}

        results: Dict[str, Any] = {
            "valid": True,
            "mismatches": [],
            "missing_files": [],
            "io_errors": [],
            "valid_files": [],
        }

        for symbol in symbols:
            file_path = self.data_dir / f"{symbol}.csv"

            if not file_path.exists():
                results["missing_files"].append(symbol)
                results["valid"] = False
                continue

            # Compute current file hash with exception handling
            try:
                with open(file_path, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
            except (OSError, PermissionError) as e:
                results["io_errors"].append({"symbol": symbol, "error": str(e)})
                results["valid"] = False
                print(f"Warning: Failed to read file for symbol {symbol}: {e}")
                continue
            except Exception as e:
                results["io_errors"].append({"symbol": symbol, "error": str(e)})
                results["valid"] = False
                print(
                    f"Warning: Unexpected error reading file for symbol {symbol}: {e}"
                )
                continue

            # Get expected hash from manifest
            expected_hash = manifest.get("symbols", {}).get(symbol, {}).get("sha256")

            if expected_hash and current_hash != expected_hash:
                results["mismatches"].append(symbol)
                results["valid"] = False
            else:
                results["valid_files"].append(symbol)

        return results


def generate_data_manifest(
    data_dir: str = "data/raw", symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate manifest for data directory.

    Args:
        data_dir: Directory containing data files
        symbols: List of symbols to include (defaults to common symbols)

    Returns:
        Generated manifest dictionary
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    generator = ManifestGenerator(data_dir)
    manifest = generator.generate_manifest(symbols)
    generator.save_manifest(manifest)

    return manifest


def validate_data_manifest(
    data_dir: str = "data/raw", symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate existing manifest.

    Args:
        data_dir: Directory containing data files
        symbols: List of symbols to validate (defaults to common symbols)

    Returns:
        Validation results dictionary
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    generator = ManifestGenerator(data_dir)
    return generator.validate_manifest(symbols)
