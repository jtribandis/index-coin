"""Manifest generator for Index-Coin data pipeline."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional


class ManifestGenerator:
    """Generates and validates data ingestion manifests."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "_ingest_manifest.json"
    
    def generate_manifest(self, symbols: List[str]) -> Dict:
        """
        Generate manifest with file checksums for given symbols.
        
        Args:
            symbols: List of symbol names to include in manifest
            
        Returns:
            Dictionary containing manifest data
        """
        manifest = {
            "generated_at": None,  # Will be set by caller if needed
            "symbols": {}
        }
        
        for symbol in symbols:
            file_path = self.data_dir / f"{symbol}.csv"
            
            try:
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    manifest["symbols"][symbol] = {
                        "file": f"{symbol}.csv",
                        "sha256": file_hash,
                        "size_bytes": file_path.stat().st_size
                    }
                else:
                    manifest["symbols"][symbol] = {
                        "file": f"{symbol}.csv",
                        "sha256": None,
                        "size_bytes": 0,
                        "error": "File not found"
                    }
            except OSError as e:
                manifest["symbols"][symbol] = {
                    "file": f"{symbol}.csv",
                    "sha256": None,
                    "size_bytes": 0,
                    "error": str(e)
                }
        
        return manifest
    
    def save_manifest(self, manifest: Dict) -> None:
        """Save manifest to JSON file."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def load_manifest(self) -> Optional[Dict]:
        """Load existing manifest from JSON file."""
        if not self.manifest_path.exists():
            return None
        
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
    
    def validate_manifest(self, symbols: List[str]) -> Dict:
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
        
        results = {
            "valid": True,
            "mismatches": [],
            "missing_files": [],
            "valid_files": []
        }
        
        for symbol in symbols:
            file_path = self.data_dir / f"{symbol}.csv"
            
            if not file_path.exists():
                results["missing_files"].append(symbol)
                results["valid"] = False
                continue
            
            # Compute current file hash
            with open(file_path, 'rb') as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Get expected hash from manifest
            expected_hash = manifest.get("symbols", {}).get(symbol, {}).get("sha256")
            
            if expected_hash and current_hash != expected_hash:
                results["mismatches"].append(symbol)
                results["valid"] = False
            else:
                results["valid_files"].append(symbol)
        
        return results


def generate_data_manifest(data_dir: str = "data/raw", symbols: List[str] = None) -> Dict:
    """
    Convenience function to generate manifest for data directory.
    
    Args:
        data_dir: Directory containing data files
        symbols: List of symbols to include (defaults to common symbols)
        
    Returns:
        Generated manifest dictionary
    """
    if symbols is None:
        symbols = ['GLD', 'QQQ', 'SPY', 'SMH', 'IYR', 'ANGL', 'BTC-USD']
    
    generator = ManifestGenerator(data_dir)
    manifest = generator.generate_manifest(symbols)
    generator.save_manifest(manifest)
    
    return manifest


def validate_data_manifest(data_dir: str = "data/raw", symbols: List[str] = None) -> Dict:
    """
    Convenience function to validate existing manifest.
    
    Args:
        data_dir: Directory containing data files
        symbols: List of symbols to validate (defaults to common symbols)
        
    Returns:
        Validation results dictionary
    """
    if symbols is None:
        symbols = ['GLD', 'QQQ', 'SPY', 'SMH', 'IYR', 'ANGL', 'BTC-USD']
    
    generator = ManifestGenerator(data_dir)
    return generator.validate_manifest(symbols)
