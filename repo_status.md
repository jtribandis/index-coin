# Repository Status

## Directory Structure
.
./artifacts
./artifacts/models
./artifacts/portfolio
./artifacts/reports
./artifacts/weights
./backtests
./backtests/configs
./data
./data/curated
./data/raw
./data/staging
./logs
./src
./src/allocation
./src/data
./src/eval
./src/features
./src/modeling
./src/rebalance
./src/sim
./src/utils
./tests
./tests/golden
./tests/golden/fixtures
./tests/golden/snapshots
./tests/integration
./tests/unit
./web
./web/public
./web/public/data

## Source Files
src/features/metrics.py
src/features/z_scores.py
src/features/standardization.py
src/features/__init__.py
src/features/returns.py
src/features/momentum.py
src/features/volatility.py
src/__init__.py
src/utils/__init__.py
src/modeling/__init__.py
src/rebalance/__init__.py
src/eval/__init__.py
src/sim/__init__.py
src/data/ingest.py
src/data/__init__.py
src/allocation/__init__.py

## Test Files
tests/unit/test_metrics.py
tests/unit/test_volatility.py
tests/unit/test_returns.py
tests/unit/test_z_scores.py
tests/unit/__init__.py
tests/unit/test_ingest.py
tests/unit/test_momentum.py
tests/unit/test_standardization.py
tests/integration/test_placeholder.py
tests/integration/__init__.py
tests/golden/test_placeholder.py
tests/golden/__init__.py
tests/__init__.py

## Data Files
The data directory contains three subdirectories:
- `curated/` - Processed and validated data ready for analysis
- `raw/` - Original data files from external sources  
- `staging/` - Intermediate data files during processing pipeline
