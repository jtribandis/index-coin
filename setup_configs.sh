#!/bin/bash
echo "🚀 Setting up Index-Coin configuration files..."

cat > .coderabbit.yaml << 'EOF'
version: 1
policy:
  required_approvals: 1
  block_on:
    - test_failures
    - security_findings
    - golden_drift_unexplained
  reviewers:
    - "@jtribandis"

review:
  depth: standard
  rules:
    - id: metrics-determinism
      description: "Flag if determinism or metric ranges fail"
      require_checks:
        - "determinism"
        - "metrics"
    
    - id: golden-drift
      description: "Block if golden snapshots changed without PR explanation"
      require_checks:
        - "golden"
    
    - id: static-analysis
      require_checks:
        - "ruff"
        - "mypy"
        - "bandit"

ui:
  summarize_findings: true
  request_fixes_as_suggestions: true

labels:
  on_block: ["needs-fix"]
  on_pass: ["qa-pass"]
EOF

mkdir -p .github/workflows

cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## What & Why
- Summary of change:
- Expected impact on metrics/outputs:

## Safety & Repro
- [ ] Ran `make check` locally
- [ ] Golden tests pass
- [ ] Determinism check passes (seeded)
- [ ] No strategy drift (snapshots unchanged or explained)

## Artifacts
- Link to run log / screenshots:

## Golden Update Rationale (if applicable)
<!-- If golden snapshots changed, explain why -->
EOF

cat > CODEOWNERS << 'EOF'
# Require at least one review from these code owners
* @jtribandis
EOF

cat > .github/workflows/ci.yml << 'EOF'
name: CI Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  ruff:
    name: Ruff Linting
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install ruff
        run: pip install ruff
      - name: Run ruff check
        run: ruff check src tests
      - name: Run ruff format check
        run: ruff format --check src tests

  mypy:
    name: Type Checking (mypy)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install mypy pandas numpy scikit-learn
      - name: Run mypy
        run: mypy --strict --ignore-missing-imports src

  bandit:
    name: Security Scan (bandit)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install bandit
        run: pip install bandit
      - name: Run bandit
        run: bandit -ll -q -r src

  unit:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest -q tests/unit

  integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest -q tests/integration

  golden:
    name: Golden File Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run golden tests
        run: pytest -q tests/golden

  determinism:
    name: Determinism Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run determinism check
        run: python -m src.modeling.determinism_check --seed 42

  metrics:
    name: Metrics Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Validate metrics
        run: python -m src.eval.validate_metrics artifacts/portfolio/perf_metrics.json
EOF

cat > Makefile << 'EOF'
.PHONY: check test unit integration golden determinism metrics lint type sec clean

check: lint type sec unit integration golden determinism metrics

lint:
	python -m ruff check src tests
	python -m ruff format --check src tests

type:
	mypy --strict --ignore-missing-imports src

sec:
	bandit -ll -q -r src

unit:
	pytest -q tests/unit

integration:
	pytest -q tests/integration

golden:
	pytest -q tests/golden

determinism:
	python -m src.modeling.determinism_check --seed 42

metrics:
	python -m src.eval.validate_metrics artifacts/portfolio/perf_metrics.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
EOF

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/

# Data
data/raw/*.csv
data/staging/*.parquet
*.h5
*.hdf5

# Models
models/checkpoints/*.pkl
models/checkpoints/*.joblib

# Logs
logs/*.txt
logs/*.json

# OS
.DS_Store
Thumbs.db
EOF

echo ""
echo "✅ Configuration files created successfully!"
echo ""
echo "Files created:"
ls -la | grep -E "^\-|^d" | grep -v "total"
echo ""
echo ".github/ contents:"
ls -la .github/
echo ""
echo ".github/workflows/ contents:"
ls -la .github/workflows/
