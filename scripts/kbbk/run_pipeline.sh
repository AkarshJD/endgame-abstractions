#!/usr/bin/env bash
# Full KBBK pipeline: dataset → features → train → validate
set -e

cd "$(dirname "$0")/../.."
source venv/bin/activate

echo "========================================"
echo " KBBK Pipeline"
echo "========================================"

echo ""
echo "[1/4] Building dataset..."
python scripts/kbbk/build_dataset.py

echo ""
echo "[2/4] Extracting features..."
python scripts/kbbk/build_features.py

echo ""
echo "[3/4] Training classifier..."
python scripts/kbbk/train_classifier.py

echo ""
echo "[4/4] Validating..."
python scripts/kbbk/validate.py

echo ""
echo "========================================"
echo " KBBK Pipeline complete."
echo "========================================"
