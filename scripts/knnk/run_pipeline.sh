#!/usr/bin/env bash
# Full KNNK pipeline: dataset → features → train → validate
set -e

cd "$(dirname "$0")/../.."
source venv/bin/activate

echo "========================================"
echo " KNNK Pipeline"
echo "========================================"

echo ""
echo "[1/4] Building dataset..."
python scripts/knnk/build_dataset.py

echo ""
echo "[2/4] Extracting features..."
python scripts/knnk/build_features.py

echo ""
echo "[3/4] Training classifier..."
python scripts/knnk/train_classifier.py

echo ""
echo "[4/4] Validating..."
python scripts/knnk/validate.py

echo ""
echo "========================================"
echo " KNNK Pipeline complete."
echo "========================================"
