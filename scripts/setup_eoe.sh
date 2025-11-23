#!/bin/bash
# Complete EOE dataset setup and model training pipeline
#
# This script:
# 1. Computes card win rates from game data
# 2. Converts draft CSV to JSONL
# 3. Builds Scryfall cache for EOE
# 4. Trains ultra model
#
# Usage: bash scripts/setup_eoe.sh

set -e  # Exit on error

echo "=========================================="
echo "EOE Dataset Setup Pipeline"
echo "=========================================="
echo ""

# Step 1: Compute card statistics from game data
echo "[1/4] Computing card win rates from game data..."
python scripts/compute_card_stats.py compute \
    --game-data "17L dataset/game_data_public.EOE.PremierDraft.csv.gz" \
    --output data/eoe_card_stats.csv

echo ""
echo "[1/4] ✓ Card statistics computed"
echo ""

# Step 2: Convert draft data to JSONL
echo "[2/4] Converting draft CSV to JSONL..."
python scripts/convert_17l_csv_to_jsonl.py convert \
    --input "17L dataset/draft_data_public.EOE.PremierDraft.csv.gz" \
    --output data/eoe_drafts.jsonl

echo ""
echo "[2/4] ✓ Draft data converted"
echo ""

# Step 3: Build Scryfall cache for EOE
echo "[3/4] Building Scryfall cache for EOE set..."
python scripts/cache_scryfall.py update EOE

echo ""
echo "[3/4] ✓ Scryfall cache built"
echo ""

# Step 4: Train ultra model
echo "[4/4] Training ultra model..."
python scripts/train.py ultra \
    --drafts-path data/eoe_drafts.jsonl \
    --metadata-path data/eoe_card_stats.csv \
    --model-type xgboost \
    --output-path artifacts/eoe_ultra_model.joblib

echo ""
echo "[4/4] ✓ Model trained"
echo ""

echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Model saved to: artifacts/eoe_ultra_model.joblib"
echo ""
echo "Next steps:"
echo "  1. Start API server:"
echo "     python scripts/serve.py \\"
echo "       --model-path artifacts/eoe_ultra_model.joblib \\"
echo "       --metadata-path data/eoe_card_stats.csv \\"
echo "       --port 8000"
echo ""
echo "  2. Test predictions:"
echo "     curl -X POST http://localhost:8000/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"pack\": [\"Lightning Bolt\", \"Counterspell\"], \"deck\": []}'"
echo ""
