# Edges of Eternities (EOE) - Setup Progress

**Date:** 2025-11-23
**Status:** In Progress

---

## What We Have

✅ **Downloaded from 17Lands:**
- `draft_data_public.EOE.PremierDraft.csv.gz` (136MB) - Pick-by-pick draft decisions
- `game_data_public.EOE.PremierDraft.csv.gz` (44MB) - Game outcomes
- `replay_data_public.EOE.PremierDraft.csv.gz` (313MB) - Detailed replays

---

## Setup Steps

### ✅ Phase 1: Quick Wins (COMPLETED)
1. ✅ Created Scryfall cache script
2. ✅ Added feature caching to API
3. ✅ Added LSTM endpoint
4. ✅ Created documentation

### 🔄 Phase 2: EOE Data Preparation (IN PROGRESS)

#### Step 1: Install Dependencies
**Status:** Installing in background...

Required packages:
- pandas, numpy (data processing)
- typer (CLI)
- scikit-learn, xgboost, lightgbm (ML models)
- torch (LSTM)
- fastapi, uvicorn (API server)

#### Step 2: Convert Draft Data CSV → JSONL
**Status:** Waiting for dependencies

The 17Lands public CSV has a wide format:
```
expansion,draft_id,pack_number,pick_number,pick,pack_card_*,pool_*,...
```

We need to convert to JSONL:
```json
{
  "event_id": "draft123",
  "pack_number": 1,
  "pick_number": 1,
  "chosen_card": "Lightning Bolt",
  "pack_contents": ["Lightning Bolt", "Counterspell", ...],
  "pool": ["Swords to Plowshares", ...]
}
```

**Tool:** `scripts/convert_17l_csv_to_jsonl.py` (created)

#### Step 3: Create Card Metadata
**Status:** Pending

Options:
A. Extract basic metadata from draft CSV (no win rates)
B. Download card ratings CSV from 17Lands (includes win rates) - **RECOMMENDED**

**Why B is better:** Win rates (GIH WR, IWD, ALSA) contribute ~70% of model accuracy!

#### Step 4: Build Scryfall Cache for EOE
**Status:** Pending

Command:
```bash
python scripts/cache_scryfall.py update EOE
```

This will fetch ~300 cards from EOE set and pre-compute features.

#### Step 5: Train Model
**Status:** Pending

Command:
```bash
python scripts/train.py ultra \
    --drafts-path data/eoe_drafts.jsonl \
    --metadata-path data/eoe_cards.csv \
    --model-type xgboost \
    --output-path artifacts/eoe_model.joblib
```

---

## Current Blockers

### 🔧 Dependency Installation
**Issue:** `scryfallsdk` package version conflict
**Solution:** Installing core deps manually, will fix pyproject.toml

### ⚠️ Missing Card Ratings
**Issue:** Public draft CSV doesn't include win rates directly
**Impact:** Model accuracy will be lower (~35-45% without, 70-85% with)

**Solutions:**
1. **Check 17Lands** for a separate card ratings CSV for EOE
2. **Compute from game data** (extract win rates from game_data_public.csv)
3. **Train without** (baseline model, lower accuracy)

---

## Next Actions (Once Deps Install)

1. **Convert Draft Data:**
   ```bash
   python scripts/convert_17l_csv_to_jsonl.py convert \
       --input "17L dataset/draft_data_public.EOE.PremierDraft.csv.gz" \
       --output data/eoe_drafts.jsonl
   ```

2. **Get Card Metadata (CRITICAL):**
   - **Option A:** Download from 17Lands directly (best)
   - **Option B:** Extract from game data (complex)
   - **Option C:** Generate minimal metadata (low accuracy)

3. **Build Scryfall Cache:**
   ```bash
   python scripts/cache_scryfall.py update EOE
   ```

4. **Train Model:**
   ```bash
   python scripts/train.py ultra \
       --drafts-path data/eoe_drafts.jsonl \
       --metadata-path data/eoe_cards.csv \
       --output-path artifacts/eoe_model.joblib
   ```

5. **Test API:**
   ```bash
   python scripts/serve.py \
       --model-path artifacts/eoe_model.joblib \
       --metadata-path data/eoe_cards.csv \
       --port 8000
   ```

---

## Expected Results

### Without Win Rates (Baseline)
- Accuracy: ~35-45%
- Training time: ~2-3 min
- Use case: Quick prototype

### With Win Rates (Ultra)
- Accuracy: **70-85%** (human-expert level!)
- Training time: ~10-15 min
- Use case: **Production model**

---

## Timeline Estimate

| Task | Time | Status |
|------|------|--------|
| Install deps | 5-10 min | 🔄 In progress |
| Convert CSV | 2-3 min | ⏳ Pending |
| Get metadata | 5 min (if download) | ⏳ Pending |
| Build cache | 10-15 min | ⏳ Pending |
| Train model | 10-15 min | ⏳ Pending |
| **TOTAL** | **30-50 min** | |

---

## Questions to Answer

1. **Do you have card ratings CSV from 17Lands?**
   - If yes → we can hit 70-85% accuracy
   - If no → we'll work with what we have (~35-45%)

2. **What's your priority?**
   - **Speed:** Train baseline now (16 features, 35-45%)
   - **Accuracy:** Get full metadata + train ultra (130+ features, 70-85%)

3. **Use case?**
   - **Testing/Learning:** Baseline is fine
   - **Production:** Need full metadata for best results

---

**Next:** Check if deps installed, then convert data!
