# AI Draft Bot: Improvements Roadmap

## Executive Summary

After comprehensive code review, we identified 40+ improvement opportunities across 9 categories. This document prioritizes them for implementation before testing on The Last Airbender set.

---

## Priority 1: CRITICAL (Implement Now) 

###  1. Add Logging System ✅ IN PROGRESS
**Status**: Started (created `utils/logging_config.py`)
**Effort**: 1 hour
**Impact**: Essential for debugging The Last Airbender data

**Actions**:
- [x] Create logging infrastructure
- [ ] Add logging to data ingestion
- [ ] Add logging to feature extraction
- [ ] Add logging to model training

### 2. Missing Card Reporting
**Effort**: 30 minutes
**Impact**: Critical for data quality on new sets

**Problem**: Cards in draft logs but missing from metadata CSV are silently skipped
**Solution**: Track and report which cards are missing

### 3. Better CLI Error Messages
**Effort**: 30 minutes
**Impact**: User experience when testing new sets

**Current**: "No picks loaded. Aborting."
**Better**: Explain WHAT went wrong and HOW to fix it

---

## Priority 2: QUICK WINS (Easy & Valuable)

### 4. Performance: Cache Deck Stats
**Effort**: 2 hours
**Impact**: 5-10x faster feature extraction

**Problem**: `compute_deck_stats()` called redundantly in loop
**Solution**: Incremental updates instead of full recalculation

**Files**: `draft_context.py:189-201`

### 5. Add Progress Bars
**Effort**: 30 minutes  
**Impact**: UX for large datasets

**Solution**: Add `tqdm` progress bars for long operations

### 6. Model Versioning
**Effort**: 1 hour
**Impact**: Production readiness

**Add to joblib payload**:
```python
{
    "version": "2.0",
    "feature_dim": 78,
    "model_type": "xgboost",
    "trained_date": "2025-11-20",
    "set_name": "The Last Airbender",
    ...
}
```

---

## Priority 3: FEATURE IMPROVEMENTS (Accuracy Gains)

### 7. Wheel & Seat Position Signals
**Effort**: 4 hours
**Impact**: +5-10% accuracy (HIGH VALUE)

**Missing Signals**:
- Wheel likelihood (ALSA integration)
- Seat position effects (early vs late)
- Signal strength (how open are your colors?)

**Implementation**: Extend `draft_context.py` with 3-5 new features

### 8. Fix Archetype Detection Bug
**Effort**: 15 minutes
**Impact**: Correctness

**File**: `synergies.py:224`
```python
# BUG: Parentheses issue
elif archetype == "spells" and "instant" in card_keywords or "sorcery" in card_keywords:
# Should be:
elif archetype == "spells" and ("instant" in card_keywords or "sorcery" in card_keywords):
```

### 9. Improve Removal Detection
**Effort**: 2 hours
**Impact**: Better deck quality features

**Current**: Name-based heuristics ("murder", "bolt")
**Better**: Type-based + simple text analysis

---

## Priority 4: ARCHITECTURE (Long-term Health)

### 10. Feature Configuration System
**Effort**: 3 hours
**Impact**: Experimentation flexibility

**Goal**: Toggle features on/off without code changes

```python
@dataclass
class FeatureConfig:
    use_winrates: bool = True
    use_deck_context: bool = True  
    use_synergies: bool = True
    use_wheel_signals: bool = False  # Easy A/B testing
```

### 11. Fix Type Hints
**Effort**: 20 minutes
**Impact**: Code correctness

**Files**: `draft_state.py:61,68`
```python
# Wrong:
color_counts: Mapping[str, int] = field(default_factory=dict)
# Right:
color_counts: dict[str, int] = field(default_factory=dict)
```

### 12. Add Basic Test Suite
**Effort**: 4 hours
**Impact**: Robustness

**Critical Tests**:
- Data ingestion (malformed JSONL)
- Feature extraction (correct dimensions)
- Model save/load
- Edge cases (empty datasets)

---

## Priority 5: ADVANCED FEATURES (Big Accuracy Gains)

### 13. Card Text/Ability Parsing
**Effort**: 8+ hours
**Impact**: +10-15% accuracy (HIGHEST POTENTIAL)

**Goal**: Extract synergies from Oracle text
- Keywords: Flying, Deathtouch, Trample
- Mechanics: Prowess, Sacrifice triggers
- Synergy detection: "Whenever you cast a spell..."

**Approach**:
1. Simple keyword extraction (regex)
2. Build synergy lookup table from 17L data
3. Later: NLP embeddings (sentence transformers)

### 14. Hyperparameter Tuning
**Effort**: 4 hours
**Impact**: +2-5% accuracy

**Current**: Fixed hyperparameters (max_depth=8, lr=0.1)
**Better**: Bayesian optimization (Optuna)

```python
pip install optuna
```

### 15. Ensemble Model
**Effort**: 4 hours
**Impact**: +2-3% accuracy

**Combine**:
- Baseline (interpretable)
- XGBoost (accurate)
- LightGBM (fast)

Average predictions for robustness

---

## Testing Plan for The Last Airbender

### Before Testing:
1. ✅ Implement Priority 1 (logging, error handling)
2. ✅ Implement Priority 2 item #4 (performance)
3. ✅ Fix archetype bug (Priority 3 item #8)

### Data Preparation:
1. Download TLA draft logs from 17Lands
2. Download TLA card metadata CSV (ensure win rate columns present)
3. Verify data quality:
   ```bash
   wc -l drafts.jsonl  # How many lines?
   head -n 1 drafts.jsonl | jq .  # Valid JSON?
   head -n 1 cards.csv  # Has required columns?
   ```

### Training Baseline:
```bash
python scripts/train.py run \
    --drafts-path data/tla/drafts.jsonl \
    --metadata-path data/tla/cards.csv \
    --output-path artifacts/tla_baseline.joblib
```

### Training Advanced:
```bash
python scripts/train.py advanced \
    --drafts-path data/tla/drafts.jsonl \
    --metadata-path data/tla/cards.csv \
    --model-type xgboost \
    --output-path artifacts/tla_advanced.joblib
```

### Evaluation Metrics:
- Pick prediction accuracy (target: 55-70%)
- Feature importance (which features matter most?)
- Training time
- Model file size

---

## Implementation Timeline

**Today (2-3 hours)**:
- [x] Logging system
- [ ] Missing card reporting
- [ ] Better error messages
- [ ] Fix type hints
- [ ] Fix archetype bug

**This Week (8-10 hours)**:
- [ ] Performance optimization (caching)
- [ ] Wheel/seat signals
- [ ] Feature configuration
- [ ] Progress bars
- [ ] Model versioning

**Next Week (8-12 hours)**:
- [ ] Improve removal detection
- [ ] Basic test suite
- [ ] Hyperparameter tuning
- [ ] Better CLI evaluation

**Future (20+ hours)**:
- [ ] Card text parsing
- [ ] Ensemble model
- [ ] Comprehensive testing
- [ ] Documentation updates

---

## Success Metrics

**Minimum Viable** (must achieve):
- ✅ Baseline model: 35-45% accuracy on TLA
- ✅ Advanced model: 55-70% accuracy on TLA
- ✅ Training completes without errors
- ✅ Feature extraction handles TLA data

**Target** (stretch goals):
- 🎯 Advanced model: 65%+ accuracy
- 🎯 Training time: <15 minutes
- 🎯 Feature importance: GIH WR in top 3
- 🎯 All logging shows no data quality issues

**Excellence** (aspirational):
- 🌟 Advanced model: 70%+ accuracy (approaching human expert)
- 🌟 Identify TLA-specific archetypes automatically
- 🌟 Feature importance reveals new insights about TLA meta

---

## Known Risks

1. **TLA data quality**: New set may have incomplete win rate data
   - Mitigation: Check for NULL/missing win rates, report % coverage

2. **Set complexity**: TLA may have unique mechanics not captured
   - Mitigation: Review feature importance, identify gaps

3. **Performance**: Large TLA dataset may be slow
   - Mitigation: Implement caching FIRST

4. **Archetype mismatch**: Hard-coded archetypes may not fit TLA
   - Mitigation: Use learned archetypes from training data

---

## Next Actions

**Immediate (before testing)**:
1. Finish logging implementation
2. Add missing card tracking
3. Fix synergy bug
4. Add performance caching

**Then**:
5. Download TLA data
6. Train both models
7. Analyze results
8. Iterate based on findings

---

**Status**: Ready to proceed with Priority 1-2 implementation, then test on The Last Airbender! 🚀
