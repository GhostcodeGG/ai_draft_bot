# AI Draft Bot - Optimization Roadmap

**Date:** 2025-11-23
**Goal:** Production-ready deployment with optimal performance

---

## Quick Reference

| Priority | Item | Impact | Effort | Timeline |
|----------|------|--------|--------|----------|
| 🔴 P0 | Pre-compute Scryfall features | High | Low | 1 day |
| 🔴 P0 | Add LSTM to API | Medium | Low | 1 day |
| 🔴 P0 | Feature caching layer | High | Medium | 2 days |
| 🟡 P1 | Model quantization | High | Medium | 2 days |
| 🟡 P1 | Comprehensive tests | High | High | 1 week |
| 🟡 P1 | Batch prediction endpoint | Medium | Low | 1 day |
| 🟢 P2 | Prometheus metrics | Low | Medium | 3 days |
| 🟢 P2 | Natural language explanations | Medium | Medium | 4 days |

---

## Phase 1: Quick Wins (1 Week)

### 1. Pre-compute Scryfall Card Features
**Current Problem:** API calls to Scryfall during feature extraction slow down training (75ms per card × 2000 cards = 2.5 minutes overhead)

**Solution:**
```bash
# Create offline card database
python scripts/cache_scryfall.py \
    --sets "BRO,ONE,MOM,WOE,LCI" \
    --output cache/scryfall/card_features.json
```

**Implementation:**
1. Create `scripts/cache_scryfall.py`:
   - Fetch all cards from specified sets
   - Extract text features (keywords, removal, card advantage)
   - Save to JSON with card name as key
   - Update quarterly when new sets release

2. Update `features/card_text.py`:
   - Load from cache first
   - Fall back to API if card not found
   - Add cache invalidation logic

**Expected Impact:**
- ✅ 10x faster training (2.5 min → 15 sec for Scryfall portion)
- ✅ No API rate limiting issues
- ✅ Offline training capability

---

### 2. Add LSTM Sequence Endpoint to API
**Current Problem:** LSTM model exists but not exposed in production API

**Solution:**
```python
# In api/server.py
@app.post("/predict/sequence")
async def predict_sequence(request: PredictRequest):
    """LSTM-based sequence prediction (considers draft history)."""
    if _lstm_model is None:
        raise HTTPException(503, "LSTM model not loaded")

    # Use request.deck as draft history
    # Use request.pack as current pack
    recommendation = _lstm_model.predict(
        picked_cards=request.deck,
        pack_cards=request.pack,
        encoder=_lstm_encoder
    )
    return PredictResponse(recommendations=[recommendation])
```

**Implementation:**
1. Add LSTM model loading in `lifespan()` function
2. Create `/predict/sequence` endpoint
3. Add LSTM-specific request validation
4. Update API_README.md with new endpoint

**Expected Impact:**
- ✅ +5-8% accuracy for sequence-aware predictions
- ✅ Better handling of new cards (via embeddings)
- ✅ User choice between XGBoost (fast) and LSTM (accurate)

---

### 3. Feature Caching Layer
**Current Problem:** Static card features (mana value, color, rarity) recomputed every request

**Solution:**
```python
# In api/server.py
from functools import lru_cache

@lru_cache(maxsize=2000)
def get_card_static_features(card_name: str) -> dict:
    """Cache static features that never change."""
    card = _metadata[card_name]
    return {
        "mana_value": card.mana_value,
        "color": card.color,
        "rarity": card.rarity,
        # ... other static features
    }
```

**Implementation:**
1. Identify static vs dynamic features
   - Static: mana value, color, rarity, card text features
   - Dynamic: win rate context, deck synergy, pack position
2. Add LRU cache decorator to static feature extraction
3. Benchmark latency improvement

**Expected Impact:**
- ✅ 30-40% latency reduction (50ms → 30-35ms)
- ✅ Lower CPU usage
- ✅ Better scalability

---

### 4. Comprehensive API Tests
**Current Problem:** Only one test file exists (test_feature_dimensions.py)

**Solution:**
```python
# tests/test_api.py
def test_predict_endpoint_success():
    response = client.post("/predict", json={
        "pack": ["Lightning Bolt", "Counterspell"],
        "deck": [],
        "pack_number": 1,
        "pick_number": 1
    })
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 2

def test_predict_endpoint_unknown_card():
    response = client.post("/predict", json={
        "pack": ["Invalid Card Name"],
        "deck": [],
        "pack_number": 1,
        "pick_number": 1
    })
    assert response.status_code == 400
    assert "Unknown cards" in response.json()["error"]
```

**Implementation:**
1. Create test fixtures (small card metadata, mock models)
2. Add tests for `/predict`, `/explain`, `/health` endpoints
3. Test error handling (unknown cards, empty packs, invalid JSON)
4. Test edge cases (single card pack, full deck)
5. Add CI/CD integration in `.github/workflows/ci.yml`

**Expected Impact:**
- ✅ >80% test coverage
- ✅ Catch regressions early
- ✅ Confidence in refactoring

---

## Phase 2: Production Readiness (2 Weeks)

### 5. Model Quantization
**Problem:** Model size and inference speed could be optimized

**Solution:**
```python
# Quantize XGBoost model
import xgboost as xgb
model_quantized = xgb.Booster()
model_quantized.load_model("model.json")
model_quantized.save_model("model_quantized.json", format="json_fp16")
```

**Expected Impact:**
- ✅ 40-50% faster inference
- ✅ 60% smaller model size (500MB → 200MB)
- ✅ Better mobile deployment

---

### 6. Rate Limiting and Authentication
**Problem:** Open API vulnerable to abuse

**Solution:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("30/minute")  # 30 requests per minute
async def predict(request: PredictRequest):
    ...
```

**Implementation:**
1. Add `slowapi` for rate limiting
2. Add API key authentication (optional)
3. Add CORS configuration per environment
4. Add IP whitelist for trusted clients

**Expected Impact:**
- ✅ Protection against abuse
- ✅ Fair resource allocation
- ✅ API key tracking for analytics

---

### 7. Prometheus Metrics
**Problem:** No observability into API performance

**Solution:**
```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter("api_requests_total", "Total API requests")
request_latency = Histogram("api_request_latency_seconds", "Request latency")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**Metrics to track:**
- Request count by endpoint
- Request latency (p50, p95, p99)
- Error rate by type
- Model inference time
- Cache hit rate

**Expected Impact:**
- ✅ Real-time performance monitoring
- ✅ Alerting on degradation
- ✅ Capacity planning data

---

### 8. Environment Variable Documentation
**Problem:** Config scattered across code, no single reference

**Solution:**
Create `.env.example`:
```bash
# Model Configuration
MODEL_PATH=artifacts/ultra_model.joblib
METADATA_PATH=data/cards.csv
MODEL_TYPE=xgboost  # xgboost, lightgbm, lstm

# Cache Configuration
SCRYFALL_CACHE_ENABLED=true
SCRYFALL_CACHE_DIR=cache/scryfall
FEATURE_CACHE_ENABLED=true

# API Configuration
LOG_LEVEL=INFO
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,https://example.com

# Performance
USE_GPU=false
MAX_WORKERS=4
BATCH_SIZE=32

# Rate Limiting
RATE_LIMIT_PER_MINUTE=30
```

**Expected Impact:**
- ✅ Easy deployment configuration
- ✅ Better developer onboarding
- ✅ Production/staging separation

---

## Phase 3: Advanced Features (1 Month)

### 9. Natural Language Explanations
**Problem:** SHAP values not user-friendly

**Solution:**
```python
def generate_explanation(card_name: str, shap_values: dict) -> str:
    top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    explanations = []
    for feature, value in top_features:
        if feature == "gih_wr":
            explanations.append(f"high win rate ({value:.0%} contribution)")
        elif feature == "color_synergy":
            explanations.append(f"fits your deck colors ({value:.0%} contribution)")

    return f"{card_name} is recommended because it has {', '.join(explanations)}."
```

**Expected Impact:**
- ✅ Better user trust
- ✅ Learning tool for players
- ✅ Differentiation from competitors

---

### 10. Batch Prediction Endpoint
**Problem:** Analyzing multiple picks requires multiple API calls

**Solution:**
```python
@app.post("/predict/batch")
async def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    """Process multiple packs in parallel."""
    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return results
```

**Expected Impact:**
- ✅ 3-5x throughput for bulk analysis
- ✅ Post-draft review feature
- ✅ Training data generation

---

### 11. Ensemble Model Serving
**Problem:** Ensemble models not exposed in API

**Solution:**
```python
@app.post("/predict/ensemble")
async def predict_ensemble(request: PredictRequest):
    """Weighted ensemble of XGBoost + LightGBM + LSTM."""
    xgb_pred = _xgb_model.predict(...)
    lgb_pred = _lgb_model.predict(...)
    lstm_pred = _lstm_model.predict(...)

    # Weighted average (0.4, 0.3, 0.3)
    ensemble_pred = 0.4 * xgb_pred + 0.3 * lgb_pred + 0.3 * lstm_pred
    return ensemble_pred
```

**Expected Impact:**
- ✅ +3-5% accuracy
- ✅ More robust predictions
- ⚠️ 3x slower inference (use for high-stakes picks)

---

### 12. Automated Data Pipeline
**Problem:** Manual download and processing of 17Lands exports

**Solution:**
1. Scheduled scraper (respecting 17Lands ToS)
2. Incremental model updates (weekly retraining)
3. Drift detection and alerting
4. Automated backtesting on new data

**Expected Impact:**
- ✅ Always up-to-date with latest meta
- ✅ New set support within 1 week of release
- ✅ Continuous improvement

---

## Implementation Priority Matrix

```
High Impact, Low Effort (DO FIRST)
├─ Pre-compute Scryfall features (1 day)
├─ Add LSTM to API (1 day)
└─ Feature caching (2 days)

High Impact, Medium Effort (DO NEXT)
├─ Model quantization (2 days)
├─ Comprehensive tests (1 week)
└─ Batch prediction (1 day)

Medium Impact, Low Effort (QUICK WINS)
├─ Environment variable docs (2 hours)
├─ Health check improvements (1 hour)
└─ Error message improvements (2 hours)

Low Impact, High Effort (DEFER)
├─ Kubernetes manifests (1 week)
├─ Distributed tracing (1 week)
└─ Automated data pipeline (2 weeks)
```

---

## Success Metrics

### Performance Targets
- ✅ API Latency: <30ms (from current 50ms)
- ✅ Throughput: 50+ req/sec (from current 20-30)
- ✅ Model Size: <200MB (from current 500MB)
- ✅ Cold Start: <5 sec (from current 10-15 sec)

### Quality Targets
- ✅ Test Coverage: >80% (from current <20%)
- ✅ Accuracy: 75-85% (maintain current)
- ✅ Uptime: >99.9%
- ✅ Error Rate: <0.1%

### Developer Experience Targets
- ✅ Setup Time: <5 min (with Docker)
- ✅ CI/CD Pipeline: <5 min
- ✅ Documentation: 9/10 (from current 8/10)

---

## Timeline Summary

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Quick Wins | Scryfall cache, LSTM API, feature caching, basic tests |
| **Week 2** | Production Prep | Model quantization, rate limiting, comprehensive tests |
| **Week 3** | Monitoring | Prometheus metrics, dashboards, alerting |
| **Week 4** | Advanced Features | NL explanations, batch endpoint, ensemble API |

---

## Next Steps

### Immediate Actions (Today)
1. ✅ Create `scripts/cache_scryfall.py` script
2. ✅ Update `features/card_text.py` to use cache
3. ✅ Add LSTM loading to `api/server.py`
4. ✅ Create test fixtures in `tests/data/`

### This Week
1. ✅ Implement feature caching with `@lru_cache`
2. ✅ Write comprehensive API tests
3. ✅ Benchmark latency improvements
4. ✅ Document environment variables

### This Month
1. ✅ Model quantization and deployment
2. ✅ Prometheus metrics and monitoring
3. ✅ Natural language explanations
4. ✅ Production deployment with Docker

---

**Questions or blockers?** See PROJECT_REVIEW.md for detailed analysis.
