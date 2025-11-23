# AI Draft Bot API

Real-time Magic: The Gathering draft pick predictions via REST API.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start the server
python scripts/serve.py \
    --model-path artifacts/advanced_model.joblib \
    --metadata-path data/cards.csv \
    --port 8000

# Server will be available at http://localhost:8000
```

### Docker Deployment

```bash
# Build and start with docker-compose
docker-compose up -d

# Or build manually
docker build -t ai-draft-bot-api:latest .
docker run -p 8000:8000 \
    -v $(pwd)/artifacts:/app/models:ro \
    -v $(pwd)/data:/app/data:ro \
    ai-draft-bot-api:latest
```

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "0.1.0",
  "model_type": "AdvancedDraftModel",
  "num_classes": 281,
  "uptime_seconds": 3600.5
}
```

### Predict

Get ranked card recommendations for a pack.

```bash
POST /predict
Content-Type: application/json

{
  "pack": ["Lightning Bolt", "Counterspell", "Llanowar Elves"],
  "deck": ["Swords to Plowshares", "Path to Exile"],
  "pack_number": 1,
  "pick_number": 3
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "card_name": "Lightning Bolt",
      "confidence": 0.94,
      "rank": 1
    },
    {
      "card_name": "Counterspell",
      "confidence": 0.78,
      "rank": 2
    },
    {
      "card_name": "Llanowar Elves",
      "confidence": 0.52,
      "rank": 3
    }
  ],
  "pack_size": 3,
  "deck_size": 2,
  "model_version": "0.1.0",
  "inference_time_ms": 45.2
}
```

### Explain

Get SHAP explanation for why a card was recommended.

```bash
POST /explain
Content-Type: application/json

{
  "pack": ["Lightning Bolt", "Counterspell"],
  "deck": ["Swords to Plowshares"],
  "card_to_explain": "Lightning Bolt",
  "pack_number": 1,
  "pick_number": 2,
  "top_k_features": 5
}
```

**Response:**
```json
{
  "card_name": "Lightning Bolt",
  "confidence": 0.94,
  "top_features": [
    {
      "feature_name": "gih_wr",
      "contribution": 0.18,
      "direction": "positive",
      "description": "Games in Hand Win Rate"
    },
    {
      "feature_name": "color_synergy",
      "contribution": 0.14,
      "direction": "positive"
    }
  ],
  "alternative_picks": [
    {"card_name": "Counterspell", "confidence": 0.78, "rank": 2}
  ],
  "explanation_text": "The model recommends Lightning Bolt with 94.0% confidence.",
  "inference_time_ms": 123.5
}
```

## Integration Examples

### Python

```python
import requests

# Get recommendations
response = requests.post("http://localhost:8000/predict", json={
    "pack": ["Lightning Bolt", "Counterspell", "Llanowar Elves"],
    "deck": [],
    "pack_number": 1,
    "pick_number": 1
})

recommendations = response.json()["recommendations"]
top_pick = recommendations[0]
print(f"Pick: {top_pick['card_name']} ({top_pick['confidence']:.1%})")
```

### JavaScript (Browser Extension)

```javascript
async function getRecommendation(pack, deck) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      pack: pack,
      deck: deck,
      pack_number: 1,
      pick_number: 1
    })
  });

  const data = await response.json();
  return data.recommendations[0];
}

// Usage
const pack = ["Lightning Bolt", "Counterspell"];
const recommendation = await getRecommendation(pack, []);
console.log(`Pick: ${recommendation.card_name}`);
```

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pack": ["Lightning Bolt", "Counterspell"],
    "deck": [],
    "pack_number": 1,
    "pick_number": 1
  }'
```

## Performance

- **Latency**: <100ms typical (depends on pack size and feature set)
- **Throughput**: ~20-50 requests/second (single worker)
- **Memory**: ~500MB-1GB (depending on model size)
- **Scaling**: Horizontal scaling via multiple workers/containers

## Configuration

Environment variables:

```bash
MODEL_PATH=/path/to/model.joblib         # Required: Path to trained model
METADATA_PATH=/path/to/cards.csv        # Required: Card metadata
LOG_LEVEL=INFO                          # Optional: DEBUG, INFO, WARNING, ERROR
SCRYFALL_CACHE_ENABLED=true            # Optional: Enable Scryfall caching
SCRYFALL_CACHE_DIR=/app/cache          # Optional: Cache directory
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid card names, malformed JSON)
- `500` - Internal Server Error (model error, feature extraction failure)
- `503` - Service Unavailable (model not loaded)

Error response format:
```json
{
  "error": "Unknown cards in pack",
  "detail": "Card 'Invalid Card' not found in metadata"
}
```

## Production Deployment

### Docker Swarm

```bash
docker stack deploy -c docker-compose.yml ai-draft-bot
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests.

### Monitoring

- Health endpoint: `/health`
- Metrics endpoint: `/metrics` (if Prometheus enabled)
- Logs: JSON structured logs to stdout

## Security

- CORS enabled for browser extensions (configure `allow_origins` in production)
- Rate limiting recommended (use nginx or API gateway)
- HTTPS recommended for production (use reverse proxy)
- No authentication by default (add if needed)

## Troubleshooting

**Model not loading:**
```bash
# Check paths
docker logs ai_draft_bot_api

# Verify model exists
docker exec ai_draft_bot_api ls -la /app/models/
```

**Slow inference:**
- Check model size (quantize for faster inference)
- Enable caching
- Use GPU if available (set `use_gpu=True` in config)

**Memory issues:**
- Reduce model complexity
- Use quantized models
- Limit worker count

## Next Steps

- [ ] Add authentication/API keys
- [ ] Add rate limiting
- [ ] Add request caching (Redis)
- [ ] Add batch prediction endpoint
- [ ] Add model versioning/A-B testing
- [ ] Add Prometheus metrics
- [ ] Add distributed tracing

## License

MIT License - See LICENSE file
