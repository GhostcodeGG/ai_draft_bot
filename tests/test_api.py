import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_draft_bot.api import server
from ai_draft_bot.data.ingest_17l import CardMetadata
from ai_draft_bot.features.draft_context import PickFeatures


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Ensure global API state is clean between tests."""
    server._model = None
    server._metadata = None
    server._lstm_model = None
    server._lstm_encoder = None
    server._metadata_version = 0
    server._cached_feature_rows.cache_clear()
    yield
    server._model = None
    server._metadata = None
    server._lstm_model = None
    server._lstm_encoder = None
    server._metadata_version = 0
    server._cached_feature_rows.cache_clear()


def test_health_unhealthy_without_model() -> None:
    client = TestClient(server.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert body["status"] == "unhealthy"


def test_predict_returns_sorted_recommendations(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(server.app)

    # Minimal metadata
    server._metadata = {
        "Alpha": CardMetadata(
            name="Alpha",
            color="R",
            rarity="common",
            type_line="Creature",
            mana_value=2,
        ),
        "Beta": CardMetadata(
            name="Beta",
            color="U",
            rarity="uncommon",
            type_line="Instant",
            mana_value=3,
        ),
    }

    class DummyModel:
        def predict_proba(self, features: np.ndarray) -> dict[str, float]:
            # Favor Alpha regardless of input
            return {"Alpha": 0.9, "Beta": 0.1}

        def get_label_encoder(self):
            class _Encoder:
                classes_ = np.array(["Alpha", "Beta"])

            return _Encoder()

    server._model = DummyModel()

    def fake_build(picks, metadata, archetypes):
        return [
            PickFeatures(
                features=np.array([0.1, 0.2]),
                label=pick.chosen_card,
                event_id=pick.event_id,
            )
            for pick in picks
        ]

    monkeypatch.setattr(server, "build_ultra_advanced_pick_features", fake_build)

    resp = client.post(
        "/predict",
        json={
            "pack": ["Alpha", "Beta"],
            "deck": [],
            "pack_number": 1,
            "pick_number": 1,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    names = [rec["card_name"] for rec in body["recommendations"]]
    assert names == ["Alpha", "Beta"], "Recommendations should be sorted by confidence"
    assert body["pack_size"] == 2


def test_load_lstm_endpoint_uses_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(server.app)
    calls: list[tuple[str, str]] = []

    def fake_loader(model_path, encoder_path):
        calls.append((str(model_path), str(encoder_path)))

    monkeypatch.setattr(server, "load_lstm_model", fake_loader)

    resp = client.post(
        "/load/lstm",
        params={"model_path": "model.pt", "encoder_path": "encoder.joblib"},
    )

    assert resp.status_code == 200
    assert calls == [("model.pt", "encoder.joblib")]
