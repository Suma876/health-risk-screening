from tests.conftest import GOOD


def test_predict_valid_input(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk"] <= 1
    assert body["band"] in {"low", "borderline", "high"}
    assert len(body["explanation"]) == 3


def test_predict_bad_input_gives_clear_error(client):
    r = client.post("/predict", json={**GOOD, "age": 0})
    assert r.status_code == 422
    assert r.json()["errors"][0]["message"] == "age must be between 1 and 120"
    r = client.post("/predict", json={**GOOD, "glucose": "abc"})
    assert r.status_code == 422
    assert "glucose must be a number" in r.json()["errors"][0]["message"]


def test_stats_counts_saved_requests(client):
    assert client.get("/stats").json()["total_requests"] == 0
    client.post("/predict", json=GOOD)
    client.post("/predict", json={**GOOD, "glucose": 300, "bmi": 45})
    s = client.get("/stats").json()
    assert s["total_requests"] == 2
    assert 0 < s["average_risk"] < 1
    assert 0 <= s["high_risk_share"] <= 1


def test_missing_model_returns_503_not_crash(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.json"))
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 503
    assert "Model file not found" in r.json()["detail"]
