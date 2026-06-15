def test_prediction_returns_return_based_contract(client):
    response = client.get("/api/v1/prediction/AAPL")
    assert response.status_code == 200
    data = response.json()

    assert data["trend"] in {"alcista", "bajista", "neutral"}
    assert isinstance(data["predicted_return_pct"], float)
    assert isinstance(data["predicted_price"], float)
    assert isinstance(data["current_price"], float)
    assert data["horizon_days"] == 10
    assert "trained_at" in data

    metrics = data["metrics"]
    assert set(metrics) == {"rmse", "mae", "directional_accuracy"}

    # Price-prediction leftovers must be gone.
    assert "pct_change" not in data
    assert "mape" not in metrics
    assert "r2" not in metrics


def test_prediction_rejects_invalid_ticker(client):
    response = client.get("/api/v1/prediction/toolong")
    assert response.status_code == 422
