from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
VALID = {"cabbage_pct": 71, "radish_pct": 7, "seasoning_pct": 20, "salt_pct": 2, "brine_salinity_pct": 11.1, "brining_hours": 12,
         "fermentation_temp_c": 4.8, "fermentation_hours": 42, "storage_temp_c": 1, "cabbage_origin": "평창", "line": "K1"}

def test_health_and_models():
    assert client.get("/api/health").json()["product"] == "평창꽃순이김치"
    body = client.get("/api/ml/overview").json(); assert body["data_mode"] == "demo" and len(body["metrics"]) == 3

def test_predict_and_unbalanced_recipe_rejected():
    response = client.post("/api/quality/predict", json=VALID); assert response.status_code == 200 and response.json()["prediction_score"] > 0
    assert client.post("/api/quality/predict", json={**VALID, "cabbage_pct": 60}).status_code == 422

def test_optimization_recipe_is_balanced_and_fallback_labeled():
    body = client.post("/api/ml/optimize", json={"target_score": 80, "max_material_cost": 85, "cabbage_origin": "평창", "line": "K2"}).json()
    assert abs(sum(body["settings"][key] for key in ["cabbage_pct", "radish_pct", "seasoning_pct", "salt_pct"]) - 100) < .01
    if not body["feasible"]: assert body["unmet_constraints"]

def test_records_are_bounded_filtered_and_json_safe():
    body = client.get("/api/data/records?limit=7&line=K1&cabbage_origin=%ED%8F%89%EC%B0%BD").json(); assert len(body["records"]) <= 7 and all(row["line"] == "K1" for row in body["records"])
    assert client.get("/api/data/records?limit=101").status_code == 422

