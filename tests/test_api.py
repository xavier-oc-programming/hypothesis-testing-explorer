from fastapi.testclient import TestClient
from main import app
import numpy as np

client = TestClient(app)

np.random.seed(42)
GROUP1 = (np.random.normal(50, 10, 100)).tolist()
GROUP2 = (np.random.normal(55, 10, 100)).tolist()
GROUP1_SAME = (np.random.normal(50, 10, 100)).tolist()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "pyspark_threshold" in data
    assert "supported_tests" in data


def test_independent_ttest_significant():
    response = client.post("/test/text", json={
        "group1": GROUP1, "group2": GROUP2,
        "test_name": "independent_ttest",
        "alpha": 0.05,
        "group1_name": "Group A", "group2_name": "Group B"
    })
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "p_value" in data
    assert "significant" in data
    assert data["engine"] == "pandas/scipy"


def test_mannwhitney():
    response = client.post("/test/text", json={
        "group1": GROUP1, "group2": GROUP2,
        "test_name": "mannwhitney", "alpha": 0.05,
        "group1_name": "A", "group2_name": "B"
    })
    assert response.status_code == 200


def test_unsupported_test():
    response = client.post("/test/text", json={
        "group1": GROUP1, "group2": GROUP2,
        "test_name": "invalid_test", "alpha": 0.05,
        "group1_name": "A", "group2_name": "B"
    })
    assert response.status_code == 422


def test_example_data():
    response = client.get("/api/example-data")
    assert response.status_code == 200
    data = response.json()
    assert "exam_scores" in data
    assert "sales_by_region" in data


def test_supported_tests():
    response = client.get("/api/tests")
    assert response.status_code == 200
    tests = response.json()
    names = [t["name"] for t in tests]
    assert "independent_ttest" in names
    assert "mannwhitney" in names
