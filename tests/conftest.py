import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "pedigree", "age"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    model = {
        "features": FEATURES,
        "w": [0.3, 1.2, 0.1, 0.1, 0.1, 0.7, 0.3, 0.4], "b": -0.8,
        "mean": dict(zip(FEATURES, [3.8, 121, 72, 29, 125, 32, 0.47, 33])),
        "std": dict(zip(FEATURES, [3.3, 30, 12, 9, 80, 7, 0.33, 11])),
        "medians": dict(zip(FEATURES, [3, 117, 72, 29, 125, 32, 0.37, 29])),
        "threshold": 0.3, "threshold_table": [],
    }
    (tmp_path / "model.json").write_text(json.dumps(model))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.json"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from api.main import app
    return TestClient(app)


GOOD = {"pregnancies": 2, "glucose": 120, "blood_pressure": 70,
        "bmi": 28, "pedigree": 0.4, "age": 33}
