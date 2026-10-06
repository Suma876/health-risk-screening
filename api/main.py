"""FastAPI app. Run:  uvicorn api.main:app --reload"""
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import db

log = logging.getLogger("risk-api")
app = FastAPI(title="Diabetes Risk Screening API")

# ---------- input validation ----------
RANGES = {
    "pregnancies": (0, 20), "glucose": (40, 500), "blood_pressure": (30, 200),
    "skin_thickness": (1, 100), "insulin": (1, 900), "bmi": (10, 70),
    "pedigree": (0.05, 2.5), "age": (1, 120),
}


class PatientIn(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=40, le=500)
    blood_pressure: float = Field(ge=30, le=200)
    skin_thickness: Optional[float] = Field(default=None, ge=1, le=100)  # optional
    insulin: Optional[float] = Field(default=None, ge=1, le=900)         # optional
    bmi: float = Field(ge=10, le=70)
    pedigree: float = Field(ge=0.05, le=2.5)
    age: int = Field(ge=1, le=120)


@app.exception_handler(RequestValidationError)
async def friendly_validation_errors(request: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        field = str(e["loc"][-1]) if e["loc"] else "body"
        if field in RANGES:
            lo, hi = RANGES[field]
            if e["type"] == "missing":
                msg = f"{field} is required"
            elif e["type"] in ("greater_than_equal", "less_than_equal"):
                msg = f"{field} must be between {lo} and {hi}"
            else:
                msg = f"{field} must be a number between {lo} and {hi}"
        else:
            msg = e["msg"]
        errors.append({"field": field, "message": msg})
    return JSONResponse(status_code=422, content={"errors": errors})


# ---------- model loading (fails with a clear 503, never a crash) ----------
_cache = {"key": None, "model": None}


def get_model() -> dict:
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.json"))
    if not path.exists():
        raise HTTPException(503, f"Model file not found at '{path}'. "
                                 "Run: python -m risk_model.run_experiments")
    key = (str(path), path.stat().st_mtime)
    if _cache["key"] != key:
        try:
            _cache.update(key=key, model=json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            raise HTTPException(503, f"Model file '{path}' is unreadable or corrupt.")
    return _cache["model"]


# ---------- plain-language output ----------
MESSAGES = {
    "en": {"high": "Higher risk. Please see a doctor soon for a blood sugar test.",
           "borderline": "Moderate risk. Please get a check-up when you can.",
           "low": "Lower risk. Keep up a healthy diet and regular activity."},
    "hi": {"high": "जोखिम अधिक है। कृपया जल्द डॉक्टर से मिलें और शुगर की जाँच कराएँ।",
           "borderline": "जोखिम मध्यम है। कृपया डॉक्टर से जाँच कराएँ।",
           "low": "जोखिम कम है। स्वस्थ भोजन और नियमित व्यायाम जारी रखें।"},
    "kn": {"high": "ಅಪಾಯ ಹೆಚ್ಚಿದೆ. ದಯವಿಟ್ಟು ಶೀಘ್ರದಲ್ಲೇ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
           "borderline": "ಅಪಾಯ ಮಧ್ಯಮವಾಗಿದೆ. ದಯವಿಟ್ಟು ವೈದ್ಯರಿಂದ ತಪಾಸಣೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ.",
           "low": "ಅಪಾಯ ಕಡಿಮೆ ಇದೆ. ಆರೋಗ್ಯಕರ ಜೀವನಶೈಲಿಯನ್ನು ಮುಂದುವರಿಸಿ."},
}
LABELS = {"pregnancies": "Number of pregnancies", "glucose": "Glucose level",
          "blood_pressure": "Blood pressure", "skin_thickness": "Skin-fold thickness",
          "insulin": "Insulin level", "bmi": "Body mass index (BMI)",
          "pedigree": "Family history score", "age": "Age"}
DISCLAIMER = "This is a screening aid, not a diagnosis. Please consult a doctor."


@app.get("/health")
def health():
    get_model()
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientIn, lang: str = Query("en")):
    if lang not in MESSAGES:
        raise HTTPException(400, "lang must be one of: en, hi, kn")
    t0 = time.perf_counter()
    m = get_model()

    vals = patient.model_dump()
    for k in ("skin_thickness", "insulin"):          # unknown -> training median
        if vals[k] is None:
            vals[k] = m["medians"][k]

    z = {f: (vals[f] - m["mean"][f]) / m["std"][f] for f in m["features"]}
    contrib = {f: w * z[f] for f, w in zip(m["features"], m["w"])}
    score = sum(contrib.values()) + m["b"]
    risk = 1 / (1 + math.exp(-score))

    t = m["threshold"]
    band = "high" if risk >= t else "borderline" if risk >= 0.7 * t else "low"

    top = sorted(contrib, key=lambda f: abs(contrib[f]), reverse=True)[:3]
    explanation = [f"{LABELS[f]} {'raises' if contrib[f] > 0 else 'lowers'} your risk"
                   for f in top]
    warnings = [f"{LABELS[f]} is unusual compared with the training data; "
                "double-check the value" for f in m["features"] if abs(z[f]) > 3]

    latency_ms = (time.perf_counter() - t0) * 1000
    try:
        db.save_request(vals, risk, band, latency_ms)
        saved = True
    except sqlite3.Error:
        log.exception("could not save request")
        saved = False                                 # still answer the user

    return {"risk": round(risk, 4), "risk_percent": round(risk * 100, 1),
            "band": band, "message": MESSAGES[lang][band],
            "explanation": explanation, "warnings": warnings,
            "threshold_used": t, "saved": saved, "disclaimer": DISCLAIMER}


@app.get("/stats")
def stats():
    try:
        return db.get_stats()
    except sqlite3.Error:
        raise HTTPException(503, "Database unavailable.")
