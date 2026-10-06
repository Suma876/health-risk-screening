# Diabetes Risk Screening (Questions A + B)

Dataset: **Pima Indians Diabetes** (768 rows, 8 numeric features, ~35% positive).
Downloaded automatically from a public GitHub mirror; or pass a local CSV path:
`python -m risk_model.run_experiments path/to/pima.csv`

## Run
    pip install -r requirements.txt
    python -m risk_model.run_experiments        # Question A -> artifacts/model.json + plots
    uvicorn api.main:app --reload               # API  (docs at /docs)
    streamlit run app_ui.py                     # front end (second terminal)
    python -m pytest -q                         # tests

## Level 3 (Q.B) break-it demo
1. Missing model: `mv artifacts/model.json /tmp/` then POST /predict.
   BEFORE (naive code: `json.load(open(path))` at import) -> server crashes at startup / 500.
   AFTER -> clean 503 "Model file not found ... Run: python -m risk_model.run_experiments".
2. Text for a number: `"glucose": "abc"`.
   BEFORE (no validation) -> 500 from `float("abc")`.  AFTER -> 422 "glucose must be a number between 40 and 500".

## Safe for 100 concurrent users
- `uvicorn api.main:app --workers 4` (or gunicorn + uvicorn workers); model is loaded once and cached.
- SQLite runs in WAL mode with a timeout; for real load move to PostgreSQL with a connection pool.
- Put nginx in front for rate limiting and TLS; keep `/predict` stateless; add a load test (locust/k6).
- DB failure never blocks a prediction (`saved: false` is returned instead).
