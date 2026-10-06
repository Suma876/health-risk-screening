
"""Front end. Run: streamlit run app_ui.py (API must be running)"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

from health_knowledge.symptom_checker import load_condition, check_symptoms


API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Personal Health Risk Check",
    page_icon="🩺"
)

st.title("🩺 Personal Health Risk Check")
st.caption("A screening aid, not a diagnosis.")


# ============================================================
# LANGUAGE
# ============================================================

lang = st.selectbox(
    "Language",
    ["en", "hi", "kn"],
    format_func={
        "en": "English",
        "hi": "हिन्दी",
        "kn": "ಕನ್ನಡ"
    }.get
)


# ============================================================
# DIABETES RISK SCREENING
# ============================================================

st.header("🩸 Diabetes Risk Screening")

c1, c2 = st.columns(2)

age = c1.number_input(
    "Age",
    0,
    130,
    35
)

preg = c2.number_input(
    "Pregnancies",
    0,
    30,
    0
)

glucose = c1.number_input(
    "Glucose (mg/dL)",
    0.0,
    600.0,
    110.0
)

bp = c2.number_input(
    "Blood pressure (diastolic, mmHg)",
    0.0,
    250.0,
    72.0
)

bmi = c1.number_input(
    "BMI",
    0.0,
    100.0,
    27.0
)

ped = c2.number_input(
    "Family history score",
    0.0,
    5.0,
    0.4
)

unknown = st.checkbox(
    "I don't know my skin-fold / insulin values",
    value=True
)

skin = None
insulin = None

if not unknown:
    skin = st.number_input(
        "Skin-fold thickness (mm)",
        0.0,
        150.0,
        20.0
    )

    insulin = st.number_input(
        "Insulin (mu U/ml)",
        0.0,
        1000.0,
        80.0
    )


# ============================================================
# MODEL / THRESHOLD
# ============================================================

model_path = Path(
    os.getenv("MODEL_PATH", "artifacts/model.json")
)

if model_path.exists():
    table = json.loads(
        model_path.read_text(encoding="utf-8")
    )["threshold_table"]
else:
    table = []


thr = st.slider(
    "Screening threshold",
    0.05,
    0.95,
    0.20,
    0.05
)


if table:
    row = min(
        table,
        key=lambda r: abs(r["threshold"] - thr)
    )

    st.caption(
        f"At threshold {thr:.2f} on test data: "
        f"recall {row['recall']:.0%} "
        f"(share of real cases caught), "
        f"precision {row['precision']:.0%}."
    )


# ============================================================
# DIABETES PREDICTION
# ============================================================

if st.button(
    "Check my diabetes risk",
    type="primary"
):

    body = {
        "pregnancies": preg,
        "glucose": glucose,
        "blood_pressure": bp,
        "skin_thickness": skin,
        "insulin": insulin,
        "bmi": bmi,
        "pedigree": ped,
        "age": age
    }

    try:
        r = requests.post(
            f"{API}/predict",
            params={"lang": lang},
            json=body,
            timeout=10
        )

    except requests.RequestException:
        st.error(
            "Cannot reach the server. "
            "Please make sure the FastAPI server is running."
        )
        st.stop()

    if r.status_code == 422:

        try:
            errors = r.json()["errors"]

            for e in errors:
                st.error(e["message"])

        except Exception:
            st.error("Invalid input.")

    elif r.status_code != 200:

        try:
            st.error(
                r.json().get(
                    "detail",
                    "Something went wrong."
                )
            )
        except Exception:
            st.error("Something went wrong.")

    else:

        d = r.json()

        flagged = d["risk"] >= thr

        if flagged:
            st.error(d["message"])
        else:
            st.success(d["message"])

        st.metric(
            "Estimated risk",
            f"{d['risk_percent']}%"
        )

        st.write("**Main reasons**")

        for line in d["explanation"]:
            st.write(f"- {line}")

        for w in d["warnings"]:
            st.warning(w)

        st.caption(d["disclaimer"])


# ============================================================
# COMMON SYMPTOM CHECKER
# ============================================================

st.divider()

st.header("🤒 Common Symptom Check")

st.caption(
    "Select your symptoms to see which common-cold symptoms "
    "match our knowledge base. This is not a diagnosis."
)


cold_symptoms = st.multiselect(
    "What symptoms are you currently experiencing?",
    [
        "runny nose",
        "stuffy nose",
        "sneezing",
        "sore throat",
        "cough",
        "mild fatigue",
        "mild headache"
    ]
)


if st.button("Check symptoms"):

    if not cold_symptoms:

        st.warning(
            "Please select at least one symptom."
        )

    else:

        try:

            condition = load_condition(
                "common_cold.json"
            )

            result = check_symptoms(
                cold_symptoms,
                condition
            )

            st.subheader("Screening result")

            st.info(
                f"Possible pattern: "
                f"{result['condition']} "
                f"(symptom match: "
                f"{result['match_score']}%)"
            )

            st.write("**Matched symptoms**")

            if result["matched_symptoms"]:

                for symptom in result["matched_symptoms"]:
                    st.write(f"- {symptom}")

            else:

                st.write(
                    "No matching symptoms found."
                )


            st.write("**Self-care**")

            for item in result["self_care"]:
                st.write(f"- {item}")


            st.write("**⚠️ Red flags**")

            for item in result["red_flags"]:
                st.warning(item)


            st.write("**When to seek medical care**")

            for item in result["seek_medical_care"]:
                st.write(f"- {item}")


            st.caption(
                result["disclaimer"]
            )

        except Exception as e:

            st.error(
                f"Unable to load the symptom knowledge base: {e}"
            )

