import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from health_knowledge.symptom_checker import check_symptoms, load_condition

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Personal Health Risk Check",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# PATHS / CONFIG
# =========================================================

API = os.getenv("API_URL", "http://127.0.0.1:8000")

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "artifacts/model.json")
)

CONDITION_PATH = Path(
    "health_knowledge/common_cold.json"
)

# =========================================================
# TRANSLATIONS
# =========================================================

TEXT = {
    "en": {
        "title": "🩺 Personal Health Risk Check",
        "caption": "A screening aid for awareness and early risk identification — not a medical diagnosis.",
        "diabetes": "🩸 Diabetes Risk",
        "symptoms": "🤒 Symptom Checker",
        "evaluation": "📊 Model Evaluation",
        "diabetes_title": "🩸 Diabetes Risk Screening",
        "diabetes_info": "This section estimates screening risk from health measurements. It does not diagnose diabetes.",
        "unknown": "I don't know my skin-fold / insulin values",
        "check": "🔎 Check My Risk",
        "age": "Age",
        "preg": "Pregnancies",
        "glucose": "Glucose (mg/dL)",
        "bp": "Blood pressure — diastolic (mmHg)",
        "bmi": "BMI",
        "family": "Family history score",
        "skin": "Skin-fold thickness (mm)",
        "insulin": "Insulin (μU/ml)",
        "symptom_info": "Select your symptoms to screen for possible common health patterns.",
        "select_symptoms": "Select your symptoms",
        "check_symptoms": "🩺 Check Symptoms",
        "model_info": "📊 Model Information",
        "threshold": "Screening threshold",
        "recall": "Recall",
        "precision": "Precision",
        "accuracy": "Accuracy",
        "threshold_help": "The threshold controls how easily the model flags a person as potentially at risk.",
        "evaluation_title": "📊 Model Evaluation",
        "evaluation_info": "These metrics are calculated on held-out test data and are intended to demonstrate model behaviour, not clinical performance.",
        "threshold_tradeoff": "Threshold Trade-off",
        "confusion_matrix": "Confusion Matrix",
        "cost_sensitive": "💰 Cost-Sensitive Learning",
        "visualizations": "📈 Experiment Visualizations",
        "self_care": "Self-care",
        "red_flags": "⚠️ Red Flags",
        "seek_care": "When to seek medical care",
        "matched": "Matched symptoms",
        "no_match": "No matching symptoms found.",
        "disclaimer": "This is a screening aid and does not diagnose disease or replace professional medical advice.",
        "lower_threshold": "A lower threshold generally catches more possible cases, but may also create more false alarms.",
        "recall_explanation": "Recall answers: Of the real positive cases, how many did we catch?",
        "precision_explanation": "Precision answers: Of the cases we flagged, how many were actually positive?",
        "tn": "True Negative (TN)",
        "fp": "False Positive (FP)",
        "fn": "False Negative (FN)",
        "tp": "True Positive (TP)",
        "cost_info": "Increasing the false-negative penalty makes the model more sensitive to missed positive cases. This can increase recall while reducing precision.",
        "medical_warning": "⚠️ These results are for educational screening only. They are not a medical diagnosis."
    },
    "hi": {
        "title": "🩺 व्यक्तिगत स्वास्थ्य जोखिम जाँच",
        "caption": "यह प्रारंभिक स्क्रीनिंग सहायता है, चिकित्सा निदान नहीं।",
        "diabetes": "🩸 मधुमेह जोखिम",
        "symptoms": "🤒 लक्षण जाँच",
        "evaluation": "📊 मॉडल मूल्यांकन",
        "diabetes_title": "🩸 मधुमेह जोखिम स्क्रीनिंग",
        "diabetes_info": "यह अनुभाग स्वास्थ्य मापदंडों के आधार पर प्रारंभिक जोखिम का अनुमान लगाता है। यह मधुमेह का निदान नहीं करता।",
        "unknown": "मुझे त्वचा की मोटाई / इंसुलिन का मान नहीं पता",
        "check": "🔎 मेरा जोखिम जाँचें",
        "age": "आयु",
        "preg": "गर्भधारण की संख्या",
        "glucose": "ग्लूकोज़ (mg/dL)",
        "bp": "रक्तचाप — डायस्टोलिक (mmHg)",
        "bmi": "BMI",
        "family": "पारिवारिक इतिहास स्कोर",
        "skin": "त्वचा की मोटाई (mm)",
        "insulin": "इंसुलिन (μU/ml)",
        "symptom_info": "संभावित स्वास्थ्य पैटर्न की जाँच के लिए अपने लक्षण चुनें।",
        "select_symptoms": "अपने लक्षण चुनें",
        "check_symptoms": "🩺 लक्षण जाँचें",
        "model_info": "📊 मॉडल जानकारी",
        "threshold": "स्क्रीनिंग सीमा",
        "recall": "रिकॉल",
        "precision": "प्रिसीजन",
        "accuracy": "सटीकता",
        "threshold_help": "यह सीमा नियंत्रित करती है कि मॉडल किसी व्यक्ति को संभावित जोखिम के रूप में कितनी आसानी से चिन्हित करता है।",
        "evaluation_title": "📊 मॉडल मूल्यांकन",
        "evaluation_info": "ये मेट्रिक्स अलग रखे गए टेस्ट डेटा पर आधारित हैं और मॉडल के व्यवहार को दिखाने के लिए हैं, चिकित्सा प्रदर्शन को नहीं।",
        "threshold_tradeoff": "स्क्रीनिंग सीमा का प्रभाव",
        "confusion_matrix": "कन्फ्यूज़न मैट्रिक्स",
        "cost_sensitive": "💰 लागत-संवेदनशील लर्निंग",
        "visualizations": "📈 प्रयोग के ग्राफ",
        "self_care": "स्व-देखभाल",
        "red_flags": "⚠️ चेतावनी संकेत",
        "seek_care": "चिकित्सकीय सहायता कब लें",
        "matched": "मिले हुए लक्षण",
        "no_match": "कोई मेल खाने वाला लक्षण नहीं मिला।",
        "disclaimer": "यह केवल स्क्रीनिंग सहायता है और बीमारी का निदान नहीं करती।",
        "lower_threshold": "कम सीमा से अधिक संभावित मामले पकड़े जा सकते हैं, लेकिन गलत चेतावनियाँ भी बढ़ सकती हैं।",
        "recall_explanation": "रिकॉल बताता है: वास्तविक सकारात्मक मामलों में से हमने कितने पकड़े?",
        "precision_explanation": "प्रिसीजन बताता है: जिन मामलों को हमने चिन्हित किया, उनमें से कितने वास्तव में सकारात्मक थे?",
        "tn": "सही नकारात्मक (TN)",
        "fp": "गलत सकारात्मक (FP)",
        "fn": "गलत नकारात्मक (FN)",
        "tp": "सही सकारात्मक (TP)",
        "cost_info": "गलत नकारात्मक मामलों की लागत बढ़ाने से मॉडल छूटे हुए सकारात्मक मामलों के प्रति अधिक संवेदनशील होता है। इससे रिकॉल बढ़ सकता है लेकिन प्रिसीजन कम हो सकता है।",
        "medical_warning": "⚠️ ये परिणाम केवल शैक्षणिक स्क्रीनिंग के लिए हैं। यह चिकित्सा निदान नहीं है।"
    },
    "kn": {
        "title": "🩺 ವೈಯಕ್ತಿಕ ಆರೋಗ್ಯ ಅಪಾಯ ಪರಿಶೀಲನೆ",
        "caption": "ಇದು ಪ್ರಾಥಮಿಕ ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯ ಮಾತ್ರ, ವೈದ್ಯಕೀಯ ರೋಗನಿರ್ಣಯವಲ್ಲ.",
        "diabetes": "🩸 ಮಧುಮೇಹ ಅಪಾಯ",
        "symptoms": "🤒 ಲಕ್ಷಣ ಪರಿಶೀಲನೆ",
        "evaluation": "📊 ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ",
        "diabetes_title": "🩸 ಮಧುಮೇಹ ಅಪಾಯ ಸ್ಕ್ರೀನಿಂಗ್",
        "diabetes_info": "ಈ ವಿಭಾಗವು ಆರೋಗ್ಯದ ಅಳತೆಗಳ ಆಧಾರದ ಮೇಲೆ ಪ್ರಾಥಮಿಕ ಅಪಾಯವನ್ನು ಅಂದಾಜಿಸುತ್ತದೆ. ಇದು ಮಧುಮೇಹವನ್ನು ನಿರ್ಣಯಿಸುವುದಿಲ್ಲ.",
        "unknown": "ನನ್ನ ಚರ್ಮದ ದಪ್ಪ / ಇನ್ಸುಲಿನ್ ಮೌಲ್ಯ ನನಗೆ ತಿಳಿದಿಲ್ಲ",
        "check": "🔎 ನನ್ನ ಅಪಾಯ ಪರಿಶೀಲಿಸಿ",
        "age": "ವಯಸ್ಸು",
        "preg": "ಗರ್ಭಧಾರಣೆಗಳ ಸಂಖ್ಯೆ",
        "glucose": "ಗ್ಲೂಕೋಸ್ (mg/dL)",
        "bp": "ರಕ್ತದೊತ್ತಡ — ಡಯಾಸ್ಟೋಲಿಕ್ (mmHg)",
        "bmi": "BMI",
        "family": "ಕುಟುಂಬದ ಇತಿಹಾಸ ಸ್ಕೋರ್",
        "skin": "ಚರ್ಮದ ದಪ್ಪ (mm)",
        "insulin": "ಇನ್ಸುಲಿನ್ (μU/ml)",
        "symptom_info": "ಸಂಭಾವ್ಯ ಆರೋಗ್ಯ ಮಾದರಿಗಳನ್ನು ಪರಿಶೀಲಿಸಲು ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ.",
        "select_symptoms": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "check_symptoms": "🩺 ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ",
        "model_info": "📊 ಮಾದರಿ ಮಾಹಿತಿ",
        "threshold": "ಸ್ಕ್ರೀನಿಂಗ್ ಮಿತಿ",
        "recall": "ರಿಕಾಲ್",
        "precision": "ಪ್ರಿಸಿಷನ್",
        "accuracy": "ನಿಖರತೆ",
        "threshold_help": "ಈ ಮಿತಿ ಮಾದರಿಯು ವ್ಯಕ್ತಿಯನ್ನು ಸಂಭವನೀಯ ಅಪಾಯ ಎಂದು ಗುರುತಿಸುವ ಮಟ್ಟವನ್ನು ನಿಯಂತ್ರಿಸುತ್ತದೆ.",
        "evaluation_title": "📊 ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ",
        "evaluation_info": "ಈ ಅಂಕಿಅಂಶಗಳನ್ನು ಪ್ರತ್ಯೇಕವಾಗಿ ಇರಿಸಲಾದ ಪರೀಕ್ಷಾ ಡೇಟಾದಿಂದ ಲೆಕ್ಕಹಾಕಲಾಗಿದೆ. ಇವು ಮಾದರಿಯ ವರ್ತನೆಯನ್ನು ತೋರಿಸುತ್ತವೆ, ವೈದ್ಯಕೀಯ ಕಾರ್ಯಕ್ಷಮತೆಯನ್ನು ಅಲ್ಲ.",
        "threshold_tradeoff": "ಸ್ಕ್ರೀನಿಂಗ್ ಮಿತಿಯ ಪರಿಣಾಮ",
        "confusion_matrix": "ಕನ್ಫ್ಯೂಷನ್ ಮ್ಯಾಟ್ರಿಕ್ಸ್",
        "cost_sensitive": "💰 ವೆಚ್ಚ-ಸೂಕ್ಷ್ಮ ಕಲಿಕೆ",
        "visualizations": "📈 ಪ್ರಯೋಗದ ಗ್ರಾಫ್‌ಗಳು",
        "self_care": "ಸ್ವಯಂ ಆರೈಕೆ",
        "red_flags": "⚠️ ಎಚ್ಚರಿಕೆ ಸೂಚನೆಗಳು",
        "seek_care": "ವೈದ್ಯಕೀಯ ಸಹಾಯ ಯಾವಾಗ ಪಡೆಯಬೇಕು",
        "matched": "ಹೊಂದಿಕೆಯಾದ ಲಕ್ಷಣಗಳು",
        "no_match": "ಯಾವುದೇ ಹೊಂದಿಕೆಯಾದ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
        "disclaimer": "ಇದು ಕೇವಲ ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯವಾಗಿದೆ ಮತ್ತು ರೋಗವನ್ನು ನಿರ್ಣಯಿಸುವುದಿಲ್ಲ.",
        "lower_threshold": "ಕಡಿಮೆ ಮಿತಿಯು ಹೆಚ್ಚು ಸಂಭವನೀಯ ಪ್ರಕರಣಗಳನ್ನು ಹಿಡಿಯಬಹುದು, ಆದರೆ ತಪ್ಪು ಎಚ್ಚರಿಕೆಗಳನ್ನೂ ಹೆಚ್ಚಿಸಬಹುದು.",
        "recall_explanation": "ರಿಕಾಲ್ ಎಂದರೆ: ನಿಜವಾದ ಧನಾತ್ಮಕ ಪ್ರಕರಣಗಳಲ್ಲಿ ಎಷ್ಟು ಪ್ರಕರಣಗಳನ್ನು ನಾವು ಪತ್ತೆಹಚ್ಚಿದ್ದೇವೆ?",
        "precision_explanation": "ಪ್ರಿಸಿಷನ್ ಎಂದರೆ: ನಾವು ಗುರುತಿಸಿದ ಪ್ರಕರಣಗಳಲ್ಲಿ ಎಷ್ಟು ನಿಜವಾಗಿಯೂ ಧನಾತ್ಮಕವಾಗಿದ್ದವು?",
        "tn": "ಸರಿಯಾದ ಋಣಾತ್ಮಕ (TN)",
        "fp": "ತಪ್ಪು ಧನಾತ್ಮಕ (FP)",
        "fn": "ತಪ್ಪು ಋಣಾತ್ಮಕ (FN)",
        "tp": "ಸರಿಯಾದ ಧನಾತ್ಮಕ (TP)",
        "cost_info": "ತಪ್ಪು ಋಣಾತ್ಮಕ ಪ್ರಕರಣಗಳಿಗೆ ಹೆಚ್ಚಿನ ದಂಡ ನೀಡುವುದರಿಂದ ತಪ್ಪಿಹೋದ ಧನಾತ್ಮಕ ಪ್ರಕರಣಗಳ ಬಗ್ಗೆ ಮಾದರಿ ಹೆಚ್ಚು ಸಂವೇದನಾಶೀಲವಾಗುತ್ತದೆ. ಇದರಿಂದ ರಿಕಾಲ್ ಹೆಚ್ಚಬಹುದು ಆದರೆ ಪ್ರಿಸಿಷನ್ ಕಡಿಮೆಯಾಗಬಹುದು.",
        "medical_warning": "⚠️ ಈ ಫಲಿತಾಂಶಗಳು ಶೈಕ್ಷಣಿಕ ಸ್ಕ್ರೀನಿಂಗ್‌ಗಾಗಿ ಮಾತ್ರ. ಇದು ವೈದ್ಯಕೀಯ ರೋಗನಿರ್ಣಯವಲ್ಲ."
    }
}

# =========================================================
# LANGUAGE SELECTION
# =========================================================

st.sidebar.header("🌐 Language")

selected_language = st.sidebar.selectbox(
    "Language / भाषा / ಭಾಷೆ",
    ["en", "hi", "kn"],
    format_func=lambda x: {
        "en": "English",
        "hi": "हिन्दी",
        "kn": "ಕನ್ನಡ"
    }[x]
)

T = TEXT[selected_language]

# =========================================================
# LOAD MODEL
# =========================================================

model_data = {}

if MODEL_PATH.exists():
    try:
        model_data = json.loads(
            MODEL_PATH.read_text(encoding="utf-8")
        )
    except Exception as e:
        st.sidebar.warning(
            f"Could not load model information: {e}"
        )
else:
    st.sidebar.warning(
        f"Model file not found: {MODEL_PATH}"
    )

# =========================================================
# MODEL METRICS
# =========================================================

threshold = float(
    model_data.get("threshold", 0.19)
)

metrics_data = model_data.get(
    "test_metrics_at_threshold",
    {}
)

accuracy = float(
    metrics_data.get("accuracy", 0)
)

precision = float(
    metrics_data.get("precision", 0)
)

recall = float(
    metrics_data.get("recall", 0)
)

# =========================================================
# SIDEBAR MODEL INFORMATION
# =========================================================

st.sidebar.markdown("---")
st.sidebar.header(T["model_info"])

st.sidebar.metric(
    T["threshold"],
    f"{threshold:.2f}"
)

st.sidebar.metric(
    T["recall"],
    f"{recall:.1%}"
)

st.sidebar.metric(
    T["precision"],
    f"{precision:.1%}"
)

st.sidebar.metric(
    T["accuracy"],
    f"{accuracy:.1%}"
)

st.sidebar.caption(T["lower_threshold"])

# =========================================================
# MAIN TITLE
# =========================================================

st.title(T["title"])
st.caption(T["caption"])

# =========================================================
# TABS
# =========================================================

tab_diabetes, tab_symptoms, tab_evaluation = st.tabs(
    [
        T["diabetes"],
        T["symptoms"],
        T["evaluation"]
    ]
)

# =========================================================
# DIABETES RISK TAB
# =========================================================

with tab_diabetes:
    st.header(T["diabetes_title"])
    st.write(T["diabetes_info"])

    c1, c2 = st.columns(2)

    age = c1.number_input(
        T["age"],
        min_value=0,
        max_value=130,
        value=35
    )

    preg = c2.number_input(
        T["preg"],
        min_value=0,
        max_value=30,
        value=0
    )

    glucose = c1.number_input(
        T["glucose"],
        min_value=0.0,
        max_value=600.0,
        value=110.0
    )

    bp = c2.number_input(
        T["bp"],
        min_value=0.0,
        max_value=250.0,
        value=72.0
    )

    bmi = c1.number_input(
        T["bmi"],
        min_value=0.0,
        max_value=100.0,
        value=27.0
    )

    ped = c2.number_input(
        T["family"],
        min_value=0.0,
        max_value=5.0,
        value=0.4
    )

    unknown = st.checkbox(
        T["unknown"],
        value=True
    )

    skin = None
    insulin = None

    if not unknown:
        skin = st.number_input(
            T["skin"],
            min_value=0.0,
            max_value=150.0,
            value=20.0
        )

        insulin = st.number_input(
            T["insulin"],
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

    st.info(
        f"{T['threshold']}: {threshold:.2f} | "
        f"{T['recall']}: {recall:.1%} | "
        f"{T['precision']}: {precision:.1%}"
    )

    if st.button(
        T["check"],
        type="primary",
        key="check_risk"
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
            response = requests.post(
                f"{API}/predict",
                params={"lang": selected_language},
                json=body,
                timeout=10
            )

        except requests.RequestException:
            st.error(
                "Cannot reach the prediction server. "
                "Make sure the API is running."
            )
            st.stop()

        if response.status_code == 422:
            try:
                errors = response.json().get("errors", [])
                for error in errors:
                    st.error(
                        error.get(
                            "message",
                            "Invalid input."
                        )
                    )
            except Exception:
                st.error("The server rejected the input.")

        elif response.status_code != 200:
            try:
                detail = response.json().get(
                    "detail",
                    "Something went wrong."
                )
            except Exception:
                detail = "Something went wrong."
            st.error(detail)

        else:
            data = response.json()
            risk = float(data.get("risk", 0))
            risk_percent = data.get(
                "risk_percent",
                round(risk * 100, 1)
            )

            flagged = risk >= threshold

            if flagged:
                st.warning(
                    data.get(
                        "message",
                        "The screening model flags this result for further attention."
                    )
                )
            else:
                st.success(
                    data.get(
                        "message",
                        "The screening model does not flag this result."
                    )
                )

            st.metric(
                "Estimated screening risk",
                f"{risk_percent}%"
            )

            explanation = data.get("explanation", [])
            if explanation:
                st.subheader("Main factors")
                for line in explanation:
                    st.write(f"- {line}")

            warnings = data.get("warnings", [])
            for warning in warnings:
                st.warning(warning)

            disclaimer = data.get(
                "disclaimer",
                T["medical_warning"]
            )
            st.caption(disclaimer)

# =========================================================
# SYMPTOM CHECKER TAB
# =========================================================

with tab_symptoms:
    st.header(T["symptoms"])
    st.write(T["symptom_info"])

    symptom_options = [
        "cough",
        "runny nose",
        "sore throat",
        "mild headache",
        "fever",
        "fatigue",
        "body ache",
        "congestion",
        "sneezing"
    ]

    selected_symptoms = st.multiselect(
        T["select_symptoms"],
        symptom_options
    )

    if st.button(
        T["check_symptoms"],
        type="primary",
        key="check_symptoms"
    ):
        try:
            condition = load_condition("common_cold.json")
            result = check_symptoms(
                selected_symptoms,
                condition
            )

            st.subheader("Screening result")
            st.info(
                f"Possible pattern: "
                f"{result['condition']} "
                f"(symptom match: "
                f"{result['match_score']:.1f}%)"
            )

            st.subheader(T["matched"])
            matched = result.get("matched_symptoms", [])

            if matched:
                for symptom in matched:
                    st.write(f"- {symptom}")
            else:
                st.write(T["no_match"])

            st.subheader(T["self_care"])
            for item in result.get("self_care", []):
                st.write(f"- {item}")

            st.subheader(T["red_flags"])
            for item in result.get("red_flags", []):
                st.warning(item)

            st.subheader(T["seek_care"])
            for item in result.get("seek_medical_care", []):
                st.write(f"- {item}")

            st.caption(
                result.get(
                    "disclaimer",
                    T["disclaimer"]
                )
            )

        except Exception as e:
            st.error(
                f"Could not run symptom checker: {e}"
            )

# =========================================================
# MODEL EVALUATION TAB
# =========================================================

with tab_evaluation:
    st.header(T["evaluation_title"])
    st.write(T["evaluation_info"])

    st.subheader("Performance at selected threshold")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(T["threshold"], f"{threshold:.2f}")
    m2.metric(T["accuracy"], f"{accuracy:.1%}")
    m3.metric(T["precision"], f"{precision:.1%}")
    m4.metric(T["recall"], f"{recall:.1%}")

    st.info(
        f"{T['recall_explanation']} "
        f"{T['precision_explanation']}"
    )

    # -----------------------------------------------------
    # THRESHOLD TABLE
    # -----------------------------------------------------

    st.subheader(T["threshold_tradeoff"])

    threshold_table = model_data.get("threshold_table", [])

    if threshold_table:
        df_threshold = pd.DataFrame(threshold_table)
        df_threshold["recall"] = df_threshold["recall"] * 100
        df_threshold["precision"] = df_threshold["precision"] * 100

        df_threshold = df_threshold.rename(
            columns={
                "threshold": "Threshold",
                "recall": "Recall (%)",
                "precision": "Precision (%)"
            }
        )

        st.dataframe(
            df_threshold,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Threshold evaluation data is not available.")

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader(T["confusion_matrix"])
    st.write("The confusion matrix shows correct and incorrect screening decisions.")

    cm = model_data.get("confusion_matrix", None)

    if cm is None:
        st.info(
            "The current model.json does not store the confusion-matrix "
            "counts at the selected threshold. Re-run the experiment "
            "after saving the counts to model.json."
        )
    else:
        cm_df = pd.DataFrame(
            [
                [cm.get("tn", 0), cm.get("fp", 0)],
                [cm.get("fn", 0), cm.get("tp", 0)]
            ],
            index=["Predicted Negative", "Predicted Positive"],
            columns=["Actual Negative", "Actual Positive"]
        )

        st.dataframe(cm_df, use_container_width=True)

    # -----------------------------------------------------
    # COST SENSITIVE LEARNING
    # -----------------------------------------------------

    st.subheader(T["cost_sensitive"])
    st.write(T["cost_info"])

    cost_data = {
        "False-negative penalty": [1, 2, 3, 5, 10],
        "Recall": [0.500, 0.741, 0.833, 0.889, 0.963],
        "Precision": [0.600, 0.588, 0.570, 0.533, 0.486],
        "Accuracy": [0.708, 0.727, 0.721, 0.688, 0.630]
    }

    cost_df = pd.DataFrame(cost_data)

    st.dataframe(
        cost_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These values come from the from-scratch cost-sensitive "
        "experiment using false-negative weights of 1, 2, 3, 5 and 10."
    )

    # -----------------------------------------------------
    # ARTIFACT VISUALIZATIONS
    # -----------------------------------------------------

    st.subheader(T["visualizations"])

    threshold_plot = Path("artifacts/threshold_tradeoff.png")
    reliability_plot = Path("artifacts/reliability.png")
    optimizer_plot = Path("artifacts/optimizer_loss.png")

    if threshold_plot.exists():
        st.image(
            str(threshold_plot),
            caption="Threshold vs Precision / Recall",
            use_container_width=True
        )

    if reliability_plot.exists():
        st.image(
            str(reliability_plot),
            caption="Model Calibration / Reliability",
            use_container_width=True
        )

    if optimizer_plot.exists():
        st.image(
            str(optimizer_plot),
            caption="Gradient Descent vs Adam Loss",
            use_container_width=True
        )

    # -----------------------------------------------------
    # WHY EVALUATION
    # -----------------------------------------------------

    st.subheader("🧠 Why we use these evaluation measures")

    st.markdown(
        """
* **Threshold** — controls how sensitive the screening system is.
* **Recall** — measures how many real positive cases were caught.
* **Precision** — measures how many flagged cases were actually positive.
* **Accuracy** — measures overall correct predictions.
* **Confusion matrix** — exposes false positives and false negatives.
* **Cost-sensitive loss** — allows missed positive cases to receive a higher penalty.
* **Calibration** — checks whether predicted risk approximately reflects observed risk.
* **Optimizer comparison** — demonstrates the training process using implemented optimization methods.
        """
    )

    st.warning(T["medical_warning"])
    