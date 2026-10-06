import json
import os
from pathlib import Path

import requests
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.json"

API = os.getenv("API_URL", "http://127.0.0.1:8000")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal Health Risk Check",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LANGUAGE TEXT
# ============================================================

TEXT = {
    "en": {
        "title": "🩺 Personal Health Risk Check",
        "caption": "A screening aid for awareness and early risk identification — not a medical diagnosis.",

        "language": "Language",
        "english": "English",
        "hindi": "हिन्दी",
        "kannada": "ಕನ್ನಡ",

        "diabetes": "🩸 Diabetes Risk",
        "symptoms": "🤒 Symptom Checker",
        "evaluation": "📊 Model Evaluation",

        "diabetes_title": "🩸 Diabetes Risk Screening",
        "diabetes_info": (
            "This section estimates screening risk from health measurements. "
            "It does not diagnose diabetes."
        ),

        "personal": "👤 Personal Information",
        "sex": "Sex",
        "male": "Male",
        "female": "Female",
        "other": "Other / Prefer not to say",

        "age": "Age",
        "preg": "Pregnancies",

        "measurements": "📏 Health Measurements",
        "glucose": "Glucose (mg/dL)",
        "bp": "Blood pressure — diastolic (mmHg)",
        "bmi": "BMI",
        "family": "Family history score",
        "skin": "Skin-fold thickness (mm)",
        "insulin": "Insulin (μU/ml)",

        "bmi_help": "BMI = weight (kg) ÷ height² (m²).",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "calculate_bmi": "Calculate BMI",

        "unknown": "I don't know my skin-fold / insulin values",

        "check": "🔎 Check My Risk",

        "symptom_info": (
            "Select your symptoms to screen for possible common symptom patterns. "
            "This is not a diagnosis."
        ),
        "select_symptoms": "Select your symptoms",
        "check_symptoms": "🩺 Check Symptoms",

        "model_info": "📊 Model Information",
        "threshold": "Screening threshold",
        "threshold_help": (
            "The threshold is the probability level at which the system changes "
            "from lower screening risk to flag for further assessment."
        ),
        "recall": "Recall",
        "precision": "Precision",
        "accuracy": "Accuracy",

        "performance": "Performance at selected threshold",
        "confusion": "Confusion Matrix",
        "tradeoff": "Threshold Trade-off",
        "cost": "💰 Cost-Sensitive Learning",
        "calibration": "Model Calibration / Reliability",
        "optimizer": "Optimizer Comparison",

        "low_risk": "Lower screening risk",
        "flagged": "Flagged for further assessment",

        "no_api": "Cannot reach the prediction server.",
        "api_hint": "Start the API in another terminal with:",

        "disclaimer": (
            "This is a screening aid and does not diagnose disease or replace "
            "professional medical advice."
        ),

        "bmi_under": "Underweight range",
        "bmi_normal": "Healthy-weight range",
        "bmi_over": "Overweight range",
        "bmi_obese": "Higher BMI range",
    },

    "hi": {
        "title": "🩺 व्यक्तिगत स्वास्थ्य जोखिम जाँच",
        "caption": "यह जागरूकता और प्रारंभिक जोखिम पहचान के लिए स्क्रीनिंग सहायता है — चिकित्सा निदान नहीं।",

        "language": "भाषा",
        "english": "English",
        "hindi": "हिन्दी",
        "kannada": "ಕನ್ನಡ",

        "diabetes": "🩸 मधुमेह जोखिम",
        "symptoms": "🤒 लक्षण जाँच",
        "evaluation": "📊 मॉडल मूल्यांकन",

        "diabetes_title": "🩸 मधुमेह जोखिम स्क्रीनिंग",
        "diabetes_info": "यह अनुभाग स्वास्थ्य मापदंडों के आधार पर प्रारंभिक स्क्रीनिंग जोखिम का अनुमान लगाता है। यह मधुमेह का निदान नहीं करता।",

        "personal": "👤 व्यक्तिगत जानकारी",
        "sex": "लिंग",
        "male": "पुरुष",
        "female": "महिला",
        "other": "अन्य / बताना नहीं चाहते",

        "age": "आयु",
        "preg": "गर्भधारण की संख्या",

        "measurements": "📏 स्वास्थ्य माप",
        "glucose": "ग्लूकोज़ (mg/dL)",
        "bp": "रक्तचाप — डायस्टोलिक (mmHg)",
        "bmi": "BMI",
        "family": "पारिवारिक इतिहास स्कोर",
        "skin": "त्वचा की मोटाई (mm)",
        "insulin": "इंसुलिन (μU/ml)",

        "bmi_help": "BMI = वजन (kg) ÷ ऊँचाई² (m²)।",
        "height": "ऊँचाई (cm)",
        "weight": "वजन (kg)",
        "calculate_bmi": "BMI की गणना करें",

        "unknown": "मुझे त्वचा की मोटाई / इंसुलिन का मान नहीं पता",

        "check": "🔎 मेरा जोखिम जाँचें",

        "symptom_info": "संभावित सामान्य स्वास्थ्य पैटर्न की जाँच के लिए अपने लक्षण चुनें। यह निदान नहीं है।",
        "select_symptoms": "अपने लक्षण चुनें",
        "check_symptoms": "🩺 लक्षण जाँचें",

        "model_info": "📊 मॉडल जानकारी",
        "threshold": "स्क्रीनिंग सीमा",
        "threshold_help": "यह वह संभाव्यता स्तर है जिस पर सिस्टम कम स्क्रीनिंग जोखिम से आगे मूल्यांकन के लिए फ्लैग करने की स्थिति में जाता है।",
        "recall": "रिकॉल",
        "precision": "प्रिसीजन",
        "accuracy": "सटीकता",

        "performance": "चयनित सीमा पर प्रदर्शन",
        "confusion": "कन्फ्यूजन मैट्रिक्स",
        "tradeoff": "थ्रेशोल्ड ट्रेड-ऑफ",
        "cost": "💰 लागत-संवेदी लर्निंग",
        "calibration": "मॉडल कैलिब्रेशन / विश्वसनीयता",
        "optimizer": "ऑप्टिमाइज़र तुलना",

        "low_risk": "कम स्क्रीनिंग जोखिम",
        "flagged": "आगे के मूल्यांकन के लिए फ्लैग किया गया",

        "no_api": "प्रेडिक्शन सर्वर तक पहुँचा नहीं जा सकता।",
        "api_hint": "दूसरे टर्मिनल में API शुरू करें:",

        "disclaimer": "यह स्क्रीनिंग सहायता है और बीमारी का निदान नहीं करती तथा पेशेवर चिकित्सा सलाह का विकल्प नहीं है।",

        "bmi_under": "कम BMI श्रेणी",
        "bmi_normal": "स्वस्थ BMI श्रेणी",
        "bmi_over": "अधिक BMI श्रेणी",
        "bmi_obese": "उच्च BMI श्रेणी",
    },

    "kn": {
        "title": "🩺 ವೈಯಕ್ತಿಕ ಆರೋಗ್ಯ ಅಪಾಯ ಪರಿಶೀಲನೆ",
        "caption": "ಇದು ಅರಿವು ಮತ್ತು ಆರಂಭಿಕ ಅಪಾಯ ಗುರುತಿಸುವಿಕೆಗಾಗಿ ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯ ಮಾತ್ರ — ವೈದ್ಯಕೀಯ ರೋಗನಿರ್ಣಯವಲ್ಲ.",

        "language": "ಭಾಷೆ",
        "english": "English",
        "hindi": "हिन्दी",
        "kannada": "ಕನ್ನಡ",

        "diabetes": "🩸 ಮಧುಮೇಹ ಅಪಾಯ",
        "symptoms": "🤒 ಲಕ್ಷಣ ಪರಿಶೀಲನೆ",
        "evaluation": "📊 ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ",

        "diabetes_title": "🩸 ಮಧುಮೇಹ ಅಪಾಯ ಸ್ಕ್ರೀನಿಂಗ್",
        "diabetes_info": "ಈ ವಿಭಾಗವು ಆರೋಗ್ಯದ ಅಳತೆಗಳ ಆಧಾರದ ಮೇಲೆ ಆರಂಭಿಕ ಸ್ಕ್ರೀನಿಂಗ್ ಅಪಾಯವನ್ನು ಅಂದಾಜಿಸುತ್ತದೆ. ಇದು ಮಧುಮೇಹವನ್ನು ನಿರ್ಣಯಿಸುವುದಿಲ್ಲ.",

        "personal": "👤 ವೈಯಕ್ತಿಕ ಮಾಹಿತಿ",
        "sex": "ಲಿಂಗ",
        "male": "ಪುರುಷ",
        "female": "ಮಹಿಳೆ",
        "other": "ಇತರೆ / ಹೇಳಲು ಇಷ್ಟವಿಲ್ಲ",

        "age": "ವಯಸ್ಸು",
        "preg": "ಗರ್ಭಧಾರಣೆಗಳ ಸಂಖ್ಯೆ",

        "measurements": "📏 ಆರೋಗ್ಯದ ಅಳತೆಗಳು",
        "glucose": "ಗ್ಲೂಕೋಸ್ (mg/dL)",
        "bp": "ರಕ್ತದೊತ್ತಡ — ಡಯಾಸ್ಟೋಲಿಕ್ (mmHg)",
        "bmi": "BMI",
        "family": "ಕುಟುಂಬದ ಇತಿಹಾಸ ಸ್ಕೋರ್",
        "skin": "ಚರ್ಮದ ದಪ್ಪ (mm)",
        "insulin": "ಇನ್ಸುಲಿನ್ (μU/ml)",

        "bmi_help": "BMI = ತೂಕ (kg) ÷ ಎತ್ತರ² (m²).",
        "height": "ಎತ್ತರ (cm)",
        "weight": "ತೂಕ (kg)",
        "calculate_bmi": "BMI ಲೆಕ್ಕ ಹಾಕಿ",

        "unknown": "ನನ್ನ ಚರ್ಮದ ದಪ್ಪ / ಇನ್ಸುಲಿನ್ ಮೌಲ್ಯ ನನಗೆ ತಿಳಿದಿಲ್ಲ",

        "check": "🔎 ನನ್ನ ಅಪಾಯ ಪರಿಶೀಲಿಸಿ",

        "symptom_info": "ಸಂಭಾವ್ಯ ಸಾಮಾನ್ಯ ಆರೋಗ್ಯ ಮಾದರಿಗಳನ್ನು ಪರಿಶೀಲಿಸಲು ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ. ಇದು ರೋಗನಿರ್ಣಯವಲ್ಲ.",
        "select_symptoms": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "check_symptoms": "🩺 ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಿ",

        "model_info": "📊 ಮಾದರಿ ಮಾಹಿತಿ",
        "threshold": "ಸ್ಕ್ರೀನಿಂಗ್ ಮಿತಿ",
        "threshold_help": "ಈ ಮಿತಿಯು ಸಿಸ್ಟಮ್ ಕಡಿಮೆ ಸ್ಕ್ರೀನಿಂಗ್ ಅಪಾಯದಿಂದ ಮುಂದಿನ ಮೌಲ್ಯಮಾಪನಕ್ಕೆ ಫ್ಲ್ಯಾಗ್ ಮಾಡುವ ಹಂತವನ್ನು ಸೂಚಿಸುತ್ತದೆ.",
        "recall": "ರಿಕಾಲ್",
        "precision": "ಪ್ರಿಸಿಷನ್",
        "accuracy": "ನಿಖರತೆ",

        "performance": "ಆಯ್ಕೆ ಮಾಡಿದ ಮಿತಿಯಲ್ಲಿನ ಕಾರ್ಯಕ್ಷಮತೆ",
        "confusion": "ಕನ್ಫ್ಯೂಷನ್ ಮ್ಯಾಟ್ರಿಕ್ಸ್",
        "tradeoff": "ಥ್ರೇಶೋಲ್ಡ್ ಟ್ರೇಡ್-ಆಫ್",
        "cost": "💰 ವೆಚ್ಚ-ಸಂವೇದಿ ಲರ್ನಿಂಗ್",
        "calibration": "ಮಾದರಿ ಕ್ಯಾಲಿಬ್ರೇಷನ್ / ವಿಶ್ವಾಸಾರ್ಹತೆ",
        "optimizer": "ಆಪ್ಟಿಮೈಸರ್ ಹೋಲಿಕೆ",

        "low_risk": "ಕಡಿಮೆ ಸ್ಕ್ರೀನಿಂಗ್ ಅಪಾಯ",
        "flagged": "ಮುಂದಿನ ಮೌಲ್ಯಮಾಪನಕ್ಕಾಗಿ ಫ್ಲ್ಯಾಗ್ ಮಾಡಲಾಗಿದೆ",

        "no_api": "ಪ್ರಿಡಿಕ್ಷನ್ ಸರ್ವರ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
        "api_hint": "ಇನ್ನೊಂದು ಟರ್ಮಿನಲ್‌ನಲ್ಲಿ API ಪ್ರಾರಂಭಿಸಿ:",

        "disclaimer": "ಇದು ಸ್ಕ್ರೀನಿಂಗ್ ಸಹಾಯ ಮಾತ್ರ. ಇದು ರೋಗವನ್ನು ನಿರ್ಣಯಿಸುವುದಿಲ್ಲ ಅಥವಾ ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ.",

        "bmi_under": "ಕಡಿಮೆ BMI ಶ್ರೇಣಿ",
        "bmi_normal": "ಆರೋಗ್ಯಕರ BMI ಶ್ರೇಣಿ",
        "bmi_over": "ಹೆಚ್ಚಿನ BMI ಶ್ರೇಣಿ",
        "bmi_obese": "ಅತಿ ಹೆಚ್ಚಿನ BMI ಶ್ರೇಣಿ",
    },
}


# ============================================================
# LANGUAGE SELECTION
# ============================================================

language_names = {
    "English": "en",
    "हिन्दी": "hi",
    "ಕನ್ನಡ": "kn",
}

selected_language = st.sidebar.selectbox(
    "🌐 Language / भाषा / ಭಾಷೆ",
    list(language_names.keys()),
)

lang = language_names[selected_language]
T = TEXT[lang]


# ============================================================
# LOAD MODEL
# ============================================================

model_data = {}

if MODEL_PATH.exists():
    try:
        model_data = json.loads(
            MODEL_PATH.read_text(encoding="utf-8")
        )
    except Exception as e:
        st.sidebar.warning(f"Could not load model.json: {e}")
else:
    st.sidebar.warning(
        "artifacts/model.json was not found."
    )


# ============================================================
# MODEL METRICS
# ============================================================

threshold = float(model_data.get("threshold", 0.19))

metrics_data = model_data.get(
    "test_metrics_at_threshold",
    {}
)

accuracy = float(metrics_data.get("accuracy", 0))
precision = float(metrics_data.get("precision", 0))
recall = float(metrics_data.get("recall", 0))


# ============================================================
# SIDEBAR MODEL INFORMATION
# ============================================================

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

st.sidebar.info(T["threshold_help"])

st.sidebar.caption(
    "These are experimental test-data metrics, not clinical performance claims."
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title(T["title"])
st.caption(T["caption"])


# ============================================================
# TABS
# ============================================================

tab_diabetes, tab_symptoms, tab_evaluation = st.tabs(
    [
        T["diabetes"],
        T["symptoms"],
        T["evaluation"],
    ]
)


# ============================================================
# DIABETES TAB
# ============================================================

with tab_diabetes:

    st.header(T["diabetes_title"])
    st.write(T["diabetes_info"])

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader(T["personal"])

    col1, col2, col3 = st.columns(3)

    with col1:
        sex_options = [
            T["male"],
            T["female"],
            T["other"],
        ]

        sex_display = st.selectbox(
            T["sex"],
            sex_options,
        )

    with col2:
        age = st.number_input(
            T["age"],
            min_value=1,
            max_value=120,
            value=35,
            step=1,
        )

    with col3:
        preg = st.number_input(
            T["preg"],
            min_value=0,
            max_value=30,
            value=0,
            step=1,
        )

    # --------------------------------------------------------
    # HEALTH MEASUREMENTS
    # --------------------------------------------------------

    st.subheader(T["measurements"])

    col1, col2, col3 = st.columns(3)

    with col1:
        glucose = st.number_input(
            T["glucose"],
            min_value=0.0,
            max_value=600.0,
            value=110.0,
            step=1.0,
        )

    with col2:
        bp = st.number_input(
            T["bp"],
            min_value=0.0,
            max_value=250.0,
            value=72.0,
            step=1.0,
        )

    with col3:
        ped = st.number_input(
            T["family"],
            min_value=0.0,
            max_value=5.0,
            value=0.4,
            step=0.1,
        )

    # --------------------------------------------------------
    # BMI CALCULATOR
    # --------------------------------------------------------

    st.subheader("⚖️ BMI Calculator")

    st.info(T["bmi_help"])

    bmi_col1, bmi_col2, bmi_col3 = st.columns(3)

    with bmi_col1:
        height_cm = st.number_input(
            T["height"],
            min_value=50.0,
            max_value=250.0,
            value=165.0,
            step=1.0,
        )

    with bmi_col2:
        weight_kg = st.number_input(
            T["weight"],
            min_value=10.0,
            max_value=300.0,
            value=65.0,
            step=1.0,
        )

    with bmi_col3:
        calculated_bmi = weight_kg / (
            (height_cm / 100.0) ** 2
        )

        st.metric(
            "Calculated BMI",
            f"{calculated_bmi:.1f}",
        )

    bmi = calculated_bmi

    if bmi < 18.5:
        st.caption(T["bmi_under"])
    elif bmi < 25:
        st.caption(T["bmi_normal"])
    elif bmi < 30:
        st.caption(T["bmi_over"])
    else:
        st.caption(T["bmi_obese"])

    # --------------------------------------------------------
    # OPTIONAL VALUES
    # --------------------------------------------------------

    st.subheader("🧪 Optional measurements")

    unknown = st.checkbox(
        T["unknown"],
        value=True,
    )

    skin = None
    insulin = None

    if not unknown:

        col1, col2 = st.columns(2)

        with col1:
            skin = st.number_input(
                T["skin"],
                min_value=0.0,
                max_value=150.0,
                value=20.0,
                step=1.0,
            )

        with col2:
            insulin = st.number_input(
                T["insulin"],
                min_value=0.0,
                max_value=1000.0,
                value=80.0,
                step=1.0,
            )

    st.divider()

    # --------------------------------------------------------
    # CHECK RISK
    # --------------------------------------------------------

    if st.button(
        T["check"],
        type="primary",
        use_container_width=True,
    ):

        body = {
            "pregnancies": preg,
            "glucose": glucose,
            "blood_pressure": bp,
            "skin_thickness": skin,
            "insulin": insulin,
            "bmi": bmi,
            "pedigree": ped,
            "age": age,
        }

        try:

            response = requests.post(
                f"{API}/predict",
                params={"lang": lang},
                json=body,
                timeout=10,
            )

        except requests.RequestException:

            st.error(T["no_api"])

            st.code(
                "python -m uvicorn api.main:app --reload"
            )

            st.stop()

        # ----------------------------------------------------
        # API VALIDATION ERROR
        # ----------------------------------------------------

        if response.status_code == 422:

            try:
                errors = response.json().get(
                    "errors",
                    []
                )

                for error in errors:
                    st.error(
                        error.get(
                            "message",
                            "Invalid input.",
                        )
                    )

            except Exception:
                st.error(
                    "The server rejected the supplied values."
                )

        # ----------------------------------------------------
        # OTHER SERVER ERROR
        # ----------------------------------------------------

        elif response.status_code != 200:

            try:
                detail = response.json().get(
                    "detail",
                    "Something went wrong.",
                )
            except Exception:
                detail = "Something went wrong."

            st.error(detail)

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        else:

            data = response.json()

            risk = float(
                data.get(
                    "risk",
                    data.get("risk_probability", 0),
                )
            )

            risk_percent = float(
                data.get(
                    "risk_percent",
                    risk * 100,
                )
            )

            flagged = risk >= threshold

            st.subheader("Screening Result")

            if flagged:
                st.warning(
                    f"⚠️ {T['flagged']}"
                )
            else:
                st.success(
                    f"✅ {T['low_risk']}"
                )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Estimated screening risk",
                    f"{risk_percent:.1f}%",
                )

            with result_col2:
                st.metric(
                    T["threshold"],
                    f"{threshold:.2f}",
                )

            if "message" in data:
                st.write(data["message"])

            if data.get("explanation"):

                st.subheader("Main factors")

                for line in data["explanation"]:
                    st.write(f"- {line}")

            if data.get("warnings"):

                for warning in data["warnings"]:
                    st.warning(warning)

            st.caption(
                data.get(
                    "disclaimer",
                    T["disclaimer"],
                )
            )


# ============================================================
# SYMPTOM TAB
# ============================================================

with tab_symptoms:

    st.header(T["symptoms"])
    st.write(T["symptom_info"])

    symptoms = [
        "cough",
        "runny nose",
        "sore throat",
        "mild headache",
        "fever",
        "fatigue",
        "body aches",
        "sneezing",
        "congestion",
    ]

    symptom_labels = {
        "cough": "Cough",
        "runny nose": "Runny nose",
        "sore throat": "Sore throat",
        "mild headache": "Mild headache",
        "fever": "Fever",
        "fatigue": "Fatigue",
        "body aches": "Body aches",
        "sneezing": "Sneezing",
        "congestion": "Nasal congestion",
    }

    selected_symptoms = st.multiselect(
        T["select_symptoms"],
        symptoms,
        format_func=lambda x: symptom_labels.get(
            x,
            x.title(),
        ),
    )

    if st.button(
        T["check_symptoms"],
        type="primary",
        use_container_width=True,
    ):

        # Common-cold knowledge-base symptoms.
        # This intentionally remains a simple knowledge-base
        # screening component, not a disease classifier.

        cold_symptoms = {
            "cough",
            "runny nose",
            "sore throat",
            "mild headache",
            "fever",
            "fatigue",
            "sneezing",
            "congestion",
            "body aches",
        }

        matched = sorted(
            set(selected_symptoms).intersection(
                cold_symptoms
            )
        )

        match_score = (
            len(matched)
            / len(cold_symptoms)
            * 100
        )

        st.subheader("Screening Result")

        st.info(
            f"Possible common-symptom pattern "
            f"(knowledge-base match: {match_score:.1f}%)."
        )

        st.write("**Matched symptoms**")

        if matched:
            for symptom in matched:
                st.write(
                    f"- {symptom_labels.get(symptom, symptom)}"
                )
        else:
            st.write(
                "No matching symptoms found."
            )

        st.write("**Self-care information**")

        st.write("- Get adequate rest.")
        st.write("- Drink adequate fluids.")
        st.write("- Monitor your symptoms.")

        st.write("**⚠️ Red flags**")

        red_flags = [
            "difficulty breathing",
            "dehydration",
            "symptoms that are getting worse",
            "symptoms that improve and then become worse",
        ]

        for item in red_flags:
            st.warning(item)

        st.write("**When to seek medical care**")

        seek = [
            "Symptoms are severe or worsening.",
            "You develop difficulty breathing.",
            "You become dehydrated.",
            "Symptoms persist without improvement.",
        ]

        for item in seek:
            st.write(f"- {item}")

        st.caption(T["disclaimer"])


# ============================================================
# MODEL EVALUATION TAB
# ============================================================

with tab_evaluation:

    st.header(T["evaluation"])

    st.write(
        "These results come from the held-out test data used "
        "during model development. They describe model behaviour, "
        "not clinical performance."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(T["performance"])

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            T["threshold"],
            f"{threshold:.2f}",
        )

    with m2:
        st.metric(
            T["accuracy"],
            f"{accuracy:.1%}",
        )

    with m3:
        st.metric(
            T["precision"],
            f"{precision:.1%}",
        )

    with m4:
        st.metric(
            T["recall"],
            f"{recall:.1%}",
        )

    st.info(
        "Recall answers: Of the real positive cases, how many "
        "did the model catch?\n\n"
        "Precision answers: Of the cases flagged by the model, "
        "how many were actually positive?"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(T["confusion"])

    confusion = model_data.get(
        "confusion_matrix",
        None,
    )

    if confusion:

        tn = int(confusion.get("tn", 0))
        fp = int(confusion.get("fp", 0))
        fn = int(confusion.get("fn", 0))
        tp = int(confusion.get("tp", 0))

        matrix_data = {
            "Actual Negative": [
                f"TN = {tn}",
                f"FP = {fp}",
            ],
            "Actual Positive": [
                f"FN = {fn}",
                f"TP = {tp}",
            ],
        }

        st.table(
            {
                "": [
                    "Predicted Negative",
                    "Predicted Positive",
                ],
                "Actual Negative": [
                    f"TN = {tn}",
                    f"FP = {fp}",
                ],
                "Actual Positive": [
                    f"FN = {fn}",
                    f"TP = {tp}",
                ],
            }
        )

    else:

        st.warning(
            "Confusion-matrix counts are not currently stored "
            "in model.json."
        )

        st.write(
            "Your experiment script calculates the confusion "
            "matrix, but the current model artifact only stores "
            "accuracy, precision and recall."
        )

        st.code(
            'print("confusion matrix:", cm)'
        )

        st.info(
            "To display exact TN, FP, FN and TP values here, "
            "save the confusion matrix into artifacts/model.json "
            "from risk_model/run_experiments.py."
        )

    # --------------------------------------------------------
    # THRESHOLD TABLE
    # --------------------------------------------------------

    st.subheader(T["tradeoff"])

    threshold_table = model_data.get(
        "threshold_table",
        [],
    )

    if threshold_table:

        st.dataframe(
            threshold_table,
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            "The threshold controls how easily the system flags "
            "a possible positive case. Lower thresholds generally "
            "increase recall but can create more false alarms."
        )

    else:
        st.warning(
            "Threshold trade-off data is not available."
        )

    # --------------------------------------------------------
    # THRESHOLD PLOT
    # --------------------------------------------------------

    threshold_plot = (
        BASE_DIR
        / "artifacts"
        / "threshold_tradeoff.png"
    )

    if threshold_plot.exists():

        st.image(
            str(threshold_plot),
            caption="Threshold vs Precision / Recall",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # COST-SENSITIVE LEARNING
    # --------------------------------------------------------

    st.subheader(T["cost"])

    st.write(
        "The from-scratch logistic regression implementation "
        "allows a higher penalty to be assigned to false negatives. "
        "This demonstrates how the model can prioritize catching "
        "more possible positive cases."
    )

    cost_results = [
        {
            "FN weight": 1,
            "Recall": 0.500,
            "Precision": 0.600,
            "Accuracy": 0.708,
        },
        {
            "FN weight": 2,
            "Recall": 0.741,
            "Precision": 0.588,
            "Accuracy": 0.727,
        },
        {
            "FN weight": 3,
            "Recall": 0.833,
            "Precision": 0.570,
            "Accuracy": 0.721,
        },
        {
            "FN weight": 5,
            "Recall": 0.889,
            "Precision": 0.533,
            "Accuracy": 0.688,
        },
        {
            "FN weight": 10,
            "Recall": 0.963,
            "Precision": 0.486,
            "Accuracy": 0.630,
        },
    ]

    st.dataframe(
        cost_results,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Increasing the false-negative penalty from 1 to 10 "
        "increases recall substantially, but precision and overall "
        "accuracy decrease. This demonstrates the screening trade-off."
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    st.subheader(T["calibration"])

    reliability_plot = (
        BASE_DIR
        / "artifacts"
        / "reliability.png"
    )

    if reliability_plot.exists():

        st.image(
            str(reliability_plot),
            caption="Model Calibration / Reliability",
            use_container_width=True,
        )

    else:

        st.warning(
            "Reliability plot was not found."
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    st.subheader(T["optimizer"])

    optimizer_plot = (
        BASE_DIR
        / "artifacts"
        / "optimizer_loss.png"
    )

    if optimizer_plot.exists():

        st.image(
            str(optimizer_plot),
            caption="Gradient Descent vs Adam Loss",
            use_container_width=True,
        )

    else:

        st.warning(
            "Optimizer comparison plot was not found."
        )

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.subheader("🧠 Model Implementation")

    st.write(
        "This project implements a NumPy logistic regression model "
        "from scratch and compares its behaviour with standard "
        "machine-learning approaches during experimentation."
    )

    features = model_data.get(
        "features",
        [],
    )

    weights = model_data.get(
        "w",
        [],
    )

    if features and weights:

        st.write("**Learned feature weights**")

        feature_rows = []

        for feature, weight in zip(
            features,
            weights,
        ):

            feature_rows.append(
                {
                    "Feature": feature,
                    "Weight": round(
                        float(weight),
                        4,
                    ),
                }
            )

        st.dataframe(
            feature_rows,
            use_container_width=True,
            hide_index=True,
        )

    st.write(
        "The standardized feature weights show the direction "
        "and relative contribution learned by the logistic model."
    )

    # --------------------------------------------------------
    # WHY THESE METRICS
    # --------------------------------------------------------

    st.subheader("🧠 Why these metrics matter")

    st.markdown(
        """
- **Threshold** — controls how sensitive the screening system is.
- **Recall** — measures how many actual positive cases are caught.
- **Precision** — measures how many flagged cases are actually positive.
- **Accuracy** — measures overall correct predictions.
- **Confusion matrix** — exposes true positives, true negatives, false positives and false negatives.
- **Cost-sensitive learning** — explicitly increases the penalty for missed positive cases.
- **Calibration** — checks whether predicted probabilities approximately match observed outcomes.
- **Optimizer comparison** — demonstrates the training process rather than treating the model as a black box.
"""
    )

    st.warning(
        "These are experimental model-evaluation results, "
        "not clinical performance claims."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    T["disclaimer"]
)