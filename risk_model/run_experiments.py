import json
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_PATH = Path("artifacts/model.json")

with MODEL_PATH.open("r", encoding="utf-8") as file:
    artifact = json.load(file)

confusion = artifact.get("confusion_matrix")

st.subheader("Confusion Matrix")

if confusion is not None:
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    tp = int(confusion["tp"])

    confusion_dataframe = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        columns=[
            "Predicted No Risk",
            "Predicted Risk",
        ],
        index=[
            "Actual No Risk",
            "Actual Risk",
        ],
    )

    st.dataframe(
        confusion_dataframe,
        use_container_width=True,
    )

else:
    st.error(
        "Confusion matrix was not found in artifacts/model.json. "
        "Run: python -m risk_model.run_experiments"
    )
