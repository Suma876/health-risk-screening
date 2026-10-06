"""Pima Indians Diabetes: load, clean, split, scale (no leakage)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
       "pima-indians-diabetes.data.csv")
FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "pedigree", "age"]
# In this dataset a 0 here means "not measured", not a real value.
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def prepare(seed=42, test_size=0.2, path=None):
    df = pd.read_csv(path or URL, header=None, names=FEATURES + ["outcome"])
    df = df.drop_duplicates()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    X, y = df[FEATURES], df["outcome"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    # Fit imputation + scaling on TRAIN only, then apply to test.
    medians = X_tr.median()
    X_tr, X_te = X_tr.fillna(medians), X_te.fillna(medians)
    mean, std = X_tr.mean(), X_tr.std(ddof=0)
    return {
        "X_train": ((X_tr - mean) / std).to_numpy(),
        "X_test": ((X_te - mean) / std).to_numpy(),
        "y_train": y_tr, "y_test": y_te,
        "features": FEATURES,
        "medians": medians.to_dict(), "mean": mean.to_dict(), "std": std.to_dict(),
    }
