"""
Question A - all three levels.

Run:
    python -m risk_model.run_experiments

Optional dataset path:
    python -m risk_model.run_experiments path/to/data.csv
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .data import prepare
from .scratch import (
    ScratchLogReg,
    brier_score,
    confusion_matrix,
    metrics,
    reliability_table,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# Level 3 prediction written before running the experiment
PREDICTION = (
    "Precision will drop when I push recall to 0.9, because a lower "
    "threshold lets in many more false alarms."
)


# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------

D = prepare(
    seed=SEED,
    path=sys.argv[1] if len(sys.argv) > 1 else None,
)

Xtr = D["X_train"]
Xte = D["X_test"]
ytr = D["y_train"]
yte = D["y_test"]

print(f"Train {Xtr.shape}, test {Xte.shape}, positive rate {ytr.mean():.2f}\n")


# =========================================================
# LEVEL 1: Scikit-learn Logistic Regression and Random Forest
# =========================================================

print("=== LEVEL 1: sklearn models ===")

sk_lr = LogisticRegression(
    C=1e6,
    max_iter=2000,
)

sk_lr.fit(Xtr, ytr)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
)

rf.fit(Xtr, ytr)

for name, fitted_model in [
    ("LogisticRegression", sk_lr),
    ("RandomForest", rf),
]:
    predictions = fitted_model.predict(Xte)

    print(
        f"{name:20s} "
        f"acc={accuracy_score(yte, predictions):.3f} "
        f"prec={precision_score(yte, predictions, zero_division=0):.3f} "
        f"rec={recall_score(yte, predictions, zero_division=0):.3f}"
    )


# =========================================================
# LEVEL 2: Logistic Regression from scratch
# =========================================================

print("\n=== LEVEL 2: NumPy logistic regression ===")

model = ScratchLogReg(
    lr=0.1,
    epochs=3000,
    fn_weight=1.0,
).fit(Xtr, ytr)

scratch_predictions = model.predict(Xte)

scratch_cm = confusion_matrix(
    yte,
    scratch_predictions,
)

scratch_metrics = metrics(scratch_cm)

print("confusion matrix:", scratch_cm)

print(
    f"scratch  "
    f"acc={scratch_metrics['accuracy']:.3f} "
    f"prec={scratch_metrics['precision']:.3f} "
    f"rec={scratch_metrics['recall']:.3f}"
)

print(
    f"sklearn  "
    f"acc={accuracy_score(yte, sk_lr.predict(Xte)):.3f}"
)


# ---------------------------------------------------------
# Top features by absolute weight
# ---------------------------------------------------------

features = np.array(D["features"])

print("\nTop-3 features by |weight| (standardized inputs):")

for tag, weights in [
    ("scratch", model.w),
    ("sklearn", sk_lr.coef_[0]),
]:
    indices = np.argsort(-np.abs(weights))[:3]

    feature_text = ", ".join(
        f"{features[index]}={weights[index]:+.3f}"
        for index in indices
    )

    print(f"  {tag}: {feature_text}")


# ---------------------------------------------------------
# Optimizer comparison: gradient descent vs Adam
# ---------------------------------------------------------

adam = ScratchLogReg(
    lr=0.05,
    epochs=3000,
    optimizer="adam",
).fit(Xtr, ytr)

plt.figure(figsize=(6, 4))
plt.plot(model.loss_history, label="gradient descent")
plt.plot(adam.loss_history, label="adam")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "optimizer_loss.png", dpi=120)
plt.close()


# ---------------------------------------------------------
# Cost-sensitive sweep
# ---------------------------------------------------------

print(
    "\nCost-sensitive sweep "
    "(threshold 0.5): fn_weight -> recall / precision"
)

for weight in [1, 2, 3, 5, 10]:
    weighted_model = ScratchLogReg(
        lr=0.1,
        epochs=3000,
        fn_weight=weight,
    ).fit(Xtr, ytr)

    weighted_predictions = weighted_model.predict(Xte)

    weighted_cm = confusion_matrix(
        yte,
        weighted_predictions,
    )

    weighted_metrics = metrics(weighted_cm)

    print(
        f"  k={weight:<2d} "
        f"recall={weighted_metrics['recall']:.3f} "
        f"precision={weighted_metrics['precision']:.3f} "
        f"acc={weighted_metrics['accuracy']:.3f}"
    )


# =========================================================
# Calibration of the standard scratch model
# =========================================================

p_te = model.predict_proba(Xte)

print(f"\nBrier score: {brier_score(yte, p_te):.3f}")

print("Reliability (predicted vs actual positive rate, n):")

rel = reliability_table(yte, p_te)

for predicted_rate, actual_rate, count in rel:
    print(
        f"  predicted {predicted_rate:.2f} "
        f"actual {actual_rate:.2f} "
        f"(n={count})"
    )

plt.figure(figsize=(4.5, 4.5))
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.plot(
    [row[0] for row in rel],
    [row[1] for row in rel],
    "o-",
)
plt.xlabel("predicted risk")
plt.ylabel("actual rate")
plt.tight_layout()
plt.savefig(OUT / "reliability.png", dpi=120)
plt.close()


# =========================================================
# LEVEL 3: Select threshold to achieve recall >= 0.9
# =========================================================

print("\n=== LEVEL 3: lower threshold until recall >= 0.9 ===")

print(
    "PREDICTION written before running:",
    PREDICTION,
)

p_tr = model.predict_proba(Xtr)

chosen = 0.5

# Select the threshold using training data only
for threshold in np.arange(0.5, 0.0, -0.01):
    train_predictions_at_threshold = (
        p_tr >= threshold
    ).astype(int)

    train_cm_at_threshold = confusion_matrix(
        ytr,
        train_predictions_at_threshold,
    )

    train_metrics_at_threshold = metrics(
        train_cm_at_threshold
    )

    if train_metrics_at_threshold["recall"] >= 0.9:
        chosen = float(round(threshold, 2))
        break


# Metrics on the test set at threshold 0.5
base_predictions = (p_te >= 0.5).astype(int)

base_cm = confusion_matrix(
    yte,
    base_predictions,
)

base_metrics = metrics(base_cm)


# Metrics on the test set at selected threshold
low_threshold_predictions = (
    p_te >= chosen
).astype(int)

low_threshold_cm = confusion_matrix(
    yte,
    low_threshold_predictions,
)

low_threshold_metrics = metrics(
    low_threshold_cm
)

print(
    f"threshold 0.50 : "
    f"precision={base_metrics['precision']:.3f} "
    f"recall={base_metrics['recall']:.3f} "
    f"acc={base_metrics['accuracy']:.3f}"
)

print(
    f"threshold {chosen:.2f} : "
    f"precision={low_threshold_metrics['precision']:.3f} "
    f"recall={low_threshold_metrics['recall']:.3f} "
    f"acc={low_threshold_metrics['accuracy']:.3f}"
)

print(
    "Always predicting 'no diabetes' gets accuracy "
    f"{1 - yte.mean():.3f} with recall 0 -> "
    "accuracy alone misleads."
)


# ---------------------------------------------------------
# Threshold/precision/recall table and plot
# ---------------------------------------------------------

threshold_table = []

for threshold in np.arange(0.05, 0.96, 0.05):
    threshold_predictions = (
        p_te >= threshold
    ).astype(int)

    threshold_cm = confusion_matrix(
        yte,
        threshold_predictions,
    )

    threshold_metrics = metrics(threshold_cm)

    threshold_table.append(
        {
            "threshold": round(float(threshold), 2),
            "recall": threshold_metrics["recall"],
            "precision": threshold_metrics["precision"],
        }
    )

plt.figure(figsize=(6, 4))

plt.plot(
    [row["threshold"] for row in threshold_table],
    [row["recall"] for row in threshold_table],
    label="recall",
)

plt.plot(
    [row["threshold"] for row in threshold_table],
    [row["precision"] for row in threshold_table],
    label="precision",
)

plt.axvline(
    chosen,
    linestyle="--",
    color="gray",
    label=f"selected threshold = {chosen:.2f}",
)

plt.xlabel("threshold")
plt.ylabel("score")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "threshold_tradeoff.png", dpi=120)
plt.close()


# =========================================================
# FINAL CONFUSION MATRIX
# =========================================================

# Final predictions use the selected threshold
final_predictions = (
    p_te >= chosen
).astype(int)
