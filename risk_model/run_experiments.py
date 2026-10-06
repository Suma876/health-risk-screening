
"""
Question A - complete experiment script.

Run:
    python -m risk_model.run_experiments

This script:
1. Trains sklearn Logistic Regression and Random Forest models.
2. Trains NumPy logistic regression from scratch.
3. Compares Gradient Descent and Adam.
4. Performs cost-sensitive learning.
5. Finds a screening threshold targeting recall >= 0.90.
6. Saves confusion matrices at:
       - selected threshold
       - threshold 0.50
7. Saves cost-sensitive results to artifacts/model.json.
8. Saves threshold/calibration/optimizer plots.
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

from .data import prepare
from .scratch import (
    ScratchLogReg,
    brier_score,
    confusion_matrix,
    metrics,
    reliability_table,
)


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# Required Level-3 prediction
PREDICTION = (
    "Precision will drop when recall is pushed toward 0.9, "
    "because lowering the threshold flags more borderline cases "
    "and therefore creates more false positives."
)


# =========================================================
# LOAD DATA
# =========================================================

D = prepare(
    seed=SEED,
    path=sys.argv[1] if len(sys.argv) > 1 else None
)

Xtr = D["X_train"]
Xte = D["X_test"]
ytr = D["y_train"]
yte = D["y_test"]

print(
    f"Train {Xtr.shape}, "
    f"test {Xte.shape}, "
    f"positive rate {ytr.mean():.2f}\n"
)


# =========================================================
# LEVEL 1
# SKLEARN MODELS
# =========================================================

print("=== LEVEL 1: sklearn models ===")

sk_lr = LogisticRegression(
    C=1e6,
    max_iter=2000
).fit(Xtr, ytr)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED
).fit(Xtr, ytr)

for name, model in [
    ("LogisticRegression", sk_lr),
    ("RandomForest", rf),
]:

    predictions = model.predict(Xte)

    print(
        f"{name:20s} "
        f"acc={accuracy_score(yte, predictions):.3f} "
        f"prec={precision_score(yte, predictions):.3f} "
        f"rec={recall_score(yte, predictions):.3f}"
    )


# =========================================================
# LEVEL 2
# NUMPY LOGISTIC REGRESSION FROM SCRATCH
# =========================================================

print("\n=== LEVEL 2: NumPy logistic regression ===")

model = ScratchLogReg(
    lr=0.1,
    epochs=3000,
    fn_weight=1.0
).fit(Xtr, ytr)

predictions = model.predict(Xte)

cm = confusion_matrix(
    yte,
    predictions
)

ms = metrics(cm)

print("confusion matrix:", cm)

print(
    f"scratch  "
    f"acc={ms['accuracy']:.3f} "
    f"prec={ms['precision']:.3f} "
    f"rec={ms['recall']:.3f}"
)

print(
    f"sklearn  "
    f"acc={accuracy_score(yte, sk_lr.predict(Xte)):.3f}"
)


# =========================================================
# TOP FEATURES
# =========================================================

features = np.array(D["features"])

print("\nTop-3 features by |weight| (standardized inputs):")

for tag, weights in [
    ("scratch", model.w),
    ("sklearn", sk_lr.coef_[0])
]:

    indices = np.argsort(-np.abs(weights))[:3]

    print(
        f"  {tag}: "
        + ", ".join(
            f"{features[i]}={weights[i]:+.3f}"
            for i in indices
        )
    )


# =========================================================
# OPTIMIZER COMPARISON
# GRADIENT DESCENT VS ADAM
# =========================================================

adam = ScratchLogReg(
    lr=0.05,
    epochs=3000,
    optimizer="adam"
).fit(Xtr, ytr)

plt.figure(figsize=(6, 4))

plt.plot(
    model.loss_history,
    label="gradient descent"
)

plt.plot(
    adam.loss_history,
    label="adam"
)

plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "optimizer_loss.png",
    dpi=120
)

plt.close()


# =========================================================
# COST-SENSITIVE LEARNING
# =========================================================

print(
    "\nCost-sensitive sweep "
    "(threshold 0.5): fn_weight -> recall / precision"
)

cost_sensitive_results = []

for k in [1, 2, 3, 5, 10]:

    cost_model = ScratchLogReg(
        lr=0.1,
        epochs=3000,
        fn_weight=k
    ).fit(Xtr, ytr)

    cost_predictions = cost_model.predict(Xte)

    cost_cm = confusion_matrix(
        yte,
        cost_predictions
    )

    cost_metrics = metrics(cost_cm)

    result = {
        "fn_weight": k,
        "recall": float(cost_metrics["recall"]),
        "precision": float(cost_metrics["precision"]),
        "accuracy": float(cost_metrics["accuracy"]),
        "confusion_matrix": {
            "tn": int(cost_cm["tn"]),
            "fp": int(cost_cm["fp"]),
            "fn": int(cost_cm["fn"]),
            "tp": int(cost_cm["tp"]),
        }
    }

    cost_sensitive_results.append(result)

    print(
        f"  k={k:<2d} "
        f"recall={cost_metrics['recall']:.3f} "
        f"precision={cost_metrics['precision']:.3f} "
        f"acc={cost_metrics['accuracy']:.3f}"
    )


# =========================================================
# STANDARD MODEL PROBABILITIES
# =========================================================

p_te = model.predict_proba(Xte)

print(
    f"\nBrier score: "
    f"{brier_score(yte, p_te):.3f}"
)


# =========================================================
# RELIABILITY / CALIBRATION
# =========================================================

print(
    "Reliability "
    "(predicted vs actual positive rate, n):"
)

rel = reliability_table(
    yte,
    p_te
)

for pred, actual, n in rel:

    print(
        f"  predicted {pred:.2f} "
        f"actual {actual:.2f} "
        f"(n={n})"
    )


plt.figure(figsize=(4.5, 4.5))

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray"
)

plt.plot(
    [r[0] for r in rel],
    [r[1] for r in rel],
    "o-"
)

plt.xlabel("predicted risk")
plt.ylabel("actual rate")

plt.tight_layout()

plt.savefig(
    OUT / "reliability.png",
    dpi=120
)

plt.close()


# =========================================================
# LEVEL 3
# SELECT THRESHOLD FOR RECALL >= 0.90
# =========================================================

print(
    "\n=== LEVEL 3: lower threshold until recall >= 0.9 ==="
)

print(
    "PREDICTION (written before running):",
    PREDICTION
)


p_tr = model.predict_proba(Xtr)

chosen = 0.5

for threshold in np.arange(
    0.5,
    0.0,
    -0.01
):

    train_predictions = (
        p_tr >= threshold
    ).astype(int)

    train_cm = confusion_matrix(
        ytr,
        train_predictions
    )

    train_metrics = metrics(
        train_cm
    )

    if train_metrics["recall"] >= 0.9:

        chosen = float(
            round(threshold, 2)
        )

        break


# =========================================================
# TEST METRICS AT 0.50
# =========================================================

pred_050 = (
    p_te >= 0.50
).astype(int)

cm_050 = confusion_matrix(
    yte,
    pred_050
)

metrics_050 = metrics(
    cm_050
)


# =========================================================
# TEST METRICS AT SELECTED THRESHOLD
# =========================================================

pred_selected = (
    p_te >= chosen
).astype(int)

cm_selected = confusion_matrix(
    yte,
    pred_selected
)

metrics_selected = metrics(
    cm_selected
)


# =========================================================
# PRINT THRESHOLD RESULTS
# =========================================================

print(
    f"threshold 0.50 : "
    f"precision={metrics_050['precision']:.3f} "
    f"recall={metrics_050['recall']:.3f} "
    f"acc={metrics_050['accuracy']:.3f}"
)

print(
    f"threshold {chosen:.2f} : "
    f"precision={metrics_selected['precision']:.3f} "
    f"recall={metrics_selected['recall']:.3f} "
    f"acc={metrics_selected['accuracy']:.3f}"
)


print("\nSelected threshold confusion matrix:")
print(cm_selected)


print(
    "\nAlways predicting 'no diabetes' gets accuracy "
    f"{1 - yte.mean():.3f} with recall 0 "
    "-> accuracy alone misleads."
)


# =========================================================
# THRESHOLD TRADE-OFF TABLE
# =========================================================

threshold_table = []

for threshold in np.arange(
    0.05,
    0.96,
    0.05
):

    predictions_at_threshold = (
        p_te >= threshold
    ).astype(int)

    cm_t = confusion_matrix(
        yte,
        predictions_at_threshold
    )

    m_t = metrics(cm_t)

    threshold_table.append(
        {
            "threshold": round(
                float(threshold),
                2
            ),
            "recall": float(
                m_t["recall"]
            ),
            "precision": float(
                m_t["precision"]
            )
        }
    )


plt.figure(figsize=(6, 4))

plt.plot(
    [r["threshold"] for r in threshold_table],
    [r["recall"] for r in threshold_table],
    label="recall"
)

plt.plot(
    [r["threshold"] for r in threshold_table],
    [r["precision"] for r in threshold_table],
    label="precision"
)

plt.axvline(
    chosen,
    ls="--",
    color="gray"
)

plt.xlabel("threshold")
plt.ylabel("score")

plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "threshold_tradeoff.png",
    dpi=120
)

plt.close()


# =========================================================
# MODEL JSON
# =========================================================

model_json = {

    # -----------------------------
    # Model
    # -----------------------------

    "features": D["features"],

    "w": model.w.tolist(),

    "b": float(model.b),

    "mean": D["mean"],

    "std": D["std"],

    "medians": D["medians"],

    "seed": SEED,


    # -----------------------------
    # Selected screening threshold
    # -----------------------------

    "threshold": chosen,


    # -----------------------------
    # Threshold table
    # -----------------------------

    "threshold_table": threshold_table,


    # -----------------------------
    # Metrics at selected threshold
    # -----------------------------

    "test_metrics_at_threshold": {
        "accuracy": float(
            metrics_selected["accuracy"]
        ),
        "precision": float(
            metrics_selected["precision"]
        ),
        "recall": float(
            metrics_selected["recall"]
        )
    },


    # -----------------------------
    # Confusion matrix at selected
    # threshold
    # -----------------------------

    "confusion_matrix_at_threshold": {

        "tn": int(
            cm_selected["tn"]
        ),

        "fp": int(
            cm_selected["fp"]
        ),

        "fn": int(
            cm_selected["fn"]
        ),

        "tp": int(
            cm_selected["tp"]
        )
    },


    # -----------------------------
    # Metrics at threshold 0.50
    # -----------------------------

    "test_metrics_at_0_50": {

        "accuracy": float(
            metrics_050["accuracy"]
        ),

        "precision": float(
            metrics_050["precision"]
        ),

        "recall": float(
            metrics_050["recall"]
        )
    },


    # -----------------------------
    # Confusion matrix at 0.50
    # -----------------------------

    "confusion_matrix_at_0_50": {

        "tn": int(
            cm_050["tn"]
        ),

        "fp": int(
            cm_050["fp"]
        ),

        "fn": int(
            cm_050["fn"]
        ),

        "tp": int(
            cm_050["tp"]
        )
    },


    # -----------------------------
    # Cost-sensitive results
    # -----------------------------

    "cost_sensitive_results":
        cost_sensitive_results,


    # -----------------------------
    # Calibration
    # -----------------------------

    "brier_score": float(
        brier_score(
            yte,
            p_te
        )
    )
}


# =========================================================
# SAVE MODEL
# =========================================================

model_path = OUT / "model.json"

model_path.write_text(
    json.dumps(
        model_json,
        indent=2
    ),
    encoding="utf-8"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)

print(
    "Saved artifacts/model.json"
)

print(
    "Saved artifacts/optimizer_loss.png"
)

print(
    "Saved artifacts/reliability.png"
)

print(
    "Saved artifacts/threshold_tradeoff.png"
)

print(
    f"\nSelected threshold: {chosen:.2f}"
)

print(
    "Selected-threshold confusion matrix:",
    cm_selected
)

print(
    "0.50 confusion matrix:",
    cm_050
)

print(
    "\nCost-sensitive results saved:",
    len(cost_sensitive_results),
    "experiments"
)

print("=" * 60)
