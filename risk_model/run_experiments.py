<<<<<<< HEAD

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
=======
```python
"""
Run all model experiments and save results.

Run from project root:

    python -m risk_model.run_experiments

This script:
1. Prepares the dataset.
2. Trains sklearn comparison models.
3. Trains the NumPy logistic regression model from scratch.
4. Compares Gradient Descent vs Adam.
5. Runs cost-sensitive learning experiments.
6. Finds a threshold targeting recall >= 0.90.
7. Calculates confusion matrices at threshold 0.50 and the chosen threshold.
8. Calculates calibration/reliability.
9. Saves ALL important evaluation results into artifacts/model.json.
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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

<<<<<<< HEAD
# Required Level-3 prediction
PREDICTION = (
    "Precision will drop when recall is pushed toward 0.9, "
    "because lowering the threshold flags more borderline cases "
    "and therefore creates more false positives."
=======
# Prediction required before experiment results are seen.
PREDICTION = (
    "Precision will drop when I push recall toward 0.9, "
    "because a lower threshold creates more false positives."
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
)


# =========================================================
<<<<<<< HEAD
# LOAD DATA
=======
# HELPER: convert confusion matrix into JSON-safe format
# =========================================================

def confusion_to_json(cm):
    """
    Convert our scratch confusion matrix dictionary into
    a JSON-friendly dictionary containing TN, FP, FN and TP.
    """

    return {
        "tn": int(cm["tn"]),
        "fp": int(cm["fp"]),
        "fn": int(cm["fn"]),
        "tp": int(cm["tp"]),
    }


# =========================================================
# LOAD / PREPARE DATA
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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
<<<<<<< HEAD
# LEVEL 1
# SKLEARN MODELS
=======
# LEVEL 1: STANDARD MODELS FOR COMPARISON
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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

<<<<<<< HEAD
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
=======

sklearn_results = {}

for name, m in [
    ("LogisticRegression", sk_lr),
    ("RandomForest", rf)
]:

    pr = m.predict(Xte)

    acc = accuracy_score(yte, pr)
    prec = precision_score(yte, pr, zero_division=0)
    rec = recall_score(yte, pr, zero_division=0)

    sklearn_results[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
    }

    print(
        f"{name:20s} "
        f"acc={acc:.3f} "
        f"prec={prec:.3f} "
        f"rec={rec:.3f}"
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    )


# =========================================================
<<<<<<< HEAD
# LEVEL 2
# NUMPY LOGISTIC REGRESSION FROM SCRATCH
=======
# LEVEL 2: NUMPY LOGISTIC REGRESSION FROM SCRATCH
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
# =========================================================

print("\n=== LEVEL 2: NumPy logistic regression ===")

model = ScratchLogReg(
    lr=0.1,
    epochs=3000,
    fn_weight=1.0
).fit(Xtr, ytr)

<<<<<<< HEAD
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
=======

# Default 0.50 predictions

p_te = model.predict_proba(Xte)

pred_050 = (p_te >= 0.50).astype(int)

cm_050 = confusion_matrix(
    yte,
    pred_050
)

metrics_050 = metrics(cm_050)

print("\nConfusion matrix at threshold 0.50:")
print(cm_050)

print(
    f"scratch  acc={metrics_050['accuracy']:.3f} "
    f"prec={metrics_050['precision']:.3f} "
    f"rec={metrics_050['recall']:.3f}"
)

print(
    f"sklearn  acc="
    f"{accuracy_score(yte, sk_lr.predict(Xte)):.3f}"
)


# =========================================================
# TOP FEATURES
# =========================================================

feats = np.array(D["features"])

print(
    "\nTop-3 features by |weight| "
    "(standardized inputs):"
)

top_features = {}

for tag, w in [
    ("scratch", model.w),
    ("sklearn", sk_lr.coef_[0])
]:

    idx = np.argsort(-np.abs(w))[:3]

    values = []

    for i in idx:
        values.append({
            "feature": str(feats[i]),
            "weight": float(w[i])
        })

    top_features[tag] = values
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4

    print(
        f"  {tag}: "
        + ", ".join(
<<<<<<< HEAD
            f"{features[i]}={weights[i]:+.3f}"
            for i in indices
=======
            f"{feats[i]}={w[i]:+.3f}"
            for i in idx
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
        )
    )


# =========================================================
# OPTIMIZER COMPARISON
<<<<<<< HEAD
# GRADIENT DESCENT VS ADAM
# =========================================================

=======
# =========================================================

print("\n=== Optimizer comparison ===")

>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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
<<<<<<< HEAD
    "\nCost-sensitive sweep "
    "(threshold 0.5): fn_weight -> recall / precision"
=======
    "\n=== Cost-sensitive learning ==="
)

print(
    "FN weight -> recall / precision / accuracy"
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
)

cost_sensitive_results = []

for k in [1, 2, 3, 5, 10]:

<<<<<<< HEAD
    cost_model = ScratchLogReg(
=======
    mk = ScratchLogReg(
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
        lr=0.1,
        epochs=3000,
        fn_weight=k
    ).fit(Xtr, ytr)

<<<<<<< HEAD
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
=======
    pk = mk.predict_proba(Xte)

    pred_k = (
        pk >= 0.50
    ).astype(int)

    cm_k = confusion_matrix(
        yte,
        pred_k
    )

    r = metrics(cm_k)

    result = {
        "fn_weight": int(k),
        "threshold": 0.50,
        "accuracy": float(r["accuracy"]),
        "precision": float(r["precision"]),
        "recall": float(r["recall"]),
        "confusion_matrix": confusion_to_json(cm_k),
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    }

    cost_sensitive_results.append(result)

    print(
        f"  k={k:<2d} "
<<<<<<< HEAD
        f"recall={cost_metrics['recall']:.3f} "
        f"precision={cost_metrics['precision']:.3f} "
        f"acc={cost_metrics['accuracy']:.3f}"
=======
        f"recall={r['recall']:.3f} "
        f"precision={r['precision']:.3f} "
        f"acc={r['accuracy']:.3f}"
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    )


# =========================================================
<<<<<<< HEAD
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
=======
# CALIBRATION / RELIABILITY
# =========================================================

brier = brier_score(
    yte,
    p_te
)

print(
    f"\nBrier score: {brier:.3f}"
)
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4

print(
    "Reliability "
    "(predicted vs actual positive rate, n):"
)

rel = reliability_table(
    yte,
    p_te
)

<<<<<<< HEAD
for pred, actual, n in rel:

    print(
        f"  predicted {pred:.2f} "
        f"actual {actual:.2f} "
        f"(n={n})"
    )


plt.figure(figsize=(4.5, 4.5))

=======
for pred, act, n in rel:

    print(
        f"  predicted {pred:.2f} "
        f"actual {act:.2f} "
        f"(n={n})"
    )


plt.figure(figsize=(4.5, 4.5))

>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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
<<<<<<< HEAD

=======
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
plt.tight_layout()

plt.savefig(
    OUT / "reliability.png",
    dpi=120
)

plt.close()


# =========================================================
<<<<<<< HEAD
# LEVEL 3
# SELECT THRESHOLD FOR RECALL >= 0.90
# =========================================================

print(
    "\n=== LEVEL 3: lower threshold until recall >= 0.9 ==="
)

print(
    "PREDICTION (written before running):",
=======
# LEVEL 3: SELECT THRESHOLD FOR RECALL >= 0.90
# =========================================================

print(
    "\n=== LEVEL 3: "
    "lower threshold until recall >= 0.9 ==="
)

print(
    "PREDICTION "
    "(written before running):",
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    PREDICTION
)


<<<<<<< HEAD
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
=======
# Use TRAINING data to select threshold.
p_tr = model.predict_proba(Xtr)

chosen = 0.50

for t in np.arange(
    0.50,
    0.0,
    -0.01
):

    train_pred = (
        p_tr >= t
    ).astype(int)

    train_cm = confusion_matrix(
        ytr,
        train_pred
    )

    train_metrics = metrics(
        train_cm
    )

    if train_metrics["recall"] >= 0.90:

        chosen = float(
            round(t, 2)
        )

        break


# =========================================================
# METRICS AT CHOSEN THRESHOLD
# =========================================================

pred_chosen = (
    p_te >= chosen
).astype(int)

cm_chosen = confusion_matrix(
    yte,
    pred_chosen
)

metrics_chosen = metrics(
    cm_chosen
)


print(
    f"threshold 0.50 : "
    f"precision={metrics_050['precision']:.3f} "
    f"recall={metrics_050['recall']:.3f} "
    f"acc={metrics_050['accuracy']:.3f}"
)

print(
    f"threshold {chosen:.2f} : "
    f"precision={metrics_chosen['precision']:.3f} "
    f"recall={metrics_chosen['recall']:.3f} "
    f"acc={metrics_chosen['accuracy']:.3f}"
)

print(
    "\nConfusion matrix at chosen threshold:"
)

print(cm_chosen)


# =========================================================
# BASELINE
# =========================================================

baseline_accuracy = (
    1 - yte.mean()
)

print(
    "Always predicting 'no diabetes' "
    f"gets accuracy "
    f"{baseline_accuracy:.3f} "
    "with recall 0 -> "
    "accuracy alone misleads."
)


# =========================================================
# THRESHOLD TABLE
# =========================================================

table = []

for t in np.arange(
    0.05,
    0.96,
    0.05
):

    pred_t = (
        p_te >= t
    ).astype(int)

    cm_t = confusion_matrix(
        yte,
        pred_t
    )

    r = metrics(
        cm_t
    )

    table.append({
        "threshold": round(
            float(t),
            2
        ),
        "recall": float(
            r["recall"]
        ),
        "precision": float(
            r["precision"]
        )
    })


# =========================================================
# THRESHOLD TRADE-OFF PLOT
# =========================================================

plt.figure(figsize=(6, 4))

plt.plot(
    [r["threshold"] for r in table],
    [r["recall"] for r in table],
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    label="recall"
)

plt.plot(
<<<<<<< HEAD
    [r["threshold"] for r in threshold_table],
    [r["precision"] for r in threshold_table],
=======
    [r["threshold"] for r in table],
    [r["precision"] for r in table],
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    label="precision"
)

plt.axvline(
    chosen,
    ls="--",
    color="gray"
)

plt.xlabel("threshold")
plt.ylabel("score")
<<<<<<< HEAD

=======
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "threshold_tradeoff.png",
    dpi=120
)

plt.close()


# =========================================================
<<<<<<< HEAD
# MODEL JSON
=======
# SAVE MODEL + ALL EVALUATION RESULTS
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
# =========================================================

model_json = {

    # -----------------------------
<<<<<<< HEAD
    # Model
=======
    # Model definition
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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
<<<<<<< HEAD
    # Threshold table
    # -----------------------------

    "threshold_table": threshold_table,


    # -----------------------------
    # Metrics at selected threshold
=======
    # Threshold performance table
    # -----------------------------

    "threshold_table": table,


    # -----------------------------
    # Performance at chosen threshold
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
    # -----------------------------

    "test_metrics_at_threshold": {
        "accuracy": float(
<<<<<<< HEAD
            metrics_selected["accuracy"]
        ),
        "precision": float(
            metrics_selected["precision"]
        ),
        "recall": float(
            metrics_selected["recall"]
=======
            metrics_chosen["accuracy"]
        ),
        "precision": float(
            metrics_chosen["precision"]
        ),
        "recall": float(
            metrics_chosen["recall"]
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
        )
    },


    # -----------------------------
<<<<<<< HEAD
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

=======
    # NEW: confusion matrix at
    # selected threshold
    # -----------------------------

    "confusion_matrix_at_threshold": (
        confusion_to_json(
            cm_chosen
        )
    ),


    # -----------------------------
    # NEW: confusion matrix at
    # standard threshold 0.50
    # -----------------------------

    "confusion_matrix_at_0_50": (
        confusion_to_json(
            cm_050
        )
    ),


    # -----------------------------
    # NEW: metrics at 0.50
    # -----------------------------

    "test_metrics_at_0_50": {
        "accuracy": float(
            metrics_050["accuracy"]
        ),
        "precision": float(
            metrics_050["precision"]
        ),
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
        "recall": float(
            metrics_050["recall"]
        )
    },


    # -----------------------------
<<<<<<< HEAD
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
=======
    # NEW: cost-sensitive results
    # -----------------------------

    "cost_sensitive_results": (
        cost_sensitive_results
    ),
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4


    # -----------------------------
    # Calibration
    # -----------------------------

    "brier_score": float(
<<<<<<< HEAD
        brier_score(
            yte,
            p_te
        )
    )
=======
        brier
    ),

    "reliability": [
        {
            "predicted": float(pred),
            "actual": float(act),
            "n": int(n)
        }
        for pred, act, n in rel
    ],


    # -----------------------------
    # Feature importance
    # -----------------------------

    "top_features": top_features,


    # -----------------------------
    # Comparison models
    # -----------------------------

    "sklearn_comparison": (
        sklearn_results
    ),


    # -----------------------------
    # Baseline
    # -----------------------------

    "baseline_always_negative_accuracy": (
        float(baseline_accuracy)
    ),


    # -----------------------------
    # Experiment notes
    # -----------------------------

    "experiment_notes": {
        "threshold_selection": (
            "Selected on training data "
            "as the first threshold from "
            "0.50 downward achieving "
            "recall >= 0.90."
        ),

        "cost_sensitive_threshold": (
            "Cost-sensitive models evaluated "
            "at threshold 0.50."
        ),

        "clinical_warning": (
            "These are experimental model "
            "evaluation results and are not "
            "clinical performance claims."
        )
    }
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
}


# =========================================================
<<<<<<< HEAD
# SAVE MODEL
=======
# WRITE MODEL.JSON
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
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
<<<<<<< HEAD
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
=======
# FINAL OUTPUT
# =========================================================

print("\n========================================")
print("EXPERIMENT COMPLETE")
print("========================================")

print(
    f"Chosen threshold: {chosen:.2f}"
)

print(
    "\nChosen-threshold confusion matrix:"
)

print(
    json.dumps(
        confusion_to_json(cm_chosen),
        indent=2
    )
)

print(
    "\n0.50 confusion matrix:"
)

print(
    json.dumps(
        confusion_to_json(cm_050),
        indent=2
    )
)

print(
    "\nSaved:"
)

print(
    f"  {model_path}"
)

print(
    f"  {OUT / 'optimizer_loss.png'}"
)

print(
    f"  {OUT / 'reliability.png'}"
)

print(
    f"  {OUT / 'threshold_tradeoff.png'}"
)
```
>>>>>>> 17dac0048a6574f94a2f1324baecfb6448b4d0b4
