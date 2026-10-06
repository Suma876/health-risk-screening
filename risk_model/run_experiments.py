"""Question A, all three levels. Run:  python -m risk_model.run_experiments"""
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
from .scratch import (ScratchLogReg, brier_score, confusion_matrix, metrics,
                      reliability_table)

SEED = 42                       # <- your seed "S"
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)

# LEVEL 3: write YOUR prediction here BEFORE you look at the results.
PREDICTION = ("Precision will drop when I push recall to 0.9, because a lower "
              "threshold lets in many more false alarms. (Edit with your own numbers.)")

D = prepare(seed=SEED, path=sys.argv[1] if len(sys.argv) > 1 else None)
Xtr, Xte, ytr, yte = D["X_train"], D["X_test"], D["y_train"], D["y_test"]
print(f"Train {Xtr.shape}, test {Xte.shape}, positive rate {ytr.mean():.2f}\n")

# ---------------- LEVEL 1: sklearn LR + RF ----------------
print("=== LEVEL 1: sklearn models ===")
sk_lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
for name, m in [("LogisticRegression", sk_lr), ("RandomForest", rf)]:
    pr = m.predict(Xte)
    print(f"{name:20s} acc={accuracy_score(yte, pr):.3f} "
          f"prec={precision_score(yte, pr):.3f} rec={recall_score(yte, pr):.3f}")

# ---------------- LEVEL 2: from scratch ----------------
print("\n=== LEVEL 2: NumPy logistic regression ===")
model = ScratchLogReg(lr=0.1, epochs=3000, fn_weight=1.0).fit(Xtr, ytr)
cm = confusion_matrix(yte, model.predict(Xte))
ms = metrics(cm)
print("confusion matrix:", cm)
print(f"scratch  acc={ms['accuracy']:.3f} prec={ms['precision']:.3f} rec={ms['recall']:.3f}")
print(f"sklearn  acc={accuracy_score(yte, sk_lr.predict(Xte)):.3f}")

feats = np.array(D["features"])
print("\nTop-3 features by |weight| (standardized inputs):")
for tag, w in [("scratch", model.w), ("sklearn", sk_lr.coef_[0])]:
    idx = np.argsort(-np.abs(w))[:3]
    print(f"  {tag}: " + ", ".join(f"{feats[i]}={w[i]:+.3f}" for i in idx))

# Optimizer comparison (plain GD vs Adam)
adam = ScratchLogReg(lr=0.05, epochs=3000, optimizer="adam").fit(Xtr, ytr)
plt.figure(figsize=(6, 4))
plt.plot(model.loss_history, label="gradient descent")
plt.plot(adam.loss_history, label="adam")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "optimizer_loss.png", dpi=120); plt.close()

# Cost-sensitive sweep: how much worse is a missed case than a false alarm?
print("\nCost-sensitive sweep (threshold 0.5): fn_weight -> recall / precision")
for k in [1, 2, 3, 5, 10]:
    mk = ScratchLogReg(lr=0.1, epochs=3000, fn_weight=k).fit(Xtr, ytr)
    r = metrics(confusion_matrix(yte, mk.predict(Xte)))
    print(f"  k={k:<2d} recall={r['recall']:.3f} precision={r['precision']:.3f} acc={r['accuracy']:.3f}")

# Calibration of the standard model
p_te = model.predict_proba(Xte)
print(f"\nBrier score: {brier_score(yte, p_te):.3f}")
print("Reliability (predicted vs actual positive rate, n):")
rel = reliability_table(yte, p_te)
for pred, act, n in rel:
    print(f"  predicted {pred:.2f}  actual {act:.2f}  (n={n})")
plt.figure(figsize=(4.5, 4.5))
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.plot([r[0] for r in rel], [r[1] for r in rel], "o-")
plt.xlabel("predicted risk"); plt.ylabel("actual rate"); plt.tight_layout()
plt.savefig(OUT / "reliability.png", dpi=120); plt.close()

# ---------------- LEVEL 3: threshold to recall 0.9 ----------------
print("\n=== LEVEL 3: lower threshold until recall >= 0.9 ===")
print("PREDICTION (written before running):", PREDICTION)
p_tr = model.predict_proba(Xtr)
chosen = 0.5
for t in np.arange(0.5, 0.0, -0.01):          # pick threshold on TRAIN data
    if metrics(confusion_matrix(ytr, (p_tr >= t).astype(int)))["recall"] >= 0.9:
        chosen = float(round(t, 2)); break
base = metrics(confusion_matrix(yte, (p_te >= 0.5).astype(int)))
low = metrics(confusion_matrix(yte, (p_te >= chosen).astype(int)))
print(f"threshold 0.50 : precision={base['precision']:.3f} recall={base['recall']:.3f} acc={base['accuracy']:.3f}")
print(f"threshold {chosen:.2f} : precision={low['precision']:.3f} recall={low['recall']:.3f} acc={low['accuracy']:.3f}")
print("Always predicting 'no diabetes' gets accuracy "
      f"{1 - yte.mean():.3f} with recall 0 -> accuracy alone misleads.")

table = []
for t in np.arange(0.05, 0.96, 0.05):
    r = metrics(confusion_matrix(yte, (p_te >= t).astype(int)))
    table.append({"threshold": round(float(t), 2), "recall": r["recall"], "precision": r["precision"]})
plt.figure(figsize=(6, 4))
plt.plot([r["threshold"] for r in table], [r["recall"] for r in table], label="recall")
plt.plot([r["threshold"] for r in table], [r["precision"] for r in table], label="precision")
plt.axvline(chosen, ls="--", color="gray"); plt.xlabel("threshold"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "threshold_tradeoff.png", dpi=120); plt.close()

# Save the model for the API (calibrated standard model + screening threshold)
(OUT / "model.json").write_text(json.dumps({
    "features": D["features"], "w": model.w.tolist(), "b": float(model.b),
    "mean": D["mean"], "std": D["std"], "medians": D["medians"],
    "threshold": chosen, "threshold_table": table, "seed": SEED,
    "test_metrics_at_threshold": low,
}, indent=2))
