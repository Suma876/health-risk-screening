"""Logistic regression from scratch (NumPy only) + metrics from scratch."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def metrics(cm):
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability_table(y, p, bins=5):
    """For each probability bin: mean predicted risk vs. actual positive rate."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        hi_ok = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi_ok
        if m.sum():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


class ScratchLogReg:
    """Binary logistic regression.

    fn_weight : cost of missing a positive case (label 1) relative to a false
                alarm. 1.0 = standard logistic regression.
    optimizer : "gd" (plain gradient descent) or "adam".
    """

    def __init__(self, lr=0.1, epochs=3000, l2=0.0, fn_weight=1.0, optimizer="gd"):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.fn_weight, self.optimizer = fn_weight, optimizer
        self.w, self.b, self.loss_history = None, 0.0, []

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        self.w, self.b, self.loss_history = np.zeros(d), 0.0, []
        sw = np.where(y == 1, self.fn_weight, 1.0)
        mw, vw, mb, vb = np.zeros(d), np.zeros(d), 0.0, 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            p = sigmoid(X @ self.w + self.b)
            loss = -np.mean(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            self.loss_history.append(loss + 0.5 * self.l2 * np.sum(self.w ** 2))

            err = sw * (p - y)
            gw = X.T @ err / n + self.l2 * self.w
            gb = err.mean()

            if self.optimizer == "adam":
                mw, vw = b1 * mw + (1 - b1) * gw, b2 * vw + (1 - b2) * gw ** 2
                mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb ** 2
                self.w -= self.lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
                self.b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
            else:
                self.w -= self.lr * gw
                self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
