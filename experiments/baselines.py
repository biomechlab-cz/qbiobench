"""Tier-1 same-feature classical baselines (THE fair comparison).

logreg / RBF-SVM / XGBoost / compact MLP on the SAME PCA features as the QML
models. Tier-2 baselines (matched learned encoders) are not implemented, and
raw-signal results from the literature (tier 3) are context only. Neither is a
same-input comparison.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def make_baselines(seed: int = 0) -> dict:
    """Return the tier-1 estimator zoo. Tunable models (RBF-SVM, logreg) are wrapped in
    train-only GridSearchCV so the same-feature classical *ceiling* gets at least as much
    hyperparameter search as the QML SVMs (PQK grid-searches gamma/C). GridSearchCV fits
    only on the training fold passed to .fit(), so this introduces no leakage.
    XGBoost added lazily (optional import)."""
    models = {
        "logreg": GridSearchCV(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            {"C": [0.01, 0.1, 1, 10, 100]}, cv=3, n_jobs=-1),
        # mirror (and exceed) the PQK SVM grid for the headline classical ceiling
        "rbf_svm": GridSearchCV(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
            {"gamma": ["scale", 0.001, 0.01, 0.1, 1.0], "C": [0.1, 1, 10, 100]}, cv=3, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = GridSearchCV(
            XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.03, 0.1]},
            cv=3, n_jobs=-1)
    except ImportError:
        pass
    return models
