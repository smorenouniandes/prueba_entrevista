from __future__ import annotations

import math
import warnings

import mlflow.pyfunc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CAT, feature_columns

SEED = 42
SPECS = {
    "prevalence": {"kind": "prevalence", "history": False},
    "logistic": {"kind": "logistic", "history": False, "C": 0.1},
    "logistic_history": {"kind": "logistic", "history": True, "C": 0.1},
    "catboost_d4": {"kind": "catboost", "history": False, "depth": 4, "iterations": 140},
    "catboost_d6_history": {"kind": "catboost", "history": True, "depth": 6, "iterations": 140},
    "extra_trees": {"kind": "extra_trees", "history": False},
    "mlp_1": {"kind": "mlp", "history": False, "layers": [24]},
    "mlp_2": {"kind": "mlp", "history": False, "layers": [32, 12]},
    "lightgbm": {"kind": "lightgbm", "history": False},
    "xgboost": {"kind": "xgboost", "history": False},
    "hybrid": {"kind": "hybrid", "history": False},
    "stacking": {"kind": "stacking", "history": False},
}


def make_estimator(spec: dict, columns: list[str]):
    kind = spec["kind"]
    if kind == "prevalence":
        return None
    if kind == "catboost":
        return CatBoostClassifier(iterations=spec["iterations"], depth=spec["depth"],
                                  learning_rate=0.06, loss_function="Logloss", verbose=False,
                                  random_seed=SEED, thread_count=2, l2_leaf_reg=5,
                                  cat_features=[c for c in CAT if c in columns])
    numerics = [c for c in columns if c not in CAT]
    scale = kind in ("logistic", "mlp")
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(numeric_steps), numerics),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), [c for c in CAT if c in columns]),
    ], sparse_threshold=0)
    if kind == "logistic":
        model = LogisticRegression(C=spec["C"], max_iter=300, random_state=SEED)
    elif kind == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=100, max_depth=12, min_samples_leaf=12,
                                     max_features=0.8, n_jobs=2, random_state=SEED)
    elif kind == "mlp":
        model = MLPClassifier(hidden_layer_sizes=tuple(spec["layers"]), max_iter=55,
                              early_stopping=True, n_iter_no_change=7, random_state=SEED,
                              alpha=0.01, batch_size=512)
    elif kind == "lightgbm":
        model = LGBMClassifier(n_estimators=140, num_leaves=15, learning_rate=0.04,
                               min_child_samples=80, reg_lambda=5, verbosity=-1,
                               n_jobs=2, random_state=SEED)
    elif kind == "xgboost":
        model = XGBClassifier(n_estimators=140, max_depth=3, learning_rate=0.04,
                              min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=5, n_jobs=2, random_state=SEED,
                              eval_metric="logloss", tree_method="hist")
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("model", model)])


def fit_one(spec: dict, train: pd.DataFrame):
    cols = feature_columns(spec["history"])
    if spec["kind"] == "prevalence":
        return float(train.default_simulado.mean())
    model = make_estimator(spec, cols)
    x = train[cols].copy()
    if spec["kind"] == "catboost":
        x[[c for c in CAT if c in cols]] = x[[c for c in CAT if c in cols]].fillna("__MISSING__").astype(str)
        x[[c for c in cols if c not in CAT]] = x[[c for c in cols if c not in CAT]].fillna(-999)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, train.default_simulado.to_numpy())
    return model


def predict_one(model, spec: dict, data: pd.DataFrame) -> np.ndarray:
    if spec["kind"] == "prevalence":
        return np.full(len(data), model)
    cols = feature_columns(spec["history"])
    x = data[cols].copy()
    if spec["kind"] == "catboost":
        x[[c for c in CAT if c in cols]] = x[[c for c in CAT if c in cols]].fillna("__MISSING__").astype(str)
        x[[c for c in cols if c not in CAT]] = x[[c for c in cols if c not in CAT]].fillna(-999)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def apply_calibrator(p: np.ndarray, calibrator):
    if calibrator is None:
        return p
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return calibrator.predict_proba(z)[:, 1]


def fit_calibrator(y, p):
    if len(np.unique(y)) < 2 or np.std(p) < 1e-10:
        return None
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return LogisticRegression(C=1, max_iter=200, random_state=SEED).fit(z, y)


def metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1-1e-8)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()),
           "pr_auc": float(average_precision_score(y, p)),
           "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
           "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
           "alerts": int(pred.sum()), "alert_rate": float(pred.mean())}
    order = np.argsort(-p, kind="stable")
    for fraction in (0.01, 0.05, 0.10):
        k = max(1, math.ceil(len(y) * fraction))
        found = int(y[order[:k]].sum())
        prefix = f"top_{int(fraction*100)}pct"
        out[prefix + "_n"] = k
        out[prefix + "_positive"] = found
        out[prefix + "_precision"] = found / k
        out[prefix + "_capture"] = found / max(1, int(y.sum()))
        out[prefix + "_lift"] = (found / k) / y.mean() if y.mean() else None
    return out


class RegisteredRiskModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        b = self.bundle
        kind = b["kind"]
        if kind in ("hybrid", "stacking"):
            a = predict_one(b["models"][0], SPECS["logistic"], model_input)
            c = predict_one(b["models"][1], SPECS["catboost_d4"], model_input)
            p = (a + c) / 2 if kind == "hybrid" else b["meta"].predict_proba(np.c_[a, c])[:, 1]
        else:
            p = predict_one(b["model"], SPECS[b["spec_name"]], model_input)
        p = apply_calibrator(p, b["calibrator"])
        return pd.DataFrame({"probabilidad": p, "clase": (p >= b["threshold"]).astype(int)})
