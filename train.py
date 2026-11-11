from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .data import CAT, KEY, audit, feature_columns, fingerprint, make_features, read_con, read_fm
from .modeling import (SEED, SPECS, RegisteredRiskModel, apply_calibrator, fit_calibrator,
                       fit_one, metrics, predict_one)

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ["2026-01-31", "2026-02-28", "2026-03-31"]
EVAL = ["2026-04-30", "2026-05-31"]
MODEL_NAME = "riesgo_default_simulado"


def configure(root=ROOT):
    db = Path(os.getenv("RISK_MLFLOW_DB", str(root / "mlruns" / "tracking.db"))).resolve()
    art = Path(os.getenv("RISK_MLFLOW_ARTIFACTS", str(root / "mlruns" / "artifacts"))).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + str(db)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name("prueba_riesgo_default_simulado")
    if experiment is None:
        exp_id = mlflow.create_experiment("prueba_riesgo_default_simulado", artifact_location=art.as_uri())
    else:
        exp_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return {"tracking_uri": uri, "artifact_root": str(art), "experiment_id": exp_id}


def smoke():
    cfg = configure()
    with mlflow.start_run(run_name="smoke_configuracion") as run:
        mlflow.log_param("check", "sqlite_local")
        mlflow.log_metric("check_value", 1.0)
    assert mlflow.get_run(run.info.run_id).data.metrics["check_value"] == 1.0
    return cfg | {"smoke_run_id": run.info.run_id}


def log_metrics(prefix, d):
    for key, value in d.items():
        if value is not None and isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            mlflow.log_metric(f"{prefix}_{key}", float(value))


def candidate_predict(name, models, fold):
    if name in ("hybrid", "stacking"):
        a = predict_one(models["logistic"], SPECS["logistic"], fold)
        b = predict_one(models["catboost_d4"], SPECS["catboost_d4"], fold)
        return (a+b)/2
    return predict_one(models[name], SPECS[name], fold)


def bootstrap_ci(df, winner, benchmark, repetitions=200):
    # Clientes, no obligaciones, son las unidades de remuestreo.
    rng = np.random.default_rng(SEED)
    groups = df.groupby("customer_id", sort=False).indices
    values = list(groups.values())
    n = len(values)
    draws = []
    for _ in range(repetitions):
        idx = np.concatenate([values[i] for i in rng.integers(0, n, n)])
        y = df.default_simulado.to_numpy()[idx]
        if len(np.unique(y)) < 2:
            continue
        w = average_precision_score(y, winner[idx])
        b = average_precision_score(y, benchmark[idx])
        draws.append((w, w-b))
    a = np.asarray(draws)
    return {"bootstrap_groups": n, "bootstrap_repetitions": len(a),
            "pr_auc_ci95": np.quantile(a[:, 0], [0.025, 0.975]).tolist(),
            "pr_auc_delta_vs_logistic_ci95": np.quantile(a[:, 1], [0.025, 0.975]).tolist()}


def run():
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    cfg = smoke()
    (ROOT / "reports" / "mlflow_config.json").write_text(json.dumps(cfg, indent=2))
    packages = ["pandas", "numpy", "scipy", "scikit-learn", "mlflow", "catboost", "lightgbm", "xgboost", "openpyxl"]
    versions = {p: importlib.metadata.version(p) for p in packages}
    versions["python"] = platform.python_version()
    (ROOT / "reports" / "versions.json").write_text(json.dumps(versions, indent=2))
    fm_path = ROOT / "fm02_impago.csv"
    con_path = ROOT / "consolida (1).csv"
    fm, con = read_fm(fm_path), read_con(con_path)
    audit_result = audit(fm, con, ROOT / "reports" / "auditoria.json")
    data = make_features(fm, con)
    # Comprobación directa: alterar todas las etiquetas abril/mayo no cambia sus variables históricas.
    changed = fm.copy()
    changed.loc[changed.partition_id.gt("2026-03-31"), "default_simulado"] = 1 - changed.loc[changed.partition_id.gt("2026-03-31"), "default_simulado"]
    alt = make_features(changed, con)
    check_cols = ["customer_default_prior_rate", "contract_default_prior_count"]
    assert data.loc[data.partition_id.gt("2026-03-31"), check_cols].equals(
        alt.loc[alt.partition_id.gt("2026-03-31"), check_cols])
    del alt, changed, fm, con
    code_hash = fingerprint(ROOT / "risk_pipeline" / "train.py")
    input_hash = {"fm02_sha256": fingerprint(fm_path), "consolida_sha256": fingerprint(con_path)}
    all_names = list(SPECS)
    base_names = [n for n in all_names if n not in ("hybrid", "stacking")]
    predictions = {n: {} for n in all_names}
    fold_models = {}
    run_ids = {}
    durations = {}
    comparison = []
    with mlflow.start_run(run_name="pipeline_principal") as parent:
        parent_id = parent.info.run_id
        mlflow.log_params({"seed": SEED, "selection": "mean_validation_average_precision",
                           "train_max": "2026-03-31", "eval_start": "2026-04-30", **input_hash})
        mlflow.set_tags({"code_sha256": code_hash, "target": "default_simulado", "task": "contemporaneous"})
        mlflow.log_dict(versions, "versions.json")
        mlflow.log_dict(audit_result, "audit.json")
        mlflow.log_dict({"features_no_history": feature_columns(False),
                         "features_history": feature_columns(True)}, "features.json")
        # Cada partición se ajusta exclusivamente con meses anteriores.
        for month in MONTHS:
            train = data.loc[data.partition_id.lt(month)]
            valid = data.loc[data.partition_id.eq(month)]
            fold_models[month] = {}
            for name in base_names:
                spec = SPECS[name]
                t = time.monotonic()
                with mlflow.start_run(run_name=f"{name}_{month}", nested=True) as fold_run:
                    mlflow.log_params({"algorithm": spec["kind"], "candidate": name, "validation_month": month,
                                       "train_end_exclusive": month, "train_n": len(train), "seed": SEED,
                                       "history_labels": spec["history"], **{k: v for k, v in spec.items() if k not in ("kind", "history") and not isinstance(v, list)}})
                    mlflow.set_tags({"code_sha256": code_hash, **input_hash})
                    try:
                        model = fit_one(spec, train)
                        p = predict_one(model, spec, valid)
                        predictions[name][month] = p
                        fold_models[month][name] = model
                        log_metrics("validation", metrics(valid.default_simulado, p))
                        mlflow.log_metric("duration_seconds", time.monotonic()-t)
                    except Exception as exc:
                        mlflow.set_tag("failure_reason", repr(exc))
                        raise
            predictions["hybrid"][month] = (predictions["logistic"][month] + predictions["catboost_d4"][month])/2
        # Meta-aprendizaje solo con predicciones fuera de muestra de enero y febrero.
        meta_y = np.concatenate([data.loc[data.partition_id.eq(m), "default_simulado"].to_numpy() for m in MONTHS[:2]])
        meta_x = np.vstack([np.c_[predictions["logistic"][m], predictions["catboost_d4"][m]] for m in MONTHS[:2]])
        meta = LogisticRegression(C=0.1, max_iter=200, random_state=SEED).fit(meta_x, meta_y)
        month = MONTHS[-1]
        x = np.c_[predictions["logistic"][month], predictions["catboost_d4"][month]]
        predictions["stacking"][month] = meta.predict_proba(x)[:, 1]
        # Stacking se desarrolla con enero/febrero y se compara únicamente en marzo.
        for name in all_names:
            spec = SPECS[name]
            t = time.monotonic()
            with mlflow.start_run(run_name=f"candidato_{name}", nested=True) as candidate_run:
                run_ids[name] = candidate_run.info.run_id
                mlflow.log_params({"candidate": name, "algorithm": spec["kind"], "seed": SEED,
                                   "history_labels": spec["history"], "selection_metric": "mean_validation_pr_auc",
                                   "threshold_rule": "march_p95", "calibration": "platt_jan_feb"})
                mlflow.set_tags({"code_sha256": code_hash, **input_hash})
                mlflow.log_dict({"features": feature_columns(spec["history"]), "spec": spec}, "configuration.json")
                aps = []
                for month in ([MONTHS[-1]] if name == "stacking" else MONTHS):
                    y = data.loc[data.partition_id.eq(month), "default_simulado"]
                    p = predictions[name][month]
                    m = metrics(y, p)
                    aps.append(m["pr_auc"])
                    log_metrics(f"validation_{month}", m)
                    comparison.append({"candidate": name, "run_id": run_ids[name], "phase": "validation", "month": month,
                                       "threshold": 0.5, **m})
                selection_ap = float(np.mean(aps))
                mlflow.log_metric("selection_pr_auc", selection_ap)
                mlflow.log_metric("validation_pr_auc_mean_descriptive", float(np.mean(aps)))
                durations[name] = time.monotonic()-t
        # Stacking tiene solo marzo evaluable con el meta; para comparar de manera justa,
        # usamos AP de marzo para todos si su AP supera al mejor promedio únicamente
        # como dato secundario. El criterio principal excluye stacking del ganador.
        eligible = [n for n in all_names if n != "stacking"]
        best = max(eligible, key=lambda n: (np.mean([average_precision_score(
            data.loc[data.partition_id.eq(m), "default_simulado"], predictions[n][m]) for m in MONTHS]),
            average_precision_score(data.loc[data.partition_id.eq(MONTHS[-1]), "default_simulado"], predictions[n][MONTHS[-1]]),
            -metrics(data.loc[data.partition_id.eq(MONTHS[-1]), "default_simulado"], predictions[n][MONTHS[-1]])["brier"], n))
        calibrated = {}
        thresholds = {}
        for name in all_names:
            raw = np.concatenate([predictions[name][m] for m in MONTHS[:2]]) if name != "stacking" else None
            cal = fit_calibrator(meta_y, raw) if name != "stacking" else None
            calibrated[name] = cal
            march = apply_calibrator(predictions[name][MONTHS[-1]], cal)
            thresholds[name] = float(np.quantile(march, 0.95))
        final_models = {}
        eval_predictions = {}
        train_final = data.loc[data.partition_id.le("2026-03-31")]
        evaluation = data.loc[data.partition_id.isin(EVAL)].copy().reset_index(drop=True)
        for name in all_names:
            spec = SPECS[name]
            with mlflow.start_run(run_id=run_ids[name], nested=True):
                mlflow.log_param("threshold", thresholds[name])
                mlflow.log_param("calibration_fitted", calibrated[name] is not None)
                t = time.monotonic()
                if name in ("hybrid", "stacking"):
                    if "logistic" not in final_models:
                        final_models["logistic"] = fit_one(SPECS["logistic"], train_final)
                    if "catboost_d4" not in final_models:
                        final_models["catboost_d4"] = fit_one(SPECS["catboost_d4"], train_final)
                    a = predict_one(final_models["logistic"], SPECS["logistic"], evaluation)
                    b = predict_one(final_models["catboost_d4"], SPECS["catboost_d4"], evaluation)
                    raw = (a+b)/2 if name == "hybrid" else meta.predict_proba(np.c_[a,b])[:, 1]
                    final_models[name] = [final_models["logistic"], final_models["catboost_d4"]]
                else:
                    if name not in final_models:
                        final_models[name] = fit_one(spec, train_final)
                    raw = predict_one(final_models[name], spec, evaluation)
                p = apply_calibrator(raw, calibrated[name])
                eval_predictions[name] = p
                for month in EVAL + ["combined"]:
                    mask = evaluation.partition_id.eq(month).to_numpy() if month != "combined" else np.ones(len(evaluation), bool)
                    for rule, threshold in [("p05", 0.5), ("operational", thresholds[name])]:
                        m = metrics(evaluation.default_simulado.to_numpy()[mask], p[mask], threshold)
                        log_metrics(f"exploratory_{month}_{rule}", m)
                        comparison.append({"candidate": name, "run_id": run_ids[name], "phase": "exploratory", "month": month,
                                           "threshold_rule": rule, "threshold": threshold, **m})
                mlflow.log_metric("final_fit_seconds", time.monotonic()-t)
                mlflow.log_dict({"selection_winner": best, "selected_before_evaluation": name == best}, "selection.json")
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(ROOT / "reports" / "comparacion_modelos.csv", index=False)
        mlflow.log_artifact(str(ROOT / "reports" / "comparacion_modelos.csv"))
        ci = bootstrap_ci(evaluation, eval_predictions[best], eval_predictions["logistic"])
        (ROOT / "reports" / "incertidumbre.json").write_text(json.dumps(ci, indent=2))
        mlflow.log_dict(ci, "incertidumbre.json")
        # Registrar en el run del ganador. El modelo recibe variables ya calculadas;
        # contiene imputación/codificación, estimador, calibrador y umbral.
        b = {"kind": SPECS[best]["kind"], "spec_name": best,
             "model": final_models[best] if best not in ("hybrid", "stacking") else None,
             "models": final_models[best] if best in ("hybrid", "stacking") else None,
             "meta": meta if best == "stacking" else None,
             "calibrator": calibrated[best], "threshold": thresholds[best]}
        bundle_path = ROOT / "outputs" / "candidate_bundle.joblib"
        joblib.dump(b, bundle_path)
        example = evaluation[feature_columns(SPECS[best]["history"])].head(5).copy()
        for c in CAT:
            example[c] = example[c].astype(str)
        for c in example.columns.difference(CAT):
            example[c] = pd.to_numeric(example[c], errors="coerce").astype(float)
        expected = RegisteredRiskModel()
        expected.bundle = b
        signature = infer_signature(example, expected.predict(None, example))
        with mlflow.start_run(run_id=run_ids[best], nested=True):
            info = mlflow.pyfunc.log_model(artifact_path="model", python_model=RegisteredRiskModel(),
                                           artifacts={"bundle": str(bundle_path)},
                                           signature=signature, input_example=example,
                                           registered_model_name=MODEL_NAME)
            mlflow.log_artifact(str(bundle_path))
        client = mlflow.MlflowClient()
        versions_model = client.search_model_versions(f"name='{MODEL_NAME}'")
        model_version = max((v for v in versions_model if v.run_id == run_ids[best]), key=lambda v: int(v.version))
        client.set_registered_model_alias(MODEL_NAME, "candidate", model_version.version)
        loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version.version}")
        sample = evaluation[feature_columns(SPECS[best]["history"])].head(100).copy()
        for c in sample.columns.difference(CAT):
            sample[c] = pd.to_numeric(sample[c], errors="coerce").astype(float)
        actual = loaded.predict(sample).probabilidad.to_numpy()
        assert np.allclose(actual, eval_predictions[best][:100], rtol=1e-7, atol=1e-9)
        selected_rows = evaluation[KEY + ["default_simulado", "consolida_match"]].copy()
        selected_rows["probabilidad"] = eval_predictions[best]
        selected_rows["clase"] = (selected_rows.probabilidad >= thresholds[best]).astype(int)
        selected_rows["ranking_mes"] = selected_rows.groupby("partition_id").probabilidad.rank(ascending=False, method="first").astype(int)
        selected_rows.to_csv(ROOT / "outputs" / "predicciones_exploratorias.csv", index=False)
        (ROOT / "reports" / "resultado.json").write_text(json.dumps({
            "parent_run_id": parent_id, "selected": best, "run_id": run_ids[best],
            "model_name": MODEL_NAME, "model_version": str(model_version.version),
            "model_uri": info.model_uri, "threshold": thresholds[best],
            "selection_metric": float(np.mean([average_precision_score(
                data.loc[data.partition_id.eq(m), "default_simulado"], predictions[best][m]) for m in MONTHS])),
            "reload_max_abs_diff": float(np.max(np.abs(actual-eval_predictions[best][:100]))),
            "runs": run_ids, "uncertainty": ci,
        }, indent=2, ensure_ascii=False))
        mlflow.log_artifact(str(ROOT / "reports" / "resultado.json"))
        mlflow.log_artifact(str(ROOT / "outputs" / "predicciones_exploratorias.csv"))
    print((ROOT / "reports" / "resultado.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke(), indent=2) if args.smoke else run())
