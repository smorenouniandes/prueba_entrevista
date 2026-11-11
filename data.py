from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["customer_id", "contract_id", "partition_id"]
FM_NUM = ["days_past_due", "ead_amount", "redesc_capital_amount", "current_interest_amount",
          "pastdue_cap_amount", "overdue_interest_amount", "obligation_approval_amount", "available_amount"]
CON_NUM = ["current_balance_amount", "contract_duration_number", "limit_amount",
           "total_average_balance1_amount", "pasivoM", "activoM",
           "segment_customer_asset_amount", "segment_customer_liability_amount"]
CAT = ["commercial_product_id", "epi_product_classification", "current_segment_id",
       "customer_subgroup_classif_id"]
CON_COLS = KEY + CON_NUM + CAT + ["contract_register_date", "contract_cancel_date", "current_expiry_date"]
NUM = FM_NUM + CON_NUM + ["utilization", "interest_to_exposure", "overdue_to_exposure",
      "balance_to_approval", "balance_change", "dpd_change", "dpd_volatility_3m",
      "ead_volatility_3m", "contract_age_days", "days_to_expiry", "customer_contracts",
      "customer_ead_total", "customer_max_dpd", "customer_mean_dpd", "customer_pastdue_total",
      "customer_default_prior_rate", "contract_default_prior_count"]
HIST = ["customer_default_prior_rate", "contract_default_prior_count"]


def fingerprint(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fm(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={k: "string" for k in KEY}, low_memory=False)
    missing = set(KEY + FM_NUM) - set(d.columns)
    if missing:
        raise ValueError(f"FM02 sin columnas: {sorted(missing)}")
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError("FM02 tiene claves nulas o duplicadas")
    if "default" in d:
        if not d["default"].isin([0, 1]).all():
            raise ValueError("Etiqueta no binaria")
        d["default_simulado"] = d["default"].astype(int)
        assert d["default_simulado"].equals(d["default"].astype(int))
    return d


def read_con(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={k: "string" for k in KEY + CAT}, usecols=CON_COLS, low_memory=False)
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError("Consolida tiene claves nulas o duplicadas")
    return d


def audit(fm: pd.DataFrame, con: pd.DataFrame, out: Path) -> dict:
    keys = fm[KEY].merge(con[KEY], on=KEY, how="left", indicator=True, validate="one_to_one")
    cov = keys.groupby("partition_id", observed=True)["_merge"].value_counts().unstack(fill_value=0)
    result = {
        "fm_rows": len(fm), "con_rows": len(con),
        "fm_customers": int(fm.customer_id.nunique()), "fm_contracts": int(fm.contract_id.nunique()),
        "fm_dtypes": {k: str(v) for k, v in fm.dtypes.items()},
        "con_dtypes": {k: str(v) for k, v in con.dtypes.items()},
        "fm_duplicate_keys": int(fm.duplicated(KEY).sum()), "con_duplicate_keys": int(con.duplicated(KEY).sum()),
        "fm_missing": {k: int(v) for k, v in fm.isna().sum().items()},
        "con_missing": {k: int(v) for k, v in con.isna().sum().items()},
        "matched": int((keys._merge == "both").sum()), "unmatched": int((keys._merge == "left_only").sum()),
        "coverage_by_month": {str(i): {str(k): int(v) for k, v in row.items()} for i, row in cov.iterrows()},
        "label_equal": bool(fm.default_simulado.equals(fm["default"].astype(int))),
        "prevalence_by_month": {str(k): {"n": int(g.size), "positive": int(g.sum()), "prevalence": float(g.mean())}
                                for k, g in fm.groupby("partition_id")["default_simulado"]},
        "client_prevalence_by_month": {str(k): {"clients": int(g.size), "positive_clients": int(g.sum()),
                                                "prevalence": float(g.mean())}
                                       for k, g in fm.groupby(["partition_id", "customer_id"])
                                       .default_simulado.max().groupby(level=0)},
        "contracts_by_month": {str(k): int(g.contract_id.nunique())
                               for k, g in fm.groupby("partition_id")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def make_features(fm: pd.DataFrame, con: pd.DataFrame, label_cutoff="2026-03-31") -> pd.DataFrame:
    """Construye variables actuales y pasadas; jamás usa la etiqueta del mismo corte como predictor."""
    d = fm.merge(con, on=KEY, how="left", validate="one_to_one", indicator=True)
    d["consolida_match"] = (d.pop("_merge") == "both").astype(int)
    d = d.sort_values(KEY[0:2] + [KEY[2]], kind="stable").reset_index(drop=True)
    for col in FM_NUM + CON_NUM:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in CAT:
        d[col] = d[col].fillna("__MISSING__").astype(str)
    d["utilization"] = d.ead_amount / d.obligation_approval_amount.replace(0, np.nan)
    d["interest_to_exposure"] = (d.current_interest_amount + d.overdue_interest_amount) / d.ead_amount.replace(0, np.nan)
    d["overdue_to_exposure"] = (d.pastdue_cap_amount + d.overdue_interest_amount) / d.ead_amount.replace(0, np.nan)
    d["balance_to_approval"] = d.current_balance_amount / d.obligation_approval_amount.replace(0, np.nan)
    d["balance_change"] = d.groupby(["customer_id", "contract_id"], sort=False).current_balance_amount.diff()
    d["dpd_change"] = d.groupby(["customer_id", "contract_id"], sort=False).days_past_due.diff()
    groups = d.groupby(["customer_id", "contract_id"], sort=False)
    for src, dst in [("days_past_due", "dpd_volatility_3m"), ("ead_amount", "ead_volatility_3m")]:
        previous = groups[src].shift(1)
        d[dst] = previous.groupby([d.customer_id, d.contract_id], sort=False).rolling(
            3, min_periods=2).std().reset_index(level=[0, 1], drop=True).sort_index()
    cut = pd.to_datetime(d.partition_id)
    reg = pd.to_datetime(d.contract_register_date.replace("0001-01-01", pd.NA), errors="coerce")
    expiry = pd.to_datetime(d.current_expiry_date.replace("0001-01-01", pd.NA), errors="coerce")
    d["contract_age_days"] = (cut - reg).dt.days.clip(lower=0)
    d["days_to_expiry"] = (expiry - cut).dt.days
    by_customer = d.groupby(["customer_id", "partition_id"], sort=False)
    d["customer_contracts"] = by_customer.contract_id.transform("size")
    d["customer_ead_total"] = by_customer.ead_amount.transform("sum")
    d["customer_max_dpd"] = by_customer.days_past_due.transform("max")
    d["customer_mean_dpd"] = by_customer.days_past_due.transform("mean")
    d["customer_pastdue_total"] = by_customer.pastdue_cap_amount.transform("sum")
    # La historia etiquetada se congela al cierre de marzo, también para abril y mayo.
    if "default_simulado" in d:
        allowed = d.partition_id.le(label_cutoff)
        y = d.default_simulado.where(allowed)
        d["contract_default_prior_count"] = y.fillna(0).groupby([d.customer_id, d.contract_id], sort=False).cumsum() - y.fillna(0)
        monthly = d.assign(_label=y).groupby(["customer_id", "partition_id"], as_index=False, sort=False)._label.agg(["sum", "count"])
        monthly = monthly.sort_values(["customer_id", "partition_id"])
        g = monthly.groupby("customer_id", sort=False)
        monthly["prior_sum"] = g["sum"].cumsum() - monthly["sum"]
        monthly["prior_n"] = g["count"].cumsum() - monthly["count"]
        monthly["customer_default_prior_rate"] = monthly.prior_sum / monthly.prior_n.replace(0, np.nan)
        d = d.merge(monthly[["customer_id", "partition_id", "customer_default_prior_rate"]],
                    on=["customer_id", "partition_id"], how="left", validate="many_to_one")
    else:
        d["contract_default_prior_count"] = np.nan
        d["customer_default_prior_rate"] = np.nan
    d[NUM] = d[NUM].replace([np.inf, -np.inf], np.nan)
    return d


def feature_columns(with_history: bool):
    return [x for x in NUM if with_history or x not in HIST] + ["consolida_match"] + CAT
