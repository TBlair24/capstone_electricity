"""Step 4 - Model evaluation: three feature specifications x four learners + naive persistence.

Validation designs
  * LOEO-CV  : leave one licensee out (13 folds) - transfer to an unseen licensee.
  * Rolling-origin : train on all years < Y, test on year Y - next-year forecast for known licensees.
Uncertainty: 2,000 licensee-level bootstrap resamples of out-of-fold residuals.
All imputation and scaling are fitted inside each training fold (sklearn Pipeline).
Hyperparameters are fixed in advance (Capstone I tuned values for XGBoost), so no tuning
leaks into the reported folds; Ridge chooses its penalty by internal CV on the training fold only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import SEED, SPECS, TAB, STRUCTURAL

ALPHAS = np.logspace(-2, 3, 40)


def make_models():
    return {
        "Ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "Random Forest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                               RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                                                     random_state=SEED, n_jobs=-1)),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8,
                                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2.0,
                                            min_child_weight=5, random_state=SEED, verbosity=0),
    }


def metrics(y, p):
    r = y - p
    return {"RMSE": np.sqrt(np.mean(r ** 2)), "MAE": np.mean(np.abs(r)),
            "R2": 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2), "Bias": np.mean(r)}


def loeo_predict(df, feats, model_name):
    y = df["loss_pct"].values
    pred = np.full(len(df), np.nan)
    for ent in df["entity"].unique():
        te = (df["entity"] == ent).values
        if model_name == "Mean baseline":
            pred[te] = y[~te].mean()
        elif model_name == "Naive persistence":
            pred[te] = df.loc[te, "loss_lag1"].values
        else:
            m = make_models()[model_name]()
            m.fit(df.loc[~te, feats], y[~te])
            pred[te] = m.predict(df.loc[te, feats])
    return pred


def entity_bootstrap(df, resid, ref_resid, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    ents = df["entity"].unique()
    idx_by = {e: np.where(df["entity"].values == e)[0] for e in ents}
    rm, d = [], []
    for _ in range(n):
        idx = np.concatenate([idx_by[e] for e in rng.choice(ents, len(ents))])
        a = np.sqrt(np.mean(resid[idx] ** 2)); b = np.sqrt(np.mean(ref_resid[idx] ** 2))
        rm.append(a); d.append(a - b)
    return np.percentile(rm, [2.5, 97.5]), np.percentile(d, [2.5, 97.5])


def run_loeo(df, tag="main"):
    y = df["loss_pct"].values
    rows, preds, folds = [], {}, []
    base = loeo_predict(df, [], "Mean baseline")
    pers = loeo_predict(df, [], "Naive persistence")
    for name, p in [("Mean baseline", base), ("Naive persistence", pers)]:
        preds[("Reference", name)] = p
    for spec, feats in SPECS.items():
        for mname in make_models():
            preds[(spec, mname)] = loeo_predict(df, feats, mname)
    for (spec, mname), p in preds.items():
        r = y - p
        ci, dci_mean = entity_bootstrap(df, r, y - base)
        _, dci_pers = entity_bootstrap(df, r, y - pers)
        rows.append({"Specification": spec, "Model": mname, **metrics(y, p),
                     "RMSE_lo": ci[0], "RMSE_hi": ci[1],
                     "dRMSE_vs_mean_lo": dci_mean[0], "dRMSE_vs_mean_hi": dci_mean[1],
                     "dRMSE_vs_persist_lo": dci_pers[0], "dRMSE_vs_persist_hi": dci_pers[1]})
        for ent in df["entity"].unique():
            m = (df["entity"] == ent).values
            folds.append({"Specification": spec, "Model": mname, "entity": ent, "n": int(m.sum()),
                          "RMSE": np.sqrt(np.mean(r[m] ** 2)), "MAE": np.mean(np.abs(r[m])), "Bias": r[m].mean()})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / f"loeo_results_{tag}.csv", index=False)
    pd.DataFrame(folds).to_csv(TAB / f"loeo_fold_metrics_{tag}.csv", index=False)
    out = df[["entity", "year", "loss_pct"]].copy()
    for (spec, mname), p in preds.items():
        out[f"{spec}|{mname}"] = p
    out.to_csv(TAB / f"loeo_predictions_{tag}.csv", index=False)
    return res


def run_rolling(df, min_train=20):
    rows = []
    for Y in sorted(df["year"].unique()):
        tr, te = df[df.year < Y], df[df.year == Y]
        if len(tr) < min_train or len(te) < 3:
            continue
        yt = te["loss_pct"].values
        cand = {"Mean baseline": np.full(len(te), tr["loss_pct"].mean()),
                "Naive persistence": te["loss_lag1"].values}
        for spec in ["Structural", "Forecast"]:
            for mname in ["Ridge", "XGBoost"]:
                m = make_models()[mname]()
                m.fit(tr[SPECS[spec]], tr["loss_pct"])
                cand[f"{spec} {mname}"] = m.predict(te[SPECS[spec]])
        for k, p in cand.items():
            rows.append({"test_year": Y, "n_train": len(tr), "n_test": len(te), "Model": k, **metrics(yt, p)})
    r = pd.DataFrame(rows)
    r.to_csv(TAB / "rolling_origin_metrics.csv", index=False)
    return r


def run_ablation(df):
    """Forecast Ridge with feature groups removed (LOEO)."""
    y = df["loss_pct"].values
    gis = ["line_density_gis", "trans_access_clipped", "gis_imputed"]
    ops = ["customers_per_km", "transformer_density", "staff_per_1000_cust", "om_per_customer", "network_growth_rate"]
    full = SPECS["Forecast"]
    variants = {
        "Forecast (all features)": full,
        "minus GIS features": [f for f in full if f not in gis],
        "minus operational features": [f for f in full if f not in ops],
        "lags only (loss_lag1 + slope)": ["loss_lag1", "loss_slope_lag3"],
        "Structural + clipped transmission": STRUCTURAL,
        "Structural + join transmission (Capstone I)": [f if f != "trans_access_clipped" else "trans_access_join" for f in STRUCTURAL],
    }
    rows = [{"Variant": k, **metrics(y, loeo_predict(df, v, "Ridge"))} for k, v in variants.items()]
    a = pd.DataFrame(rows)
    a.to_csv(TAB / "ablation_ridge.csv", index=False)
    return a


if __name__ == "__main__":
    df = pd.read_csv(TAB / "panel_v2.csv")
    pd.set_option("display.width", 200)
    main = run_loeo(df, "main")
    print(main[["Specification", "Model", "RMSE", "MAE", "R2", "Bias", "RMSE_lo", "RMSE_hi",
                "dRMSE_vs_mean_lo", "dRMSE_vs_mean_hi", "dRMSE_vs_persist_lo", "dRMSE_vs_persist_hi"]].round(3).to_string(index=False))
    cc = run_loeo(df[df.complete_case == 1].reset_index(drop=True), "complete_case")
    print("\nComplete-case sensitivity:")
    print(cc[["Specification", "Model", "RMSE", "MAE", "R2"]].round(3).to_string(index=False))
    print("\nAblation (Ridge, LOEO):")
    print(run_ablation(df).round(3).to_string(index=False))
    ro = run_rolling(df)
    print("\nRolling-origin RMSE by year:")
    print(ro.pivot(index="test_year", columns="Model", values="RMSE").round(2).to_string())
