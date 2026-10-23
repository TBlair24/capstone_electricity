"""Step 5 - Out-of-fold explanations.

Changes from Capstone I:
  * SHAP is computed only for HELD-OUT licensee rows: in each LOEO fold the model is trained on
    the other 12 licensees and explained on the withheld one. Capstone I explained a single model
    fitted to all rows, i.e. in-sample behaviour of a model that did not generalise.
  * Ridge (the linear model) gets exact linear attributions, coef_j * (x_j - mean_train_j) in
    standardised units, plus coefficient stability across the 13 folds.
"""
import numpy as np
import pandas as pd
import shap

from config import LABELS, SPECS, TAB
from models import make_models

SPEC = "Persistence"   # best-performing Ridge-free spec under LOEO; also run Forecast


def oof_attributions(df, spec):
    feats = SPECS[spec]
    y = df["loss_pct"].values
    xgb_rows, ridge_rows, coefs = [], [], []
    for ent in df["entity"].unique():
        te = (df["entity"] == ent).values
        Xtr, Xte = df.loc[~te, feats], df.loc[te, feats]
        m = make_models()["XGBoost"]().fit(Xtr, y[~te])
        sv = shap.TreeExplainer(m).shap_values(Xte)
        xgb_rows.append(pd.DataFrame(sv, columns=feats).assign(entity=ent))
        r = make_models()["Ridge"]().fit(Xtr, y[~te])
        imp, sc, rr = r.named_steps["simpleimputer"], r.named_steps["standardscaler"], r.named_steps["ridgecv"]
        Z = sc.transform(imp.transform(Xte))           # standardised with training-fold statistics
        ridge_rows.append(pd.DataFrame(Z * rr.coef_, columns=feats).assign(entity=ent))
        coefs.append(pd.Series(rr.coef_, index=feats).rename(ent))
    X = pd.concat(xgb_rows); R = pd.concat(ridge_rows); C = pd.concat(coefs, axis=1).T
    summ = pd.DataFrame({
        "feature": feats, "label": [LABELS[f] for f in feats],
        "xgb_mean_abs_shap": X[feats].abs().mean().values,
        "xgb_sd_abs_shap": X[feats].abs().std().values,
        "ridge_mean_abs_contrib": R[feats].abs().mean().values,
        "ridge_coef_mean": C.mean().values, "ridge_coef_sd": C.std().values,
        "ridge_coef_sign_stable": ((C > 0).all() | (C < 0).all()).values,
    }).sort_values("xgb_mean_abs_shap", ascending=False)
    summ.to_csv(TAB / f"oof_attribution_{spec.lower()}.csv", index=False)
    X.to_csv(TAB / f"oof_shap_values_{spec.lower()}.csv", index=False)
    return summ


if __name__ == "__main__":
    df = pd.read_csv(TAB / "panel_v2.csv")
    for spec in ["Structural", "Persistence", "Forecast"]:
        print(f"\n== {spec} ==")
        print(oof_attributions(df, spec).drop(columns="feature").round(3).to_string(index=False))
