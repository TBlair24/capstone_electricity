"""Step 2 - Build the leakage-safe licensee-year panel.

Input: the Capstone I annual matrix (data/processed/feature_matrix_annual.csv), which already
contains audited, sanity-filtered annual energy totals and operational ratios.

Changes from Capstone I:
  * The contemporaneous rolling loss-trend slope is DROPPED. It used a window that included
    the current year's quarters, so the outcome leaked into its own predictor.
  * loss_lag1 = loss in year t-1 for the same licensee (only if year t-1 exists).
  * loss_slope_lag3 = OLS slope of annual loss over years t-3..t-1 (>= 2 points), so only
    past outcomes enter. Both are computed per licensee on the full annual series.
  * GIS features come from the clipped, flagged Step 1 output.
  * Rows with a missing outcome or missing loss_lag1 are dropped, so all three specifications
    are evaluated on exactly the same rows. Other missing predictors are imputed INSIDE each
    model pipeline (training-fold medians only).
"""
import numpy as np
import pandas as pd

from config import PROC, TAB, ENTITY_TYPES, SPECS


def lagged_slope(years, values, window=3):
    """Slope of values over the `window` years strictly before each year."""
    out = []
    s = pd.Series(values, index=years)
    for y in years:
        past = s[(s.index >= y - window) & (s.index <= y - 1)].dropna()
        out.append(np.polyfit(past.index.values, past.values, 1)[0] if len(past) >= 2 else np.nan)
    return np.array(out)


def build():
    a = pd.read_csv(PROC / "feature_matrix_annual.csv")
    a = a[a["entity"].isin(ENTITY_TYPES)].sort_values(["entity", "year"]).copy()
    a = a.drop(columns=["loss_trend_slope"])          # leaky Capstone I feature

    parts = []
    for ent, g in a.groupby("entity"):
        g = g.copy()
        prev = g.set_index("year")["loss_pct"]
        g["loss_lag1"] = g["year"].map(lambda y: prev.get(y - 1, np.nan))
        g["loss_slope_lag3"] = lagged_slope(g["year"].values, g["loss_pct"].values)
        parts.append(g)
    a = pd.concat(parts)

    gis = pd.read_csv(TAB / "gis_features_v2.csv")[
        ["entity", "line_density_gis", "trans_access_clipped", "trans_access_join", "gis_imputed"]]
    p = a.merge(gis, on="entity", how="left")
    p["is_rural"] = p["entity"].map(ENTITY_TYPES).eq("rural_cooperative").astype(int)

    n_all = len(p)
    p = p.dropna(subset=["loss_pct", "loss_lag1"]).reset_index(drop=True)
    # ERA network-length fields are blank before ~2016 for most licensees, so network ratios are
    # missing there. Capstone I median-filled them BEFORE validation; here they stay missing and are
    # imputed inside each training fold. complete_case marks rows usable for the sensitivity check.
    p["complete_case"] = p[["customers_per_km", "transformer_density"]].notna().all(axis=1).astype(int)

    cols = ["entity", "year", "entity_type", "loss_pct", "complete_case", "trans_access_join"] + SPECS["Forecast"]
    p = p[cols]
    p.to_csv(TAB / "panel_v2.csv", index=False)

    miss = p[SPECS["Forecast"]].isna().mean().mul(100).round(1)
    print(f"Annual rows: {n_all}; analytical panel: {len(p)} licensee-years, "
          f"{p.entity.nunique()} licensees, {p.year.min()}-{p.year.max()}; complete-case rows: {p.complete_case.sum()}")
    print("Missing % (imputed inside model pipelines):\n" + miss[miss > 0].to_string())
    print(p.groupby("entity").year.agg(["count", "min", "max"]).to_string())
    return p


if __name__ == "__main__":
    build()
