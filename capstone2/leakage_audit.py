"""Step 4b - Leakage audit of the trend feature.

Compares three ways of building a 'historical loss slope', all added to the Persistence
specification and evaluated with Ridge under LOEO-CV on identical rows:
  1. Capstone I contemporaneous slope: rolling 5-quarter window ending in Q4 of year t.
  2. One-quarter-shifted slope: same window on the loss series shifted back one quarter.
     Its window still covers Q1-Q3 of year t, so three-quarters of the target year leak in.
  3. Strict annual lag (Capstone II): slope over years t-3..t-1 only.
A large jump in R2 for 1 or 2 relative to 3 is the signature of target leakage.
"""
import numpy as np
import pandas as pd

from config import PROC, SPECS, TAB
from models import loeo_predict, metrics


def rolling_slope(s, w=5):
    v = s.values; out = [np.nan] * len(v)
    for i in range(w - 1, len(v)):
        y = v[i - w + 1:i + 1]; x = np.arange(w); m = ~np.isnan(y)
        out[i] = np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 3 else np.nan
    return pd.Series(out, index=s.index)


def main():
    q = pd.read_csv(PROC / "feature_matrix_quarterly.csv").sort_values(["entity", "year", "quarter_num"])
    q["slope_contemp"] = q.groupby("entity")["loss_pct"].transform(rolling_slope)
    q["slope_qshift"] = q.groupby("entity")["loss_pct"].transform(lambda s: rolling_slope(s.shift(1)))
    ann = q.groupby(["entity", "year"])[["slope_contemp", "slope_qshift"]].last().reset_index()
    df = pd.read_csv(TAB / "panel_v2.csv").merge(ann, on=["entity", "year"], how="left")
    y = df["loss_pct"].values
    change = y - df["loss_lag1"].values
    rows = []
    for label, col in [("Capstone I contemporaneous slope", "slope_contemp"),
                       ("One-quarter-shifted slope", "slope_qshift"),
                       ("Strict annual lag, years t-3..t-1 (Capstone II)", "loss_slope_lag3")]:
        m = metrics(y, loeo_predict(df, SPECS["Persistence"] + [col], "Ridge"))
        ok = df[col].notna()
        rows.append({"Trend feature": label, **m,
                     "corr_with_current_change": np.corrcoef(df.loc[ok, col], change[ok])[0, 1]})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "leakage_audit.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
