"""Automated checks for the Capstone II pipeline. Run:  python -m pytest capstone2/tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_panel import lagged_slope            # noqa: E402
from config import SPECS, TAB                   # noqa: E402

PANEL = pd.read_csv(TAB / "panel_v2.csv")


def test_lagged_slope_ignores_current_year():
    """Changing year t's outcome must not change the slope feature for year t."""
    years = np.arange(2010, 2020)
    vals = np.linspace(20, 30, 10)
    base = lagged_slope(years, vals)
    for i in range(len(years)):
        bumped = vals.copy(); bumped[i] += 25.0
        assert np.allclose(lagged_slope(years, bumped)[i], base[i], equal_nan=True)


def test_lag1_is_previous_year():
    a = pd.read_csv(TAB.parents[2] / "data" / "processed" / "feature_matrix_annual.csv")
    prev = a.set_index(["entity", "year"])["loss_pct"]
    for _, r in PANEL.iterrows():
        assert np.isclose(r.loss_lag1, prev[(r.entity, r.year - 1)])


def test_leaky_trend_feature_removed():
    for feats in SPECS.values():
        assert "loss_trend_slope" not in feats and "loss_pct" not in feats


def test_specifications_share_rows_and_are_nested():
    assert PANEL[["loss_pct", "loss_lag1"]].notna().all().all()
    assert set(SPECS["Structural"]) < set(SPECS["Persistence"]) < set(SPECS["Forecast"])


def test_clipped_transmission_never_exceeds_join():
    g = pd.read_csv(TAB / "gis_features_v2.csv").dropna(subset=["trans_km_join"])
    assert (g.trans_km_clipped <= g.trans_km_join + 1e-6).all()


def test_loeo_folds_hold_out_whole_licensees():
    p = pd.read_csv(TAB / "loeo_fold_metrics_main.csv")
    counts = p.groupby(["Specification", "Model"]).entity.nunique()
    assert (counts == PANEL.entity.nunique()).all()
