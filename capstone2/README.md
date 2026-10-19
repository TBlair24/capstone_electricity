# Capstone II (CAI 8103) — leakage-safe, entity-level extension

This folder extends the Capstone I notebook (`notebooks/uganda_electricity_loss_risk.ipynb`).
Capstone I outputs in `data/processed/` are the inputs; every Capstone II output is written to
`outputs/capstone2/` (tables as CSV, figures as PNG).

## Run

```bash
pip install -r requirements.txt pytest
python capstone2/run_all.py        # ~1 minute; runs every step, then the tests
```

## Steps

| Step | Script | What changed relative to Capstone I |
|---|---|---|
| 1 | `gis_features.py` | Transmission length measured by geometric **clipping** to each service area (Capstone I spatial join double-counted lines); `gis_imputed` flag; district coverage table |
| 2 | `build_panel.py` | Leaky contemporaneous trend slope **removed**; `loss_lag1` and `loss_slope_lag3` built from past years only; missing network ratios imputed **inside** each training fold, not before validation |
| 3 | `archetypes.py` | K-Means on one median profile **per licensee**; 200 within-licensee bootstrap resamples, Adjusted Rand Index |
| 4 | `models.py` | Structural / Persistence / Forecast specifications × Ridge, Random Forest, XGBoost; mean **and naïve-persistence** baselines; LOEO-CV with licensee-bootstrap CIs; rolling-origin temporal validation; complete-case sensitivity; feature-group ablation |
| 4b | `leakage_audit.py` | Compares contemporaneous, one-quarter-shifted and strict annual-lag trend features |
| 5 | `explain.py` | SHAP on **held-out** licensee rows only; Ridge coefficient stability across folds |
| 6 | `figures.py` | Report figures and full-boundary Uganda map |
| — | `tests/test_pipeline.py` | 6 automated checks (no current-year leakage in lag features, whole-licensee folds, clipping ≤ join, nested specifications) |

`district_lookup.py` is the Capstone I district-to-licensee lookup, unchanged. It is an
analytical approximation; ERA does not publish official concession polygons.
