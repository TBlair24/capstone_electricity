# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is an MSc capstone (UTAMU) on predicting electricity distribution-loss % across Uganda's 13 licensed distribution entities. It uses ERA regulatory data (2009–2022) plus GIS network geometry. The work has two phases:

- **Capstone I (CAI 7202):** one Jupyter notebook, `notebooks/uganda_electricity_loss_risk.ipynb`, with outputs in `data/processed/` and `outputs/figures/`. Tag `capstone1-final`.
- **Capstone II (CAI 8103):** a script package, `capstone2/`, with a pytest suite. Its outputs are in `outputs/capstone2/`.

There is no linter. See README.md for data download URLs.

## Git conventions that matter

- **Leave the submitted state alone.** Branch `capstone2` and tag `capstone2-cat1` are the exact CAT 1 submission (commit `eb6bbce`, 19 Oct 2026), and the submitted report points graders to them. Never commit to that branch or move that tag. New work goes on `main` or a new branch.
- **`capstone_2/` is a frozen archive.** It holds the submitted ZIP: the progress report PDF, the work plan, `evidence/` (tables, figures and console log from the submitted run) and `source_code_capstone_electricity/` (a full repo snapshot plus `capstone2_branch.bundle`). Don't edit it, and don't confuse it with the live `capstone2/` package.
- **An earlier draft was dropped on purpose.** An interim "improved" notebook was superseded, because the CAT 1 report found its R² of 0.77 came from leakage. It's archived outside the repo in `~/capstone_electricity_interim_archive/`. Don't restore it.

## Commands

A local virtualenv lives at `venv/` (Python 3.14, git-ignored), and the notebook kernelspec is named `venv`.

```bash
source venv/bin/activate
pip install -r requirements.txt

python capstone2/run_all.py      # all 7 steps in order, then pytest (~1 min)
python capstone2/models.py       # one step; each script runs on its own
python -m pytest capstone2/tests -q
python -m pytest capstone2/tests/test_pipeline.py::test_lag1_is_previous_year -q

jupyter nbconvert --to notebook --execute --inplace notebooks/uganda_electricity_loss_risk.ipynb
```

- **Tests check generated files.** They read the CSVs in `outputs/capstone2/tables/`, so run the pipeline before the tests if those files are stale.
- **Reruns don't exactly match the committed outputs.** The submitted outputs were made on a different machine and library versions. Running here reproduces Ridge, the baselines and `panel_v2.csv` exactly. XGBoost moves by up to ~0.06 pp RMSE and Random Forest by ~0.002 pp, and the SHAP values shift.
- **Don't commit rerun noise.** Commit regenerated `outputs/capstone2/` only when a change is intended, and otherwise run `git restore outputs/capstone2`.

The required inputs are:
- `data/raw/Distribution_Database_Q3_2025-Updated.xlsm`. Only the `Distribution` sheet is used, which has one row per entity per quarter.
- `data/gis/distribution_lines.geojson` (~82 MB and slow to load), `transmission_network.geojson` and `uganda_adm2.json` (GADM ADM2).

## Capstone II architecture (`capstone2/`)

- **Steps.** The steps are independent scripts that pass data only through CSVs in `outputs/capstone2/tables/`. They run in this order: `gis_features` → `build_panel` → `archetypes` → `models` → `leakage_audit` → `explain` → `figures`. `run_all.py` runs each one as a subprocess and then runs pytest.
- **Central table.** `build_panel.py` writes `panel_v2.csv` (119 licensee-years). Every later step and the tests read it.
- **Inputs.** The upstream inputs are Capstone I's `data/processed/feature_matrix_annual.csv` and the GIS layers. `leakage_audit` also reads `feature_matrix_quarterly.csv`. Capstone II doesn't regenerate these files, so changing the Capstone I notebook changes Capstone II's inputs.
- **`config.py`.**
  - `ROOT = Path(__file__).resolve().parents[1]`, so paths resolve against whatever checkout holds `capstone2/`.
  - It also holds `SEED`, `ENTITY_TYPES` and `LABELS`.
  - It defines the nested feature sets `SPECS`: Structural ⊂ Persistence (+`loss_lag1`) ⊂ Forecast (+`loss_slope_lag3`). The tests check that each set contains the previous one and that no set includes `loss_trend_slope` or `loss_pct`.
  - It defines `ARCHETYPE_FEATURES`, which deliberately leaves out every loss-derived feature.
- **Shared modules.**
  - `models.py` provides `make_models()`, `loeo_predict()` and `metrics()`, which `leakage_audit.py` and `explain.py` reuse.
  - `district_lookup.py` (`lookup_licensee`, `norm`) is the district→licensee table, shared by `gis_features.py` and `figures.py`.
- **Leakage rules.**
  - Lag features use earlier years only. `build_panel.lagged_slope` is tested by perturbing each year's outcome and checking that year's own slope doesn't change.
  - Imputation and scaling are fitted inside each training fold through sklearn pipelines.
  - Hyperparameters are fixed in advance, and Ridge picks its penalty with `RidgeCV` inside each training fold.
- **Validation.**
  - Leave-one-entity-out (LOEO) validation holds out one whole licensee per fold.
  - Rolling-origin validation tests each year on a model trained on all earlier years.
  - Confidence intervals come from 2,000 bootstrap resamples drawn by licensee.
  - Every result is compared with a mean baseline and a naïve persistence baseline (ŷ = `loss_lag1`).
  - The run also includes a complete-case sensitivity check (`*_complete_case.csv`) and a Ridge feature-group ablation.
- **SHAP** is computed out-of-fold, on held-out licensee rows only.
- **Transmission length** is clipped to each service polygon geometrically. `trans_km_clipped` must be ≤ `trans_km_join`, and a test checks this.

## Capstone I notebook traps

- Each phase reloads its inputs from `data/processed/*.csv`, so you can rerun from a later phase.
- Cells 3-C/3-D/3-E were superseded by `3-FIX-A`/`3-FIX-B`, but they still run and write files.
- Cells 4-F, 5-C and 6-C each overwrite `feature_matrix_final.csv`, so its contents depend on how far the notebook ran.
- `loss_trend_slope` contains the current-year outcome, which is target leakage. That is why Capstone II exists. The README's Capstone I findings depend on it.
- All paths are relative to `notebooks/` (`../data/...`). `nbconvert` runs with that directory as its working directory.

## Domain conventions

- **Target.**
  - `loss_pct = 100 × (energy_purchased_kwh − energy_sold_kwh) / energy_purchased_kwh`.
  - Annual loss is recomputed from annual kWh totals, never averaged from quarterly rates.
  - Values outside `[0, 70)` are treated as invalid.
- **Entities.**
  - `UMEME` is major_urban.
  - Eight UEDCL territories: CNST, EST, MWST, NEST, NNWST, NWST, SST, SWST.
  - Four rural cooperatives: BECS, KIL, KRECS, PACMECS.
  - `UEDCL` and `RST` are excluded, and the raw spelling `Umeme` is normalised to `UMEME`.
  - The Umeme concession ended on 31 Mar 2025, so later data reflects a different operator.
- **Spatial.**
  - Use `EPSG:32636` (UTM 36N) for any metric work.
  - Only distribution and transmission lines with `Status == 'Operational'` are kept.
  - Voltage ≥33 kV is classed as hv33, ≥11 kV as mv11, and anything lower as lv.
- **District → licensee mapping.**
  - The mapping is an analytical approximation, because ERA publishes no concession polygons.
  - It assigns 61 of 166 districts.
  - Licensees with little GIS coverage get entity-type median values and are flagged with `gis_imputed`.
- **Small n.** There are only 13 licensees, so clustering uses k=3 for interpretability (silhouette is weak at every k) and trees are kept shallow and fixed. Frame results as screening and benchmarking, not as causal claims or a validated forecaster. The CAT 1 finding is that no model beats naïve persistence.
