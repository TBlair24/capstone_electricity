# Predicting Electricity Distribution Loss Risk Across Uganda's Licensed Service Areas
### Using Machine Learning and Geospatial Feature Engineering

**Author:** Tony Blair Ochieng | Jan26/MAIDS/0077U  
**Course:** CAI 7202 Capstone Project I · CAI 8103 Capstone Project II  
**Institution:** Uganda Technology and Management University (UTAMU)  
**Year:** 2026

---

## Capstone II (CAI 8103): leakage-safe re-evaluation

Capstone II rebuilds the Capstone I analysis as a modular, tested Python pipeline in `capstone2/`.
Its outputs are in `outputs/capstone2/`. See [`capstone2/README.md`](capstone2/README.md) for the
step-by-step changes relative to Capstone I.

```bash
pip install -r requirements.txt
python capstone2/run_all.py      # 7 pipeline steps, then 6 automated tests (~1 minute)
```

- **Leakage removed.** Capstone I's `loss_trend_slope` feature contained the current year's outcome. Capstone II replaces it with features built only from earlier years: `loss_lag1` and `loss_slope_lag3`.
- **Two validation designs.** Models are tested by leave-one-licensee-out validation (holding out each licensee in turn) and by year-by-year forecasting. Results are compared against both a mean baseline and a naïve "same as last year" baseline (`loss_lag1`).
- **GIS and clustering fixes.** Transmission length is clipped to each service area. Archetypes are clustered on one profile per licensee, and their stability is checked by bootstrap resampling.
- **Out-of-fold SHAP.** Feature importance is calculated only on licensees the model did not see during training.

**CAT 1 submission (19 Oct 2026):** the exact submitted code is on branch `capstone2` and tag `capstone2-cat1`.
The submitted progress report, work plan, evidence and source snapshot are archived in [`capstone_2/`](capstone_2/).
The Capstone I state is tagged `capstone1-final`.

---

## Capstone I Summary

This study applies machine learning to identify the spatial and operational drivers of electricity
distribution losses across Uganda's 13 licensed service areas, using ERA regulatory data (2009–2022)
and GIS network geometry as the sole data sources.

**Key findings:**
- Top SHAP predictors: staff per 1,000 customers (1.605), loss trend slope (1.534), customers per km (1.456), transmission access (1.439)
- K-Means (k=3) identified 3 network archetypes: rural cooperative (worsening losses), peri-urban UEDCL (stable), urban/dense (improving)
- Highest predicted risk: KIL (#1, 37.0%), BECS (#2, 34.8%), NNWST (#3, 31.9%), NWST (#4, 31.4%)
- XGBoost entity-level LOOCV: RMSE=11.25 pp, MAE=8.55 pp

---

## Repository Structure

```
capstone_electricity/
├── data/
│   ├── raw/                  ← ERA Excel files (download — see below)
│   ├── gis/                  ← GeoJSON files (download — see below)
│   └── processed/            ← Capstone I CSVs (also the inputs to Capstone II)
├── notebooks/
│   └── uganda_electricity_loss_risk.ipynb   ← Capstone I analysis notebook
├── capstone2/                ← Capstone II pipeline (7 scripts + run_all.py + tests/)
├── outputs/
│   ├── figures/              ← Capstone I: 14 figures
│   └── capstone2/            ← Capstone II: tables/ (CSV) and figures/ (PNG)
├── capstone_2/               ← Archived Capstone II CAT 1 submission (do not edit)
├── requirements.txt
└── README.md
```

---

## Data Download Instructions

### Step 1 — ERA Excel files (direct browser download, no login required)

| File | URL | Save to |
|---|---|---|
| Distribution Database Q3 2025 | https://www.era.go.ug/media/2026/01/Distribution_Database_Q3_2025-Updated.xlsm | `data/raw/` |
| Distribution Statistics Q4 2025 | https://www.era.go.ug/media/2026/04/Distribution_Statistics_Q4_2025_20260402.xlsm | `data/raw/` |

### Step 2 — GIS files (free, CC-BY 4.0)

| File | URL | Save as |
|---|---|---|
| Uganda Distribution Lines (85 MB) | https://energydata.info/dataset/distribution-lines-2023 → Download → GeoJSON | `data/gis/distribution_lines.geojson` |
| Uganda Transmission Network | https://energydata.info/dataset/uganda-electricity-transmission-network-2017 → Download → GeoJSON | `data/gis/transmission_network.geojson` |
| Uganda Admin Boundaries ADM2 | https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_UGA_2.json | `data/gis/uganda_adm2.json` |

---

## Setup and Running

```bash
# 1. Clone the repository
git clone https://github.com/TBlair24/capstone_electricity
cd capstone_electricity

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data files (see above)

# 4a. Capstone I: launch Jupyter
jupyter notebook notebooks/uganda_electricity_loss_risk.ipynb

# 4b. Capstone II: run the pipeline and tests
python capstone2/run_all.py
```

For the Capstone I notebook, run all cells in order. Each phase prints a completion message and saves intermediate CSVs
to `data/processed/` and figures to `outputs/figures/`.

---

## Capstone I Notebook Phases

| Phase | Cells | Description |
|---|---|---|
| 1 | 2-A | Data audit — confirm ERA sheet structure |
| 2 | 2-B to 2-F | Cleaning, entity consolidation, target variable, feature engineering |
| 3 | 3-FIX-A to 3-FIX-B | GIS loading, service area mapping, spatial feature computation |
| 4 | 4-A to 4-F | EDA, correlation matrix, VIF analysis, scatter plots |
| 5 | 5-A to 5-E | K-Means elbow/silhouette, k=3 clustering, cluster map |
| 6 | 6-A to 6-G | Model comparison, XGBoost tuning, LOOCV, SHAP, risk map |

---

## Capstone I Key Technical Decisions

**Why entity-level LOOCV?**  
Observations from the same licensee across years are not independent. Entity-level LOOCV holds
out all records from one licensee and evaluates generalisation to a wholly unseen entity type
— the operationally relevant test for ERA.

**Why k=3 for clustering despite k=6 having highest silhouette?**  
With 13 entities, k=6 produces clusters of 1–2 entities that cannot be characterised.
k=3 gives three interpretable network archetypes with silhouette=0.237 (above weak-structure threshold).

**Why were 4 features removed by VIF?**  
`hv33_ratio` (VIF=41.3) and `lv_ratio` (VIF=32.8) are near-perfect complements (they sum to ~1).
`mv_ratio_gis` (15.3) and `service_area_km2` (11.0) are collinear with `line_density_gis`.
Removing them is mathematically and theoretically justified.

---

## Capstone I Figures

| Figure | File | Description |
|---|---|---|
| 1 | fig1_loss_timeseries.png | Loss % time series per entity (2009–2022) |
| 2 | fig2_loss_distribution.png | Boxplots and violin by entity type |
| 3 | fig3_correlation.png | Feature correlation matrix + bar chart vs loss_pct |
| 4 | fig4_vif.png | VIF multicollinearity assessment |
| 5 | fig5_feature_scatter.png | Top-6 feature scatter plots vs loss_pct |
| 6 | fig6_elbow_silhouette.png | K-Means elbow + silhouette plots |
| 7 | fig7_cluster_profiles.png | Cluster centroid profiles + entity loss bar |
| 8 | fig8_cluster_map.png | Spatial cluster map — district and service area |
| 9 | fig9_model_comparison.png | RF vs LightGBM vs XGBoost comparison |
| 10 | fig10_regularization.png | L1 and L2 regularisation sensitivity |
| 11 | fig11_residuals.png | Residual analysis — scatter, distribution, MAE per entity |
| 12 | fig12_shap_importance.png | SHAP global importance + beeswarm |
| 13 | fig13_shap_waterfall.png | SHAP waterfall for 4 entities |
| 14 | fig14_risk_map.png | Predicted vs actual loss choropleth map |

---

## Data Sources

| Dataset | Source | Licence |
|---|---|---|
| ERA Distribution Database Q3 2025 | Electricity Regulatory Authority Uganda | Public domain (statutory publisher) |
| Uganda Distribution Lines 2023 | GIZ / Uganda Energy Sector GIS Working Group via energydata.info | CC-BY 4.0 |
| Uganda Transmission Network 2017 | GIZ / energydata.info | CC-BY 4.0 |
| Uganda Admin Boundaries ADM2 | GADM v4.1 | CC-BY 4.0 (non-commercial) |

No synthetic or simulated data is used in this study.

---

## Citation

> Ochieng, T.B. (2026). *Predicting Electricity Distribution Loss Risk Across Uganda's Licensed
> Service Areas Using Machine Learning and Geospatial Feature Engineering.* Capstone Project I,
> MSc Artificial Intelligence and Data Science, Uganda Technology and Management University (UTAMU).