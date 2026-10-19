"""Shared configuration for the Capstone II (CAI 8103) pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GIS_DIR = DATA / "gis"
PROC = DATA / "processed"           # Capstone I outputs (inputs here)
OUT = ROOT / "outputs" / "capstone2"  # all Capstone II outputs
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in (OUT, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

UTM = "EPSG:32636"   # WGS 84 / UTM zone 36N
SEED = 42

ENTITY_TYPES = {
    "UMEME": "major_urban",
    "CNST": "uedcl_territory", "EST": "uedcl_territory", "MWST": "uedcl_territory",
    "NEST": "uedcl_territory", "NNWST": "uedcl_territory", "NWST": "uedcl_territory",
    "SST": "uedcl_territory", "SWST": "uedcl_territory",
    "BECS": "rural_cooperative", "KIL": "rural_cooperative",
    "KRECS": "rural_cooperative", "PACMECS": "rural_cooperative",
}

# Capstone I predictor set, minus the leaky contemporaneous trend slope
STRUCTURAL = [
    "customers_per_km", "transformer_density", "staff_per_1000_cust",
    "om_per_customer", "network_growth_rate",
    "line_density_gis", "trans_access_clipped", "is_rural", "gis_imputed",
]
PERSISTENCE = STRUCTURAL + ["loss_lag1"]
FORECAST = PERSISTENCE + ["loss_slope_lag3"]
SPECS = {"Structural": STRUCTURAL, "Persistence": PERSISTENCE, "Forecast": FORECAST}

# Archetype features: time-varying operational + spatial structure only (no outcome terms)
ARCHETYPE_FEATURES = [
    "customers_per_km", "transformer_density", "staff_per_1000_cust",
    "om_per_customer", "network_growth_rate",
    "line_density_gis", "trans_access_clipped", "is_rural",
]

LABELS = {
    "customers_per_km": "Customers per km",
    "transformer_density": "Transformer density",
    "staff_per_1000_cust": "Staff per 1,000 customers",
    "om_per_customer": "O&M per customer",
    "network_growth_rate": "Network growth rate",
    "line_density_gis": "Line density (GIS)",
    "trans_access_clipped": "Transmission access (clipped)",
    "is_rural": "Rural flag",
    "gis_imputed": "GIS-imputed flag",
    "loss_lag1": "Previous-year loss",
    "loss_slope_lag3": "Lagged 3-year loss slope",
}
