"""Step 1 - Recompute licensee GIS features with geometric clipping and an imputation flag.

Changes from Capstone I:
  * Transmission length is the length of each line *inside* a service area
    (geometric intersection), not the full length of every line that touches it.
    Capstone I used a spatial join, which double-counts lines crossing several areas.
  * Licensees whose GIS features had to be imputed carry gis_imputed = 1.
  * A district coverage table is written so unmapped areas are documented.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from config import GIS_DIR, TAB, UTM, ENTITY_TYPES
from district_lookup import lookup_licensee, norm


def main():
    dist = gpd.read_file(GIS_DIR / "distribution_lines.geojson")
    trans = gpd.read_file(GIS_DIR / "transmission_network.geojson")
    adm = gpd.read_file(GIS_DIR / "uganda_adm2.json")

    dist = dist[dist["Status"] == "Operational"].to_crs(UTM)
    trans = trans[trans["Status"] == "Operational"].to_crs(UTM)
    adm = adm.to_crs(UTM)

    dist["length_km"] = dist.geometry.length / 1000
    adm["area_km2"] = adm.geometry.area / 1e6
    dist["licensee"] = dist["District"].apply(norm).apply(lookup_licensee)
    adm["licensee"] = adm["NAME_2"].apply(norm).apply(lookup_licensee)

    # --- coverage diagnostics ---------------------------------------------------
    cov = adm[["NAME_1", "NAME_2", "licensee", "area_km2"]].copy()
    cov["assigned"] = ~cov["licensee"].isin(["UNMAPPED", "EXCLUDE"])
    cov.drop(columns="area_km2").to_csv(TAB / "district_coverage.csv", index=False)
    total_km = dist["length_km"].sum()
    mapped_km = dist.loc[~dist["licensee"].isin(["UNMAPPED", "EXCLUDE"]), "length_km"].sum()
    summary = {
        "districts_total": len(adm),
        "districts_assigned": int(cov["assigned"].sum()),
        "area_assigned_pct": round(100 * cov.loc[cov.assigned, "area_km2"].sum() / cov["area_km2"].sum(), 1),
        "operational_dist_km": round(total_km, 1),
        "dist_km_assigned_pct": round(100 * mapped_km / total_km, 1),
        "operational_trans_km": round(trans.geometry.length.sum() / 1000, 1),
    }

    # --- distribution lengths by licensee --------------------------------------
    d = dist[~dist["licensee"].isin(["UNMAPPED", "EXCLUDE"])]
    v = pd.to_numeric(d["VOLTAGE"], errors="coerce")
    d = d.assign(mv11=np.where((v >= 11) & (v < 33), d["length_km"], 0.0),
                 hv33=np.where(v >= 33, d["length_km"], 0.0))
    lines = d.groupby("licensee").agg(mv11_km_gis=("mv11", "sum"), hv33_km_gis=("hv33", "sum")).reset_index()
    lines["total_dist_km_gis"] = lines["mv11_km_gis"] + lines["hv33_km_gis"]

    # --- service areas and clipped transmission ---------------------------------
    svc = adm[~adm["licensee"].isin(["UNMAPPED", "EXCLUDE"])].dissolve(by="licensee", aggfunc={"area_km2": "sum"}).reset_index()
    rows = []
    for _, s in svc.iterrows():
        inter = trans.geometry.intersection(s.geometry)
        clipped = inter.length.sum() / 1000
        touched = trans.loc[trans.intersects(s.geometry)].geometry.length.sum() / 1000
        rows.append({"licensee": s["licensee"], "service_area_km2": s["area_km2"],
                     "trans_km_clipped": clipped, "trans_km_join": touched})
    area = pd.DataFrame(rows)

    g = pd.DataFrame({"licensee": list(ENTITY_TYPES)}).merge(lines, how="left").merge(area, how="left")
    g["line_density_gis"] = g["total_dist_km_gis"] / g["service_area_km2"]
    g["trans_access_clipped"] = g["trans_km_clipped"] / g["service_area_km2"]
    g["trans_access_join"] = g["trans_km_join"] / g["service_area_km2"]

    # Impute only what is missing: line density needs mapped lines (>10 km) and an area;
    # transmission access needs only an area. gis_imputed = 1 if either was imputed.
    has_area = g["service_area_km2"].notna()
    has_lines = g["total_dist_km_gis"].fillna(0) > 10
    need = {"line_density_gis": ~(has_area & has_lines),
            "trans_access_clipped": ~has_area, "trans_access_join": ~has_area}
    g["entity_type"] = g["licensee"].map(ENTITY_TYPES)
    for col, miss in need.items():
        ok = ~miss
        med = g[ok].groupby("entity_type")[col].median()
        fill = g["entity_type"].map(med)
        # no observed rural cooperative for some fields: fall back to observed UEDCL-territory median
        fill = fill.fillna(g[ok & (g.entity_type == "uedcl_territory")][col].median())
        g.loc[miss, col] = fill[miss]
    g["gis_imputed"] = (need["line_density_gis"] | need["trans_access_clipped"]).astype(int)

    g = g.rename(columns={"licensee": "entity"})
    g.to_csv(TAB / "gis_features_v2.csv", index=False)
    pd.Series(summary).to_csv(TAB / "gis_coverage_summary.csv", header=["value"])
    print(pd.Series(summary).to_string())
    print(g[["entity", "gis_imputed", "line_density_gis", "trans_km_join", "trans_km_clipped",
             "trans_access_join", "trans_access_clipped"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
