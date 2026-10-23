"""Step 6 - Figures and maps for the Capstone II progress report (static, print-ready)."""
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from config import FIG, GIS_DIR, TAB, UTM, LABELS
from district_lookup import lookup_licensee, norm

# Reference categorical slots (validated order) + neutral greys
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
SPEC_COL = {"Structural": S2, "Persistence": S1, "Forecast": S3}


def fig_loeo():
    r = pd.read_csv(TAB / "loeo_results_main.csv")
    ref = r[r.Specification == "Reference"].set_index("Model")
    m = r[r.Specification != "Reference"]
    models = ["Ridge", "Random Forest", "XGBoost"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(models)); w = 0.26
    for i, (spec, col) in enumerate(SPEC_COL.items()):
        s = m[m.Specification == spec].set_index("Model").loc[models]
        xs = x + (i - 1) * w
        ax.bar(xs, s.RMSE, w * 0.9, color=col, label=spec, zorder=2)
        ax.errorbar(xs, s.RMSE, yerr=[s.RMSE - s.RMSE_lo, s.RMSE_hi - s.RMSE], fmt="none",
                    ecolor=INK2, capsize=2, lw=0.9, zorder=3)
        for xx, v in zip(xs, s.RMSE):
            ax.text(xx, 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
    ax.axhline(ref.loc["Mean baseline", "RMSE"], color=INK2, ls="--", lw=1.1)
    ax.axhline(ref.loc["Naive persistence", "RMSE"], color=INK, ls=":", lw=1.4)
    ax.text(2.42, ref.loc["Mean baseline", "RMSE"] + 0.25, f"Mean\nbaseline\n{ref.loc['Mean baseline','RMSE']:.2f}",
            ha="left", va="bottom", fontsize=7, color=INK2)
    ax.text(2.42, ref.loc["Naive persistence", "RMSE"] - 0.25, f"Naïve\npersistence\n{ref.loc['Naive persistence','RMSE']:.2f}",
            ha="left", va="top", fontsize=7, color=INK)
    ax.set_xlim(-0.5, 2.85)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("LOEO RMSE (percentage points)")
    ax.set_ylim(0, 18.5); ax.grid(axis="x", visible=False)
    ax.legend(title="Specification", fontsize=7.5, title_fontsize=8, ncol=3, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "c2_loeo_rmse.png", dpi=220); plt.close(fig)


def fig_rolling():
    r = pd.read_csv(TAB / "rolling_origin_metrics.csv")
    piv = r.pivot(index="test_year", columns="Model", values="RMSE")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    spec = [("Mean baseline", GREY, "--"), ("Naive persistence", INK, ":"),
            ("Structural Ridge", S2, "-"), ("Forecast Ridge", S3, "-"), ("Forecast XGBoost", S1, "-")]
    for name, col, ls in spec:
        ax.plot(piv.index, piv[name], ls=ls, color=col, lw=2, marker="o", ms=4.5, label=name.replace("Naive", "Naïve"))
    ax.set_xlabel("Test year (model trained on all earlier years)"); ax.set_ylabel("RMSE (pp)")
    ax.set_xticks(piv.index); ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")
    ax.set_ylim(0, 19)
    fig.tight_layout(); fig.savefig(FIG / "c2_rolling_origin.png", dpi=220); plt.close(fig)


def fig_attribution():
    a = pd.read_csv(TAB / "oof_attribution_forecast.csv").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    y = np.arange(len(a))
    axes[0].barh(y, a.xgb_mean_abs_shap, xerr=a.xgb_sd_abs_shap, color=S1, height=0.6,
                 error_kw=dict(ecolor=GREY, lw=0.8, capsize=2))
    axes[0].set_yticks(y); axes[0].set_yticklabels(a.label, fontsize=8)
    axes[0].set_xlabel("Mean |SHAP| on held-out rows (pp)"); axes[0].set_title("(a) XGBoost, out-of-fold SHAP", fontsize=9)
    cols = [S3 if s else GREY for s in a.ridge_coef_sign_stable]
    axes[1].barh(y, a.ridge_coef_mean, xerr=a.ridge_coef_sd, color=cols, height=0.6,
                 error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("Std. coefficient (mean ± SD, 13 folds)"); axes[1].set_title("(b) Ridge coefficient stability", fontsize=9)
    axes[1].legend(handles=[mpatches.Patch(color=S3, label="Same sign in all folds"),
                            mpatches.Patch(color=GREY, label="Sign changes")], fontsize=7, frameon=False, loc="lower right")
    for ax in axes: ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "c2_oof_attribution.png", dpi=220); plt.close(fig)


def fig_stability_and_gis():
    ari = pd.read_csv(TAB / "cluster_bootstrap_ari.csv").ARI
    g = pd.read_csv(TAB / "gis_features_v2.csv").dropna(subset=["trans_km_join"])
    g = g[g.trans_km_join > 0].sort_values("trans_km_join")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].hist(ari, bins=np.linspace(0, 1, 21), color=S1, edgecolor="white", lw=1)
    axes[0].axvline(ari.mean(), color=INK, ls="--", lw=1)
    axes[0].text(ari.mean() - 0.02, axes[0].get_ylim()[1] * 0.9, f"mean {ari.mean():.2f}", ha="right", fontsize=7.5)
    axes[0].set_xlabel("Adjusted Rand Index vs original partition"); axes[0].set_ylabel("Bootstrap resamples")
    axes[0].set_title("(a) Archetype stability (200 resamples)", fontsize=9)
    y = np.arange(len(g)); h = 0.38
    axes[1].barh(y + h / 2, g.trans_km_join, h, color=GREY, label="Spatial join (Capstone I)")
    axes[1].barh(y - h / 2, g.trans_km_clipped, h, color=S2, label="Geometric clip (Capstone II)")
    axes[1].set_yticks(y); axes[1].set_yticklabels(g.entity)
    axes[1].set_xlabel("Transmission line length in service area (km)")
    axes[1].set_title("(b) Transmission double-counting removed", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right"); axes[1].grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "c2_stability_and_clipping.png", dpi=220); plt.close(fig)


def fig_map():
    adm = gpd.read_file(GIS_DIR / "uganda_adm2.json").to_crs(UTM)
    tr = gpd.read_file(GIS_DIR / "transmission_network.geojson")
    tr = tr[tr.Status == "Operational"].to_crs(UTM)
    arch = pd.read_csv(TAB / "entity_archetypes_v2.csv")
    adm["entity"] = adm["NAME_2"].apply(norm).apply(lookup_licensee)
    adm = adm.merge(arch[["entity", "archetype", "loss_pct"]], on="entity", how="left")
    COL = {"A: Rural cooperative": S2, "B: Peri-urban UEDCL": S1, "C: Urban / dense": S3}
    outline = gpd.GeoSeries([adm.geometry.buffer(300).union_all().buffer(-300)], crs=UTM)
    tr = gpd.clip(tr, outline.buffer(5000))
    fig, ax = plt.subplots(figsize=(5.6, 5.9))
    adm.plot(ax=ax, color="#ecebe7", edgecolor="white", linewidth=0.3)
    for k, c in COL.items():
        sub = adm[adm.archetype == k]
        if len(sub): sub.plot(ax=ax, color=c, edgecolor="white", linewidth=0.3)
    outline.boundary.plot(ax=ax, color=INK, linewidth=0.9)
    tr.plot(ax=ax, color=INK, linewidth=0.8)
    svc = adm[adm.archetype.notna()].dissolve(by="entity", aggfunc="first").reset_index()
    for _, r in svc.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(f"{r.entity}\n{r.loss_pct:.0f}%", (p.x, p.y), ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="#333", alpha=0.75, lw=0))
    x0, y0, x1, y1 = adm.total_bounds
    ax.plot([x0 + 60000, x0 + 160000], [y0 + 15000] * 2, color=INK, lw=3); ax.text(x0 + 110000, y0 + 24000, "100 km", ha="center", fontsize=7)
    ax.annotate("N", xy=(x1 - 30000, y1 - 15000), xytext=(x1 - 30000, y1 - 60000), ha="center", fontsize=9,
                fontweight="bold", arrowprops=dict(facecolor=INK, width=2.5, headwidth=8))
    h = [mpatches.Patch(color=c, label=k) for k, c in COL.items()] + [
        mpatches.Patch(color="#ecebe7", label="Not assigned in lookup"),
        Line2D([0], [0], color=INK, lw=1, label="Operational transmission line")]
    ax.legend(handles=h, loc="lower right", fontsize=6.5, frameon=True, title="Entity-level archetype", title_fontsize=7)
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(FIG / "c2_archetype_map.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    fig_loeo(); fig_rolling(); fig_attribution(); fig_stability_and_gis(); fig_map()
    print("Figures written to", FIG)
