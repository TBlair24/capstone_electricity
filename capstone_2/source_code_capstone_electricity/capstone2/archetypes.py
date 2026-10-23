"""Step 3 - Entity-level archetypes with bootstrap stability.

Changes from Capstone I:
  * Clustering is applied to ONE median profile per licensee (13 points), not to 107
    licensee-years. Clustering years let a single long-history licensee dominate and split
    licensees across clusters.
  * The outcome-derived trend slope is excluded from clustering inputs.
  * Stability: 200 within-licensee bootstrap resamples of years -> recompute medians ->
    re-cluster -> Adjusted Rand Index (ARI) against the original partition.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import ARCHETYPE_FEATURES, SEED, TAB


def cluster(profile, k=3, seed=SEED):
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(profile[ARCHETYPE_FEATURES]))
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)
    return km.labels_, X


def profiles(df):
    return df.groupby("entity")[ARCHETYPE_FEATURES + ["loss_pct"]].median()


def main(n_boot=200):
    df = pd.read_csv(TAB / "panel_v2.csv")
    prof = profiles(df)
    base, X = cluster(prof)

    sil = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(X))
           for k in range(2, 7)}

    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(n_boot):
        bs = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                        for _, g in df.groupby("entity")])
        lab, _ = cluster(profiles(bs).loc[prof.index])
        aris.append(adjusted_rand_score(base, lab))
    aris = np.array(aris)

    # name archetypes from centroid profiles
    prof["cluster"] = base
    cent = prof.groupby("cluster").mean()
    names = {}
    for c, r in cent.iterrows():
        if r["is_rural"] > 0.5:
            names[c] = "A: Rural cooperative"
        elif r["customers_per_km"] == cent["customers_per_km"].max() or r["line_density_gis"] == cent["line_density_gis"].max():
            names[c] = "C: Urban / dense"
        else:
            names[c] = "B: Peri-urban UEDCL"
    prof["archetype"] = prof["cluster"].map(names)
    prof.reset_index().to_csv(TAB / "entity_archetypes_v2.csv", index=False)
    summ = prof.groupby("archetype")[ARCHETYPE_FEATURES + ["loss_pct"]].mean()
    summ.insert(0, "licensees", prof.groupby("archetype").size())
    summ.insert(1, "members", prof.reset_index().groupby("archetype")["entity"].apply(", ".join))
    summ.to_csv(TAB / "archetype_profiles_v2.csv")
    pd.DataFrame({"ARI": aris}).to_csv(TAB / "cluster_bootstrap_ari.csv", index=False)
    pd.Series(sil, name="silhouette").to_csv(TAB / "entity_silhouette.csv")

    print("Silhouette (entity level):", {k: round(v, 3) for k, v in sil.items()})
    print(f"Bootstrap ARI: mean {aris.mean():.3f}, median {np.median(aris):.3f}, "
          f"share ARI=1: {(aris > 0.999).mean():.2f}, share >=0.8: {(aris >= 0.8).mean():.2f}, "
          f"5th pct {np.percentile(aris, 5):.3f}")
    print(summ.round(3).T.to_string())


if __name__ == "__main__":
    main()
